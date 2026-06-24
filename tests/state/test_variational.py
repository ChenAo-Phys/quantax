import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
import quantax as qtx
from quantax.sites import Chain
from quantax.symmetry import Translation, Z2Inversion
from quantax.state import Variational, VS_TYPE
from quantax.model import SingleDense, RBM_Dense
from quantax.utils import array_extend, to_distributed_array
from _dtype import use_dtype

# `Variational` wraps an Equinox model and exposes the forward pass (`__call__` /
# `fast_forward`), the log-derivative Jacobian (1/psi dpsi/dtheta), parameter
# (un)flattening, in-place updates, and serialization. The four `VS_TYPE`s
# distinguish the parameter/output dtype combinations, which mainly change the
# Jacobian layout (see `quantax.state.VS_TYPE`).
#
# All standard models return amplitudes as `LogArray` (via `prod_by_log`), so the
# Jacobian tests below exercise the numerically-stable log-space gradient path for
# every dtype combination, cross-checked against finite differences.

# A genuinely non-holomorphic complex activation (the conj term breaks holomorphy).
NONHOLO_ACT = lambda z: jnp.cosh(z) + 0.3 * jnp.conj(z)
# A real-parameter network with complex output (a pure phase): real_to_complex.
PHASE_ACT = lambda x: jnp.exp(1j * x)


def _make_state(kind, *, symm=False, max_parallel=None, features=2):
    """
    Build a `Variational` of the requested kind on a 4-site chain. Sets the
    default dtype first (it determines whether outputs are complex) and creates
    the lattice before the model (models read the global `Sites`).
    """
    if kind == "real":
        use_dtype(jnp.float64)
        Chain(4, boundary=1)
        model = SingleDense(features, jnp.cosh, dtype=jnp.float64)
    elif kind == "holomorphic":
        use_dtype(jnp.complex128)
        Chain(4, boundary=1)
        model = RBM_Dense(features, dtype=jnp.complex128)
    elif kind == "non_holomorphic":
        use_dtype(jnp.complex128)
        Chain(4, boundary=1)
        model = SingleDense(
            features, NONHOLO_ACT, holomorphic=False, dtype=jnp.complex128
        )
    elif kind == "real_to_complex":
        use_dtype(jnp.complex128)
        Chain(4, boundary=1)
        model = SingleDense(features, PHASE_ACT, dtype=jnp.float64)
    else:
        raise ValueError(kind)

    symmetry = (Translation((1,)) @ Z2Inversion()) if symm else None
    return Variational(model, symm=symmetry, max_parallel=max_parallel)


def _spins(n, nmodes, seed=0):
    """A reproducible batch of +-1 configurations."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.choice([-1, 1], size=(n, nmodes)).astype(np.int8))


def _shard(s):
    """
    Pad to a multiple of `device_count` and distribute along axis 0, the shape
    `fast_forward` / `jacobian` expect (the samplers feed them sharded arrays).
    """
    s = array_extend(jnp.asarray(s), jax.device_count())
    return to_distributed_array(s)


def _fd_jacobian(vs, spins, eps=1e-6):
    """
    Finite-difference ground truth for the Jacobian 1/psi dpsi/dtheta, laid out to
    match `vs.vs_type`. Reconstructs psi from a flat parameter vector through the
    public (un)flatten / combine API, independent of `Variational.jacobian`.
    """
    _, others = vs.partition()
    flat0 = np.asarray(vs.get_params_flatten())
    nparams = vs.nparams
    symm = vs.symm

    def psi_at(flat, s):
        model = vs.combine(vs.get_params_unflatten(jnp.asarray(flat)), others)
        psi = symm.symmetrize(jax.vmap(model)(symm.get_symm_spins(s)), s)
        return complex(np.asarray(psi))

    def deriv(flat, s, psi, e):
        return (psi_at(flat + e, s) - psi_at(flat - e, s)) / (2 * eps) / psi

    n = spins.shape[0]
    if vs.vs_type == VS_TYPE.non_holomorphic:
        # complex parameters; columns are [d/dRe(theta), d/dIm(theta)]
        gt = np.zeros((n, 2 * nparams), complex)
        for i in range(n):
            psi = psi_at(flat0, spins[i])
            for j in range(nparams):
                er = np.zeros(nparams, complex)
                er[j] = eps
                ei = np.zeros(nparams, complex)
                ei[j] = 1j * eps
                gt[i, j] = deriv(flat0, spins[i], psi, er)
                gt[i, nparams + j] = deriv(flat0, spins[i], psi, ei)
    else:
        # holomorphic complex (real-eps step gives the complex derivative),
        # real_or_holomorphic real, or real_to_complex
        dt = complex if np.issubdtype(np.dtype(vs.dtype), np.complexfloating) else float
        gt = np.zeros((n, nparams), complex)
        for i in range(n):
            psi = psi_at(flat0, spins[i])
            for j in range(nparams):
                e = np.zeros(nparams, dt)
                e[j] = eps
                gt[i, j] = deriv(flat0, spins[i], psi, e)
    return gt


# ====================================================================
# Construction and VS_TYPE / dtype inference
# ====================================================================


@pytest.mark.parametrize(
    "kind,expected,holomorphic,complex_params",
    [
        ("real", VS_TYPE.real_or_holomorphic, False, False),
        ("holomorphic", VS_TYPE.real_or_holomorphic, True, True),
        ("non_holomorphic", VS_TYPE.non_holomorphic, False, True),
        ("real_to_complex", VS_TYPE.real_to_complex, False, False),
    ],
)
def test_vs_type_inference(kind, expected, holomorphic, complex_params, x64):
    vs = _make_state(kind)
    assert vs.vs_type == expected
    assert vs.holomorphic is holomorphic
    assert np.issubdtype(np.dtype(vs.dtype), np.complexfloating) == complex_params


def test_complex_params_real_output_raises(x64):
    # Complex parameters with a real default dtype is an unsupported combination.
    use_dtype(jnp.float64)
    Chain(4, boundary=1)
    with pytest.raises(RuntimeError):
        Variational(RBM_Dense(2, dtype=jnp.complex128))


def test_mixed_param_dtypes_raises():
    # All parameter arrays must be uniformly real or uniformly complex.
    class MixedDtype(eqx.Module):
        a: jax.Array
        b: jax.Array

        def __init__(self):
            self.a = jnp.ones(2, dtype=jnp.float32)
            self.b = jnp.ones(2, dtype=jnp.complex64)

        def __call__(self, s):
            return jnp.sum(self.a) + jnp.sum(self.b)

    qtx.set_default_dtype(jnp.complex64)
    Chain(4, boundary=1)
    with pytest.raises(ValueError):
        Variational(MixedDtype())


def test_nparams_matches_param_count(x64):
    vs = _make_state("real", features=3)
    params, _ = vs.partition()
    expected = sum(leaf.size for leaf in jax.tree.leaves(params))
    assert vs.nparams == expected
    assert vs.get_params_flatten().size == vs.nparams


# ====================================================================
# max_parallel parsing
# ====================================================================


def test_max_parallel_none(x64):
    vs = _make_state("real")
    assert (vs.forward_chunk, vs.backward_chunk, vs.ref_chunk) == (None, None, None)


def test_max_parallel_int(x64):
    vs = _make_state("real", max_parallel=8)
    assert (vs.forward_chunk, vs.backward_chunk, vs.ref_chunk) == (8, 8, 8)


def test_max_parallel_pair_defaults_ref_to_forward(x64):
    vs = _make_state("real", max_parallel=(4, 8))
    assert (vs.forward_chunk, vs.backward_chunk, vs.ref_chunk) == (4, 8, 4)


def test_max_parallel_triple(x64):
    vs = _make_state("real", max_parallel=(4, 8, 16))
    assert (vs.forward_chunk, vs.backward_chunk, vs.ref_chunk) == (4, 8, 16)


# ====================================================================
# Forward pass
# ====================================================================


def test_call_identity_matches_model(x64):
    # With the default Identity symmetry, psi(s) equals the bare model output.
    vs = _make_state("holomorphic")
    spins = _spins(5, vs.Nmodes)
    expected = np.array([complex(np.asarray(vs.model(s))) for s in spins])
    np.testing.assert_allclose(np.asarray(vs(spins)), expected, rtol=1e-6)


def test_call_pads_and_truncates_to_nsamples(x64):
    # __call__ pads the batch to a multiple of device_count internally and
    # returns exactly nsamples amplitudes (here a non-multiple of device_count).
    vs = _make_state("real")
    nsamples = 4 * jax.device_count() + 1
    spins = _spins(nsamples, vs.Nmodes)
    psi = vs(spins)
    assert psi.shape == (nsamples,)
    assert np.all(np.isfinite(np.asarray(psi)))


def test_fast_forward_matches_call(x64):
    vs = _make_state("non_holomorphic")
    spins = _spins(6, vs.Nmodes)
    expected = np.asarray(vs(spins))
    fast = np.asarray(vs.fast_forward(_shard(spins)))[: spins.shape[0]]
    np.testing.assert_allclose(fast, expected, rtol=1e-6)


def test_forward_chunking_is_consistent(x64):
    # The chunk size must not change the result, only the per-device batching.
    use_dtype(jnp.complex128)
    Chain(4, boundary=1)
    model = RBM_Dense(3, dtype=jnp.complex128)
    vs_full = Variational(model)
    vs_chunked = Variational(model, max_parallel=2)
    spins = _spins(16, vs_full.Nmodes)
    np.testing.assert_allclose(
        np.asarray(vs_full(spins)), np.asarray(vs_chunked(spins)), rtol=1e-6
    )


def test_symmetry_translation_invariance(x64):
    # A trivial-sector translation symmetry makes psi invariant under lattice
    # translations: psi(s) = (1/|G|) sum_t f(T_t s) is unchanged by rolling s.
    use_dtype(jnp.complex128)
    Chain(4, boundary=1)
    vs = Variational(RBM_Dense(2, dtype=jnp.complex128), symm=Translation((1,)))
    spins = _spins(4, vs.Nmodes)
    rolled = jnp.roll(spins, 1, axis=1)
    np.testing.assert_allclose(np.asarray(vs(rolled)), np.asarray(vs(spins)), rtol=1e-6)


# ====================================================================
# Parameter (un)flatten / partition / combine
# ====================================================================


def test_params_flatten_unflatten_roundtrip(x64):
    vs = _make_state("non_holomorphic", features=3)
    flat = vs.get_params_flatten()
    rebuilt = vs.combine(vs.get_params_unflatten(flat), vs.partition()[1])
    spins = _spins(4, vs.Nmodes)
    np.testing.assert_allclose(
        np.asarray(jax.vmap(rebuilt)(spins)),
        np.asarray(jax.vmap(vs.model)(spins)),
        rtol=1e-6,
    )


def test_partition_combine_reconstructs_forward(x64):
    vs = _make_state("real", features=3)
    params, others = vs.partition()
    rebuilt = vs.combine(params, others)
    spins = _spins(4, vs.Nmodes)
    np.testing.assert_allclose(
        np.asarray(vs(spins)),
        np.array([float(np.asarray(rebuilt(s))) for s in spins]),
        rtol=1e-6,
    )


# ====================================================================
# update
# ====================================================================


def test_update_subtracts_step(x64):
    # theta' = theta - step (note the minus sign convention).
    vs = _make_state("holomorphic")
    before = np.asarray(vs.get_params_flatten())
    step = jnp.asarray(
        np.linspace(0.1, 0.5, vs.nparams) + 1j * np.linspace(-0.2, 0.2, vs.nparams),
        dtype=vs.dtype,
    )
    vs.update(step)
    after = np.asarray(vs.get_params_flatten())
    np.testing.assert_allclose(after, before - np.asarray(step), rtol=1e-6)


def test_update_ignores_nonfinite_step(x64):
    vs = _make_state("real")
    before = np.asarray(vs.get_params_flatten())
    step = jnp.asarray(np.full(vs.nparams, np.nan), dtype=vs.dtype)
    vs.update(step)
    np.testing.assert_array_equal(np.asarray(vs.get_params_flatten()), before)


def test_update_real_params_discard_imaginary_step(x64):
    # For real parameters only the real part of a complex step is applied.
    vs = _make_state("real")
    before = np.asarray(vs.get_params_flatten())
    real_part = np.linspace(0.1, 0.5, vs.nparams)
    step = jnp.asarray(real_part + 1j * np.ones(vs.nparams))
    vs.update(step)
    np.testing.assert_allclose(
        np.asarray(vs.get_params_flatten()), before - real_part, rtol=1e-6
    )


# ====================================================================
# save / load
# ====================================================================


def test_save_load_roundtrip(tmp_path, x64):
    vs = _make_state("holomorphic")
    spins = _spins(4, vs.Nmodes)
    psi = np.asarray(vs(spins))

    file = tmp_path / "params.eqx"
    vs.save(file)

    # A fresh model (different random init) loaded from the file must reproduce psi.
    loaded = Variational(RBM_Dense(2, dtype=jnp.complex128), param_file=file)
    np.testing.assert_allclose(np.asarray(loaded(spins)), psi, rtol=1e-6)


# ====================================================================
# jacobian  (1/psi dpsi/dtheta)
# ====================================================================


@pytest.mark.parametrize(
    "kind", ["real", "holomorphic", "non_holomorphic", "real_to_complex"]
)
def test_jacobian_matches_finite_differences(kind, x64):
    vs = _make_state(kind)
    spins = _spins(4, vs.Nmodes)
    jac = np.asarray(vs.jacobian(_shard(spins)))[: spins.shape[0]]
    np.testing.assert_allclose(jac, _fd_jacobian(vs, spins), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("kind", ["holomorphic", "non_holomorphic", "real_to_complex"])
def test_jacobian_matches_finite_differences_with_symmetry(kind, x64):
    # Symmetrization sums amplitudes in log space; the phase gradient must survive
    # the log-sum-exp (regression for the complex-sign / LogArray gradient fix).
    vs = _make_state(kind, symm=True)
    spins = _spins(4, vs.Nmodes)
    jac = np.asarray(vs.jacobian(_shard(spins)))[: spins.shape[0]]
    np.testing.assert_allclose(jac, _fd_jacobian(vs, spins), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize(
    "kind,doubled",
    [
        ("real", False),
        ("holomorphic", False),
        ("real_to_complex", False),
        ("non_holomorphic", True),
    ],
)
def test_jacobian_shape_and_dtype(kind, doubled, x64):
    # non_holomorphic stacks d/dRe and d/dIm, so it has 2*nparams columns; all
    # complex-output types store a complex Jacobian.
    vs = _make_state(kind)
    spins = _spins(4, vs.Nmodes)
    jac = np.asarray(vs.jacobian(_shard(spins)))[: spins.shape[0]]
    assert jac.shape == (spins.shape[0], (2 if doubled else 1) * vs.nparams)
    assert jac.dtype == np.dtype(qtx.get_default_dtype())


def test_jacobian_complex_output_has_nonzero_imaginary_part(x64):
    # Guards against silently dropping the phase gradient of a complex network.
    vs = _make_state("non_holomorphic")
    spins = _spins(4, vs.Nmodes)
    jac = np.asarray(vs.jacobian(_shard(spins)))[: spins.shape[0]]
    assert np.abs(jac.imag).max() > 1e-3


def test_jacobian_chunking_is_consistent(x64):
    use_dtype(jnp.complex128)
    Chain(4, boundary=1)
    model = SingleDense(3, NONHOLO_ACT, holomorphic=False, dtype=jnp.complex128)
    vs_full = Variational(model)
    vs_chunked = Variational(model, max_parallel=(2, 2, 2))
    spins = _spins(16, vs_full.Nmodes)
    j_full = np.asarray(vs_full.jacobian(_shard(spins)))
    j_chunked = np.asarray(vs_chunked.jacobian(_shard(spins)))
    np.testing.assert_allclose(j_full, j_chunked, rtol=1e-6)


# ====================================================================
# to_netket_model
# ====================================================================


@pytest.mark.parametrize("kind", ["real", "holomorphic"])
def test_to_netket_model_returns_logpsi(kind, x64):
    vs = _make_state(kind)
    spins = _spins(5, vs.Nmodes)
    logpsi = np.asarray(vs.to_netket_model()(spins))
    np.testing.assert_allclose(np.exp(logpsi), np.asarray(vs(spins)), rtol=1e-5)
