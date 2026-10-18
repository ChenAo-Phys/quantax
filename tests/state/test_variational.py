import warnings
from typing import Callable
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
import quantax as qtx
from quantax.sites import Chain
from quantax.symmetry import Translation, Z2Inversion
from quantax.state import Variational, VS_TYPE
from quantax.nn import RefModel, prod_by_log
from quantax.utils import array_extend, to_distributed_array
from _dtype import use_dtype

# `Variational` wraps an Equinox model and exposes the forward pass (`__call__` /
# `fast_forward`), the log-derivative Jacobian (1/psi dpsi/dtheta), parameter
# (un)flattening, in-place updates, the local-update (`RefModel`) paths, and
# serialization. The four `VS_TYPE`s distinguish the parameter/output dtype
# combinations, which mainly change the Jacobian layout (see `quantax.state.VS_TYPE`).
#
# The tests use `_Dense`, the `SingleDense` architecture psi(s) = prod f(W s + b)
# with a numpy init: the library init calibrates its weight scale with eager ops
# that take seconds to compile for every new dtype and width, while the
# `Variational` logic under test doesn't depend on the init. Like the standard
# models, `_Dense` returns amplitudes as `LogArray` (via `prod_by_log`), so the
# Jacobian tests exercise the log-space gradient path; `log_output=False` covers
# models returning plain arrays.

NDEV = jax.device_count()

# A genuinely non-holomorphic complex activation (the conj term breaks holomorphy).
NONHOLO_ACT = lambda z: jnp.cosh(z) + 0.3 * jnp.conj(z)
# A real-parameter network with complex output; both log|psi| and the phase depend
# on the parameters, so neither half of the real_to_complex Jacobian is trivial.
R2C_ACT = lambda x: jnp.cosh(x) * jnp.exp(1j * x)

# kind -> (parameter dtype, default dtype, activation, holomorphic, VS_TYPE)
KINDS = {
    "real": (jnp.float64, jnp.float64, jnp.cosh, False, VS_TYPE.real_or_holomorphic),
    "holomorphic": (
        jnp.complex128,
        jnp.complex128,
        jnp.cosh,
        True,
        VS_TYPE.real_or_holomorphic,
    ),
    "non_holomorphic": (
        jnp.complex128,
        jnp.complex128,
        NONHOLO_ACT,
        False,
        VS_TYPE.non_holomorphic,
    ),
    "real_to_complex": (
        jnp.float64,
        jnp.complex128,
        R2C_ACT,
        False,
        VS_TYPE.real_to_complex,
    ),
}

# Configurations of period 4 on the 4-site ring, none of which is annihilated by
# the symmetries used below (e.g. momentum pi/2 kills the period-2 Neel states).
SPINS = np.array(
    [[1, 1, 1, -1], [1, -1, -1, -1], [1, 1, -1, -1], [-1, 1, 1, -1]], dtype=np.int8
)


class _Dense(eqx.Module):
    """psi(s) = prod_j f(W s + b)_j, as a LogArray or a plain array."""

    W: jax.Array
    b: jax.Array
    actfn: Callable = eqx.field(static=True)
    holomorphic: bool = eqx.field(static=True)
    log_output: bool = eqx.field(static=True)

    def __call__(self, s):
        x = self.actfn(self.W @ s + self.b)
        return prod_by_log(x) if self.log_output else jnp.prod(x)


class _ExpLinearRef(RefModel):
    """
    psi(s) = exp(w . s) with the internal value w . s, which `ref_forward` updates by
    w . (s - s_old). The update is only right if s, s_old and the internal value of
    the same symmetry image are paired. ``modes`` are the required update modes, and
    ``ref=False`` turns the local updates off as `RefModel.use_ref` allows.
    """

    w: jax.Array
    modes: tuple[str, ...] = eqx.field(static=True, default=("nflips",))
    ref: bool = eqx.field(static=True, default=True)

    @property
    def use_ref(self):
        return self.ref

    @property
    def required_update_modes(self):
        return self.modes

    def __call__(self, s):
        return jnp.exp(self.w @ s)

    def init_internal(self, s):
        x = self.w @ s
        return jnp.exp(x), x

    def ref_forward(self, s, s_old, update_mode, internal, return_update=False):
        x = internal + self.w @ (s - s_old)
        return (jnp.exp(x), x) if return_update else jnp.exp(x)


def _dense_model(kind, *, log_output=True, features=2, seed=0, lattice=True):
    """
    A `_Dense` model of the requested kind on a 4-site chain. Sets the default
    dtype first (it determines whether outputs are complex) and creates the lattice
    unless ``lattice=False``.
    """
    param_dtype, dtype, actfn, holomorphic, _ = KINDS[kind]
    if lattice:
        use_dtype(dtype)
        Chain(4, boundary=1)
    rng = np.random.default_rng(seed)
    W = 0.5 * rng.standard_normal((features, 4))
    b = 0.5 * rng.standard_normal(features)
    if jnp.issubdtype(param_dtype, jnp.complexfloating):
        W = W + 0.5j * rng.standard_normal(W.shape)
        b = b + 0.5j * rng.standard_normal(b.shape)
    W, b = jnp.asarray(W, param_dtype), jnp.asarray(b, param_dtype)
    return _Dense(W, b, actfn, holomorphic, log_output)


def _variational(kind, *, symm=None, max_parallel=None, log_output=True):
    """A `Variational` of `_dense_model(kind)` with a named symmetry."""
    model = _dense_model(kind, log_output=log_output)
    if symm == "real":
        symm = Translation((1,)) @ Z2Inversion()
    elif symm == "complex":
        symm = Translation((1,), sector=1)  # complex characters
    elif symm == "odd_z2":
        symm = Z2Inversion(-1)
    return Variational(model, symm=symm, max_parallel=max_parallel)


def _ref_model(seed=0, **kwargs):
    """An `_ExpLinearRef` in the default dtype on a 4-site chain."""
    Chain(4, boundary=1)
    w = 0.5 * np.random.default_rng(seed).standard_normal(4)
    return _ExpLinearRef(jnp.asarray(w, qtx.get_default_dtype()), **kwargs)


def _spins(n, seed=0):
    """A reproducible batch of ±1 configurations on 4 sites."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=(n, 4)).astype(np.int8)


def _shard(s):
    """
    Pad to a multiple of `device_count` and distribute along axis 0, the shape
    `fast_forward` / `jacobian` expect (the samplers feed them sharded arrays).
    """
    s = array_extend(jnp.asarray(s), NDEV)
    return to_distributed_array(s)


def _bare_psi(model, spins):
    """The bare model output for a batch, bypassing `Variational` (one compile)."""
    return np.asarray(jax.jit(jax.vmap(model))(jnp.asarray(spins)))


def _fd_jacobian(vs, spins, eps=1e-6):
    """
    Central finite-difference ground truth for the Jacobian 1/psi dpsi/dtheta, laid
    out like `vs.vs_type`. Reconstructs psi from a flat parameter vector through the
    public (un)flatten / combine API, independent of `Variational.jacobian`, and
    evaluates all perturbations in one jitted call.
    """
    _, others = vs.partition()
    flat0 = vs.get_params_flatten()
    eye = np.eye(flat0.size)
    if vs.vs_type == VS_TYPE.non_holomorphic:
        # complex parameters; columns are [d/dRe(theta), d/dIm(theta)]
        steps = np.concatenate([eye, 1j * eye])
    else:
        # holomorphic complex (a real step gives the complex derivative),
        # real_or_holomorphic real, or real_to_complex
        steps = eye
    steps = jnp.asarray(eps * steps, flat0.dtype)
    symm = vs.symm

    def psi_at(flat, s):
        model = vs.combine(vs.get_params_unflatten(flat), others)
        psi = symm.symmetrize(jax.vmap(model)(symm.get_symm_spins(s)), s)
        return jnp.asarray(psi)

    @jax.jit
    def ratio(s):
        diff = lambda e: psi_at(flat0 + e, s) - psi_at(flat0 - e, s)
        return jax.vmap(diff)(steps) / (2 * eps) / psi_at(flat0, s)

    return np.asarray(jax.vmap(ratio)(jnp.asarray(spins)))


# ====================================================================
# Construction: VS_TYPE / dtype inference and validation
# ====================================================================


def test_complex_params_real_output_raises(x64):
    # Complex parameters with a real default dtype is an unsupported combination.
    model = _dense_model("holomorphic")
    use_dtype(jnp.float64)
    with pytest.raises(RuntimeError):
        Variational(model)


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


# ====================================================================
# max_parallel parsing and low-precision warnings
# ====================================================================


@pytest.mark.parametrize(
    "max_parallel,expected",
    [
        (None, (None, None, None)),
        (8, (8, 8, 8)),
        ((4, 8), (4, 8, 4)),  # the ref chunk defaults to the forward chunk
        ((4, 8, 16), (4, 8, 16)),
    ],
)
def test_max_parallel_parsing(max_parallel, expected):
    vs = Variational(_ref_model(), max_parallel=max_parallel)
    assert (vs.forward_chunk, vs.backward_chunk, vs.ref_chunk) == expected


@pytest.mark.parametrize(
    "max_parallel,use_ref,expected",
    [
        # Without a chunk, the batch size of the forward pass isn't fixed, which may
        # change psi of the same configuration in low precision.
        (None, True, "`forward_chunk` is None"),
        (8, True, None),
        # Local updates evaluate psi in batches of ref_chunk.
        ((4, 8, 16), True, "`ref_chunk` doesn't match"),
        ((4, 8, 16), False, None),
        ((4, 8), True, None),
    ],
)
def test_low_precision_chunk_warnings(max_parallel, use_ref, expected, gpu_precision):
    model = _ref_model()  # float32
    with jax.default_matmul_precision("tensorfloat32"):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            Variational(model, max_parallel=max_parallel, use_ref=use_ref)
    messages = [str(w.message) for w in record if "chunk" in str(w.message)]
    if expected is None:
        assert messages == []
    else:
        assert len(messages) == 1 and expected in messages[0]

    # float32 is full precision with "highest" matmuls
    with jax.default_matmul_precision("highest"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Variational(model, max_parallel=max_parallel, use_ref=use_ref)


# ====================================================================
# Forward pass
# ====================================================================


@pytest.mark.parametrize("max_parallel", [None, 2])
def test_forward_paths_match_model(max_parallel, x64):
    # Both forward paths equal the bare model with the default Identity symmetry,
    # for any chunk size. __call__ pads the batch to a multiple of device_count
    # (and of the chunk) internally and returns exactly nsamples amplitudes.
    vs = Variational(_dense_model("holomorphic"), max_parallel=max_parallel)
    assert not vs.use_ref  # not a RefModel
    spins = _spins(4 * NDEV + 1)
    expected = _bare_psi(vs.model, spins)

    psi = np.asarray(vs(spins))
    assert psi.shape == (spins.shape[0],)
    np.testing.assert_allclose(psi, expected, rtol=1e-12)

    n = 4 * NDEV  # fast_forward expects a sharded batch
    fast = np.asarray(vs.fast_forward(_shard(spins[:n])))
    np.testing.assert_allclose(fast, expected[:n], rtol=1e-12)


def test_symmetry_sector_phase(x64):
    # psi(s) = (1/|G|) sum_g chi_g f(T_g s) is an eigenstate of the symmetry:
    # psi(T s) = conj(chi_T) psi(s). Momentum pi/2 on 4 sites has the complex
    # character chi = exp(-i pi/2), with T s = roll(s, -1); the odd Z2 sector flips
    # the sign under s -> -s.
    vs = Variational(
        _dense_model("holomorphic"),
        symm=Translation((1,), sector=1) @ Z2Inversion(-1),
    )
    psi = np.asarray(vs(SPINS))
    assert np.all(np.abs(psi) > 1e-2)
    np.testing.assert_allclose(np.asarray(vs(np.roll(SPINS, -1, axis=1))), 1j * psi)
    np.testing.assert_allclose(np.asarray(vs(-SPINS)), -psi)


# ====================================================================
# update
# ====================================================================


def test_update_subtracts_step(x64):
    # theta' = theta - step (note the minus sign convention), and the compiled
    # forward pass sees the new parameters.
    vs = _variational("non_holomorphic")
    _, others = vs.partition()
    before = np.asarray(vs.get_params_flatten())
    rng = np.random.default_rng(1)
    step = 0.1 * (
        rng.standard_normal(vs.nparams) + 1j * rng.standard_normal(vs.nparams)
    )
    psi_before = np.asarray(vs(SPINS))
    vs.update(jnp.asarray(step, vs.dtype))

    after = before - step
    np.testing.assert_allclose(np.asarray(vs.get_params_flatten()), after, rtol=1e-12)
    # get_params_unflatten inverts get_params_flatten
    model = vs.combine(vs.get_params_unflatten(jnp.asarray(after)), others)
    expected = _bare_psi(model, SPINS)
    np.testing.assert_allclose(np.asarray(vs(SPINS)), expected, rtol=1e-12)
    assert not np.allclose(expected, psi_before)


def test_update_ignores_nonfinite_step(x64):
    vs = _variational("real")
    before = np.asarray(vs.get_params_flatten())
    step = np.linspace(0.1, 0.5, vs.nparams)
    step[1] = np.nan
    with pytest.warns(UserWarning, match="invalid update step"):
        vs.update(jnp.asarray(step))
    np.testing.assert_array_equal(np.asarray(vs.get_params_flatten()), before)


def test_update_real_params_discard_imaginary_step(x64):
    # For real parameters only the real part of a complex step is applied.
    vs = _variational("real")
    before = np.asarray(vs.get_params_flatten())
    real_part = np.linspace(0.1, 0.5, vs.nparams)
    step = jnp.asarray(real_part + 1j * np.ones(vs.nparams))
    vs.update(step)
    after = vs.get_params_flatten()
    assert after.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(after), before - real_part, rtol=1e-12)


# ====================================================================
# save / load
# ====================================================================


def test_save_load_roundtrip(tmp_path, x64):
    vs = _variational("holomorphic")
    file = tmp_path / "params.eqx"
    vs.save(file)

    # A fresh model (different random init) loaded from the file must reproduce the
    # parameters, hence psi.
    fresh = _dense_model("holomorphic", seed=1, lattice=False)
    params = np.asarray(vs.get_params_flatten())
    assert not np.allclose(np.asarray(Variational(fresh).get_params_flatten()), params)
    loaded = Variational(fresh, param_file=file)
    np.testing.assert_array_equal(np.asarray(loaded.get_params_flatten()), params)


# ====================================================================
# jacobian  (1/psi dpsi/dtheta)
# ====================================================================


@pytest.mark.parametrize(
    "kind,log_output,symm,max_parallel",
    [
        # a model returning plain arrays instead of LogArray, with 4 samples per
        # device in chunks of 2
        ("real", False, "real", 2),
        # plain complex output: the phase gradient comes from psi / stop_grad(psi)
        ("non_holomorphic", False, "complex", None),
        ("holomorphic", True, "complex", None),
        ("non_holomorphic", True, "complex", 2),
        ("real_to_complex", True, "complex", None),
    ],
)
def test_jacobian_matches_finite_differences(kind, log_output, symm, max_parallel, x64):
    # Symmetrization sums amplitudes in log space; the phase gradient must survive
    # the log-sum-exp (regression for the complex-sign / LogArray gradient fix).
    vs = _variational(kind, symm=symm, max_parallel=max_parallel, log_output=log_output)
    param_dtype, _, _, holomorphic, vs_type = KINDS[kind]
    assert vs.vs_type == vs_type
    assert vs.dtype == param_dtype  # the parameter dtype, not the output dtype
    assert vs.holomorphic is holomorphic
    assert vs.nparams == sum(x.size for x in jax.tree.leaves(vs.partition()[0]))

    # The 4-site ring has only 3-4 symmetry orbits, and the Jacobian is constant on
    # an orbit, so rows repeat; cycling SPINS makes the per-device blocks differ, so
    # rows returned in the wrong device order are caught.
    spins = np.concatenate([np.roll(SPINS, -d, axis=0) for d in range(NDEV)])
    jac = np.asarray(vs.jacobian(_shard(spins)))
    # non_holomorphic stacks d/dRe and d/dIm, so it has 2*nparams columns; all
    # complex-output types store a complex Jacobian.
    ncols = (2 if vs_type == VS_TYPE.non_holomorphic else 1) * vs.nparams
    assert jac.shape == (spins.shape[0], ncols)
    assert jac.dtype == qtx.get_default_dtype()

    fd = _fd_jacobian(vs, spins)
    if jnp.issubdtype(qtx.get_default_dtype(), jnp.complexfloating):
        # dropping either the magnitude or the phase gradient would be caught
        assert np.abs(fd.real).max() > 0.1 and np.abs(fd.imag).max() > 0.1
    np.testing.assert_allclose(jac, fd, rtol=1e-6, atol=1e-8)


# ====================================================================
# Local updates (RefModel)
# ====================================================================


def test_ref_paths_match_call(x64):
    # init_internal, ref_forward and segment_ref_forward must pair the 8 symmetry
    # images of s, s_old and the internal values in the same order, chunk by chunk
    # (2 per chunk, 2 or 4 chunks per device), to agree with the direct forward pass.
    use_dtype(jnp.float64)
    model = _ref_model()
    symm = Translation((1,)) @ Z2Inversion()
    vs = Variational(model, symm=symm, max_parallel=2)
    assert vs.use_ref and vs.required_update_modes == ("nflips",)
    n = 4 * NDEV
    s_old = _spins(n)
    s_new = s_old.copy()
    s_new[:, [0, 2]] *= -1
    # Connected configurations as in the local energy: flip site j of every sample,
    # with idx_segment pointing each one to its sample.
    flips = [np.where(np.arange(4) == j, -1, 1).astype(np.int8) for j in (1, 3)]
    s_conn = np.concatenate([s_old * flip for flip in flips])
    idx_segment = np.tile(np.arange(n), 2)
    psi_direct = np.asarray(vs(np.concatenate([s_old, s_new, s_conn])))

    psi_old, internal = vs.init_internal(_shard(s_old))
    np.testing.assert_allclose(np.asarray(psi_old), psi_direct[:n])

    args = (_shard(s_new), _shard(s_old), {"nflips": 2}, internal)
    np.testing.assert_allclose(np.asarray(vs.ref_forward(*args)), psi_direct[n : 2 * n])
    psi_new, internal_new = vs.ref_forward(*args, return_update=True)
    np.testing.assert_allclose(np.asarray(psi_new), psi_direct[n : 2 * n])
    _, internal_expected = vs.init_internal(_shard(s_new))
    np.testing.assert_allclose(np.asarray(internal_new), np.asarray(internal_expected))

    psi_conn = vs.segment_ref_forward(
        _shard(s_conn), _shard(s_old), {"nflips": 1}, _shard(idx_segment), internal
    )
    np.testing.assert_allclose(np.asarray(psi_conn), psi_direct[2 * n :])


def test_ref_misuse_raises(x64):
    use_dtype(jnp.float64)
    model = _ref_model(modes=("nflips", "idx_flips"))
    s = _shard(_spins(NDEV))
    vs = Variational(model)
    _, internal = vs.init_internal(s)
    # every required mode must be given
    for update_mode in ({}, {"nflips": 1}):
        with pytest.raises(ValueError, match="required modes"):
            vs.ref_forward(s, s, update_mode, internal)

    # Local updates are off with use_ref=False, or when the RefModel itself turns
    # them off (e.g. backflow with rank > Ntotal).
    no_ref = _ExpLinearRef(model.w, ref=False)
    for vs in (Variational(model, use_ref=False), Variational(no_ref)):
        assert not vs.use_ref
        with pytest.raises(RuntimeError):
            vs.init_internal(s)


# ====================================================================
# to_netket_model
# ====================================================================


@pytest.mark.parametrize(
    "kind,log_output,symm",
    [
        # the odd Z2 sector makes the real amplitudes take both signs
        ("real", False, "odd_z2"),
        ("holomorphic", True, None),
    ],
)
def test_to_netket_model_returns_logpsi(kind, log_output, symm, x64):
    vs = _variational(kind, symm=symm, log_output=log_output)
    s = _spins(3)
    spins = np.concatenate([s, -s])  # psi(-s) = -psi(s) in the odd Z2 sector
    psi = np.asarray(vs(spins))
    if kind == "real":
        assert np.any(psi < 0) and np.any(psi > 0)
    # NetKet jits the module
    logpsi = np.asarray(eqx.filter_jit(vs.to_netket_model())(spins))
    assert np.iscomplexobj(logpsi)
    np.testing.assert_allclose(np.exp(logpsi), psi, rtol=1e-10)
