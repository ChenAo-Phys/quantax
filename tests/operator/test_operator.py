import numpy as np
import jax
import jax.numpy as jnp
import pytest
import quantax as qtx
from quantax.sites import Chain
from quantax.operator import (
    Operator,
    sigma_x,
    sigma_z,
    sigma_p,
    sigma_m,
    Heisenberg,
)
from quantax.operator.operator import _get_conn_size
from quantax.model import RBM_Dense
from quantax.sampler import Samples
from quantax.state import DenseState, Variational
from quantax.symmetry import Identity
from quantax.utils import ints_to_array, to_distributed_array
from _dtype import use_dtype


def _dense(op: Operator) -> np.ndarray:
    """Dense matrix of an operator as a plain numpy array."""
    return np.asarray(op.todense())


# --- algebra against the QuSpin dense matrix (ground truth) ---


def test_add_matches_dense_sum():
    Chain(3, boundary=1)
    A = sigma_z(0) @ sigma_z(1)
    B = sigma_x(1) @ sigma_x(2)
    assert np.allclose(_dense(A + B), _dense(A) + _dense(B))


def test_sub_matches_dense():
    Chain(3, boundary=1)
    A = sigma_z(0) @ sigma_z(1)
    B = sigma_z(1) @ sigma_z(2)
    assert np.allclose(_dense(A - B), _dense(A) - _dense(B))


def test_scalar_mul_div_neg():
    Chain(3, boundary=1)
    A = sigma_z(0) @ sigma_z(1)
    DA = _dense(A)
    assert np.allclose(_dense(2.0 * A), 2.0 * DA)
    assert np.allclose(_dense(A * 2.0), 2.0 * DA)
    assert np.allclose(_dense(A / 2.0), DA / 2.0)
    assert np.allclose(_dense(-A), -DA)


def test_matmul_operator_product_is_ordered():
    # Operator product must respect matrix order: (A @ B) acts B first.
    Chain(3, boundary=1)
    A = sigma_x(0) @ sigma_x(1)
    B = sigma_z(1) @ sigma_z(2)
    assert np.allclose(_dense(A @ B), _dense(A) @ _dense(B))
    # the product is non-commuting here, guarding against accidental symmetry
    assert not np.allclose(_dense(A @ B), _dense(B @ A))


def test_hermitian_conjugate(x64):
    use_dtype(jnp.complex128)
    Chain(3, boundary=1)
    A = 1j * (sigma_p(0) @ sigma_m(1)) + sigma_z(2)
    assert np.allclose(_dense(A.H), _dense(A).conj().T)
    # H does not mutate the original operator
    assert np.allclose(
        _dense(A), (1j * _dense(sigma_p(0) @ sigma_m(1))) + _dense(sigma_z(2))
    )


def test_expression_runs():
    Chain(2, boundary=1)
    H = Heisenberg(J=1.0)
    # smoke test: expression is the same as repr and contains the spin symbols
    assert H.expression == repr(H)
    assert "S" in H.expression


# --- regressions for in-place operators (cache invalidation) ---


def test_imul_invalidates_cache():
    Chain(2, boundary=1)
    H = sigma_z(0) @ sigma_z(1)
    before = _dense(H)  # populates the QuSpin / jax caches
    H *= 3.0
    assert np.allclose(_dense(H), 3.0 * before)
    # the jax op list (used by Oloc) must also reflect the new strength
    strengths = [J for _, terms in H.jax_op_list for t in terms for J in t.strength]
    assert np.allclose(np.real(strengths), 3.0 * 4.0)  # sigma_z(0)@sigma_z(1) -> 4.0


def test_itruediv_invalidates_cache():
    Chain(2, boundary=1)
    H = sigma_z(0) @ sigma_z(1)
    before = _dense(H)
    H /= 2.0
    assert np.allclose(_dense(H), before / 2.0)


def test_iadd_returns_self_and_invalidates_cache():
    Chain(3, boundary=1)
    H = sigma_z(0) @ sigma_z(1)
    B = sigma_z(1) @ sigma_z(2)
    before = _dense(H)  # populate cache before mutating
    hid = id(H)
    H += B
    assert hid == id(H)  # __iadd__ returns the same object
    assert np.allclose(_dense(H), before + _dense(sigma_z(1) @ sigma_z(2)))


def test_isub_returns_self():
    Chain(2, boundary=1)
    H = 2.0 * (sigma_z(0) @ sigma_z(1))
    before = _dense(H)
    hid = id(H)
    H -= sigma_z(0) @ sigma_z(1)
    assert hid == id(H)
    assert np.allclose(_dense(H), before - _dense(sigma_z(0) @ sigma_z(1)))


# --- regressions for duplicate-term merging in __add__ / __iadd__ ---


def test_add_merges_same_opstr_from_other():
    # `other` contributes two terms sharing opstr 'zz' that is absent from `self`.
    Chain(4, boundary=1)
    A = sigma_x(0) @ sigma_x(1)
    B = (sigma_z(0) @ sigma_z(1)) + (sigma_z(2) @ sigma_z(3))
    C = A + B
    opstrs = [t.opstr for t in C.op_list]
    assert opstrs.count("zz") == 1
    assert opstrs.count("xx") == 1
    assert np.allclose(_dense(C), _dense(A) + _dense(B))


def test_iadd_merges_same_opstr_from_other():
    Chain(4, boundary=1)
    A = sigma_x(0) @ sigma_x(1)
    expected = (
        _dense(A) + _dense(sigma_z(0) @ sigma_z(1)) + _dense(sigma_z(2) @ sigma_z(3))
    )
    A += (sigma_z(0) @ sigma_z(1)) + (sigma_z(2) @ sigma_z(3))
    opstrs = [t.opstr for t in A.op_list]
    assert opstrs.count("zz") == 1
    assert np.allclose(_dense(A), expected)


# --- exact expectation through the dense application path ---


def test_expectation_matches_ground_energy(x64):
    use_dtype(jnp.complex128)
    Chain(4, boundary=1)
    H = Heisenberg(J=1.0)
    w, v = H.diagonalize(k=1)
    gs = DenseState(np.asarray(v[:, 0]))
    energy = gs @ H @ gs
    assert np.isclose(complex(energy).real, float(w[0]), atol=1e-6)
    assert np.isclose(complex(energy).imag, 0.0, atol=1e-6)


def test_oloc_weighted_mean_matches_exact_expectation(x64):
    # The core VMC identity: sum_s |psi_s|^2 Oloc(s) / sum_s |psi_s|^2 equals
    # <psi|H|psi> / <psi|psi>. Checked over the full basis with a random psi, so it
    # exercises the off-diagonal connected-config machinery, not just diagonal terms.
    use_dtype(jnp.complex128)
    Chain(4, boundary=1)
    symm = Identity()
    symm.basis_make()
    H = Heisenberg(J=1.0)
    rng = np.random.default_rng(0)
    psi = rng.standard_normal(symm.basis.Ns) + 1j * rng.standard_normal(symm.basis.Ns)
    state = DenseState(jnp.asarray(psi))
    samples = jnp.asarray(ints_to_array(symm.basis.states))

    Oloc = np.asarray(H.Oloc(state, samples))
    w = np.abs(psi) ** 2
    est = np.sum(w * Oloc) / w.sum()
    exact = (psi.conj() @ _dense(H) @ psi) / (psi.conj() @ psi)
    assert np.isclose(est, exact, atol=1e-8)


def test_expectation_with_unnormalized_r(x64):
    # r with mean != 1 (e.g. given by hand) must still give the weighted mean and
    # variance, normalized by sum(r)
    use_dtype(jnp.complex128)
    Chain(4, boundary=1)
    symm = Identity()
    symm.basis_make()
    H = Heisenberg(J=1.0)
    rng = np.random.default_rng(0)
    psi = rng.standard_normal(symm.basis.Ns) + 1j * rng.standard_normal(symm.basis.Ns)
    state = DenseState(jnp.asarray(psi))
    spins = to_distributed_array(jnp.asarray(ints_to_array(symm.basis.states)))
    r = rng.uniform(0.5, 1.5, symm.basis.Ns) * 3.0
    samples = Samples(spins, r=to_distributed_array(jnp.asarray(r)))

    Omean, Ovar = H.expectation(state, samples, return_var=True)
    Oloc = np.asarray(H.Oloc(state, spins))
    w = r / r.sum()
    mean = np.sum(w * Oloc)
    assert np.isclose(Omean, mean, atol=1e-10)
    assert np.isclose(Ovar, np.sum(w * np.abs(Oloc - mean) ** 2), atol=1e-10)


def test_expectation_var_no_fp32_cancellation():
    # a large constant shift makes <|O|^2> - |<O>|^2 cancel catastrophically in fp32,
    # while the variance itself is unchanged
    Chain(4, boundary=1)
    symm = Identity()
    symm.basis_make()
    # sigma_z(0) @ sigma_z(0) is the identity
    H = Heisenberg(J=1.0) + 1e4 * (sigma_z(0) @ sigma_z(0))
    rng = np.random.default_rng(0)
    psi = rng.standard_normal(symm.basis.Ns)
    state = DenseState(jnp.asarray(psi, jnp.float32))
    spins = to_distributed_array(jnp.asarray(ints_to_array(symm.basis.states)))
    r = rng.uniform(0.5, 1.5, symm.basis.Ns)
    samples = Samples(
        spins, r=to_distributed_array(jnp.asarray(r / r.mean(), jnp.float32))
    )

    _, Ovar = H.expectation(state, samples, return_var=True)
    Oloc = np.asarray(H.Oloc(state, spins), np.float64)
    w = r / r.sum()
    var = np.sum(w * np.abs(Oloc - np.sum(w * Oloc)) ** 2)
    assert np.isclose(Ovar, var, rtol=1e-4)


def test_diagonalize_full_matches_dense_eigh():
    Chain(3, boundary=1)
    H = Heisenberg(J=1.0)
    w_full, v = H.diagonalize(k="full")
    Hd = _dense(H)
    # the full spectrum matches a direct dense eigendecomposition...
    assert np.allclose(np.sort(w_full), np.linalg.eigvalsh(Hd), atol=1e-6)
    # ...and the returned eigenvectors actually diagonalize H
    assert np.allclose(v.conj().T @ Hd @ v, np.diag(w_full), atol=1e-6)


def test_diagonalize_invalid_k_raises():
    Chain(2, boundary=1)
    H = Heisenberg(J=1.0)
    with pytest.raises(ValueError):
        H.diagonalize(k="lowest")


# --- Oloc with chunking and fused sample forward pass ---


def _full_basis_samples():
    """All configurations in QuSpin basis order, as a jax array."""
    symm = Identity()
    symm.basis_make()
    return jnp.asarray(ints_to_array(symm.basis.states))


def test_oloc_chunked_matches_dense():
    # ref_chunk (= max_parallel) smaller than the per-device batch forces the
    # chunk_map loop in _Oloc, and raw samples without psi exercise the fused
    # forward pass where the samples ride in the tail of the conn buffer.
    Chain(4, boundary=1)
    H = Heisenberg(J=1.0)
    state = Variational(RBM_Dense(features=4), max_parallel=2)

    samples = _full_basis_samples()
    Oloc = np.asarray(H.Oloc(state, samples))

    psi = np.asarray(state(samples))
    ref = (_dense(H) @ psi) / psi
    assert np.allclose(Oloc, ref, rtol=1e-4, atol=1e-5)


def test_oloc_raw_samples_match_samples_with_psi():
    # Oloc from raw configurations (psi extracted from the conn forward pass)
    # must agree with Oloc from Samples carrying a precomputed psi.
    Chain(4, boundary=1)
    H = Heisenberg(J=1.0)
    state = Variational(RBM_Dense(features=4), max_parallel=2)

    spins = to_distributed_array(_full_basis_samples())
    Oloc_raw = np.asarray(H.Oloc(state, spins))
    Oloc_psi = np.asarray(H.Oloc(state, Samples(spins, psi=state(spins))))
    assert np.allclose(Oloc_raw, Oloc_psi, rtol=1e-5, atol=1e-6)


# --- Oloc fills a missing psi of the samples in place and updates r ---


def test_oloc_fills_psi_and_r_in_place():
    # psi evaluated in the forward pass of the connected configurations is written
    # back into the samples carrying only prob, and r is updated from it.
    Chain(4, boundary=1)
    H = Heisenberg(J=1.0)
    state = Variational(RBM_Dense(features=4), max_parallel=2)
    spins = to_distributed_array(_full_basis_samples())
    psi = np.asarray(state(spins))
    samples = Samples(spins, prob=jnp.abs(jnp.asarray(psi)))
    Oloc = np.asarray(H.Oloc(state, samples))
    np.testing.assert_allclose(np.asarray(samples.psi), psi, rtol=1e-6)
    w = np.abs(psi)  # |psi|^2 / prob
    np.testing.assert_allclose(np.asarray(samples.r), w / w.mean(), rtol=1e-5)
    ref = (_dense(H) @ psi) / psi
    assert np.allclose(Oloc, ref, rtol=1e-4, atol=1e-5)


def test_oloc_keeps_given_psi():
    # A given psi is used as it is, while r is updated from it.
    Chain(4, boundary=1)
    H = Heisenberg(J=1.0)
    state = Variational(RBM_Dense(features=4), max_parallel=2)
    spins = to_distributed_array(_full_basis_samples())
    psi = state(spins)
    samples = Samples(spins, psi=psi, prob=jnp.ones(spins.shape[0]))
    H.Oloc(state, samples)
    assert samples.psi is psi
    w = np.abs(np.asarray(psi)) ** 2
    np.testing.assert_allclose(np.asarray(samples.r), w / w.mean(), rtol=1e-5)


def test_diagonal_oloc_fills_psi_only_for_reweighting():
    # A diagonal operator doesn't evaluate any connected configuration, so psi is
    # evaluated by a separate forward pass only if prob requires it for r.
    Chain(4, boundary=1)
    Z = sigma_z(0) @ sigma_z(1)
    state = Variational(RBM_Dense(features=4))
    spins = to_distributed_array(_full_basis_samples())
    psi = np.asarray(state(spins))
    calls = []
    direct_forward = state._direct_forward
    state._direct_forward = lambda *args: (calls.append(1), direct_forward(*args))[1]

    samples = Samples(spins)
    Z.Oloc(state, samples)
    assert samples.psi is None and samples.r is None
    assert calls == []

    samples = Samples(spins, prob=jnp.ones(spins.shape[0]))
    Z.Oloc(state, samples)
    assert calls != []
    np.testing.assert_allclose(np.asarray(samples.psi), psi, rtol=1e-6)
    w = np.abs(psi) ** 2
    np.testing.assert_allclose(np.asarray(samples.r), w / w.mean(), rtol=1e-5)


def test_oloc_skips_term_without_connections():
    # sigma^+ has no non-zero connection from the all-up configuration. With a given
    # psi, the forward pass of the empty connected configurations is skipped and only
    # the diagonal term is left.
    Chain(4, boundary=1)
    state = Variational(RBM_Dense(features=4), max_parallel=2)
    spins = to_distributed_array(jnp.ones((2 * jax.device_count(), 4), jnp.int8))
    samples = Samples(spins, psi=state(spins))
    Oloc = (sigma_p(0) + sigma_z(1)).Oloc(state, samples)
    np.testing.assert_allclose(np.asarray(Oloc), 1.0)


def test_expectation_reweights_prob_only_samples(x64):
    # Samples carrying only prob get psi and r filled in by Oloc before r is used,
    # for off-diagonal and diagonal operators. All configurations drawn once follow
    # the uniform prob, so the reweighted mean is the exact expectation value.
    use_dtype(jnp.complex128)
    Chain(4, boundary=1)
    symm = Identity()
    symm.basis_make()
    rng = np.random.default_rng(0)
    psi = rng.standard_normal(symm.basis.Ns) + 1j * rng.standard_normal(symm.basis.Ns)
    state = DenseState(jnp.asarray(psi))
    spins = to_distributed_array(jnp.asarray(ints_to_array(symm.basis.states)))
    prob = to_distributed_array(jnp.ones(symm.basis.Ns))
    for op in (Heisenberg(J=1.0), sigma_z(0) @ sigma_z(1)):
        samples = Samples(spins, prob=prob)
        Omean = op.expectation(state, samples)
        exact = (psi.conj() @ _dense(op) @ psi) / (psi.conj() @ psi)
        assert np.isclose(Omean, exact, atol=1e-10)
        assert samples.psi is not None and samples.r is not None


def test_get_conn_size():
    ndev = jax.device_count()
    # identical 6x3 blocks on every device so the expected count is
    # independent of the device number
    block = np.ones((6, 3), np.float32)
    block[0, :] = np.nan  # invalid connections
    block[1, 1] = 0.0  # explicit zeros are not valid connections
    H_conn = jnp.asarray(np.tile(block, (ndev, 1)))  # 14 valid per device

    assert int(_get_conn_size(H_conn, None, False)) == 14
    # counts the whole device batch even when it exceeds forward_chunk,
    # then rounds up to a multiple of forward_chunk
    assert int(_get_conn_size(H_conn, 2, False)) == 14
    assert int(_get_conn_size(H_conn, 4, False)) == 16
    # psi_needed reserves one extra slot per sample on each device
    assert int(_get_conn_size(H_conn, None, True)) == 20
    assert int(_get_conn_size(H_conn, 4, True)) == 20


# --- constant shift is explicitly unsupported ---


def test_nonzero_constant_shift_raises():
    Chain(2, boundary=1)
    H = sigma_z(0) @ sigma_z(1)
    with pytest.raises(ValueError):
        H + 1.0
    # adding zero is a no-op and allowed
    assert (H + 0.0) is H
