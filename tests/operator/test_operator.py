import numpy as np
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
from quantax.state import DenseState
from quantax.symmetry import Identity
from quantax.utils import ints_to_array
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


# --- constant shift is explicitly unsupported ---


def test_nonzero_constant_shift_raises():
    Chain(2, boundary=1)
    H = sigma_z(0) @ sigma_z(1)
    with pytest.raises(ValueError):
        H + 1.0
    # adding zero is a no-op and allowed
    assert (H + 0.0) is H
