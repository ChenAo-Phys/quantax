import numpy as np
import jax
import jax.numpy as jnp
import pytest
import quantax as qtx
from quantax.sites import Chain
from quantax.symmetry import Identity, Symmetry
from quantax.state import State, DenseState
from quantax.operator import Heisenberg, sigma_z
from quantax.utils import LogArray, ScaleArray, ints_to_array
from _dtype import use_dtype

# The full wavefunction of a `DenseState` is stored in QuSpin `basis.states` order.
# With the trivial `Identity` symmetry the stored amplitudes equal the input `psi`
# (symm_norm = 1), so a plain numpy array is the ground truth for every method.
#
# `DenseState` accepts any `PsiArray` for `psi`; the `rep` fixture exercises the three
# representations (plain array / LogArray / ScaleArray) so the protocol-based
# materialization in norm / __array__ / __jax_array__ is covered for all of them.


@pytest.fixture(params=["plain", "log", "scale"])
def rep(request):
    """Wrap a dense array into one of the three `PsiArray` representations."""

    def wrap(x):
        x = jnp.asarray(x)
        if request.param == "log":
            return LogArray.from_value(x)
        if request.param == "scale":
            return ScaleArray.from_value(x)
        return x

    return wrap


def _psi(dim: int, *, cplx: bool = True, seed: int = 0, scale: float = 1.0):
    """A reproducible random dense wavefunction as a plain numpy array."""
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(dim)
    if cplx:
        psi = psi + 1j * rng.standard_normal(dim)
    return psi * scale


# ====================================================================
# Abstract State base class
# ====================================================================


def test_default_symmetry_is_identity():
    Chain(3, boundary=1)
    state = State()
    # Identity() is a memoized factory returning the singleton trivial symmetry
    assert isinstance(state.symm, Symmetry)
    assert state.symm is Identity()


def test_explicit_symmetry_is_stored():
    Chain(3, boundary=1)
    symm = Identity()
    assert State(symm).symm is symm


def test_geometry_properties_delegate_to_symm():
    Chain(4, boundary=1)
    state = State()
    assert state.Nsites == state.symm.Nsites == 4
    assert state.Nmodes == state.symm.Nmodes
    assert state.nsymm == state.symm.nsymm
    assert state.Nparticles == state.symm.Nparticles
    assert state.basis is state.symm.basis


def test_base_defaults():
    Chain(3, boundary=1)
    state = State()
    assert state.use_ref is False
    assert state.required_update_modes == ()
    assert state.dtype == qtx.get_default_dtype()


def test_abstract_methods_raise_not_implemented():
    Chain(3, boundary=1)
    state = State()
    s = jnp.ones((1, state.Nmodes))
    with pytest.raises(NotImplementedError):
        state(s)
    with pytest.raises(NotImplementedError):
        state.init_internal(s)
    with pytest.raises(NotImplementedError):
        state.ref_forward(s, s, {}, None)
    with pytest.raises(NotImplementedError):
        state.segment_ref_forward(s, s, {}, jnp.array([0]), None)


# ====================================================================
# DenseState construction and evaluation
# ====================================================================


def test_psi_roundtrip_and_dtype(x64):
    use_dtype(jnp.complex128)
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    psi = _psi(symm.basis.Ns)
    state = DenseState(jnp.asarray(psi), symm)
    np.testing.assert_allclose(np.asarray(state.psi), psi)
    assert state.psi.dtype == jnp.complex128
    assert state.dtype == jnp.complex128


def test_size_mismatch_raises():
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    with pytest.raises(ValueError):
        DenseState(jnp.ones(symm.basis.Ns + 1), symm)


def test_getitem_and_call_match_psi(x64):
    use_dtype(jnp.complex128)
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    basis = symm.basis
    psi = _psi(basis.Ns)
    state = DenseState(jnp.asarray(psi), symm)
    # __getitem__ slices by basis integers; __call__ converts ±1 fock states first
    np.testing.assert_allclose(np.asarray(state[basis.states]), psi, rtol=1e-6)
    fock = ints_to_array(basis.states)
    np.testing.assert_allclose(np.asarray(state(fock)), psi, rtol=1e-6)


def test_getitem_missing_basis_int_is_zero(x64):
    # An integer outside the Hilbert space must evaluate to 0 (not found).
    use_dtype(jnp.complex128)
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    state = DenseState(jnp.asarray(_psi(symm.basis.Ns)), symm)
    out_of_range = np.array([2**3], dtype=symm.basis.states.dtype)  # 2^Nsites
    assert np.asarray(state[out_of_range]).item() == 0.0


# ====================================================================
# norm  (returns a plain real scalar for every representation)
# ====================================================================


def test_norm_matches_numpy_for_all_reps(rep, x64):
    use_dtype(jnp.complex128)
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    psi = _psi(symm.basis.Ns)
    state = DenseState(rep(psi), symm)
    n = state.norm()
    assert isinstance(n, jax.Array)
    assert jnp.issubdtype(n.dtype, jnp.floating)  # real scalar, usable as a number
    assert np.isclose(float(n), np.linalg.norm(psi), rtol=1e-6)


@pytest.mark.parametrize("ord", [1, 2, 3])
def test_norm_ord(ord, x64):
    use_dtype(jnp.complex128)
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    psi = _psi(symm.basis.Ns)
    state = DenseState(jnp.asarray(psi), symm)
    assert np.isclose(float(state.norm(ord)), np.linalg.norm(psi, ord), rtol=1e-6)


def test_normalize_returns_unit_norm(rep, x64):
    use_dtype(jnp.complex128)
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    state = DenseState(rep(_psi(symm.basis.Ns, scale=1e3)), symm)
    normalized = state.normalize()
    assert normalized is not state
    assert np.isclose(float(normalized.norm()), 1.0, rtol=1e-6)


def test_normalize_inplace(rep, x64):
    use_dtype(jnp.complex128)
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    state = DenseState(rep(_psi(symm.basis.Ns, scale=1e3)), symm)
    assert not np.isclose(float(state.norm()), 1.0)
    state.normalize_()
    assert np.isclose(float(state.norm()), 1.0, rtol=1e-6)


# ====================================================================
# numpy / jax array protocols
# ====================================================================


def test_array_protocol_materializes_all_reps(rep, x64):
    use_dtype(jnp.complex128)
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    psi = _psi(symm.basis.Ns)
    state = DenseState(rep(psi), symm)

    arr = np.asarray(state)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_allclose(arr, psi, rtol=1e-6)

    jarr = jnp.asarray(state)
    assert isinstance(jarr, jax.Array)
    np.testing.assert_allclose(np.asarray(jarr), psi, rtol=1e-6)


def test_array_honors_dtype(x64):
    use_dtype(jnp.complex128)
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    state = DenseState(jnp.asarray(_psi(symm.basis.Ns)), symm)
    assert np.asarray(state, dtype=np.complex64).dtype == np.complex64


def test_array_copy_does_not_alias_internal_psi(x64):
    # np.array(...) requests copy=True by default; the returned buffer must be
    # independent of the state's internal `_psi` (regression for a numpy-backed psi).
    use_dtype(jnp.complex128)
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    psi = _psi(symm.basis.Ns)
    state = DenseState(np.asarray(psi), symm)
    copied = np.array(state)
    copied[0] = 12345.0
    np.testing.assert_allclose(np.asarray(state.psi), psi, rtol=1e-6)


# ====================================================================
# arithmetic
# ====================================================================


def test_neg_add_sub_mul(x64):
    use_dtype(jnp.complex128)
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    a, b = _psi(symm.basis.Ns, seed=1), _psi(symm.basis.Ns, seed=2)
    sa, sb = DenseState(jnp.asarray(a), symm), DenseState(jnp.asarray(b), symm)

    np.testing.assert_allclose(np.asarray((-sa).psi), -a, rtol=1e-6)
    np.testing.assert_allclose(np.asarray((sa + sb).psi), a + b, rtol=1e-6)
    np.testing.assert_allclose(np.asarray((sa - sb).psi), a - b, rtol=1e-6)
    np.testing.assert_allclose(np.asarray((sa * 2.0).psi), a * 2.0, rtol=1e-6)
    np.testing.assert_allclose(np.asarray((2.0 * sa).psi), a * 2.0, rtol=1e-6)


def test_add_sub_with_non_densestate_raises(x64):
    use_dtype(jnp.complex128)
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    state = DenseState(jnp.asarray(_psi(symm.basis.Ns)), symm)
    with pytest.raises(RuntimeError):
        state + 1.0
    with pytest.raises(RuntimeError):
        state - 1.0


# ====================================================================
# overlap (@)
# ====================================================================


def test_matmul_overlap_matches_vdot(x64):
    use_dtype(jnp.complex128)
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    a, b = _psi(symm.basis.Ns, seed=3), _psi(symm.basis.Ns, seed=4)
    sa, sb = DenseState(jnp.asarray(a), symm), DenseState(jnp.asarray(b), symm)
    assert np.isclose(sa @ sb, np.vdot(a, b), rtol=1e-6)
    # <a|a> is real and equals the squared 2-norm
    assert np.isclose((sa @ sa).real, float(sa.norm()) ** 2, rtol=1e-6)


def test_matmul_with_non_state_raises_type_error(x64):
    use_dtype(jnp.complex128)
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    state = DenseState(jnp.asarray(_psi(symm.basis.Ns)), symm)
    assert state.__matmul__(5) is NotImplemented
    with pytest.raises(TypeError):
        state @ 5


def test_operator_expectation_via_matmul_is_ground_energy(x64):
    # <gs|H|gs> through the dense path must equal the ground-state energy.
    use_dtype(jnp.complex128)
    Chain(4, boundary=1)
    H = Heisenberg(J=1.0)
    w, v = H.diagonalize(k=1)
    gs = DenseState(np.asarray(v[:, 0]))
    energy = gs @ H @ gs
    assert np.isclose(complex(energy).real, float(w[0]), atol=1e-6)
    assert np.isclose(complex(energy).imag, 0.0, atol=1e-6)


# ====================================================================
# todense
# ====================================================================


def test_densestate_todense_is_identity(x64):
    use_dtype(jnp.complex128)
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    state = DenseState(jnp.asarray(_psi(symm.basis.Ns)), symm)
    # DenseState short-circuits todense() to itself for the same symmetry
    assert state.todense() is state
    assert state.todense(state.symm) is state


# ====================================================================
# expectation (delegates to Operator.expectation, exposing return_var)
# ====================================================================


def test_expectation_delegates_to_operator(x64):
    use_dtype(jnp.complex128)
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    state = DenseState(jnp.asarray(_psi(symm.basis.Ns)), symm)
    H = sigma_z(0) @ sigma_z(1)
    samples = jnp.asarray(ints_to_array(symm.basis.states))

    mean = state.expectation(H, samples)
    assert np.isclose(mean, H.expectation(state, samples))


def test_expectation_return_var(x64):
    use_dtype(jnp.complex128)
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    state = DenseState(jnp.asarray(_psi(symm.basis.Ns)), symm)
    H = sigma_z(0) @ sigma_z(1)
    samples = jnp.asarray(ints_to_array(symm.basis.states))

    mean = state.expectation(H, samples)
    mean_v, var = state.expectation(H, samples, return_var=True)
    assert isinstance(var, float)
    assert np.isclose(mean, mean_v)
    # matches calling the operator directly with return_var
    mean_o, var_o = H.expectation(state, samples, return_var=True)
    assert np.isclose(var, var_o)
