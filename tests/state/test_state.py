import itertools
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from quantax.sites import Chain
from quantax.symmetry import Identity, Translation
from quantax.state import State, DenseState
from quantax.operator import Ising
from quantax.utils import LogArray, ints_to_array
from _dtype import use_dtype

# The full wavefunction of a `DenseState` is stored in QuSpin `basis.states` order.
# With the trivial `Identity` symmetry the stored amplitudes equal the input `psi`
# (symm_norm = 1), so a plain numpy array is the ground truth for every method.
#
# `DenseState` accepts any `PsiArray` for `psi`; the `rep` fixture exercises the two
# representations (plain array / LogArray) so the protocol-based materialization in
# norm / __array__ / __jax_array__ is covered for both of them.


@pytest.fixture(params=["plain", "log"])
def rep(request):
    """Wrap a dense array into one of the two `PsiArray` representations."""

    def wrap(x):
        x = jnp.asarray(x)
        if request.param == "log":
            return LogArray.from_value(x)
        return x

    return wrap


def _psi(dim: int, *, cplx: bool = True, seed: int = 0, scale: float = 1.0):
    """A reproducible random dense wavefunction as a plain numpy array."""
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(dim)
    if cplx:
        psi = psi + 1j * rng.standard_normal(dim)
    return psi * scale


def _dense(*seeds: int) -> list[DenseState]:
    """Random complex `DenseState`s on a 3-site chain with the Identity symmetry."""
    use_dtype(jnp.complex128)
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    return [DenseState(jnp.asarray(_psi(symm.basis.Ns, seed=k)), symm) for k in seeds]


def _all_fock(nsites: int) -> np.ndarray:
    """All ±1 configurations, enumerated independently of the QuSpin basis order."""
    return np.array(list(itertools.product([1, -1], repeat=nsites)), dtype=np.int8)


class _FuncState(State):
    """A user-defined state implementing only `__call__`, evaluated in numpy."""

    def __init__(self, fn, symm=None):
        super().__init__(symm)
        self._fn = fn

    def __call__(self, s):
        return self._fn(np.asarray(s))


def _product_fn(nsites: int, cplx: bool, seed: int = 0):
    """psi(s) = prod_i (a_i + b_i s_i): a generic wavefunction without any symmetry."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((2, nsites))
    if cplx:
        a = a + 1j * rng.standard_normal((2, nsites))
    return lambda s: np.prod(a[0] + a[1] * s.reshape(-1, nsites), axis=-1)


def _momentum_projected(fn, nsites: int, sector: int):
    """
    psi(s) = (1/L) sum_t chi^t fn(T^t s) with T s = roll(s, -1) and
    chi = exp(-2i pi sector / L), the convention of `Symmetry.symmetrize`.
    Then psi(T s) = conj(chi) psi(s).
    """
    chi = np.exp(-2j * np.pi * sector / nsites)

    def projected(s):
        s = s.reshape(-1, nsites)
        rolled = (chi**t * fn(np.roll(s, -t, axis=1)) for t in range(nsites))
        return sum(rolled) / nsites

    return projected


# ====================================================================
# Generic State methods, derived from `__call__` alone
# ====================================================================


def test_generic_state_derives_dense_methods(x64):
    # A State subclass that only implements `__call__` gets the lookup, todense,
    # norm and overlap from the base class. The references sum over an independent
    # enumeration of all configurations, so they don't rely on the basis order.
    use_dtype(jnp.complex128)
    Chain(3, boundary=1)
    fn = _product_fn(3, cplx=True)
    state = _FuncState(fn)
    fock = _all_fock(3)
    psi = fn(fock)

    assert state.use_ref is False and state.required_update_modes == ()
    np.testing.assert_allclose(np.asarray(state.fast_forward(fock)), psi)

    dense = state.todense()  # evaluates state[basis.states]
    assert isinstance(dense, DenseState) and dense.symm is state.symm
    np.testing.assert_allclose(np.asarray(dense(fock)), psi, rtol=1e-12)
    assert np.isclose(float(state.norm()), np.linalg.norm(psi), rtol=1e-12)
    fn_other = _product_fn(3, cplx=True, seed=1)
    overlap = np.vdot(psi, fn_other(fock))
    assert np.isclose(state @ _FuncState(fn_other), overlap, rtol=1e-12)


# ====================================================================
# DenseState construction and evaluation
# ====================================================================


def test_size_mismatch_raises():
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    with pytest.raises(ValueError):
        DenseState(jnp.ones(symm.basis.Ns + 1), symm)


def test_densestate_lookup_by_basis_ints_and_fock_states(x64):
    use_dtype(jnp.complex128)
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    basis = symm.basis
    psi = _psi(basis.Ns, cplx=False)
    state = DenseState(jnp.asarray(psi))
    assert state.symm is symm  # Identity by default
    # psi is cast to the default dtype
    assert state.psi.dtype == state.dtype == jnp.complex128

    # __getitem__ slices by basis integers keeping the batch shape; __call__
    # converts ±1 fock states first
    ints = basis.states.reshape(2, -1)
    np.testing.assert_allclose(np.asarray(state[ints]), psi.reshape(2, -1))
    fock = ints_to_array(basis.states)
    np.testing.assert_allclose(np.asarray(state(fock)), psi)

    # An integer outside the Hilbert space evaluates to 0 (not found).
    out_of_range = np.array([2**3], dtype=basis.states.dtype)  # 2^Nsites
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
    for ord in (1, 3):
        assert np.isclose(float(state.norm(ord)), np.linalg.norm(psi, ord), rtol=1e-6)


def test_normalize(rep, x64):
    use_dtype(jnp.complex128)
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    psi = _psi(symm.basis.Ns, scale=1e3)
    state = DenseState(rep(psi), symm)

    normalized = state.normalize()
    assert normalized is not state
    np.testing.assert_allclose(
        np.asarray(normalized), psi / np.linalg.norm(psi), rtol=1e-6
    )
    assert np.isclose(float(state.norm()), np.linalg.norm(psi), rtol=1e-6)

    state.normalize_()  # in place
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
    assert np.asarray(state, dtype=np.complex64).dtype == np.complex64

    jarr = jnp.asarray(state)
    assert isinstance(jarr, jax.Array)
    np.testing.assert_allclose(np.asarray(jarr), psi, rtol=1e-6)


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
    sa, sb = _dense(1, 2)
    a, b = np.asarray(sa.psi), np.asarray(sb.psi)

    np.testing.assert_allclose(np.asarray((-sa).psi), -a, rtol=1e-6)
    np.testing.assert_allclose(np.asarray((sa + sb).psi), a + b, rtol=1e-6)
    np.testing.assert_allclose(np.asarray((sa - sb).psi), a - b, rtol=1e-6)
    c = 2.0 - 0.5j  # complex, so a conjugated scalar would show up
    np.testing.assert_allclose(np.asarray((sa * c).psi), a * c, rtol=1e-6)
    np.testing.assert_allclose(np.asarray((c * sa).psi), a * c, rtol=1e-6)


def test_add_sub_with_incompatible_state_raises(x64):
    (state,) = _dense(0)
    # amplitudes in different symmetric bases can't be added elementwise
    symm = Translation((1,))
    symm.basis_make()
    other_symm = DenseState(jnp.asarray(_psi(symm.basis.Ns)), symm)
    for other in (1.0, other_symm):
        with pytest.raises(RuntimeError):
            state + other
        with pytest.raises(RuntimeError):
            state - other


# ====================================================================
# overlap (@)
# ====================================================================


def test_matmul_overlap_matches_vdot(x64):
    sa, sb = _dense(3, 4)
    a, b = np.asarray(sa.psi), np.asarray(sb.psi)
    assert np.isclose(sa @ sb, np.vdot(a, b), rtol=1e-6)
    # <a|a> is real and equals the squared 2-norm
    assert np.isclose((sa @ sa).real, float(sa.norm()) ** 2, rtol=1e-6)
    # a non-State operand returns NotImplemented, so that e.g. `state @ H` can be
    # handled by `Operator.__rmatmul__`
    assert sa.__matmul__(5) is NotImplemented
    with pytest.raises(TypeError):
        sa @ 5


# ====================================================================
# todense in a symmetric basis
# ====================================================================


@pytest.mark.parametrize(
    "dtype,sector",
    [(jnp.float64, 2), (jnp.complex128, 1)],  # momentum pi (real) and pi/2 (complex)
)
def test_todense_symmetric_basis_matches_full_basis(dtype, sector, x64):
    # A momentum eigenstate stored in QuSpin's symmetric basis (amplitudes of the
    # representatives divided by their symmetry norms) must reconstruct the full
    # wavefunction, keep its norm and energy, and overlap through the full basis.
    # On a 4-site ring the orbits have different sizes, so a wrong symmetry norm or
    # character convention shows up in the reconstruction.
    use_dtype(dtype)
    Chain(4, boundary=1)
    cplx = dtype == jnp.complex128
    projected = _momentum_projected(_product_fn(4, cplx), 4, sector)
    fn = projected if cplx else (lambda s: projected(s).real)
    symm = Translation((1,), sector=sector)
    fock = _all_fock(4)
    psi = fn(fock)

    dense_symm = _FuncState(fn, symm).todense()
    assert dense_symm.basis.Ns < 2**4
    # no copy (and no QuSpin lookup) for the same symmetry
    assert dense_symm.todense() is dense_symm
    assert dense_symm.todense(symm) is dense_symm
    full = dense_symm.todense(Identity())
    assert np.isrealobj(full.psi) != cplx
    np.testing.assert_allclose(np.asarray(full(fock)), psi, atol=1e-12)

    # QuSpin's symmetric basis is orthonormal
    assert np.isclose(float(dense_symm.norm()), np.linalg.norm(psi), rtol=1e-12)
    # Mismatched symmetries are contracted in the full basis, also with a state
    # outside the sector on either side (projecting it onto the sector would be
    # wrong).
    psi_full = np.asarray(full.psi)
    phi = _psi(2**4, cplx=cplx, seed=1)
    assert np.isclose(full @ dense_symm, np.vdot(psi, psi), rtol=1e-12)
    assert np.isclose(dense_symm @ DenseState(phi), np.vdot(psi_full, phi), rtol=1e-12)
    assert np.isclose(DenseState(phi) @ dense_symm, np.vdot(phi, psi_full), rtol=1e-12)
    # <psi|H|psi> through `state @ H @ state` matches the dense matrix, and H
    # commutes with translations, so the sector and the full space agree
    H = Ising(h=0.7)
    Hd = np.asarray(H.todense())
    E_exact = np.vdot(psi_full, Hd @ psi_full) / np.vdot(psi_full, psi_full)
    E_full = (full @ H @ full) / (full @ full)
    E_symm = (dense_symm @ H @ dense_symm) / (dense_symm @ dense_symm)
    assert abs(E_exact) > 1e-3  # not trivially zero; |E| ~ 0.03-1 across seeds
    assert np.isclose(E_full, E_exact, rtol=1e-10)
    assert np.isclose(E_symm, E_exact, rtol=1e-10)
