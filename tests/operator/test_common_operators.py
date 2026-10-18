import itertools
import numpy as np
import pytest
import quantax as qtx
from quantax.sites import Chain
from quantax.operator import Heisenberg, Ising, Hubbard, tJ, tV
from quantax.symmetry import Identity
from quantax.utils import ints_to_array
from _op_helpers import dense

PT = qtx.PARTICLE_TYPE


def _basis_configs() -> np.ndarray:
    """The ±1 configurations of the basis states, in QuSpin basis order."""
    symm = Identity()
    symm.basis_make()
    return ints_to_array(symm.basis.states)


# --- explicit Pauli reference for the spin path ---

_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SY = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_I2 = np.eye(2, dtype=complex)


def _embed(op: np.ndarray, i: int, L: int) -> np.ndarray:
    """Place a single-site 2x2 operator at site ``i`` of an ``L``-site chain."""
    mats = [op if k == i else _I2 for k in range(L)]
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _pair(op: np.ndarray, i: int, j: int, L: int) -> np.ndarray:
    return _embed(op, i, L) @ _embed(op, j, L)


# --- particle-type guards ---


@pytest.mark.parametrize(
    "build, particle_type",
    [
        (lambda: Heisenberg(), PT.spinful_fermion),
        (lambda: Ising(), PT.spinful_fermion),
        (lambda: Hubbard(U=1.0), PT.spin),
        (lambda: tJ(J=1.0), PT.spin),
        (lambda: tV(V=1.0), PT.spin),
    ],
)
def test_wrong_particle_type_raises(build, particle_type):
    Chain(4, boundary=1, particle_type=particle_type)
    with pytest.raises(ValueError):
        build()


# --- length-mismatch guards between couplings and neighbor orders ---


@pytest.mark.parametrize(
    "build, particle_type",
    [
        (lambda: Heisenberg(J=[1.0, 0.5], n_neighbor=1), PT.spin),
        (lambda: Hubbard(U=1.0, t=[1.0, 0.5], n_neighbor=1), PT.spinful_fermion),
        (lambda: tJ(J=[1.0, 0.5], J_neighbor=1), PT.spinful_fermion),
        (lambda: tJ(J=1.0, t=[1.0, 0.5], t_neighbor=1), PT.spinful_fermion),
        (lambda: tV(V=[1.0, 0.5], V_neighbor=1), PT.spinless_fermion),
        (lambda: tV(V=1.0, t=[1.0, 0.5], t_neighbor=1), PT.spinless_fermion),
    ],
)
def test_length_mismatch_raises(build, particle_type):
    Chain(4, boundary=1, particle_type=particle_type)
    with pytest.raises(ValueError):
        build()


# --- complex hopping is explicitly unsupported (scalar and list forms) ---


@pytest.mark.parametrize("t", [1.0j, [1.0j]])
@pytest.mark.parametrize(
    "build, particle_type",
    [
        (lambda t: Hubbard(U=1.0, t=t), PT.spinful_fermion),
        (lambda t: tJ(J=1.0, t=t), PT.spinful_fermion),
        (lambda t: tV(V=1.0, t=t), PT.spinless_fermion),
    ],
)
def test_complex_t_raises(build, particle_type, t):
    Chain(4, boundary=1, particle_type=particle_type)
    with pytest.raises(NotImplementedError):
        build(t)


# --- spin models against the explicit Pauli construction ---

_NN = [(0, 1), (1, 2), (2, 3), (3, 0)]  # periodic 4-site chain
_NNN = [(0, 2), (1, 3)]


@pytest.mark.parametrize("msr", [False, True])
def test_heisenberg_matches_explicit_paulis(msr):
    L = 4
    Chain(L, boundary=1)
    H = Heisenberg(J=[1.0, 0.5], n_neighbor=[1, 2], msr=msr)

    # MSR flips the sign of the transverse (xx+yy) part on nearest-neighbor bonds,
    # leaving the zz part and the further bonds untouched.
    def bond(i, j, sign=1):
        xy = _pair(_SX, i, j, L) + _pair(_SY, i, j, L)
        return sign * xy + _pair(_SZ, i, j, L)

    ref = sum(bond(i, j, -1 if msr else 1) for i, j in _NN)
    ref += 0.5 * sum(bond(i, j) for i, j in _NNN)
    assert np.allclose(dense(H), ref, atol=1e-5)


def test_ising_matches_explicit_paulis():
    L = 4
    Chain(L, boundary=1)
    H = Ising(h=0.7, J=1.3)
    ref = -1.3 * sum(_pair(_SZ, i, j, L) for i, j in _NN)
    ref -= 0.7 * sum(_embed(_SX, i, L) for i in range(L))
    assert np.allclose(dense(H), ref, atol=1e-5)


# --- fermionic models against free fermions and their interactions ---


def _assert_free_fermion_sectors(H: np.ndarray, L: int, boundary: int, t: list):
    """
    Each sector of the particle numbers of each spin, for hopping ``t[d - 1]`` over
    distance ``d`` on an ``L``-site ring, has the sums of the occupied single-particle
    energies eps(k) = -2 sum_d t_d cos(d k) as its spectrum, with k shifted by pi/L
    for APBC.
    """
    k = 2 * np.pi * (np.arange(L) + (boundary == -1) / 2) / L
    eps = -2 * sum(td * np.cos(d * k) for d, td in enumerate(t, start=1))
    s = _basis_configs() > 0
    n = np.sum(s.reshape(s.shape[0], -1, L), axis=2)  # (states, spins)
    for numbers in np.unique(n, axis=0):
        sector = np.flatnonzero(np.all(n == numbers, axis=1))
        energies = [[sum(c) for c in itertools.combinations(eps, N)] for N in numbers]
        spectrum = np.sort([sum(e) for e in itertools.product(*energies)])
        eigs = np.linalg.eigvalsh(H[np.ix_(sector, sector)])
        assert np.allclose(eigs, spectrum, atol=1e-5)


# t-t' hopping on the 5-site ring: it is not bipartite, so the spectrum of a sector
# also fixes the signs of t and t', and the bonds across the boundary hop over the
# middle modes, probing Jordan-Wigner signs.
_T = [0.8, 0.3]


@pytest.mark.parametrize("boundary", [1, -1])  # PBC and APBC
def test_hubbard_free_spectrum_and_onsite_interaction(boundary):
    L = 5
    # two fermions of each spin give Jordan-Wigner signs in both spin sectors, in a
    # basis small enough for cheap dense matrices
    Chain(L, boundary=boundary, particle_type=PT.spinful_fermion, Nparticles=(2, 2))
    H0 = dense(Hubbard(U=0.0, t=_T, n_neighbor=[1, 2]))
    assert np.allclose(H0, H0.conj().T)
    _assert_free_fermion_sectors(H0, L, boundary, _T)

    # U counts the doubly occupied sites (spin-down modes after the spin-up ones)
    s = _basis_configs() > 0
    ndouble = np.sum(s[:, :L] & s[:, L:], axis=1)
    HU = dense(Hubbard(U=3.0, t=_T, n_neighbor=[1, 2])) - H0
    assert np.allclose(HU, np.diag(3.0 * ndouble), atol=1e-5)


@pytest.mark.parametrize("boundary", [1, -1])
def test_tV_free_spectrum_and_neighbor_interaction(boundary):
    L = 5
    Chain(L, boundary=boundary, particle_type=PT.spinless_fermion)
    H0 = dense(tV(V=0.0, t=_T, t_neighbor=[1, 2]))
    assert np.allclose(H0, H0.conj().T)
    _assert_free_fermion_sectors(H0, L, boundary, _T)

    # V_d couples the occupations at distance d, without boundary signs
    s = _basis_configs() > 0
    V = [1.5, 0.4]
    pairs = [np.sum(s & np.roll(s, -d, axis=1), axis=1) for d in (1, 2)]
    HV = dense(tV(V=V, V_neighbor=[1, 2], t=_T, t_neighbor=[1, 2])) - H0
    assert np.allclose(HV, np.diag(V[0] * pairs[0] + V[1] * pairs[1]), atol=1e-5)


def test_tJ_exchange_and_hopping():
    L = 5
    Chain(L, boundary=-1, particle_type=PT.spinful_fermion, double_occ=False)
    J = [0.4, 0.1]
    H = dense(tJ(J=J, J_neighbor=[1, 2], t=_T, t_neighbor=[1, 2]))
    assert np.allclose(H, H.conj().T)
    s = _basis_configs() > 0
    n = np.sum(s, axis=1)

    # at half filling no hopping is allowed, leaving the J1-J2 ring of spins with
    # sum_d J_d sum_i (S_i S_{i+d} - 1/4). Its spectrum is basis independent, and
    # the ring is not bipartite, so it fixes the sign of the exchange.
    half = np.flatnonzero(n == L)
    ref = sum(
        Jd
        * (sum(_pair(P, i, (i + d) % L, L) for P in (_SX, _SY, _SZ)) - np.eye(2**L))
        / 4
        for d, Jd in enumerate(J, start=1)
        for i in range(L)
    )
    eigs = np.linalg.eigvalsh(H[np.ix_(half, half)])
    assert np.allclose(eigs, np.linalg.eigvalsh(ref), atol=1e-5)

    # a single particle hops by -t_d over distance d, with -1 across the
    # anti-periodic boundary
    hop = np.zeros((L, L))
    for d, td in enumerate(_T, start=1):
        for i in range(L):
            j, sign = (i + d) % L, (-1 if i + d >= L else 1)
            hop[i, j] = hop[j, i] = -td * sign
    single = np.flatnonzero(n == 1)
    mode = np.argmax(s[single], axis=1)
    site, spin = mode % L, mode // L
    ref = np.where(spin[:, None] == spin[None, :], hop[np.ix_(site, site)], 0)
    assert np.allclose(H[np.ix_(single, single)], ref, atol=1e-5)
