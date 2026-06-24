import numpy as np
import jax.numpy as jnp
import pytest
import quantax as qtx
from quantax.sites import Chain
from quantax.operator import Operator, Heisenberg, Ising, Hubbard, tJ, tV
from _dtype import use_dtype

PT = qtx.PARTICLE_TYPE


def _dense(op: Operator) -> np.ndarray:
    """Dense matrix of an operator as a plain numpy array."""
    return np.asarray(op.todense())


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


def _sdot(i: int, j: int, L: int) -> np.ndarray:
    r""":math:`\boldsymbol{\sigma}_i \cdot \boldsymbol{\sigma}_j` on an ``L``-site chain."""
    return sum(_embed(s, i, L) @ _embed(s, j, L) for s in (_SX, _SY, _SZ))


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


# --- Heisenberg correctness against the explicit Pauli construction ---


def test_heisenberg_matches_explicit_paulis(x64):
    use_dtype(jnp.complex128)
    L = 3
    Chain(L, boundary=0)  # open chain -> bonds (0,1) and (1,2)
    H = Heisenberg(J=1.0)
    ref = _sdot(0, 1, L) + _sdot(1, 2, L)
    assert np.allclose(_dense(H), ref, atol=1e-5)


def test_heisenberg_msr_flips_offdiagonal(x64):
    use_dtype(jnp.complex128)
    L = 3
    Chain(L, boundary=0)
    H = Heisenberg(J=1.0, msr=True)

    # MSR flips the sign of the transverse (xx+yy) part on nearest-neighbor bonds,
    # leaving the zz part untouched.
    def _msr_bond(i, j):
        xy = _embed(_SX, i, L) @ _embed(_SX, j, L) + _embed(_SY, i, L) @ _embed(
            _SY, j, L
        )
        zz = _embed(_SZ, i, L) @ _embed(_SZ, j, L)
        return -xy + zz

    ref = _msr_bond(0, 1) + _msr_bond(1, 2)
    assert np.allclose(_dense(H), ref, atol=1e-5)


def test_heisenberg_multi_neighbor_hermitian(x64):
    use_dtype(jnp.complex128)
    Chain(6, boundary=1)
    H = Heisenberg(J=[1.0, 0.5], n_neighbor=[1, 2])
    assert np.allclose(_dense(H.H), _dense(H).conj().T, atol=1e-5)


# --- fermionic models are Hermitian (guards hopping/exchange sign bugs) ---


@pytest.mark.parametrize("boundary", [1, -1])  # PBC and APBC
def test_hubbard_hermitian(boundary):
    Chain(4, boundary=boundary, particle_type=PT.spinful_fermion)
    H = Hubbard(U=8.0, t=1.0)
    assert np.allclose(_dense(H.H), _dense(H).conj().T, atol=1e-5)


@pytest.mark.parametrize("boundary", [1, -1])
def test_tJ_hermitian(boundary):
    Chain(4, boundary=boundary, particle_type=PT.spinful_fermion)
    H = tJ(J=0.4, t=1.0)
    assert np.allclose(_dense(H.H), _dense(H).conj().T, atol=1e-5)


@pytest.mark.parametrize("boundary", [1, -1])
def test_tV_hermitian(boundary):
    Chain(4, boundary=boundary, particle_type=PT.spinless_fermion)
    H = tV(V=2.0, t=1.0)
    assert np.allclose(_dense(H.H), _dense(H).conj().T, atol=1e-5)
