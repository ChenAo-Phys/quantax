import itertools
import pytest
import numpy as np
import jax.numpy as jnp
import quantax as qtx
from quantax.sites import Grid, Square, Chain
from quantax.symmetry import Translation, TransND, Z2Inversion, C4v
from _dtype import use_dtype


def _fermion_configs(Nmodes, Npart):
    combs = list(itertools.combinations(range(Nmodes), Npart))
    configs = np.full((len(combs), Nmodes), -1, dtype=np.int8)
    for i, c in enumerate(combs):
        configs[i, list(c)] = 1
    return configs


def _dense_projector(symm, configs):
    # P[s', s] = sum_g coeff_g(s) [T_g s == s'], built by feeding unit vectors
    # to `symmetrize`. A correct symmetry sector gives an idempotent projector.
    idx = {tuple(c.tolist()): i for i, c in enumerate(configs)}
    M = len(configs)
    P = np.zeros((M, M), complex)
    for j, s in enumerate(configs):
        sj = jnp.asarray(s)
        gathered = np.asarray(symm.get_symm_spins(sj))
        for r in range(gathered.shape[0]):
            e = np.zeros(gathered.shape[0], complex)
            e[r] = 1.0
            coeff = complex(np.asarray(symm.symmetrize(jnp.asarray(e), sj)))
            P[idx[tuple(gathered[r].tolist())], j] += coeff
    return P


# --- __matmul__ Z2 inversion combination rules ---


def test_matmul_z2_inversion_rules():
    Square(2)
    # 0 (no Z2) composes with the other operand's Z2 value
    assert (TransND() @ Z2Inversion(1)).Z2_inversion == 1
    assert (Z2Inversion(-1) @ TransND()).Z2_inversion == -1
    # equal Z2 values are preserved; conflicting ones are rejected
    assert (Z2Inversion(1) @ Z2Inversion(1)).Z2_inversion == 1
    with pytest.raises(ValueError):
        Z2Inversion(1) @ Z2Inversion(-1)


# --- basis: unsupported sectors are rejected ---


def test_basis_rejects_apbc_perm_sign():
    # QuSpin can't represent the non-trivial perm_sign from anti-periodic fermions.
    Chain(4, boundary=-1, particle_type="spinless_fermion", Nparticles=2)
    with pytest.raises(RuntimeError):
        Translation(1).basis


def test_basis_rejects_highdim_character():
    # The 2D E irrep has |character| = 2, which QuSpin can't represent.
    Square(4)
    with pytest.raises(RuntimeError):
        C4v(repr="E").basis


# --- get_symm_spins / nsymm ---


def test_get_symm_spins_z2_doubles():
    Square(4)
    t = TransND() @ Z2Inversion(1)
    out = t.get_symm_spins(jnp.ones(16))
    assert t.nsymm == 32
    assert out.shape == (t.nsymm, 16)


# --- multi-generator APBC fermion signs (regression) ---


@pytest.mark.parametrize("shape", [(4, 2), (3, 2), (4, 4)])
def test_multigen_apbc_perm_sign_matches_matmul(shape):
    # Building the N-D translation group with `_get_perm` must give the same
    # perm_sign as composing 1-D translations with `@`. The old code re-scattered
    # the accumulated signs once per generator, so the two disagreed for fermions
    # with anti-periodic boundaries and >= 2 generators.
    Grid(list(shape), boundary=-1, particle_type="spinless_fermion", Nparticles=2)
    ndim = len(shape)
    A = Translation(np.eye(ndim, dtype=int))  # via _get_perm
    B = Translation(np.eye(ndim, dtype=int)[0])
    for d in range(1, ndim):
        B = B @ Translation(np.eye(ndim, dtype=int)[d])  # via __matmul__
    assert np.array_equal(np.asarray(A._perm), np.asarray(B._perm))
    assert np.allclose(np.asarray(A._character), np.asarray(B._character))
    assert np.array_equal(np.asarray(A._perm_sign), np.asarray(B._perm_sign))


def test_symmetrize_valid_projector_apbc_fermion(x64):
    # 4x2 anti-periodic spinless fermions at even filling: k=0 is a valid sector,
    # so the momentum projector must satisfy P^2 = P. The old multi-generator sign
    # bug produced a non-idempotent operator here.
    use_dtype(jnp.complex128)
    Grid([4, 2], boundary=-1, particle_type="spinless_fermion", Nparticles=2)
    P = _dense_projector(TransND(0), _fermion_configs(8, 2))
    assert np.max(np.abs(P @ P - P)) < 1e-9


def test_symmetrize_valid_projector_pbc_spin(x64):
    # Sanity check on the non-fermionic path: a spin translation sector is a
    # clean projector regardless of the sign machinery.
    use_dtype(jnp.complex128)
    Grid([2, 2], particle_type="spin", Nparticles=(2, 2))
    configs = _fermion_configs(4, 2)  # Sz=0 spin configs share this enumeration
    P = _dense_projector(TransND(0), configs)
    assert np.max(np.abs(P @ P - P)) < 1e-9
