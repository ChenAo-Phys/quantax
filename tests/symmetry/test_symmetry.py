import itertools
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import quantax as qtx
from quantax.sites import Grid, Square
from quantax.symmetry import (
    Symmetry,
    Identity,
    Translation,
    TransND,
    Z2Inversion,
    SpinInverse,
    C4v,
)
from quantax.utils import ints_to_array
from _dtype import use_dtype


def _configs(Nmodes, Npart):
    combs = list(itertools.combinations(range(Nmodes), Npart))
    configs = np.full((len(combs), Nmodes), -1, dtype=np.int8)
    for i, c in enumerate(combs):
        configs[i, list(c)] = 1
    return configs


def _coefficients(symm, configs):
    # coeff[s, g] with symmetrize(psi, s) = sum_g coeff[s, g] psi[g]; symmetrize is
    # linear in psi, so feeding it the unit vectors reads off the coefficients
    eye = jnp.eye(symm.nsymm, dtype=qtx.get_default_dtype())
    coeff = jax.vmap(lambda s: jax.vmap(symm.symmetrize, in_axes=(0, None))(eye, s))
    return np.asarray(coeff(jnp.asarray(configs)))


def _dense_projector(symm, configs):
    # P[s', s] = sum_g coeff_g(s) [T_g s == s'], so that psi_symm = P^T psi
    coeff = _coefficients(symm, configs)
    gathered = np.asarray(jax.vmap(symm.get_symm_spins)(jnp.asarray(configs)))
    index = {c.tobytes(): i for i, c in enumerate(configs)}
    rows = np.array([[index[g.tobytes()] for g in gs] for gs in gathered])
    P = np.zeros((len(configs), len(configs)), coeff.dtype)
    np.add.at(P, (rows, np.arange(len(configs))[:, None]), coeff)
    return P


def _hopping(configs, bonds, signs):
    # brute-force H = -sum_<ij> sign_ij (c_i^dag c_j + h.c.) in the Fock basis,
    # with the Jordan-Wigner string over the modes between i and j
    index = {c.tobytes(): i for i, c in enumerate(configs)}
    H = np.zeros((len(configs), len(configs)))
    for col, s in enumerate(configs):
        occ = s > 0
        for (i, j), t in zip(bonds, signs):
            for a, b in ((i, j), (j, i)):
                if occ[b] and not occ[a]:
                    new = s.copy()
                    new[b], new[a] = -1, 1
                    string = np.sum(occ[min(a, b) + 1 : max(a, b)])
                    H[index[new.tobytes()], col] -= t * (-1) ** string
    return H


# --- __matmul__ Z2 inversion combination rules ---


def test_matmul_z2_inversion_rules():
    Square(4)
    # 0 (no Z2) composes with the other operand's Z2 value
    assert (Identity() @ Z2Inversion(1)).Z2_inversion == 1
    assert (Z2Inversion(-1) @ Identity()).Z2_inversion == -1
    # equal Z2 values are preserved; conflicting ones are rejected
    assert (Z2Inversion(1) @ Z2Inversion(1)).Z2_inversion == 1
    with pytest.raises(ValueError):
        Z2Inversion(1) @ Z2Inversion(-1)


# --- input validation ---


def test_invalid_generator_and_sector_raise():
    Square(4)
    with pytest.raises(ValueError):
        Symmetry(np.arange(8))  # generator size != Nmodes
    with pytest.raises(ValueError):
        Translation((1, 0), sector=4)  # the x translation only has sectors 0..3
    with pytest.raises(ValueError):
        # sector 1 has complex characters, not representable in the real default
        Translation((1, 0), sector=1)


def test_symmetrize_input_validation():
    Square(4, particle_type="spinless_fermion", Nparticles=2)
    symm = TransND()
    spins = jnp.asarray(_configs(16, 2)[0])
    psi = jnp.ones(symm.nsymm)
    with pytest.raises(RuntimeError):
        symm.symmetrize(psi)  # fermion signs need the configuration
    # batched inputs are a misuse: these functions act on a single configuration
    with pytest.raises(ValueError):
        symm.symmetrize(psi[None], spins)
    with pytest.raises(ValueError):
        symm.symmetrize(psi, spins[None])
    with pytest.raises(ValueError):
        symm.get_symm_spins(spins[None])


def test_symmetrize_real_psi_in_complex_sector(x64):
    # The projection of a real psi on a complex sector is complex: the characters
    # must not be cast to the real psi.dtype, which dropped the imaginary part. The
    # precision of psi is kept, float32 -> complex64 under complex128 characters.
    use_dtype(jnp.complex128)
    Grid([4, 2], particle_type="spinless_fermion", Nparticles=3)
    symm = TransND(sector=(1, 0))
    spins = jnp.asarray(_configs(8, 3)[5])
    psi = jnp.asarray(np.random.default_rng(0).standard_normal(8), jnp.float32)
    got = symm.symmetrize(psi, spins)
    assert got.dtype == jnp.complex64
    expected = symm.symmetrize(psi.astype(jnp.complex64), spins)
    assert abs(expected.imag) > 0.1 * abs(expected)
    assert np.isclose(got, expected, rtol=1e-6)
    # a real sector keeps a real psi real (no imaginary part to differentiate), also
    # at k = pi, whose characters are exactly real
    for sector in [(0, 0), (2, 1)]:
        out = TransND(sector=sector).symmetrize(psi, spins)
        assert out.dtype == jnp.float32
        assert np.isclose(
            out,
            TransND(sector=sector).symmetrize(psi.astype(jnp.complex64), spins),
            rtol=1e-6,
        )


# --- get_symm_spins / symmetrize with Z2 inversion ---


def test_z2_inversion_spins_and_symmetrize():
    use_dtype(jnp.complex64)
    Square(4)
    symm = TransND((1, 2)) @ Z2Inversion(-1)
    assert symm.nsymm == 32
    rng = np.random.default_rng(0)
    spins = jnp.asarray(rng.choice([-1, 1], 16).astype(np.int8))
    out = np.asarray(symm.get_symm_spins(spins))
    # the translated images come first, followed by their spin-flipped copies
    assert np.array_equal(out[:16], np.asarray(spins)[np.asarray(symm._perm)])
    assert np.array_equal(out[16:], -out[:16])
    # psi_symm(s) = sum_g chi_g [psi(T_g s) - psi(-T_g s)] / 32, with chi_g (not
    # its conjugate) as in the docstring and eigenvalue -1 for the spin flip
    psi = rng.standard_normal(32) + 1j * rng.standard_normal(32)
    chi = np.asarray(symm.character)
    expected = (chi @ psi[:16] - chi @ psi[16:]) / 32
    got = symm.symmetrize(jnp.asarray(psi, dtype=jnp.complex64))
    assert np.isclose(got, expected, atol=1e-6)


# --- basis: unsupported sectors are rejected ---


def test_basis_rejects_apbc_perm_sign():
    # QuSpin can't represent the non-trivial perm_sign from anti-periodic fermions.
    Square(4, boundary=-1, particle_type="spinless_fermion", Nparticles=2)
    with pytest.raises(RuntimeError):
        TransND().basis


def test_basis_rejects_highdim_character():
    # The 2D E irrep has |character| = 2, which QuSpin can't represent.
    Square(4)
    with pytest.raises(RuntimeError):
        C4v(repr="E").basis


# --- basis: QuSpin sector agrees with quantax's own symmetrization ---


def _spin_z2():
    Square(4, Nparticles=(8, 8))
    return TransND((2, 0)) @ Z2Inversion(-1), _configs(16, 8)


def _spinless():
    # a complex sector, whose dimension (8) differs from the trivial one (6)
    use_dtype(jnp.complex64)
    Square(4, particle_type="spinless_fermion", Nparticles=2)
    return TransND((1, 0)), _configs(16, 2)


def _spinful():
    # an int Nparticles makes the basis sum over all (Nup, Ndn) splits
    Grid([4, 2], particle_type="spinful_fermion", Nparticles=3, double_occ=False)
    configs = _configs(16, 3)
    double = np.any((configs[:, :8] > 0) & (configs[:, 8:] > 0), axis=1)
    return TransND((2, 1)) @ SpinInverse(-1), configs[~double]


@pytest.mark.parametrize("system", [_spin_z2, _spinless, _spinful])
def test_sector_dimension_matches_quspin(system):
    # The trace of quantax's projector onto the sector is the sector dimension,
    # which QuSpin computes independently. The Z2 / spin inversion eigenvalue and
    # the fermion reordering signs (a translation that swaps two fermions gives -1,
    # removing that orbit from some sectors) both change the count.
    symm, configs = system()
    coeff = _coefficients(symm, configs)
    gathered = np.asarray(jax.vmap(symm.get_symm_spins)(jnp.asarray(configs)))
    fixed = np.all(gathered == configs[:, None, :], axis=-1)
    trace = np.sum(coeff * fixed)
    symm.basis_make()
    assert np.isclose(trace, symm.basis.Ns, atol=1e-3)


def test_spin_basis_magnetized_sector():
    # For spins Nparticles = (Nup, Ndn). Spin flip maps Nup = 9 to Nup = 7 without
    # changing any sector dimension, so check the magnetization of the states.
    Square(4, Nparticles=(9, 7))
    symm = TransND((2, 0))
    symm.basis_make()
    states = ints_to_array(symm.basis.states)
    assert len(states) > 0
    assert np.all(np.sum(states > 0, axis=1) == 9)


# --- multi-generator APBC fermion signs (regression) ---


def test_multigen_apbc_perm_sign_matches_matmul():
    # Building the N-D translation group with `_get_perm` must give the same
    # perm_sign as composing 1-D translations with `@`. The old code re-scattered
    # the accumulated signs once per generator, so the two disagreed for fermions
    # with anti-periodic boundaries and >= 2 generators.
    Grid([4, 4], boundary=-1, particle_type="spinless_fermion", Nparticles=2)
    A = Translation(np.eye(2, dtype=int))  # via _get_perm
    B = Translation((1, 0)) @ Translation((0, 1))  # via __matmul__
    assert np.array_equal(np.asarray(A._perm), np.asarray(B._perm))
    assert np.allclose(np.asarray(A._character), np.asarray(B._character))
    assert np.array_equal(np.asarray(A._perm_sign), np.asarray(B._perm_sign))


@pytest.mark.parametrize("sector", [0, (2, 0)])
def test_translation_projector_apbc_fermion(sector):
    # 4x4 anti-periodic spinless fermions at even filling, where T^L = (-1)^N = 1,
    # so every momentum sector is a valid one. The projector must be idempotent
    # and Hermitian (the old multi-generator sign bug broke this), and commute with
    # the anti-periodic hopping, i.e. the signed translations are its symmetries.
    lattice = Grid([4, 4], boundary=-1, particle_type="spinless_fermion", Nparticles=2)
    configs = _configs(16, 2)
    P = _dense_projector(TransND(sector), configs)
    assert np.trace(P).real > 1  # non-empty sector, P = 0 would pass trivially
    assert np.max(np.abs(P @ P - P)) < 1e-5
    assert np.max(np.abs(P - P.conj().T)) < 1e-5
    H = _hopping(configs, *lattice.get_neighbor(1, return_sign=True))
    assert np.max(np.abs(P @ H - H @ P)) < 1e-5
