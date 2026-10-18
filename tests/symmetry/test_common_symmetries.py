import pytest
import numpy as np
import jax.numpy as jnp
from quantax.sites import Grid, Square, Triangular
import quantax.symmetry as sym


def _site_map(lattice, f):
    # perm[i] = index of the site at cell coordinates f(x_i, y_i), modulo the extent
    xyz = lattice.xyz_from_index.copy()
    xyz[:, 1:] = np.stack(f(*xyz[:, 1:].T), axis=1) % lattice.shape[1:]
    return lattice.index_from_xyz[tuple(xyz.T)]


# --- Identity ---


def test_identity():
    Square(4)
    I = sym.Identity()
    assert I.nsymm == 1
    assert np.array_equal(np.asarray(I.character), [1.0])
    assert np.array_equal(np.asarray(I._perm), [np.arange(16)])
    # Identity is cached and returns the same instance
    assert sym.Identity() is I


# --- Z2Inversion ---


@pytest.mark.parametrize("eigval", [1, -1])
def test_z2inversion(eigval):
    Square(4)
    z = sym.Z2Inversion(eigval)
    assert z.Z2_inversion == eigval
    # Z2 inversion doubles the group size relative to the trivial group
    assert z.nsymm == 2
    # cached per eigenvalue
    assert sym.Z2Inversion(eigval) is z


@pytest.mark.parametrize("bad", [0, 2])
def test_z2inversion_invalid_raises(bad):
    Square(4)
    with pytest.raises(ValueError):
        sym.Z2Inversion(bad)


# --- SpinInverse ---


@pytest.mark.parametrize("eigval", [1, -1])
def test_spininverse_spin(eigval):
    # In a spin-1/2 system SpinInverse falls back to Z2Inversion.
    Square(4)
    s = sym.SpinInverse(eigval)
    assert s is sym.Z2Inversion(eigval)


@pytest.mark.parametrize("eigval", [1, -1])
def test_spininverse_spinful_fermion(eigval):
    # In a spinful fermion system SpinInverse exchanges up/down sites.
    Grid([4, 2], particle_type="spinful_fermion", Nparticles=(2, 2))
    s = sym.SpinInverse(eigval)
    assert s.Z2_inversion == 0
    # the generator swaps the two spin blocks of length Nsites
    N = s.Nsites
    expected = np.concatenate([np.arange(N, 2 * N), np.arange(N)])
    assert np.array_equal(np.asarray(s._generator[0]), expected)
    # the eigenvalue is the character of the swap
    assert np.array_equal(np.asarray(s.character), [1, eigval])


def test_spininverse_eigval0_raises():
    Square(4)
    with pytest.raises(ValueError):
        sym.SpinInverse(0)


def test_spininverse_spinless_raises():
    Square(4, particle_type="spinless_fermion", Nparticles=2)
    with pytest.raises(ValueError):
        sym.SpinInverse(1)


# --- ParticleHole ---


def test_particlehole_fermion():
    # particle-hole is the Z2 inversion of a fermion system
    Square(4, particle_type="spinless_fermion", Nparticles=8)
    assert sym.ParticleHole(-1) is sym.Z2Inversion(-1)


def test_particlehole_spin_raises():
    Square(4)
    with pytest.raises(ValueError):
        sym.ParticleHole()


# --- Flip / Rotation / LinearTransform ---


def test_flip():
    lattice = Square(4)
    # the default center is the middle of the lattice: x -> 3 - x
    perm = np.asarray(sym.Flip()._perm)
    assert np.array_equal(perm[1], _site_map(lattice, lambda x, y: (3 - x, y)))
    # flipping y about the origin site instead: y -> -y
    perm = np.asarray(sym.Flip(axis=1, center=[0, 0])._perm)
    assert np.array_equal(perm[1], _site_map(lattice, lambda x, y: (x, -y)))


def test_rotation_square():
    lattice = Square(4)
    perm = np.asarray(sym.Rotation(np.pi / 2)._perm)
    assert perm.shape[0] == 4
    # perm[1][i] is the site that the 90-degree rotation about the lattice center
    # moves onto site i, and two rotations give the point inversion
    assert np.array_equal(perm[1], _site_map(lattice, lambda x, y: (y, 3 - x)))
    assert np.array_equal(perm[2], _site_map(lattice, lambda x, y: (3 - x, 3 - y)))


def test_lineartransform_spinful_acts_on_both_spins():
    lattice = Grid([4, 2], particle_type="spinful_fermion", Nparticles=(2, 2))
    g = _site_map(lattice, lambda x, y: (3 - x, y))
    perm = np.asarray(sym.Flip()._perm)
    assert np.array_equal(perm[1], np.concatenate([g, g + 8]))


def test_rotation_axis_out_of_bound_raises():
    Square(4)
    with pytest.raises(ValueError):
        sym.Rotation(np.pi / 2, axes=(0, 2))


def test_lineartransform_not_symmetry_raises():
    Square(4)
    with pytest.raises(ValueError):
        # scaling does not map the lattice onto itself
        sym.LinearTransform(np.array([[2.0, 0.0], [0.0, 1.0]]))


# --- C4v ---


@pytest.mark.parametrize(
    "repr_, character",
    [
        ("A1", [1, 1, 1, 1, 1, 1, 1, 1]),
        ("A2", [1, -1, 1, -1, 1, -1, 1, -1]),
        ("B1", [1, 1, -1, -1, 1, 1, -1, -1]),
        ("B2", [1, -1, -1, 1, 1, -1, -1, 1]),
    ],
)
def test_c4v_1d_reps(repr_, character):
    # elements ordered as R^k F^f with index 2k + f
    Square(4)
    s = sym.C4v(repr=repr_)
    assert len({tuple(p) for p in np.asarray(s._perm).tolist()}) == s.nsymm == 8
    assert np.allclose(np.asarray(s.character), character)


def test_c4v_e_rep():
    Square(4)
    s = sym.C4v(repr="E")
    # the 2D E irrep is realized on the C2 subgroup with doubled character
    assert s.nsymm == 2
    assert np.allclose(np.asarray(s.character), [2, -2])
    # symmetrize includes the irrep dimension d = 2: (d / |G|) sum_g chi_g psi_g
    psi = jnp.array([1.0, 0.25])
    assert np.isclose(s.symmetrize(psi), (2 / 2) * (2 * 1.0 - 2 * 0.25))


def test_c4v_invalid_repr_raises():
    Square(4)
    with pytest.raises(ValueError):
        sym.C4v(repr="X")


# --- D6 ---
# On an even triangular torus the mean coordinate is no C6 center, so pass a site.


@pytest.mark.parametrize(
    "repr_, character",
    [
        ("A1", [1] * 12),
        ("A2", [1, -1] * 6),
        ("B1", [1, 1, -1, -1] * 3),
        ("B2", [1, -1, -1, 1] * 3),
    ],
)
def test_d6_1d_reps(repr_, character):
    Triangular(4)
    s = sym.D6(center=[0, 0], repr=repr_)
    assert len({tuple(p) for p in np.asarray(s._perm).tolist()}) == s.nsymm == 12
    assert np.allclose(np.asarray(s.character), character)


@pytest.mark.parametrize(
    "repr_, character",
    [
        ("E1", [2, 1, -1, -2, -1, 1]),
        ("E2", [2, -1, -1, 2, -1, -1]),
    ],
)
def test_d6_e_reps(repr_, character):
    Triangular(4)
    s = sym.D6(center=[0, 0], repr=repr_)
    assert s.nsymm == 6
    assert np.allclose(np.asarray(s.character), character)


def test_d6_invalid_repr_raises():
    Triangular(4)
    with pytest.raises(ValueError):
        sym.D6(repr="X")


def test_d6_e_rep_uses_center():
    # Regression test: D6's E representations must thread `center` into Rotation.
    lattice = Triangular(4)
    origin = np.asarray(sym.D6(center=lattice.coord[0], repr="E1")._perm)
    center = lattice.coord[5]
    shifted = np.asarray(sym.D6(center=center, repr="E1")._perm)
    # a different center must change the generated permutations
    assert not np.array_equal(origin, shifted)
    # and must match a direct Rotation built with the same center and character
    character = jnp.array([2, 1, -1, -2, -1, 1], dtype=jnp.float32)
    direct = np.asarray(
        sym.Rotation(np.pi / 3, center=center, character=character)._perm
    )
    assert np.array_equal(shifted, direct)
