import pytest
import numpy as np
import jax.numpy as jnp
from quantax.sites import Sites, Square, Chain, Triangular
import quantax.symmetry as sym

# --- Identity ---


def test_identity():
    Square(4)
    I = sym.Identity()
    assert I.nsymm == 1
    assert np.array_equal(np.asarray(I.character), [1.0])
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


@pytest.mark.parametrize("bad", [0, 2, -2])
def test_z2inversion_invalid_raises(bad):
    Square(4)
    with pytest.raises(ValueError):
        sym.Z2Inversion(bad)


# --- SpinInverse ---


@pytest.mark.parametrize("eigval", [1, -1])
def test_spininverse_spin(eigval):
    # In a spin-1/2 system SpinInverse falls back to Z2Inversion.
    Chain(4)
    s = sym.SpinInverse(eigval)
    assert s is sym.Z2Inversion(eigval)
    assert s.Z2_inversion == eigval


@pytest.mark.parametrize("eigval", [1, -1])
def test_spininverse_spinful_fermion(eigval):
    # In a spinful fermion system SpinInverse exchanges up/down sites.
    Square(2, particle_type="spinful_fermion", Nparticles=(2, 2))
    s = sym.SpinInverse(eigval)
    assert s.nsymm == 2
    # the generator swaps the two spin blocks of length Nsites
    N = s.Nsites
    expected = np.concatenate([np.arange(N, 2 * N), np.arange(N)])
    assert np.array_equal(np.asarray(s._generator[0]), expected)


def test_spininverse_eigval0_raises():
    Chain(4)
    with pytest.raises(ValueError):
        sym.SpinInverse(0)


def test_spininverse_eigval0_raises_fermion():
    Square(2, particle_type="spinful_fermion", Nparticles=(2, 2))
    with pytest.raises(ValueError):
        sym.SpinInverse(0)


def test_spininverse_spinless_raises():
    Chain(4, particle_type="spinless_fermion", Nparticles=2)
    with pytest.raises(ValueError):
        sym.SpinInverse(1)


# --- ParticleHole ---


def test_particlehole_fermion():
    Chain(4, particle_type="spinless_fermion", Nparticles=2)
    ph = sym.ParticleHole(1)
    assert ph.Z2_inversion == 1
    assert ph.nsymm == 2


def test_particlehole_spin_raises():
    Square(4)
    with pytest.raises(ValueError):
        sym.ParticleHole()


# --- Flip / Rotation / LinearTransform ---


def test_flip():
    Square(4)
    f = sym.Flip()
    assert f.nsymm == 2
    # flipping twice is the identity -> the non-trivial element is an involution
    perm = np.asarray(f._perm)
    assert np.array_equal(perm[1][perm[1]], np.arange(perm.shape[1]))


def test_rotation_square():
    Square(4)
    r = sym.Rotation(np.pi / 2)
    # 90-degree rotation generates a cyclic group of order 4
    assert r.nsymm == 4


def test_rotation_axis_out_of_bound_raises():
    Square(4)
    with pytest.raises(ValueError):
        sym.Rotation(np.pi / 2, axes=(0, 2))


def test_lineartransform_identity():
    Square(4)
    s = sym.LinearTransform(np.eye(2))
    assert s.nsymm == 1


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
    Square(4)
    s = sym.C4v(repr=repr_)
    assert s.nsymm == 8
    assert np.allclose(np.asarray(s.character), character)


def test_c4v_e_rep():
    Square(4)
    s = sym.C4v(repr="E")
    # the 2D E irrep is realized on the C2 subgroup with doubled character
    assert s.nsymm == 2
    assert np.allclose(np.asarray(s.character), [2, -2])


def test_c4v_invalid_repr_raises():
    Square(4)
    with pytest.raises(ValueError):
        sym.C4v(repr="X")


# --- D6 ---


@pytest.mark.parametrize("repr_", ["A1", "A2", "B1", "B2"])
def test_d6_1d_reps(repr_):
    Triangular(3)
    s = sym.D6(repr=repr_)
    assert s.nsymm == 12


@pytest.mark.parametrize(
    "repr_, character",
    [
        ("E1", [2, 1, -1, -2, -1, 1]),
        ("E2", [2, -1, -1, 2, -1, -1]),
    ],
)
def test_d6_e_reps(repr_, character):
    Triangular(3)
    s = sym.D6(repr=repr_)
    assert s.nsymm == 6
    assert np.allclose(np.asarray(s.character), character)


def test_d6_invalid_repr_raises():
    Triangular(3)
    with pytest.raises(ValueError):
        sym.D6(repr="X")


def test_d6_e_rep_uses_center():
    # Regression test: D6's E representations must thread `center` into Rotation.
    lattice = Triangular(3)
    default = np.asarray(sym.D6(repr="E1")._perm)
    center = lattice.coord[1]
    shifted = np.asarray(sym.D6(repr="E1", center=center)._perm)
    # a non-default center must change the generated permutations
    assert not np.array_equal(default, shifted)
    # and must match a direct Rotation built with the same center and character
    character = jnp.array([2, 1, -1, -2, -1, 1], dtype=jnp.float32)
    direct = np.asarray(
        sym.Rotation(np.pi / 3, center=center, character=character)._perm
    )
    assert np.array_equal(shifted, direct)


# --- get_symm_spins ---


def test_get_symm_spins_shape():
    Square(4)
    s = sym.C4v(repr="A1")
    spins = jnp.ones(16)
    out = s.get_symm_spins(spins)
    assert out.shape == (s.nsymm, 16)
