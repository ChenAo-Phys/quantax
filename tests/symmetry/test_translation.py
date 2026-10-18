import pytest
import numpy as np
import jax.numpy as jnp
from quantax.sites import Grid, Lattice, Square
from quantax.symmetry import Translation, TransND
from _dtype import use_dtype


def _translated_sites(lattice, vec):
    # index of the site at r_i + vec, which is what a translation generator stores
    xyz = lattice.xyz_from_index.copy()
    xyz[:, 1:] = (xyz[:, 1:] + np.asarray(vec)) % lattice.shape[1:]
    return lattice.index_from_xyz[tuple(xyz.T)]


# --- group elements and characters ---


def test_transnd_elements_and_characters():
    # Every element of the 4x4 translation group is a lattice translation by some
    # (a, b), with character exp[-2 pi i (a q1 + b q2) / L] in the sector (q1, q2).
    use_dtype(jnp.complex64)
    lattice = Square(4)
    q = np.array([1, 2])
    t = TransND(tuple(q))
    vecs = []
    for perm, chi in zip(np.asarray(t._perm), np.asarray(t.character)):
        vec = lattice.xyz_from_index[perm[0], 1:]  # where site (0, 0) is sent
        assert np.array_equal(perm, _translated_sites(lattice, vec))
        assert np.isclose(chi, np.exp(-2j * np.pi * (vec @ q) / 4), atol=1e-6)
        vecs.append(tuple(vec))
    assert len(set(vecs)) == t.nsymm == 16
    assert np.all(np.asarray(t._perm_sign) == 1)


def test_translation_incommensurate_raises():
    Square(4)
    with pytest.raises(ValueError):
        Translation((3, 0))  # 3 does not divide 4


def test_translation_open_boundary_raises():
    Square(4, boundary=0)
    with pytest.raises(ValueError):
        Translation((1, 0))


# --- negative translation vectors (regression) ---


def test_negative_translation_is_inverse():
    # A negative translation vector must not crash and should generate the same
    # cyclic group as its positive counterpart (it is the group inverse).
    Square(4)
    tp = Translation((1, 0))
    tn = Translation((-1, 0))
    assert tp.nsymm == tn.nsymm

    gp = np.asarray(tp._generator[0])
    gn = np.asarray(tn._generator[0])
    # the two generators are inverse permutations of each other
    assert np.array_equal(gp[gn], np.arange(gp.size))
    # and they span the same set of group permutations
    sp = set(map(tuple, np.asarray(tp._perm).tolist()))
    sn = set(map(tuple, np.asarray(tn._perm).tolist()))
    assert sp == sn


@pytest.mark.parametrize(
    "extent, particle_type",
    [([4, 4], "spinless_fermion"), ([4, 2], "spinful_fermion")],
    ids=["spinless", "spinful"],
)
def test_negative_translation_apbc_sign(extent, particle_type):
    # Anti-periodic x boundary: the sites that wrap across it pick up a -1,
    # regardless of the translation direction; the periodic y boundary gives none.
    # Spinful fermions repeat the site map and the signs on the spin-down block.
    lattice = Grid(extent, boundary=[-1, 1], particle_type=particle_type)
    nblocks = lattice.Nmodes // lattice.Nsites
    x = np.tile(lattice.xyz_from_index[:, 1], nblocks)
    block = np.repeat(np.arange(nblocks), lattice.Nsites) * lattice.Nsites
    t = Translation((1, 0))
    expected = np.tile(_translated_sites(lattice, (1, 0)), nblocks) + block
    assert np.array_equal(np.asarray(t._generator[0]), expected)
    # +1 moves the x = 3 column across the boundary, -1 moves the x = 0 column
    assert np.array_equal(np.asarray(t._generator_sign[0]), np.where(x == 3, -1, 1))
    sign = np.asarray(Translation((-1, 0))._generator_sign[0])
    assert np.array_equal(sign, np.where(x == 0, -1, 1))
    assert np.all(np.asarray(Translation((0, -1))._generator_sign[0]) == 1)


def test_negative_translation_momentum_mirror():
    # T(1) in sector q projects onto the same momentum subspace as
    # T(-1) in sector (L - q), since T(-1) = T(1)^{-1}.
    use_dtype(jnp.complex64)
    Square(4)
    N = 16

    def projector(symm):
        P = np.zeros((N, N), complex)
        for g, c in zip(np.asarray(symm._perm), np.asarray(symm._character)):
            Pg = np.zeros((N, N), complex)
            Pg[np.arange(N), g] = 1
            P += np.conj(c) * Pg
        return P / symm.nsymm

    for q in range(4):
        Pp = projector(Translation((1, 0), sector=q))
        Pn = projector(Translation((-1, 0), sector=(4 - q) % 4))
        assert np.allclose(Pp, Pn, atol=1e-6)


# --- sublattice coordinates ---


def _order(g):
    # order of the permutation g, i.e. the period of its translation
    k, p = 1, g
    while not np.array_equal(p, np.arange(g.size)):
        k, p = k + 1, p[g]
    return k


@pytest.mark.parametrize(
    "lattice, vectors",
    [
        (lambda: Square(4), [[2, 0], [0, 2]]),
        # negative vectors once gave a negative period (regression)
        (lambda: Square(4), [[-2, 0], [0, 2]]),
        # a vector off the lattice axes, with periods 2 and 4
        (lambda: Square(4), [[2, 0], [1, 1]]),
        # two sites per unit cell enlarge the set of sublattice channels
        (lambda: Lattice([4, 2], np.eye(2), [[0, 0], [0.5, 0.5]]), [[2, 0], [0, 1]]),
    ],
    ids=["square", "negative_vector", "skewed", "two_site_cell"],
)
def test_sublattice_coord(lattice, vectors):
    # [channel, x', y'] labels each site uniquely, and translating a site by a
    # sublattice vector keeps its channel and moves it by one sublattice step.
    # This is what lets Jastrow / pairing parameters be shared between sites.
    lattice = lattice()
    t = Translation(vectors)
    coord = t.get_sublattice_coord()
    # The coordinate along a sublattice vector has the period of its translation.
    # A coordinate system [channel, x', y'] needs the group to be the direct
    # product of its generators' cyclic groups, i.e. |G| = product of the periods.
    period = np.array([_order(g) for g in np.asarray(t._generator)])
    assert len({tuple(p) for p in np.asarray(t._perm).tolist()}) == np.prod(period)
    nchannel = lattice.Nsites // np.prod(period)
    assert len({tuple(c) for c in coord}) == lattice.Nsites
    assert np.all((coord[:, 1:] >= 0) & (coord[:, 1:] < period))
    assert set(coord[:, 0]) == set(range(nchannel))
    for k, g in enumerate(np.asarray(t._generator)):
        step = np.zeros(lattice.ndim, dtype=int)
        step[k] = 1
        shifted = coord.copy()
        shifted[:, 1:] = (shifted[:, 1:] + step) % period
        assert np.array_equal(coord[g], shifted)


def test_sublattice_coord_needs_one_vector_per_axis():
    Square(4)
    with pytest.raises(ValueError):
        Translation((2, 0)).get_sublattice_coord()
