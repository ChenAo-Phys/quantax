import itertools
import pytest
import numpy as np
import jax.numpy as jnp
from quantax.sites import (
    Lattice,
    Grid,
    Chain,
    Square,
    Cube,
    Triangular,
    TriangularB,
    Pyrochlore,
)
from quantax.symmetry import Translation


def _min_image(lattice):
    """Brute-force minimum-image distance and boundary sign of all site pairs.

    The sign is that of the sum of the boundary signs of all shortest images, 0 if
    they cancel.
    """
    r = lattice.coord
    periodic = np.flatnonzero(lattice.boundary != 0)
    supercell = np.asarray(lattice.shape[1:])[:, None] * lattice.basis_vectors
    dists, signs = [], []
    for n in itertools.product(range(-2, 3), repeat=len(periodic)):
        shift = np.zeros(lattice.ndim)
        sign = 1
        for axis, k in zip(periodic, n):
            shift += k * supercell[axis]
            sign *= lattice.boundary[axis] ** abs(k)
        dists.append(np.linalg.norm(r[None, :, :] + shift - r[:, None, :], axis=-1))
        signs.append(sign)
    dists = np.stack(dists)
    dist = dists.min(axis=0)
    total = np.einsum("nij,n->ij", np.isclose(dists, dist), np.asarray(signs))
    return dist, np.sign(total)


def _shell(dist, n):
    # i < j pairs at the n'th smallest non-zero distance, empty past the last one
    values = np.unique(np.round(dist[dist > 1e-9], 8))
    if n > len(values):
        return np.zeros((0, 2), dtype=np.int64)
    pairs = np.argwhere(np.isclose(dist, values[n - 1]))
    return pairs[pairs[:, 0] < pairs[:, 1]]


# --- geometry ---


def test_lattice_geometry():
    a = np.array([[1.0, 0.0], [1.0, 1.0]])
    offsets = np.array([[0.0, 0.0], [0.5, 0.25]])
    lattice = Lattice(extent=(4, 3), basis_vectors=a, site_offsets=offsets)
    assert lattice.shape == (2, 4, 3)
    assert lattice.Nsites == 24 and lattice.ncells == 12 and lattice.ndim == 2
    # xyz_from_index[i] = [sublattice, x, y] inverts index_from_xyz
    xyz = lattice.xyz_from_index
    assert np.array_equal(lattice.index_from_xyz[tuple(xyz.T)], np.arange(24))
    # r = x a1 + y a2 + offset of the sublattice
    coord = xyz[:, 1:] @ a + offsets[xyz[:, 0]]
    assert np.allclose(lattice.coord, coord)
    # reciprocal vectors are dual to the basis: a_i . b_j = 2 pi delta_ij
    assert np.allclose(a @ lattice.reciprocal_vectors.T, 2 * np.pi * np.eye(2))


# --- distance, sign and neighbors against brute-force minimum images ---


@pytest.mark.parametrize(
    "lattice, z1",
    [
        (lambda: Chain(4, boundary=-1, particle_type="spinless_fermion"), 2),
        (lambda: Square(3), 4),
        # mixed anti-periodic / open boundaries, edge sites lose a neighbor
        (lambda: Grid([4, 3], [-1, 0], particle_type="spinless_fermion"), None),
        (lambda: Triangular(4, boundary=-1, particle_type="spinless_fermion"), 6),
        # next-nearest pairs have three shortest images with signs of both kinds
        (lambda: Triangular(3, boundary=-1, particle_type="spinless_fermion"), 6),
        (lambda: Cube(3), 6),
        (lambda: Pyrochlore(2), 6),
        # skewed basis with two sites per cell
        (lambda: Lattice([3, 3], [[1, 0], [1, 1]], [[0, 0], [0.5, 0.5]]), 4),
        # skewed supercells: the shortest image can need a wrap along an axis
        # without displacement, or along both axes
        (lambda: TriangularB(2), 6),
        (lambda: TriangularB(3, boundary=-1, particle_type="spinless_fermion"), 6),
        (lambda: Triangular([6, 2], [1, -1], particle_type="spinless_fermion"), None),
        (lambda: Triangular([8, 2], boundary=[0, 1]), None),
        (lambda: Pyrochlore([2, 2, 1]), None),
    ],
    ids=[
        "chain_apbc",
        "square",
        "grid_mixed_bc",
        "triangular",
        "triangular_3x3_apbc",
        "cube",
        "pyrochlore",
        "two_site_cell",
        "triangularB",
        "triangularB_apbc",
        "triangular_6x2_mixed_bc",
        "triangular_cylinder",
        "pyrochlore_2x2x1",
    ],
)
def test_dist_sign_neighbors_match_min_image(lattice, z1):
    lattice = lattice()
    dist, sign = _min_image(lattice)
    # sign is read before anything computes the lazy dist and sign
    assert np.array_equal(lattice.sign, sign)
    # the shell-1 cache is filled first, so the sequence below extends it
    nn = lattice.get_neighbor(1)
    shells = lattice.get_neighbor([1, 2, 3])
    assert np.allclose(lattice.dist, dist)
    assert np.array_equal(nn, shells[0])
    for n, bonds in enumerate(shells, start=1):
        assert np.array_equal(bonds, _shell(dist, n))
        bond_sign = sign[bonds[:, 0], bonds[:, 1]]
        if np.all(bond_sign != 0):
            _, (got,) = lattice.get_neighbor([n], return_sign=True)
            assert np.array_equal(got, bond_sign)
        else:  # shortest images of opposite signs cancel
            with pytest.raises(ValueError, match="no well-defined sign"):
                lattice.get_neighbor(n, return_sign=True)
    if z1 is not None:  # coordination number of the lattice
        assert np.all(np.bincount(nn.ravel(), minlength=lattice.Nsites) == z1)


def test_triangularB_neighbor_shells():
    # every shell of the hexagonal 27-site cluster, up to the farthest one
    lattice = TriangularB(3)
    dist, _ = _min_image(lattice)
    shells = lattice.get_neighbor(list(range(1, 7)))
    for n, bonds in enumerate(shells, start=1):
        assert np.array_equal(bonds, _shell(dist, n))
    # each site sees 6, 6, 6, 6 and 2 sites in the first five shells, which hold
    # all of the other 26 sites
    counts = [np.bincount(b.ravel(), minlength=lattice.Nsites) for b in shells[:5]]
    assert np.array_equal(np.stack(counts), np.repeat([[6], [6], [6], [6], [2]], 27, 1))
    assert shells[5].shape == (0, 2)


def test_triangularB_j2_bonds():
    # sites 0 and 6 of TriangularB(2) are 3 a1 = (3, 0) apart in the cell, but the
    # image 3 a1 - 2 a2 = (0, -sqrt(3)) needs a wrap along a2, an axis without
    # displacement; all 12 sites have 3 sites at sqrt(3), i.e. 18 J2 bonds
    lattice = TriangularB(2)
    assert np.isclose(lattice.dist[0, 6], np.sqrt(3))
    j2 = lattice.get_neighbor(2)
    assert np.allclose(lattice.dist[j2[:, 0], j2[:, 1]], np.sqrt(3))
    assert len(j2) == 18


def test_cancelling_images_have_no_bond_sign():
    # sites 0 and 2 of an anti-periodic 4-site chain are 2 apart both directly and
    # through the boundary, with opposite signs: the folded hopping cancels, so the
    # sign is 0 and the second shell can't be requested with signs
    lattice = Chain(4, boundary=-1, particle_type="spinless_fermion")
    assert np.array_equal(lattice.sign[0], [1, 1, 0, -1])
    # the nearest neighbors are well defined, one of them crosses the boundary
    _, sign = lattice.get_neighbor(1, return_sign=True)
    assert sorted(sign.tolist()) == [-1, 1, 1, 1]
    with pytest.raises(ValueError, match="neighbor shell n_neighbor=2 have no"):
        lattice.get_neighbor([1, 2], return_sign=True)
    assert np.array_equal(lattice.get_neighbor(2), [[0, 2], [1, 3]])


@pytest.mark.parametrize(
    "lattice, n_neighbor",
    [
        # the images of shells 3-5 cancel, sign 0
        (
            lambda: Square(4, boundary=-1, particle_type="spinless_fermion"),
            [1, 2, 3, 4, 5],
        ),
        # shell 2 has three images of signs (+, -, -), sign -1
        (lambda: Triangular(3, boundary=-1, particle_type="spinless_fermion"), [1, 2]),
        # shell 5 has cancelling and equal-sign pairs, shell 6 mixed signs
        (lambda: Triangular(6, boundary=-1, particle_type="spinless_fermion"), [5, 6]),
        (lambda: Triangular(8, [1, -1], particle_type="spinless_fermion"), [1, 2, 9]),
    ],
    ids=[
        "square_apbc",
        "triangular_3x3_apbc",
        "triangular_6x6_apbc",
        "triangular_8x8_mixed_bc",
    ],
)
def test_bond_signs_are_translation_covariant(lattice, n_neighbor):
    # the hopping sum_ij sign_ij (c_i^dag c_j + h.c.) of each shell commutes with
    # the translations c_i -> s_i c_g(i), also on pairs with tied shortest images
    lattice = lattice()
    N = lattice.Nsites
    # read from Lattice.sign, as get_neighbor raises for shells with sign 0
    shells = lattice.get_neighbor(n_neighbor)
    for vector in np.eye(lattice.ndim, dtype=np.int64):
        T = Translation(vector)
        U = np.zeros((N, N))
        U[np.asarray(T._generator[0]), np.arange(N)] = T._generator_sign[0]
        for bonds in shells:
            sign = lattice.sign[bonds[:, 0], bonds[:, 1]]
            h = np.zeros((N, N))
            h[bonds[:, 0], bonds[:, 1]] = h[bonds[:, 1], bonds[:, 0]] = sign
            assert np.allclose(U @ h @ U.T, h)


def test_get_neighbor_beyond_last_shell_is_empty():
    # an open chain of 4 sites only has distances 1, 2 and 3
    lattice = Chain(4, boundary=0)
    assert lattice.get_neighbor(4).shape == (0, 2)
    # extending the cache past an empty shell keeps returning empty arrays
    third, fifth = lattice.get_neighbor([3, 5])
    assert np.array_equal(third, [[0, 3]])
    assert fifth.shape == (0, 2)


# --- orbitals ---


def _assert_sorted_eigenbasis(orbs, H):
    # an orthonormal basis of eigenvectors of H, in ascending energy
    N = H.shape[0]
    assert orbs.shape == (N, N)
    assert np.allclose(orbs.conj().T @ orbs, np.eye(N))
    E = orbs.conj().T @ H @ orbs
    assert np.allclose(E, np.diag(np.diag(E)))
    assert np.all(np.diff(np.diag(E).real) > -1e-9)


@pytest.mark.parametrize("use_real", [False, True])
@pytest.mark.parametrize("extent", [[3, 3], [4, 4], [4, 3], [5]])
def test_orbitals_are_sorted_tight_binding_eigenstates(extent, use_real):
    # nearest-neighbor tight-binding eigenstates in ascending energy, so the first
    # N orbitals are the free-fermion ground state
    lattice = Grid(extent)
    N = lattice.Nsites
    orbs = lattice.orbitals(use_real=use_real)
    if use_real:
        assert not np.iscomplexobj(orbs)
    H = np.zeros((N, N))
    i, j = lattice.get_neighbor(1).T
    H[i, j] = H[j, i] = -1.0
    _assert_sorted_eigenbasis(orbs, H)


@pytest.mark.parametrize("use_real", [False, True])
def test_orbitals_block_diagonal_in_sublattice(use_real):
    # With two sites per cell, each orbital is a plane wave on one sublattice. The
    # sites are sublattice-major, so the orbitals are two equal diagonal blocks,
    # each the sorted tight-binding basis of the 4 x 3 periodic grid of cells.
    lattice = Lattice([4, 3], np.eye(2), [[0, 0], [0.5, 0.5]])
    n = lattice.ncells
    assert np.all(lattice.xyz_from_index[:n, 0] == 0)
    orbs = lattice.orbitals(use_real=use_real)
    assert orbs.shape == (2 * n, 2 * n)
    assert np.allclose(orbs[:n, n:], 0) and np.allclose(orbs[n:, :n], 0)
    assert np.allclose(orbs[n:, n:], orbs[:n, :n])
    cells = lattice.xyz_from_index[:n, 1:]
    d = (cells[:, None] - cells[None]) % lattice.shape[1:]
    d = np.minimum(d, lattice.shape[1:] - d)
    H = -(d.sum(-1) == 1).astype(float)  # hopping between neighboring cells
    _assert_sorted_eigenbasis(orbs[:n, :n], H)


# --- representations ---


def test_triangularB_neighbor_repr():
    L = 3
    lattice = TriangularB(L)
    N = lattice.Nsites
    site = lattice.to_neighbor_repr(np.arange(N)).reshape(3 * L, L)
    # array neighbors along (1, 0), (0, 1) and (-1, 1) are lattice neighbors, so
    # convolutions see a triangular stencil (x is periodic, y is not wrapped)
    nn = set(map(tuple, lattice.get_neighbor(1)))
    for x, y in itertools.product(range(3 * L), range(L - 1)):
        for dx, dy in [(1, 0), (0, 1), (-1, 1)]:
            a, b = site[x, y], site[(x + dx) % (3 * L), y + dy]
            assert (min(a, b), max(a, b)) in nn
    # to_original_repr undoes it, also for batched JAX input
    x = jnp.arange(2 * N).reshape(2, N)
    roundtrip = lattice.to_original_repr(lattice.to_neighbor_repr(x))
    assert isinstance(roundtrip, type(x))
    assert np.array_equal(roundtrip, x)


# --- plot ---


def test_plot_hides_boundary_bonds():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 3x3 cells with a second site at the cell center: each corner site bonds to
    # the 4 centers around it, 36 bonds in total
    lattice = Lattice([3, 3], np.eye(2), [[0, 0], [0.5, 0.5]])
    fig = lattice.plot()
    axes = fig.axes[0]
    # bonds through the periodic boundary are not drawn: a corner (x, y) only
    # reaches the centers of cells (x - 1, y - 1)..(x, y) directly if x, y >= 1,
    # so 5 x 5 of the 36 bonds remain
    assert len(axes.lines) == 25
    # one index label per site, one color per sublattice
    assert len(axes.texts) == lattice.Nsites
    colors = axes.collections[0].get_facecolors()
    assert np.all(colors[:9] == colors[0]) and np.all(colors[9:] == colors[9])
    assert not np.array_equal(colors[0], colors[9])
    plt.close(fig)


# --- error handling ---


def test_spin_apbc_raises():
    with pytest.raises(ValueError):
        Chain(4, boundary=-1)


def test_triangular_wrong_extent_raises():
    with pytest.raises(ValueError):
        Triangular([2, 2, 2])


def test_pyrochlore_wrong_extent_raises():
    with pytest.raises(ValueError):
        Pyrochlore([2, 2])
