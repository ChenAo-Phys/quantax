import pytest
import numpy as np
from quantax.sites import (
    Sites,
    Lattice,
    Chain,
    Square,
    Cube,
    Triangular,
    TriangularB,
    Pyrochlore,
)


def test_define():
    lattice = Lattice(
        extent=(4, 4),
        basis_vectors=np.array([[1, 0], [1, 1]]),
        site_offsets=np.array([[0.0, 0.0], [0.5, 0.5]]),
    )
    assert lattice.shape == (2, 4, 4)
    assert lattice.Nsites == 32
    assert lattice.ncells == 16


def test_shape_and_cells():
    lattice = Square(3)
    assert lattice.shape == (1, 3, 3)
    assert lattice.ncells == 9
    assert lattice.Nsites == 9
    assert lattice.ndim == 2


def test_index_xyz_roundtrip():
    lattice = Square(3)
    idx = lattice.index_from_xyz
    xyz = lattice.xyz_from_index
    # xyz_from_index inverts index_from_xyz
    assert np.array_equal(idx[tuple(xyz.T)], np.arange(lattice.Nsites))


def test_reciprocal_vectors():
    lattice = Square(4)
    # orthonormal basis -> reciprocal vectors are 2*pi times identity
    assert np.allclose(lattice.reciprocal_vectors, 2 * np.pi * np.eye(2))


def test_basis_and_offsets_default():
    lattice = Chain(4)
    assert np.array_equal(lattice.basis_vectors, np.eye(1))
    assert lattice.site_offsets.shape == (1, 1)
    assert np.array_equal(lattice.site_offsets, np.zeros((1, 1)))


def test_boundary_default_pbc():
    lattice = Square(4)
    assert np.array_equal(lattice.boundary, np.ones(2, dtype=np.int64))


# --- neighbors, distance and sign ---


def test_chain_pbc_vs_obc_neighbors():
    pbc = Chain(4)  # periodic: wraps around
    assert pbc.get_neighbor(1).shape[0] == 4
    Sites._SITES = None
    obc = Chain(4, boundary=0)  # open: no wrap
    assert obc.get_neighbor(1).shape[0] == 3


def test_square_nearest_neighbors():
    lattice = Square(3)
    nn = lattice.get_neighbor(1)
    # 2 bonds per site, counted once each (i < j) -> 2 * Nsites
    assert nn.shape == (2 * lattice.Nsites, 2)
    assert np.all(nn[:, 0] < nn[:, 1])


def test_get_neighbor_sequence():
    lattice = Square(4)
    neighbors = lattice.get_neighbor([1, 2])
    assert isinstance(neighbors, list)
    assert len(neighbors) == 2
    # nearest and next-nearest each have 2 * Nsites bonds on the square lattice
    assert neighbors[0].shape == (2 * lattice.Nsites, 2)
    assert neighbors[1].shape == (2 * lattice.Nsites, 2)


def test_dist_matrix_properties():
    lattice = Chain(4)
    dist = lattice.dist
    assert dist.shape == (4, 4)
    assert np.allclose(dist, dist.T)  # symmetric
    assert np.allclose(np.diag(dist), 0.0)  # zero on diagonal


def test_sign_pbc_all_positive():
    lattice = Square(3)
    assert np.all(lattice.sign == 1)


def test_sign_apbc_fermion():
    # anti-periodic boundary: bonds crossing the boundary get sign -1
    lattice = Chain(4, boundary=-1, particle_type="spinless_fermion", Nparticles=2)
    nn, sign = lattice.get_neighbor(1, return_sign=True)
    crosses = np.array([abs(j - i) != 1 for i, j in nn])
    assert np.all(sign[crosses] == -1)
    assert np.all(sign[~crosses] == 1)


# --- orbitals ---


def test_orbitals_orthonormal():
    lattice = Square(3)
    orbs = lattice.orbitals()
    assert orbs.shape == (lattice.Nsites, lattice.Nsites)
    # k-orbitals form an orthonormal basis
    assert np.allclose(orbs.conj().T @ orbs, np.eye(lattice.Nsites))


@pytest.mark.parametrize("L", [3, 4, 5])
def test_orbitals_real(L):
    lattice = Square(L)
    orbs = lattice.orbitals(use_real=True)
    assert np.iscomplexobj(orbs) is False or np.allclose(orbs.imag, 0.0)
    # a complete real basis: Nsites orthonormal orbitals, for even and odd L alike
    assert orbs.shape == (lattice.Nsites, lattice.Nsites)
    assert np.allclose(orbs.T @ orbs, np.eye(lattice.Nsites))


@pytest.mark.parametrize("L", [3, 4])
def test_orbitals_real_spans_complex(L):
    lattice = Square(L)
    real = lattice.orbitals(use_real=True)
    cplx = lattice.orbitals(use_real=False)
    # the real orbitals span the same single-particle subspace as the complex ones
    assert np.allclose(real @ real.T, (cplx @ cplx.conj().T).real)


# --- representations ---


def test_neighbor_repr_identity_generic():
    lattice = Square(3)
    x = np.arange(lattice.Nsites)
    assert np.array_equal(lattice.to_neighbor_repr(x), x)
    assert np.array_equal(lattice.to_original_repr(x), x)


def test_triangularB_repr_roundtrip():
    lattice = TriangularB(2)
    x = np.arange(lattice.Nsites)
    roundtrip = lattice.to_original_repr(lattice.to_neighbor_repr(x))
    assert np.array_equal(roundtrip, x)


# --- common lattices ---


def test_cube():
    lattice = Cube(2)
    assert lattice.Nsites == 8
    assert lattice.shape == (1, 2, 2, 2)
    assert lattice.ndim == 3


def test_triangular():
    lattice = Triangular(3)
    assert lattice.shape == (1, 3, 3)
    assert lattice.Nsites == 9
    # triangular lattice has 6 nearest neighbors per site -> 3 * Nsites bonds
    assert lattice.get_neighbor(1).shape[0] == 3 * lattice.Nsites


def test_triangularB():
    L = 2
    lattice = TriangularB(L)
    assert lattice.Nsites == 3 * L**2
    assert lattice.shape == (1, 3 * L, L)


def test_pyrochlore():
    lattice = Pyrochlore(1)
    assert lattice.shape == (4, 1, 1, 1)
    assert lattice.Nsites == 4
    assert lattice.ndim == 3


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
