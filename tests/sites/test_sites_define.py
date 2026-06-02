import pytest
import numpy as np
import quantax as qtx
from quantax.sites import Sites, Lattice


def test_Sites_define():
    coord = np.array([[0, 0], [1, 0], [0, 1]])
    sites = Sites(3, coord=coord)
    assert sites.Nsites == 3
    assert np.array_equal(sites.coord, coord)


def test_Lattice_define():
    extent = (4, 4)
    basis_vectors = np.array([[1, 0], [1, 1]])
    site_offsets = np.array([[0, 0], [0.5, 0.5]])
    lattice = Lattice(
        extent, basis_vectors, site_offsets
    )
    assert lattice.Nsites == 32
    assert lattice.shape == (2, 4, 4)
