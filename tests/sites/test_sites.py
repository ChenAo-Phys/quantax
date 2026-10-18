import pytest
import numpy as np
import quantax as qtx
from quantax.sites import Sites, Chain

# --- particle types ---


@pytest.mark.parametrize(
    "particle_type, Nparticles, double_occ, expected",
    [
        # expected: Nmodes, Nfmodes, Nparticles, Ntotal, double_occ
        # spins always hold Nsites particles; a spin maps to 2 fermion modes
        (qtx.PARTICLE_TYPE.spin, None, None, (4, 8, 4, 4, False)),
        ("spin", (3, 1), None, (4, 8, (3, 1), 4, False)),
        # spinful fermions store up and down modes and allow double occupancy
        ("Spinful Fermion", [2, 1], None, (8, 8, (2, 1), 3, True)),
        ("spinful_fermion", 3, False, (8, 8, 3, 3, False)),
        ("spinless_fermion", 2, None, (4, 4, 2, 2, False)),
        # no particle number conservation
        ("spinless_fermion", None, None, (4, 4, None, None, False)),
    ],
)
def test_particle_types(particle_type, Nparticles, double_occ, expected):
    chain = Chain(
        4, particle_type=particle_type, Nparticles=Nparticles, double_occ=double_occ
    )
    assert chain.Nsites == 4
    got = (
        chain.Nmodes,
        chain.Nfmodes,
        chain.Nparticles,
        chain.Ntotal,
        chain.double_occ,
    )
    assert got == expected
    name = chain.particle_type.name
    assert chain.is_fermion == (name != "spin")
    assert chain.is_spinful == (name != "spinless_fermion")


# --- neighbors of sites without a lattice ---


def test_sites_neighbors_without_lattice():
    # plain Euclidean distances, no boundary and no signs
    coord = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [3, 0]])
    sites = Sites(5, coord=coord)
    assert sites.ndim == 2
    assert np.allclose(
        sites.dist, np.linalg.norm(coord[:, None] - coord[None], axis=-1)
    )
    shells, signs = sites.get_neighbor([1, 2, 3], return_sign=True)
    assert np.array_equal(shells[0], [[0, 1], [0, 2], [1, 3], [2, 3]])
    assert np.array_equal(shells[1], [[0, 3], [1, 2]])  # sqrt(2)
    assert np.array_equal(shells[2], [[1, 4]])  # distance 2
    assert all(np.all(s == 1) for s in signs)


# --- error handling ---


@pytest.mark.parametrize(
    "kwargs",
    [
        # a spin magnetization sector needs a tuple (Nup, Ndn) summing to Nsites
        dict(particle_type="spin", Nparticles=2),
        dict(particle_type="spin", Nparticles=(1, 1)),
        dict(particle_type="spinless_fermion", Nparticles=(1, 1)),
        dict(particle_type="spinful_fermion", Nparticles=(1, 1, 1)),
        dict(particle_type="spin", double_occ=True),
        dict(particle_type="nonsense"),
    ],
)
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        Sites(4, **kwargs)


def test_coord_unavailable_raises():
    sites = Sites(3)
    with pytest.raises(RuntimeError):
        sites.coord
    with pytest.raises(RuntimeError):
        sites.ndim


def test_multiple_sites_warns():
    Chain(4)
    with pytest.warns(UserWarning):
        Chain(4)
