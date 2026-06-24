import pytest
import numpy as np
import quantax as qtx
from quantax.sites import Sites, Chain


def test_get_sites():
    chain = Chain(4)
    assert qtx.get_sites() is chain
    assert qtx.get_lattice() is chain


def test_define():
    coord = np.array([[0, 0], [1, 0], [0, 1]])
    sites = Sites(3, coord=coord)
    assert sites.Nsites == 3
    assert sites.ndim == 2
    assert np.array_equal(sites.coord, coord)


# --- particle types ---


def test_spin():
    L = 4
    chain = Chain(L, particle_type="spin")
    assert chain.ndim == 1
    assert chain.Nsites == L
    assert chain.Nmodes == L
    assert chain.Nfmodes == 2 * L
    assert chain.Nparticles == L
    assert chain.Ntotal == L
    assert chain.particle_type == qtx.PARTICLE_TYPE.spin


def test_spinful_fermion():
    L = 4
    chain = Chain(L, particle_type="spinful_fermion", Nparticles=(2, 2))
    assert chain.ndim == 1
    assert chain.Nsites == L
    assert chain.Nmodes == 2 * L
    assert chain.Nfmodes == 2 * L
    assert chain.Nparticles == (2, 2)
    assert chain.Ntotal == 4
    assert chain.particle_type == qtx.PARTICLE_TYPE.spinful_fermion


def test_spinless_fermion():
    L = 4
    chain = Chain(L, particle_type="spinless_fermion", Nparticles=2)
    assert chain.ndim == 1
    assert chain.Nsites == L
    assert chain.Nmodes == L
    assert chain.Nfmodes == L
    assert chain.Nparticles == 2
    assert chain.Ntotal == 2
    assert chain.particle_type == qtx.PARTICLE_TYPE.spinless_fermion


# --- error handling ---


def test_spin_int_nparticles_raises():
    with pytest.raises(ValueError):
        Sites(4, particle_type="spin", Nparticles=2)


def test_spinless_tuple_nparticles_raises():
    with pytest.raises(ValueError):
        Sites(4, particle_type="spinless_fermion", Nparticles=(1, 1))


def test_double_occ_non_spinful_raises():
    with pytest.raises(ValueError):
        Sites(4, particle_type="spin", double_occ=True)


def test_unknown_particle_type_raises():
    with pytest.raises(ValueError):
        Sites(4, particle_type="nonsense")


def test_bad_nparticles_tuple_length_raises():
    with pytest.raises(ValueError):
        Sites(4, particle_type="spinful_fermion", Nparticles=(1, 1, 1))


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
