import pytest
import quantax as qtx
from quantax.sites import Chain


def test_get_sites():
    chain = Chain(4)
    sites = qtx.get_sites()
    assert sites is chain

    lattice = qtx.get_lattice()
    assert lattice is chain


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
