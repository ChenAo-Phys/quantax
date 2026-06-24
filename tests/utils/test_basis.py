import pytest
import numpy as np
import jax.numpy as jnp
import quantax as qtx
from quantax.sites import Chain, Square, Sites
from quantax.utils import (
    ints_to_array,
    array_to_ints,
    neel,
    stripe,
    Sqz_factor,
    rand_states,
)


def test_ints_array_roundtrip():
    Chain(4)
    s = rand_states(5)
    ints = array_to_ints(s)
    back = ints_to_array(ints)
    assert np.array_equal(np.asarray(s), back)


def test_ints_to_array_values():
    Chain(4)
    arr = ints_to_array(np.array([0]))
    # integer 0 -> all spins down (-1)
    assert np.all(arr == -1)
    assert arr.dtype == np.int8


def test_neel():
    Chain(4)
    spins = np.asarray(neel())
    assert spins.tolist() == [1, -1, 1, -1]
    Sites._SITES = None
    Chain(4)
    spins_b = np.asarray(neel(bipartiteA=False))
    assert spins_b.tolist() == [-1, 1, -1, 1]


def test_neel_multisite_raises():
    from quantax.sites import Pyrochlore

    Pyrochlore(1)  # 4 sites per unit cell
    with pytest.raises(ValueError):
        neel()


def test_stripe():
    Square(4)
    spins = np.asarray(stripe(alternate_dim=1)).reshape(4, 4)
    # alternates along the chosen dimension, constant along the other
    for row in spins:
        assert np.array_equal(np.abs(np.diff(row)), np.full(3, 2))
    for col in spins.T:
        assert len(np.unique(col)) == 1  # constant -> stripe, not a checkerboard


def test_Sqz_factor():
    Square(4)
    f = Sqz_factor(np.pi, np.pi)
    # structure factor at q=(pi,pi): the (pi,pi) Fourier component of S^z.
    # uniform config: all signs cancel -> 0
    assert np.isclose(np.asarray(f(jnp.ones(16))), 0.0, atol=1e-6)
    # Neel order is exactly the (pi,pi) modulation -> peaks at sqrt(N)/2
    assert np.isclose(np.asarray(f(neel().astype(jnp.float32))), np.sqrt(16) / 2)


def test_rand_states_spin():
    Chain(6)
    s = rand_states(10)
    assert s.shape == (10, 6)
    assert set(np.unique(np.asarray(s)).tolist()).issubset({-1, 1})


def test_rand_states_no_batch():
    Chain(6)
    s = rand_states()
    assert s.shape == (6,)


def test_rand_states_spin_conserved():
    Chain(6, Nparticles=(3, 3))
    s = rand_states(8)
    # each configuration has exactly 3 up-spins
    assert np.all(np.sum(np.asarray(s) == 1, axis=1) == 3)


def test_rand_states_spinful_fermion_conserved():
    Square(2, particle_type="spinful_fermion", Nparticles=(2, 1))
    s = np.asarray(rand_states(8))
    Nsites = qtx.get_sites().Nsites
    up, down = s[:, :Nsites], s[:, Nsites:]
    assert np.all(np.sum(up == 1, axis=1) == 2)
    assert np.all(np.sum(down == 1, axis=1) == 1)


def test_rand_states_spinless_fermion_conserved():
    Chain(6, particle_type="spinless_fermion", Nparticles=2)
    s = np.asarray(rand_states(8))
    assert np.all(np.sum(s == 1, axis=1) == 2)
