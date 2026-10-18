import pytest
import numpy as np
import jax
import jax.numpy as jnp
from quantax.sites import Chain, Square, Pyrochlore
from quantax.utils import (
    ints_to_array,
    array_to_ints,
    neel,
    stripe,
    Sqz_factor,
    rand_states,
    get_distributed_sharding,
)

NDEV = jax.device_count()


def test_ints_array_match_quspin_basis():
    # The conversion must follow the basis integers of QuSpin, which the exact
    # references (ED, todense) are built on: the first mode is the most significant
    # bit, and a set bit is spin up / occupied (+1).
    from quspin.basis import spin_basis_general

    basis = spin_basis_general(4, pauli=-1)
    labels = [basis.int_to_state(int(i)) for i in basis.states]  # e.g. "|1 1 0 1>"
    bits = [[int(c) for c in label[1:-1].split()] for label in labels]
    expected = np.array(bits) * 2 - 1

    Chain(4)  # Nmodes defaults to the global sites
    arr = ints_to_array(basis.states)
    assert arr.dtype == np.int8
    np.testing.assert_array_equal(arr, expected)
    np.testing.assert_array_equal(ints_to_array(basis.states, Nmodes=4), expected)
    np.testing.assert_array_equal(array_to_ints(jnp.asarray(arr)), basis.states)


def test_neel():
    Square(4)
    spins = np.asarray(neel()).reshape(4, 4)
    # checkerboard with (0, 0) up: nearest neighbours are anti-aligned
    assert spins[0, 0] == 1
    assert np.all(spins[1:] == -spins[:-1]) and np.all(spins[:, 1:] == -spins[:, :-1])
    np.testing.assert_array_equal(neel(bipartiteA=False), -spins.flatten())


def test_neel_multisite_raises():
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
    lattice = Square(4)
    s = np.random.default_rng(0).choice([-1.0, 1.0], size=16).astype(np.float32)
    # brute force: sum_r e^{-iqr} s_r / (2 sqrt(N)); (pi, pi) gives real phases,
    # (pi/2, 0) complex ones
    for q in [(np.pi, np.pi), (np.pi / 2, 0.0)]:
        f = Sqz_factor(*q)
        expected = np.sum(np.exp(-1j * lattice.coord @ np.array(q)) * s) / 8
        np.testing.assert_allclose(complex(f(jnp.asarray(s))), expected, atol=1e-5)

    f = Sqz_factor(np.pi, np.pi)
    # uniform config: all signs cancel -> 0
    assert np.isclose(np.asarray(f(jnp.ones(16))), 0.0, atol=1e-6)
    # Neel order is exactly the (pi,pi) modulation -> peaks at sqrt(N)/2
    assert np.isclose(np.asarray(f(neel().astype(jnp.float32))), np.sqrt(16) / 2)


# ---------- rand_states ----------
# The jitted generators are specialized on the shape and the particle number, so the
# cases below share them where possible: NS samples everywhere, 12 modes for random
# states and blocks of 6 modes with 4 or 2 particles for conserved ones.
NS = 16 * NDEV
SPINFUL = dict(particle_type="spinful_fermion")


def test_rand_states_spin():
    Chain(12)
    s = rand_states(NS)
    assert s.shape == (NS, 12) and s.dtype == jnp.int8
    assert set(np.unique(np.asarray(s)).tolist()) == {-1, 1}
    # a batch divisible by the device count is distributed over the devices
    assert s.sharding.is_equivalent_to(get_distributed_sharding(), 2)
    # a batch not divisible by the device count (here 1, unless there is only one
    # device) can't be split evenly and is replicated instead; without a batch, a
    # single state without batch dimension. Both share one compiled (1, 12) kernel.
    assert rand_states(1).sharding.is_fully_replicated
    assert rand_states().shape == (12,)


@pytest.mark.parametrize(
    "L, kwargs, counts",
    [
        (6, dict(Nparticles=(4, 2)), (4,)),  # spin: number of up spins
        (6, dict(particle_type="spinless_fermion", Nparticles=2), (2,)),
        # spinful: total number over all modes, or (Nup, Ndown) over the two halves
        (3, dict(**SPINFUL, Nparticles=4), (4,)),
        (6, dict(**SPINFUL, Nparticles=(4, 2)), (4, 2)),
        # t-J like: no doubly occupied site
        (6, dict(**SPINFUL, Nparticles=(4, 2), double_occ=False), (4, 2)),
    ],
    ids=["spin", "spinless", "spinful_total", "spinful", "spinful_no_double_occ"],
)
def test_rand_states_conserved(L, kwargs, counts):
    sites = Chain(L, **kwargs)
    s = np.asarray(rand_states(NS))
    assert s.shape == (NS, sites.Nmodes)
    assert set(np.unique(s).tolist()) == {-1, 1}
    occ = np.split(s == 1, len(counts), axis=1)
    for n, block in zip(counts, occ):
        assert np.all(np.sum(block, axis=1) == n)
    if not sites.double_occ and sites.Nmodes > sites.Nsites:
        assert not np.any(occ[0] & occ[1])
    # every configuration is permuted independently, not one shared permutation
    assert len(np.unique(s, axis=0)) > 1


@pytest.mark.parametrize(
    "L, kwargs",
    [
        (12, dict(particle_type="spinless_fermion")),
        (6, SPINFUL),
        (6, dict(**SPINFUL, double_occ=False)),
    ],
    ids=["spinless", "spinful", "spinful_no_double_occ"],
)
def test_rand_states_fermion_not_conserved(L, kwargs):
    sites = Chain(L, **kwargs)
    s = np.asarray(rand_states(NS))
    assert s.shape == (NS, 12) and set(np.unique(s).tolist()) == {-1, 1}
    # the particle number fluctuates
    assert len(np.unique(np.sum(s == 1, axis=1))) > 1
    if sites.Nmodes > sites.Nsites:
        up, down = s[:, :6] == 1, s[:, 6:] == 1
        # without double occupancy, each site is empty, up or down
        assert np.any(up & down) == sites.double_occ
        assert np.any(up & ~down) and np.any(~up & down) and np.any(~up & ~down)


def test_rand_states_single_occ_total_number_not_implemented():
    Chain(6, particle_type="spinful_fermion", Nparticles=4, double_occ=False)
    with pytest.raises(NotImplementedError):
        rand_states(NDEV)
