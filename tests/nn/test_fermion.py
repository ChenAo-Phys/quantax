"""
Unit tests for the Jordan-Wigner sign bookkeeping in quantax/nn/fermion.py.

These pure functions (occupied-index extraction, hop detection, and the three
fermion-ordering signs) are load-bearing for the determinant/Pfaffian models and
are exactly the error-prone sign logic that needs an independent reference.
"""

import numpy as np
import jax.numpy as jnp
import pytest
from quantax.sites import Square, Chain
from quantax.nn import (
    fermion_idx,
    changed_inds,
    permute_sign,
    fermion_inverse_sign,
    fermion_reorder_sign,
)

# ---------- fermion_idx ----------


def test_fermion_idx_spinless():
    Square(2, particle_type="spinless_fermion", Nparticles=2)
    s = jnp.array([1.0, -1.0, 1.0, -1.0])  # occupied at modes 0, 2
    np.testing.assert_array_equal(np.asarray(fermion_idx(s)), [0, 2])


def test_fermion_idx_spin_maps_to_two_modes():
    # A spin config maps to 2N fermion modes: up occupation then down occupation
    # (down = "spin not up"), so every site contributes exactly one occupied mode.
    Square(2)  # spin: 4 sites -> 8 modes
    s = jnp.array([1.0, -1.0, 1.0, -1.0])
    # up modes occupied where s>0: {0,2}; down modes (offset 4) where s<=0: {1,3}->{5,7}
    np.testing.assert_array_equal(np.asarray(fermion_idx(s)), [0, 2, 5, 7])


def test_fermion_idx_separate_spins():
    Square(2, particle_type="spinful_fermion", Nparticles=(2, 1))
    # up half (modes 0-3), down half (modes 4-7)
    s = jnp.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0])
    idx_up, idx_dn = fermion_idx(s, separate_spins=True)
    np.testing.assert_array_equal(np.asarray(idx_up), [0, 2])
    np.testing.assert_array_equal(np.asarray(idx_dn), [0])  # within the down block


def test_fermion_idx_separate_spins_requires_spinful():
    Square(2, particle_type="spinless_fermion", Nparticles=2)
    with pytest.raises(ValueError):
        fermion_idx(jnp.array([1.0, -1.0, 1.0, -1.0]), separate_spins=True)


# ---------- changed_inds ----------


def test_changed_inds_single_hop():
    Square(2, particle_type="spinless_fermion", Nparticles=2)
    s_old = jnp.array([1.0, -1.0, 1.0, -1.0])  # occ 0, 2
    s_new = jnp.array([1.0, -1.0, -1.0, 1.0])  # particle hops 2 -> 3
    ann, cre = changed_inds(s_new, s_old, 1)
    np.testing.assert_array_equal(np.asarray(ann), [2])
    np.testing.assert_array_equal(np.asarray(cre), [3])


# ---------- permute_sign ----------


def test_permute_sign_no_fermion_between():
    # Occupied {0,2}; hop 2->3 passes no occupied site -> sign +1.
    Square(2, particle_type="spinless_fermion", Nparticles=2)
    idx = jnp.array([0, 2], dtype=jnp.uint16)
    sgn = permute_sign(idx, jnp.array([2], jnp.uint16), jnp.array([3], jnp.uint16))
    assert int(sgn) == 1


def test_permute_sign_one_fermion_between():
    # Occupied {0,1,3}; hop 0->2 passes over the fermion at site 1 -> sign -1.
    Square(2, particle_type="spinless_fermion", Nparticles=3)
    idx = jnp.array([0, 1, 3], dtype=jnp.uint16)
    sgn = permute_sign(idx, jnp.array([0], jnp.uint16), jnp.array([2], jnp.uint16))
    assert int(sgn) == -1


# ---------- fermion_inverse_sign ----------


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_fermion_inverse_sign_matches_reversal_parity(n):
    # Reordering N fermions from 0..N-1 to N-1..0 is N(N-1)/2 transpositions.
    Chain(4, particle_type="spinless_fermion", Nparticles=max(n, 1))
    s = jnp.array([1.0] * n + [-1.0] * (4 - n))
    assert int(fermion_inverse_sign(s)) == (-1) ** (n * (n - 1) // 2)


# ---------- fermion_reorder_sign ----------


def _brute_reorder_sign(s):
    """Parity of the permutation channel-first (all up, then all down) ->
    channel-last (site-interleaved: site0-up, site0-down, site1-up, ...)."""
    s = np.asarray(s)
    nsite = s.size // 2
    occ = (s > 0).astype(int)
    channel_first = [
        c * nsite + i for c in range(2) for i in range(nsite) if occ[c * nsite + i]
    ]
    to_last = {c * nsite + i: i * 2 + c for c in range(2) for i in range(nsite)}
    target = [to_last[m] for m in channel_first]
    inversions = sum(
        target[i] > target[j]
        for i in range(len(target))
        for j in range(i + 1, len(target))
    )
    return 1 - 2 * (inversions % 2)


def test_fermion_reorder_sign_matches_brute_force():
    Square(2, particle_type="spinful_fermion", Nparticles=(2, 2), double_occ=True)
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = jnp.asarray(rng.choice([-1.0, 1.0], size=8))
        assert int(fermion_reorder_sign(s)) == _brute_reorder_sign(s)
