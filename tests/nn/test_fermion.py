"""
Unit tests for the Jordan-Wigner sign bookkeeping in quantax/nn/fermion.py.

These pure functions (occupied-index extraction, hop detection, and the three
fermion-ordering signs) are load-bearing for the determinant/Pfaffian models and
are exactly the error-prone sign logic that needs an independent reference. They
are evaluated jitted and vmapped over many configurations, compared against
brute-force references.
"""

import itertools
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from quantax.sites import Sites, Lattice, Square, Chain
from quantax.nn import (
    fermion_idx,
    changed_inds,
    permute_sign,
    fermion_inverse_sign,
    fermion_reorder_sign,
)


def _all_configs(N, n=None):
    """All +-1 configurations of N modes, or those with n occupied modes."""
    configs = np.array(list(itertools.product([1.0, -1.0], repeat=N)), np.float32)
    if n is not None:
        configs = configs[np.sum(configs > 0, axis=1) == n]
    return configs


def _parity(seq) -> int:
    """(-1) ** (number of inversions of ``seq``)."""
    seq = list(seq)
    inv = sum(a > b for i, a in enumerate(seq) for b in seq[i + 1 :])
    return 1 - 2 * (inv % 2)


# ---------- fermion_idx ----------


def test_fermion_idx_spinless():
    Chain(6, particle_type="spinless_fermion", Nparticles=3)
    s = _all_configs(6, 3)
    idx = np.asarray(jax.jit(jax.vmap(fermion_idx))(s))
    assert idx.dtype == np.uint16
    np.testing.assert_array_equal(idx, [np.flatnonzero(si > 0) for si in s])


def test_fermion_idx_spin_maps_to_two_modes():
    # A spin config maps to 2N fermion modes: up occupation then down occupation
    # (down = "spin not up"), so every site contributes exactly one occupied mode.
    Chain(4)
    s = _all_configs(4)
    idx = np.asarray(jax.jit(jax.vmap(fermion_idx))(s))
    expected = [np.flatnonzero(np.concatenate([si > 0, si < 0])) for si in s]
    np.testing.assert_array_equal(idx, expected)


def test_fermion_idx_separate_spins():
    Chain(4, particle_type="spinful_fermion", Nparticles=(2, 1))
    # up half (modes 0-3), down half (modes 4-7)
    s = _all_configs(8)
    s = s[(np.sum(s[:, :4] > 0, 1) == 2) & (np.sum(s[:, 4:] > 0, 1) == 1)]
    f = jax.vmap(lambda s: fermion_idx(s, separate_spins=True))
    idx_up, idx_dn = jax.jit(f)(s)
    np.testing.assert_array_equal(idx_up, [np.flatnonzero(si[:4] > 0) for si in s])
    # indices within the down block
    np.testing.assert_array_equal(idx_dn, [np.flatnonzero(si[4:] > 0) for si in s])


def test_fermion_idx_separate_spins_requires_spinful():
    Square(2, particle_type="spinless_fermion", Nparticles=2)
    with pytest.raises(ValueError):
        fermion_idx(jnp.array([1.0, -1.0, 1.0, -1.0]), separate_spins=True)


# ---------- changed_inds / permute_sign ----------


def _hop_reference(occ_old, occ_new, nhops):
    """Sorted annihilated / created modes, padded with the fill value ``nmodes``."""
    ann = np.flatnonzero(occ_old & ~occ_new)
    cre = np.flatnonzero(occ_new & ~occ_old)
    pad = lambda m: np.pad(m, (0, nhops - m.size), constant_values=occ_old.size)
    return pad(ann), pad(cre)


def _inplace_sign(occ_old, ann, cre):
    """Sign relating the determinant rows of the new configuration in sorted order
    to the rows after a local update, which overwrites the row of the k-th
    annihilated mode by the k-th created mode: the parity of sorting the
    overwritten list."""
    rows = list(np.flatnonzero(occ_old))
    for a, c in zip(ann, cre):
        if a < occ_old.size:
            rows[rows.index(a)] = c
    return _parity(rows)


def _check_hops(s_old, s_new, nhops, to_modes):
    """Compare changed_inds and permute_sign with the brute-force references."""

    @jax.jit
    @jax.vmap
    def f(s_new, s_old):
        idx_ann, idx_cre = changed_inds(s_new, s_old, nhops)
        sign = permute_sign(fermion_idx(s_old), idx_ann, idx_cre)
        return idx_ann, idx_cre, sign

    ann, cre, sign = (np.asarray(a) for a in f(s_new, s_old))
    for i in range(s_old.shape[0]):
        occ_old, occ_new = to_modes(s_old[i]), to_modes(s_new[i])
        ann_ref, cre_ref = _hop_reference(occ_old, occ_new, nhops)
        np.testing.assert_array_equal(ann[i], ann_ref)
        np.testing.assert_array_equal(cre[i], cre_ref)
        assert sign[i] == _inplace_sign(occ_old, ann_ref, cre_ref)
    return sign


def _random_hops(s_old, nhops, rng):
    """Move ``nhops`` random particles of each configuration to random holes."""
    s_new = s_old.copy()
    for s, k in zip(s_new, nhops):
        src = rng.choice(np.flatnonzero(s > 0), k, replace=False)
        dst = rng.choice(np.flatnonzero(s < 0), k, replace=False)
        s[src], s[dst] = -1.0, 1.0
    return s_new


def test_permute_sign_spinless_hops():
    # 1 and 2 hops with nhops = 2, an upper bound as in the samplers: missing hops
    # are filled with the number of modes and must not change the sign. Two hops
    # can cross each other's endpoints, which the moving fermions must not count.
    Chain(8, particle_type="spinless_fermion", Nparticles=4)
    rng = np.random.default_rng(0)
    s_old = _all_configs(8, 4)
    s_old = s_old[rng.permutation(len(s_old))[:64]]
    s_new = _random_hops(s_old, rng.integers(1, 3, size=len(s_old)), rng)
    sign = _check_hops(s_old, s_new, 2, lambda s: s > 0)
    assert np.any(sign == 1) and np.any(sign == -1)


def test_permute_sign_spin_exchange_and_flip():
    # On the 2N modes of fermion_idx, a spin exchange moves one up fermion and one
    # down fermion (2 hops), and a single spin flip moves a fermion between modes i
    # and N + i (1 hop, filled up to nhops = 2).
    Chain(6)
    rng = np.random.default_rng(1)
    s_old = _all_configs(6)
    s_old = s_old[np.abs(s_old.sum(1)) < 6]  # at least one anti-aligned pair
    s_old = s_old[rng.permutation(len(s_old))[:64]]
    s_new = s_old.copy()
    for k, s in enumerate(s_new):
        if k % 2:
            s[rng.integers(6)] *= -1
        else:
            i = rng.choice(np.flatnonzero(s > 0))
            j = rng.choice(np.flatnonzero(s < 0))
            s[i], s[j] = s[j], s[i]
    to_modes = lambda s: np.concatenate([s > 0, s < 0])
    sign = _check_hops(s_old, s_new, 2, to_modes)
    assert np.any(sign == 1) and np.any(sign == -1)


# ---------- fermion_inverse_sign ----------


def test_fermion_inverse_sign_matches_reversal_parity():
    # QuSpin labels mode i as N-1-i, so its ordering of the occupied modes is the
    # reverse of the usual one.
    Chain(5, particle_type="spinless_fermion")
    s = _all_configs(5)
    sign = np.asarray(jax.jit(jax.vmap(fermion_inverse_sign))(s))
    expected = [_parity(4 - np.flatnonzero(si > 0)) for si in s]
    np.testing.assert_array_equal(sign, expected)


# ---------- fermion_reorder_sign ----------


def _reorder_sign_reference(s, nchannels):
    """Parity of the permutation of the occupied modes from channel-first order
    (mode c * ncells + i) to channel-last order (mode i * nchannels + c)."""
    ncells = s.size // nchannels
    occ = np.flatnonzero(s > 0)
    return _parity([(m % ncells) * nchannels + m // ncells for m in occ])


def _two_site_chain(particle_type):
    return Lattice((3,), [[1.0]], [[0.0], [0.3]], particle_type=particle_type)


@pytest.mark.parametrize(
    "make_sites, nchannels",
    [
        # the channels are (spin, sublattice) on lattices with several sites per cell
        (lambda: _two_site_chain("spinful_fermion"), 4),
        (lambda: _two_site_chain("spinless_fermion"), 2),
        # without a lattice the channels are the spins only
        (lambda: Sites(4, particle_type="spinful_fermion"), 2),
    ],
    ids=["two_site_spinful", "two_site_spinless", "sites_spinful"],
)
def test_fermion_reorder_sign_matches_brute_force(make_sites, nchannels):
    sites = make_sites()
    s = _all_configs(sites.Nmodes)
    sign = np.asarray(jax.jit(jax.vmap(fermion_reorder_sign))(s))
    expected = [_reorder_sign_reference(si, nchannels) for si in s]
    np.testing.assert_array_equal(sign, expected)
