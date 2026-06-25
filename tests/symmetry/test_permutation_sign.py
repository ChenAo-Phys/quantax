import numpy as np
import jax
import jax.numpy as jnp
import pytest
import quantax as qtx
from quantax.sites import Square, Grid
from quantax.symmetry import TransND, Rotation, Flip, SpinInverse
from quantax.symmetry.symmetry import _inversion_parity, _permutation_sign


def _bruteforce_inversion_parity(a: np.ndarray) -> int:
    """Parity of #{(i, j): i < j, a[i] > a[j]} (strict; ties are not inversions)."""
    n = len(a)
    inv = 0
    for i in range(n):
        for j in range(i + 1, n):
            if a[i] > a[j]:
                inv += 1
    return inv % 2


# --- unit test: the O(N) cycle-decomposition parity matches the definition ---


@pytest.mark.parametrize("n", [0, 1, 2, 5, 17, 100, 200, 400])
def test_inversion_parity_matches_bruteforce_distinct(n):
    rng = np.random.default_rng(n)
    f = jax.jit(_inversion_parity)
    for _ in range(20):
        a = rng.permutation(n).astype(np.int64)
        got = int(f(jnp.asarray(a)))
        assert got == _bruteforce_inversion_parity(a)


@pytest.mark.parametrize("n", [5, 17, 100, 200])
def test_inversion_parity_handles_ties(n):
    # The Ntotal=None branch feeds arrays padded with a repeated dummy value, so
    # ties must be broken by position (stable) exactly like the brute-force count.
    rng = np.random.default_rng(n + 1)
    f = jax.jit(_inversion_parity)
    for _ in range(20):
        a = rng.integers(0, max(2, n // 3), size=n).astype(np.int64)
        got = int(f(jnp.asarray(a)))
        assert got == _bruteforce_inversion_parity(a)


# --- reference for the full _permutation_sign, used in integration tests ---


def _ref_permutation_sign(spins, perm, perm_sign, Ntotal):
    p = np.argsort(perm)
    if Ntotal is None:
        Nmodes = int(p.max()) + 1
        pp = np.where(spins > 0, p, Nmodes)
        arg = np.argsort(pp == Nmodes, kind="stable")
        q = pp[arg]
    else:
        q = p[np.flatnonzero(spins > 0)[:Ntotal]]
    sign = -1 if _bruteforce_inversion_parity(q) else 1
    additional = np.sum((spins > 0) & (perm_sign < 0))
    psign = 1 if additional % 2 == 0 else -1
    return sign * psign


def _check_against_reference(syms, configs):
    Ntotal = qtx.get_sites().Ntotal
    batched = jax.jit(jax.vmap(_permutation_sign, in_axes=(0, None, None)))
    for sym in syms:
        perm = np.asarray(sym._perm)
        perm_sign = np.asarray(sym._perm_sign)
        got = np.asarray(batched(jnp.asarray(configs), sym._perm, sym._perm_sign))
        ref = np.array(
            [
                [
                    _ref_permutation_sign(s, perm[g], perm_sign[g], Ntotal)
                    for g in range(perm.shape[0])
                ]
                for s in configs
            ]
        )
        assert np.array_equal(got, ref)


def _fermion_configs(Nmodes, Nsites, nup, ndn, nconf, seed):
    rng = np.random.default_rng(seed)
    configs = np.full((nconf, Nmodes), -1, dtype=np.int8)
    for b in range(nconf):
        up = rng.choice(Nsites, nup, replace=False)
        dn = rng.choice(Nsites, ndn, replace=False) + Nsites
        configs[b, up] = 1
        configs[b, dn] = 1
    return configs


# --- integration: _permutation_sign on real lattice-symmetry perms ---


def test_permutation_sign_fixed_Nparticles():
    # Ntotal-known branch, exercised by translations, rotations, flips and spin
    # inversion on a spinful-fermion lattice at half filling.
    L = 4
    Square(L, particle_type="spinful_fermion", Nparticles=(L * L // 2, L * L // 2))
    jax.clear_caches()  # _permutation_sign bakes in get_sites().Ntotal at trace time
    syms = [TransND(0), Rotation(np.pi / 2, sector=0), Flip(sector=0), SpinInverse()]
    configs = _fermion_configs(2 * L * L, L * L, L * L // 2, L * L // 2, 12, seed=0)
    _check_against_reference(syms, configs)


def test_permutation_sign_Ntotal_none():
    # Ntotal=None branch (Nparticles unspecified): the slow padded path that ranks
    # an array containing repeated dummy values.
    Grid([3, 3], particle_type="spinless_fermion")
    assert qtx.get_sites().Ntotal is None
    jax.clear_caches()
    syms = [TransND(0), Rotation(np.pi / 2, sector=0), Flip(sector=0)]
    rng = np.random.default_rng(3)
    configs = (rng.integers(0, 2, size=(16, 9)) * 2 - 1).astype(np.int8)
    _check_against_reference(syms, configs)
