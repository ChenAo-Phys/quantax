import numpy as np
import jax
import jax.numpy as jnp
import pytest
from quantax.sites import Chain
from quantax.state import DenseState
from quantax.symmetry import Identity
from quantax.sampler import SpinExchange, ParticleHop, SiteExchange, SiteFlip
from quantax.utils import ints_to_array
from quantax.global_defs import get_sites

NDEV = jax.device_count()
# All sample counts are multiples of NDEV so that `nsamples % device_count == 0`
# holds for any device layout (CPU CI uses 4, the workstation uses GPUs).
STAT_NS = 10000 * NDEV


def _uniform_dense_state() -> DenseState:
    """A flat, positive `DenseState` over the current (sector-restricted) basis."""
    symm = Identity()
    symm.basis_make()
    return DenseState(jnp.ones(symm.basis.Ns), symm)


def _biased_dense_state() -> tuple[DenseState, np.ndarray, np.ndarray]:
    """
    A real, positive `DenseState` over the current basis with amplitude 2 on
    configs whose first mode is occupied (else 1). Returns the state together
    with the basis-ordered configs and (un-normalized) wave function so a test
    can compute exact expectations of ``s[0]`` under :math:`|\\psi|^2`.
    """
    symm = Identity()
    symm.basis_make()
    configs = ints_to_array(symm.basis.states)  # (Ns, Nmodes), in basis order
    psi = np.where(configs[:, 0] > 0, 2.0, 1.0).astype(np.float64)
    return DenseState(jnp.asarray(psi), symm), configs, psi


# ====================================================================
# SpinExchange  (spin systems, fixed (Nup, Ndown))
# ====================================================================


def test_spinexchange_requires_magnetization_sector():
    # Without a (Nup, Ndown) tuple the Sz sector is undefined.
    Chain(4, boundary=1)  # spin, Nparticles defaults to the int Nsites
    state = _uniform_dense_state()
    with pytest.raises(ValueError):
        SpinExchange(state, 4 * NDEV)


def test_spinexchange_conserves_magnetization():
    Nup, Ndn = 3, 1
    Chain(4, boundary=1, Nparticles=(Nup, Ndn))
    ns = 8 * NDEV
    samples = SpinExchange(_uniform_dense_state(), ns, thermal_steps=20).sweep()
    spins = np.asarray(samples.spins)
    assert spins.shape == (ns, get_sites().Nmodes)
    assert set(np.unique(spins).tolist()).issubset({-1, 1})
    # neighbor exchanges never change the up / down spin counts
    assert np.all((spins == 1).sum(axis=1) == Nup)
    assert np.all((spins == -1).sum(axis=1) == Ndn)


def test_spinexchange_samples_target_distribution():
    # |psi|^2 sampling within a fixed-Sz sector must reproduce <s0>.
    Chain(4, boundary=1, Nparticles=(2, 2))
    state, configs, psi = _biased_dense_state()
    p = psi**2 / np.sum(psi**2)
    exact_s0 = float(np.sum(p * configs[:, 0]))

    samples = SpinExchange(state, STAT_NS).sweep()  # default reweight=2.0
    spins = np.asarray(samples.spins)
    r = np.asarray(samples.reweight_factor)
    est = float(np.mean(r * spins[:, 0]))
    assert np.isclose(est, exact_s0, atol=0.05)


# ====================================================================
# ParticleHop  (fermions, fixed particle number)
# ====================================================================


def test_particlehop_requires_particle_number():
    Chain(4, particle_type="spinless_fermion")  # Nparticles None -> Ntotal None
    state = _uniform_dense_state()
    with pytest.raises(ValueError):
        ParticleHop(state, 4 * NDEV)


def test_particlehop_conserves_particle_number_spinless():
    Ntot = 2
    Chain(4, particle_type="spinless_fermion", Nparticles=Ntot)
    ns = 8 * NDEV
    samples = ParticleHop(_uniform_dense_state(), ns, thermal_steps=20).sweep()
    spins = np.asarray(samples.spins)
    assert spins.shape == (ns, get_sites().Nmodes)
    assert set(np.unique(spins).tolist()).issubset({-1, 1})
    assert np.all((spins == 1).sum(axis=1) == Ntot)


def test_particlehop_conserves_spin_blocks_spinful():
    # Hops stay within a spin block, so up and down counts are each conserved.
    # This exercises the down-mode neighbor offset in `_get_site_neighbors`.
    Nup, Ndn = 2, 1
    Chain(4, particle_type="spinful_fermion", Nparticles=(Nup, Ndn))
    N = get_sites().Nsites
    ns = 8 * NDEV
    samples = ParticleHop(_uniform_dense_state(), ns, thermal_steps=20).sweep()
    spins = np.asarray(samples.spins)
    up, dn = spins[:, :N], spins[:, N:]
    assert set(np.unique(spins).tolist()).issubset({-1, 1})
    assert np.all((up == 1).sum(axis=1) == Nup)
    assert np.all((dn == 1).sum(axis=1) == Ndn)


def test_particlehop_samples_target_distribution():
    Chain(4, particle_type="spinless_fermion", Nparticles=2)
    state, configs, psi = _biased_dense_state()
    p = psi**2 / np.sum(psi**2)
    exact_s0 = float(np.sum(p * configs[:, 0]))

    samples = ParticleHop(state, STAT_NS).sweep()  # default reweight=2.0
    spins = np.asarray(samples.spins)
    r = np.asarray(samples.reweight_factor)
    est = float(np.mean(r * spins[:, 0]))
    assert np.isclose(est, exact_s0, atol=0.05)


# ====================================================================
# SiteExchange  (spinful fermions, swap whole sites)
# ====================================================================


def test_siteexchange_requires_particle_number():
    Chain(4, particle_type="spinful_fermion")  # Nparticles None
    state = _uniform_dense_state()
    with pytest.raises(ValueError):
        SiteExchange(state, 4 * NDEV)


def test_siteexchange_conserves_site_occupation_multiset():
    # Swapping whole sites only permutes the per-site (n_up, n_dn) types, so both
    # the per-spin counts and the multiset of site-occupation types are invariant.
    Nup, Ndn = 2, 1
    Chain(4, particle_type="spinful_fermion", Nparticles=(Nup, Ndn))
    N = get_sites().Nsites
    # site 0 doubly occupied, site 1 single up -> a distinctive occupation multiset
    init = np.array([1, 1, -1, -1, 1, -1, -1, -1], dtype=np.int8)
    ns = 8 * NDEV
    samples = SiteExchange(
        _uniform_dense_state(), ns, thermal_steps=20, initial_spins=jnp.asarray(init)
    ).sweep()
    spins = np.asarray(samples.spins)
    up, dn = spins[:, :N], spins[:, N:]
    assert spins.shape == (ns, 2 * N)
    assert set(np.unique(spins).tolist()).issubset({-1, 1})
    assert np.all((up == 1).sum(axis=1) == Nup)
    assert np.all((dn == 1).sum(axis=1) == Ndn)
    # encode each site as 2 * n_up + n_dn in {0, 1, 2, 3}; the sorted pattern is fixed
    site_type = (up == 1) * 2 + (dn == 1)
    expected = np.sort((init[:N] == 1) * 2 + (init[N:] == 1))
    assert np.all(np.sort(site_type, axis=1) == expected)


# ====================================================================
# SiteFlip  (spinful fermions, swap up/down on a site)
# ====================================================================


def test_siteflip_conserves_per_site_and_total_number():
    # Flipping up<->down on a site preserves the per-site occupation number and the
    # total particle number (but not Nup / Ndn individually), so it needs a
    # total-N sector rather than a (Nup, Ndown) one.
    Ntot = 4
    Chain(4, particle_type="spinful_fermion", Nparticles=Ntot)
    N = get_sites().Nsites
    init = np.array([1, 1, -1, -1, -1, -1, 1, 1], dtype=np.int8)  # all sites singly occ
    ns = 8 * NDEV
    samples = SiteFlip(
        _uniform_dense_state(), ns, thermal_steps=20, initial_spins=jnp.asarray(init)
    ).sweep()
    spins = np.asarray(samples.spins)
    up, dn = spins[:, :N], spins[:, N:]
    n_site_init = (init[:N] == 1).astype(int) + (init[N:] == 1).astype(int)
    n_site = (up == 1).astype(int) + (dn == 1).astype(int)
    assert set(np.unique(spins).tolist()).issubset({-1, 1})
    assert np.all(n_site == n_site_init)  # per-site occupation conserved
    assert np.all((spins == 1).sum(axis=1) == Ntot)  # total particle number conserved
