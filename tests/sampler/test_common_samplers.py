import numpy as np
import jax
import jax.numpy as jnp
import pytest
from quantax.sites import Chain
from quantax.state import DenseState
from quantax.symmetry import Identity
from quantax.sampler import (
    SpinExchange,
    ParticleHop,
    ParticleHopUp,
    ParticleHopDn,
    SiteExchange,
    SiteFlip,
    MixSampler,
)
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
    assert samples.reweight_factor is None  # reweight=2 -> trivial factor
    est = float(np.mean(spins[:, 0]))
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
    assert samples.reweight_factor is None  # reweight=2 -> trivial factor
    est = float(np.mean(spins[:, 0]))
    assert np.isclose(est, exact_s0, atol=0.05)


# ====================================================================
# ParticleHopUp / ParticleHopDn  (spinful fermions, single-sector hops)
# ====================================================================


@pytest.mark.parametrize("sampler_cls", [ParticleHopUp, ParticleHopDn])
def test_particlehop_sector_requires_sector_numbers(sampler_cls):
    # A total particle number is not enough; the sector numbers must be fixed.
    Chain(4, particle_type="spinful_fermion", Nparticles=3)
    state = _uniform_dense_state()
    with pytest.raises(ValueError):
        sampler_cls(state, 4 * NDEV)


@pytest.mark.parametrize(
    "sampler_cls, sector", [(ParticleHopUp, 0), (ParticleHopDn, 1)]
)
def test_particlehop_sector_moves_only_own_sector(sampler_cls, sector):
    # Nup > Nsites / 2 also exercises the hole-hopping branch for ParticleHopUp.
    Nup, Ndn = 3, 1
    Chain(4, particle_type="spinful_fermion", Nparticles=(Nup, Ndn))
    N = get_sites().Nsites
    ns = 8 * NDEV

    initial = np.tile(np.array([1, 1, 1, -1] + [1, -1, -1, -1], dtype=np.int8), (ns, 1))
    sampler = sampler_cls(
        _uniform_dense_state(), ns, thermal_steps=20, initial_spins=jnp.asarray(initial)
    )
    spins = np.asarray(sampler.sweep().spins)

    up, dn = spins[:, :N], spins[:, N:]
    assert np.all((up == 1).sum(axis=1) == Nup)
    assert np.all((dn == 1).sum(axis=1) == Ndn)
    # the other sector is never touched
    frozen = np.split(initial, 2, axis=1)[1 - sector]
    assert np.array_equal([up, dn][1 - sector], frozen)


def test_particlehop_up_dn_mix_samples_target_distribution():
    # Neither sampler is ergodic alone; their mixture must reproduce <s0>.
    Chain(4, particle_type="spinful_fermion", Nparticles=(2, 1))
    state, configs, psi = _biased_dense_state()
    p = psi**2 / np.sum(psi**2)
    exact_s0 = float(np.sum(p * configs[:, 0]))

    up = ParticleHopUp(state, STAT_NS // 2, thermal_steps=0)
    dn = ParticleHopDn(state, STAT_NS // 2, thermal_steps=0)
    samples = MixSampler([up, dn]).sweep()  # default reweight=2.0
    spins = np.asarray(samples.spins)
    assert samples.reweight_factor is None
    est = float(np.mean(spins[:, 0]))
    assert np.isclose(est, exact_s0, atol=0.05)


def test_particlehop_sector_update_modes():
    # The single-sector samplers report exact flip numbers per spin sector,
    # while the plain ParticleHop only reports the total.
    Chain(4, particle_type="spinful_fermion", Nparticles=(2, 2))
    state = _uniform_dense_state()
    ns = 4 * NDEV
    hop = ParticleHop(state, ns, thermal_steps=0)
    up = ParticleHopUp(state, ns, thermal_steps=0)
    dn = ParticleHopDn(state, ns, thermal_steps=0)
    assert hop.update_mode == {"nflips": 2}
    assert up.update_mode == {"nflips": 2, "nflips_up": 2, "nflips_dn": 0}
    assert dn.update_mode == {"nflips": 2, "nflips_up": 0, "nflips_dn": 2}


@pytest.mark.parametrize("sampler_cls", [ParticleHopUp, ParticleHopDn])
def test_particlehop_sector_rejects_wrong_particle_type(sampler_cls):
    # Spin sectors of hopping particles are only defined for spinful fermions.
    Chain(4, boundary=1, Nparticles=(2, 2))  # spin system
    state = _uniform_dense_state()
    with pytest.raises(ValueError, match="not supported"):
        sampler_cls(state, 4 * NDEV)


@pytest.mark.parametrize("sampler_cls", [ParticleHopUp, ParticleHopDn])
def test_particlehop_sector_respects_double_occupancy(sampler_cls):
    # Single-sector hops can propose doubly occupied sites, which must be
    # rejected when double occupancy is forbidden.
    Chain(4, particle_type="spinful_fermion", Nparticles=(2, 2), double_occ=False)
    N = get_sites().Nsites
    ns = 8 * NDEV
    samples = sampler_cls(_uniform_dense_state(), ns, thermal_steps=20).sweep()
    spins = np.asarray(samples.spins)
    up, dn = spins[:, :N], spins[:, N:]
    assert not np.any((up == 1) & (dn == 1))
    assert np.all((up == 1).sum(axis=1) == 2)
    assert np.all((dn == 1).sum(axis=1) == 2)


def test_particlehop_sector_fast_updates_with_singletpair():
    # The sector samplers provide the update modes required by SingletPair for
    # spinful fermions, while the plain ParticleHop falls back to direct forward.
    from quantax.model import SingletPair
    from quantax.state import Variational

    Chain(4, particle_type="spinful_fermion", Nparticles=(2, 2))
    state = Variational(SingletPair())
    ns = 4 * NDEV
    up = ParticleHopUp(state, ns, thermal_steps=0)
    dn = ParticleHopDn(state, ns, thermal_steps=0)
    assert up.use_ref and dn.use_ref
    assert MixSampler([up, dn], thermal_steps=0).use_ref

    with pytest.warns(UserWarning, match="update_modes required by the state"):
        hop = ParticleHop(state, ns, thermal_steps=0)
    assert not hop.use_ref


def test_particlehop_up_dn_mix_ref_samples_target_distribution():
    # Sampling through the low-rank ref updates of SingletPair must reproduce
    # the exact |psi|^2 distribution of the state.
    from quantax.model import SingletPair
    from quantax.state import Variational

    Chain(4, particle_type="spinful_fermion", Nparticles=(2, 2))
    state = Variational(SingletPair())

    dense = state.todense()
    configs = ints_to_array(dense.basis.states)
    psi = np.asarray(dense.psi)
    p = np.abs(psi) ** 2 / np.sum(np.abs(psi) ** 2)
    exact_s0 = float(np.sum(p * configs[:, 0]))

    up = ParticleHopUp(state, STAT_NS // 2, thermal_steps=0)
    dn = ParticleHopDn(state, STAT_NS // 2, thermal_steps=0)
    mix = MixSampler([up, dn])
    assert mix.use_ref
    spins = np.asarray(mix.sweep().spins)
    est = float(np.mean(spins[:, 0]))
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
