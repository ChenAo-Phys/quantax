from typing import Callable
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from scipy.stats import chi2
from quantax.sites import Chain
from quantax.state import DenseState
from quantax.symmetry import Identity
from quantax.sampler import (
    LocalFlip,
    SpinExchange,
    ParticleHop,
    ParticleHopUp,
    ParticleHopDn,
    SiteExchange,
    SiteFlip,
    MixSampler,
)
from quantax.global_defs import get_sites, get_subkeys
from _stats import sector_configs, config_index, histogram

NDEV = jax.device_count()
# All sample counts are multiples of NDEV so that `nsamples % device_count == 0`
# holds for any device layout (CPU CI uses 4, the workstation uses GPUs).
SPINLESS = dict(particle_type="spinless_fermion")
SPINFUL = dict(particle_type="spinful_fermion")


def _uniform_dense_state() -> DenseState:
    """A flat, positive `DenseState` over the current (sector-restricted) basis."""
    symm = Identity()
    symm.basis_make()
    return DenseState(jnp.ones(symm.basis.Ns), symm)


# ====================================================================
# Validation of the sector required by the samplers
# ====================================================================


def test_spinexchange_requires_magnetization_sector():
    # Without a (Nup, Ndown) tuple the Sz sector is undefined.
    Chain(4, boundary=1)  # spin, Nparticles defaults to the int Nsites
    state = _uniform_dense_state()
    with pytest.raises(ValueError):
        SpinExchange(state, 4 * NDEV)


def test_particlehop_requires_particle_number():
    Chain(4, particle_type="spinless_fermion")  # Nparticles None -> Ntotal None
    state = _uniform_dense_state()
    with pytest.raises(ValueError):
        ParticleHop(state, 4 * NDEV)


def test_particlehop_sector_requires_sector_numbers():
    # A total particle number is not enough; the sector numbers must be fixed. The
    # check is shared by ParticleHopUp and ParticleHopDn, which differ only by
    # their sector.
    Chain(4, particle_type="spinful_fermion", Nparticles=3)
    state = _uniform_dense_state()
    with pytest.raises(ValueError):
        ParticleHopUp(state, 4 * NDEV)


@pytest.mark.parametrize("sampler_cls", [ParticleHopUp, ParticleHopDn])
def test_particlehop_sector_rejects_wrong_particle_type(sampler_cls):
    # Spin sectors of hopping particles are only defined for spinful fermions.
    Chain(4, boundary=1, Nparticles=(2, 2))  # spin system
    state = _uniform_dense_state()
    with pytest.raises(ValueError, match="not supported"):
        sampler_cls(state, 4 * NDEV)


def test_siteexchange_requires_particle_number():
    Chain(4, particle_type="spinful_fermion")  # Nparticles None
    state = _uniform_dense_state()
    with pytest.raises(ValueError):
        SiteExchange(state, 4 * NDEV)


# ====================================================================
# Proposal kernels
# ====================================================================
#
# None of these samplers returns a proposal ratio, so Metropolis-Hastings samples
# |psi|^2 only if the proposal probability Q(s -> s') is symmetric. The kernel is
# measured directly by proposing from every configuration of the sector many times,
# without any Metropolis step. On an open chain the end sites have fewer neighbors
# than the bulk, which the neighbor table pads with -1 (no hop); choosing among the
# actual neighbors only would make Q asymmetric. Hops across the periodic boundary
# or to second neighbors change the order of the hopping particles, so a choice of
# the particle depending on its position would make Q asymmetric as well.


def _hops(blocks: tuple[int, ...]) -> Callable:
    """Exchange of different occupations on a bond, within each block of N modes."""

    def moves(c, N, bonds):
        for b in blocks:
            for i, j in bonds + b * N:
                if c[i] != c[j]:
                    new = c.copy()
                    new[[i, j]] = c[[j, i]]
                    yield new

    return moves


def _site_exchanges(c, N, bonds):
    """Exchange of the whole (up, down) content of the two sites of a bond."""
    for i, j in bonds:
        new = c.copy()
        new[[i, i + N, j, j + N]] = c[[j, j + N, i, i + N]]
        if np.any(new != c):
            yield new


def _site_flips(c, N, bonds):
    """Exchange of the up and down occupations of a site."""
    for i in range(N):
        new = c.copy()
        new[[i, i + N]] = c[[i + N, i]]
        if np.any(new != c):
            yield new


def _single_flips(c, N, bonds):
    """Flip of any single mode."""
    for m in range(len(c)):
        new = c.copy()
        new[m] *= -1
        yield new


def _mix_up_dn(state, nsamples, **kwargs):
    # Neither component is ergodic alone, the mixture hops both spin species.
    up = ParticleHopUp(state, nsamples, **kwargs)
    dn = ParticleHopDn(state, nsamples, **kwargs)
    return MixSampler([up, dn], thermal_steps=0)


# id: (lattice of a 4-site chain, open except for the "ring" case, sampler, neighbor
# orders, legal moves). 2 * Nparticles > Nmodes hops holes instead of particles
# (the "holes" cases).
_PROPOSERS = {
    "LocalFlip": (dict(), LocalFlip, None, _single_flips),
    "SpinExchange-nn12": (dict(Nparticles=(2, 2)), SpinExchange, (1, 2), _hops((0,))),
    "SpinExchange-holes": (dict(Nparticles=(3, 1)), SpinExchange, 1, _hops((0,))),
    "ParticleHop-spinless-ring": (
        dict(**SPINLESS, Nparticles=2, boundary=1),
        ParticleHop,
        1,
        _hops((0,)),
    ),
    "ParticleHop-spinful": (
        dict(**SPINFUL, Nparticles=(2, 1)),
        ParticleHop,
        1,
        _hops((0, 1)),
    ),
    "ParticleHopUp-holes": (
        dict(**SPINFUL, Nparticles=(3, 1)),
        ParticleHopUp,
        1,
        _hops((0,)),
    ),
    "ParticleHopDn": (
        dict(**SPINFUL, Nparticles=(3, 1)),
        ParticleHopDn,
        1,
        _hops((1,)),
    ),
    "ParticleHopUp+Dn": (
        dict(**SPINFUL, Nparticles=(2, 1)),
        _mix_up_dn,
        1,
        _hops((0, 1)),
    ),
    "SiteExchange": (
        dict(**SPINFUL, Nparticles=(2, 1)),
        SiteExchange,
        1,
        _site_exchanges,
    ),
    "SiteFlip": (dict(**SPINFUL, Nparticles=3), SiteFlip, None, _site_flips),
}


@pytest.mark.parametrize("case", list(_PROPOSERS))
def test_proposals_are_symmetric_and_reach_all_legal_moves(case):
    # The empirical kernel must be supported exactly on the legal moves (which
    # implies the conservation laws of the sampler, e.g. the other spin block is
    # never touched by ParticleHopUp, and its ergodicity within them), symmetric,
    # and changing no more modes than the update mode tells the state.
    lattice, sampler_cls, n_neighbor, legal_moves = _PROPOSERS[case]
    Chain(4, **{"boundary": 0, **lattice})
    sites = get_sites()
    N = sites.Nsites
    configs = sector_configs()
    kwargs = dict(thermal_steps=0, initial_spins=jnp.asarray(configs[0]))
    if n_neighbor is not None:
        kwargs["n_neighbor"] = n_neighbor
    sampler = sampler_cls(_uniform_dense_state(), NDEV, **kwargs)

    reps = 2000
    old = np.repeat(configs, reps, axis=0)
    new, ratio = sampler._propose_spins_and_ratio(get_subkeys(), jnp.asarray(old))
    assert ratio is None
    new = np.asarray(new)
    n = len(configs)
    src = np.repeat(np.arange(n), reps)
    dst = config_index(new, configs)
    q = np.bincount(src * n + dst, minlength=n * n).reshape(n, n) / reps

    bonds = None
    if n_neighbor is not None:
        orders = [n_neighbor] if isinstance(n_neighbor, int) else list(n_neighbor)
        bonds = np.concatenate(sites.get_neighbor(orders), axis=0)
    legal = np.zeros((n, n), dtype=bool)
    for a, c in enumerate(configs):
        for c_new in legal_moves(c, N, bonds):
            legal[a, config_index(c_new[None], configs)[0]] = True
    off_diag = ~np.eye(n, dtype=bool)
    np.testing.assert_array_equal(q[off_diag] > 0, legal[off_diag])
    if legal_moves is _single_flips:
        assert np.all(np.diag(q) == 0)  # a flip always changes the configuration

    # detailed balance without a proposal ratio, within 5 sigma per transition
    sigma = np.sqrt((q * (1 - q) + q.T * (1 - q.T)) / reps)
    assert np.all(np.abs(q - q.T) <= 5 * sigma)

    changed = old != new
    mode = sampler.update_mode
    assert changed.sum(axis=1).max() == mode["nflips"]
    if "nflips_up" in mode:
        assert changed[:, :N].sum(axis=1).max() == mode["nflips_up"]
        assert changed[:, N:].sum(axis=1).max() == mode["nflips_dn"]


# ====================================================================
# Sampling with Metropolis
# ====================================================================


def test_particlehop_spinful_samples_target_distribution():
    # Hops of both spin species through the Metropolis sweep must reproduce the
    # exact |psi|^2 over the (Nup, Ndown) sector. From a uniform start, the chain
    # on this ring relaxes as 0.82^t (exact transition matrix), so the 30
    # thermalization steps plus one sweep of 16 steps leave a bias ~6e-6, far below
    # the 5 sigma ~ 0.02 of the histogram of the independent chains. Sampling
    # |psi| instead would be off by 9 sigma. A distortion spread over many
    # configurations is detected by the chi-square of the whole histogram: a frozen
    # spin-down fermion gives a mean chi2 ~ 85 against the threshold 71 (p = 1e-6),
    # while no single configuration is expected off by more than 4 sigma.
    Chain(4, particle_type="spinful_fermion", Nparticles=(2, 1))
    symm = Identity()
    symm.basis_make()
    configs = sector_configs()
    occ = configs > 0
    psi = 1.0 + 2.0 * occ[:, 0] + 1.0 * occ[:, 1] + 0.5 * occ[:, 4]
    p = psi**2 / np.sum(psi**2)
    state = DenseState(jnp.asarray(psi), symm)

    ns = 4000 // NDEV * NDEV  # the power quoted above assumes 4000 chains
    # uniform start drawn on the host, which saves compiling `rand_states`
    init = configs[np.random.default_rng(0).integers(len(configs), size=ns)]
    sampler = ParticleHop(state, ns, thermal_steps=30, initial_spins=jnp.asarray(init))
    samples = sampler.sweep()
    # the sweep psi is returned, so the reweighting is trivial
    np.testing.assert_allclose(np.asarray(samples.r), 1.0, rtol=1e-5)
    counts = histogram(samples.spins, configs)
    assert np.all(np.abs(counts / ns - p) <= 5 * np.sqrt(p * (1 - p) / ns))
    stat = np.sum((counts - ns * p) ** 2 / (ns * p))
    assert stat < chi2.isf(1e-6, df=len(configs) - 1)
