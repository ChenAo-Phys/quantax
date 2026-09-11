import logging
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from quantax import set_random_seed
from quantax.sites import Chain
from quantax.state import DenseState, Variational, GeneralDetState
from quantax.model import RBM_Dense, SingletPair
from quantax.symmetry import Identity
from quantax.sampler import (
    LocalFlip,
    SiteFlip,
    SpinExchange,
    ParticleHopUp,
    ParticleHopDn,
    MixSampler,
)
from quantax.sampler.samples import Samples
from quantax.sampler.metropolis import Metropolis
from quantax.utils import LogArray, ints_to_array
from quantax.global_defs import get_sites

NDEV = jax.device_count()
# All sample counts are multiples of NDEV so that `nsamples % device_count == 0`
# holds for any device layout (CPU CI uses 4, the workstation uses GPUs).
STAT_NS = 10000 * NDEV


def _skewed_dense_state(L: int) -> tuple[DenseState, np.ndarray, np.ndarray]:
    """
    A real, positive `DenseState` on a length-``L`` spin chain biased toward
    ``s[0] == +1`` (amplitude 2 vs 1), with no Sz conservation so the full
    ``2**L`` Hilbert space is reachable by single-spin flips.
    """
    Chain(L, boundary=1)
    symm = Identity()
    symm.basis_make()
    configs = ints_to_array(symm.basis.states)  # (Ns, L), in basis order
    psi = np.where(configs[:, 0] > 0, 2.0, 1.0).astype(np.float64)
    return DenseState(jnp.asarray(psi), symm), configs, psi


# --- LocalFlip ---


def test_localflip_outputs_valid_pm1_samples():
    Chain(3, boundary=1)
    ns = 8 * NDEV
    samples = LocalFlip(DenseState(jnp.ones(8)), ns).sweep()
    spins = np.asarray(samples.spins)
    assert spins.shape == (ns, get_sites().Nmodes)
    assert set(np.unique(spins).tolist()).issubset({-1, 1})
    assert samples.state_internal is None
    assert samples.reweight_factor is None  # reweight=2 -> trivial factor


def test_localflip_reweighting_recovers_observable():
    # Markov chain sampling |psi|^1; the reweighting factor must recover
    # <s0>_{|psi|^2} = 0.6 from a proposal whose own <s0> is far lower.
    state, configs, psi = _skewed_dense_state(3)
    p = psi**2 / np.sum(psi**2)
    exact_s0 = float(np.sum(p * configs[:, 0]))

    samples = LocalFlip(state, STAT_NS, reweight=1.0).sweep()
    spins = np.asarray(samples.spins)
    r = np.asarray(samples.reweight_factor)
    est = float(np.mean(r * spins[:, 0]))
    assert np.isclose(est, exact_s0, atol=0.05)


# --- reweight normalization in the chunked reference-forward path ---


def test_chunked_ref_reweight_is_globally_normalized():
    # Regression: with `use_ref` + chunking the reweighting factor must be
    # normalized over ALL samples, not per chunk. Per-chunk normalization used to
    # bias the mean by the zero-padding ratio (here 100/96) for reweight != 2.
    Chain(8, particle_type="spinful_fermion", Nparticles=(4, 4))
    # max_parallel < per-device batch (96) forces chunking; 96 % 50 != 0 forces
    # zero-padding of the final chunk, which is what exposed the bug.
    state = GeneralDetState(max_parallel=50)
    sampler = SiteFlip(state, nsamples=96 * NDEV, reweight=1.0, thermal_steps=10)

    assert sampler.use_ref
    chunk = state.ref_chunk
    assert chunk is not None and chunk < sampler.nsamples // NDEV  # chunking active

    samples = sampler.sweep(5)
    r = np.asarray(samples.reweight_factor)
    assert r.shape == (96 * NDEV,)
    assert np.all(np.isfinite(r))
    assert np.isclose(r.mean(), 1.0, atol=1e-4)


# --- MixSampler ---


def test_mixsampler_rejects_mismatched_reweight():
    Chain(4, boundary=1)
    state = Variational(RBM_Dense(features=4))
    s1 = LocalFlip(state, 4 * NDEV, reweight=2.0, thermal_steps=0)
    s2 = LocalFlip(state, 4 * NDEV, reweight=1.0, thermal_steps=0)
    with pytest.raises(ValueError):
        MixSampler([s1, s2])


def test_mixsampler_inherits_subsampler_reweight():
    Chain(4, boundary=1)
    state = Variational(RBM_Dense(features=4))
    s1 = LocalFlip(state, 4 * NDEV, reweight=1.5, thermal_steps=0)
    s2 = LocalFlip(state, 4 * NDEV, reweight=1.5, thermal_steps=0)
    mix = MixSampler([s1, s2], thermal_steps=0)
    assert float(mix.reweight) == 1.5
    assert mix.nsamples == 8 * NDEV


def test_mixsampler_sweep_smoke():
    Chain(4, boundary=1)
    state = Variational(RBM_Dense(features=4))
    s1 = LocalFlip(state, 4 * NDEV, reweight=1.0, thermal_steps=5)
    s2 = LocalFlip(state, 4 * NDEV, reweight=1.0, thermal_steps=5)
    mix = MixSampler([s1, s2], thermal_steps=5)
    samples = mix.sweep()
    spins = np.asarray(samples.spins)
    assert spins.shape == (8 * NDEV, get_sites().Nmodes)
    assert set(np.unique(spins).tolist()).issubset({-1, 1})
    r = np.asarray(samples.reweight_factor)
    assert r.shape == (8 * NDEV,)
    assert np.isclose(r.mean(), 1.0, atol=1e-4)


# --- chunked sweep: max_parallel < ns/dev sweeps the Markov chains chunk by chunk
#     through chunk_map(_partial_sweep), for the direct path as well as the ref one ---


def _fixed_initial_spins() -> jax.Array:
    return jnp.asarray(np.resize([1, -1], get_sites().Nmodes).astype(np.int8))


def _exact_probs(state) -> tuple[np.ndarray, np.ndarray]:
    """All configurations of the current chain and their exact |psi|^2 weights."""
    symm = Identity()
    symm.basis_make()
    configs = ints_to_array(symm.basis.states)
    psi = np.asarray(state(jnp.asarray(configs)), dtype=np.complex128)
    p = np.abs(psi) ** 2
    return configs, p / p.sum()


def _config_codes(spins: np.ndarray) -> np.ndarray:
    """Bijective integer code of +-1 configurations, for histogramming."""
    weights = 1 << np.arange(spins.shape[1])
    return ((spins > 0).astype(np.int64) * weights).sum(axis=1)


def _psi_aligned(sampler, state, samples) -> None:
    """The psi carried through the sweep must belong to the returned spins."""
    psi_carried = np.asarray(sampler._psi)
    psi_fresh = np.asarray(state(samples.spins))
    assert psi_carried.shape == (sampler.nsamples,)
    np.testing.assert_allclose(psi_carried, psi_fresh, rtol=1e-5)


def test_chunked_direct_sweep_calls_partial_sweep_per_chunk():
    # A non-RefModel takes the direct path. With forward_chunk 4 < ns/dev = 10 the
    # sweep must be routed through chunk_map(_partial_sweep): 10 is padded to 12
    # and _partial_sweep is called on 3 chunks of 4 walkers per device.
    Chain(4, boundary=1)
    state = Variational(RBM_Dense(features=4), max_parallel=4)
    ns = 10 * NDEV
    sampler = LocalFlip(state, ns, thermal_steps=0)
    assert not sampler.use_ref and state.forward_chunk == 4 < ns // NDEV

    batches = []
    partial_sweep = sampler._partial_sweep

    def spy(nsweeps, spins):
        batches.append(spins.shape[0])
        return partial_sweep(nsweeps, spins)

    sampler._partial_sweep = spy
    samples = sampler.sweep(20)
    assert batches == [4 * NDEV] * 3

    # padded walkers are truncated away and the outputs stay aligned
    spins = np.asarray(samples.spins)
    assert spins.shape == (ns, get_sites().Nmodes)
    assert set(np.unique(spins).tolist()).issubset({-1, 1})
    _psi_aligned(sampler, state, samples)


def test_chunked_direct_sweep_samples_target_distribution():
    # Sweeping chunk by chunk must not bias the stationary distribution: the
    # histogram over all 2**L configurations of the chunked sampler must match the
    # exact |psi|^2 of the state. ns/dev = 200 is not a multiple of the chunk 24.
    Chain(4, boundary=1)
    set_random_seed(0)
    state = Variational(RBM_Dense(features=4), max_parallel=24)
    ns = 200 * NDEV
    sampler = LocalFlip(state, ns, thermal_steps=40)
    assert not sampler.use_ref and state.forward_chunk == 24 < ns // NDEV

    configs, p_exact = _exact_probs(state)
    lookup = np.empty(len(configs), dtype=np.int64)
    lookup[_config_codes(configs)] = np.arange(len(configs))

    counts = np.zeros(len(configs))
    for _ in range(25):
        spins = np.asarray(sampler.sweep().spins)
        counts += np.bincount(lookup[_config_codes(spins)], minlength=len(configs))
    freq = counts / counts.sum()
    assert np.abs(freq - p_exact).max() < 0.02


def test_mixsampler_chunked_direct_sweep():
    # MixSampler goes through the same chunk_map(_partial_sweep) path, drawing the
    # component sampler per step inside every chunk.
    Chain(4, boundary=1)
    state = Variational(RBM_Dense(features=4), max_parallel=4)
    s1 = LocalFlip(state, 5 * NDEV, thermal_steps=0)
    s2 = LocalFlip(state, 5 * NDEV, thermal_steps=0)
    mix = MixSampler([s1, s2], initial_spins=_fixed_initial_spins(), thermal_steps=0)
    assert not mix.use_ref and state.forward_chunk == 4 < mix.nsamples // NDEV

    samples = mix.sweep(20)
    spins = np.asarray(samples.spins)
    assert spins.shape == (10 * NDEV, get_sites().Nmodes)
    assert set(np.unique(spins).tolist()).issubset({-1, 1})
    _psi_aligned(mix, state, samples)


# --- accept rule (_update): special acceptance conditions for nan/inf/zero psi ---


def _accepted_mask(old_psi, new_psi) -> np.ndarray:
    """
    Run ``Metropolis._update`` with crafted wavefunction values (proposed spins
    always differ from the old ones) and return the per-walker accept mask.
    """
    ns = old_psi.shape[0]
    sampler = LocalFlip(DenseState(jnp.ones(2 ** get_sites().Nsites)), ns)
    old_spins = jnp.tile(_fixed_initial_spins(), (ns, 1))
    new_spins = np.asarray(-old_spins)
    _, out = sampler._update(
        jax.random.key(0),
        None,
        Samples(old_spins, old_psi),
        Samples(-old_spins, new_psi),
    )
    return (np.asarray(out.spins) == new_spins).all(axis=1)


def test_update_escapes_bad_psi():
    # A nan psi is never adopted, and never adopted as a new proposal either.
    # nan old psi always escapes (any non-nan ratio is nan, treated as a forced
    # accept) unless the new psi is also nan. old == new (0 == 0 or inf == inf)
    # is treated like any other equal-magnitude pair and accepted; old > new
    # (e.g. inf -> finite) is correctly rejected by the ordinary rate.
    Chain(3, boundary=1)
    nan, inf = np.nan, np.inf
    old = [nan, nan, 1.0, 1.0, inf, inf, 0.0, 0.0]
    new = [1.0, nan, nan, 2.0, inf, 1.0, 0.0, 1.0]
    expect = [True, False, False, True, True, False, True, True]

    old_psi = jnp.tile(jnp.asarray(old, jnp.float32), NDEV)
    new_psi = jnp.tile(jnp.asarray(new, jnp.float32), NDEV)
    accepted = _accepted_mask(old_psi, new_psi)
    assert np.array_equal(accepted, np.tile(expect, NDEV))


def test_update_escapes_bad_psi_logarray():
    # Same special conditions with LogArray psi: nan-ness must be read from
    # components (isnan), not the densified value which overflows at large
    # finite logabs.
    Chain(3, boundary=1)
    nan, inf = np.nan, np.inf
    old_sign, old_logabs = [nan, 1.0, 1.0, 1.0], [0.0, inf, 0.0, inf]
    new_sign, new_logabs = [1.0, 1.0, nan, 1.0], [0.0, inf, 0.0, 0.0]
    expect = [True, True, False, False]

    def make(sign, logabs):
        return LogArray(
            jnp.tile(jnp.asarray(sign, jnp.float32), NDEV),
            jnp.tile(jnp.asarray(logabs, jnp.float32), NDEV),
        )

    accepted = _accepted_mask(make(old_sign, old_logabs), make(new_sign, new_logabs))
    assert np.array_equal(accepted, np.tile(expect, NDEV))


# --- free steps: proposals that leave the spins unchanged are counted as steps
#     without evaluating the wavefunction, and every chain takes exactly nsweeps steps ---


def _count_forward_calls(state) -> list[int]:
    """Spy on ``state.fast_forward`` and record the batch size of every call."""
    batches = []
    fast_forward = state.fast_forward

    def spy(s):
        batches.append(s.shape[0])
        return fast_forward(s)

    state.fast_forward = spy
    return batches


def _sector_dense_state(amplitude) -> DenseState:
    """A `DenseState` on the current (sector-restricted) basis with the given amplitudes."""
    symm = Identity()
    symm.basis_make()
    configs = ints_to_array(symm.basis.states)
    return DenseState(jnp.asarray(amplitude(configs), dtype=jnp.float32), symm)


def test_free_steps_reduce_wavefunction_evaluations():
    # SpinExchange proposes unchanged spins whenever the chosen bond is parallel.
    # Those proposals are counted as free steps, so a sweep of nsweeps steps needs
    # fewer than nsweeps batched evaluations (plus one for the initial spins), each
    # on the full batch. LocalFlip always changes the spins: exactly nsweeps rounds.
    Chain(6, boundary=1, Nparticles=(3, 3))
    state = _sector_dense_state(lambda c: np.ones(len(c)))
    ns = 8 * NDEV
    nsweeps = 40
    batches = _count_forward_calls(state)
    SpinExchange(state, ns, thermal_steps=0).sweep(nsweeps)
    assert all(n == ns for n in batches)
    n_rounds = len(batches) - 1
    assert 1 <= n_rounds < nsweeps


def test_localflip_evaluates_every_step():
    Chain(4, boundary=1)
    state = DenseState(jnp.ones(16))
    ns = 8 * NDEV
    nsweeps = 20
    batches = _count_forward_calls(state)
    LocalFlip(state, ns, thermal_steps=0).sweep(nsweeps)
    assert batches == [ns] * (nsweeps + 1)


def test_every_chain_takes_exactly_nsweeps_steps():
    # With a flat state every changed proposal is accepted. After sweep(1) a
    # LocalFlip chain differs from its start in exactly one site, and a
    # SpinExchange chain in 0 (free step) or 2 sites, with a single evaluation
    # round in both cases. sweep(0) returns the current spins without evaluating.
    Chain(6, boundary=1, Nparticles=(3, 3))
    state = _sector_dense_state(lambda c: np.ones(len(c)))
    ns = 16 * NDEV

    sampler = SpinExchange(state, ns, thermal_steps=0)
    s0 = np.asarray(sampler._spins)
    batches = _count_forward_calls(state)
    s1 = np.asarray(sampler.sweep(1).spins)
    assert len(batches) == 2
    assert set(np.sum(s1 != s0, axis=1).tolist()).issubset({0, 2})

    batches.clear()
    s2 = np.asarray(sampler.sweep(0).spins)
    assert len(batches) == 0
    assert np.array_equal(s2, s1)


def test_localflip_sweep_one_flips_every_chain_once():
    Chain(4, boundary=1)
    state = DenseState(jnp.ones(16))
    sampler = LocalFlip(state, 16 * NDEV, thermal_steps=0)
    s0 = np.asarray(sampler._spins)
    s1 = np.asarray(sampler.sweep(1).spins)
    assert np.all(np.sum(s1 != s0, axis=1) == 1)


def test_free_steps_sample_target_distribution():
    # Regression for the sampling bias of evaluating only changed proposals: the
    # probability of an unchanged proposal depends on the configuration (0 for
    # the Neel states, 1/2 for the domain states of this ring), so a chain that
    # stopped right after a changed proposal would sample |psi|^2 (1 - u) instead
    # of |psi|^2. Counting the free steps and stopping every chain at exactly
    # nsweeps steps must reproduce the exact distribution.
    Chain(4, boundary=1, Nparticles=(2, 2))
    set_random_seed(0)
    state = _sector_dense_state(
        lambda c: 1.0 + 0.8 * (c[:, 0] > 0) + 0.4 * (c[:, 1] > 0) + 0.2 * (c[:, 2] > 0)
    )
    configs, p_exact = _exact_probs(state)
    lookup = np.empty(1 << get_sites().Nmodes, dtype=np.int64)
    lookup[_config_codes(configs)] = np.arange(len(configs))

    sampler = SpinExchange(state, 2000 * NDEV, thermal_steps=40)
    counts = np.zeros(len(configs))
    for _ in range(10):
        spins = np.asarray(sampler.sweep().spins)
        counts += np.bincount(lookup[_config_codes(spins)], minlength=len(configs))
    freq = counts / counts.sum()
    assert np.abs(freq - p_exact).max() < 0.01


def test_mixsampler_merges_update_modes_as_upper_bounds():
    # Proposals of one batch come from different components, so the modes passed
    # to the state must be upper bounds over the components.
    Chain(4, particle_type="spinful_fermion", Nparticles=(2, 1))
    state = _sector_dense_state(lambda c: np.ones(len(c)))
    up = ParticleHopUp(state, 4 * NDEV, thermal_steps=0)
    dn = ParticleHopDn(state, 4 * NDEV, thermal_steps=0)
    with pytest.warns(UserWarning, match="merged into its maximum"):
        mix = MixSampler([up, dn], thermal_steps=0)
    assert mix.update_mode == {"nflips": 2, "nflips_up": 2, "nflips_dn": 2}


def test_mixsampler_ref_free_steps_sample_target_distribution():
    # Mixed ParticleHopUp/Dn proposals evaluated in one batch through the low-rank
    # updates of SingletPair (merged update modes, free steps for hops onto occupied
    # sites) must reproduce the exact |psi|^2 over all configurations.
    Chain(4, particle_type="spinful_fermion", Nparticles=(2, 2))
    set_random_seed(0)
    # A smooth pairing matrix: the default paired Fermi sea is peaked on 6 of the
    # 36 configurations and its Metropolis chain relaxes only in ~2000 steps,
    # whereas this one relaxes in ~15 steps (spectral gap 0.065).
    N = get_sites().Nsites
    F = 1 + 0.5 * np.cos(np.subtract.outer(np.arange(N), np.arange(N)) * np.pi / 2)
    F += 0.2 * np.random.default_rng(0).standard_normal((N, N))
    state = Variational(SingletPair(F=jnp.asarray(F, jnp.float32)))
    configs, p_exact = _exact_probs(state)
    lookup = np.empty(1 << get_sites().Nmodes, dtype=np.int64)
    lookup[_config_codes(configs)] = np.arange(len(configs))

    up = ParticleHopUp(state, 1000 * NDEV, thermal_steps=0)
    dn = ParticleHopDn(state, 1000 * NDEV, thermal_steps=0)
    mix = MixSampler([up, dn], thermal_steps=100)
    assert mix.use_ref
    counts = np.zeros(len(configs))
    for _ in range(10):
        samples = mix.sweep()
        spins = np.asarray(samples.spins)
        counts += np.bincount(lookup[_config_codes(spins)], minlength=len(configs))
    freq = counts / counts.sum()
    assert np.abs(freq - p_exact).max() < 0.01
    _psi_aligned(mix, state, samples)


# --- proposal ratios: an asymmetric proposer must have its P(s|s')/P(s'|s) carried
#     along with the pending proposal through the free-step loop, and MixSampler
#     must unify components with and without ratios ---


class _BiasedFlip(Metropolis):
    """
    Single spin flips with an asymmetric, configuration-dependent proposal. When
    ``s[0] == +1`` site 0 is flipped with probability 3/4 (else a uniformly random
    other site); otherwise one of the ``N + 1`` options "flip site i" / "do nothing"
    is drawn uniformly, so unchanged proposals occur with a state-dependent
    probability. Returns the proposal ratio required by Metropolis-Hastings.
    """

    @partial(jax.jit, static_argnums=0)
    def propose(self, key, old_spins):
        ns, N = old_spins.shape
        k1, k2, k3 = jax.random.split(key, 3)
        first_up = old_spins[:, 0] > 0
        pos_uniform = jax.random.choice(k1, N + 1, (ns,))  # N means "do nothing"
        use_first = jax.random.uniform(k2, (ns,)) < 0.75
        pos_other = 1 + jax.random.choice(k3, N - 1, (ns,))
        pos_biased = jnp.where(use_first, 0, pos_other)
        pos = jnp.where(first_up, pos_biased, pos_uniform)
        flip = pos < N
        flipped = old_spins.at[jnp.arange(ns), jnp.minimum(pos, N - 1)].multiply(-1)
        new_spins = jnp.where(flip[:, None], flipped, old_spins)

        def q(first_up, pos):  # probability of proposing to flip ``pos``
            q_biased = jnp.where(pos == 0, 0.75, 0.25 / (N - 1))
            return jnp.where(first_up, q_biased, 1.0 / (N + 1))

        ratio = q(new_spins[:, 0] > 0, pos) / q(first_up, pos)
        return new_spins, jnp.where(flip, ratio, 1.0)


def _histogram_matches_exact(sampler, state, nsweeps: int, atol: float) -> None:
    configs, p_exact = _exact_probs(state)
    lookup = np.empty(1 << get_sites().Nmodes, dtype=np.int64)
    lookup[_config_codes(configs)] = np.arange(len(configs))
    counts = np.zeros(len(configs))
    for _ in range(nsweeps):
        spins = np.asarray(sampler.sweep().spins)
        counts += np.bincount(lookup[_config_codes(spins)], minlength=len(configs))
    freq = counts / counts.sum()
    assert np.abs(freq - p_exact).max() < atol


def test_propose_ratio_carried_through_free_steps():
    state, _, _ = _skewed_dense_state(3)
    set_random_seed(0)
    sampler = _BiasedFlip(state, 2000 * NDEV, thermal_steps=20)
    _histogram_matches_exact(sampler, state, nsweeps=10, atol=0.01)


def test_mixsampler_unifies_propose_ratios():
    # LocalFlip returns no ratio and gets ratio 1 inside the mixture.
    state, _, _ = _skewed_dense_state(3)
    set_random_seed(0)
    biased = _BiasedFlip(state, 1000 * NDEV, thermal_steps=0)
    plain = LocalFlip(state, 1000 * NDEV, thermal_steps=0)
    mix = MixSampler([biased, plain], thermal_steps=20)
    _histogram_matches_exact(mix, state, nsweeps=10, atol=0.01)


def test_free_steps_under_x64(x64):
    # Under x64, argmax and Python scalars promote to int64; the step counters must
    # keep a single dtype in the carry of the free-step loop.
    Chain(4, boundary=1, Nparticles=(2, 2))
    state = _sector_dense_state(lambda c: np.ones(len(c)))
    sampler = SpinExchange(state, 8 * NDEV, thermal_steps=0)
    spins = np.asarray(sampler.sweep(10).spins)
    assert spins.shape == (8 * NDEV, get_sites().Nmodes)
    assert np.all(spins.sum(axis=1) == 0)


# --- device sweep: Variational states run the whole sweep in one jitted call ---


def test_variational_sweep_runs_on_device():
    # Neither the host loop nor state.fast_forward is used; the sweep calls the
    # jittable forward of the state with the model passed explicitly.
    Chain(4, boundary=1, Nparticles=(2, 2))
    state = Variational(RBM_Dense(features=4))
    sampler = SpinExchange(state, 8 * NDEV, thermal_steps=0)
    called = []
    sampler._sweep_host = lambda *args: called.append("host")
    state.fast_forward = lambda s: called.append("fast_forward")
    samples = sampler.sweep(20)
    assert called == []
    spins = np.asarray(samples.spins)
    assert spins.shape == (8 * NDEV, get_sites().Nmodes)
    assert np.all(spins.sum(axis=1) == 0)
    _psi_aligned(sampler, state, samples)


def test_variational_sweep_fixed_rounds_take_one_step_per_round():
    # LocalFlip (_n_candidates == 1) takes exactly one step per chain and round:
    # after sweep(1) every chain differs from its start in at most one site.
    Chain(4, boundary=1)
    state = Variational(RBM_Dense(features=4))
    sampler = LocalFlip(state, 16 * NDEV, thermal_steps=0)
    s0 = np.asarray(sampler._spins)
    samples = sampler.sweep(1)
    s1 = np.asarray(samples.spins)
    assert set(np.sum(s1 != s0, axis=1).tolist()).issubset({0, 1})
    _psi_aligned(sampler, state, samples)


def test_variational_sweep_under_x64(x64):
    # The loop carries (step counters, samples) must keep their dtypes under x64.
    Chain(4, boundary=1, Nparticles=(2, 2))
    state = Variational(RBM_Dense(features=4))
    for sampler in (
        SpinExchange(state, 8 * NDEV, thermal_steps=0),
        LocalFlip(state, 8 * NDEV, thermal_steps=0),
    ):
        spins = np.asarray(sampler.sweep(10).spins)
        assert spins.shape == (8 * NDEV, get_sites().Nmodes)


def test_variational_sweep_reuses_compilation_after_parameter_update(caplog):
    # The model is an argument of the jitted sweep, so an optimizer update of the
    # parameters must not trigger a recompilation of the whole sweep.
    Chain(4, boundary=1)
    state = Variational(RBM_Dense(features=4))
    sampler = LocalFlip(state, 8 * NDEV, thermal_steps=0)
    jax.config.update("jax_log_compiles", True)
    try:
        with caplog.at_level(logging.INFO, logger="jax"):
            sampler.sweep(4)
            n_first = sum("Compiling" in r.getMessage() for r in caplog.records)
            state.update(0.01 * jnp.ones(state.nparams, jnp.float32))
            caplog.clear()
            sampler.sweep(4)
            n_second = sum("Compiling" in r.getMessage() for r in caplog.records)
    finally:
        jax.config.update("jax_log_compiles", False)
    assert n_first >= 1  # positive control: the first sweep compiles the device loop
    assert n_second == 0


def test_mixsampler_warns_on_unmergeable_update_mode():
    # Only the flip numbers have an upper-bound meaning; any other mode that differs
    # among the components is dropped to None with a warning.
    Chain(4, boundary=1)
    state = DenseState(jnp.ones(16))

    class _Tagged(LocalFlip):
        def __init__(self, tag, *args, **kwargs):
            self._tag = tag
            super().__init__(*args, **kwargs)

        @property
        def update_mode(self):
            return {"nflips": 1, "tag": self._tag}

    s1 = _Tagged("a", state, 4 * NDEV, thermal_steps=0)
    s2 = _Tagged("b", state, 4 * NDEV, thermal_steps=0)
    with pytest.warns(UserWarning, match="update mode 'tag' differs"):
        mix = MixSampler([s1, s2], thermal_steps=0)
    assert mix.update_mode == {"nflips": 1, "tag": None}


def test_mixsampler_candidates_follow_components():
    # None if any component uses the default, else the largest component value.
    Chain(4, boundary=1)
    state = DenseState(jnp.ones(16))

    class _TwoCandidates(LocalFlip):
        _n_candidates = 2

    flips = [LocalFlip(state, 4 * NDEV, thermal_steps=0) for _ in range(2)]
    assert MixSampler(flips, thermal_steps=0)._n_candidates == 1
    two = _TwoCandidates(state, 4 * NDEV, thermal_steps=0)
    assert MixSampler([flips[0], two], thermal_steps=0)._n_candidates == 2

    class _Default(LocalFlip):
        _n_candidates = None

    default = _Default(state, 4 * NDEV, thermal_steps=0)
    assert MixSampler([flips[0], default], thermal_steps=0)._n_candidates is None
