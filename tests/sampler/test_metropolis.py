import logging
import warnings
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
from quantax.operator import Ising, Hubbard
from quantax.sampler import (
    LocalFlip,
    SiteFlip,
    SpinExchange,
    ParticleHopUp,
    ParticleHopDn,
    MixSampler,
)
from quantax.sampler.samples import Samples, SamplesTree
from quantax.sampler.metropolis import (
    Metropolis,
)
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
    with jax.default_matmul_precision("highest"):
        samples = LocalFlip(DenseState(jnp.ones(8)), ns).sweep()
    spins = np.asarray(samples.spins)
    assert spins.shape == (ns, get_sites().Nmodes)
    assert set(np.unique(spins).tolist()).issubset({-1, 1})
    assert samples.state_internal is None
    np.testing.assert_allclose(np.asarray(samples.r), 1.0, rtol=1e-5)


def test_localflip_reweighting_recovers_observable():
    # Markov chain sampling |psi|^1; the reweighting factor must recover
    # <s0>_{|psi|^2} = 0.6 from a proposal whose own <s0> is far lower.
    state, configs, psi = _skewed_dense_state(3)
    p = psi**2 / np.sum(psi**2)
    exact_s0 = float(np.sum(p * configs[:, 0]))

    samples = LocalFlip(state, STAT_NS, reweight=1.0).sweep()
    spins = np.asarray(samples.spins)
    r = np.asarray(samples.r)
    est = float(np.mean(r * spins[:, 0]))
    assert np.isclose(est, exact_s0, atol=0.05)


# --- psi filled in the chunked reference-forward path ---


def test_chunked_ref_oloc_fills_psi():
    # With `use_ref` + chunking the sampler returns no psi, which `Operator.Oloc`
    # fills in by `init_internal` chunk by chunk, including a zero-padded chunk.
    Chain(8, particle_type="spinful_fermion", Nparticles=(4, 4))
    # max_parallel < per-device batch (96) forces chunking; 96 % 50 != 0 forces
    # zero-padding of the final chunk.
    state = GeneralDetState(max_parallel=50)
    sampler = SiteFlip(state, nsamples=96 * NDEV, reweight=1.0, thermal_steps=10)

    assert sampler.use_ref
    chunk = state.ref_chunk
    assert chunk is not None and chunk < sampler.nsamples // NDEV  # chunking active

    samples = sampler.sweep(5)
    assert samples.psi is None and samples.r is None
    Hubbard(U=4.0).Oloc(state, samples)
    psi = np.asarray(jnp.asarray(state(samples.spins)))
    np.testing.assert_allclose(np.asarray(jnp.asarray(samples.psi)), psi, rtol=1e-5)


# --- precision check: low-precision sweep psi is dropped and the samples are
#     reweighted towards the psi filled in by Operator.Oloc ---


@pytest.mark.parametrize("reweight", [2.0, 1.0])
def test_low_precision_reweights_towards_full_forward(reweight, gpu_precision):
    # The sweep psi is distorted by f(s) from psi of `state.__call__`, mimicking the
    # rounding error of a different batch size. In low precision, the sweep psi is
    # not returned, and the samples drawn from |psi f|^n must be reweighted towards
    # |psi|^2 once psi is filled in by `Operator.Oloc`.
    state, _, _ = _skewed_dense_state(3)
    distortion = lambda s: jnp.where(s[:, 1] > 0, 2.0, 1.0)
    fast_forward = state.fast_forward
    state.fast_forward = lambda s: fast_forward(s) * distortion(s)
    # psi is dropped only if the forward chunk exceeds the samples per device
    state.forward_chunk = 128
    # the precision is checked when the sampler is constructed
    with jax.default_matmul_precision("tensorfloat32"):
        sampler = LocalFlip(state, 64 * NDEV, reweight=reweight, thermal_steps=0)
        samples = sampler.sweep()
    assert samples.psi is None and samples.r is None
    psi = np.abs(np.asarray(state(samples.spins)))
    f = np.asarray(distortion(samples.spins))
    # prob is the probability the chains actually followed, from the sweep psi,
    # which is still recorded by the sampler
    np.testing.assert_allclose(np.asarray(samples.prob), (psi * f) ** reweight)
    np.testing.assert_allclose(np.abs(np.asarray(sampler._psi)), psi * f, rtol=1e-6)
    Ising(h=1.0).Oloc(state, samples)
    np.testing.assert_allclose(np.abs(np.asarray(samples.psi)), psi, rtol=1e-6)
    w = psi**2 / (psi * f) ** reweight
    np.testing.assert_allclose(np.asarray(samples.r), w / w.mean(), rtol=1e-5)


@pytest.mark.parametrize("scale", [1e25, 1e-25])
@pytest.mark.parametrize("reweight", [2.0, 1.0])
def test_plain_psi_beyond_float32_range_of_prob(scale, reweight):
    # |psi|^2 of a plain float32 psi ~ 1e25 or 1e-25 overflows or underflows
    # float32, so prob (for n = 2) and r are computed in the LogArray representation
    # and stay finite and accurate.
    state, _, psi = _skewed_dense_state(3)
    state = DenseState(jnp.asarray(psi * scale, jnp.float32), state.symm)
    sampler = LocalFlip(state, 64 * NDEV, reweight=reweight, thermal_steps=0)
    samples = sampler.sweep()
    assert isinstance(samples.prob, LogArray)
    psi = np.abs(np.asarray(state(samples.spins), np.float64)) / scale
    w = psi ** (2 - reweight)
    np.testing.assert_allclose(np.asarray(samples.r), w / w.mean(), rtol=1e-4)
    assert np.isfinite(Ising(h=1.0).expectation(state, samples))


def test_reweight_zero_prob_is_one_for_zero_psi():
    # prob = |psi|^0 = 1 also for psi = 0, so prob stays valid for reweighting the
    # samples towards another wave function, e.g. after sweep(0).
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    configs = ints_to_array(symm.basis.states)
    state = DenseState(jnp.asarray(np.where(configs[:, 0] > 0, 1.0, 0.0)), symm)
    samples = LocalFlip(state, 64 * NDEV, reweight=0.0, thermal_steps=0).sweep()
    assert np.any(np.asarray(samples.psi) == 0)
    np.testing.assert_array_equal(np.asarray(samples.prob), 1.0)


@pytest.mark.parametrize("precision", ["tensorfloat32", "highest"])
@pytest.mark.parametrize("max_parallel", [None, 4, 8, 16])
def test_low_precision_drops_sweep_psi_only_for_smaller_batches(
    max_parallel, precision, gpu_precision
):
    # In low precision, the sweep psi is dropped only if the chains are evaluated in
    # a batch smaller than the forward chunk, with 8 samples per device here,
    # silently as it costs little. Without a chunk, the batch size isn't fixed
    # anyway, which is warned by the state instead. In full precision, the sweep psi
    # is always returned.
    Chain(4, boundary=1)
    low = precision == "tensorfloat32"
    with jax.default_matmul_precision(precision):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            state = Variational(RBM_Dense(features=4), max_parallel=max_parallel)
            sampler = LocalFlip(state, 8 * NDEV, thermal_steps=0)
        samples = sampler.sweep(5)
    messages = [str(w.message) for w in record]
    dropped = low and max_parallel == 16
    no_chunk = any("`forward_chunk` is None" in m for m in messages)
    assert no_chunk == (low and max_parallel is None)
    assert all("`forward_chunk` is None" in m for m in messages)
    assert (samples.psi is None) == dropped
    assert (samples.r is None) == dropped
    psi = np.asarray(state(samples.spins))
    np.testing.assert_allclose(np.asarray(sampler._psi), psi, rtol=1e-5)


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
    r = np.asarray(samples.r)
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


def _prob_aligned(sampler, state, samples) -> None:
    """
    psi carried through the sweep, recorded by the sampler, and the sampling
    probability |psi|^n computed from it at the end of the sweep must belong to the
    returned spins. psi is not returned for states with local updates, and r is
    `Samples.update_r` of psi and prob whenever psi is returned.
    """
    assert samples.state_internal is None
    psi_fresh = np.asarray(state(samples.spins))
    assert np.asarray(sampler._psi).shape == (sampler.nsamples,)
    np.testing.assert_allclose(np.asarray(sampler._psi), psi_fresh, rtol=1e-5)
    if state.use_ref:
        assert samples.psi is None
    if samples.psi is None:
        assert samples.r is None
    else:
        np.testing.assert_array_equal(np.asarray(samples.psi), np.asarray(sampler._psi))
        expected = Samples(samples.spins, psi=samples.psi, prob=samples.prob)
        expected.update_r()
        r = np.asarray(samples.r)
        np.testing.assert_allclose(r, np.asarray(expected.r), rtol=1e-6)
    prob = np.asarray(samples.prob)
    prob_fresh = np.abs(psi_fresh) ** float(sampler.reweight)
    assert prob.shape == (sampler.nsamples,)
    np.testing.assert_allclose(prob, prob_fresh, rtol=1e-5)


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
    _prob_aligned(sampler, state, samples)


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
    _prob_aligned(mix, state, samples)


# --- accept rule (_update): special acceptance conditions for nan/inf/zero psi ---


def _accepted_mask(
    old_psi, new_psi, propose_ratio=None, seed=0, reweight=2.0
) -> np.ndarray:
    """
    Run ``Metropolis._update`` with crafted wave functions (proposed spins always
    differ from the old ones) and return the per-walker accept mask.
    """
    ns = old_psi.shape[0]
    state = DenseState(jnp.ones(2 ** get_sites().Nsites))
    sampler = LocalFlip(state, ns, reweight=reweight, thermal_steps=0)
    old_spins = jnp.tile(_fixed_initial_spins(), (ns, 1))
    new_spins = np.asarray(-old_spins)
    _, out = sampler._update(
        jax.random.key(seed),
        propose_ratio,
        SamplesTree(old_spins, old_psi),
        SamplesTree(-old_spins, new_psi),
    )
    return (np.asarray(out.spins) == new_spins).all(axis=1)


def test_update_escapes_bad_psi():
    # A nan psi is never adopted, and never adopted as a new proposal either.
    # A chain stuck at a nan psi always escapes unless the new psi is also nan.
    # Equal special values (0 -> 0, inf -> inf) are accepted; otherwise the
    # special values are ordered 0 < finite < inf, e.g. inf -> finite is rejected.
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


# The special-value table of `Metropolis._update`, TABLE[old][new] = accepted.
_SPECIAL_TABLE = {
    "0": {"0": True, "1": True, "nan": False, "inf": True},
    "1": {"0": False, "1": True, "nan": False, "inf": True},
    "nan": {"0": True, "1": True, "nan": False, "inf": True},
    "inf": {"0": False, "1": False, "nan": False, "inf": True},
}

# Encodings of the special values of psi: plain floats of both signs, and
# (sign, logabs) of LogArray, with the zero of from_value (sign 0) and of abs or
# negation (sign +-1), a NaN in either component, and a finite value whose dense
# form overflows float32.
_SPECIAL_ENCODINGS = {
    "plain": {
        "0": [0.0, -0.0],
        "1": [1.0, -1.0],
        "nan": [np.nan],
        "inf": [np.inf, -np.inf],
    },
    "logarray": {
        "0": [(0.0, -np.inf), (1.0, -np.inf), (-1.0, -np.inf)],
        "1": [(1.0, 0.0), (-1.0, 200.0)],
        "nan": [(np.nan, 0.0), (1.0, np.nan)],
        "inf": [(1.0, np.inf), (-1.0, np.inf)],
    },
}


@pytest.mark.parametrize("propose_ratio", [None, 0.5, 2.0])
@pytest.mark.parametrize("kind", ["plain", "logarray"])
def test_accept_follows_special_value_table(kind, propose_ratio):
    # Every entry of the table for every pair of encodings, with and without a
    # proposal ratio. Finite -> finite is decided by R, so 1 -> 1 is only taken
    # between equal values and without a proposal ratio (R = 1).
    Chain(3, boundary=1)
    enc = _SPECIAL_ENCODINGS[kind]
    with_ratio = propose_ratio is not None
    old, new, expect = zip(
        *[
            (o, n, accepted)
            for old_key, row in _SPECIAL_TABLE.items()
            for new_key, accepted in row.items()
            for o in enc[old_key]
            for n in enc[new_key]
            if not (old_key == new_key == "1" and (o != n or with_ratio))
        ]
    )

    def make(values):
        x = np.asarray(values, np.float32)
        x = np.tile(x, (NDEV,) + (1,) * (x.ndim - 1))
        if kind == "plain":
            return jnp.asarray(x)
        return LogArray(jnp.asarray(x[:, 0]), jnp.asarray(x[:, 1]))

    if propose_ratio is not None:
        propose_ratio = jnp.full(len(expect) * NDEV, propose_ratio, jnp.float32)
    accepted = _accepted_mask(make(old), make(new), propose_ratio)
    assert np.array_equal(accepted, np.tile(expect, NDEV))


def test_update_accepts_in_float64_with_32bit_random_bits():
    # The acceptance test runs in float64 inside the jitted update, with the uniform
    # random number assembled from 32-bit draws (64-bit random bits can't be
    # generated inside jit with x64 disabled). The returned psi keeps float32, and
    # no prob is carried through the update.
    Chain(3, boundary=1)
    ns = 4 * NDEV
    state = DenseState(jnp.ones(2 ** get_sites().Nsites))
    sampler = LocalFlip(state, ns, thermal_steps=0)
    spins = jnp.tile(_fixed_initial_spins(), (ns, 1))
    # distinct buffers for the old and new samples, which are donated by _update
    args = (
        jax.random.key(0),
        None,
        SamplesTree(spins, jnp.ones(ns, jnp.float32)),
        SamplesTree(-spins, jnp.ones(ns, jnp.float32)),
    )
    jaxpr = str(jax.make_jaxpr(lambda *a: sampler._update(*a))(*args))
    assert "f64[" in jaxpr
    assert "u64" not in jaxpr
    _, out = sampler._update(*args)
    assert out.psi.dtype == jnp.float32


@pytest.mark.parametrize("reweight", [2.0, 1.0])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
def test_update_accept_frequency_matches_rate(dtype, reweight):
    # The float64 test accepts with probability R = |psi'/psi|^n, independent of the
    # sign or phase of psi; checked against binomial noise for a rate below and
    # above 1/2.
    Chain(3, boundary=1)
    n = 100_000 * NDEV
    phase = 1j if jnp.issubdtype(dtype, jnp.complexfloating) else -1.0
    for rate in (0.3, 0.8):
        new = jnp.full(n, phase * rate ** (1 / reweight), dtype)
        old = jnp.ones(n, dtype)
        accepted = _accepted_mask(old, new, seed=3, reweight=reweight)
        sigma = np.sqrt(rate * (1 - rate) / n)
        assert abs(accepted.mean() - rate) < 5 * sigma


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
    # to the state must be upper bounds over the components. The merge only
    # affects the fast local updates, so a state that doesn't use them is not
    # warned about it.
    Chain(4, particle_type="spinful_fermion", Nparticles=(2, 1))
    state = _sector_dense_state(lambda c: np.ones(len(c)))
    up = ParticleHopUp(state, 4 * NDEV, thermal_steps=0)
    dn = ParticleHopDn(state, 4 * NDEV, thermal_steps=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mix = MixSampler([up, dn], thermal_steps=0)
    assert mix.update_mode == {"nflips": 2, "nflips_up": 2, "nflips_dn": 2}


def test_mixsampler_warns_on_merged_update_modes_with_ref_state():
    # A state with fast local updates may become less efficient due to the merged
    # upper bounds, so this is the case where the warning is raised.
    Chain(4, particle_type="spinful_fermion", Nparticles=(2, 2))
    state = Variational(SingletPair())
    assert state.use_ref
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
    _prob_aligned(mix, state, samples)


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
    _prob_aligned(sampler, state, samples)


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
    _prob_aligned(sampler, state, samples)


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
    # among the components is dropped to None, again with a warning only when the
    # state uses the fast local updates.
    Chain(4, particle_type="spinful_fermion", Nparticles=(2, 2))
    ref_state = Variational(SingletPair())
    dense_state = _sector_dense_state(lambda c: np.ones(len(c)))

    class _Tagged(ParticleHopUp):
        def __init__(self, tag, *args, **kwargs):
            self._tag = tag
            super().__init__(*args, **kwargs)

        @property
        def update_mode(self):
            return {**super().update_mode, "tag": self._tag}

    merged = {"nflips": 2, "nflips_up": 2, "nflips_dn": 0, "tag": None}
    s1 = _Tagged("a", ref_state, 4 * NDEV, thermal_steps=0)
    s2 = _Tagged("b", ref_state, 4 * NDEV, thermal_steps=0)
    with pytest.warns(UserWarning, match="update mode 'tag' differs"):
        mix = MixSampler([s1, s2], thermal_steps=0)
    assert mix.update_mode == merged

    s1 = _Tagged("a", dense_state, 4 * NDEV, thermal_steps=0)
    s2 = _Tagged("b", dense_state, 4 * NDEV, thermal_steps=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mix = MixSampler([s1, s2], thermal_steps=0)
    assert mix.update_mode == merged


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


# --- prob: the sampling probability is computed from psi at the end of the sweep in
#     the dtype and array representation of psi, and returned with the samples ---


def test_sweep_without_steps_returns_recorded_prob():
    # nsweeps <= 0 returns the current configurations with the probability
    # recorded by the last sweep (None before the first sweep), but no psi, as the
    # state may have changed since. The samples are reweighted from the recorded
    # prob once psi is filled in.
    state, _, _ = _skewed_dense_state(3)
    sampler = LocalFlip(state, 8 * NDEV, reweight=1.0, thermal_steps=0)
    samples0 = sampler.sweep(0)
    assert samples0.prob is None and samples0.psi is None
    assert sampler._psi is None
    samples = sampler.sweep(5)
    samples0 = sampler.sweep(0)
    assert samples0.psi is None and samples0.r is None
    np.testing.assert_array_equal(np.asarray(samples0.spins), np.asarray(samples.spins))
    np.testing.assert_array_equal(np.asarray(samples0.prob), np.asarray(samples.prob))
    _prob_aligned(sampler, state, samples0)
    Ising(h=1.0).Oloc(state, samples0)
    np.testing.assert_allclose(np.asarray(samples0.r), np.asarray(samples.r), rtol=1e-6)


@pytest.mark.parametrize("reweight", [2.0, 1.0])
def test_prob_keeps_single_precision_under_x64(x64, reweight):
    # Under x64, prob must stay in the dtype of the single-precision wave
    # function on both sweep paths.
    Chain(4, boundary=1)
    state = Variational(RBM_Dense(features=4))
    sampler = LocalFlip(state, 8 * NDEV, reweight=reweight, thermal_steps=0)
    samples = sampler.sweep(5)
    assert samples.prob.dtype == jnp.float32

    symm = Identity()
    symm.basis_make()
    psi = jnp.linspace(1.0, 2.0, symm.basis.Ns, dtype=jnp.float32)
    dense = DenseState(psi, symm)
    # the dense lookup itself promotes to float64 under x64, so the host sweep is
    # fed a float32 wave function explicitly
    fast_forward = dense.fast_forward
    dense.fast_forward = lambda s: jnp.asarray(fast_forward(s), jnp.float32)
    sampler = LocalFlip(dense, 8 * NDEV, reweight=reweight, thermal_steps=0)
    samples = sampler.sweep(5)
    assert samples.prob.dtype == jnp.float32


@pytest.mark.parametrize("log", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_prob_promotes_low_precision_psi(dtype, log):
    # psi below single precision is promoted to float32 before prob = |psi|^n is
    # computed as a LogArray, so log|psi| keeps the float32 accuracy.
    Chain(2, boundary=1)
    symm = Identity()
    symm.basis_make()
    psi = jnp.asarray([1.0, -300.0, 3.0, 4.0], jnp.float32)
    state = DenseState(psi, symm)
    fast_forward = state.fast_forward

    def low_precision_forward(s):
        psi = jnp.asarray(fast_forward(s), dtype)
        return LogArray(jnp.sign(psi), jnp.log(jnp.abs(psi))) if log else psi

    state.fast_forward = low_precision_forward
    sampler = LocalFlip(state, 16 * NDEV, reweight=2.0, thermal_steps=0)
    samples = sampler.sweep(5)
    prob = samples.prob
    psi = low_precision_forward(samples.spins)
    assert isinstance(prob, LogArray)
    assert prob.logabs.dtype == jnp.float32
    if log:
        expected = np.exp(2 * np.asarray(psi.logabs, np.float32))
    else:
        expected = np.asarray(psi, np.float32) ** 2
    np.testing.assert_allclose(np.asarray(prob), expected, rtol=1e-5)


def test_prob_keeps_logarray_representation():
    # A LogArray wave function yields a LogArray prob, also through the chunked
    # sweep of a state with local updates.
    Chain(8, particle_type="spinful_fermion", Nparticles=(4, 4))
    state = GeneralDetState(max_parallel=50)
    sampler = SiteFlip(state, nsamples=96 * NDEV, reweight=1.0, thermal_steps=0)
    assert sampler.use_ref and state.ref_chunk < 96
    samples = sampler.sweep(5)
    assert isinstance(samples.prob, LogArray)
    assert samples.prob.shape == (96 * NDEV,)
    assert samples.prob.logabs.dtype == jnp.float32
    psi = state(samples.spins)
    np.testing.assert_allclose(
        np.asarray(samples.prob.logabs), np.asarray(psi.logabs), rtol=1e-4, atol=1e-4
    )


def test_update_keeps_psi_of_accepted_configurations():
    # After every acceptance step, psi belongs to the configuration kept by the
    # chain, whether the proposal was accepted or rejected.
    state, _, _ = _skewed_dense_state(3)
    sampler = LocalFlip(state, 64 * NDEV, reweight=1.0, thermal_steps=0)
    old_spins = sampler._spins
    old = SamplesTree(old_spins, state(old_spins))
    new_spins = sampler.propose(jax.random.key(1), old_spins)
    new = SamplesTree(new_spins, state(new_spins))
    new_spins = np.asarray(new_spins)  # the inputs of _update are donated
    _, out = sampler._update(jax.random.key(2), None, old, new)
    accepted = np.all(np.asarray(out.spins) == new_spins, axis=1)
    assert 0 < accepted.sum() < accepted.size  # both branches are exercised
    expected = np.asarray(state(out.spins))
    np.testing.assert_allclose(np.asarray(out.psi), expected, rtol=1e-6)
