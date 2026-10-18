import logging
import warnings
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from scipy.stats import chi2
from quantax.sites import Chain
from quantax.state import State, DenseState, Variational
from quantax.model import SingletPair
from quantax.symmetry import Identity
from quantax.operator import create_u, annihilate_u, sigma_z
from quantax.sampler import (
    LocalFlip,
    SpinExchange,
    ParticleHop,
    ParticleHopUp,
    ParticleHopDn,
    MixSampler,
)
from quantax.sampler.samples import Samples, SamplesTree
from quantax.sampler import metropolis
from quantax.sampler.metropolis import Metropolis
from quantax.utils import LogArray, ints_to_array
from quantax.global_defs import get_sites
from _stats import sector_configs, skewed_dense_state, config_codes, histogram

NDEV = jax.device_count()
# All sample counts are multiples of NDEV so that `nsamples % device_count == 0`
# holds for any device layout (CPU CI uses 4, the workstation uses GPUs). Tests share
# the sample counts where they don't matter, which shares the compilations of the
# operations on the samples.
NS = 16 * NDEV


def _stat_ns(n: int) -> int:
    """
    ``n`` rounded up to a multiple of NDEV, so that the statistical tests keep their
    power for any device count.
    """
    return -(-n // NDEV) * NDEV


STAT_NS = _stat_ns(16000)


class _ExpLinear(eqx.Module):
    """psi(s) = exp(w . s), a model without random initialization, cheap to compile."""

    w: jax.Array

    def __call__(self, s):
        return jnp.exp(jnp.dot(self.w, s.astype(self.w.dtype)))


def _exp_linear_state(**kwargs) -> Variational:
    """A `Variational` state of `_ExpLinear` with distinct weights on all modes."""
    w = np.linspace(-0.4, 0.6, get_sites().Nmodes)
    return Variational(_ExpLinear(jnp.asarray(w, jnp.float32)), **kwargs)


def _sector_dense_state(amplitude) -> DenseState:
    """A `DenseState` with the given amplitudes on the current (sector) basis."""
    symm = Identity()
    symm.basis_make()
    configs = ints_to_array(symm.basis.states)
    return DenseState(jnp.asarray(amplitude(configs), dtype=jnp.float32), symm)


def _fixed_initial_spins() -> jax.Array:
    return jnp.asarray(np.resize([1, -1], get_sites().Nmodes).astype(np.int8))


def _exact_probs(state, reweight: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """All configurations of the current basis and their exact |psi|^n weights."""
    configs = sector_configs()
    psi = np.asarray(state(jnp.asarray(configs)), dtype=np.complex128)
    p = np.abs(psi) ** reweight
    return configs, p / p.sum()


def _histogram(sampler, configs, nsweeps: int = 1) -> tuple[np.ndarray, Samples]:
    """Counts of ``configs`` in ``nsweeps`` sweeps, and the samples of the last one."""
    counts = np.zeros(len(configs))
    for _ in range(nsweeps):
        samples = sampler.sweep()
        counts += histogram(samples.spins, configs)
    return counts, samples


def _assert_follows(counts: np.ndarray, p: np.ndarray) -> None:
    """
    Chi-square test of the counts of independent samples, e.g. one per Markov chain,
    against the exact distribution ``p``, failing by chance with probability 1e-6
    for expected counts >~ 5 (a few times more with expected counts ~1, where the
    tail of the chi-square approximation is poor).
    """
    n = counts.sum()
    stat = np.sum((counts - n * p) ** 2 / (n * p))
    assert stat < chi2.isf(1e-6, len(p) - 1)


def _prob_aligned(sampler, state, samples, rtol: float = 1e-5) -> None:
    """
    psi carried through the sweep, recorded by the sampler, and the sampling
    probability |psi|^n computed from it at the end of the sweep must belong to the
    returned spins. psi is not returned for states with local updates, and r is
    `Samples.update_r` of psi and prob whenever psi is returned. ``rtol`` allows for
    the rounding errors accumulated by many low-rank updates of states with local
    updates.
    """
    assert samples.state_internal is None
    psi_fresh = np.asarray(state(samples.spins))
    assert np.asarray(sampler._psi).shape == (sampler.nsamples,)
    psi = np.asarray(sampler._psi)
    np.testing.assert_allclose(psi, psi_fresh, rtol=rtol)
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
    np.testing.assert_allclose(prob, prob_fresh, rtol=rtol)


def _count_forward_calls(state) -> list[int]:
    """Spy on ``state.fast_forward`` and record the batch size of every call."""
    batches = []
    fast_forward = state.fast_forward

    def spy(s):
        batches.append(s.shape[0])
        return fast_forward(s)

    state.fast_forward = spy
    return batches


# --- sweep length: every chain takes exactly nsweeps steps, and proposals that leave
#     the spins unchanged are counted as steps without evaluating the wavefunction ---


def test_localflip_takes_one_step_per_round():
    # LocalFlip always changes the spins (_n_candidates == 1), so every step is one
    # batched evaluation round after the one for the initial spins, in the
    # thermalization (by default 20 steps per mode), in sweep() of `sweep_steps` and
    # in reset(nsweeps) alike. With reweight 0 the chains sample |psi|^0 uniformly,
    # accepting every flip also from and onto psi = 0: after sweep(1) every chain
    # differs from its start in exactly one site. prob = |psi|^0 = 1 also for
    # psi = 0, so prob stays valid for reweighting the samples towards another wave
    # function, e.g. after sweep(0). Each of the n >= 64 thermalized chains sits on
    # psi = 0 with probability 1/2, so none does with probability < 1e-19.
    state, configs, _ = skewed_dense_state(3)
    state = DenseState(jnp.asarray(np.where(configs[:, 0] > 0, 1.0, 0.0)), state.symm)
    batches = _count_forward_calls(state)
    n = _stat_ns(64)
    sampler = LocalFlip(state, n, reweight=0.0, sweep_steps=5)
    assert batches == [n] * (20 * 3 + 1)
    batches.clear()
    sampler.sweep()
    assert batches == [n] * 6
    batches.clear()
    s0 = np.asarray(sampler._spins)
    samples = sampler.sweep(1)
    assert batches == [n] * 2
    assert np.all(np.sum(np.asarray(samples.spins) != s0, axis=1) == 1)
    assert np.any(np.asarray(samples.psi) == 0)
    np.testing.assert_array_equal(np.asarray(samples.prob), 1.0)
    batches.clear()
    sampler.reset(nsweeps=2)
    assert batches == [n] * 3


def test_spinexchange_free_steps_save_evaluations():
    # SpinExchange proposes unchanged spins whenever the chosen bond is parallel. Those
    # proposals are counted as free steps, so a sweep of nsweeps steps needs fewer
    # than nsweeps batched evaluations (plus one for the initial spins), each on the
    # full batch. With a flat state every changed proposal is accepted: after sweep(1)
    # a chain differs from its start in 0 (free step) or 2 sites, with a single
    # evaluation round, and both occur among the chains: from a random start a bond
    # is antiparallel with probability 0.6, so all n >= 64 chains take the same kind
    # of step with probability < 1e-14. sweep(0) returns the current spins without
    # evaluating.
    Chain(6, boundary=1, Nparticles=(3, 3))
    state = _sector_dense_state(lambda c: np.ones(len(c)))
    n = _stat_ns(64)
    sampler = SpinExchange(state, n, thermal_steps=0)
    batches = _count_forward_calls(state)
    s0 = np.asarray(sampler._spins)
    s1 = np.asarray(sampler.sweep(1).spins)
    assert batches == [n] * 2
    assert set(np.sum(s1 != s0, axis=1).tolist()) == {0, 2}

    batches.clear()
    s2 = np.asarray(sampler.sweep(0).spins)
    assert batches == []
    assert np.array_equal(s2, s1)

    nsweeps = 40
    sampler.sweep(nsweeps)
    assert all(b == n for b in batches)
    assert 1 <= len(batches) - 1 < nsweeps


def test_free_steps_sample_target_distribution():
    # Regression for the sampling bias of evaluating only changed proposals: the
    # probability of an unchanged proposal depends on the configuration (0 for
    # the Neel states, 1/2 for the domain states of this ring), so a chain that
    # stopped right after a changed proposal would sample |psi|^2 (1 - u) instead
    # of |psi|^2, off by ~0.1 here. Counting the free steps and stopping every chain
    # at exactly nsweeps steps must reproduce the exact distribution.
    Chain(4, boundary=1, Nparticles=(2, 2))
    state = _sector_dense_state(
        lambda c: 1.0 + 0.8 * (c[:, 0] > 0) + 0.4 * (c[:, 1] > 0) + 0.2 * (c[:, 2] > 0)
    )
    configs, p_exact = _exact_probs(state)
    sampler = SpinExchange(state, STAT_NS, thermal_steps=20)
    counts, _ = _histogram(sampler, configs)
    _assert_follows(counts, p_exact)


# --- device sweep: Variational states run the whole sweep in one jitted call ---


def test_variational_sweep_runs_on_device(caplog):
    # Neither the host loop nor state.fast_forward is used; the sweep calls the
    # jittable forward of the state with the model passed explicitly, so an optimizer
    # update of the parameters doesn't recompile the whole sweep.
    Chain(4, boundary=1)
    state = _exp_linear_state()
    sampler = LocalFlip(state, NS, thermal_steps=0)
    called = []
    sampler._sweep_host = lambda *args: called.append("host")
    state.fast_forward = lambda s: called.append("fast_forward")
    jax.config.update("jax_log_compiles", True)
    try:
        with caplog.at_level(logging.INFO, logger="jax"):
            sampler.sweep(20)
            n_first = sum("Compiling" in r.getMessage() for r in caplog.records)
            state.update(0.01 * jnp.ones(state.nparams, jnp.float32))
            caplog.clear()
            samples = sampler.sweep(20)
            n_second = sum("Compiling" in r.getMessage() for r in caplog.records)
    finally:
        jax.config.update("jax_log_compiles", False)
    assert called == []
    assert n_first >= 1  # positive control: the first sweep compiles the device loop
    assert n_second == 0
    _prob_aligned(sampler, state, samples)

    # With zero parameters psi is flat and every flip is accepted, so the distance of
    # a LocalFlip chain from its start changes parity at every step: it tells whether
    # each chain took exactly nsweeps steps in the device loop.
    state.update(state.get_params_flatten())
    spins = [np.asarray(sampler._spins)]
    for nsweeps in (1, 2, 3):
        spins.append(np.asarray(sampler.sweep(nsweeps).spins))
        dist = np.sum(spins[-1] != spins[-2], axis=1)
        assert np.all(dist % 2 == nsweeps % 2) and np.all(dist <= nsweeps)


def test_variational_sweep_under_x64(x64):
    # Under x64, argmax and Python scalars promote to int64; the step counters must
    # keep a single dtype in the carry of the device loop. psi and prob = |psi|^n keep
    # the single precision of the model.
    Chain(4, boundary=1, Nparticles=(2, 2))
    state = _exp_linear_state()
    sampler = SpinExchange(state, NS, reweight=1.0, thermal_steps=0)
    samples = sampler.sweep(10)
    assert samples.psi.dtype == jnp.float32
    assert samples.prob.dtype == jnp.float32
    assert np.all(np.asarray(samples.spins).sum(axis=1) == 0)


# --- chunked sweep: max_parallel < ns/dev sweeps the Markov chains chunk by chunk
#     through chunk_map(_partial_sweep), for the direct path here and the ref one in
#     test_mixsampler_chunked_ref_sweep_samples_target_distribution ---


def test_chunked_direct_sweep_samples_target_distribution():
    # A non-RefModel takes the direct path. With forward_chunk < ns/dev the sweep is
    # routed through chunk_map(_partial_sweep): the walkers per device are padded to
    # a multiple of the chunk and swept chunk by chunk. The padded walkers are
    # truncated away, the outputs stay aligned, and sweeping chunk by chunk must not
    # bias the stationary distribution. Every chunk draws its own random numbers, so
    # the chains of different chunks, all started from the same spins, aren't copies
    # of each other (which the histogram alone barely detects). The sampler is a
    # MixSampler, which goes through the same path drawing the component per proposal
    # inside every chunk.
    Chain(4, boundary=1)
    chunk = 900
    state = _exp_linear_state(max_parallel=chunk)
    s1 = LocalFlip(state, _stat_ns(8000), thermal_steps=0)
    s2 = LocalFlip(state, _stat_ns(8000), thermal_steps=0)
    mix = MixSampler([s1, s2], initial_spins=_fixed_initial_spins(), thermal_steps=20)
    ns_dev = mix.nsamples // NDEV
    assert not mix.use_ref and chunk < ns_dev and ns_dev % chunk != 0

    outputs = []
    partial_sweep = mix._partial_sweep

    def spy(nsweeps, spins):
        out = partial_sweep(nsweeps, spins)
        outputs.append(np.asarray(out.spins))
        return out

    mix._partial_sweep = spy
    configs, p_exact = _exact_probs(state)
    counts, samples = _histogram(mix, configs)
    assert [len(s) for s in outputs] == [chunk * NDEV] * (ns_dev // chunk + 1)
    assert not np.array_equal(outputs[0], outputs[1])
    spins = np.asarray(samples.spins)
    assert spins.shape == (mix.nsamples, get_sites().Nmodes)
    assert set(np.unique(spins).tolist()).issubset({-1, 1})
    _prob_aligned(mix, state, samples)
    _assert_follows(counts, p_exact)


class _ProductRefState(State):
    """
    The product state psi(s) = prod_i a_i^{(1 + s_i) / 2} with local updates, which
    isn't `Variational` and is therefore swept by the host loop. psi itself is the
    internal value, updated from the flipped sites only. The batch sizes of
    ``ref_forward`` are recorded.
    """

    required_update_modes = ("nflips",)
    use_ref = True
    forward_chunk = None

    def __init__(self, a, ref_chunk: int):
        super().__init__()
        self._a = jnp.asarray(a, jnp.float32)
        self.ref_chunk = ref_chunk
        self.batches = []

    def __call__(self, s):
        return jnp.prod(jnp.where(jnp.asarray(s) > 0, self._a, 1.0), axis=1)

    def init_internal(self, s):
        psi = self(s)
        return psi, psi + 0  # distinct buffers, both are donated by _update

    def ref_forward(self, s, s_old, update_mode, internal, return_update=False):
        assert update_mode == {"nflips": 1} and return_update
        self.batches.append(s.shape[0])
        ratio = jnp.where(s != s_old, jnp.where(s > 0, self._a, 1 / self._a), 1.0)
        psi = internal * jnp.prod(ratio, axis=1)
        return psi, psi + 0


def test_host_sweep_of_ref_state_carries_internal():
    # A state with local updates that isn't `Variational` is swept by the host loop
    # through init_internal and ref_forward, with the update mode of the sampler. The
    # internal values must follow the accepted proposals, so that psi stays that of
    # the spins, and the chains are chunked by ref_chunk, not forward_chunk.
    Chain(3, boundary=1)
    state = _ProductRefState([0.5, 2.0, 3.0], ref_chunk=8)
    sampler = LocalFlip(state, NS, thermal_steps=0)
    assert sampler.use_ref and NS // NDEV == 16
    samples = sampler.sweep(10)
    # 2 chunks per device, of 10 rounds each
    assert state.batches == [8 * NDEV] * 20
    _prob_aligned(sampler, state, samples)


# --- psi dropped from the samples: low precision with a batch size that differs from
#     the forward pass, and sweep(0). The samples are reweighted towards the psi
#     filled in by Operator.Oloc ---


def test_samples_without_psi_are_reweighted_towards_full_forward(gpu_precision):
    # The sweep psi is distorted by f(s) from psi of `state.__call__`, mimicking the
    # rounding error of a different batch size. In low precision, the sweep psi is
    # not returned, and the samples drawn from |psi f|^n must be reweighted towards
    # |psi|^2 once psi is filled in by `Operator.Oloc` (here by the cheap forward pass
    # of a diagonal operator). sweep(0) returns the current configurations with the
    # prob recorded by the last sweep (None before the first sweep), but no psi, as
    # the state may have changed since.
    reweight = 1.0
    state, _, _ = skewed_dense_state(3)
    op = sigma_z(0) @ sigma_z(1)
    distortion = lambda s: jnp.where(s[:, 1] > 0, 2.0, 1.0)
    fast_forward = state.fast_forward
    state.fast_forward = lambda s: fast_forward(s) * distortion(s)
    # psi is dropped only if the forward chunk exceeds the samples per device
    state.forward_chunk = 128
    # the precision is checked when the sampler is constructed
    with jax.default_matmul_precision("tensorfloat32"):
        sampler = LocalFlip(state, NS, reweight=reweight, thermal_steps=0)
        samples0 = sampler.sweep(0)
        assert samples0.prob is None and samples0.psi is None
        assert sampler._psi is None
        samples = sampler.sweep()
    assert samples.psi is None and samples.r is None
    psi = np.abs(np.asarray(state(samples.spins)))
    f = np.asarray(distortion(samples.spins))
    # prob is the probability the chains actually followed, from the sweep psi,
    # which is still recorded by the sampler
    np.testing.assert_allclose(
        np.asarray(samples.prob), (psi * f) ** reweight, rtol=1e-6
    )
    np.testing.assert_allclose(np.abs(np.asarray(sampler._psi)), psi * f, rtol=1e-6)
    op.Oloc(state, samples)
    np.testing.assert_allclose(np.abs(np.asarray(samples.psi)), psi, rtol=1e-6)
    w = psi**2 / (psi * f) ** reweight
    np.testing.assert_allclose(np.asarray(samples.r), w / w.mean(), rtol=1e-5)

    samples0 = sampler.sweep(0)
    assert samples0.psi is None and samples0.r is None
    np.testing.assert_array_equal(np.asarray(samples0.spins), np.asarray(samples.spins))
    np.testing.assert_array_equal(np.asarray(samples0.prob), np.asarray(samples.prob))
    op.Oloc(state, samples0)
    np.testing.assert_allclose(np.asarray(samples0.r), np.asarray(samples.r), rtol=1e-6)


def test_low_precision_drops_sweep_psi_only_for_smaller_batches(monkeypatch):
    # In low precision, the sweep psi is dropped only if the chains are evaluated in
    # a batch smaller than the forward chunk, with 16 samples per device here,
    # silently as it costs little. Without a chunk, the batch size isn't fixed anyway
    # (warned by `Variational` instead). In full precision, the sweep psi is always
    # returned. The choice is made when the sampler is constructed, by
    # `is_low_precision` of the state dtype, which is replaced here because matmul
    # precision contexts would recompile every operation. The sweep is replaced by a
    # stub returning the psi of the spins, which is also called chunk by chunk for
    # forward_chunk 8. The real sweep of a dropped psi in emulated TF32 is tested
    # above.
    Chain(3, boundary=1)
    state = DenseState(jnp.arange(1.0, 9.0))
    assert NS // NDEV == 16
    cases = [  # (low precision, forward_chunk, psi dropped)
        (True, None, False),
        (True, 8, False),
        (True, 16, False),
        (True, 32, True),
        (False, 32, False),
    ]
    for low, chunk, dropped in cases:
        state.forward_chunk = chunk
        dtypes = []
        is_low_precision = lambda dtype: dtypes.append(dtype) or low
        monkeypatch.setattr(metropolis, "is_low_precision", is_low_precision)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sampler = LocalFlip(state, NS, thermal_steps=0)
            sampler._partial_sweep = lambda nsweeps, spins: SamplesTree(
                spins, jnp.asarray(state(spins))
            )
            samples = sampler.sweep(1)
        assert dtypes in ([], [state.dtype])
        assert (samples.psi is None) == dropped
        assert (samples.r is None) == dropped
        psi = np.asarray(state(samples.spins))
        np.testing.assert_allclose(np.asarray(sampler._psi), psi)
        np.testing.assert_allclose(np.asarray(samples.prob), psi**2, rtol=1e-5)


# --- prob: the sampling probability is computed from psi at the end of the sweep ---


def test_prob_is_float32_logarray_of_psi():
    # prob = |psi|^n is computed as a LogArray from psi promoted to at least float32,
    # so it stays accurate where the dense |psi|^2 of a plain float32 psi ~ 1e25 or
    # 1e-25 overflows or underflows float32 (which used to zero r silently), and
    # where psi is given below single precision: |psi| > 256 overflows |psi|^2 in
    # float16, whose 11-bit mantissa would also round it. A LogArray psi of low
    # precision also gives a float32 logabs. A complex psi keeps its phase until the
    # modulus is taken. The chains carry psi in its own dtype.
    Chain(3, boundary=1)
    state = DenseState(jnp.asarray([101, -299, 153, 407, -251, 303, 127, 353.0]))
    fast_forward = state.fast_forward
    sampler = LocalFlip(state, NS, thermal_steps=0)
    cases = [  # (dtype, scale, LogArray)
        (jnp.float32, 1e25, False),
        (jnp.float32, 1e-25, False),
        (jnp.float16, 1.0, False),
        (jnp.float16, 1.0, True),
        (jnp.complex64, 1.0, False),
    ]
    for dtype, scale, log in cases:

        def forward(s):
            psi = np.asarray(fast_forward(s)) * np.float32(scale)
            if jnp.issubdtype(dtype, jnp.complexfloating):
                # phases e^{i pi k / 3} of odd k, both real and complex
                psi = psi * np.exp(1j * np.pi / 3 * (np.asarray(s) @ [1, 2, 4]))
            psi = jnp.asarray(psi, dtype)
            return LogArray(jnp.sign(psi), jnp.log(jnp.abs(psi))) if log else psi

        state.fast_forward = forward
        samples = sampler.sweep(4)
        assert samples.psi.dtype == dtype
        psi = forward(samples.spins)
        if log:
            logabs = np.asarray(psi.logabs, np.float64)
        else:
            logabs = np.log(np.abs(np.asarray(psi)).astype(np.float64))
        assert isinstance(samples.prob, LogArray)
        assert samples.prob.logabs.dtype == jnp.float32
        np.testing.assert_allclose(samples.prob.logabs, 2 * logabs, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(samples.r), 1.0, rtol=1e-5)


# --- accept rule (_update) ---


def _update_sampler(reweight: float = 2.0) -> LocalFlip:
    state = DenseState(jnp.ones(2 ** get_sites().Nsites))
    return LocalFlip(state, NS, reweight=reweight, thermal_steps=0)


def _accepted_mask(
    sampler, old_psi, new_psi, propose_ratio=None, seed=0
) -> tuple[np.ndarray, SamplesTree]:
    """
    Run ``Metropolis._update`` with crafted wave functions (proposed spins always
    differ from the old ones) and return the per-walker accept mask and the output.
    """
    ns = old_psi.shape[0]
    old_spins = jnp.tile(_fixed_initial_spins(), (ns, 1))
    new_spins = np.asarray(-old_spins)  # the inputs of _update are donated
    _, out = sampler._update(
        jax.random.key(seed),
        propose_ratio,
        SamplesTree(old_spins, old_psi),
        SamplesTree(-old_spins, new_psi),
    )
    return (np.asarray(out.spins) == new_spins).all(axis=1), out


# The special-value table of `Metropolis._update`, TABLE[old][new] = accepted.
# A nan psi is never adopted, and a chain stuck at a nan psi always escapes unless
# the new psi is also nan. Equal special values (0 -> 0, inf -> inf) are accepted;
# otherwise the special values are ordered 0 < finite < inf.
_SPECIAL_TABLE = {
    "0": {"0": True, "1": True, "nan": False, "inf": True},
    "1": {"0": False, "1": True, "nan": False, "inf": True},
    "nan": {"0": True, "1": True, "nan": False, "inf": True},
    "inf": {"0": False, "1": False, "nan": False, "inf": True},
}

# Encodings of the special values of psi: plain floats of both signs, and
# (sign, logabs) of LogArray, with the zero of from_value (sign 0) and of abs or
# negation (sign +-1), a NaN in either component, and a finite value whose dense
# form overflows float32, so nan-ness must be read from the components.
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


def test_accept_follows_special_value_table():
    # Every entry of the table for every pair of encodings, with and without
    # proposal ratios (0.5 and 2). Finite -> finite is decided by R, so 1 -> 1 is only
    # taken between equal values and without a proposal ratio (R = 1).
    Chain(3, boundary=1)
    sampler = _update_sampler()
    for kind, enc in _SPECIAL_ENCODINGS.items():

        def make(values):
            x = np.asarray(values, np.float32)
            x = np.tile(x, (NDEV,) + (1,) * (x.ndim - 1))
            if kind == "plain":
                return jnp.asarray(x)
            return LogArray(jnp.asarray(x[:, 0]), jnp.asarray(x[:, 1]))

        for with_ratio in (False, True):
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
            ratio = None
            if with_ratio:
                ratio = np.resize(np.float32([0.5, 2.0]), len(expect))
                ratio = jnp.asarray(np.tile(ratio, NDEV))
            accepted, _ = _accepted_mask(sampler, make(old), make(new), ratio)
            assert np.array_equal(accepted, np.tile(expect, NDEV)), (kind, with_ratio)


def test_update_accepts_in_float64_with_32bit_random_bits():
    # The acceptance test runs in float64 inside the jitted update, with the uniform
    # random number assembled from 32-bit draws (64-bit random bits can't be
    # generated inside jit with x64 disabled). The returned psi keeps float32.
    Chain(3, boundary=1)
    sampler = _update_sampler()
    ns = 4 * NDEV
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


@pytest.mark.parametrize("dtype, reweight", [(jnp.float32, 2.0), (jnp.complex64, 1.0)])
def test_update_accept_frequency_matches_rate(dtype, reweight):
    # The float64 test accepts with probability R = |psi'/psi|^n, independent of the
    # sign or phase of psi; checked against binomial noise for a rate below and
    # above 1/2.
    Chain(3, boundary=1)
    sampler = _update_sampler(reweight)
    n = 20_000 * NDEV
    rates = np.repeat([0.3, 0.8], n)
    phase = 1j if jnp.issubdtype(dtype, jnp.complexfloating) else -1.0
    new = jnp.asarray(phase * rates ** (1 / reweight), dtype)
    old = jnp.ones(2 * n, dtype)
    accepted, _ = _accepted_mask(sampler, old, new, seed=3)
    for i, rate in enumerate((0.3, 0.8)):
        sigma = np.sqrt(rate * (1 - rate) / n)
        assert abs(accepted[i * n : (i + 1) * n].mean() - rate) < 5 * sigma


def test_update_keeps_psi_of_accepted_configurations():
    # After every acceptance step, psi belongs to the configuration kept by the
    # chain, whether the proposal was accepted or rejected. A chain rejects only a
    # flip of s[0] = +1 with u > 1/2, with probability 1/12, so 256 chains exercise
    # both branches for any device count (all accepted with probability 2e-10).
    state, _, _ = skewed_dense_state(3)
    sampler = LocalFlip(state, _stat_ns(256), reweight=1.0, thermal_steps=0)
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


def test_update_rejects_double_occupancy():
    # Without double occupancy, a proposal putting a spin-up and a spin-down fermion
    # on the same site is rejected whatever its wave function, while an allowed one
    # with the same wave function is accepted. A sampler for spins only is rejected
    # for fermions.
    Chain(3, particle_type="spinful_fermion", Nparticles=(1, 1), double_occ=False)
    state = _sector_dense_state(lambda c: np.ones(len(c)))
    with pytest.raises(ValueError, match="not supported"):
        LocalFlip(state, NS)
    sampler = ParticleHop(state, NS, thermal_steps=0)
    # modes: spin-up on sites 0-2, then spin-down on sites 0-2
    old = [1, -1, -1, -1, 1, -1]  # up on 0, down on 1
    double = [1, -1, -1, 1, -1, -1]  # up and down on 0
    allowed = [1, -1, -1, -1, -1, 1]  # up on 0, down on 2
    old_spins = jnp.asarray(np.tile([old, old], (NDEV, 1)), jnp.int8)
    new_spins = np.tile([double, allowed], (NDEV, 1)).astype(np.int8)
    _, out = sampler._update(
        jax.random.key(0),
        None,
        SamplesTree(old_spins, jnp.ones(2 * NDEV, jnp.float32)),
        SamplesTree(jnp.asarray(new_spins), jnp.full(2 * NDEV, 10.0, jnp.float32)),
    )
    accepted = np.all(np.asarray(out.spins) == new_spins, axis=1)
    assert np.array_equal(accepted, np.tile([False, True], NDEV))


# --- MixSampler ---


def test_mixsampler_requires_common_state_and_reweight():
    # The components must share the state and the reweight factor, which the mixture
    # inherits, and the mixture draws the sum of their samples. Its first run starts
    # from the spins of the components; a later reset draws random spins instead
    # (equal to the previous ones by chance with probability 8^-(12 NDEV)).
    Chain(3, boundary=1)
    state = DenseState(jnp.ones(8))
    s1 = LocalFlip(state, 4 * NDEV, reweight=1.5, thermal_steps=0)
    other = DenseState(jnp.ones(8))
    other_state = LocalFlip(other, 4 * NDEV, reweight=1.5, thermal_steps=0)
    with pytest.raises(ValueError, match="states"):
        MixSampler([s1, other_state])
    other_reweight = LocalFlip(state, 4 * NDEV, reweight=1.0, thermal_steps=0)
    with pytest.raises(ValueError, match="reweight"):
        MixSampler([s1, other_reweight])
    s2 = LocalFlip(state, 8 * NDEV, reweight=1.5, thermal_steps=0)
    spins = np.concatenate([np.asarray(s._spins) for s in (s1, s2)])
    mix = MixSampler([s1, s2], thermal_steps=0)
    assert float(mix.reweight) == 1.5
    assert mix.nsamples == 12 * NDEV
    mix_spins = np.asarray(mix._spins)
    mix_codes = config_codes(mix_spins)
    np.testing.assert_array_equal(np.sort(mix_codes), np.sort(config_codes(spins)))
    mix.reset(nsweeps=0)
    assert not np.array_equal(np.asarray(mix._spins), mix_spins)


def test_mixsampler_merges_update_modes():
    # Proposals of one batch come from different components, so the flip numbers
    # passed to the state must be upper bounds over the components, and any other
    # mode that differs among them is dropped to None. The merge only affects the
    # fast local updates of the state, so it is warned about only for a state that
    # uses them. A sampler lacking the update modes required by such a state falls
    # back to the plain forward pass with a warning.
    Chain(4, particle_type="spinful_fermion", Nparticles=(2, 2))
    ref_state = Variational(SingletPair())
    dense_state = _sector_dense_state(lambda c: np.ones(len(c)))
    assert ref_state.use_ref and not dense_state.use_ref

    class _Tagged(ParticleHopUp):
        def __init__(self, tag, *args, **kwargs):
            self._tag = tag
            super().__init__(*args, **kwargs)

        @property
        def update_mode(self):
            return {**super().update_mode, "tag": self._tag}

    def hops(state):
        up = ParticleHopUp(state, 4 * NDEV, thermal_steps=0)
        return [up, ParticleHopDn(state, 8 * NDEV, thermal_steps=0)]

    def tagged(state):
        return [_Tagged(tag, state, 4 * NDEV, thermal_steps=0) for tag in "ab"]

    merged = {"nflips": 2, "nflips_up": 2, "nflips_dn": 2}
    dropped = {"nflips": 2, "nflips_up": 2, "nflips_dn": 0, "tag": None}
    with pytest.warns(UserWarning, match="merged into its maximum"):
        mix = MixSampler(hops(ref_state), thermal_steps=0)
    assert mix.update_mode == merged and mix.use_ref
    with pytest.warns(UserWarning, match="update mode 'tag' differs"):
        mix = MixSampler(tagged(ref_state), thermal_steps=0)
    assert mix.update_mode == dropped
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mix = MixSampler(hops(dense_state), thermal_steps=0)
        assert mix.update_mode == merged
        assert MixSampler(tagged(dense_state), thermal_steps=0).update_mode == dropped

    with pytest.warns(UserWarning, match="not all provided"):
        hop = ParticleHop(ref_state, 4 * NDEV, thermal_steps=0)
    assert not hop.use_ref


def test_mixsampler_draws_components_by_nsamples_with_their_ratios():
    # Every proposal comes from a component chosen with probability proportional to
    # its nsamples, together with the proposal ratio of that component, or 1 for a
    # component that returns none. Up and down fermions sit on the same two sites,
    # so that the two components change the spins equally often: 1/3 of the changed
    # proposals move up fermions. The up hops come with the proposal ratio 3 (not a
    # valid one, just a label).
    Chain(4, particle_type="spinful_fermion", Nparticles=(2, 2))
    state = _sector_dense_state(lambda c: np.ones(len(c)))

    class _RatioHopUp(ParticleHopUp):
        def propose(self, key, old_spins):
            ratio = jnp.full(old_spins.shape[0], 3.0, jnp.float32)
            return super().propose(key, old_spins), ratio

    n = 6000 * NDEV
    spins = jnp.asarray(np.tile([1, 1, -1, -1] * 2, (n, 1)), jnp.int8)
    key = jax.random.key(0)
    dn_hop = ParticleHopDn(state, 8 * NDEV, thermal_steps=0)
    # no ratio at all without a component providing one (only traced here)
    up_hop = ParticleHopUp(state, 4 * NDEV, thermal_steps=0)
    mix = MixSampler([up_hop, dn_hop], thermal_steps=0)
    assert jax.eval_shape(mix._propose_spins_and_ratio, key, spins)[1] is None
    up_hop = _RatioHopUp(state, 4 * NDEV, thermal_steps=0)
    mix = MixSampler([up_hop, dn_hop], thermal_steps=0)
    new_spins, ratio = mix._propose_spins_and_ratio(key, spins)
    changed = np.asarray(new_spins) != np.asarray(spins)
    up, dn = changed[:, :4].any(axis=1), changed[:, 4:].any(axis=1)
    assert not np.any(up & dn)
    ratio = np.asarray(ratio)
    assert np.all(ratio[up] == 3.0) and np.all(ratio[dn] == 1.0)
    nchanged = up.sum() + dn.sum()
    # 5 sigma of the binomial noise
    assert abs(up.sum() / nchanged - 1 / 3) < 5 * np.sqrt(2 / 9 / nchanged)


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


def test_mixsampler_chunked_ref_sweep_samples_target_distribution():
    # Mixed ParticleHopUp/Dn proposals evaluated in one batch through the low-rank
    # updates of SingletPair (merged update modes, free steps for hops onto occupied
    # sites) must reproduce the exact |psi|^n over all configurations. With `use_ref`
    # the sweep is chunked by `ref_chunk`, and the sampler returns no psi, which
    # `Operator.Oloc` (here of a single hopping term, cheap to compile) fills in by
    # `init_internal` chunk by chunk, including a zero-padded chunk. Regression: r
    # must be normalized over all samples, not per chunk, which biased the mean by
    # the zero-padding ratio for reweight != 2.
    Chain(4, particle_type="spinful_fermion", Nparticles=(2, 2))
    # A smooth pairing matrix: the default paired Fermi sea is peaked on 6 of the
    # 36 configurations and its Metropolis chain relaxes only in ~2000 steps,
    # whereas this one relaxes in ~15 steps (spectral gap 0.065).
    N = get_sites().Nsites
    F = 1 + 0.5 * np.cos(np.subtract.outer(np.arange(N), np.arange(N)) * np.pi / 2)
    F += 0.2 * np.random.default_rng(0).standard_normal((N, N))
    chunk = 40
    state = Variational(SingletPair(F=jnp.asarray(F, jnp.float32)), max_parallel=chunk)
    reweight = 1.0
    configs, p_exact = _exact_probs(state, reweight)

    # Few chains keep the device loop cheap, with the samples of 20 sweeps of 16 steps,
    # which are nearly independent: over 10 seeds the chi-square statistic averages
    # 37 for 35 degrees of freedom (at most 54), and stopping a chain right after a
    # changed proposal gives 260-330, well above the threshold 90.
    up = ParticleHopUp(state, _stat_ns(192), reweight=reweight, thermal_steps=0)
    dn = ParticleHopDn(state, _stat_ns(192), reweight=reweight, thermal_steps=0)
    with pytest.warns(UserWarning, match="merged into its maximum"):
        mix = MixSampler([up, dn], thermal_steps=100)
    ns_dev = mix.nsamples // NDEV
    assert mix.use_ref and chunk < ns_dev and ns_dev % chunk != 0
    counts, samples = _histogram(mix, configs, nsweeps=20)
    _assert_follows(counts, p_exact)
    assert isinstance(samples.prob, LogArray)
    assert samples.prob.logabs.dtype == jnp.float32
    # the 420 low-rank updates of every chain accumulate relative errors up to 1e-5
    _prob_aligned(mix, state, samples, rtol=1e-4)

    (create_u(0) @ annihilate_u(1)).Oloc(state, samples)
    psi = np.asarray(jnp.asarray(state(samples.spins)))
    np.testing.assert_allclose(np.asarray(jnp.asarray(samples.psi)), psi, rtol=1e-5)
    # r reweights from prob, the distribution the chains followed
    w = np.abs(psi) ** 2 / np.asarray(samples.prob)
    np.testing.assert_allclose(np.asarray(samples.r), w / w.mean(), rtol=1e-5)


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


def test_propose_ratio_carried_through_free_steps():
    # psi is smaller for s[0] == +1, so that the proposal ratio 3 decides whether a
    # flip of site 0 to +1 is accepted (R = 3/4), which is often proposed after "do
    # nothing" free steps: the ratio of another candidate biases the samples.
    state, _, _ = skewed_dense_state(3, up=0.5)
    configs, p_exact = _exact_probs(state)
    sampler = _BiasedFlip(state, STAT_NS, thermal_steps=20)
    counts, _ = _histogram(sampler, configs)
    _assert_follows(counts, p_exact)


def test_mixsampler_unifies_propose_ratios():
    # A mixture of a proposer with ratios and one without (LocalFlip, which gets
    # ratio 1 inside the mixture, as checked directly in
    # test_mixsampler_draws_components_by_nsamples_with_their_ratios). The chains
    # sample |psi|^1, and the reweighting factor must recover <s0>_{|psi|^2} = -0.6
    # from samples whose own <s0> is -1/3.
    state, configs, psi = skewed_dense_state(3, up=0.5)
    _, p_exact = _exact_probs(state, reweight=1.0)
    biased = _BiasedFlip(state, _stat_ns(8000), reweight=1.0, thermal_steps=0)
    plain = LocalFlip(state, _stat_ns(8000), reweight=1.0, thermal_steps=0)
    mix = MixSampler([biased, plain], thermal_steps=20)
    counts, samples = _histogram(mix, configs)
    _assert_follows(counts, p_exact)

    p2 = psi**2 / np.sum(psi**2)
    exact_s0 = float(np.sum(p2 * configs[:, 0]))
    r = np.asarray(samples.r)
    est = float(np.mean(r * np.asarray(samples.spins)[:, 0]))
    # r * s0 has variance 0.72: sigma = 0.0067 for 16000 samples
    assert abs(est - exact_s0) < 0.035
