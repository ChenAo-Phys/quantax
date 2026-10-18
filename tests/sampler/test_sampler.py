import itertools
import numpy as np
import jax
import jax.numpy as jnp
import pytest
import quantax as qtx
from quantax.sites import Chain
from quantax.state import DenseState
from quantax.symmetry import Identity, Translation
from quantax.sampler import Sampler, ExactSampler, RandomSampler
from quantax.utils import ints_to_array, LogArray
from _stats import skewed_dense_state, config_index, histogram

NDEV = jax.device_count()
# All sample counts are multiples of NDEV so that `nsamples % device_count == 0`
# holds for any device layout (CPU CI uses 4, the workstation uses GPUs).
# ExactSampler runs op by op, so every new shape (nsamples, support size) costs a
# few seconds of small compilations: most tests share the same shapes.
STAT_NS = 10000 * NDEV


def _assert_frequencies(spins, configs: np.ndarray, p: np.ndarray) -> None:
    """The histogram of independent samples follows p within 5 sigma per entry."""
    n = len(spins)
    freq = histogram(spins, configs) / n
    assert np.all(np.abs(freq - p) <= 5 * np.sqrt(p * (1 - p) / n))


def _assert_reweighted(spins, r, configs: np.ndarray, p: np.ndarray) -> None:
    """The r-weighted histogram of the samples follows p within 5 sigma per entry."""
    r = np.asarray(r, dtype=np.float64)
    weights = np.eye(len(configs))[config_index(spins, configs)] * r[:, None]
    est = weights.mean(axis=0)
    sigma = weights.std(axis=0) / np.sqrt(len(r))
    assert np.all(np.abs(est - p) <= 5 * sigma + 1e-12)


# --- base Sampler ---


def test_sampler_rejects_indivisible_nsamples():
    # nsamples must split evenly across devices.
    if NDEV == 1:
        pytest.skip("every nsamples is divisible by a single device")
    Chain(2, boundary=1)
    state = DenseState(jnp.ones(4))
    with pytest.raises(ValueError):
        Sampler(state, nsamples=NDEV + 1)


# --- ExactSampler ---


def test_exact_sampler_frequencies_match_born_rule():
    # With reweight = 2 the configurations are drawn directly from |psi|^2.
    # Zero-amplitude configurations are masked out of the support and are never
    # drawn: their tolerance in `_assert_frequencies` is exactly 0.
    state, configs, psi = skewed_dense_state(3)
    psi[[1, 4, 6]] = 0.0
    state = DenseState(jnp.asarray(psi), state.symm)
    samples = ExactSampler(state, STAT_NS, reweight=2.0).sweep()
    p = psi**2 / np.sum(psi**2)
    _assert_frequencies(samples.spins, configs, p)

    # the recorded probability is the Born probability of the normalized state, so
    # the reweighting is trivial
    born = np.abs(np.asarray(samples.psi)) ** 2 / np.sum(psi**2)
    np.testing.assert_allclose(np.asarray(samples.prob), born, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(samples.r), 1.0, rtol=1e-5)


def test_exact_sampler_reweighting_recovers_born_distribution():
    # With reweight = 1 the samples follow |psi|, the recorded probability is |psi|
    # up to a constant (only |psi|^2 is normalized), and the reweighting factor
    # recovers |psi|^2, which differs from |psi| by far more than the tolerance.
    # The sign structure of psi must not enter the probability.
    state, configs, psi = skewed_dense_state(3)
    psi *= np.where(configs[:, 2] > 0, -1.0, 1.0)
    state = DenseState(jnp.asarray(psi), state.symm)
    samples = ExactSampler(state, STAT_NS, reweight=1.0).sweep()
    ratio = np.asarray(samples.prob) / np.abs(np.asarray(samples.psi))
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-5)
    _assert_frequencies(samples.spins, configs, np.abs(psi) / np.sum(np.abs(psi)))
    _assert_reweighted(samples.spins, samples.r, configs, psi**2 / np.sum(psi**2))


@pytest.mark.parametrize("reweight", [2.0, 1.0])
def test_exact_sampler_prob_counts_symmetry_images(reweight):
    # A configuration s is drawn by choosing a representative with probability
    # ∝ |psi|^n and then one of the |G| symmetry images uniformly. Configurations
    # invariant under a subgroup appear several times among the images and are
    # drawn more often, so q_s ∝ |psi(rep)|^n * multiplicity / |G|.
    L = 4
    Chain(L, boundary=1)
    symm = Translation([1])
    symm.basis_make()
    reps = np.asarray(ints_to_array(symm.basis.states))  # (Nrep, L)
    # The amplitude depends on the representative only through the translation
    # invariant number of up spins, so psi(rep(s)) is known from s directly.
    amp = lambda s: 1.0 + np.count_nonzero(np.asarray(s) > 0, axis=-1)
    state = DenseState(jnp.asarray(amp(reps), dtype=jnp.float32), symm)

    # m_s = number of translations leaving s invariant
    mult = lambda s: np.all(
        np.stack([np.roll(s, k, axis=-1) for k in range(L)], axis=-2)
        == np.asarray(s)[..., None, :],
        axis=-1,
    ).sum(axis=-1)
    # exact q over all configurations, normalized
    configs = np.array(list(itertools.product([1, -1], repeat=L)))
    q = amp(configs) ** reweight * mult(configs) / L / np.sum(amp(reps) ** reweight)
    assert {1, 2, 4} <= set(mult(configs).tolist())  # orbits of different sizes
    np.testing.assert_allclose(q.sum(), 1.0)

    samples = ExactSampler(state, STAT_NS, reweight=reweight).sweep()
    spins = np.asarray(samples.spins)
    ratio = np.asarray(samples.prob, np.float64) / q[config_index(spins, configs)]
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-5)
    # the samples follow the recorded probability
    _assert_frequencies(spins, configs, q)

    if reweight == 2.0:
        # Born rule of the symmetric state, so the reweighting factor is trivial
        np.testing.assert_allclose(np.asarray(samples.r), 1.0, rtol=1e-5)


def test_exact_sampler_symm_argument_reweights_to_the_state():
    # A sampler symmetry the state doesn't carry draws the samples from the dense
    # wave function in the symmetric basis, while psi is evaluated by the state
    # itself, so that r = |psi|^2 / prob recovers the distribution of the state.
    L = 4
    Chain(L, boundary=1)
    configs = np.array(list(itertools.product([1, -1], repeat=L)))
    psi_fn = lambda s: 1.0 + (s[:, 0] > 0) + 0.5 * (s[:, 1] > 0)
    symm = Identity()
    symm.basis_make()
    basis_configs = np.asarray(ints_to_array(symm.basis.states))
    state = DenseState(jnp.asarray(psi_fn(basis_configs)), symm)

    samples = ExactSampler(state, STAT_NS, symm=Translation([1])).sweep()
    spins = np.asarray(samples.spins)
    np.testing.assert_allclose(np.asarray(samples.psi), psi_fn(spins), rtol=1e-6)
    # prob is translation invariant, unlike |psi|^2 of the state
    orbit = np.min([config_index(np.roll(spins, k, 1), configs) for k in range(L)], 0)
    prob = np.asarray(samples.prob)
    for o in np.unique(orbit):
        np.testing.assert_allclose(prob[orbit == o], prob[orbit == o][0], rtol=1e-5)
    p = psi_fn(configs) ** 2 / np.sum(psi_fn(configs) ** 2)
    _assert_reweighted(spins, samples.r, configs, p)


def test_exact_sampler_prob_reflects_sampling_forward_pass():
    # prob is evaluated from the dense wave function used for drawing, psi from
    # the state itself. A mismatch between the two, e.g. from the precision of
    # the forward passes, is corrected by r = |psi|^2 / prob.
    state, configs, psi = skewed_dense_state(3)
    distortion = np.where(configs[:, 1] > 0, 2.0, 1.0)
    todense = state.todense
    state.todense = lambda symm=None: DenseState(
        jnp.asarray(psi * distortion), todense(symm).symm
    )

    samples = ExactSampler(state, STAT_NS, reweight=2.0).sweep()
    spins = np.asarray(samples.spins)
    f = np.where(spins[:, 1] > 0, 2.0, 1.0)
    ratio = np.asarray(samples.prob) / (np.abs(np.asarray(samples.psi)) * f) ** 2
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-5)
    _assert_reweighted(spins, samples.r, configs, psi**2 / np.sum(psi**2))


def test_exact_sampler_handles_more_than_32_modes():
    # Regression: with more than 32 modes the basis integers exceed uint32. The
    # sampler used to draw the integers themselves with jr.choice, which casts
    # them to uint32 when x64 is off (the default) and silently truncated the
    # high bits, yielding configurations outside the particle-number sector.
    # Drawing indices into the basis keeps the integers on the host as uint64.
    Chain(20, particle_type="spinful_fermion", Nparticles=(1, 1))  # 40 modes
    symm = Identity()
    symm.basis_make()
    assert int(symm.basis.states.max()) >= 2**32
    state = DenseState(jnp.ones(symm.basis.Ns), symm)
    samples = ExactSampler(state, 64 * NDEV, reweight=2.0).sweep()
    spins = np.asarray(samples.spins)
    assert spins.shape == (64 * NDEV, 40)
    n_up = np.count_nonzero(spins[:, :20] > 0, axis=1)
    n_dn = np.count_nonzero(spins[:, 20:] > 0, axis=1)
    assert np.all(n_up == 1) and np.all(n_dn == 1)


def test_exact_sampler_psi_is_the_unnormalized_state():
    # Regression: `Operator.Oloc` divides psi of the connected configurations,
    # evaluated from the state, by `samples.psi`. Normalizing psi of the samples
    # would therefore scale the off-diagonal local estimators by the norm of the
    # state (~3e3 here), although the sampling probability is normalized.
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    psi = 1e3 * (1.0 + np.random.default_rng(0).random(symm.basis.Ns))
    state = DenseState(jnp.asarray(psi), symm)
    samples = ExactSampler(state, STAT_NS, reweight=2.0).sweep()
    np.testing.assert_allclose(
        np.asarray(samples.psi), np.asarray(state(samples.spins)), rtol=1e-6
    )
    born = np.abs(np.asarray(samples.psi)) ** 2 / np.sum(psi**2)
    np.testing.assert_allclose(np.asarray(samples.prob), born, rtol=1e-5)


def test_exact_sampler_probability_survives_large_amplitudes():
    # `|psi| ** reweight` of an unnormalized state overflows float32, so the
    # sampling probability is computed from the normalized wave function while
    # psi of the samples keeps the log representation of the state.
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    configs = np.asarray(ints_to_array(symm.basis.states))
    # |psi|**2 = exp(100) overflows float32, while the norm itself does not
    logabs = np.full(symm.basis.Ns, 50.0)
    logabs[0] += np.log(2.0)  # one configuration with twice the amplitude
    state = DenseState(LogArray(jnp.ones(symm.basis.Ns), jnp.asarray(logabs)), symm)

    samples = ExactSampler(state, STAT_NS, reweight=2.0).sweep()
    idx = config_index(samples.spins, configs)
    np.testing.assert_allclose(np.asarray(samples.psi.logabs), logabs[idx], atol=1e-5)
    p = np.exp(2 * (logabs - logabs.max()))
    _assert_frequencies(samples.spins, configs, p / p.sum())


# --- RandomSampler ---


def test_random_sampler_draws_uniformly_and_reweights_to_born():
    # The samples are uniform, i.e. reweight 0 with prob = 1 for every sample, and
    # the reweighting factor recovers |psi|^2. For a complex state the probability
    # and the reweighting factor stay real.
    qtx.set_default_dtype(jnp.complex64)  # reset by the conftest
    state, configs, psi = skewed_dense_state(3)
    psi = psi * np.exp(1j * np.pi / 2 * np.arange(len(psi)))
    state = DenseState(jnp.asarray(psi), state.symm)
    sampler = RandomSampler(state, STAT_NS)
    assert sampler.reweight == 0.0
    samples = sampler.sweep()
    assert np.isrealobj(samples.prob) and np.isrealobj(samples.r)
    np.testing.assert_array_equal(np.asarray(samples.prob), 1.0)
    _assert_frequencies(samples.spins, configs, np.full(len(configs), 1 / len(configs)))
    p = np.abs(psi) ** 2 / np.sum(np.abs(psi) ** 2)
    _assert_reweighted(samples.spins, samples.r, configs, p)
