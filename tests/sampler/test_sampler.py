import numpy as np
import jax
import jax.numpy as jnp
import pytest
from quantax.sites import Chain
from quantax.state import DenseState, Variational
from quantax.model import RBM_Dense
from quantax.symmetry import Identity, Translation
from quantax.operator import Ising
from quantax.sampler import Sampler, ExactSampler, RandomSampler
from quantax.utils import ints_to_array, LogArray
from quantax.global_defs import get_sites

NDEV = jax.device_count()
# All sample counts are multiples of NDEV so that `nsamples % device_count == 0`
# holds for any device layout (CPU CI uses 4, the workstation uses GPUs).
STAT_NS = 10000 * NDEV


def _skewed_dense_state(L: int) -> tuple[DenseState, np.ndarray, np.ndarray]:
    """
    A real, positive `DenseState` on a length-``L`` spin chain biased toward
    ``s[0] == +1`` (amplitude 2 vs 1), with no Sz conservation so the full
    ``2**L`` Hilbert space is covered.

    Returns the state together with the basis-ordered configurations and the
    (un-normalized) wave function, so a test can compute exact expectations.
    """
    Chain(L, boundary=1)
    symm = Identity()
    symm.basis_make()
    configs = ints_to_array(symm.basis.states)  # (Ns, L), in basis order
    psi = np.where(configs[:, 0] > 0, 2.0, 1.0).astype(np.float64)
    return DenseState(jnp.asarray(psi), symm), configs, psi


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


def test_exact_sampler_output_shapes():
    # The sampler returns psi, the sampling probability, and the reweighting
    # factor computed from them by `Samples.update_r`.
    Chain(3, boundary=1)
    state = DenseState(jnp.ones(8))
    ns = 4 * NDEV
    samples = ExactSampler(state, ns, reweight=2.0).sweep()
    assert np.asarray(samples.spins).shape == (ns, get_sites().Nmodes)
    assert np.asarray(samples.psi).shape == (ns,)
    assert np.asarray(samples.prob).shape == (ns,)
    assert samples.state_internal is None
    np.testing.assert_allclose(np.asarray(samples.r), 1.0, rtol=1e-5)


def test_exact_sampler_frequencies_match_born_rule():
    # With reweight = 2 the configurations are drawn directly from |psi|^2.
    state, _, psi = _skewed_dense_state(3)
    samples = ExactSampler(state, STAT_NS, reweight=2.0).sweep()
    spins = np.asarray(samples.spins)

    uniq, counts = np.unique(spins, axis=0, return_counts=True)
    freq = counts / counts.sum()
    amp = np.asarray(state(jnp.asarray(uniq)))
    target = np.abs(amp) ** 2 / np.sum(psi**2)
    assert np.allclose(freq, target, atol=0.02)

    # the recorded probability is the Born probability of the normalized state
    born = np.abs(np.asarray(samples.psi)) ** 2 / np.sum(psi**2)
    np.testing.assert_allclose(np.asarray(samples.prob), born, rtol=1e-5)


def test_exact_sampler_prob_follows_reweight():
    # With reweight = n the samples follow |psi|^n, and the recorded probability
    # is |psi|^n up to a constant factor (only |psi|^2 is normalized).
    state, _, psi = _skewed_dense_state(3)
    samples = ExactSampler(state, STAT_NS, reweight=1.0).sweep()
    spins = np.asarray(samples.spins)
    ratio = np.asarray(samples.prob) / np.abs(np.asarray(samples.psi))
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-5)

    uniq, counts = np.unique(spins, axis=0, return_counts=True)
    freq = counts / counts.sum()
    target = np.abs(np.asarray(state(jnp.asarray(uniq)))) / np.sum(psi)
    assert np.allclose(freq, target, atol=0.02)


def test_exact_sampler_reweighting_recovers_observable():
    # Sample from |psi|^1 and let the reweighting factor recover <s0>_{|psi|^2}.
    # The biased state has <s0>_p = 0.6 while the *proposal* mean <s0>_q ≈ 0.33,
    # so a broken reweighting would miss the target by far more than the tolerance.
    state, configs, psi = _skewed_dense_state(3)
    p = psi**2 / np.sum(psi**2)
    exact_s0 = float(np.sum(p * configs[:, 0]))

    samples = ExactSampler(state, STAT_NS, reweight=1.0).sweep()
    spins = np.asarray(samples.spins)
    r = np.asarray(samples.r)
    est = float(np.mean(r * spins[:, 0]))
    assert np.isclose(est, exact_s0, atol=0.05)


def test_exact_sampler_skips_zero_amplitude_configs():
    # Regression for the `prob > 0.0` support mask: zero-amplitude configurations
    # must never be sampled, and every non-zero one should appear.
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    psi = np.ones(symm.basis.Ns)
    psi[[1, 4, 6]] = 0.0
    state = DenseState(jnp.asarray(psi), symm)

    samples = ExactSampler(state, 4000 * NDEV, reweight=2.0).sweep()
    assert np.all(np.abs(np.asarray(samples.psi)) > 0)
    assert np.all(np.asarray(samples.prob) > 0)
    uniq = np.unique(np.asarray(samples.spins), axis=0)
    assert uniq.shape[0] == np.count_nonzero(psi)


def test_exact_sampler_with_symmetry_and_variational_smoke():
    # End-to-end path through a real ansatz and a non-trivial symmetry orbit
    # (the random symmetry-image selection in `sweep`).
    Chain(4, boundary=1)
    state = Variational(RBM_Dense(features=4), symm=Translation([1]))
    ns = 4 * NDEV
    samples = ExactSampler(state, ns, reweight=1.0).sweep()

    spins = np.asarray(samples.spins)
    assert spins.shape == (ns, get_sites().Nmodes)
    assert set(np.unique(spins).tolist()).issubset({-1, 1})
    assert samples.nsamples == ns
    prob = np.asarray(samples.prob)
    assert prob.shape == (ns,)
    assert np.all(np.isfinite(prob)) and np.all(prob > 0)
    r = np.asarray(samples.r)
    assert r.shape == (ns,)
    assert np.all(np.isfinite(r))
    assert np.isclose(r.mean(), 1.0, atol=1e-4)


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
    reps = ints_to_array(symm.basis.states)  # (Nrep, L)
    # The amplitude depends on the representative only through the translation
    # invariant number of up spins, so psi(rep(s)) is known from s directly.
    amp = lambda s: 1.0 + np.count_nonzero(np.asarray(s) > 0, axis=-1)
    state = DenseState(jnp.asarray(amp(reps), dtype=jnp.float32), symm)

    samples = ExactSampler(state, STAT_NS, reweight=reweight).sweep()
    spins = np.asarray(samples.spins)
    prob = np.asarray(samples.prob, dtype=np.float64)

    # m_s = number of translations leaving s invariant
    shifts = np.stack([np.roll(spins, k, axis=1) for k in range(L)], axis=1)
    mult = np.all(shifts == spins[:, None, :], axis=2).sum(axis=1)
    assert {1, 2, 4} <= set(np.unique(mult).tolist())  # orbits of different sizes
    p_rep = amp(spins) ** reweight / np.sum(amp(reps) ** reweight)
    expected = p_rep * mult / L  # normalized over all configurations
    ratio = prob / expected
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-5)

    # the samples follow the recorded probability
    uniq, index, counts = np.unique(
        spins, axis=0, return_index=True, return_counts=True
    )
    assert np.allclose(counts / counts.sum(), expected[index], atol=0.02)

    if reweight == 2.0:
        # Born rule of the symmetric state, so the reweighting factor is trivial
        np.testing.assert_allclose(np.asarray(samples.r), 1.0, rtol=1e-5)


def test_exact_sampler_prob_reflects_sampling_forward_pass():
    # prob is evaluated from the dense wave function used for drawing, psi from
    # the state itself. A mismatch between the two, e.g. from the precision of
    # the forward passes, is corrected by r = |psi|^2 / prob.
    state, configs, psi = _skewed_dense_state(3)
    distortion = np.where(configs[:, 1] > 0, 2.0, 1.0)
    todense = state.todense
    state.todense = lambda symm=None: DenseState(
        jnp.asarray(psi * distortion), todense(symm).symm
    )
    p = psi**2 / np.sum(psi**2)
    exact_s0 = float(np.sum(p * configs[:, 0]))

    samples = ExactSampler(state, STAT_NS, reweight=2.0).sweep()
    spins = np.asarray(samples.spins)
    f = np.where(spins[:, 1] > 0, 2.0, 1.0)
    ratio = np.asarray(samples.prob) / (np.abs(np.asarray(samples.psi)) * f) ** 2
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-5)

    r = np.asarray(samples.r)
    est = float(np.mean(r * spins[:, 0]))
    assert np.isclose(est, exact_s0, atol=0.05)


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
    # `Operator.Oloc` divides psi of the connected configurations, evaluated from
    # the state, by `samples.psi`. Normalizing psi of the samples would therefore
    # scale every local estimator by the norm of the state.
    Chain(6, boundary=1)
    state = Variational(RBM_Dense(features=4))
    samples = ExactSampler(state, 1000 * NDEV, reweight=2.0).sweep()
    psi = np.asarray(samples.psi)
    assert np.allclose(psi, np.asarray(state(samples.spins)), rtol=1e-5)

    # the local energies are then the ones of the state, so the sampled mean
    # reproduces <H> of the full wave function
    H = Ising(h=1.0)
    psi_dense = np.asarray(state.todense().psi)
    p = psi_dense**2 / np.sum(psi_dense**2)
    Eexact = float(p @ (np.asarray(H.todense()) @ psi_dense / psi_dense))
    assert np.isclose(H.expectation(state, samples).real, Eexact, rtol=0.05)


def test_exact_sampler_probability_survives_large_amplitudes():
    # `|psi| ** reweight` of an unnormalized state overflows float32, so the
    # sampling probability is computed from the normalized wave function while
    # psi of the samples keeps the log representation of the state.
    Chain(3, boundary=1)
    symm = Identity()
    symm.basis_make()
    # |psi|**2 = exp(100) overflows float32, while the norm itself does not
    logabs = np.full(symm.basis.Ns, 50.0)
    logabs[0] += np.log(2.0)  # one configuration with twice the amplitude
    state = DenseState(LogArray(jnp.ones(symm.basis.Ns), jnp.asarray(logabs)), symm)

    samples = ExactSampler(state, 4000 * NDEV, reweight=2.0).sweep()
    assert np.allclose(np.asarray(samples.psi.logabs), logabs[1], atol=1.0)

    uniq, counts = np.unique(np.asarray(samples.spins), axis=0, return_counts=True)
    freq = counts / counts.sum()
    weights = np.exp(2 * (np.asarray(state(jnp.asarray(uniq)).logabs) - logabs.max()))
    assert np.allclose(freq, weights / weights.sum(), atol=0.02)


# --- RandomSampler ---


def test_random_sampler_forces_zero_reweight():
    Chain(3, boundary=1)
    sampler = RandomSampler(DenseState(jnp.ones(8)), nsamples=4 * NDEV)
    assert sampler.reweight == 0.0


def test_random_sampler_outputs_valid_pm1_samples():
    Chain(3, boundary=1)
    ns = 4 * NDEV
    samples = RandomSampler(DenseState(jnp.ones(8)), ns).sweep()
    spins = np.asarray(samples.spins)
    assert spins.shape == (ns, get_sites().Nmodes)
    assert set(np.unique(spins).tolist()).issubset({-1, 1})
    assert samples.state_internal is None
    np.testing.assert_array_equal(np.asarray(samples.prob), np.ones(ns))
    assert samples.r is not None  # included in the returned samples
    r = np.asarray(samples.r)
    assert r.shape == (ns,)
    assert np.isclose(r.mean(), 1.0, atol=1e-4)


def test_random_sampler_reweighting_recovers_observable():
    # Uniform proposals reweighted by |psi|^2 recover <s0>_{|psi|^2} = 0.6,
    # whereas the un-reweighted uniform mean of s0 is 0.
    state, configs, psi = _skewed_dense_state(3)
    p = psi**2 / np.sum(psi**2)
    exact_s0 = float(np.sum(p * configs[:, 0]))

    samples = RandomSampler(state, STAT_NS).sweep()
    spins = np.asarray(samples.spins)
    r = np.asarray(samples.r)
    est = float(np.mean(r * spins[:, 0]))
    assert np.isclose(est, exact_s0, atol=0.05)
