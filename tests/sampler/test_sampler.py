import numpy as np
import jax
import jax.numpy as jnp
import pytest
from quantax.sites import Chain
from quantax.state import DenseState, Variational
from quantax.model import RBM_Dense
from quantax.symmetry import Identity, Translation
from quantax.sampler import Sampler, ExactSampler, RandomSampler
from quantax.utils import ints_to_array
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


def test_reweight_factor_is_normalized_to_unit_mean():
    # r_s = |psi|^(2-n) / <|psi|^(2-n)> ; here n = 1, so r ∝ |psi| with mean 1.
    Chain(2, boundary=1)
    psi = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    sampler = Sampler(DenseState(psi), nsamples=4 * NDEV, reweight=1.0)
    r = np.asarray(sampler._get_reweight_factor(psi))
    expected = np.abs(np.asarray(psi)) ** (2 - 1.0)
    expected = expected / expected.mean()
    assert np.allclose(r, expected, atol=1e-5)
    assert np.isclose(r.mean(), 1.0, atol=1e-5)


def test_reweight_factor_trivial_when_reweight_is_two():
    # n = 2 is the |psi|^2 case: every reweighting factor collapses to 1.
    Chain(2, boundary=1)
    psi = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    sampler = Sampler(DenseState(psi), nsamples=4 * NDEV, reweight=2.0)
    r = np.asarray(sampler._get_reweight_factor(psi))
    assert np.allclose(r, 1.0)


# --- ExactSampler ---


def test_exact_sampler_output_shapes():
    Chain(3, boundary=1)
    state = DenseState(jnp.ones(8))
    ns = 4 * NDEV
    samples = ExactSampler(state, ns, reweight=2.0).sweep()
    assert np.asarray(samples.spins).shape == (ns, get_sites().Nmodes)
    assert np.asarray(samples.psi).shape == (ns,)
    assert np.asarray(samples.reweight_factor).shape == (ns,)
    assert samples.state_internal is None


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


def test_exact_sampler_reweighting_recovers_observable():
    # Sample from |psi|^1 and let the reweighting factor recover <s0>_{|psi|^2}.
    # The biased state has <s0>_p = 0.6 while the *proposal* mean <s0>_q ≈ 0.33,
    # so a broken reweighting would miss the target by far more than the tolerance.
    state, configs, psi = _skewed_dense_state(3)
    p = psi**2 / np.sum(psi**2)
    exact_s0 = float(np.sum(p * configs[:, 0]))

    samples = ExactSampler(state, STAT_NS, reweight=1.0).sweep()
    spins = np.asarray(samples.spins)
    r = np.asarray(samples.reweight_factor)
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
    r = np.asarray(samples.reweight_factor)
    assert r.shape == (ns,)
    assert np.all(np.isfinite(r))
    assert np.isclose(r.mean(), 1.0, atol=1e-4)


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
    r = np.asarray(samples.reweight_factor)
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
    r = np.asarray(samples.reweight_factor)
    est = float(np.mean(r * spins[:, 0]))
    assert np.isclose(est, exact_s0, atol=0.05)
