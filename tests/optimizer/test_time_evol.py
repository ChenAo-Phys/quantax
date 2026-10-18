"""
Characterization tests for TimeEvol in quantax/optimizer/sr.py: the S/F construction
(direct and memory-efficient chunked paths), the solve_SF packing of every vs_type,
the equivalence with real-time SR, and the check that the samples are not reweighted.
"""

import math
import numpy as np
import jax
import jax.numpy as jnp
import pytest
from quantax.state import Variational
from quantax.operator import Heisenberg
from quantax.sampler import Samples
from quantax.optimizer import TimeEvol, SR, lstsq_pinv_eig, pinvh_solve
from quantax.symmetry import TransND
from quantax.utils import to_distributed_array

from _refs import make_state, make_samples, prob_only, ref_obar, ref_energy_ebar

NDEV = jax.device_count()
# At least 16 samples, more than the 10 parameters as TimeEvol assumes, with an even
# number of at least 4 per device: more than max_parallel=2, so a chunked state takes
# the memory-efficient path, in whole chunks
NS = NDEV * max(4, 2 * math.ceil(8 / NDEV))


def _assert_sf_matches_reference(te, state, H, samples, S, F):
    """S = Obar^† Obar, F = Obar^† Ebar and the statistics of the unweighted samples."""
    unweighted = Samples(samples.spins, psi=samples.psi)
    Obar = ref_obar(state, unweighted)
    Ebar, energy, VarE = ref_energy_ebar(state, H, unweighted, clip=None)
    tol = dict(rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(np.asarray(S), Obar.conj().T @ Obar, **tol)
    np.testing.assert_allclose(np.asarray(F), Obar.conj().T @ Ebar, **tol)
    np.testing.assert_allclose(float(np.asarray(te.energy)), energy, rtol=1e-10)
    np.testing.assert_allclose(float(np.asarray(te.VarE)), VarE, rtol=1e-10)


@pytest.mark.parametrize(
    "kind, rtol",
    [("holomorphic", None), ("non_holomorphic", None), ("real_to_complex", 0.1)],
)
def test_step_matches_reference_and_real_time_sr(kind, rtol, x64):
    # TimeEvol solves S theta = i F, projected onto the real parameters as
    # S.real theta = -F.imag for non-holomorphic and real_to_complex states;
    # real-time SR solves the same system in least-squares form, so with matching
    # pinv regularization the steps agree. A user solver replaces the default
    # pinvh_solve(): rtol=0.1 truncates part of the spectrum of S, changing the step.
    state = make_state(kind)
    H = Heisenberg()
    te = TimeEvol(state, H, solver=None if rtol is None else pinvh_solve(rtol))
    # local-energy clipping is biased and must stay off in real-time dynamics,
    # regardless of SR's default
    assert te._grad._clip is None

    samples = make_samples(state, NS)
    S, F = te.get_SF(samples)
    _assert_sf_matches_reference(te, state, H, samples, S, F)

    step = np.asarray(te.solve_SF(S, F))
    sr = SR(state, H, imag_time=False, solver=lstsq_pinv_eig(rtol), clip=None)
    sr_step = np.asarray(sr.get_step(samples))
    assert step.shape == (state.nparams,) and step.dtype == sr_step.dtype
    np.testing.assert_allclose(step, sr_step, rtol=1e-8, atol=1e-10)
    # raw spins are accepted as samples
    np.testing.assert_allclose(np.asarray(te.get_step(samples.spins)), step, rtol=1e-12)


@pytest.mark.parametrize("max_parallel", [None, 2])
def test_reweight_check(max_parallel, x64, monkeypatch):
    # The samples must be drawn from |psi|^2 and are used with r = 1, on the direct
    # and the chunked path alike. The check runs after the local energy, which fills
    # psi and r of the samples carrying only prob.
    state = make_state("holomorphic")
    if max_parallel is not None:
        state = Variational(state.model, max_parallel=max_parallel)
    H = Heisenberg()
    te = TimeEvol(state, H)
    if max_parallel is not None:
        # more samples than max_parallel per device must take the chunked path
        def direct(samples):
            raise AssertionError("the direct S/F path is taken")

        monkeypatch.setattr(te, "_get_SF_direct", direct)
    samples = make_samples(state, NS)
    spins, psi = samples.spins, samples.psi

    # r = 1 up to a few percent, e.g. from a low-precision sweep reweighted towards
    # the direct forward pass, is ignored, and so is r = 0 of a sample whose
    # |psi|^2 / prob is not finite
    r = 1 + 0.03 * np.random.default_rng(0).uniform(-1, 1, NS)
    r /= r.mean()
    r[3] = 0.0
    noisy = Samples(spins, psi=psi, r=to_distributed_array(jnp.asarray(r)))
    S, F = te.get_SF(noisy)
    _assert_sf_matches_reference(te, state, H, samples, S, F)

    # samples drawn from |psi|^2 carrying only prob are accepted with r = 1
    born = Samples(spins, prob=jnp.abs(jnp.asarray(psi)) ** 2)
    S, F = te.get_SF(born)
    np.testing.assert_allclose(np.asarray(born.r), 1.0, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(born.psi), np.asarray(psi), rtol=1e-12)
    _assert_sf_matches_reference(te, state, H, samples, S, F)

    # reweighted samples raise, whether r is given or only known from prob
    with pytest.raises(ValueError, match="non-reweighted"):
        te.get_step(make_samples(state, NS, reweight=True))
    with pytest.raises(ValueError, match="non-reweighted"):
        te.get_step(prob_only(samples)[0])


def _force_chunked(te, monkeypatch):
    def direct(samples):
        raise AssertionError("the direct S/F path is taken")

    monkeypatch.setattr(te, "_get_SF_direct", direct)


@pytest.mark.parametrize("kind", ["holomorphic", "non_holomorphic", "real_to_complex"])
def test_chunked_sf_padded_last_chunk(kind, x64, monkeypatch):
    # The memory-efficient path splits each device's samples into chunks of
    # max_parallel. An odd number of samples per device pads the last chunk, and
    # the padded samples must not enter S, F or the mean Jacobian. The Jacobian of a
    # non-holomorphic state has 2 * nparams columns, so S is 2 nparams x 2 nparams.
    state = Variational(make_state(kind).model, max_parallel=2)
    H = Heisenberg()
    te = TimeEvol(state, H)
    _force_chunked(te, monkeypatch)
    # NS / NDEV is even, so one more sample per device leaves 1 padded row per device
    samples = make_samples(state, NS + NDEV)
    S, F = te.get_SF(samples)
    _assert_sf_matches_reference(te, state, H, samples, S, F)


def test_chunked_sf_padding_with_vanishing_psi(x64, monkeypatch):
    # In the k = pi sector the translation characters are exactly +-1 for a real
    # dtype, so psi vanishes on the translation-invariant all-up padding and its
    # Jacobian row is NaN: the padded rows must be dropped, not weighted by 0.
    model = make_state("real").model
    state = Variational(model, symm=TransND(sector=2), max_parallel=2)
    up = to_distributed_array(jnp.ones((NDEV, state.Nmodes), jnp.int8))
    assert np.all(np.asarray(state(up)) == 0)
    H = Heisenberg()
    te = TimeEvol(state, H)
    _force_chunked(te, monkeypatch)

    # samples drawn from |psi|^2 never have psi = 0, e.g. the Neel states of this
    # spin-flip-even model
    candidates = make_samples(state, 8 * NS)
    nonzero = np.flatnonzero(np.abs(np.asarray(candidates.psi)) > 1e-8)
    idx = nonzero[: NS + NDEV]
    spins = np.asarray(candidates.spins)[idx]
    psi = np.asarray(candidates.psi)[idx]
    samples = Samples(
        to_distributed_array(jnp.asarray(spins)), psi=to_distributed_array(psi)
    )
    S, F = te.get_SF(samples)
    _assert_sf_matches_reference(te, state, H, samples, S, F)
