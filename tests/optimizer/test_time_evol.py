"""
Characterization tests for TimeEvol in quantax/optimizer/sr.py: the S/F construction
(direct and memory-efficient chunked paths), the solve_SF packing, and the
equivalence with real-time SR.
"""

import numpy as np
import jax.numpy as jnp
import pytest
import quantax as qtx
from quantax.sites import Chain
from quantax.state import Variational
from quantax.model import RBM_Dense
from quantax.operator import Heisenberg, Ising
from quantax.sampler import Samples
from quantax.optimizer import TimeEvol, SR, lstsq_pinv_eig, pinvh_solve
from quantax.utils import to_distributed_array

from _dtype import use_dtype
from _refs import make_state, make_samples, prob_only, ref_obar, ref_energy_ebar


def test_sf_direct_matches_reference(x64):
    state = make_state("holomorphic")
    H = Heisenberg()
    te = TimeEvol(state, H)
    samples = make_samples(state, 32)
    Smat, Fvec = te.get_SF(samples)
    Obar = ref_obar(state, samples)
    Ebar, energy, VarE = ref_energy_ebar(state, H, samples, clip=None)
    np.testing.assert_allclose(
        np.asarray(Smat), Obar.conj().T @ Obar, rtol=1e-10, atol=1e-14
    )
    np.testing.assert_allclose(
        np.asarray(Fvec), Obar.conj().T @ Ebar, rtol=1e-10, atol=1e-14
    )
    np.testing.assert_allclose(float(np.asarray(te.energy)), energy, rtol=1e-10)
    np.testing.assert_allclose(float(np.asarray(te.VarE)), VarE, rtol=1e-10)


def test_sf_indirect_matches_direct(x64):
    # The chunked accumulation must reproduce the direct S and F. Both states
    # share the same model parameters; only the chunking differs.
    use_dtype(jnp.complex128)
    Chain(4, boundary=1)
    model = RBM_Dense(2, dtype=jnp.complex128)
    state_full = Variational(model)
    state_chunk = Variational(model, max_parallel=2)
    H = Heisenberg()
    te_full = TimeEvol(state_full, H)
    te_chunk = TimeEvol(state_chunk, H)

    samples = make_samples(state_full, 32)
    S1, F1 = te_full.get_SF(samples)
    S2, F2 = te_chunk.get_SF(samples)
    np.testing.assert_allclose(np.asarray(S1), np.asarray(S2), rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(np.asarray(F1), np.asarray(F2), rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(
        float(np.asarray(te_full.energy)), float(np.asarray(te_chunk.energy)), rtol=1e-9
    )


def test_step_matches_real_time_sr(x64):
    # TimeEvol solves S theta = i F; real-time SR solves the same system in
    # least-squares form. With matching pinv regularization the steps agree.
    state = make_state("holomorphic")
    H = Heisenberg()
    samples = make_samples(state, 64)
    te_step = np.asarray(TimeEvol(state, H).get_step(samples))
    # clip=None to match TimeEvol, which disables the (biased) Eloc clipping
    sr = SR(state, H, imag_time=False, solver=lstsq_pinv_eig(), clip=None)
    sr_step = np.asarray(sr.get_step(samples))
    np.testing.assert_allclose(te_step, sr_step, rtol=1e-6, atol=1e-10)


def test_time_evol_disables_eloc_clip(x64):
    # Local-energy clipping is a biased operation and must stay off in
    # real-time dynamics; TimeEvol pins clip=None regardless of SR's default.
    state = make_state("holomorphic")
    te = TimeEvol(state, Heisenberg())
    assert te._grad._clip is None


def test_solve_sf_packing_real_to_complex(x64):
    # For real parameters with complex output the equation is projected onto
    # the real part: solve(S.real, -F.imag).
    state = make_state("real_to_complex")
    H = Ising(h=1.0)
    solver = pinvh_solve()
    te = TimeEvol(state, H, solver=solver)
    samples = make_samples(state, 16)
    step = np.asarray(te.get_step(samples))

    Obar = ref_obar(state, samples)
    Ebar, _, _ = ref_energy_ebar(state, H, samples, clip=None)
    Smat = Obar.conj().T @ Obar
    Fvec = Obar.conj().T @ Ebar
    expected = np.asarray(solver(jnp.asarray(Smat.real), jnp.asarray(-Fvec.imag)))
    expected = expected.astype(np.dtype(qtx.get_default_dtype()))
    np.testing.assert_allclose(step, expected, rtol=1e-8, atol=1e-12)


def _state(max_parallel):
    """A holomorphic state, chunked to take the memory-efficient TimeEvol path."""
    state = make_state("holomorphic")
    if max_parallel is None:
        return state
    return Variational(state.model, max_parallel=max_parallel)


# 16 samples exceed max_parallel=2 on up to 4 devices, so the chunked path is taken
@pytest.mark.parametrize("max_parallel", [None, 2])
def test_reweighted_samples_raise(x64, max_parallel):
    state = _state(max_parallel)
    te = TimeEvol(state, Heisenberg())
    samples = make_samples(state, 16, reweight=True)
    with pytest.raises(ValueError):
        te.get_step(samples)


@pytest.mark.parametrize("max_parallel", [None, 2])
def test_prob_only_reweighted_samples_raise(x64, max_parallel):
    # The reweighting factor of samples carrying only prob is only known after the
    # local energy fills in psi, where it's checked.
    state = _state(max_parallel)
    te = TimeEvol(state, Heisenberg())
    samples, _ = prob_only(make_samples(state, 16))
    with pytest.raises(ValueError):
        te.get_step(samples)
    # samples drawn from |psi|^2 are not reweighted
    spins = make_samples(state, 16).spins
    born = Samples(spins, prob=jnp.abs(jnp.asarray(state(spins))) ** 2)
    te.get_step(born)
    np.testing.assert_allclose(np.asarray(born.r), 1.0, rtol=1e-10)


@pytest.mark.parametrize("max_parallel", [None, 2])
def test_reweight_check_tolerates_noise_and_invalid_samples(x64, max_parallel):
    # r = 1 up to a few percent, e.g. from a low-precision sweep reweighted towards
    # the direct forward pass, is accepted, and so is r = 0 of a sample whose psi
    # is not finite.
    state = _state(max_parallel)
    te = TimeEvol(state, Heisenberg())
    samples = make_samples(state, 32)
    r = 1 + 0.03 * np.random.default_rng(0).uniform(-1, 1, 32)
    r[3] = 0.0
    samples.r = to_distributed_array(jnp.asarray(r / r.mean()))
    te.get_step(samples)


def test_samples_are_used_with_trivial_r(x64):
    # r within the tolerance, e.g. noise of a low-precision sampler, is ignored: S, F
    # and the energy are those of r = 1, on the direct and the chunked path alike.
    state = make_state("holomorphic")
    chunked = Variational(state.model, max_parallel=2)
    H = Heisenberg()
    samples = make_samples(state, 32)
    r = 1 + 0.03 * np.random.default_rng(0).uniform(-1, 1, 32)
    r = to_distributed_array(jnp.asarray(r / r.mean()))

    te = TimeEvol(state, H)
    S0, F0 = te.get_SF(Samples(samples.spins, psi=samples.psi))
    E0 = float(np.asarray(te.energy))
    for vs in (state, chunked):
        te = TimeEvol(vs, H)
        S, F = te.get_SF(Samples(samples.spins, psi=samples.psi, r=r))
        np.testing.assert_allclose(np.asarray(S), np.asarray(S0), rtol=1e-9, atol=1e-13)
        np.testing.assert_allclose(np.asarray(F), np.asarray(F0), rtol=1e-9, atol=1e-13)
        np.testing.assert_allclose(float(np.asarray(te.energy)), E0, rtol=1e-9)
