"""
Characterization tests for quantax/optimizer/sr.py: the SR energy gradient and
statistics, the SPRING / MARCH / AdamSR momentum variants, and exact
reconfiguration (ER).
"""

import numpy as np
import jax.numpy as jnp
import pytest
import quantax as qtx
from quantax.sites import Chain
from quantax.state import Variational
from quantax.model import RBM_Dense
from quantax.operator import Heisenberg, Ising
from quantax.sampler import ExactSampler
from quantax.optimizer import SR, SPRING, MARCH, AdamSR, ER, auto_shift_eig, lsmr
from quantax.utils import ints_to_array, to_distributed_array

from _refs import (
    make_state,
    make_samples,
    ref_obar,
    ref_energy_ebar,
    ref_solve,
    ref_spring_step,
    ref_march_step,
    ref_adam_step,
)

# ====================================================================
# SR statistics and estimators
# ====================================================================


@pytest.mark.parametrize("reweight", [False, True])
def test_energy_vare_and_ebar(reweight, x64):
    # energy and VarE center at the weighted mean; Ebar centers at the
    # unweighted mean. Both behaviors are pinned.
    state = make_state("holomorphic")
    H = Heisenberg()
    opt = SR(state, H, solver=auto_shift_eig())
    assert opt.energy is None and opt.VarE is None
    samples = make_samples(state, 32, reweight=reweight)
    opt.get_step(samples)
    Ebar_ref, energy_ref, VarE_ref = ref_energy_ebar(state, H, samples)
    np.testing.assert_allclose(float(np.asarray(opt.energy)), energy_ref, rtol=1e-10)
    np.testing.assert_allclose(float(np.asarray(opt.VarE)), VarE_ref, rtol=1e-10)
    np.testing.assert_allclose(
        np.asarray(opt.get_Ebar(samples)), Ebar_ref, rtol=1e-10, atol=1e-14
    )


@pytest.mark.parametrize("reweight", [False, True])
def test_get_obar(reweight, x64):
    state = make_state("holomorphic")
    opt = SR(state, Heisenberg(), solver=auto_shift_eig())
    samples = make_samples(state, 32, reweight=reweight)
    np.testing.assert_allclose(
        np.asarray(opt.get_Obar(samples)),
        ref_obar(state, samples),
        rtol=1e-10,
        atol=1e-14,
    )


# ====================================================================
# Momentum variants
# ====================================================================


def test_first_momentum_step_equals_sr(x64):
    # With zeroed buffers, the first SPRING and MARCH steps reduce to plain SR.
    state = make_state("holomorphic")
    H = Heisenberg()
    solver = auto_shift_eig()
    samples = make_samples(state, 16)
    sr_step = np.asarray(SR(state, H, solver=solver).get_step(samples))
    spring_step = np.asarray(SPRING(state, H, solver=solver).get_step(samples))
    march_step = np.asarray(MARCH(state, H, solver=solver).get_step(samples))
    np.testing.assert_allclose(spring_step, sr_step, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(march_step, sr_step, rtol=1e-12, atol=1e-15)


def _two_step_data(state, H, seeds=(0, 1)):
    out = []
    for seed in seeds:
        samples = make_samples(state, 16, seed=seed)
        Ebar, _, _ = ref_energy_ebar(state, H, samples)
        out.append((samples, ref_obar(state, samples), Ebar))
    return out


@pytest.mark.parametrize("norm_clip", [None, 1e-3])
def test_spring_two_steps(norm_clip, x64):
    state = make_state("holomorphic")
    H = Heisenberg()
    solver = auto_shift_eig()
    mu = 0.9
    opt = SPRING(state, H, solver=solver, mu=mu, norm_clip=norm_clip)
    (s1, O1, e1), (s2, O2, e2) = _two_step_data(state, H)

    step1 = np.asarray(opt.get_step(s1))
    phi0 = np.zeros(state.nparams, dtype=np.complex128)
    ref1 = ref_spring_step(state, solver, O1, e1, phi0, mu, norm_clip)
    np.testing.assert_allclose(step1, ref1, rtol=1e-9, atol=1e-13)

    step2 = np.asarray(opt.get_step(s2))
    ref2 = ref_spring_step(state, solver, O2, e2, ref1, mu, norm_clip)
    np.testing.assert_allclose(step2, ref2, rtol=1e-8, atol=1e-13)


@pytest.mark.parametrize("norm_clip", [None, 1e-3])
def test_march_two_steps(norm_clip, x64):
    state = make_state("holomorphic")
    H = Heisenberg()
    solver = auto_shift_eig()
    mu, beta = 0.95, 0.995
    opt = MARCH(state, H, solver=solver, mu=mu, beta=beta, norm_clip=norm_clip)
    (s1, O1, e1), (s2, O2, e2) = _two_step_data(state, H)

    bufs = {
        "phi": np.zeros(state.nparams, dtype=np.complex128),
        "v": np.zeros(state.nparams),
    }
    step1 = np.asarray(opt.get_step(s1))
    ref1, _ = ref_march_step(state, solver, O1, e1, bufs, mu, beta, norm_clip)
    np.testing.assert_allclose(step1, ref1, rtol=1e-9, atol=1e-13)

    # condition the step-2 reference on the actual step-1 buffers; the V^-1
    # rescaling amplifies float64 noise, hence the looser tolerance
    bufs = {"phi": step1, "v": np.abs(step1) ** 2}
    step2 = np.asarray(opt.get_step(s2))
    ref2, _ = ref_march_step(state, solver, O2, e2, bufs, mu, beta, norm_clip)
    np.testing.assert_allclose(step2, ref2, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("norm_clip", [None, 1e-3])
def test_adamsr_two_steps(norm_clip, x64):
    # The default eig-family solver has no native diag_preconditioner, so the
    # second solve uses the emulated right preconditioning; the reference
    # reproduces exactly that behavior.
    state = make_state("holomorphic")
    H = Heisenberg()
    solver = auto_shift_eig()
    mu, beta = 0.95, 0.995
    opt = AdamSR(state, H, solver=solver, mu=mu, beta=beta, norm_clip=norm_clip)
    (s1, O1, e1), (s2, O2, e2) = _two_step_data(state, H)

    bufs = {
        "m": np.zeros(state.nparams, dtype=np.complex128),
        "v": np.zeros(state.nparams),
        "t": 0,
    }
    # The clipped case makes g (and hence V) tiny, so the emulated 1/V rescaling
    # amplifies jit-vs-eager float reordering to a ~1e-6 absolute floor on the
    # small components; the unclipped case is well-conditioned and matches far
    # tighter.
    rtol, atol = (1e-5, 1e-6) if norm_clip is not None else (1e-9, 1e-13)
    step1 = np.asarray(opt.get_step(s1))
    ref1, _ = ref_adam_step(state, solver, O1, e1, bufs, mu, beta, norm_clip)
    np.testing.assert_allclose(step1, ref1, rtol=rtol, atol=atol)

    # condition the step-2 reference on the optimizer's realized buffers to avoid
    # compounding the step-1 round-off through another 1/V rescaling.
    bufs = {k: np.asarray(opt._buffers[k]) for k in ("m", "v", "t")}
    step2 = np.asarray(opt.get_step(s2))
    ref2, _ = ref_adam_step(state, solver, O2, e2, bufs, mu, beta, norm_clip)
    np.testing.assert_allclose(step2, ref2, rtol=rtol, atol=atol)


def test_adamsr_lsmr_uses_diag_preconditioner(x64):
    # lsmr is the solver that consumes diag_preconditioner. In the
    # underdetermined regime the preconditioner changes the min-norm solution,
    # so this pins that AdamSR actually plumbs it through.
    state = make_state("holomorphic", features=16)  # nparams > nsamples
    H = Heisenberg()
    solver = lsmr(rtol=1e-12, maxiter=2000)
    mu, beta = 0.95, 0.995
    opt = AdamSR(state, H, solver=solver, mu=mu, beta=beta)
    (s1, O1, e1), (s2, O2, e2) = _two_step_data(state, H)

    bufs = {
        "m": np.zeros(state.nparams, dtype=np.complex128),
        "v": np.zeros(state.nparams),
        "t": 0,
    }
    step1 = np.asarray(opt.get_step(s1))
    ref1, bufs = ref_adam_step(state, solver, O1, e1, bufs, mu, beta)
    np.testing.assert_allclose(step1, ref1, rtol=1e-6, atol=1e-10)

    step2 = np.asarray(opt.get_step(s2))
    ref2, bufs2 = ref_adam_step(state, solver, O2, e2, bufs, mu, beta)
    np.testing.assert_allclose(step2, ref2, rtol=1e-6, atol=1e-10)

    # without the preconditioner the second solve gives a different min-norm
    # solution, proving the kwarg is actually consumed
    mhat2 = bufs2["m"] / (1 - mu ** bufs2["t"])
    no_precond = ref_solve(state, solver, O2, e2 - O2 @ mhat2) + mhat2
    assert np.max(np.abs(step2 - no_precond)) > 1e-8 * np.linalg.norm(step2)


# ====================================================================
# ER (exact reconfiguration)
# ====================================================================


def test_er_step_and_energy_reference(x64):
    state = make_state("holomorphic")
    H = Heisenberg()
    solver = auto_shift_eig()
    er = ER(state, H, solver=solver)
    step = np.asarray(er.get_step())

    symm = state.symm
    basis = symm.basis
    spins = ints_to_array(basis.states)
    psi = np.asarray(state(jnp.asarray(spins)))
    psi = psi / np.asarray(basis.get_amp(basis.states))
    psi = psi / np.linalg.norm(psi)

    Omat = np.asarray(state.jacobian(to_distributed_array(jnp.asarray(spins))))
    Omat = Omat * psi[:, None]
    Obar = Omat - psi[:, None] * (psi.conj() @ Omat)[None, :]

    Hd = np.asarray(H.todense(symm))
    energy = (psi.conj() @ Hd @ psi).real
    Ebar = Hd @ psi - energy * psi

    np.testing.assert_allclose(float(np.asarray(er.energy)), energy, rtol=1e-10)
    expected = ref_solve(state, solver, Obar, Ebar)
    np.testing.assert_allclose(step, expected, rtol=1e-8, atol=1e-12)


def test_er_converges_to_ground_state(x64):
    state = make_state("holomorphic", features=8)
    H = Heisenberg()
    er = ER(state, H)
    for _ in range(60):
        state.update(0.05 * er.get_step())
    er.get_step()  # refresh the energy after the last update
    E0 = np.linalg.eigvalsh(np.asarray(H.todense()))[0]
    assert float(np.asarray(er.energy)) < E0 + 0.05 * abs(E0)


# ====================================================================
# SR integration (mirrors the quick_start tutorial)
# ====================================================================


def test_sr_converges_ising_chain():
    Chain(6, boundary=1)
    H = Ising(h=1.0)
    state = Variational(RBM_Dense(features=12))
    sampler = ExactSampler(state, 512)
    opt = SR(state, H)
    energies = []
    for _ in range(100):
        samples = sampler.sweep()
        step = opt.get_step(samples)
        state.update(step * 0.03)
        energies.append(float(np.asarray(opt.energy)))
    E0 = np.linalg.eigvalsh(np.asarray(H.todense()))[0]
    final = np.mean(energies[-10:])
    assert final < energies[0]
    assert final < E0 * 0.95  # within 5% of the ground state (E0 < 0)
