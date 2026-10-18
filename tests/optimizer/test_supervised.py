"""
Characterization tests for quantax/optimizer/supervised.py: the overlap gradient
(Supervised), the Supervised+AdamSR multiple-inheritance combination, and the
full-summation variant (SupervisedExact).
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
import quantax as qtx
from quantax.state import DenseState
from quantax.sampler import Samples
from quantax.optimizer import (
    Supervised,
    SupervisedAdam,
    SupervisedExact,
    auto_shift_eig,
)
from quantax.utils import ints_to_array, to_distributed_array

from _refs import (
    make_state,
    make_samples,
    padded_basis,
    prob_only,
    ref_obar,
    ref_overlap_ebar,
    ref_solve,
    ref_adam_step,
)

NDEV = jax.device_count()
NS = -(-16 // NDEV) * NDEV  # samples per step, 16 rounded up to a multiple of NDEV


def _dense_target(dim, seed=7, complex_=False):
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim)
    if complex_:
        vec = vec + 1j * rng.normal(size=dim)
    vec = vec / np.linalg.norm(vec)
    return DenseState(jnp.asarray(vec))


def test_supervised_step_reference(x64):
    # Samples carrying only prob get psi and r filled in place before r is used.
    # A real state keeps the psi ratio real, which is required for the clip
    # branch (jnp.clip is undefined for complex input).
    state = make_state("real")
    target = _dense_target(2**4)
    solver = auto_shift_eig()
    for clip in (None, 0.5):
        opt = Supervised(state, target, solver=solver, clip=clip)
        samples, expected = prob_only(make_samples(state, NS))
        step = np.asarray(opt.get_step(samples))
        np.testing.assert_allclose(np.asarray(samples.psi), np.asarray(expected.psi))
        np.testing.assert_allclose(np.asarray(samples.r), np.asarray(expected.r))

        Ebar = ref_overlap_ebar(state, target, expected, clip)
        np.testing.assert_allclose(
            np.asarray(opt.get_Ebar(samples)), Ebar, rtol=1e-10, atol=1e-14
        )
        step_ref = ref_solve(state, solver, ref_obar(state, expected), Ebar)
        np.testing.assert_allclose(step, step_ref, rtol=1e-8, atol=1e-12)


def test_overlap_ebar_excludes_nonfinite_ratio(x64):
    # A NaN psi sample must not poison the mean amplitude ratio (which would
    # turn the whole Ebar into NaN): it is excluded and its Ebar entry is zero.
    state = make_state("real")
    target = _dense_target(2**4)
    n = NS
    samples = make_samples(state, n)
    psi = np.asarray(samples.psi).copy()
    psi[5] = np.nan
    samples = Samples(samples.spins, psi=to_distributed_array(jnp.asarray(psi)))

    grad = qtx.optimizer.OverlapGrad(target)
    with pytest.warns(UserWarning, match="non-finite amplitude ratio"):
        Ebar = np.asarray(grad.ebar(state, samples))
    assert np.all(np.isfinite(Ebar))
    assert Ebar[5] == 0.0

    good = np.arange(n) != 5
    phi = np.asarray(target(samples.spins))
    ratio = phi[good] / psi[good]
    ref = -(ratio / ratio.mean() - 1) * np.sqrt(1.0 / n)
    np.testing.assert_allclose(Ebar[good], ref, rtol=1e-10, atol=1e-14)


def test_supervised_adam_combines_overlap_ebar_with_adam_solve(x64):
    # SupervisedAdam takes get_Ebar from Supervised, including the clip, and solve
    # from AdamSR (multiple inheritance); the composition is pinned by value.
    state = make_state("real")
    target = _dense_target(2**4)
    solver = auto_shift_eig()
    mu, beta, clip = 0.95, 0.995, 0.5
    opt = SupervisedAdam(state, target, solver=solver, clip=clip, mu=mu, beta=beta)
    # the overlap gradient defines no Hamiltonian and no energy
    assert opt.energy is None
    assert opt.hamiltonian is None

    bufs = {
        "m": np.zeros(state.nparams),
        "v": np.zeros(state.nparams),
        "t": 0,
    }
    for seed in (0, 1):
        samples = make_samples(state, NS, seed=seed)
        step = np.asarray(opt.get_step(samples))
        Ebar = ref_overlap_ebar(state, target, samples, clip)
        ref, bufs = ref_adam_step(
            state, solver, ref_obar(state, samples), Ebar, bufs, mu, beta
        )
        # the rounding error of the solve scales with the whole step, so the small
        # components need an absolute tolerance on that scale
        atol = 1e-8 * np.abs(ref).max()
        np.testing.assert_allclose(step, ref, rtol=1e-8, atol=atol)


def test_supervised_exact_step_reference(x64):
    state = make_state("holomorphic")
    target = _dense_target(2**4, complex_=True)
    solver = auto_shift_eig()
    opt = SupervisedExact(state, target, solver=solver)
    step = np.asarray(opt.get_step())

    symm = state.symm
    basis = symm.basis
    spins, Ns = padded_basis(symm)
    psi = np.asarray(state(spins))[:Ns]
    psi = psi / np.asarray(basis.get_amp(basis.states))
    psi = psi / np.linalg.norm(psi)

    target_psi = np.asarray(target.todense(symm).psi)
    Ebar = psi - target_psi / np.vdot(psi, target_psi)

    Omat = np.asarray(state.jacobian(spins))[:Ns] * psi[:, None]
    Obar = Omat - psi[:, None] * (psi.conj() @ Omat)[None, :]

    expected = ref_solve(state, solver, Obar, Ebar)
    np.testing.assert_allclose(step, expected, rtol=1e-8, atol=1e-12)


def test_steps_increase_fidelity_at_predicted_rate(x64):
    # End-to-end sign convention of the update theta' = theta - eta * step. The
    # samples enumerate the Hilbert space with a uniform prob, repeated NDEV times to
    # be divisible over the devices, so the reweighted estimators are exact and
    # Obar^† Ebar = -grad(log F) / 2 for the fidelity F with the target. Both the
    # stochastic and the exact step must then increase log F at the rate
    # 2 (Obar^† Ebar) . step > 0, compared with a central finite difference.
    state = make_state("real")
    target = _dense_target(2**4)
    phi = np.asarray(target.psi)
    basis = state.symm.basis
    spins = np.tile(ints_to_array(basis.states), (NDEV, 1))
    spins = to_distributed_array(jnp.asarray(spins))
    prob = to_distributed_array(jnp.ones(spins.shape[0]))
    samples = Samples(spins, prob=prob)

    def log_fidelity():
        psi = np.asarray(state.todense().psi)
        return np.log((psi @ phi) ** 2 / (psi @ psi))

    stochastic = Supervised(state, target).get_step(samples)
    grad = ref_obar(state, samples).T @ ref_overlap_ebar(state, target, samples)
    exact = SupervisedExact(state, target).get_step()
    for step in (np.asarray(stochastic), np.asarray(exact)):
        rate = 2 * grad @ step
        assert rate > 0
        eta = 1e-5 / np.linalg.norm(step)  # parameter displacement 1e-5
        state.update(jnp.asarray(eta * step))
        f_plus = log_fidelity()
        state.update(jnp.asarray(-2 * eta * step))
        f_minus = log_fidelity()
        state.update(jnp.asarray(eta * step))
        np.testing.assert_allclose((f_plus - f_minus) / (2 * eta), rate, rtol=1e-7)
