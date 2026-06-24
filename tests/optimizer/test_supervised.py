"""
Characterization tests for quantax/optimizer/supervised.py: the overlap gradient
(Supervised), the Supervised+AdamSR multiple-inheritance combination, and the
full-summation variant (SupervisedExact).
"""

import numpy as np
import jax.numpy as jnp
import pytest
import quantax as qtx
from quantax.state import DenseState
from quantax.operator import Ising
from quantax.sampler import RandomSampler
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
    ref_obar,
    ref_overlap_ebar,
    ref_solve,
    ref_adam_step,
)


def _dense_target(dim, seed=7, complex_=False):
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim)
    if complex_:
        vec = vec + 1j * rng.normal(size=dim)
    vec = vec / np.linalg.norm(vec)
    return DenseState(jnp.asarray(vec))


@pytest.mark.parametrize("clip", [None, 0.5])
def test_supervised_step_reference(clip, x64):
    # A real state keeps the psi ratio real, which is required for the clip
    # branch (jnp.clip is undefined for complex input).
    state = make_state("real")
    target = _dense_target(2**4)
    solver = auto_shift_eig()
    opt = Supervised(state, target, solver=solver, clip=clip)
    samples = make_samples(state, 16, reweight=True)

    step = np.asarray(opt.get_step(samples))
    Ebar = ref_overlap_ebar(state, target, samples, clip)
    np.testing.assert_allclose(
        np.asarray(opt.get_Ebar(samples)), Ebar, rtol=1e-10, atol=1e-14
    )
    expected = ref_solve(state, solver, ref_obar(state, samples), Ebar)
    np.testing.assert_allclose(step, expected, rtol=1e-8, atol=1e-12)


def test_supervised_adam_combines_overlap_ebar_with_adam_solve(x64):
    # SupervisedAdam takes get_Ebar from Supervised and solve from AdamSR
    # (multiple inheritance); the composition is pinned by value.
    state = make_state("real")
    target = _dense_target(2**4)
    solver = auto_shift_eig()
    mu, beta = 0.95, 0.995
    opt = SupervisedAdam(state, target, solver=solver, mu=mu, beta=beta)
    assert opt.energy is None
    assert opt.hamiltonian is None

    bufs = {
        "m": np.zeros(state.nparams),
        "v": np.zeros(state.nparams),
        "t": 0,
    }
    for seed in (0, 1):
        samples = make_samples(state, 16, seed=seed)
        step = np.asarray(opt.get_step(samples))
        Ebar = ref_overlap_ebar(state, target, samples, clip=None)
        ref, bufs = ref_adam_step(
            state, solver, ref_obar(state, samples), Ebar, bufs, mu, beta
        )
        np.testing.assert_allclose(step, ref, rtol=1e-8, atol=1e-13)


def test_supervised_exact_step_reference(x64):
    state = make_state("holomorphic")
    target = _dense_target(2**4, complex_=True)
    solver = auto_shift_eig()
    opt = SupervisedExact(state, target, solver=solver)
    step = np.asarray(opt.get_step())

    symm = state.symm
    basis = symm.basis
    spins = ints_to_array(basis.states)
    psi = np.asarray(state(jnp.asarray(spins)))
    psi = psi / np.asarray(basis.get_amp(basis.states))
    psi = psi / np.linalg.norm(psi)

    target_psi = np.asarray(target.todense(symm).psi)
    Ebar = psi - target_psi / np.vdot(psi, target_psi)

    Omat = np.asarray(state.jacobian(to_distributed_array(jnp.asarray(spins))))
    Omat = Omat * psi[:, None]
    Obar = Omat - psi[:, None] * (psi.conj() @ Omat)[None, :]

    expected = ref_solve(state, solver, Obar, Ebar)
    np.testing.assert_allclose(step, expected, rtol=1e-8, atol=1e-12)


def test_supervised_training_increases_overlap(x64):
    # End-to-end sign convention check: training toward a target must increase
    # the fidelity. Uses RandomSampler so the reweighted path is exercised.
    state = make_state("real")
    Hd = np.asarray(Ising(h=1.0).todense())
    gs = np.linalg.eigh(Hd)[1][:, 0]
    target = DenseState(jnp.asarray(np.abs(gs)))  # stoquastic: |gs| = gs up to sign
    sampler = RandomSampler(state, 256)
    opt = Supervised(state, target, solver=auto_shift_eig())

    def fidelity():
        psi = np.asarray(state.todense().psi)
        return np.abs(psi @ np.abs(gs)) ** 2 / (psi @ psi)

    f0 = fidelity()
    for _ in range(40):
        samples = sampler.sweep()
        state.update(0.1 * opt.get_step(samples))
    f1 = fidelity()
    assert f1 > f0
    assert f1 > 0.9
