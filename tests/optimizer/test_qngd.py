"""
Characterization tests for the shared QNGD mechanics in quantax/optimizer/qngd.py:
the Obar/Ebar estimation pipeline, the vs_type packing around the solver, the
buffer flow into and out of the solver, and buffer serialization.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
import quantax as qtx
from quantax.operator import Ising
from quantax.optimizer import SR, SPRING, MARCH, AdamSR, sgd_solver, auto_shift_eig

from _refs import (
    make_state,
    make_samples,
    ref_obar,
    ref_energy_ebar,
    ref_sgd_step,
    ref_solve,
)

# Each entry exercises one branch of the vs_type / imag_time packing in QNGD.solve.
# real-time evolution of a purely real state is ill-defined and not covered.
PACKING_CASES = [
    ("real", True),
    ("holomorphic", True),
    ("holomorphic", False),
    ("non_holomorphic", True),
    ("real_to_complex", True),
    ("real_to_complex", False),
]


@pytest.mark.parametrize("kind,imag_time", PACKING_CASES)
def test_get_step_full_pipeline_closed_form(kind, imag_time, x64):
    # End-to-end pin of get_step = Ebar/Obar estimation + vs_type packing +
    # unpacking, using the closed-form sgd_solver x = A^† b / len(b).
    state = make_state(kind)
    H = Ising(h=1.0)
    opt = SR(state, H, imag_time=imag_time, solver=sgd_solver())
    samples = make_samples(state, 16)
    step = np.asarray(opt.get_step(samples))
    Obar = ref_obar(state, samples)
    Ebar, _, _ = ref_energy_ebar(state, H, samples)
    expected = ref_sgd_step(state, Obar, Ebar, imag_time)
    np.testing.assert_allclose(step, expected, rtol=1e-10, atol=1e-12)


def test_default_solver_matches_reference_packing(x64):
    # The same auto_shift_eig instance applied to the reference-packed system
    # must reproduce get_step, including the reweighted Obar/Ebar factors.
    state = make_state("non_holomorphic")
    H = Ising(h=1.0)
    solver = auto_shift_eig()
    opt = SR(state, H, solver=solver)
    samples = make_samples(state, 16, reweight=True)
    step = np.asarray(opt.get_step(samples))
    Obar = ref_obar(state, samples)
    Ebar, _, _ = ref_energy_ebar(state, H, samples)
    expected = ref_solve(state, solver, Obar, Ebar)
    np.testing.assert_allclose(step, expected, rtol=1e-8, atol=1e-12)


def test_solver_receives_and_updates_x0(x64):
    # SR keeps an x0 buffer: it is passed to the solver as a keyword argument
    # and replaced by the returned step after each solve.
    captured = {}

    def probe(A, b, **kwargs):
        captured["keys"] = sorted(kwargs.keys())
        return kwargs["x0"] + 1

    state = make_state("holomorphic")
    opt = SR(state, Ising(h=1.0), solver=probe)
    step1 = np.asarray(opt.get_step(make_samples(state, 16, seed=0)))
    assert captured["keys"] == ["x0"]
    np.testing.assert_allclose(step1, np.ones(state.nparams))  # x0 starts at zero
    step2 = np.asarray(opt.get_step(make_samples(state, 16, seed=1)))
    np.testing.assert_allclose(step2, np.full(state.nparams, 2.0))


def test_spring_solver_receives_no_buffers(x64):
    # SPRING keeps its momentum in the phi buffer but does not forward it (or
    # any x0 guess) to the numerical solver.
    captured = {}

    def probe(A, b, **kwargs):
        captured["keys"] = sorted(kwargs.keys())
        return jnp.zeros(A.shape[1], A.dtype)

    state = make_state("holomorphic")
    opt = SPRING(state, Ising(h=1.0), solver=probe)
    opt.get_step(make_samples(state, 16))
    assert captured["keys"] == []


@pytest.mark.parametrize(
    "kind", ["real", "holomorphic", "non_holomorphic", "real_to_complex"]
)
def test_step_dtype_shape_and_replication(kind, x64):
    state = make_state(kind)
    opt = SR(state, Ising(h=1.0))
    step = opt.get_step(make_samples(state, 16))
    assert step.dtype == np.dtype(qtx.get_default_dtype())
    assert step.shape == (state.nparams,)
    assert step.sharding.is_fully_replicated
    assert np.all(np.isfinite(np.asarray(step)))


@pytest.mark.parametrize("cls", [SR, SPRING, MARCH, AdamSR])
def test_save_load_roundtrip(cls, tmp_path, x64):
    # The buffers written by save() must restore a fresh optimizer to the exact
    # same trajectory. This also pins the buffer dict layout, which is the
    # serialization format of previously saved files.
    state = make_state("holomorphic")
    H = Ising(h=1.0)
    solver = auto_shift_eig()
    opt = cls(state, H, solver=solver)
    opt.get_step(make_samples(state, 16, seed=0))
    file = tmp_path / "buffers.eqx"
    opt.save(file)

    opt2 = cls(state, H, solver=solver, file=file)
    samples = make_samples(state, 16, seed=1)
    step_a = np.asarray(opt.get_step(samples))
    step_b = np.asarray(opt2.get_step(samples))
    np.testing.assert_array_equal(step_a, step_b)
