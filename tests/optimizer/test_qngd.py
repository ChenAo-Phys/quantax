"""
Characterization tests for the shared QNGD mechanics in quantax/optimizer/qngd.py:
the Obar/Ebar estimation pipeline, the vs_type packing around the solver, the
handling of NaN Jacobian rows, and buffer serialization.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
import quantax as qtx
from quantax.operator import Ising
from quantax.sampler import Samples
from quantax.optimizer import SR, SPRING, MARCH, AdamSR, sgd_solver, auto_shift_eig
from quantax.utils import to_distributed_array

from _refs import (
    make_state,
    make_samples,
    ref_obar,
    ref_energy_ebar,
    ref_sgd_step,
    ref_solve,
)

NDEV = jax.device_count()
NS = -(-32 // NDEV) * NDEV  # samples per step, 32 rounded up to a multiple of NDEV


def _is_replicated(x):
    # an array on a single device also counts as fully replicated
    return x.sharding.is_fully_replicated and len(x.sharding.device_set) == NDEV


def _assert_step_layout(step, state):
    # the step follows the default dtype and is replicated on all devices
    assert step.dtype == np.dtype(qtx.get_default_dtype())
    assert step.shape == (state.nparams,)
    assert _is_replicated(step)


@pytest.mark.parametrize(
    "kind", ["real", "holomorphic", "non_holomorphic", "real_to_complex"]
)
def test_get_step_full_pipeline(kind, x64):
    # End-to-end pin of get_step = Ebar/Obar estimation + vs_type packing + solve +
    # unpacking. The closed-form sgd_solver x = A^† b / len(b) pins every branch of
    # the vs_type / imag_time packing in QNGD.solve_equation; real-time evolution of
    # a purely real state is ill-defined and not covered.
    state = make_state(kind)
    H = Ising(h=1.0)
    samples = make_samples(state, NS)
    Obar = ref_obar(state, samples)
    Ebar, _, _ = ref_energy_ebar(state, H, samples)
    for imag_time in (True,) if kind == "real" else (True, False):
        opt = SR(state, H, imag_time=imag_time, solver=sgd_solver())
        step = opt.get_step(samples)
        _assert_step_layout(step, state)
        expected = ref_sgd_step(state, Obar, Ebar, imag_time)
        np.testing.assert_allclose(np.asarray(step), expected, rtol=1e-10, atol=1e-12)

    # The same auto_shift_eig instance applied to the reference-packed system must
    # reproduce get_step, including the reweighted Obar/Ebar factors. Checked on the
    # most involved (real/imag stacked) packing only.
    if kind == "non_holomorphic":
        solver = auto_shift_eig()
        opt = SR(state, H, solver=solver)
        samples = make_samples(state, NS, reweight=True)
        step = opt.get_step(samples)
        _assert_step_layout(step, state)
        Obar = ref_obar(state, samples)
        Ebar, _, _ = ref_energy_ebar(state, H, samples)
        expected = ref_solve(state, solver, Obar, Ebar)
        np.testing.assert_allclose(np.asarray(step), expected, rtol=1e-8, atol=1e-12)


def test_get_obar_zeroes_nan_jacobian_entries(x64, monkeypatch):
    # NaN entries of the Jacobian (e.g. from a NaN psi sample) are zeroed before the
    # centering, with a warning counting the affected rows, so a single corrupted
    # sample can't turn the whole Obar into NaN. The mean runs over all samples.
    state = make_state("holomorphic")
    rng = np.random.default_rng(0)
    Omat = rng.normal(size=(NS, state.nparams)) + 1j * rng.normal(
        size=(NS, state.nparams)
    )
    Omat[3, 1] = np.nan
    Omat[7, :] = np.nan + 1j * np.nan
    monkeypatch.setattr(
        state, "jacobian", lambda s: to_distributed_array(jnp.asarray(Omat))
    )
    spins = to_distributed_array(jnp.ones((NS, state.Nmodes), jnp.int8))
    w = rng.uniform(0.5, 1.5, size=NS)
    samples = Samples(spins, r=to_distributed_array(jnp.asarray(w / w.mean())))

    opt = SR(state, Ising(h=1.0), solver=sgd_solver())
    with pytest.warns(UserWarning, match="2 NaN row"):
        Obar = np.asarray(opt.get_Obar(samples))
    O0 = np.where(np.isnan(Omat), 0, Omat)
    expected = (O0 - O0.mean(axis=0)) * np.sqrt(w / w.mean() / NS)[:, None]
    np.testing.assert_allclose(Obar, expected, rtol=1e-12, atol=1e-15)


def test_solve_replicates_distributed_solver_output(x64):
    # The parameters of the state are replicated, so the step is replicated on all
    # devices whatever the sharding of the solver output, e.g. of a solver
    # distributing its work, both in the eager solve_equation and the jitted solve.
    # Obar only needs a column count divisible by NDEV here.
    distributed = lambda A, b: to_distributed_array(A.T @ b)
    state = make_state("holomorphic")
    opt = SR(state, Ising(h=1.0), solver=distributed)
    Obar = jnp.ones((NS, 2 * NDEV), jnp.complex128)
    Ebar = jnp.ones(NS, jnp.complex128)
    assert NDEV == 1 or not _is_replicated(distributed(Obar, Ebar))
    step = opt.solve_equation(Obar, Ebar)
    assert _is_replicated(step)
    np.testing.assert_array_equal(np.asarray(step), NS)
    step, _ = opt.solve(Obar, Ebar, opt._buffers)
    assert _is_replicated(step)
    np.testing.assert_array_equal(np.asarray(step), NS)


def test_save_load_roundtrip(tmp_path, x64):
    # The buffers written by save() restore a fresh optimizer exactly, so it
    # continues the same trajectory: the step is a pure function of (Obar, Ebar,
    # buffers). This also pins the buffer dict layout, which is the serialization
    # format of previously saved files. SR keeps no buffers but must round-trip too.
    state = make_state("holomorphic")
    H = Ising(h=1.0)
    solver = auto_shift_eig()
    samples = make_samples(state, NS)
    for cls, keys in [
        (SR, set()),
        (SPRING, {"phi"}),
        (MARCH, {"phi", "v"}),
        (AdamSR, {"m", "v", "t"}),
    ]:
        opt = cls(state, H, solver=solver)
        assert set(opt._buffers) == keys
        if keys:
            opt.get_step(samples)  # non-trivial buffers
        file = tmp_path / f"{cls.__name__}.eqx"
        opt.save(file)

        loaded = cls(state, H, solver=solver, file=file)._buffers
        assert set(loaded) == keys
        for k in keys:
            assert loaded[k].dtype == opt._buffers[k].dtype
            assert _is_replicated(loaded[k])
            np.testing.assert_array_equal(np.asarray(loaded[k]), opt._buffers[k])
        if "t" in keys:
            assert int(loaded["t"]) == 1
