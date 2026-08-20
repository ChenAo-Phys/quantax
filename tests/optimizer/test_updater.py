"""
Unit tests for the update-strategy layer in quantax/optimizer/updater.py and the
``diag_preconditioner`` handling in QNGD.solve_equation. These pin the pieces
that the end-to-end optimizer tests in test_sr.py only exercise indirectly: the
buffer contracts of each updater, which solver keyword arguments each one
forwards, the ``updater=`` composition entry point, and the two branches of the
preconditioner emulation.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
import quantax as qtx
from quantax.operator import Heisenberg
from quantax.optimizer import (
    Updater,
    PlainUpdater,
    SpringUpdater,
    MarchUpdater,
    AdamUpdater,
    SR,
    SPRING,
    MARCH,
    AdamSR,
    StochasticQNGD,
    EnergyGrad,
    auto_shift_eig,
    lsmr,
    sgd_solver,
    pinvh_solve,
)
from quantax.optimizer.qngd import _accepts_diag_preconditioner

from _dtype import use_dtype
from _refs import make_state, make_samples

# ====================================================================
# Base class and hyperparameter storage
# ====================================================================


def test_base_class_methods_are_abstract():
    base = Updater()
    with pytest.raises(NotImplementedError):
        base.init(4)
    with pytest.raises(NotImplementedError):
        base.update(lambda O, E, **k: E, jnp.zeros((2, 4)), jnp.zeros(2), {})


def test_hyperparameters_are_inspectable():
    s = SpringUpdater(0.8)
    assert s.mu == 0.8
    m = MarchUpdater(0.9, 0.99)
    assert (m.mu, m.beta) == (0.9, 0.99)
    a = AdamUpdater(0.91, 0.992)
    assert (a.mu, a.beta) == (0.91, 0.992)


# ====================================================================
# init: buffer contracts
# ====================================================================


@pytest.mark.parametrize(
    "updater,keys,ones_keys",
    [
        (PlainUpdater(), set(), set()),
        (SpringUpdater(), {"phi"}, set()),
        # MARCH's "v" (second-order momentum) starts at 1, not 0, so the
        # diag_preconditioner (v**0.25) is neutral on the first step.
        (MarchUpdater(), {"phi", "v"}, {"v"}),
        (AdamUpdater(), {"m", "v", "t"}, set()),
    ],
)
def test_init_buffer_keys_shapes_and_zero(updater, keys, ones_keys):
    nparams = 5
    bufs = updater.init(nparams)
    assert set(bufs) == keys
    for name, buf in bufs.items():
        expected = 1 if name in ones_keys else 0
        np.testing.assert_array_equal(np.asarray(buf), expected)
        if name == "t":
            assert buf.shape == ()
            assert jnp.issubdtype(buf.dtype, jnp.integer)
        else:
            assert buf.shape == (nparams,)
        assert buf.sharding.is_fully_replicated


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.complex128])
def test_init_buffer_dtypes(dtype, x64):
    # The "momentum-of-the-step" buffers follow the default dtype; the
    # second-order (squared-magnitude) buffers are always real.
    use_dtype(dtype)
    real = jnp.finfo(dtype).dtype
    assert SpringUpdater().init(3)["phi"].dtype == dtype
    march = MarchUpdater().init(3)
    assert march["phi"].dtype == dtype and march["v"].dtype == real
    adam = AdamUpdater().init(3)
    assert adam["m"].dtype == dtype and adam["v"].dtype == real


# ====================================================================
# update: solver-kwarg contract and momentum bookkeeping (isolated)
# ====================================================================


class _RecordingSolve:
    """A stub ``core_solve`` that records each call and returns a fixed step."""

    def __init__(self, step):
        self.step = jnp.asarray(step)
        self.calls = []

    def __call__(self, Obar, Ebar, **kwargs):
        self.calls.append({"Ebar": np.asarray(Ebar), "kwargs": dict(kwargs)})
        return self.step


def _synth(nsamples=6, nparams=4, seed=0):
    use_dtype(jnp.complex128)
    rng = np.random.default_rng(seed)
    cplx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    Obar = jnp.asarray(cplx(nsamples, nparams))
    Ebar = jnp.asarray(cplx(nsamples))
    return Obar, Ebar, nparams


def test_plain_update_solves_without_kwargs_and_keeps_no_buffers(x64):
    Obar, Ebar, nparams = _synth()
    g = np.arange(1, nparams + 1) + 0j
    solver = _RecordingSolve(g)
    upd = PlainUpdater()
    bufs = upd.init(nparams)
    step, bufs = upd.update(solver, Obar, Ebar, bufs)

    assert len(solver.calls) == 1
    assert solver.calls[0]["kwargs"] == {}  # plain solve, no warm-start
    np.testing.assert_array_equal(np.asarray(step), g)
    assert bufs == {}  # no persistent buffers


def test_spring_update_centers_ebar_and_accumulates_momentum(x64):
    Obar, Ebar, nparams = _synth()
    g = np.full(nparams, 0.5 + 0.5j)
    solver = _RecordingSolve(g)
    mu = 0.9
    upd = SpringUpdater(mu)
    bufs = upd.init(nparams)

    # first step: phi == 0, so Ebar is unchanged and step == g
    step1, bufs = upd.update(solver, Obar, Ebar, bufs)
    assert solver.calls[0]["kwargs"] == {}  # no preconditioner
    np.testing.assert_allclose(solver.calls[0]["Ebar"], np.asarray(Ebar))
    np.testing.assert_allclose(np.asarray(step1), g)
    np.testing.assert_allclose(np.asarray(bufs["phi"]), g)

    # second step: Ebar is shifted by -mu*(Obar@phi) and momentum adds mu*phi
    phi = np.asarray(step1)
    step2, bufs = upd.update(solver, Obar, Ebar, bufs)
    np.testing.assert_allclose(
        solver.calls[1]["Ebar"], np.asarray(Ebar) - mu * (np.asarray(Obar) @ phi)
    )
    np.testing.assert_allclose(np.asarray(step2), g + mu * phi)


def test_march_forwards_diag_preconditioner(x64):
    Obar, Ebar, nparams = _synth()
    solver = _RecordingSolve(np.ones(nparams) + 0j)
    upd = MarchUpdater()
    bufs = upd.init(nparams)
    upd.update(solver, Obar, Ebar, bufs)
    kwargs = solver.calls[0]["kwargs"]
    assert set(kwargs) == {"diag_preconditioner"}
    # first step: v == 1, so the preconditioner is neutral (1**0.25 + 1e-8)
    np.testing.assert_allclose(
        np.asarray(kwargs["diag_preconditioner"]), 1.0, atol=1e-7
    )


def test_adam_does_two_solves_with_preconditioner_on_the_second(x64):
    Obar, Ebar, nparams = _synth()
    g = np.full(nparams, 0.3 + 0.1j)
    solver = _RecordingSolve(g)
    mu = 0.95
    upd = AdamUpdater(mu, 0.995)
    bufs = upd.init(nparams)
    step, bufs = upd.update(solver, Obar, Ebar, bufs)

    assert len(solver.calls) == 2
    assert solver.calls[0]["kwargs"] == {}  # gradient solve, no preconditioner
    assert set(solver.calls[1]["kwargs"]) == {"diag_preconditioner"}
    # t advances, and on the first step mhat == g, so step == g + mhat == 2*g
    assert int(np.asarray(bufs["t"])) == 1
    np.testing.assert_allclose(np.asarray(step), 2 * g, rtol=1e-6)


# ====================================================================
# Composition: the updater= entry point on the drivers
# ====================================================================


@pytest.mark.parametrize(
    "public_cls,updater_factory",
    [
        (SR, lambda: PlainUpdater()),
        (lambda s, H, **kw: SPRING(s, H, mu=0.7, **kw), lambda: SpringUpdater(0.7)),
        (
            lambda s, H, **kw: MARCH(s, H, mu=0.8, beta=0.9, **kw),
            lambda: MarchUpdater(0.8, 0.9),
        ),
        (
            lambda s, H, **kw: AdamSR(s, H, mu=0.8, beta=0.9, **kw),
            lambda: AdamUpdater(0.8, 0.9),
        ),
    ],
)
def test_public_optimizer_equals_qngd_with_updater(public_cls, updater_factory, x64):
    # Each public optimizer must be exactly StochasticQNGD composed with the
    # corresponding updater (same solver, same samples => identical step).
    state = make_state("holomorphic")
    H = Heisenberg()
    solver = auto_shift_eig()
    samples = make_samples(state, 16)

    ref = np.asarray(public_cls(state, H, solver=solver).get_step(samples))
    composed = StochasticQNGD(
        state, EnergyGrad(H), solver=solver, updater=updater_factory()
    )
    got = np.asarray(composed.get_step(samples))
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize(
    "updater,keys",
    [
        (PlainUpdater(), set()),
        (SpringUpdater(), {"phi"}),
        (MarchUpdater(), {"phi", "v"}),
        (AdamUpdater(), {"m", "v", "t"}),
    ],
)
def test_driver_buffers_come_from_updater(updater, keys, x64):
    state = make_state("holomorphic")
    opt = StochasticQNGD(state, EnergyGrad(Heisenberg()), updater=updater)
    assert set(opt._buffers) == keys


# ====================================================================
# diag_preconditioner handling in solve_equation
# ====================================================================


def test_accepts_diag_preconditioner_detection():
    assert _accepts_diag_preconditioner(lsmr()) is True
    assert _accepts_diag_preconditioner(auto_shift_eig()) is False
    assert _accepts_diag_preconditioner(sgd_solver()) is False
    assert _accepts_diag_preconditioner(pinvh_solve()) is False


def _solve_equation_opt(solver):
    # A holomorphic state keeps the vs_type packing trivial, so solve_equation
    # is just the (preconditioned) solver call.
    state = make_state("holomorphic")
    opt = SR(state, Heisenberg(), solver=solver)
    rng = np.random.default_rng(0)
    nparams = state.nparams
    Obar = jnp.asarray(rng.standard_normal((8, nparams)) + 0j)
    Ebar = jnp.asarray(rng.standard_normal(8) + 0j)
    d = jnp.asarray(rng.uniform(0.5, 2.0, nparams))
    return opt, Obar, Ebar, d


def test_emulated_preconditioner_divides_columns_and_rescales(x64):
    # A solver without native support: solve_equation must divide Obar's columns
    # by d, NOT forward the kwarg, and rescale the solver output by 1/d.
    captured = {}

    def plain_solver(A, b, **kwargs):
        captured["A"] = np.asarray(A)
        captured["kwargs"] = dict(kwargs)
        return jnp.arange(1, A.shape[1] + 1).astype(A.dtype)

    assert _accepts_diag_preconditioner(plain_solver) is False
    opt, Obar, Ebar, d = _solve_equation_opt(auto_shift_eig())
    opt._solver = plain_solver
    step = np.asarray(opt.solve_equation(Obar, Ebar, diag_preconditioner=d))

    np.testing.assert_allclose(captured["A"], np.asarray(Obar) / np.asarray(d))
    assert "diag_preconditioner" not in captured["kwargs"]
    raw = np.arange(1, Obar.shape[1] + 1)
    np.testing.assert_allclose(step, raw / np.asarray(d), rtol=1e-12)


def test_native_preconditioner_is_passed_through_unscaled(x64):
    # A solver that declares diag_preconditioner receives Obar unchanged and the
    # preconditioner as a keyword; solve_equation does NOT divide/rescale itself.
    captured = {}

    def native_solver(A, b, *, diag_preconditioner=None, **kwargs):
        captured["A"] = np.asarray(A)
        captured["d"] = (
            None if diag_preconditioner is None else np.asarray(diag_preconditioner)
        )
        return jnp.zeros(A.shape[1], A.dtype)

    assert _accepts_diag_preconditioner(native_solver) is True
    opt, Obar, Ebar, d = _solve_equation_opt(auto_shift_eig())
    opt._solver = native_solver
    opt.solve_equation(Obar, Ebar, diag_preconditioner=d)
    np.testing.assert_array_equal(captured["A"], np.asarray(Obar))  # not divided
    np.testing.assert_allclose(captured["d"], np.asarray(d))


def test_no_preconditioner_leaves_obar_untouched(x64):
    captured = {}

    def plain_solver(A, b, **kwargs):
        captured["A"] = np.asarray(A)
        captured["kwargs"] = dict(kwargs)
        return jnp.zeros(A.shape[1], A.dtype)

    opt, Obar, Ebar, _ = _solve_equation_opt(auto_shift_eig())
    opt._solver = plain_solver
    opt.solve_equation(Obar, Ebar)
    np.testing.assert_array_equal(captured["A"], np.asarray(Obar))
    assert "diag_preconditioner" not in captured["kwargs"]
