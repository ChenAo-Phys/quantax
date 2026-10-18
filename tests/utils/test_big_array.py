import pytest
import numpy as np
import jax
import jax.numpy as jnp
from quantax.utils import (
    LogArray,
    where,
    isnan,
    isinf,
    isfinite,
    to_distributed_array,
    to_replicated_array,
    get_distributed_sharding,
)

# Each test traces its LogArray ops into a single jitted program (see `_run`). Run
# eagerly, every op compiles its own XLA program for each new shape, which made these
# tiny tests compile-bound. The outputs are LogArrays returned through the jit
# boundary, so the pytree registration is exercised as well, and their dense values
# are computed in numpy and compared to the same operation on numpy arrays.

NDEV = jax.device_count()

# float32 / complex64 to match the default precision (x64 disabled in conftest)
REAL = np.array([[1.5, -2.0, 0.5], [3.0, -0.25, 4.0]], dtype=np.float32)
CPLX = np.array([1.0 + 2.0j, -3.0 + 0.0j, 0.5 - 1.5j], dtype=np.complex64)

RNG = np.random.default_rng(0)
REAL3 = RNG.normal(size=(2, 3, 4)).astype(np.float32)
CPLX3 = (REAL3 + 1j * RNG.normal(size=REAL3.shape)).astype(np.complex64)


def _run(f, *xs):
    """Evaluate ``f`` on ``LogArray.from_value`` of ``xs`` in one jitted program."""
    return jax.jit(lambda *xs: f(*map(LogArray.from_value, xs)))(*xs)


def dense(arr):
    # numpy instead of arr.value(), which would compile again for every shape
    sign, logabs = np.asarray(arr.sign), np.asarray(arr.logabs)
    assert sign.shape == logabs.shape
    return sign * np.exp(logabs)


def assert_value(arr, expected, msg=""):
    value, expected = dense(arr), np.asarray(expected)
    # assert_allclose broadcasts a 0-d side, e.g. () against (1, 1), so check the
    # shape (keepdims, reduced axes) explicitly
    assert value.shape == expected.shape, msg
    # log-space ops go through exp/log, so use float32-appropriate tolerances
    np.testing.assert_allclose(value, expected, rtol=1e-4, atol=1e-5, err_msg=msg)


def _check_ops(ops, *xs):
    """``ops`` maps names to ``(f, expected)``; all ``f(*LogArrays)`` share one jit."""
    outs = _run(lambda *a: [f(*a) for f, _ in ops.values()], *xs)
    for (name, (_, expected)), out in zip(ops.items(), outs):
        assert isinstance(out, LogArray), name
        assert_value(out, expected, name)


# ---------- construction / conversion ----------


def test_from_value_roundtrip():
    real, cplx, zero = _run(lambda r, c, z: (r, c, z), REAL, CPLX, 0.0)
    assert_value(real, REAL)
    assert_value(cplx, CPLX)
    # dtype promotes the sign (phase) with the real logabs
    assert real.dtype == jnp.float32 and cplx.dtype == jnp.complex64
    # zero is encoded as sign=0, logabs=-inf
    assert float(zero.sign) == 0.0 and float(zero.logabs) == -np.inf
    assert dense(zero) == 0.0

    # conversions to dense numpy (__array__) and jax (__jax_array__) arrays
    np.testing.assert_allclose(np.asarray(real), REAL, rtol=1e-6)
    out = jax.jit(lambda x: jnp.asarray(x) * 2)(real)
    np.testing.assert_allclose(np.asarray(out), 2 * REAL, rtol=1e-6)


def test_idempotent_from_value():
    # a LogArray is returned unchanged, without the dense round trip that would
    # overflow for e.g. logabs = 1e4
    arr = LogArray(sign=jnp.ones(2), logabs=jnp.full(2, 1e4))
    assert LogArray.from_value(arr) is arr


# ---------- elementwise ops ----------


def test_unary_ops():
    ops = {
        "neg": (lambda r, c, r3: -r, -REAL),
        "conj": (lambda r, c, r3: c.conj(), CPLX.conj()),
        "abs": (lambda r, c, r3: r.abs(), np.abs(REAL)),
        "abs complex": (lambda r, c, r3: abs(c), np.abs(CPLX)),
        "real": (lambda r, c, r3: c.real, CPLX.real),
        "imag": (lambda r, c, r3: c.imag, CPLX.imag),
        "real of real": (lambda r, c, r3: r.real, REAL),
        "imag of real": (lambda r, c, r3: r.imag, np.zeros_like(REAL)),
        "T": (lambda r, c, r3: r.T, REAL.T),
        "mT": (lambda r, c, r3: r3.mT, np.swapaxes(REAL3, -1, -2)),
    }
    _check_ops(ops, REAL, CPLX, REAL3)

    cast = _run(lambda r: r.astype(jnp.complex64), REAL)
    assert cast.dtype == jnp.complex64 and cast.logabs.dtype == jnp.float32
    assert_value(cast, REAL)


def test_binary_ops():
    # LogArray (op) LogArray, LogArray (op) scalar and the reflected scalar (op)
    # LogArray, which dispatches to __rmul__ / __rtruediv__ / __radd__ / __rsub__
    R1 = REAL + 1.0
    ops = {
        "mul": (lambda a, b: a * b, REAL * R1),
        "mul scalar": (lambda a, b: a * 2.0, REAL * 2.0),
        "rmul": (lambda a, b: 2.0 * a, REAL * 2.0),
        "truediv": (lambda a, b: a / b, REAL / R1),
        "truediv scalar": (lambda a, b: a / 2.0, REAL / 2.0),
        "rtruediv": (lambda a, b: 6.0 / a, 6.0 / REAL),
        "add": (lambda a, b: a + b, REAL + R1),
        "add scalar": (lambda a, b: a + 2.0, REAL + 2.0),
        "radd": (lambda a, b: 2.0 + a, REAL + 2.0),
        "sub": (lambda a, b: a - b, REAL - R1),
        "sub scalar": (lambda a, b: a - 2.0, REAL - 2.0),
        "rsub": (lambda a, b: 5.0 - a, 5.0 - REAL),
        # the left operand is the LogArray, so the representation is preserved
        "add jax array": (lambda a, b: a + jnp.asarray(REAL), 2 * REAL),
        "pow 2": (lambda a, b: a**2, REAL**2),
        "pow 3": (lambda a, b: a**3, REAL**3),
    }
    _check_ops(ops, REAL, R1)


def test_complex_binary_ops():
    # complex phases multiply and add correctly in the log representation
    C2 = CPLX[::-1] * 0.5j
    ops = {
        "mul": (lambda a, b: a * b, CPLX * C2),
        "truediv": (lambda a, b: a / b, CPLX / C2),
        "add": (lambda a, b: a + b, CPLX + C2),
        "sub": (lambda a, b: a - b, CPLX - C2),
        "pow 2": (lambda a, b: a**2, CPLX**2),
    }
    _check_ops(ops, CPLX, C2)


def test_pow_zero_is_one():
    # x**0 = 1 for every x as in numpy, also for zero, inf and nan, where
    # logabs * 0 would be nan.
    x = np.array([0.0, -2.0, np.inf, np.nan], np.float32)
    out = _run(lambda a: a**0.0, x)
    np.testing.assert_array_equal(dense(out), x**0)


# ---------- reductions ----------


def test_reductions():
    # The mean is the sum with log(N) subtracted, where N is the size of the reduced
    # axes, which must be counted correctly for negative and tuple axes.
    ops = {}
    for k in (False, True):
        for axis in (None, 0, 1):
            ops[f"sum {axis=} {k=}"] = (
                lambda r, c, r3, c3, axis=axis, k=k: r.sum(axis=axis, keepdims=k),
                REAL.sum(axis=axis, keepdims=k),
            )
            ops[f"prod {axis=} {k=}"] = (
                lambda r, c, r3, c3, axis=axis, k=k: r.prod(axis=axis, keepdims=k),
                REAL.prod(axis=axis, keepdims=k),
            )
        for axis in (None, 0, 1, -1, (0, 2), (-1, 0), (1, -1)):
            for i, x in ((2, REAL3), (3, CPLX3)):
                ops[f"mean {x.dtype} {axis=} {k=}"] = (
                    lambda *a, i=i, axis=axis, k=k: a[i].mean(axis=axis, keepdims=k),
                    np.mean(x, axis=axis, keepdims=k),
                )
    ops["sum complex"] = (lambda r, c, r3, c3: c.sum(), CPLX.sum())
    _check_ops(ops, REAL, CPLX, REAL3, CPLX3)


def test_reductions_special_values():
    # dense values would overflow (e^700 ~ 1e304), but the log representation keeps
    # the exact log-magnitude
    logabs = (700.0 + np.log(np.arange(1.0, 4.0))).astype(np.float32)
    arr = LogArray(sign=np.ones(3, np.float32), logabs=logabs)
    # sums of zeros (logabs = -inf everywhere, e.g. a symmetrized amplitude whose
    # images all vanish) stay zero instead of exp(-inf - (-inf)) = nan, and +inf ties
    # stay inf
    infs = LogArray(sign=np.ones(2, np.float32), logabs=np.full(2, np.inf, np.float32))

    @jax.jit
    def f(arr, infs):
        zeros = LogArray.from_value(jnp.zeros(3))
        return arr.mean(), arr.prod(), arr + arr, zeros.sum(), zeros + zeros, infs.sum()

    mean, prod, add, zero_sum, zero_add, inf_sum = f(arr, infs)
    # float32 rounding of the log-magnitudes ~1e3 is ~1e-7 relative
    np.testing.assert_allclose(float(mean.logabs), 700.0 + np.log(2.0), rtol=1e-6)
    assert float(mean.sign) == 1.0
    np.testing.assert_allclose(float(prod.logabs), 2100.0 + np.log(6.0), rtol=1e-6)
    np.testing.assert_allclose(add.logabs, logabs + np.log(2.0), rtol=1e-6)
    assert dense(zero_sum) == 0.0
    np.testing.assert_array_equal(dense(zero_add), [0.0, 0.0, 0.0])
    assert float(inf_sum.logabs) == np.inf and float(inf_sum.sign) == 1.0


# ---------- generated array methods ----------


def test_array_methods():
    ops = {
        "reshape": (lambda r: r.reshape(6), REAL.reshape(6)),
        "flatten": (lambda r: r.flatten(), REAL.flatten()),
        "transpose": (lambda r: r.transpose(), REAL.transpose()),
        "getitem row": (lambda r: r[0], REAL[0]),
        "getitem column": (lambda r: r[:, 1], REAL[:, 1]),
        "take": (lambda r: r.take(np.array([0, 2]), axis=1), REAL.take([0, 2], 1)),
        "swapaxes": (lambda r: r.swapaxes(0, 1), REAL.swapaxes(0, 1)),
        "repeat": (lambda r: r.repeat(2, axis=0), REAL.repeat(2, axis=0)),
    }
    _check_ops(ops, REAL)


# ---------- special values ----------


def test_predicates_dispatch():
    # The module-level predicates give consistent answers for plain arrays and
    # LogArray. NaN in either component is NaN; zero (sign=0, logabs=-inf) is finite;
    # overflow is an infinite sign or logabs=+inf.
    # value pattern: [normal, nan, nan, inf, inf, zero]
    log = LogArray(
        sign=jnp.asarray([1.0, jnp.nan, 1.0, 1.0, jnp.inf, 0.0]),
        logabs=jnp.asarray([0.0, 0.0, jnp.nan, jnp.inf, 0.0, -jnp.inf]),
    )
    plain = jnp.asarray([1.0, jnp.nan, jnp.nan, jnp.inf, jnp.inf, 0.0])
    preds = jax.jit(lambda *a: [(isnan(x), isinf(x), isfinite(x)) for x in a])
    for nan, inf, finite in preds(log, plain):
        np.testing.assert_array_equal(nan, [False, True, True, False, False, False])
        np.testing.assert_array_equal(inf, [False, False, False, True, True, False])
        np.testing.assert_array_equal(finite, [True, False, False, False, False, True])


# ---------- where ----------


def test_where():
    # A plain operand on either side is converted, so the result stays a LogArray.
    cond = np.array([True, False, True])
    x = np.array([1.0, 2.0, 3.0], np.float32)
    y = np.array([4.0, 5.0, 6.0], np.float32)

    def f(x, y):
        log_x, log_y = LogArray.from_value(x), LogArray.from_value(y)
        outs = [where(cond, log_x, log_y), where(cond, log_x, y)]
        return outs + [where(cond, x, log_y), where(cond, x, y)]

    *logs, plain = jax.jit(f)(x, y)
    for out in logs:
        assert isinstance(out, LogArray)
        assert_value(out, [1.0, 5.0, 3.0])
    assert isinstance(plain, jax.Array)
    np.testing.assert_array_equal(plain, [1.0, 5.0, 3.0])
    # numpy operands stay on the host
    out = where(cond, x, y)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, [1.0, 5.0, 3.0])


# ---------- sharding ----------


def test_sharding():
    shape = (2 * NDEV, 2)
    sign, logabs = np.ones(shape, np.float32), np.zeros(shape, np.float32)
    arr = LogArray(to_distributed_array(sign), to_distributed_array(logabs))
    assert arr.sharding.is_equivalent_to(get_distributed_sharding(), 2)
    if NDEV > 1:
        # mismatched layouts of the two leaves have no single sharding
        arr = LogArray(to_distributed_array(sign), to_replicated_array(logabs))
        with pytest.raises(ValueError, match="same sharding"):
            arr.sharding


# ---------- complex phase gradient ----------
# LogArray extracts the wavefunction phase with `_phase` (= x/|x|). `jnp.sign` has
# a *zero* JVP for complex inputs, which would silently discard the phase gradient
# of complex amplitudes; these guard the custom-JVP fix.


def test_logarray_phase_gradient():
    # d arg(x/|x|) / d(Re, Im) = (-Im, Re) / |x|^2 for complex x.
    def phase_angle(zr, zi):
        return jnp.angle(LogArray.from_value(zr + 1j * zi).sign)

    # On the real axis the sign is locally constant (+-1), so its gradient is zero,
    # matching jnp.sign and avoiding spurious/NaN gradients.
    real_sign = lambda x: LogArray.from_value(x).sign

    @jax.jit
    def grads(zr, zi, x):
        grad_cplx = jax.grad(phase_angle, argnums=(0, 1))(zr, zi)
        return grad_cplx, jax.vmap(jax.grad(real_sign))(x)

    grad_cplx, grad_real = grads(1.5, 0.7, jnp.asarray([2.0, -3.0]))
    denom = 1.5**2 + 0.7**2
    np.testing.assert_allclose(grad_cplx, (-0.7 / denom, 1.5 / denom), rtol=1e-3)
    np.testing.assert_array_equal(grad_real, [0.0, 0.0])


def test_logarray_sum_preserves_complex_phase_gradient():
    # The log-space sum (used by symmetrization) must propagate the phase gradient.
    # Multiplying every term by exp(i t) rotates the sum's phase by t, so d/dt = 1.
    def phase_of_sum(t):
        terms = jnp.array([1.0 + 1.0j, 2.0 - 0.5j, -0.5 + 0.3j]) * jnp.exp(1j * t)
        return jnp.angle(LogArray.from_value(terms).sum().sign)

    grad = jax.jit(jax.grad(phase_of_sum))(0.3)
    np.testing.assert_allclose(float(grad), 1.0, rtol=1e-3)


def test_logarray_add_preserves_complex_phase_gradient():
    # Regression: __add__ took the sign of the sum with jnp.sign, whose zero JVP for
    # complex inputs dropped the phase gradient (d/dt Im log z was 0).
    # z(t) = 2 exp(i t) + 1 and d/dt log z = 2i exp(i t) / z.
    ops = {
        "add": lambda a: a + 1.0,
        "radd": lambda a: 1.0 + a,
        "sub": lambda a: a - (-1.0),
        "rsub": lambda a: 1.0 - (-a),
    }

    def log_z(t):
        a = LogArray.from_value(2.0 * jnp.exp(1j * t))
        zs = [op(a) for op in ops.values()]
        return jnp.stack([jnp.stack([z.logabs, jnp.angle(z.sign)]) for z in zs])

    t = 0.7
    out, grad = jax.jit(lambda t: (log_z(t), jax.jacrev(log_z)(t)))(t)
    z = 2 * np.exp(1j * t) + 1
    dlog_z = 2j * np.exp(1j * t) / z
    for name, o, g in zip(ops, out, grad):
        np.testing.assert_allclose(o, [np.log(abs(z)), np.angle(z)], 1e-6, err_msg=name)
        np.testing.assert_allclose(g, [dlog_z.real, dlog_z.imag], 1e-5, err_msg=name)


# ---------- gradients at ties of the maximum ----------
# sum and __add__ shift by the largest logabs, which must not carry a gradient.
# Equal logabs are common, e.g. symmetry images of a symmetric configuration.


def _log_psi(y):
    # surrogate of log(psi) whose gradient is (1/psi) dpsi, as in Variational
    return y.sign / jax.lax.stop_gradient(y.sign) + y.logabs


def test_logarray_sum_gradient_at_ties():
    s = jnp.asarray([1.0, -1.0, 1.0])
    t = jnp.asarray([0.3, 0.3, -0.5])
    grad = jax.jit(jax.grad(lambda t: _log_psi(LogArray(s, t).sum())))(t)
    v = np.asarray(s) * np.exp(np.asarray(t, np.float64))
    np.testing.assert_allclose(grad, v / v.sum(), rtol=1e-5)


def test_logarray_complex_gradient_at_ties():
    c = jnp.asarray([1.0, 1.0j], jnp.complex64)

    def log_psi(t):
        y = LogArray(c, t)
        out = jnp.stack([_log_psi(y.sum()), _log_psi(y[0] + y[1])])
        return jnp.stack([out.real, out.imag])

    t = jnp.asarray([0.7, 0.7])
    jac = jax.jit(jax.jacrev(log_psi))(t)  # (real/imag, sum/add, t)
    v = np.asarray(c) * np.exp(0.7)
    for i, op in enumerate(("sum", "add")):
        grad = jac[0, i] + 1j * jac[1, i]
        np.testing.assert_allclose(grad, v / v.sum(), rtol=1e-5, err_msg=op)


# ---------- abs must not hide non-finite values ----------


def test_logarray_abs_keeps_nonfinite():
    # Regression: ``abs`` sets the sign to 1. It used to do so unconditionally,
    # turning a NaN / Inf sign with a finite log-magnitude into a finite value that
    # isnan / isinf / isfinite could no longer detect. A NaN / +inf logabs is kept
    # as well.
    arr = LogArray(
        sign=jnp.asarray([1.0, -1.0, jnp.nan, jnp.inf, 0.0, -1.0, -1.0]),
        logabs=jnp.asarray([0.0, 0.5, 0.0, 0.0, -jnp.inf, jnp.nan, jnp.inf]),
    )

    @jax.jit
    def f(arr):
        a = abs(arr)
        return a, a.isnan(), a.isinf(), a.isfinite()

    a, nan, inf, finite = f(arr)
    np.testing.assert_array_equal(nan, [0, 0, 1, 0, 0, 1, 0])
    np.testing.assert_array_equal(inf, [0, 0, 0, 1, 0, 0, 1])
    np.testing.assert_array_equal(finite, [1, 1, 0, 0, 1, 0, 0])
    # valid entries keep the exact unit sign and the value |x|
    np.testing.assert_array_equal(np.asarray(a.sign)[[0, 1, 4]], [1.0, 1.0, 1.0])
    assert_value(a[np.array([0, 1, 4])], [1.0, np.exp(0.5), 0.0])


def test_logarray_abs_keeps_nan_phase():
    # A complex phase head producing NaN on one sample while the log-magnitude head
    # is finite: the NaN must survive abs() and propagate through later arithmetic.
    theta = jnp.asarray([0.3, jnp.nan, 1.2])
    arr = LogArray(sign=jnp.exp(1j * theta), logabs=jnp.asarray([0.0, 0.5, -1.0]))

    @jax.jit
    def f(arr):
        a = abs(arr)
        ratio = a**2 / a  # the reweighting-style arithmetic of the samplers
        return a, a.isfinite(), isfinite(ratio), isfinite(ratio.mean())

    a, finite, finite_ratio, finite_mean = f(arr)
    assert jnp.issubdtype(a.sign.dtype, jnp.floating)  # abs is real
    np.testing.assert_array_equal(finite, [True, False, True])
    np.testing.assert_array_equal(np.asarray(a.sign)[[0, 2]], [1.0, 1.0])
    np.testing.assert_array_equal(finite_ratio, [True, False, True])
    np.testing.assert_array_equal(finite_mean, False)
