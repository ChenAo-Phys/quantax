import pytest
import numpy as np
import jax
import jax.numpy as jnp
from quantax.utils import LogArray, where, isnan, isinf, isfinite

# value() / np.asarray() materialize the dense array, which is compared to the
# equivalent operation on a plain numpy array.

# float32 / complex64 to match the default precision (x64 disabled in conftest)
REAL = np.array([[1.5, -2.0, 0.5], [3.0, -0.25, 4.0]], dtype=np.float32)
CPLX = np.array([1.0 + 2.0j, -3.0 + 0.0j, 0.5 - 1.5j], dtype=np.complex64)


def make(x):
    return LogArray.from_value(jnp.asarray(x))


def dense(arr):
    return np.asarray(arr.value())


def assert_value(arr, expected):
    # log-space ops go through exp/log, so use float32-appropriate tolerances
    np.testing.assert_allclose(dense(arr), np.asarray(expected), rtol=1e-4, atol=1e-5)


# ---------- construction / conversion ----------


def test_from_value_roundtrip_real():
    assert_value(make(REAL), REAL)


def test_from_value_roundtrip_complex():
    assert_value(make(CPLX), CPLX)


def test_idempotent_from_value():
    arr = make(REAL)
    assert LogArray.from_value(arr) is arr


def test_numpy_and_jax_conversion():
    arr = make(REAL)
    np.testing.assert_allclose(np.asarray(arr), REAL)
    np.testing.assert_allclose(np.asarray(jnp.asarray(arr)), REAL)


def test_properties():
    arr = make(REAL)
    assert arr.shape == REAL.shape
    assert arr.ndim == REAL.ndim
    assert arr.size == REAL.size
    assert jnp.issubdtype(arr.dtype, jnp.floating)
    assert jnp.issubdtype(make(CPLX).dtype, jnp.complexfloating)


# ---------- unary ops ----------


def test_neg():
    assert_value(-make(REAL), -REAL)


def test_conj():
    assert_value(make(CPLX).conj(), CPLX.conj())


def test_abs():
    assert_value(make(REAL).abs(), np.abs(REAL))
    assert_value(abs(make(CPLX)), np.abs(CPLX))


def test_real_imag_complex():
    arr = make(CPLX)
    assert_value(arr.real, CPLX.real)
    assert_value(arr.imag, CPLX.imag)


def test_real_imag_real():
    arr = make(REAL)
    assert_value(arr.real, REAL)
    assert_value(arr.imag, np.zeros_like(REAL))


def test_transpose_property():
    assert_value(make(REAL).T, REAL.T)


def test_astype():
    arr = make(REAL).astype(jnp.complex64)
    assert jnp.issubdtype(arr.dtype, jnp.complexfloating)
    assert_value(arr, REAL)


# ---------- binary ops ----------


def test_mul():
    a, b = make(REAL), make(REAL + 1.0)
    assert_value(a * b, REAL * (REAL + 1.0))
    assert_value(make(REAL) * 2.0, REAL * 2.0)
    assert_value(2.0 * make(REAL), REAL * 2.0)  # __rmul__


def test_truediv():
    a, b = make(REAL), make(REAL + 1.0)
    assert_value(a / b, REAL / (REAL + 1.0))
    assert_value(make(REAL) / 2.0, REAL / 2.0)
    assert_value(6.0 / make(REAL), 6.0 / REAL)  # __rtruediv__


@pytest.mark.parametrize("p", [2, 3])
def test_pow(p):
    assert_value(make(REAL) ** p, REAL**p)


def test_add():
    a, b = make(REAL), make(REAL + 1.0)
    assert_value(a + b, REAL + (REAL + 1.0))
    assert_value(make(REAL) + 2.0, REAL + 2.0)
    assert_value(2.0 + make(REAL), REAL + 2.0)  # __radd__


def test_sub():
    a, b = make(REAL), make(REAL + 1.0)
    assert_value(a - b, REAL - (REAL + 1.0))
    assert_value(make(REAL) - 2.0, REAL - 2.0)
    assert_value(5.0 - make(REAL), 5.0 - REAL)  # __rsub__


def test_add_with_jax_array():
    # left operand is the custom type, so the representation is preserved
    res = make(REAL) + jnp.asarray(REAL)
    assert isinstance(res, LogArray)
    assert_value(res, 2 * REAL)


# ---------- reductions ----------


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_sum(axis, keepdims):
    arr = make(REAL).sum(axis=axis, keepdims=keepdims)
    assert_value(arr, REAL.sum(axis=axis, keepdims=keepdims))


RNG = np.random.default_rng(0)
REAL3 = RNG.normal(size=(2, 3, 4)).astype(np.float32)
CPLX3 = (REAL3 + 1j * RNG.normal(size=REAL3.shape)).astype(np.complex64)


@pytest.mark.parametrize("x", [REAL3, CPLX3], ids=["real", "complex"])
@pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 2), (-1, 0), (1, -1)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_mean(x, axis, keepdims):
    # The mean is the sum with log(N) subtracted, where N is the size of the reduced
    # axes, which must be counted correctly for negative and tuple axes.
    arr = make(x).mean(axis=axis, keepdims=keepdims)
    expected = np.asarray(jnp.mean(jnp.asarray(x), axis=axis, keepdims=keepdims))
    assert arr.shape == expected.shape
    assert_value(arr, expected)


def test_mean_no_overflow():
    # mean(e^700 * [1, 2, 3]) = 2 e^700 overflows the dense value, but not the logabs
    arr = LogArray(sign=jnp.ones(3), logabs=700.0 + jnp.log(jnp.arange(1.0, 4.0)))
    np.testing.assert_allclose(float(arr.mean().logabs), 700.0 + np.log(2.0))
    assert float(arr.mean().sign) == 1.0


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_prod(axis):
    assert_value(make(REAL).prod(axis=axis), REAL.prod(axis=axis))


def test_sum_complex():
    assert_value(make(CPLX).sum(), CPLX.sum())


# ---------- generated array methods ----------


def test_reshape_flatten():
    assert_value(make(REAL).reshape(6), REAL.reshape(6))
    assert_value(make(REAL).flatten(), REAL.flatten())


def test_transpose_method():
    assert_value(make(REAL).transpose(), REAL.transpose())


def test_getitem():
    assert_value(make(REAL)[0], REAL[0])
    assert_value(make(REAL)[:, 1], REAL[:, 1])


def test_take():
    arr = make(REAL).take(np.array([0, 2]), axis=1)
    assert_value(arr, REAL.take([0, 2], axis=1))


# ---------- pytree / jit ----------


def test_pytree_flatten_unflatten():
    arr = make(REAL)
    leaves, treedef = jax.tree.flatten(arr)
    assert len(leaves) == 2
    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert isinstance(rebuilt, LogArray)
    assert_value(rebuilt, REAL)


def test_pytree_map():
    # scaling both leaves is not the same as scaling the value, but it must stay a
    # valid pytree of the same type with the leaves transformed
    arr = make(REAL)
    doubled = jax.tree.map(lambda x: x * 2, arr)
    assert isinstance(doubled, LogArray)
    leaves = jax.tree.leaves(arr)
    np.testing.assert_allclose(
        np.asarray(jax.tree.leaves(doubled)[0]), 2 * np.asarray(leaves[0])
    )


def test_jit():
    @jax.jit
    def f(a):
        return (a * a).value()

    out = f(make(REAL))
    np.testing.assert_allclose(np.asarray(out), REAL**2, rtol=1e-4)


# ---------- special values / stability ----------


def test_logarray_zero_encoding():
    z = LogArray.from_value(jnp.asarray(0.0))
    assert float(z.sign) == 0.0
    assert float(z.logabs) == -np.inf
    assert float(z.value()) == 0.0


def test_logarray_isnan():
    # NaN in either component is NaN; zero (sign=0, logabs=-inf) and overflow
    # (logabs=+inf) are not.
    arr = LogArray(
        sign=jnp.asarray([1.0, jnp.nan, 1.0, 0.0, 1.0]),
        logabs=jnp.asarray([0.0, 0.0, jnp.nan, -jnp.inf, jnp.inf]),
    )
    np.testing.assert_array_equal(
        np.asarray(arr.isnan()), [False, True, True, False, False]
    )


def test_logarray_stability():
    # dense values would overflow (exp(700) ~ 1e304, product over 5 -> inf), but the
    # log representation keeps the exact log-magnitude
    big = LogArray(sign=jnp.ones(5), logabs=jnp.full(5, 700.0))
    prod = big.prod()
    np.testing.assert_allclose(np.asarray(prod.logabs), 3500.0)
    assert float(jnp.exp(jnp.asarray(700.0)) ** 5) == np.inf  # dense overflows


# ---------- where ----------


def test_where_logarray():
    cond = np.array([True, False, True])
    x = LogArray.from_value(jnp.asarray([1.0, 2.0, 3.0]))
    y = LogArray.from_value(jnp.asarray([4.0, 5.0, 6.0]))
    res = where(cond, x, y)
    assert isinstance(res, LogArray)
    assert_value(res, [1.0, 5.0, 3.0])


def test_where_mixed_promotes_to_logarray():
    # A plain operand on either side is converted, so the result stays a LogArray.
    cond = np.array([True, False, True])
    log = LogArray.from_value(jnp.asarray([1.0, 2.0, 3.0]))
    plain = jnp.asarray([4.0, 5.0, 6.0])
    res = where(cond, log, plain)
    assert isinstance(res, LogArray)
    assert_value(res, [1.0, 5.0, 3.0])
    res = where(cond, plain, log)
    assert isinstance(res, LogArray)
    assert_value(res, [4.0, 2.0, 6.0])


def test_where_plain_array():
    cond = np.array([True, False, True])
    res = where(cond, jnp.asarray([1.0, 2.0, 3.0]), jnp.asarray([4.0, 5.0, 6.0]))
    assert isinstance(res, jax.Array)
    np.testing.assert_allclose(np.asarray(res), [1.0, 5.0, 3.0])


# ---------- isnan / isinf / isfinite dispatch ----------


def test_predicates_dispatch():
    # The module-level predicates give consistent answers for plain arrays and
    # LogArray; value pattern: [normal, nan, inf, zero].
    arrays = [
        jnp.asarray([1.0, jnp.nan, jnp.inf, 0.0]),
        LogArray(
            sign=jnp.asarray([1.0, jnp.nan, 1.0, 0.0]),
            logabs=jnp.asarray([0.0, 0.0, jnp.inf, -jnp.inf]),
        ),
    ]
    for arr in arrays:
        np.testing.assert_array_equal(
            np.asarray(isnan(arr)), [False, True, False, False]
        )
        np.testing.assert_array_equal(
            np.asarray(isinf(arr)), [False, False, True, False]
        )
        np.testing.assert_array_equal(
            np.asarray(isfinite(arr)), [True, False, False, True]
        )


# ---------- complex phase gradient ----------
# LogArray extracts the wavefunction phase with `_phase` (= x/|x|). `jnp.sign` has
# a *zero* JVP for complex inputs, which would silently discard the phase gradient
# of complex amplitudes; these guard the custom-JVP fix.


def test_logarray_complex_phase_has_correct_gradient():
    # d arg(x/|x|) / d(Re, Im) = (-Im, Re) / |x|^2 for complex x.
    def phase_angle(zr, zi):
        return jnp.angle(LogArray.from_value(zr + 1j * zi).sign)

    grad = jax.grad(phase_angle, argnums=(0, 1))(1.5, 0.7)
    denom = 1.5**2 + 0.7**2
    np.testing.assert_allclose(grad, (-0.7 / denom, 1.5 / denom), rtol=1e-3)


def test_logarray_real_sign_gradient_is_zero():
    # On the real axis the sign is locally constant (+-1), so its gradient is zero,
    # matching jnp.sign and avoiding spurious/NaN gradients.
    assert float(jax.grad(lambda x: LogArray.from_value(x).sign)(2.0)) == 0.0
    assert float(jax.grad(lambda x: LogArray.from_value(x).sign)(-3.0)) == 0.0


def test_logarray_sum_preserves_complex_phase_gradient():
    # The log-space sum (used by symmetrization) must propagate the phase gradient.
    # Multiplying every term by exp(i t) rotates the sum's phase by t, so d/dt = 1.
    def phase_of_sum(t):
        terms = jnp.array([1.0 + 1.0j, 2.0 - 0.5j, -0.5 + 0.3j]) * jnp.exp(1j * t)
        return jnp.angle(LogArray.from_value(terms).sum().sign)

    np.testing.assert_allclose(float(jax.grad(phase_of_sum)(0.3)), 1.0, rtol=1e-3)


@pytest.mark.parametrize("op", ["add", "radd", "sub", "rsub"])
def test_logarray_add_preserves_complex_phase_gradient(op):
    # Regression: __add__ took the sign of the sum with jnp.sign, whose zero JVP for
    # complex inputs dropped the phase gradient (d/dt Im log z was 0).
    # z(t) = 2 exp(i t) + 1 and d/dt log z = 2i exp(i t) / z.
    def log_z(t, part):
        a = LogArray.from_value(2.0 * jnp.exp(1j * t))
        if op == "add":
            z = a + 1.0
        elif op == "radd":
            z = 1.0 + a
        elif op == "sub":
            z = a - (-1.0)
        else:
            z = 1.0 - (-a)
        return jnp.angle(z.sign) if part == "imag" else z.logabs

    t = 0.7
    z = 2 * np.exp(1j * t) + 1
    np.testing.assert_allclose(float(log_z(t, "real")), np.log(abs(z)), rtol=1e-6)
    np.testing.assert_allclose(float(log_z(t, "imag")), np.angle(z), rtol=1e-6)
    dlog_z = 2j * np.exp(1j * t) / z
    np.testing.assert_allclose(
        float(jax.grad(log_z)(t, "real")), dlog_z.real, rtol=1e-5
    )
    np.testing.assert_allclose(
        float(jax.grad(log_z)(t, "imag")), dlog_z.imag, rtol=1e-5
    )


# ---------- gradients at ties of the maximum ----------
# sum and __add__ shift by the largest logabs, which must not carry a gradient.
# Equal logabs are common, e.g. symmetry images of a symmetric configuration.


def _log_psi(y):
    # surrogate of log(psi) whose gradient is (1/psi) dpsi, as in Variational
    return y.sign / jax.lax.stop_gradient(y.sign) + y.logabs


def test_logarray_sum_gradient_at_ties():
    s = jnp.asarray([1.0, -1.0, 1.0])
    t = jnp.asarray([0.3, 0.3, -0.5])
    grad = jax.grad(lambda t: _log_psi(LogArray(s, t).sum()))(t)
    v = np.asarray(s) * np.exp(np.asarray(t, np.float64))
    np.testing.assert_allclose(grad, v / v.sum(), rtol=1e-5)


@pytest.mark.parametrize("op", ["sum", "add"])
def test_logarray_complex_gradient_at_ties(op):
    c = jnp.asarray([1.0, 1.0j], jnp.complex64)

    def log_psi(t, part):
        y = LogArray(c, t)
        z = y.sum() if op == "sum" else y[0] + y[1]
        out = _log_psi(z)
        return out.real if part == "real" else out.imag

    t = jnp.asarray([0.7, 0.7])
    grad = jax.grad(log_psi)(t, "real") + 1j * jax.grad(log_psi)(t, "imag")
    v = np.asarray(c) * np.exp(0.7)
    np.testing.assert_allclose(grad, v / v.sum(), rtol=1e-5)


# ---------- abs must not hide non-finite values ----------


def test_logarray_abs_keeps_nonfinite_sign():
    # Regression: ``abs`` sets the sign to 1. It used to do so unconditionally,
    # turning a NaN / Inf sign with a finite log-magnitude into a finite value that
    # isnan / isinf / isfinite could no longer detect.
    arr = LogArray(
        sign=jnp.asarray([1.0, -1.0, jnp.nan, jnp.inf, 0.0]),
        logabs=jnp.asarray([0.0, 0.5, 0.0, 0.0, -jnp.inf]),
    )
    a = abs(arr)
    np.testing.assert_array_equal(
        np.asarray(a.isnan()), [False, False, True, False, False]
    )
    np.testing.assert_array_equal(
        np.asarray(a.isinf()), [False, False, False, True, False]
    )
    np.testing.assert_array_equal(
        np.asarray(a.isfinite()), [True, True, False, False, True]
    )
    # valid entries keep the exact unit sign and the value |x|
    np.testing.assert_array_equal(np.asarray(a.sign)[[0, 1, 4]], [1.0, 1.0, 1.0])
    assert_value(a[jnp.array([0, 1, 4])], [1.0, np.exp(0.5), 0.0])


def test_logarray_abs_keeps_nan_phase():
    # A complex phase head producing NaN on one sample while the log-magnitude head
    # is finite: the NaN must survive abs() and propagate through later arithmetic.
    theta = jnp.asarray([0.3, jnp.nan, 1.2])
    arr = LogArray(sign=jnp.exp(1j * theta), logabs=jnp.asarray([0.0, 0.5, -1.0]))
    a = abs(arr)
    assert jnp.issubdtype(a.sign.dtype, jnp.floating)  # abs is real
    np.testing.assert_array_equal(np.asarray(a.isfinite()), [True, False, True])
    np.testing.assert_array_equal(np.asarray(a.sign)[[0, 2]], [1.0, 1.0])
    ratio = a**2 / a  # the reweighting-style arithmetic of the samplers
    np.testing.assert_array_equal(np.asarray(isfinite(ratio)), [True, False, True])
    np.testing.assert_array_equal(np.asarray(isfinite(ratio.mean())), False)


def test_logarray_abs_keeps_nonfinite_logabs():
    # abs keeps the log-magnitude, so a NaN / +inf logabs survives as well.
    arr = LogArray(
        sign=jnp.asarray([1.0, -1.0, -1.0, 1.0]),
        logabs=jnp.asarray([0.0, jnp.nan, jnp.inf, -jnp.inf]),
    )
    a = abs(arr)
    np.testing.assert_array_equal(np.asarray(a.isnan()), [False, True, False, False])
    np.testing.assert_array_equal(np.asarray(a.isinf()), [False, False, True, False])
    np.testing.assert_array_equal(np.asarray(a.isfinite()), [True, False, False, True])
    assert_value(a[jnp.array([0, 3])], [1.0, 0.0])
