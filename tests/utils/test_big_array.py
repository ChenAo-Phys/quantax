import pytest
import numpy as np
import jax
import jax.numpy as jnp
from quantax.utils import LogArray, ScaleArray, where, isnan, isinf, isfinite

# Both classes expose the same array-like API, so the shared behavior is tested
# against both via the `cls` fixture. value() / np.asarray() materialize the dense
# array, which is compared to the equivalent operation on a plain numpy array.

# float32 / complex64 to match the default precision (x64 disabled in conftest)
REAL = np.array([[1.5, -2.0, 0.5], [3.0, -0.25, 4.0]], dtype=np.float32)
CPLX = np.array([1.0 + 2.0j, -3.0 + 0.0j, 0.5 - 1.5j], dtype=np.complex64)


@pytest.fixture(params=[LogArray, ScaleArray], ids=["LogArray", "ScaleArray"])
def cls(request):
    return request.param


def make(cls, x):
    return cls.from_value(jnp.asarray(x))


def dense(arr):
    return np.asarray(arr.value())


def assert_value(arr, expected):
    # log/exponent-space ops go through exp/log, so use float32-appropriate tolerances
    np.testing.assert_allclose(dense(arr), np.asarray(expected), rtol=1e-4, atol=1e-5)


# ---------- construction / conversion ----------


def test_from_value_roundtrip_real(cls):
    assert_value(make(cls, REAL), REAL)


def test_from_value_roundtrip_complex(cls):
    assert_value(make(cls, CPLX), CPLX)


def test_idempotent_from_value(cls):
    arr = make(cls, REAL)
    assert cls.from_value(arr) is arr


def test_numpy_and_jax_conversion(cls):
    arr = make(cls, REAL)
    np.testing.assert_allclose(np.asarray(arr), REAL)
    np.testing.assert_allclose(np.asarray(jnp.asarray(arr)), REAL)


def test_properties(cls):
    arr = make(cls, REAL)
    assert arr.shape == REAL.shape
    assert arr.ndim == REAL.ndim
    assert arr.size == REAL.size
    assert jnp.issubdtype(arr.dtype, jnp.floating)
    assert jnp.issubdtype(make(cls, CPLX).dtype, jnp.complexfloating)


# ---------- unary ops ----------


def test_neg(cls):
    assert_value(-make(cls, REAL), -REAL)


def test_conj(cls):
    assert_value(make(cls, CPLX).conj(), CPLX.conj())


def test_abs(cls):
    assert_value(make(cls, REAL).abs(), np.abs(REAL))
    assert_value(abs(make(cls, CPLX)), np.abs(CPLX))


def test_real_imag_complex(cls):
    arr = make(cls, CPLX)
    assert_value(arr.real, CPLX.real)
    assert_value(arr.imag, CPLX.imag)


def test_real_imag_real(cls):
    arr = make(cls, REAL)
    assert_value(arr.real, REAL)
    assert_value(arr.imag, np.zeros_like(REAL))


def test_transpose_property(cls):
    assert_value(make(cls, REAL).T, REAL.T)


def test_astype(cls):
    arr = make(cls, REAL).astype(jnp.complex64)
    assert jnp.issubdtype(arr.dtype, jnp.complexfloating)
    assert_value(arr, REAL)


# ---------- binary ops ----------


def test_mul(cls):
    a, b = make(cls, REAL), make(cls, REAL + 1.0)
    assert_value(a * b, REAL * (REAL + 1.0))
    assert_value(make(cls, REAL) * 2.0, REAL * 2.0)
    assert_value(2.0 * make(cls, REAL), REAL * 2.0)  # __rmul__


def test_truediv(cls):
    a, b = make(cls, REAL), make(cls, REAL + 1.0)
    assert_value(a / b, REAL / (REAL + 1.0))
    assert_value(make(cls, REAL) / 2.0, REAL / 2.0)
    assert_value(6.0 / make(cls, REAL), 6.0 / REAL)  # __rtruediv__


@pytest.mark.parametrize("p", [2, 3])
def test_pow(cls, p):
    assert_value(make(cls, REAL) ** p, REAL**p)


def test_add(cls):
    a, b = make(cls, REAL), make(cls, REAL + 1.0)
    assert_value(a + b, REAL + (REAL + 1.0))
    assert_value(make(cls, REAL) + 2.0, REAL + 2.0)
    assert_value(2.0 + make(cls, REAL), REAL + 2.0)  # __radd__


def test_sub(cls):
    a, b = make(cls, REAL), make(cls, REAL + 1.0)
    assert_value(a - b, REAL - (REAL + 1.0))
    assert_value(make(cls, REAL) - 2.0, REAL - 2.0)
    assert_value(5.0 - make(cls, REAL), 5.0 - REAL)  # __rsub__


def test_add_with_jax_array(cls):
    # left operand is the custom type, so the representation is preserved
    res = make(cls, REAL) + jnp.asarray(REAL)
    assert isinstance(res, cls)
    assert_value(res, 2 * REAL)


# ---------- reductions ----------


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_sum(cls, axis, keepdims):
    arr = make(cls, REAL).sum(axis=axis, keepdims=keepdims)
    assert_value(arr, REAL.sum(axis=axis, keepdims=keepdims))


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_mean(cls, axis):
    assert_value(make(cls, REAL).mean(axis=axis), REAL.mean(axis=axis))


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_prod(cls, axis):
    assert_value(make(cls, REAL).prod(axis=axis), REAL.prod(axis=axis))


def test_sum_complex(cls):
    assert_value(make(cls, CPLX).sum(), CPLX.sum())


# ---------- generated array methods ----------


def test_reshape_flatten(cls):
    assert_value(make(cls, REAL).reshape(6), REAL.reshape(6))
    assert_value(make(cls, REAL).flatten(), REAL.flatten())


def test_transpose_method(cls):
    assert_value(make(cls, REAL).transpose(), REAL.transpose())


def test_getitem(cls):
    assert_value(make(cls, REAL)[0], REAL[0])
    assert_value(make(cls, REAL)[:, 1], REAL[:, 1])


def test_take(cls):
    arr = make(cls, REAL).take(np.array([0, 2]), axis=1)
    assert_value(arr, REAL.take([0, 2], axis=1))


# ---------- pytree / jit ----------


def test_pytree_flatten_unflatten(cls):
    arr = make(cls, REAL)
    leaves, treedef = jax.tree.flatten(arr)
    assert len(leaves) == 2
    rebuilt = jax.tree.unflatten(treedef, leaves)
    assert isinstance(rebuilt, cls)
    assert_value(rebuilt, REAL)


def test_pytree_map(cls):
    # scaling both leaves is not the same as scaling the value, but it must stay a
    # valid pytree of the same type with the leaves transformed
    arr = make(cls, REAL)
    doubled = jax.tree.map(lambda x: x * 2, arr)
    assert isinstance(doubled, cls)
    leaves = jax.tree.leaves(arr)
    np.testing.assert_allclose(
        np.asarray(jax.tree.leaves(doubled)[0]), 2 * np.asarray(leaves[0])
    )


def test_jit(cls):
    @jax.jit
    def f(a):
        return (a * a).value()

    out = f(make(cls, REAL))
    np.testing.assert_allclose(np.asarray(out), REAL**2, rtol=1e-4)


# ---------- class-specific ----------


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


def test_scalearray_isnan():
    arr = ScaleArray(
        significand=jnp.asarray([1.0, jnp.nan, 1.0, 0.0, 1.0]),
        exponent=jnp.asarray([0.0, 0.0, jnp.nan, 0.0, jnp.inf]),
    )
    np.testing.assert_array_equal(
        np.asarray(arr.isnan()), [False, True, True, False, False]
    )


def test_cross_conversion():
    la = LogArray.from_value(jnp.asarray(REAL))
    sa = ScaleArray.from_value(la)
    assert isinstance(sa, ScaleArray)
    assert_value(sa, REAL)

    back = LogArray.from_value(sa)
    assert isinstance(back, LogArray)
    assert_value(back, REAL)


def test_logarray_stability():
    # dense values would overflow (exp(700) ~ 1e304, product over 5 -> inf), but the
    # log representation keeps the exact log-magnitude
    big = LogArray(sign=jnp.ones(5), logabs=jnp.full(5, 700.0))
    prod = big.prod()
    np.testing.assert_allclose(np.asarray(prod.logabs), 3500.0)
    assert float(jnp.exp(jnp.asarray(700.0)) ** 5) == np.inf  # dense overflows


def test_scalearray_stability():
    big = ScaleArray(significand=jnp.ones(5), exponent=jnp.full(5, 700.0))
    prod = big.prod()
    np.testing.assert_allclose(np.asarray(prod.exponent), 3500.0)


# ---------- where ----------


def test_where_logarray():
    cond = np.array([True, False, True])
    x = LogArray.from_value(jnp.asarray([1.0, 2.0, 3.0]))
    y = LogArray.from_value(jnp.asarray([4.0, 5.0, 6.0]))
    res = where(cond, x, y)
    assert isinstance(res, LogArray)
    assert_value(res, [1.0, 5.0, 3.0])


def test_where_scalearray():
    cond = np.array([True, False, True])
    x = ScaleArray.from_value(jnp.asarray([1.0, 2.0, 3.0]))
    y = ScaleArray.from_value(jnp.asarray([4.0, 5.0, 6.0]))
    res = where(cond, x, y)
    assert isinstance(res, ScaleArray)
    assert_value(res, [1.0, 5.0, 3.0])


def test_where_mixed_promotes_to_scalearray():
    cond = np.array([True, False, True])
    x = ScaleArray.from_value(jnp.asarray([1.0, 2.0, 3.0]))
    y = LogArray.from_value(jnp.asarray([4.0, 5.0, 6.0]))
    res = where(cond, x, y)
    assert isinstance(res, ScaleArray)
    assert_value(res, [1.0, 5.0, 3.0])


def test_where_plain_array():
    cond = np.array([True, False, True])
    res = where(cond, jnp.asarray([1.0, 2.0, 3.0]), jnp.asarray([4.0, 5.0, 6.0]))
    assert isinstance(res, jax.Array)
    np.testing.assert_allclose(np.asarray(res), [1.0, 5.0, 3.0])


# ---------- isnan / isinf / isfinite dispatch ----------


def test_predicates_dispatch():
    # The module-level predicates give consistent answers for plain arrays and
    # both stable representations; value pattern: [normal, nan, inf, zero].
    arrays = [
        jnp.asarray([1.0, jnp.nan, jnp.inf, 0.0]),
        LogArray(
            sign=jnp.asarray([1.0, jnp.nan, 1.0, 0.0]),
            logabs=jnp.asarray([0.0, 0.0, jnp.inf, -jnp.inf]),
        ),
        ScaleArray(
            significand=jnp.asarray([1.0, jnp.nan, 1.0, 0.0]),
            exponent=jnp.asarray([0.0, 0.0, jnp.inf, 0.0]),
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
