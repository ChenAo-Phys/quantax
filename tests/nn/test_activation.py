import numpy as np
import pytest
import jax
import jax.numpy as jnp
from quantax.nn import (
    sinhp1_by_log,
    prod_by_log,
    exp_by_log,
    crelu,
    cardioid,
    pair_cpl,
)
from quantax.utils import LogArray

# float32 / complex64 to match the default precision (x64 disabled in conftest).
# Functions are evaluated under jit as in the networks; eager evaluation would
# compile every primitive separately and dominate the test time.
CPLX = np.array([0.5 + 1.0j, -1.5 - 0.5j, 2.0 + 0.0j, -0.25 + 2.0j], dtype=np.complex64)


def _value(out: LogArray) -> np.ndarray:
    """float64 / complex128 value of a LogArray, computed on the host."""
    sign = np.asarray(out.sign).astype(np.complex128)
    return sign * np.exp(np.asarray(out.logabs, np.float64))


# ---------- sinhp1_by_log ----------


def _log_sinhp1(x) -> tuple[np.ndarray, np.ndarray]:
    """float64 reference ``(log|f|, sign(f))`` of ``f = sinh(x) + 1``."""
    f = np.sinh(np.asarray(x, np.float64)) + 1
    return np.log(np.abs(f)), np.sign(f)


def test_sinhp1_by_log_real_no_overflow():
    # sinh(x) overflows float32 beyond |x| ~ 89, and sinh(x) + 1 changes sign at
    # x ~ -0.88; logabs and sign must stay accurate on both sides of both.
    neg = [-120.0, -89.5, -30.0, -3.0, -1.5, -0.9, -0.85, -0.5, -0.25]
    pos = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 40.0, 89.5, 120.0]
    x = np.array(neg + pos, dtype=np.float32)
    out = jax.jit(sinhp1_by_log)(x)
    assert isinstance(out, LogArray)
    with np.errstate(over="ignore"):
        assert not np.all(np.isfinite(np.sinh(x) + 1))  # naive computation overflows
    logabs, sign = _log_sinhp1(x)
    assert np.all(np.isfinite(np.asarray(out.logabs)))
    np.testing.assert_allclose(np.asarray(out.logabs), logabs, rtol=1e-6, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(out.sign), sign)


def test_sinhp1_by_log_complex_raises():
    with pytest.raises(TypeError, match="real"):
        sinhp1_by_log(jnp.asarray(CPLX))


# sinh(x) + 1 is exactly 0 in float32 at x0
X0 = np.float32(-0.8813735842704773)


def _log_surrogate(x):
    """The Variational surrogate of log(psi), psi = sum(sinh(x) + 1), whose
    gradient is d log(psi) / dx."""
    y = sinhp1_by_log(x).sum()
    return y.sign / jax.lax.stop_gradient(y.sign) + y.logabs


def test_sinhp1_by_log_exact_zero_has_finite_gradient():
    # Regression: sinhp1_by_log computes sig = (sinh(x) + 1) exp(-|x|). At an exact
    # zero of sig, d log|sig| = dsig / sig is infinite and the jacobian of any sum
    # containing the element was NaN.
    x = np.array([X0, 0.5, 1.5, 2.0], np.float32)
    f = jax.jit(lambda x: (sinhp1_by_log(x).logabs, jax.grad(_log_surrogate)(x)))
    logabs, grad = f(x)
    # sig of x0 is exactly 0 and gets the eps^2 shift (logabs ~ -31), while the
    # float32 rounding of a nonzero sig would give logabs ~ -16
    assert np.isfinite(logabs[0]) and logabs[0] < -25
    x64 = np.asarray(x, np.float64)
    expected = np.cosh(x64) / np.sum(np.sinh(x64) + 1)
    assert np.all(np.isfinite(np.asarray(grad)))
    np.testing.assert_allclose(np.asarray(grad), expected, rtol=1e-5)


# ---------- prod_by_log ----------


def test_prod_by_log_real_sign_without_overflow():
    # 51 factors of magnitude 10 with 25 negative ones: the product -1e51 overflows
    # float32, while LogArray keeps the sign and accumulates in log space.
    x = (10.0 * (-1.0) ** np.arange(51)).astype(np.float32)
    out = jax.jit(prod_by_log)(x)
    with np.errstate(over="ignore"):
        assert not np.isfinite(np.prod(x))  # naive product overflows
    assert int(out.sign) == -1
    # a sequential float32 sum of the 51 logs is off by 6e-7 relative
    np.testing.assert_allclose(float(out.logabs), 51 * np.log(10.0), rtol=2e-6)


def test_prod_by_log_complex():
    out = jax.jit(prod_by_log)(CPLX)
    np.testing.assert_allclose(_value(out), np.prod(CPLX.astype(np.complex128)), 1e-5)


# ---------- exp_by_log ----------


def test_exp_by_log_real_is_exact_log():
    # Real input: logabs is the input itself (no overflow for exp(90) > float32 max)
    # and the sign is identically one.
    x = np.array([-2.0, 0.0, 0.5, 80.0, 90.0], dtype=np.float32)
    out = jax.jit(exp_by_log)(x)
    np.testing.assert_array_equal(np.asarray(out.logabs), x)
    np.testing.assert_array_equal(np.asarray(out.sign), np.ones_like(x))


def test_exp_by_log_complex():
    out = jax.jit(exp_by_log)(CPLX)
    np.testing.assert_allclose(_value(out), np.exp(CPLX.astype(np.complex128)), 1e-6)


# ---------- crelu / cardioid / pair_cpl ----------


def test_complex_activations_on_axes():
    # Hand values on the real / imaginary axes pin both definitions:
    # crelu clips Re and Im separately; cardioid keeps arg 0, kills arg pi and
    # halves arg +-pi/2. Off the axes, cardioid scales by (1 + cos(arg z)) / 2.
    z = np.array([2.0, -2.0, 3.0j, -3.0j, 1.0 - 1.0j, -2.0 + 1.0j], dtype=np.complex64)
    out_crelu, out_cardioid = jax.jit(lambda z: (crelu(z), cardioid(z)))(z)
    np.testing.assert_allclose(np.asarray(out_crelu), [2.0, 0.0, 3.0j, 0.0, 1.0, 1.0j])
    expected = [2.0, 0.0, 1.5j, -1.5j, (1 + np.sqrt(0.5)) / 2 * (1.0 - 1.0j)]
    expected.append((1 - 2 / np.sqrt(5)) / 2 * (-2.0 + 1.0j))
    np.testing.assert_allclose(np.asarray(out_cardioid), expected, atol=1e-6)


def test_pair_cpl_splits_channel_axis():
    # splits along the first (channel) axis, as used in the conv models
    x = np.arange(8, dtype=np.float32).reshape(4, 2)
    out = np.asarray(jax.jit(pair_cpl)(x))
    assert np.iscomplexobj(out)
    np.testing.assert_array_equal(out, x[:2] + 1j * x[2:])
