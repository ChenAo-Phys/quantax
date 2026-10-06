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
REAL = np.array([0.5, -1.5, 2.0, -0.25, 1.0], dtype=np.float32)
CPLX = np.array([0.5 + 1.0j, -1.5 - 0.5j, 2.0 + 0.0j, -0.25 + 2.0j], dtype=np.complex64)


def dense(arr):
    return np.asarray(arr.value())


# ---------- sinhp1_by_log ----------


def _log_sinhp1(x) -> tuple[np.ndarray, np.ndarray]:
    """float64 reference ``(log|f|, sign(f))`` of ``f = sinh(x) + 1``."""
    f = np.sinh(np.asarray(x, np.float64)) + 1
    return np.log(np.abs(f)), np.sign(f)


def test_sinhp1_by_log_real():
    out = sinhp1_by_log(jnp.asarray(REAL))
    assert isinstance(out, LogArray)
    np.testing.assert_allclose(dense(out), np.sinh(REAL) + 1, rtol=1e-4, atol=1e-5)


def test_sinhp1_by_log_real_no_overflow():
    # sinh(x) overflows float32 beyond |x| ~ 89, and sinh(x) + 1 changes sign at
    # x ~ -0.88; logabs and sign must stay accurate on both sides of both.
    neg = [-120.0, -89.5, -30.0, -3.0, -1.5, -0.9, -0.85, -0.5]
    pos = [0.0, 0.25, 1.0, 5.0, 40.0, 89.5, 120.0]
    x = np.array(neg + pos, dtype=np.float32)
    out = sinhp1_by_log(jnp.asarray(x))
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
    x = jnp.asarray([X0, 0.5, 1.5, 2.0], jnp.float32)
    out = sinhp1_by_log(x)
    # sig of x0 is exactly 0 and gets the 2^-40 shift (logabs ~ -26.8), while the
    # float32 rounding of a nonzero sig would give logabs ~ -16
    assert np.isfinite(out.logabs[0]) and out.logabs[0] < -25
    x64 = np.asarray(x, np.float64)
    expected = np.cosh(x64) / np.sum(np.sinh(x64) + 1)
    for f in (jax.grad(_log_surrogate), jax.jit(jax.grad(_log_surrogate))):
        grad = np.asarray(f(x))
        assert np.all(np.isfinite(grad))
        np.testing.assert_allclose(grad, expected, rtol=1e-5)


# ---------- prod_by_log ----------


def test_prod_by_log_real():
    out = prod_by_log(jnp.asarray(REAL))
    assert isinstance(out, LogArray)
    np.testing.assert_allclose(dense(out), np.prod(REAL), rtol=1e-4, atol=1e-5)


def test_prod_by_log_complex():
    out = prod_by_log(jnp.asarray(CPLX))
    np.testing.assert_allclose(dense(out), np.prod(CPLX), rtol=1e-4, atol=1e-5)


def test_prod_by_log_no_overflow():
    # prod of fifty 10's = 1e50 overflows float32; LogArray accumulates in log space.
    x = np.full(50, 10.0, dtype=np.float32)
    out = prod_by_log(jnp.asarray(x))
    assert np.isfinite(np.asarray(out.logabs))
    with np.errstate(over="ignore"):
        assert not np.isfinite(np.prod(x))  # naive product overflows
    recon = np.asarray(out.sign, np.float64) * np.exp(
        np.asarray(out.logabs, np.float64)
    )
    np.testing.assert_allclose(recon, np.prod(x.astype(np.float64)), rtol=1e-4)


# ---------- exp_by_log ----------


def test_exp_by_log_real():
    out = exp_by_log(jnp.asarray(REAL))
    assert isinstance(out, LogArray)
    np.testing.assert_allclose(dense(out), np.exp(REAL), rtol=1e-4, atol=1e-5)
    # real input -> sign is identically one
    np.testing.assert_array_equal(np.asarray(out.sign), np.ones_like(REAL))


def test_exp_by_log_complex():
    out = exp_by_log(jnp.asarray(CPLX))
    np.testing.assert_allclose(dense(out), np.exp(CPLX), rtol=1e-4, atol=1e-5)


def test_exp_by_log_no_overflow():
    x = np.array([80.0, 85.0, 90.0], dtype=np.float32)
    out = exp_by_log(jnp.asarray(x))
    assert np.all(np.isfinite(np.asarray(out.logabs)))
    with np.errstate(over="ignore"):
        assert not np.all(np.isfinite(np.exp(x)))
    recon = np.asarray(out.sign, np.float64) * np.exp(
        np.asarray(out.logabs, np.float64)
    )
    np.testing.assert_allclose(recon, np.exp(x.astype(np.float64)), rtol=1e-4)


# ---------- crelu ----------


def test_crelu():
    out = np.asarray(crelu(jnp.asarray(CPLX)))
    expected = np.maximum(CPLX.real, 0) + 1j * np.maximum(CPLX.imag, 0)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


# ---------- cardioid ----------


def test_cardioid():
    out = np.asarray(cardioid(jnp.asarray(CPLX)))
    expected = 0.5 * (1 + np.cos(np.angle(CPLX))) * CPLX
    np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-6)


# ---------- pair_cpl ----------


def test_pair_cpl():
    x = np.arange(6, dtype=np.float32)
    out = np.asarray(pair_cpl(jnp.asarray(x)))
    expected = x[:3] + 1j * x[3:]
    np.testing.assert_allclose(out, expected)
    assert np.iscomplexobj(out)


def test_pair_cpl_multidim():
    # splits along the first (channel) axis, as used in the conv models
    x = np.arange(8, dtype=np.float32).reshape(4, 2)
    out = np.asarray(pair_cpl(jnp.asarray(x)))
    expected = x[:2] + 1j * x[2:]
    np.testing.assert_allclose(out, expected)
