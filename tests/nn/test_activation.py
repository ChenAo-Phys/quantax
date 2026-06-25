import numpy as np
import jax.numpy as jnp
from quantax.nn import (
    sinhp1_by_scale,
    prod_by_log,
    exp_by_scale,
    exp_by_log,
    crelu,
    cardioid,
    pair_cpl,
)
from quantax.utils import LogArray, ScaleArray

# float32 / complex64 to match the default precision (x64 disabled in conftest).
REAL = np.array([0.5, -1.5, 2.0, -0.25, 1.0], dtype=np.float32)
CPLX = np.array([0.5 + 1.0j, -1.5 - 0.5j, 2.0 + 0.0j, -0.25 + 2.0j], dtype=np.complex64)


def dense(arr):
    return np.asarray(arr.value())


# ---------- sinhp1_by_scale ----------


def test_sinhp1_by_scale_real():
    out = sinhp1_by_scale(jnp.asarray(REAL))
    assert isinstance(out, ScaleArray)
    np.testing.assert_allclose(dense(out), np.sinh(REAL) + 1, rtol=1e-4, atol=1e-5)


def test_sinhp1_by_scale_no_overflow():
    # sinh(90) ~ 6e38 overflows float32; the ScaleArray representation must stay finite.
    x = np.array([90.0, 0.0, -90.0], dtype=np.float32)
    out = sinhp1_by_scale(jnp.asarray(x))
    assert np.all(np.isfinite(np.asarray(out.significand)))
    assert np.all(np.isfinite(np.asarray(out.exponent)))
    with np.errstate(over="ignore"):
        assert not np.all(np.isfinite(np.sinh(x) + 1))  # naive computation overflows
    # the dominant element reconstructs correctly in higher precision
    recon = np.asarray(out.significand, np.float64) * np.exp(
        np.asarray(out.exponent, np.float64)
    )
    ref = np.sinh(np.array([90.0], np.float64)) + 1
    np.testing.assert_allclose(recon[0], ref[0], rtol=1e-4)


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


# ---------- exp_by_scale ----------


def test_exp_by_scale_real():
    out = exp_by_scale(jnp.asarray(REAL))
    assert isinstance(out, ScaleArray)
    np.testing.assert_allclose(dense(out), np.exp(REAL), rtol=1e-4, atol=1e-5)


def test_exp_by_scale_complex():
    out = exp_by_scale(jnp.asarray(CPLX))
    np.testing.assert_allclose(dense(out), np.exp(CPLX), rtol=1e-4, atol=1e-5)


def test_exp_by_scale_no_overflow():
    x = np.array([80.0, 85.0, 90.0], dtype=np.float32)
    out = exp_by_scale(jnp.asarray(x))
    sig = np.asarray(out.significand)
    exp = np.asarray(out.exponent)
    assert np.all(np.isfinite(sig)) and np.all(np.isfinite(exp))
    with np.errstate(over="ignore"):
        assert not np.all(np.isfinite(np.exp(x)))  # exp(90) overflows float32
    recon = sig.astype(np.float64) * np.exp(exp.astype(np.float64))
    np.testing.assert_allclose(recon, np.exp(x.astype(np.float64)), rtol=1e-4)


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
