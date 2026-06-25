"""
Characterization tests for the iterative solvers in quantax/optimizer/solver.py.
"""

import numpy as np
import jax.numpy as jnp
import pytest
from quantax.optimizer import lsmr


def _complex_problem(n, m, seed=0):
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))).astype(
        np.complex64
    )
    b = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    return jnp.asarray(A), jnp.asarray(b)


@pytest.mark.parametrize("shape", [(200, 50), (50, 200)])
def test_lsmr_complex64(shape):
    # lsmr runs in the input dtype (no x64 upcast), so complex64 in must give a
    # finite complex64 least-square / min-norm solution matching the float64
    # reference to single precision.
    n, m = shape
    A, b = _complex_problem(n, m)
    x = lsmr(rtol=1e-6, atol=0.0, maxiter=2000)(A, b)

    assert x.dtype == jnp.complex64
    assert jnp.all(jnp.isfinite(x))

    A64 = np.asarray(A).astype(np.complex128)
    b64 = np.asarray(b).astype(np.complex128)
    x_ref, *_ = np.linalg.lstsq(A64, b64, rcond=None)

    # the least-square residual must match the reference (tall: nonzero residual;
    # wide: min-norm solution with ~zero residual)
    res = np.linalg.norm(A64 @ np.asarray(x) - b64)
    res_ref = np.linalg.norm(A64 @ x_ref - b64)
    np.testing.assert_allclose(res, res_ref, rtol=1e-4, atol=1e-4)

    err = np.linalg.norm(np.asarray(x) - x_ref) / np.linalg.norm(x_ref)
    assert err < 1e-4
