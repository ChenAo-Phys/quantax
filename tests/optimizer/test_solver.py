"""
Characterization tests for the solvers in quantax/optimizer/solver.py.
"""

import warnings
import numpy as np
import jax.numpy as jnp
import pytest
from quantax.optimizer import (
    lsmr,
    lstsq_cg,
    minnorm_cg,
    auto_shift_eig,
    minnorm_shift_eig,
    lstsq_shift_eig,
    minnorm_pinv_eig,
    lstsq_pinv_eig,
    auto_pinv_eig,
)


def _complex_problem(n, m, seed=0):
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))).astype(
        np.complex64
    )
    b = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    return jnp.asarray(A), jnp.asarray(b)


@pytest.mark.parametrize("shape", [(200, 50), (50, 200)])
def test_lsmr_complex64(shape):
    # lsmr keeps the input dtype by default, so complex64 in must give a finite
    # complex64 least-square / min-norm solution matching the float64 reference to
    # single precision.
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


@pytest.mark.parametrize("shape", [(200, 50), (50, 200)])
def test_cg_solves_shifted_equation(shape, x64):
    # CG converges to the solution of the shifted normal equation, with the shift
    # eps = Tr(A^† A) * rshift / sqrt(min(n, m)) + ashift for both solvers.
    n, m = shape
    A, b = _complex_problem(n, m)
    A, b = A.astype(jnp.complex128), b.astype(jnp.complex128)
    A64, b64 = np.asarray(A), np.asarray(b)
    rshift, ashift = 1e-2, 1e-1
    trace = np.sum(np.abs(A64) ** 2)
    eps = trace * rshift / np.sqrt(min(n, m)) + ashift
    S = A64.conj().T @ A64 + eps * np.eye(m)
    x_lstsq = np.linalg.solve(S, A64.conj().T @ b64)
    T = A64 @ A64.conj().T + eps * np.eye(n)
    x_minnorm = A64.conj().T @ np.linalg.solve(T, b64)
    kwargs = dict(
        rshift=rshift, ashift=ashift, rtol=1e-12, maxiter=2000, dtype=jnp.float64
    )
    for solver, x_ref in [(lstsq_cg, x_lstsq), (minnorm_cg, x_minnorm)]:
        x = solver(**kwargs)(A, b)
        assert x.dtype == jnp.complex128
        err = np.linalg.norm(np.asarray(x) - x_ref) / np.linalg.norm(x_ref)
        assert err < 1e-8


def _rank_deficient_problem(n, m, rank=32, seed=0):
    """
    A float32 A of the given rank with singular values from 1 to 1e-2 and a tiny
    noise, and a generic b that can't be fitted exactly, like a VMC Jacobian.
    """
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((m, m)))
    k = min(n, m)
    s = np.zeros(k)
    s[:rank] = np.logspace(0, -2, rank)
    A = (U[:, :k] * s) @ V[:, :k].T + 1e-6 * rng.standard_normal((n, m))
    b = rng.standard_normal(n)
    return jnp.asarray(A, jnp.float32), jnp.asarray(b, jnp.float32)


@pytest.mark.parametrize("shape", [(64, 256), (256, 64)])
def test_minnorm_cg_rank_deficient(shape):
    # Without the shift, CG on the rank-deficient A A^† amplifies the small
    # eigen-directions and returns a residual far above |b|, worse than no step.
    A, b = _rank_deficient_problem(*shape)
    x = minnorm_cg()(A, b)
    A64, b64 = np.asarray(A, np.float64), np.asarray(b, np.float64)
    res = np.linalg.norm(A64 @ np.asarray(x, np.float64) - b64) / np.linalg.norm(b64)
    assert np.isfinite(res) and res < 1


@pytest.mark.parametrize("shape", [(200, 50), (50, 200)])
@pytest.mark.parametrize(
    "dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128]
)
def test_iterative_solvers_default_dtype_under_x64(shape, dtype, x64):
    # By default CG and LSMR keep the input dtype, also single precision under global
    # x64, and leave x64 enabled.
    n, m = shape
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, m))
    b = rng.standard_normal(n)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        A = A + 1j * rng.standard_normal((n, m))
        b = b + 1j * rng.standard_normal(n)
    A, b = jnp.asarray(A, dtype), jnp.asarray(b, dtype)
    A64, b64 = np.asarray(A, np.complex128), np.asarray(b, np.complex128)
    x_ref, *_ = np.linalg.lstsq(A64, b64)
    res_ref = np.linalg.norm(A64 @ x_ref - b64) / np.linalg.norm(b64)
    cg = lstsq_cg if n > m else minnorm_cg
    for solver in [cg(), lsmr()]:
        x = solver(A, b)
        assert x.dtype == dtype
        res = np.linalg.norm(A64 @ np.asarray(x) - b64) / np.linalg.norm(b64)
        assert res < res_ref + 0.05
        assert jnp.zeros(()).dtype == jnp.float64

    # double-precision inputs also iterate in double precision, reaching the exact
    # least-square solution beyond single-precision accuracy
    if jnp.finfo(dtype).bits == 64:
        x = lsmr(rtol=1e-12, maxiter=2000)(A, b)
        err = np.linalg.norm(np.asarray(x) - x_ref) / np.linalg.norm(x_ref)
        assert err < 1e-8


@pytest.mark.parametrize("is_complex", [False, True])
def test_eig_solvers_single_precision_under_x64(is_complex, x64):
    # A single-precision dtype disables x64 inside the solver, so the solution must
    # be cast back to the input dtype outside, without a dtype truncation warning.
    rng = np.random.default_rng(0)
    for shape in [(40, 10), (10, 40)]:
        A = rng.standard_normal(shape)
        b = rng.standard_normal(shape[0])
        if is_complex:
            A = A + 1j * rng.standard_normal(shape)
            b = b + 1j * rng.standard_normal(shape[0])
        A, b = jnp.asarray(A), jnp.asarray(b)
        solvers = [
            minnorm_shift_eig,
            lstsq_shift_eig,
            auto_shift_eig,
            minnorm_pinv_eig,
            lstsq_pinv_eig,
            auto_pinv_eig,
        ]
        for solver in solvers:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                x = solver(dtype=jnp.float32)(A, b)
            assert x.dtype == A.dtype
            assert jnp.all(jnp.isfinite(x))


# ====================================================================
# Shift-eig (Cholesky) solvers
# ====================================================================


def _ref_shift_eig(A, b, rshift=1e-12, ashift=1e-6):
    """numpy reference of the shifted normal equations solved by auto_shift_eig."""
    A = np.asarray(A).astype(np.complex128 if np.iscomplexobj(A) else np.float64)
    b = np.asarray(b).astype(A.dtype)
    n, m = A.shape
    if n < m:  # MinSR: x = A^† (A A^† + eps)^-1 b
        T = A @ A.conj().T
        shift = rshift * np.trace(T).real / np.sqrt(n) + ashift
        return A.conj().T @ np.linalg.solve(T + shift * np.eye(n), b)
    else:  # SR: x = (A^† A + eps)^-1 A^† b
        S = A.conj().T @ A
        shift = rshift * np.trace(S).real / np.sqrt(m) + ashift
        return np.linalg.solve(S + shift * np.eye(m), A.conj().T @ b)


@pytest.mark.parametrize("shape", [(64, 200), (200, 64), (12, 50), (50, 13)])
@pytest.mark.parametrize("is_complex", [False, True])
def test_shift_eig_reference(shape, is_complex, x64):
    # both branches of auto_shift_eig reproduce the shifted normal equations, also for
    # sizes not divisible by the device count (parameter padding in the MinSR branch)
    n, m = shape
    if is_complex:
        A, b = _complex_problem(n, m)
        A, b = A.astype(jnp.complex128), b.astype(jnp.complex128)
    else:
        rng = np.random.default_rng(1)
        A = jnp.asarray(rng.standard_normal((n, m)))
        b = jnp.asarray(rng.standard_normal(n))
    x = auto_shift_eig()(A, b)
    assert x.shape == (m,) and x.dtype == A.dtype
    np.testing.assert_allclose(np.asarray(x), _ref_shift_eig(A, b), rtol=1e-8)
