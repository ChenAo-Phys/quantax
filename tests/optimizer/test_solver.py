"""
Characterization tests for the solvers in quantax/optimizer/solver.py.
"""

import warnings
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import pytest
from quantax.sites import Chain
from quantax.state import Variational
from quantax.nn import Sequential
from quantax.optimizer import (
    lsmr,
    lstsq_cg,
    minnorm_cg,
    auto_shift_eig,
    auto_pinv_eig,
    block_pinv_eig,
)


def _problem(n, m, dtype, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, m))
    b = rng.standard_normal(n)
    if jnp.issubdtype(dtype, jnp.complexfloating):
        A = A + 1j * rng.standard_normal((n, m))
        b = b + 1j * rng.standard_normal(n)
    return jnp.asarray(A, dtype), jnp.asarray(b, dtype)


def _spread_problem(n, m, dtype, cond=10**1.5, seed=1):
    """
    A with singular values from 1 to 1 / cond, by default so that the eigenvalues of
    A^† A and A A^† span the regularization cutoffs below, and a generic b.
    """
    rng = np.random.default_rng(seed)

    def G(*shape):
        x = rng.standard_normal(shape)
        if jnp.issubdtype(dtype, jnp.complexfloating):
            x = x + 1j * rng.standard_normal(shape)
        return x

    U, _ = np.linalg.qr(G(n, n))
    V, _ = np.linalg.qr(G(m, m))
    k = min(n, m)
    A = (U[:, :k] * np.logspace(0, -np.log10(cond), k)) @ V[:, :k].conj().T
    return jnp.asarray(A, dtype), jnp.asarray(G(n), dtype)


def _to64(A, b):
    A = np.asarray(A).astype(np.complex128 if np.iscomplexobj(A) else np.float64)
    return A, np.asarray(b).astype(A.dtype)


def _ref_shift_eig(A, b, rshift, ashift):
    """
    The shifted normal equations, x = (A^† A + eps)^-1 A^† b (SR) for tall A and
    x = A^† (A A^† + eps)^-1 b (MinSR) for wide A, with
    eps = Tr(A^† A) * rshift / sqrt(min(n, m)) + ashift.
    """
    A, b = _to64(A, b)
    n, m = A.shape
    eps = rshift * np.sum(np.abs(A) ** 2) / np.sqrt(min(n, m)) + ashift
    if n < m:
        return A.conj().T @ np.linalg.solve(A @ A.conj().T + eps * np.eye(n), b)
    return np.linalg.solve(A.conj().T @ A + eps * np.eye(m), A.conj().T @ b)


def _ref_snr_sum(rho, tol_snr):
    """Sum over the sample axis 0, damping the components of low signal-to-noise."""
    x = rho.sum(axis=0)
    mean = x / rho.shape[0]
    noise = np.sqrt(np.mean(np.abs(rho - mean) ** 2, axis=0) / rho.shape[0])
    if tol_snr > 0:
        x = x / (1 + (tol_snr * noise / np.abs(mean)) ** 6)
    return x


def _ref_pinv_eig(A, b, rtol, atol=0.0, tol_snr=0.0):
    """
    The pseudo-inverse solution from the eigendecomposition of A A^† (MinSR, wide A)
    or A^† A (SR, tall A), with the eigenvalues lambda smoothly truncated as
    1 / (lambda (1 + ((rtol max|lambda| + atol) / |lambda|)^6)).
    """
    A, b = _to64(A, b)
    n, m = A.shape
    vals, U = np.linalg.eigh(A @ A.conj().T if n < m else A.conj().T @ A)
    cut = rtol * np.max(np.abs(vals)) + atol
    inv = 1 / (vals * (1 + (cut / np.abs(vals)) ** 6))
    if n < m:  # per-sample contributions conj(U_ts) b_t to U^† b
        rho = _ref_snr_sum(U.conj() * b[:, None], tol_snr)
        return A.conj().T @ U @ (inv * rho)
    # per-sample contributions to U^† A^† b
    rho = _ref_snr_sum((A @ U).conj() * b[:, None], tol_snr)
    return U @ (inv * rho)


def _rel_err(x, x_ref):
    return np.linalg.norm(np.asarray(x) - x_ref) / np.linalg.norm(x_ref)


def _cg_residual(A, b, x, rshift):
    """
    The relative residual of the shifted normal equations M z = c solved by CG:
    M = A^† A + eps, z = x and c = A^† b for tall A (lstsq_cg), M = A A^† + eps,
    x = A^† z and c = b for wide A (minnorm_cg), with eps from rshift as above.
    """
    A, b = _to64(A, b)
    x = np.asarray(x).astype(A.dtype)
    n, m = A.shape
    eps = rshift * np.sum(np.abs(A) ** 2) / np.sqrt(min(n, m))
    if n < m:
        z = np.linalg.solve(A @ A.conj().T, A @ x)
        M, c = A @ A.conj().T + eps * np.eye(n), b
    else:
        z, c = x, A.conj().T @ b
        M = A.conj().T @ A + eps * np.eye(m)
    return np.linalg.norm(M @ z - c) / np.linalg.norm(c)


def _lsmr_stopping_ratio(A, b, x):
    """
    The LSMR stopping criterion at x: |A^† r| / (|A| |r|) for the least-square
    problem of tall A without an exact solution, and |r| / (|A| |x| + |b|) for the
    consistent problem of wide A, with r = A x - b. The Frobenius norm |A| bounds
    the internal estimate of LSMR from above.
    """
    A, b = _to64(A, b)
    x = np.asarray(x).astype(A.dtype)
    r = A @ x - b
    norm_A = np.linalg.norm(A)
    if A.shape[0] > A.shape[1]:
        return np.linalg.norm(A.conj().T @ r) / (norm_A * np.linalg.norm(r))
    return np.linalg.norm(r) / (norm_A * np.linalg.norm(x) + np.linalg.norm(b))


# ====================================================================
# Iterative solvers
# ====================================================================


@pytest.mark.parametrize(
    "dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128]
)
def test_iterative_solvers_keep_input_dtype_under_x64(dtype, x64):
    # By default CG and LSMR iterate in the input dtype, also in single precision
    # under global x64, return it and leave x64 enabled. They converge to the exact
    # solution within the precision of the dtype: CG to the shifted normal equations
    # of both branches, LSMR to the unshifted least-square (tall) and min-norm (wide)
    # solutions.
    single = jnp.finfo(dtype).bits == 32
    tol, rtol = (1e-5, 1e-6) if single else (1e-8, 1e-12)
    tall, wide = _problem(40, 10, dtype), _problem(10, 40, dtype)
    rshift, ashift = 1e-2, 1e-1
    kwargs = dict(rshift=rshift, ashift=ashift, rtol=rtol, maxiter=2000)
    solve_lsmr = lsmr(rtol=rtol, maxiter=2000)
    cases = [
        (lstsq_cg(**kwargs), tall, _ref_shift_eig(*tall, rshift, ashift)),
        (minnorm_cg(**kwargs), wide, _ref_shift_eig(*wide, rshift, ashift)),
        (solve_lsmr, tall, np.linalg.lstsq(*_to64(*tall))[0]),
        (solve_lsmr, wide, np.linalg.lstsq(*_to64(*wide))[0]),
    ]
    for solver, (A, b), x_ref in cases:
        x = solver(A, b)
        assert x.dtype == dtype
        assert _rel_err(x, x_ref) < tol
    assert jnp.zeros(()).dtype == jnp.float64


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
def test_iterative_solvers_default_stopping(dtype):
    # The production setting: single precision, x64 disabled and the default
    # tolerances, which regularize by early stopping. The iterations must run until
    # the documented criteria are met: the relative residual rtol = 1e-2 of the CG
    # normal equations with the default rshift = 1e-4, and rtol / n = 0.1 / n of
    # LSMR, with n the number of rows (samples).
    tall, wide = _problem(40, 10, dtype), _problem(10, 40, dtype)
    for solver, (A, b) in [(lstsq_cg(), tall), (minnorm_cg(), wide)]:
        x = solver(A, b)
        assert x.dtype == dtype
        assert _cg_residual(A, b, x, rshift=1e-4) < 1e-2
    default_lsmr = lsmr()
    for A, b in (tall, wide):
        x = default_lsmr(A, b)
        assert x.dtype == dtype
        assert _lsmr_stopping_ratio(A, b, x) < 0.1 / A.shape[0]
    assert jnp.zeros(()).dtype == jnp.float32


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


def test_lsmr_diag_preconditioner_gives_weighted_min_norm(x64):
    # LSMR on A D^-1 followed by x = D^-1 y keeps A x = b but minimizes |D x|
    # instead of |x|: x = D^-2 A^† (A D^-2 A^†)^-1 b for an underdetermined A.
    A, b = _problem(10, 40, jnp.complex128)
    d = np.random.default_rng(2).uniform(0.5, 2.0, 40)
    x = lsmr(rtol=1e-12, maxiter=2000)(A, b, diag_preconditioner=jnp.asarray(d))
    A64, b64 = _to64(A, b)
    Ad = A64 / d**2
    x_ref = Ad.conj().T @ np.linalg.solve(A64 @ Ad.conj().T, b64)
    assert _rel_err(x, x_ref) < 1e-8


# ====================================================================
# Eigendecomposition (shift and pseudo-inverse) solvers
# ====================================================================

# The MinSR (wide) and SR (tall) branches, with sizes not divisible by the device
# count (parameter padding in the MinSR branch)
EIG_SHAPES = [(12, 50), (50, 13)]


@pytest.mark.parametrize("is_complex", [False, True])
def test_eig_solvers_reference(is_complex, x64):
    # Both branches of auto_shift_eig (default shifts in double precision) and
    # auto_pinv_eig (smooth eigenvalue truncation with rtol and atol, and the
    # signal-to-noise damping of tol_snr) reproduce the numpy references.
    shift_solver = auto_shift_eig()
    pinv_solver = auto_pinv_eig(rtol=1e-2, atol=1e-3, tol_snr=1.0)
    for shape in EIG_SHAPES:
        A, b = _spread_problem(*shape, jnp.complex128 if is_complex else jnp.float64)
        x = shift_solver(A, b)
        assert x.shape == (shape[1],) and x.dtype == A.dtype
        x_ref = _ref_shift_eig(A, b, rshift=1e-12, ashift=1e-6)
        np.testing.assert_allclose(np.asarray(x), x_ref, rtol=1e-8)

        x = pinv_solver(A, b)
        assert x.shape == (shape[1],) and x.dtype == A.dtype
        x_ref = _ref_pinv_eig(A, b, rtol=1e-2, atol=1e-3, tol_snr=1.0)
        np.testing.assert_allclose(np.asarray(x), x_ref, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("is_complex", [False, True])
def test_eig_solvers_single_precision_under_x64(is_complex, x64):
    # A single-precision dtype disables x64 inside the solver, so the solution must
    # be cast back to the input dtype outside, without a dtype truncation warning,
    # and x64 stays enabled.
    shift_solver = auto_shift_eig(dtype=jnp.float32)
    pinv_solver = auto_pinv_eig(dtype=jnp.float32)
    for shape in EIG_SHAPES:
        A, b = _problem(*shape, jnp.complex128 if is_complex else jnp.float64)
        refs = [
            _ref_shift_eig(A, b, rshift=1e-6, ashift=1e-6),
            _ref_pinv_eig(A, b, rtol=1e-6),
        ]
        for solver, x_ref in zip([shift_solver, pinv_solver], refs):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                x = solver(A, b)
            assert x.dtype == A.dtype
            assert _rel_err(x, x_ref) < 1e-5
    assert jnp.zeros(()).dtype == jnp.float64


def test_double_precision_solvers_with_x64_disabled():
    # The eigendecomposition solvers default to dtype=float64, and LSMR accepts it,
    # also with x64 disabled as by default: x64 is enabled inside the solver only, so
    # the single-precision inputs of an ill-conditioned problem are solved in double
    # precision (single precision is off by more than 1e-6), cast back without a dtype
    # truncation warning, and x64 stays disabled.
    shift_solver, pinv_solver = auto_shift_eig(), auto_pinv_eig()
    lsmr_solver = lsmr(rtol=1e-12, maxiter=4000, dtype=jnp.float64)
    for shape in EIG_SHAPES:
        A, b = _spread_problem(*shape, jnp.float32, cond=10**2.5)
        cases = [
            (shift_solver, _ref_shift_eig(A, b, rshift=1e-12, ashift=1e-6)),
            (pinv_solver, _ref_pinv_eig(A, b, rtol=1e-12)),
            (lsmr_solver, np.linalg.lstsq(*_to64(A, b))[0]),
        ]
        for solver, x_ref in cases:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                x = solver(A, b)
            assert x.dtype == jnp.float32
            assert _rel_err(x, x_ref) < 1e-6
    assert jnp.zeros(()).dtype == jnp.float32


class _Affine(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __call__(self, x):
        return self.weight @ x + self.bias


def test_block_pinv_eig_solves_each_layer(x64):
    # Each layer with parameters is solved independently with Ebar / (number of
    # those layers); the parameter-free activation layers don't form blocks.
    Chain(4)
    layers = [
        _Affine(np.ones((3, 4)), np.ones(3)),  # 15 parameters
        jnp.tanh,
        _Affine(np.ones((2, 3)), np.ones(2)),  # 8 parameters
        _Affine(np.ones((2, 2)), np.ones(2)),  # 6 parameters
        jnp.sum,
    ]
    state = Variational(Sequential(layers))
    # 12 samples: MinSR for the first block and SR for the other ones
    A, b = _problem(12, 29, jnp.float64)
    kwargs = dict(rtol=1e-2, atol=1e-3, tol_snr=1.0)
    x = block_pinv_eig(state, **kwargs)(A, b)
    blocks = np.split(np.asarray(A), [15, 23], axis=1)
    x_ref = np.concatenate([_ref_pinv_eig(Ai, b / 3, **kwargs) for Ai in blocks])
    np.testing.assert_allclose(np.asarray(x), x_ref, rtol=1e-8)


def test_block_pinv_eig_requires_sequential(x64):
    # the parameter blocks are only defined by the layers of a Sequential model
    Chain(4)
    state = Variational(_Affine(np.ones((1, 4)), np.ones(1)))
    with pytest.raises(ValueError, match="Sequential"):
        block_pinv_eig(state)
