from typing import Callable, Optional
import jax
import jax.numpy as jnp
from jax.lax import cond
from jax.scipy.linalg import solve, eigh
from jax.scipy.sparse.linalg import cg


class lstsq_shift_cg:
    def __init__(
        self,
        diag_shift: float = 0.01,
        tol: float = 1e-5,
        atol: float = 0.0,
        maxiter: Optional[int] = None,
    ):
        @jax.jit
        def S_apply(A, x):
            S_apply_x = jnp.einsum("sk,sl,l->k", A, A, x)
            S_apply_x += diag_shift * jnp.einsum("sk,sk,k->k", A, A, x)
            return S_apply_x

        self.S_apply = S_apply
        self.tol = tol
        self.atol = atol
        self.maxiter = maxiter

    def __call__(self, A: jax.Array, b: jax.Array) -> jax.Array:
        F = jnp.einsum("sk,s->k", A.conj(), b)
        Apply = lambda x: self.S_apply(A, x)
        x = cg(Apply, F, tol=self.tol, atol=self.atol, maxiter=self.maxiter)
        return x[0]


def minnorm_shift_eig(ashift: float = 1e-4) -> Callable:
    @jax.jit
    def solution(A: jax.Array, b: jax.Array) -> jax.Array:
        T = A @ A.conj().T
        T += ashift * jnp.identity(T.shape[0], T.dtype)
        T_inv_b = solve(T, b, assume_a="her")
        x = A.conj().T @ T_inv_b
        return x

    return solution


@jax.jit
def _get_eigs_inv(vals: jax.Array, tol: Optional[float], atol: float) -> jax.Array:
    vals_abs = jnp.abs(vals)
    if tol is None:
        if vals_abs.dtype == jnp.float64:
            tol = 1e-12
        elif vals_abs.dtype == jnp.float32:
            tol = 1e-6
        elif vals_abs.dtype == jnp.float16:
            tol = 1e-3
        else:
            raise ValueError(f"Invalid dtype {vals_abs.dtype} for inversion.")

    inv_factor = 1 + ((tol * jnp.max(vals_abs) + atol) / vals_abs) ** 6
    eigs_inv = 1 / (vals * inv_factor)
    return jnp.where(vals_abs > 0.0, eigs_inv, 0.0)


def pinvh_solve(tol: Optional[float] = None, atol: float = 0.0) -> Callable:
    @jax.jit
    def solve(H: jax.Array, b: jax.Array) -> jax.Array:
        eig_vals, U = eigh(H)
        eig_inv = _get_eigs_inv(eig_vals, tol, atol)
        return jnp.einsum("rs,s,ts,t->r", U, eig_inv, U.conj(), b)

    return solve


@jax.jit
def _sum_without_noise(inputs: jax.Array, tol_snr: float) -> jax.Array:
    """
    Noise truncation, see https://arxiv.org/pdf/2108.03409.pdf
    """
    x = jnp.sum(inputs, axis=0)
    x_mean = x / inputs.shape[0]
    x_var = jnp.abs(inputs - x_mean[None, :]) ** 2
    x_var = jnp.sqrt(jnp.mean(x_var, axis=0) / inputs.shape[0])
    snr = jnp.abs(x_mean) / x_var
    x = cond(tol_snr > 1e-6, lambda a: a / (1 + (tol_snr / snr) ** 6), lambda a: a, x)
    return x


def minnorm_pinv_eig(
    tol: Optional[float] = None, atol: float = 0.0, tol_snr: float = 0.0
) -> Callable:
    @jax.jit
    def solve(A: jax.Array, b: jax.Array) -> jax.Array:
        T = A @ A.conj().T
        # T_inv_b = pinv_solve(T, b, tol, atol, tol_snr)
        # x = jnp.einsum("rk,r->k", A.conj(), T_inv_b)
        eig_vals, U = eigh(T)
        eig_inv = _get_eigs_inv(eig_vals, tol, atol)
        rho_ts = jnp.einsum("ts,t->ts", U.conj(), b)
        rho = _sum_without_noise(rho_ts, tol_snr)
        x = jnp.einsum("rk,rs,s,s->k", A.conj(), U, eig_inv, rho)
        return x

    return solve


def lstsq_pinv_eig(
    tol: Optional[float] = None, atol: float = 0.0, tol_snr: float = 0.0
) -> Callable:
    @jax.jit
    def solve(A: jax.Array, b: jax.Array) -> jax.Array:
        S = A.conj().T @ A
        eig_vals, V = eigh(S)
        eig_inv = _get_eigs_inv(eig_vals, tol, atol)
        rho_sk = jnp.einsum("lk,sl,s->sk", V.conj(), A.conj(), b)
        rho = _sum_without_noise(rho_sk, tol_snr)
        return jnp.einsum("kl,l,l->k", V, eig_inv, rho)

    return solve


def auto_pinv_eig(
    tol: Optional[float] = None, atol: float = 0.0, tol_snr: float = 0.0
) -> Callable:
    minnorm_solver = minnorm_pinv_eig(tol, atol, tol_snr)
    lstsq_solver = lstsq_pinv_eig(tol, atol, tol_snr)

    @jax.jit
    def solve(A: jax.Array, b: jax.Array) -> jax.Array:
        if A.shape[0] < A.shape[1]:
            return minnorm_solver(A, b)
        else:
            return lstsq_solver(A, b)

    return solve


def minsr_pinv_eig(
    tol: Optional[float] = None, atol: float = 0.0, tol_snr: float = 0.0
) -> Callable:
    @jax.jit
    def solve(T: jax.Array, b: jax.Array) -> jax.Array:
        eig_vals, U = eigh(T)
        eig_inv = _get_eigs_inv(eig_vals, tol, atol)
        rho_ts = jnp.einsum("ts,t->ts", U.conj(), b)
        rho = _sum_without_noise(rho_ts, tol_snr)
        x = jnp.einsum("rs,s,s->r", U, eig_inv, rho)
        return x

    return solve


def sgd_solver() -> Callable:
    @jax.jit
    def solve(A: jax.Array, b: jax.Array) -> jax.Array:
        return jnp.einsum("sk,s->k", A.conj(), b) / b.shape[0]

    return solve
