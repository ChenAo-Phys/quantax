from typing import Callable, Optional
import jax
import jax.numpy as jnp
from jax.lax import cond
from jax.scipy.linalg import solve, eigh
from jax.scipy.sparse.linalg import cg
from quantax.utils import to_global_array, array_extend


def _get_rtol(dtype: jnp.dtype) -> float:
    if dtype == jnp.float64:
        rtol = 1e-12
    elif dtype == jnp.float32:
        rtol = 1e-6
    elif dtype == jnp.float16:
        rtol = 1e-3
    else:
        raise ValueError(f"Invalid dtype {dtype} for determining eigenvalue tolerance.")
    return rtol


class lstsq_shift_cg:
    def __init__(
        self,
        diag_shift: float = 0.01,
        rtol: float = 1e-5,
        atol: float = 0.0,
        maxiter: Optional[int] = None,
    ):
        @jax.jit
        def S_apply(A, x):
            S_apply_x = jnp.einsum("sk,sl,l->k", A, A, x)
            S_apply_x += diag_shift * jnp.einsum("sk,sk,k->k", A, A, x)
            return S_apply_x

        self.S_apply = S_apply
        self.rtol = rtol
        self.atol = atol
        self.maxiter = maxiter

    def __call__(self, A: jax.Array, b: jax.Array) -> jax.Array:
        F = jnp.einsum("sk,s->k", A.conj(), b)
        Apply = lambda x: self.S_apply(A, x)
        x = cg(Apply, F, tol=self.rtol, atol=self.atol, maxiter=self.maxiter)
        return x[0]


def minnorm_shift_eig(rshift: Optional[float] = None, ashift: float = 1e-4) -> Callable:
    @jax.jit
    def solution(A: jax.Array, b: jax.Array) -> jax.Array:
        T = A @ A.conj().T
        trace = jnp.linalg.trace(T).real
        rel_shift = _get_rtol(trace.dtype) if rshift is None else rshift
        shift = rel_shift * trace + ashift
        T += shift * jnp.identity(T.shape[0], T.dtype)
        T_inv_b = solve(T, b, assume_a="pos")  # cholesky solver is used internally
        x = A.conj().T @ T_inv_b
        return x

    return solution


def lstsq_shift_eig(rshift: Optional[float] = None, ashift: float = 1e-4) -> Callable:
    @jax.jit
    def solution(A: jax.Array, b: jax.Array) -> jax.Array:
        S = A.conj().T @ A
        F = A.conj().T @ b
        trace = jnp.linalg.trace(S).real
        rel_shift = _get_rtol(trace.dtype) if rshift is None else rshift
        shift = rel_shift * trace + ashift
        S += shift * jnp.identity(S.shape[0], S.dtype)
        x = solve(S, F, assume_a="pos")  # cholesky solver is used internally
        return x

    return solution


def auto_shift_eig(rshift: Optional[float] = None, ashift: float = 1e-4) -> Callable:
    minnorm_solver = minnorm_shift_eig(rshift, ashift)
    lstsq_solver = lstsq_shift_eig(rshift, ashift)

    @jax.jit
    def solve(A: jax.Array, b: jax.Array) -> jax.Array:
        if A.shape[0] < A.shape[1]:
            return minnorm_solver(A, b)
        else:
            return lstsq_solver(A, b)

    return solve


@jax.jit
def _get_eigs_inv(vals: jax.Array, rtol: Optional[float], atol: float) -> jax.Array:
    vals_abs = jnp.abs(vals)
    if rtol is None:
        rtol = _get_rtol(vals_abs.dtype)
    inv_factor = 1 + ((rtol * jnp.max(vals_abs) + atol) / vals_abs) ** 6
    eigs_inv = 1 / (vals * inv_factor)
    return jnp.where(vals_abs > 0.0, eigs_inv, 0.0)


def pinvh_solve(rtol: Optional[float] = None, atol: float = 0.0) -> Callable:
    @jax.jit
    def solve(H: jax.Array, b: jax.Array) -> jax.Array:
        eig_vals, U = eigh(H)
        eig_inv = _get_eigs_inv(eig_vals, rtol, atol)
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
    rtol: Optional[float] = None, atol: float = 0.0, tol_snr: float = 0.0
) -> Callable:
    @jax.jit
    def solve(A: jax.Array, b: jax.Array) -> jax.Array:

        Adag = A.T.conj()
        ndevices = jax.device_count()
        Adag = array_extend(Adag, ndevices)
        Adag = to_global_array(Adag)

        T = Adag.conj().T @ Adag
        # T_inv_b = pinv_solve(T, b, tol, atol, tol_snr)
        # x = jnp.einsum("rk,r->k", A.conj(), T_inv_b)
        eig_vals, U = eigh(T)
        eig_inv = _get_eigs_inv(eig_vals, rtol, atol)
        rho_ts = jnp.einsum("ts,t->ts", U.conj(), b)
        rho = _sum_without_noise(rho_ts, tol_snr)
        x = jnp.einsum("rk,rs,s,s->k", A.conj(), U, eig_inv, rho)
        return x

    return solve


def lstsq_pinv_eig(
    rtol: Optional[float] = None, atol: float = 0.0, tol_snr: float = 0.0
) -> Callable:
    @jax.jit
    def solve(A: jax.Array, b: jax.Array) -> jax.Array:
        S = A.conj().T @ A
        eig_vals, V = eigh(S)
        eig_inv = _get_eigs_inv(eig_vals, rtol, atol)
        rho_sk = jnp.einsum("lk,sl,s->sk", V.conj(), A.conj(), b)
        rho = _sum_without_noise(rho_sk, tol_snr)
        return jnp.einsum("kl,l,l->k", V, eig_inv, rho)

    return solve


def auto_pinv_eig(
    rtol: Optional[float] = None, atol: float = 0.0, tol_snr: float = 0.0
) -> Callable:
    """
    Obtain the least-square minimum-norm solver for the linear equation
    :math:`Ax=b` using pseudo-inverse. It automatically chooses between
    :math:`x = (A^† A)^{-1} A^† b` and :math:`x = A^† (A A^†)^{-1} b`, which respectively
    correspond to SR and MinSR.

    :param rtol:
        The relative tolerance for pseudo-inverse. Default to be :math:`10^{-12}` for
        double precision and :math:`10^{-6}` for single precision.

    :param atol:
        The absolute tolerance for pseudo-inverse, default to 0.

    :param tol_snr:
        The tolerence of signal-to-noise ratio (SNR), default to 0 which means no regularization
        based on SNR. For details see `this paper <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.100503>`_.

    :return:
        A solver function with two arguments A and b and one output x as the solution of
        :math:`A x = b`.
    """
    minnorm_solver = minnorm_pinv_eig(rtol, atol, tol_snr)
    lstsq_solver = lstsq_pinv_eig(rtol, atol, tol_snr)

    @jax.jit
    def solve(A: jax.Array, b: jax.Array) -> jax.Array:
        if A.shape[0] < A.shape[1]:
            return minnorm_solver(A, b)
        else:
            return lstsq_solver(A, b)

    return solve


def minsr_pinv_eig(
    rtol: Optional[float] = None, atol: float = 0.0, tol_snr: float = 0.0
) -> Callable:
    """
    Obtain the pseudo-inverse solver for the inverse problem in MinSR
    :math:`Tx=b`, where :math:`T` is a Hermitian matrix.

    :param rtol:
        The relative tolerance for pseudo-inverse. Default to be :math:`10^{-12}` for
        double precision and :math:`10^{-6}` for single precision.

    :param atol:
        The absolute tolerance for pseudo-inverse, default to 0.

    :param tol_snr:
        The tolerence of signal-to-noise ratio (SNR), default to 0 which means no regularization
        based on SNR. For details see `this paper <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.100503>`_.

    :return:
        A solver function with two arguments T and b and one output x as the solution of
        :math:`T x = b`.
    """

    @jax.jit
    def solve(T: jax.Array, b: jax.Array) -> jax.Array:
        eig_vals, U = eigh(T)
        eig_inv = _get_eigs_inv(eig_vals, rtol, atol)
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
