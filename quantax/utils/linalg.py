from typing import Tuple, Optional
import jax
import jax.numpy as jnp


@jax.custom_vjp
def det(A: jax.Array) -> jax.Array:
    return jnp.linalg.det(A)


def _det_fwd(A: jax.Array) -> Tuple[jax.Array, jax.Array]:
    detA = det(A)
    Ainv = jnp.linalg.inv(A)
    return detA, detA * Ainv.T


def _det_bwd(res: jax.Array, g: jax.Array) -> Tuple[jax.Array]:
    return (g * res,)


det.defvjp(_det_fwd, _det_bwd)


@jax.custom_vjp
def slogdet(A: jax.Array):
    return jnp.linalg.slogdet(A)


def _slogdet_fwd(A: jax.Array):
    return slogdet(A), jnp.linalg.inv(A).T


def _slogdet_bwd(res: jax.Array, g) -> Tuple[jax.Array]:
    return (g[1] * res,)


slogdet.defvjp(_slogdet_fwd, _slogdet_bwd)


@jax.jit
def _householder_n(x: jax.Array, n: int) -> Tuple[jax.Array, jax.Array, jax.Array]:
    arange = jnp.arange(x.size)
    xn = x[n]
    x = jnp.where(arange <= n, jnp.zeros_like(x), x)
    sigma = jnp.vdot(x, x)
    norm_x = jnp.sqrt(xn.conj() * xn + sigma)

    phase = xn / jnp.abs(xn)
    vn = xn + phase * norm_x
    alpha = -phase * norm_x

    v = jnp.where(arange == n, vn, x)
    v /= jnp.linalg.norm(v)

    cond = sigma == 0.0
    v = jnp.where(cond, jnp.zeros_like(x), v)
    tau = jnp.where(cond, 0, 2)
    alpha = jnp.where(cond, xn, alpha)

    return v, tau, alpha


@jax.custom_vjp
def pfaffian(A: jax.Array) -> jax.Array:
    n = A.shape[0]
    if n % 2 == 1:
        return jnp.array(0, dtype=A.dtype)

    def body_fun(i, val):
        A, pfaffian_val = val
        v, tau, alpha = _householder_n(A[:, i], i + 1)
        w = tau * A @ v.conj()
        A += jnp.outer(v, w) - jnp.outer(w, v)

        pfaffian_val *= 1 - tau
        pfaffian_val *= jnp.where(i % 2 == 0, -alpha, 1.0)
        return A, pfaffian_val

    init_val = (A, jnp.array(1.0, dtype=A.dtype))
    A, pfaffian_val = jax.lax.fori_loop(0, A.shape[0] - 2, body_fun, init_val)
    pfaffian_val *= A[n - 2, n - 1]

    return pfaffian_val


def _pfa_fwd(A: jax.Array) -> Tuple[jax.Array, jax.Array]:
    pfaA = pfaffian(A)
    Ainv = jnp.linalg.inv(A)
    return pfaA, pfaA * Ainv


def _pfa_bwd(res: jax.Array, g: jax.Array) -> Tuple[jax.Array]:
    return (-g * res / 2,)


pfaffian.defvjp(_pfa_fwd, _pfa_bwd)