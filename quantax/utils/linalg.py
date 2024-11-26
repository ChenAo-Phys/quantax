from typing import Tuple
import jax
import jax.numpy as jnp


@jax.custom_vjp
def det(A: jax.Array) -> jax.Array:
    return jnp.linalg.det(A)


def _det_fwd(A: jax.Array) -> Tuple[jax.Array, jax.Array]:
    lu, piv = jax.scipy.linalg.lu_factor(A)
    nperm = jnp.sum(piv != jnp.arange(piv.shape[-1]), axis=-1)
    diagonal = jnp.diagonal(lu, axis1=-2, axis2=-1)
    detA = jnp.prod(diagonal, axis=-1)
    detA = jnp.where(nperm % 2 == 1, -detA, detA)
    I = jnp.identity(A.shape[-1], dtype=A.dtype)
    I = jnp.expand_dims(I, axis=tuple(range(A.ndim - 2)))
    Ainv = jax.scipy.linalg.lu_solve((lu, piv), I)
    return detA, detA[..., None, None] * jnp.swapaxes(Ainv, axis1=-2, axis2=-1)


def _det_bwd(res: jax.Array, g: jax.Array) -> Tuple[jax.Array]:
    return (res * g[..., None, None],)


det.defvjp(_det_fwd, _det_bwd)


@jax.custom_vjp
def logdet(A: jax.Array) -> jax.Array:
    if not jnp.iscomplex(A):
        raise ValueError("`logdet` only accepts complex inputs.")
    sign, logabsdet = jnp.linalg.slogdet(A)
    return jax.lax.complex(logabsdet, jnp.angle(sign))


def _logdet_fwd(A: jax.Array) -> Tuple[jax.Array, jax.Array]:
    if not jnp.iscomplex(A):
        raise ValueError("`logdet` only accepts complex inputs.")

    lu, piv = jax.scipy.linalg.lu_factor(A)
    nperm = jnp.sum(piv != jnp.arange(piv.shape[-1]), axis=-1)
    diagonal = jnp.diagonal(lu, axis1=-2, axis2=-1)
    logdet = jax.scipy.special.logsumexp(diagonal, axis=-1)
    neg_logdet = logdet - jnp.sign(logdet.imag) * jnp.pi * 1j
    logdet = jnp.where(nperm % 2 == 1, neg_logdet, logdet)
    I = jnp.identity(A.shape[-1], dtype=A.dtype)
    I = jnp.expand_dims(I, axis=tuple(range(A.ndim - 2)))
    Ainv = jax.scipy.linalg.lu_solve((lu, piv), I)
    return logdet, jnp.swapaxes(Ainv, axis1=-2, axis2=-1)


def _logdet_bwd(res: jax.Array, g: jax.Array) -> Tuple[jax.Array]:
    return (res * g[..., None, None],)


logdet.defvjp(_logdet_fwd, _logdet_bwd)


@jax.jit
def _householder_n(x: jax.Array, n: int) -> Tuple[jax.Array, jax.Array, jax.Array]:
    arange = jnp.arange(x.size)
    xn = x[n]
    x = jnp.where(arange <= n, jnp.zeros_like(x), x)
    sigma = jnp.vdot(x, x)
    norm_x = jnp.sqrt(xn.conj() * xn + sigma)

    phase = jnp.where(xn == 0.0, 1.0, xn / jnp.abs(xn))
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

    if n == 2:
        return A[0, 1]

    if n == 4:
        a, b, c, d, e, f = A[jnp.triu_indices(n, 1)]
        return a * f - b * e + d * c

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


@jax.custom_vjp
def logpf(A: jax.Array) -> jax.Array:
    if not jnp.iscomplex(A):
        raise ValueError("`logpf` only accepts complex inputs.")

    n = A.shape[0]
    if n % 2 == 1:
        return jnp.array(0, dtype=A.dtype)

    def body_fun(i, val):
        A, logpf_val = val
        v, tau, alpha = _householder_n(A[:, i], i + 1)
        w = tau * A @ v.conj()
        A += jnp.outer(v, w) - jnp.outer(w, v)

        new_val = jnp.log((1 - tau) * jnp.where(i % 2 == 0, -alpha, 1.0))
        logpf_val = jnp.logaddexp(logpf_val, new_val)
        return A, logpf_val

    init_val = (A, jnp.array(0, dtype=A.dtype))
    A, pfaffian_val = jax.lax.fori_loop(0, A.shape[0] - 2, body_fun, init_val)
    pfaffian_val *= A[n - 2, n - 1]

    return pfaffian_val


def _logpf_fwd(A: jax.Array) -> Tuple[jax.Array, jax.Array]:
    logpfA = logpf(A)
    Ainv = jnp.linalg.inv(A)
    return logpfA, Ainv


def _logpf_bwd(res: jax.Array, g: jax.Array) -> Tuple[jax.Array]:
    return (-g * res / 2,)


logpf.defvjp(_logpf_fwd, _logpf_bwd)
