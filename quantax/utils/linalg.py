from typing import Tuple
import jax
import jax.numpy as jnp
from .array import array_set


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


def _single_pfaffian(A: jax.Array) -> jax.Array:
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


@jax.custom_vjp
def pfaffian(A: jax.Array) -> jax.Array:
    batch = A.shape[:-2]

    # By convention, pfaffian of 0 particle is 1
    if A.size == 0:
        return jnp.ones(batch, A.dtype)

    A = A.reshape(-1, *A.shape[-2:])
    pfa = jax.vmap(_single_pfaffian)(A)
    pfa = pfa.reshape(batch)
    return pfa


def _pfa_fwd(A: jax.Array) -> Tuple[jax.Array, jax.Array]:
    pfaA = pfaffian(A)
    Ainv = jnp.linalg.inv(A)
    Ainv = (Ainv - jnp.swapaxes(Ainv, -2, -1)) / 2
    return pfaA, pfaA[..., None, None] * Ainv


def _pfa_bwd(res: jax.Array, g: jax.Array) -> Tuple[jax.Array]:
    return (-res * g[..., None, None] / 2,)


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


def det_update_rows(
    inv_old: jax.Array, update: jax.Array, update_idx: jax.Array, return_inv: bool
):
    """ "
    Applies a low-rank update to a determinant of orbitals, where
    only the rows are updated

    The update to the orbitals is constructed.
    update = jnp.zeros([nparticle,nparticle]),
    update = update.at[update_idx].set(update)

    Args
    inv_old: The inverse of the orbital matrix before the update
    update: The update to the rows
    update_idx: indices of the rows to be updated
    return_inv: bool indicating whether to update the inverse

    Returns
    rat: The ratio between the updated determinant and the old determinant
    inv: The inverse of the updated determinant orbitals
    """

    eye = jnp.eye(len(update), dtype=update.dtype)
    mat = eye + update @ inv_old[:, update_idx]

    rat = det(mat)

    if return_inv:

        inv_times_update = update @ inv_old
        solve = jnp.linalg.solve(mat, inv_times_update)
        inv = inv_old - inv_old[:, update_idx] @ solve

        return rat, inv
    else:
        return rat


def det_update_gen(
    inv_old: jax.Array,
    row_update: jax.Array,
    column_update: jax.Array,
    overlap_update: jax.Array,
    row_idx: jax.Array,
    column_idx: jax.Array,
    return_inv: bool,
):
    """ "
    Applies a low-rank update to a determinant of orbitals, returning the
    ratio between the updated determinant and the old determinant as well
    as the inverse of the update orbitals.

    The update to the orbitals is an "L shaped update"
    constructed from low_rank_update_matrix.
    update = jnp.zeros([nparticle,nparticle]),
    update = update.at[row_idx].set(row_update)
    update = update.at[:, column_idx].set(column_update)
    update = update.at[row_idx, column_idx].set(overlap_update)

    Args
    inv_old: The inverse of the orbital matrix before the update
    row_update: The update to the rows
    column_update: The update to the columns
    overlap_update: The update to the section of the matrix where
    the rows and columns overlap
    row_idx: indices of the rows
    column_idx: indices of the columns
    return_inv: bool indicating whether to update the inverse

    Returns
    rat: The ratio between the updated determinant and the old determinant
    inv: The inverse of the updated determinant orbitals
    """

    row_update = array_set(row_update.T, overlap_update.T / 2, column_idx).T
    column_update = array_set(column_update, overlap_update / 2, row_idx)

    inv_times_update = row_update @ inv_old
    sliced_inv = inv_old[column_idx]

    mat11 = inv_times_update[:, row_idx]
    mat21 = inv_times_update @ column_update
    mat12 = sliced_inv[:, row_idx]
    mat22 = sliced_inv @ column_update

    mat = jnp.block([[mat11, mat21], [mat12, mat22]])

    mat = mat + jnp.eye(len(mat), dtype=mat.dtype)

    rat = det(mat)

    if return_inv:
        lhs = jnp.concatenate((inv_times_update, sliced_inv), 0)
        rhs = jnp.concatenate((inv_old[:, row_idx], inv_old @ column_update), 1)
        inv = inv_old - rhs @ jnp.linalg.solve(mat, lhs)

        return rat, inv
    else:
        return rat


def pfa_eye(rank, dtype):

    a = jnp.zeros([rank, rank], dtype=dtype)
    b = jnp.eye(rank, dtype=dtype)

    return jnp.block([[a, b], [-b, a]])


def pfa_update(
    inv_old: jax.Array, update: jax.Array, update_idx: jax.Array, return_inv: bool
):
    """
    Applies a low-rank update to a pfaffian matrix, returning the
    ratio between the updated pfaffian and the old pfaffian as well
    as the inverse of the update orbitals

    The update to the orbitals is an "L shaped update"
    constructed from low_rank_update_matrix.
    update = jnp.zeros([nparticle,nparticle]),
    update = update.at[update_idx].set(update_matrix)
    update = update - update.T

    Args
    inv_old: The inverse of the orbital matrix before the update
    update: The condensed form of the update
    update_idx: The indices indicating the rows/columns to be updated
    return_inv: bool indicating whether to update the inverse

    Returns
    rat: The ratio between the updated pfaffian and the old pfaffian
    inv: The inverse of the updated pfaffian orbitals
    """

    inv_times_update = update @ inv_old
    sliced_inv = inv_old[update_idx]

    mat11 = inv_times_update @ update.T
    mat21 = inv_times_update[:, update_idx]
    mat22 = sliced_inv[:, update_idx]

    mat = jnp.block([[mat11, mat21], [-1 * mat21.T, mat22]])

    mat = mat + pfa_eye(len(mat) // 2, dtype=mat.dtype)

    rat = pfaffian(mat)

    if return_inv:
        l = len(inv_times_update)
        if l == 1:
            update = 2 * sliced_inv.T @ inv_times_update / mat[0][1]
        else:
            inv_times_update = jnp.concatenate((inv_times_update, sliced_inv), 0)
            solve = jnp.linalg.solve(mat, inv_times_update)
            update = inv_times_update.T @ solve

        inv = inv_old + update
        inv = (inv - inv.T) / 2

        return rat, inv
    else:
        return rat
