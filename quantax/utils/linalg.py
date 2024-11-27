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

def _det_update_rows(update: jax.Array, idx: jax.Array, old_inv: jax.Array):
    """"
    Returns update matrix for determinant update of rows 
    psi1 = det(phi1), psi2 = det(phi2) = psi1*det(update_matrix)
    phi2[idx] - phi1[idx] = update 
    """

    eye = jnp.eye(len(update), dtype=update.dtype)
    return eye + update @ old_inv[:, idx]

def _inv_update_rows(
    update: jax.Array, 
    idx: jax.Array, 
    old_inv: jax.Array, 
    update_matrix: jax.Array
):
    """"
    Returns inverse update for update of rows 
    """

    inv_times_update = update @ old_inv
    solve = jnp.linalg.solve(update_matrix, inv_times_update)

    return old_inv - old_inv[:, idx] @ solve

def _det_update_gen(
    row_update: jax.Array, 
    column_update: jax.Array, 
    overlap_update: jax.Array,
    row_idx: jax.Array, 
    column_idx: jax.Array,   
    old_inv: jax.Array
):
    """"
    Returns low rank determinant update where update 
    is "L shaped" with form 

    overlap_update  row_update
    column_update

    Overlap matrix is where both rows and columns are updated
        
    """
    
    row_update = array_set(row_update.T, 0, column_idx).T
    column_update = array_set(column_update, overlap_update, row_idx)

    mat11 = row_update @ old_inv[:, row_idx]
    mat21 = row_update @ old_inv @ column_update
    mat12 = old_inv[column_idx][:, row_idx]
    mat22 = old_inv[column_idx] @ column_update

    mat = jnp.block([[mat11, mat21], [mat12, mat22]])

    return mat + jnp.eye(len(mat), dtype=mat.dtype)

def _inv_update_gen(
    row_update: jax.Array, 
    column_update: jax.Array, 
    overlap_update: jax.Array,
    row_idx: jax.Array, 
    column_idx: jax.Array,   
    old_inv: jax.Array,
    update_matrix: jax.Array,
):
    
    row_update = array_set(row_update.T, 0, column_idx).T
    column_update = array_set(column_update, overlap_update, row_idx)
    
    lhs = jnp.concatenate((row_update @ old_inv, old_inv[column_idx]), 0)
    rhs = jnp.concatenate((old_inv[:, row_idx], old_inv @ column_update), 1)

    return old_inv - rhs @ jnp.linalg.solve(update_matrix, lhs)

def _pfa_update(
    update: jax.Array, 
    overlap_update: jax.Array,
    idx: jax.Array, 
    old_inv: jax.Array
):
    """"
    Returns low rank pfaffian update where update 
    is "L shaped" with form 

    overlap_update  update
    - 1*update.T

    Overlap matrix is where both rows and columns are updated
        
    """
    
    update = array_set(update.T, overlap_update.T/2, idx).T

    mat11 = update @ old_inv @ update.T
    mat21 = update @ old_inv[:,idx]
    mat22 = old_inv[idx][:, idx]

    mat = jnp.block([[mat11, mat21], [-1*mat21.T, mat22]])

    return mat - _pfa_eye(len(mat)//2, dtype=mat.dtype)

def _inv_update_pfa(
    update: jax.Array, 
    overlap_update: jax.Array,
    idx: jax.Array, 
    old_inv: jax.Array,
    update_matrix: jax.Array,
):
    
    update = array_set(update.T, overlap_update.T/2, idx).T
    
    inv_times_update = jnp.concatenate((update @ old_inv, old_inv[idx]), 0)
    solve = jnp.linalg.solve(update_matrix, inv_times_update)
    inv = old_inv + inv_times_update.T @ solve

    return (inv - inv.T) / 2

def _pfa_eye(rank, dtype):

    a = jnp.zeros([rank, rank], dtype=dtype)
    b = jnp.eye(rank, dtype=dtype)

    return jnp.block([[a, -1 * b], [b, a]])
