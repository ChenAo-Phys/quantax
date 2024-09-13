from __future__ import annotations
from typing import Optional
from jaxtyping import Key
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from ..nn import Sequential
from ..utils import det, pfaffian
from ..global_defs import get_sites, get_subkeys, is_default_cpl


def _get_fermion_idx(x: jax.Array, Nparticle: int) -> jax.Array:
    particle = jnp.ones_like(x)
    hole = jnp.zeros_like(x)
    if get_sites().is_fermion:
        x = jnp.where(x > 0, particle, hole)
    else:
        particle = jnp.ones_like(x)
        hole = jnp.zeros_like(x)
        x_up = jnp.where(x > 0, particle, hole)
        x_down = jnp.where(x <= 0, particle, hole)
        x = jnp.concatenate([x_up, x_down])
    idx = jnp.flatnonzero(x, size=Nparticle)
    return idx


class Determinant(eqx.Module):
    U: jax.Array
    Nparticle: int
    holomorphic: bool = eqx.field(static=True)

    def __init__(self, Nparticle: Optional[int] = None, dtype: jnp.dtype = jnp.float32):
        N = get_sites().nsites
        self.Nparticle = N if Nparticle is None else Nparticle
        # https://www.quora.com/Suppose-A-is-an-NxN-matrix-whose-entries-are-independent-random-variables-with-mean-0-and-variance-%CF%83-2-What-is-the-mean-and-variance-of-X-det-A
        # scale = sqrt(1 / (n!)^(1/n)) ~ sqrt(e/n)
        scale = np.sqrt(np.e / Nparticle, dtype=dtype)
        is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
        shape = (2 * N, Nparticle)
        if is_default_cpl() and not is_dtype_cpl:
            shape = (2,) + shape
        self.U = jr.normal(get_subkeys(), shape, dtype) * scale
        self.holomorphic = is_default_cpl() and is_dtype_cpl

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        U = self.U if self.U.ndim == 2 else jax.lax.complex(self.U[0], self.U[1])
        idx = _get_fermion_idx(x, self.Nparticle)
        return det(U[idx, :])

    def rescale(self, maximum: jax.Array) -> Determinant:
        U = self.U / maximum.astype(self.U.dtype) ** (1 / self.Nparticle)
        return eqx.tree_at(lambda tree: tree.U, self, U)


class Pfaffian(eqx.Module):
    F: jax.Array
    Nparticle: int
    holomorphic: bool = eqx.field(static=True)

    def __init__(self, Nparticle: Optional[int] = None, dtype: jnp.dtype = jnp.float32):
        N = get_sites().nsites
        self.Nparticle = N if Nparticle is None else Nparticle

        scale = np.sqrt(np.e / Nparticle, dtype=dtype)
        is_dtype_cpl = jnp.issubdtype(dtype, jnp.complexfloating)
        shape = (N * (2 * N - 1),)
        if is_default_cpl() and not is_dtype_cpl:
            shape = (2,) + shape
        self.F = jr.normal(get_subkeys(), shape, dtype) * scale
        self.holomorphic = is_default_cpl() and is_dtype_cpl

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        F = self.F if self.F.ndim == 1 else jax.lax.complex(self.F[0], self.F[1])
        N = get_sites().nsites
        F_full = jnp.zeros((2 * N, 2 * N), F.dtype)
        F_full = F_full.at[jnp.tril_indices(2 * N, -1)].set(F)
        F_full = F_full - F_full.T
        idx = _get_fermion_idx(x, self.Nparticle)
        return pfaffian(F_full[idx, :][:, idx])

    def rescale(self, maximum: jax.Array) -> Determinant:
        F = self.F / maximum.astype(self.F.dtype) ** (2 / self.Nparticle)
        return eqx.tree_at(lambda tree: tree.F, self, F)
