from __future__ import annotations
from typing import Optional
from jaxtyping import Key
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from ..utils import det
from ..global_defs import get_sites, get_subkeys, get_params_dtype


class Determinant(eqx.Module):
    U: jax.Array

    def __init__(self, Ne: Optional[int] = None):
        N = get_sites().nsites
        if Ne is None:
            Ne = N
        shape = (2 * N, Ne)
        # https://www.quora.com/What-is-the-variance-of-det-A-where-A-is-an-n-n-matrix-whose-elements-are-randomly-and-independently-drawn-from-1-1
        # scale = sqrt(1 / (n!)^(1/n)) ~ sqrt(e/n)
        scale = np.sqrt(np.e / Ne)
        dtype = get_params_dtype()
        self.U = jr.normal(get_subkeys(), shape, dtype) * scale

    def get_U(self, x: jax.Array) -> jax.Array:
        N = self.U.shape[0] // 2
        idx = jnp.arange(N)
        idx = jnp.where(x > 0, idx, idx + N)
        arg = jnp.argsort(x <= 0)
        idx = idx[arg]
        return self.U[idx, :]

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        U = self.get_U(x)
        return det(U)

    def rescale(self, maximum: jax.Array) -> Determinant:
        Ne = self.U.shape[1]
        U = self.U * maximum ** (1/Ne)
        return eqx.tree_at(lambda tree: tree.U, self, U)
