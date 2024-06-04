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
    is_fermion: bool = eqx.field(static=True)

    def __init__(self, Nparticle: Optional[int] = None):
        sites = get_sites()
        self.is_fermion = sites.is_fermion
        N = sites.nsites
        if Nparticle is None:
            Nparticle = N
        # https://www.quora.com/What-is-the-variance-of-det-A-where-A-is-an-n-n-matrix-whose-elements-are-randomly-and-independently-drawn-from-1-1
        # scale = sqrt(1 / (n!)^(1/n)) ~ sqrt(e/n)
        dtype = get_params_dtype()
        scale = np.sqrt(np.e / Nparticle, dtype=dtype)
        self.U = jr.normal(get_subkeys(), (2 * N, Nparticle), dtype) * scale

    def get_U(self, x: jax.Array) -> jax.Array:
        Nparticle = self.U.shape[1]
        if not self.is_fermion:
            particle = jnp.ones_like(x)
            hole = -particle
            x_up = jnp.where(x > 0, particle, hole)
            x_down = jnp.where(x <= 0, particle, hole)
            x = jnp.concatenate([x_up, x_down])

        idx = jnp.argsort(x <= 0, stable=True)[:Nparticle]
        return self.U[idx, :]

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        U = self.get_U(x)
        return det(U)

    def rescale(self, maximum: jax.Array) -> Determinant:
        Ne = self.U.shape[1]
        U = self.U / maximum ** (1 / Ne)
        return eqx.tree_at(lambda tree: tree.U, self, U)
