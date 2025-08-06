from typing import Tuple
import jax
import jax.numpy as jnp
from ..global_defs import get_sites


def fermion_idx(x: jax.Array) -> jax.Array:
    """
    Get the indices of occupied fermion sites.
    """
    particle = jnp.ones_like(x)
    hole = jnp.zeros_like(x)
    if get_sites().is_fermion:
        x = jnp.where(x > 0, particle, hole)
    else:
        x_up = jnp.where(x > 0, particle, hole)
        x_down = jnp.where(x <= 0, particle, hole)
        x = jnp.concatenate([x_up, x_down])
    idx = jnp.flatnonzero(x, size=get_sites().Ntotal).astype(jnp.uint16)
    return idx


def changed_inds(
    s: jax.Array, s_old: jax.Array, nhops: int
) -> Tuple[jax.Array, jax.Array]:
    """
    Get the indices of the hopping fermions.
    """
    annihilate = jnp.logical_and(s <= 0, s_old > 0)
    idx_annihilate = jnp.flatnonzero(annihilate, size=nhops, fill_value=s.size)
    create = jnp.logical_and(s > 0, s_old <= 0)
    idx_create = jnp.flatnonzero(create, size=nhops, fill_value=s.size)
    return idx_annihilate.astype(jnp.uint16), idx_create.astype(jnp.uint16)


def permute_sign(
    idx: jax.Array, idx_annihilate: jax.Array, idx_create: jax.Array
) -> jax.Array:
    """
    Get the sign change due to fermion hopping.
    """
    parity = 0
    for idx1, idx2 in zip(idx_annihilate, idx_create):
        cond1 = jnp.logical_and(idx > idx1, idx < idx2)
        cond2 = jnp.logical_and(idx < idx1, idx > idx2)
        parity += jnp.sum(jnp.logical_or(cond1, cond2))
    parity_sign = 1 - 2 * (parity % 2)
    return parity_sign