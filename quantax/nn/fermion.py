from typing import Tuple, Union
import jax
import jax.numpy as jnp
from ..global_defs import get_sites, Lattice


def fermion_idx(
    x: jax.Array, separate_spins: bool = False
) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
    """
    Get the indices of occupied fermion sites.

    :param x:
        A 1D array representing the fermion configuration. For spinful fermions,
        the first half corresponds to spin-up sites and the second half to spin-down sites.

    :param separate_spins:
        Whether to return the indices of spin-up and spin-down fermions separately.
    """
    sites = get_sites()
    particle = jnp.ones_like(x)
    hole = jnp.zeros_like(x)
    if sites.is_fermion:
        x = jnp.where(x > 0, particle, hole)
        if separate_spins:
            x_up, x_dn = jnp.split(x, 2)
    else:
        x_up = jnp.where(x > 0, particle, hole)
        x_dn = jnp.where(x <= 0, particle, hole)
        if not separate_spins:
            x = jnp.concatenate([x_up, x_dn])

    if separate_spins:
        if not sites.is_spinful:
            raise ValueError("Cannot separate spins for spinless fermions.")
        if isinstance(sites.Nparticles, tuple):
            Nup, Ndn = sites.Nparticles
        else:
            Nup, Ndn = None, None
        idx_up = jnp.flatnonzero(x_up, size=Nup).astype(jnp.uint16)
        idx_dn = jnp.flatnonzero(x_dn, size=Ndn).astype(jnp.uint16)
        return idx_up, idx_dn
    else:
        return jnp.flatnonzero(x, size=sites.Ntotal).astype(jnp.uint16)


def changed_inds(
    s: jax.Array, s_old: jax.Array, nhops: int
) -> Tuple[jax.Array, jax.Array]:
    """
    Get the indices of the hopping fermions.

    :param s:
        A 1D array representing the new fermion configuration.

    :param s_old:
        A 1D array representing the old fermion configuration.

    :param nhops:
        The number of hopping fermions.
    """
    if not get_sites().is_fermion:
        s = jnp.concatenate([s, -s])
        s_old = jnp.concatenate([s_old, -s_old])

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

    :param idx_annihilate:
        The indices of the annihilated fermions.

    :param idx_create:
        The indices of the created fermions.
    """
    parity = jnp.array(0)
    for idx1, idx2 in zip(idx_annihilate, idx_create):
        cond1 = jnp.logical_and(idx > idx1, idx < idx2)
        cond2 = jnp.logical_and(idx < idx1, idx > idx2)
        parity += jnp.sum(jnp.logical_or(cond1, cond2))
    parity_sign = 1 - 2 * (parity % 2)
    return parity_sign


def fermion_inverse_sign(s: jax.Array) -> jax.Array:
    """
    Usual fermion ordering goes from site 0 to site N-1, while QuSpin orders from
    site N-1 to site 0. This function computes the sign difference between these two
    orderings.
    """
    n_el = jnp.sum(s > 0).astype(jnp.int32)
    return (-1) ** (n_el * (n_el - 1) // 2)


def fermion_reorder_sign(s: jax.Array) -> jax.Array:
    """
    Compute the sign difference between two fermionic orderings, one having channel dimension
    first, the other having channel dimension last.
    """
    sites = get_sites()
    nchannels = 2 if sites.is_spinful else 1
    if isinstance(sites, Lattice):
        nchannels *= sites.shape[0]
    s = s.reshape(nchannels, -1)
    s = jnp.where(s > 0, 1, 0).astype(jnp.int32)

    n_el = jnp.cumulative_sum(s[::-1], axis=0, include_initial=True)[-2::-1]
    n_el = jnp.cumulative_sum(n_el, axis=1, include_initial=True)[:, :-1]
    n_exchange = jnp.sum(s * n_el)
    sign = 1 - 2 * (n_exchange % 2)
    return sign
