import jax
import jax.numpy as jnp
from ..global_defs import get_sites, Lattice


def fermion_idx(
    x: jax.Array, separate_spins: bool = False
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """
    Get the indices of occupied fermion sites.

    :param x:
        A 1D array representing the fermion configuration. For spinful fermions,
        the first half corresponds to spin-up sites and the second half to spin-down sites.

    :param separate_spins:
        Whether to return the indices of spin-up and spin-down fermions separately.
    """
    sites = get_sites()
    if separate_spins and not sites.is_spinful:
        raise ValueError("Cannot separate spins for spinless fermions.")

    if sites.is_fermion:
        occ = jnp.where(x > 0, 1, 0)
    else:
        # A spin maps to two fermionic modes: spin-up and spin-down occupation.
        occ = jnp.concatenate([jnp.where(x > 0, 1, 0), jnp.where(x <= 0, 1, 0)])

    if separate_spins:
        occ_up, occ_dn = jnp.split(occ, 2)
        if isinstance(sites.Nparticles, tuple):
            Nup, Ndn = sites.Nparticles
        else:
            Nup, Ndn = None, None
        idx_up = jnp.flatnonzero(occ_up, size=Nup).astype(jnp.uint16)
        idx_dn = jnp.flatnonzero(occ_dn, size=Ndn).astype(jnp.uint16)
        return idx_up, idx_dn
    else:
        return jnp.flatnonzero(occ, size=sites.Ntotal).astype(jnp.uint16)


def changed_inds(
    s: jax.Array, s_old: jax.Array, nhops: int
) -> tuple[jax.Array, jax.Array]:
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

    :param idx:
        The sorted indices of the occupied fermions.

    :param idx_annihilate:
        The indices of the annihilated fermions.

    :param idx_create:
        The indices of the created fermions.
    """
    # For each hop, count the occupied fermions strictly between its endpoints,
    # excluding the other hopping fermions (which are themselves moving).
    lo = jnp.minimum(idx_annihilate, idx_create)
    hi = jnp.maximum(idx_annihilate, idx_create)
    between = (idx[:, None] > lo) & (idx[:, None] < hi)
    moving = jnp.isin(idx, idx_annihilate)
    parity = jnp.sum(between & ~moving[:, None])
    return 1 - 2 * (parity % 2)


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
