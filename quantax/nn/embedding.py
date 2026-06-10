from typing import Sequence
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.typing import DTypeLike
import equinox as eqx
from ..global_defs import get_subkeys, get_lattice, PARTICLE_TYPE


def input_to_index(s: jax.Array) -> jax.Array:
    r"""
    Convert a :math:`\pm 1` configuration into a per-unit-cell state index.

    The occupations of all sites (and spin species) within a unit cell are packed
    into the bits of a single integer, giving one index per cell in
    ``[0, 1 << s_per_cell)``, where ``s_per_cell`` is the number of binary degrees
    of freedom in a cell (``lattice.shape[0]`` for spin, twice that for spinful
    fermions).

    :param s:
        The configuration, a :math:`\pm 1` array of length ``Nsites`` (spin or
        spinless fermion) or ``2 * Nsites`` (spinful fermion, ordered as spin-up
        sites then spin-down).

    :return:
        An integer array with one entry per unit cell, i.e. of length
        ``prod(lattice.shape[1:])``.
    """
    lattice = get_lattice()
    s = s.reshape(-1, lattice.Nsites)
    bits = jnp.where(s > 0, 1, 0).astype(jnp.int32)

    # Sites are laid out as (s_per_cell, *spatial); gather the bits of one cell
    # (over species and sublattice sites) and pack them into a single integer.
    nspecies = bits.shape[0]
    s_per_cell = lattice.shape[0]
    bits = bits.reshape(nspecies * s_per_cell, -1)
    power = 1 << jnp.arange(bits.shape[0], dtype=jnp.int32)
    index = jnp.sum(bits * power[:, None], axis=0)

    return index


class Embedding(eqx.Module):
    r"""
    Learned embedding of a configuration onto the lattice grid.

    Each site's local state (see `input_to_index`) selects a column of a learnable
    table ``Et``, producing an embedding laid out on the spatial grid
    ``lattice.shape[1:]``. An optional sublattice-periodic bias ``Ep`` is tiled over
    the grid and added.
    """

    d: int
    Et: jax.Array
    Ep: jax.Array | None

    def __init__(
        self,
        d: int,
        Ep_sublattice: Sequence[int] | None = None,
        dtype: DTypeLike = jnp.float32,
    ):
        r"""
        :param d:
            The embedding dimension (number of output channels per site).

        :param Ep_sublattice:
            The spatial shape of the periodic bias ``Ep``, tiled over the grid.
            Default to ``None``, i.e. no bias.

        :param dtype:
            The data type of the parameters. Default to ``float32``.
        """
        self.d = d
        keyt, keyp = get_subkeys(2)

        lattice = get_lattice()
        s_per_cell = lattice.shape[0]
        if lattice.particle_type == PARTICLE_TYPE.spinful_fermion:
            s_per_cell *= 2
        self.Et = jr.normal(keyt, (d, 1 << s_per_cell), dtype=dtype)

        if Ep_sublattice is None:
            self.Ep = None
        else:
            self.Ep = jr.normal(keyp, (d, *Ep_sublattice), dtype=dtype)

    def __call__(self, s: jax.Array) -> jax.Array:
        shape = get_lattice().shape[1:]
        s = input_to_index(s)
        embedding = self.Et[:, s].reshape(-1, *shape)
        if self.Ep is not None:
            reps = tuple(l // subl for (l, subl) in zip(shape, self.Ep.shape[1:]))
            embedding += jnp.tile(self.Ep, reps=(1,) + reps)
        return embedding
