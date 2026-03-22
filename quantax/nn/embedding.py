from typing import Optional
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from ..global_defs import get_subkeys, get_lattice, PARTICLE_TYPE


def input_to_index(s):
    lattice = get_lattice()
    s = s.reshape(-1, lattice.Nsites)
    index = jnp.where(s > 0, 1, 0).astype(jnp.int32)

    if lattice.is_fermion:
        if index.shape[0] != 2:
            raise NotImplementedError
        index = 2 * (index[0] != index[1]) + index[1]
    else:
        power = 1 << jnp.arange(s.shape[0], dtype=jnp.int32)
        index = jnp.sum(index * power[:, None], axis=0)

    return index


class Embedding(eqx.Module):
    d: int
    Et: jax.Array
    Ep: Optional[jax.Array]

    def __init__(self, d: int, Ep_sublattice=None, dtype=jnp.float32):
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
