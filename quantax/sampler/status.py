from typing import Union, Optional, Tuple
from numbers import Number
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from ..utils import to_array_shard


@jtu.register_pytree_node_class
class SamplerStatus:
    def __init__(
        self,
        spins: Optional[jax.Array] = None,
        wave_function: Optional[jax.Array] = None,
        prob: Optional[jax.Array] = None,
        propose_prob: Optional[jax.Array] = None,
    ):
        self.spins = spins
        self.wave_function = wave_function
        self.prob = prob
        self.propose_prob = propose_prob

    def tree_flatten(self) -> Tuple:
        children = (self.spins, self.wave_function, self.prob, self.propose_prob)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


# @partial(jax.pmap, static_broadcasted_argnums=1, axis_name="i")
# def _get_reweight_pmap(wf: jax.Array, reweight: float):
#     psi_reweight = jnp.abs(wf) ** (2 - reweight)
#     return psi_reweight / jax.lax.pmean(jnp.mean(psi_reweight), "i")


@jtu.register_pytree_node_class
class Samples:
    def __init__(
        self,
        spins: jax.Array,
        wave_function: jax.Array,
        reweight: Union[float, jax.Array] = 2.0,
    ):
        self.spins = to_array_shard(spins)
        self.wave_function = to_array_shard(wave_function)
        if isinstance(reweight, Number):
            reweight_factor = jnp.abs(self.wave_function) ** (2 - reweight)
            # should be pmean for multiple hosts
            self.reweight_factor = reweight_factor / jnp.mean(reweight_factor)
        else:
            self.reweight_factor = to_array_shard(reweight)

    @property
    def nsamples(self) -> int:
        return self.spins.shape[0] # restricted to one node
    
    def tree_flatten(self) -> Tuple:
        children = (self.spins, self.wave_function, self.reweight_factor)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __getitem__(self, idx):
        return Samples(
            self.spins[idx], self.wave_function[idx], self.reweight_factor[idx]
        )
