from typing import Sequence, Any
from numbers import Number
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import with_sharding_constraint
from jax.sharding import SingleDeviceSharding, PositionalSharding


def is_sharded_array(array: Any) -> bool:
    if isinstance(array, jax.Array):
        return not isinstance(array.sharding, SingleDeviceSharding)
    else:
        return False
    

@partial(jax.jit, static_argnums=1)
def to_array_shard(array: Sequence, sharded_axis: int = 0) -> jax.Array:
    sharding = PositionalSharding(jax.local_devices())
    array = jnp.asarray(array)
    shape = [1] * array.ndim
    shape[sharded_axis] = sharding.shape[0]
    array = with_sharding_constraint(array, sharding.reshape(shape))
    return array


@jax.jit
def to_array_replicate(array: Sequence) -> jax.Array:
    sharding = PositionalSharding(jax.local_devices()).replicate()
    array = jnp.asarray(array)
    array = with_sharding_constraint(array, sharding)
    return array


def array_extend(
    array: Sequence, multiple_of_num: int, axis: int = 0, padding_values: Number = 0
) -> jax.Array:
    if isinstance(array, jax.Array):
        pad_fn = jnp.pad
    else:
        array = np.asarray(array)
        pad_fn = np.pad

    n_res = array.shape[axis] % multiple_of_num
    if n_res == 0:
        return array # fast return when the extension is not needed
    
    n_extend = multiple_of_num - n_res
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (0, n_extend)
    array = pad_fn(array, pad_width, constant_values=padding_values)
    return array
