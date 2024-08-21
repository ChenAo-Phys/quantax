from typing import Sequence, Union
from numbers import Number
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import with_sharding_constraint
from jax.sharding import SingleDeviceSharding, NamedSharding, Mesh, PartitionSpec
from jax.experimental import multihost_utils
from .sharding import global_sharding, replicate_sharding


_Array = Union[jax.Array, np.ndarray]


def is_sharded_array(array: _Array) -> bool:
    if isinstance(array, jax.Array):
        return not isinstance(array.sharding, SingleDeviceSharding)
    else:
        return False
    

@partial(jax.jit, static_argnums=1)
def to_global_array(array: Sequence, sharded_axis: int = 0) -> jax.Array:
    if sharded_axis == 0:
        sharding = global_sharding
    else:
        mesh = Mesh(jax.devices(), ('x'))
        partitions = (None,) * sharded_axis + ('x',)
        spec = PartitionSpec(*partitions)
        sharding = NamedSharding(mesh, spec)
    
    array = jnp.asarray(array)
    array = with_sharding_constraint(array, sharding)
    return array


def to_replicate_array(array: Sequence) -> jax.Array:
    array = jnp.asarray(array)
    if jax.process_count() > 1:
        array = multihost_utils.process_allgather(array, tiled=True)
    array = with_sharding_constraint(array, replicate_sharding)
    return array


def array_extend(
    array: _Array, multiple_of_num: int, axis: int = 0, padding_values: Number = 0
) -> _Array:
    n_res = array.shape[axis] % multiple_of_num
    if n_res == 0:
        return array # fast return when the extension is not needed
    
    if isinstance(array, jax.Array):
        pad_fn = jnp.pad
    else:
        array = np.asarray(array)
        pad_fn = np.pad
    
    n_extend = multiple_of_num - n_res
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (0, n_extend)
    array = pad_fn(array, pad_width, constant_values=padding_values)
    return array
