from typing import Sequence, Union
from numbers import Number
from jaxtyping import ArrayLike
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.lax import with_sharding_constraint
from jax.sharding import SingleDeviceSharding, Mesh, PartitionSpec
from jax.experimental.multihost_utils import (
    global_array_to_host_local_array,
    host_local_array_to_global_array,
)
from .sharding import get_global_sharding, get_replicate_sharding


def is_sharded_array(array: Union[jax.Array, np.ndarray]) -> bool:
    """
    Whether the input array is sharded. The array is always considered not sharded 
    if it's not a jax array.
    """
    if isinstance(array, jax.Array):
        return not isinstance(array.sharding, SingleDeviceSharding)
    else:
        return False


@jax.jit
def to_global_array(array: Sequence) -> jax.Array:
    """
    Transform the array to be sharded across all devices in the first dimension.
    See `~quantax.utils.get_global_sharding` for the sharding.
    """
    array = jnp.asarray(array)
    array = with_sharding_constraint(array, get_global_sharding())
    return array


@jax.jit
def to_replicate_array(array: Sequence) -> jax.Array:
    """
    Transform the array to be replicated across all devices.
    See `~quantax.utils.get_replicate_sharding` for the sharding.
    """
    array = jnp.asarray(array)
    array = with_sharding_constraint(array, get_replicate_sharding())
    return array


def global_to_local(array: jax.Array) -> jax.Array:
    """
    In multi-host jobs, use `jax.experimental.multihost_utils.global_array_to_host_local_array`
    to transform a sharded array to be local on each device.
    """
    if jax.process_count() > 1:
        global_mesh = Mesh(jax.devices(), "x")
        global_pspecs = PartitionSpec("x")
        array = global_array_to_host_local_array(array, global_mesh, global_pspecs)
    return array


def local_to_global(array: Sequence) -> jax.Array:
    """
    In multi-host jobs, use `jax.experimental.multihost_utils.host_local_array_to_global_array`
    to transform local arrays to be sharded.
    """
    if jax.process_count() == 1:
        array = to_global_array(array)
    else:
        global_mesh = Mesh(jax.devices(), "x")
        global_pspecs = PartitionSpec("x")
        array = host_local_array_to_global_array(array, global_mesh, global_pspecs)
        array = jnp.asarray(array)
    return array


def local_to_replicate(array: Sequence) -> jax.Array:
    """
    In multi-host jobs, use `jax.experimental.multihost_utils.host_local_array_to_global_array`
    to transform local arrays to be replicated on each device.
    """
    if jax.process_count() == 1:
        array = to_replicate_array(array)
    else:
        global_mesh = Mesh(jax.devices(), "x")
        replicate_pspecs = PartitionSpec()
        array = host_local_array_to_global_array(array, global_mesh, replicate_pspecs)
        array = jnp.asarray(array)
    return array


def to_replicate_numpy(array: jax.Array) -> np.ndarray:
    """
    In multi-host jobs, use `jax.experimental.multihost_utils.global_array_to_host_local_array`
    to transform a sharded array to be replicated numpy arrays on each device.
    """
    if jax.process_count() > 1:
        array = to_replicate_array(array)
        global_mesh = Mesh(jax.devices(), "x")
        replicate_pspecs = PartitionSpec()
        array = global_array_to_host_local_array(array, global_mesh, replicate_pspecs)
    return np.asarray(array, order="C")


def array_extend(
    array: jax.Array, multiple_of_num: int, axis: int = 0, padding_values: Number = 0
) -> jax.Array:
    """
    Extend the array.

    :param array:
        The array to be extended.

    :param multiple_of_num:
        Specify the size of the extended axis to be a multiple of this number.

    :param axis:
        The axis to be extended, default to 0 (the first dimension).

    :param padding_values:
        The padding values, default to 0.
    """
    n_res = array.shape[axis] % multiple_of_num
    if n_res == 0:
        return array  # fast return when the extension is not needed

    n_extend = multiple_of_num - n_res
    pad_width = [(0, 0)] * array.ndim
    pad_width[axis] = (0, n_extend)
    array = jnp.pad(array, pad_width, constant_values=padding_values)
    return array


def array_set(array: jax.Array, array_set: jax.Array, inds: ArrayLike) -> jax.Array:
    """
    Equivalent to `array.at[inds].set(array_set)`, but significantly faster
    for complex-valued inputs.
    """
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        real = array.real.at[inds].set(array_set.real)
        imag = array.imag.at[inds].set(array_set.imag)
        return jax.lax.complex(real, imag)
    else:
        return array.at[inds].set(array_set)


@partial(jax.jit, static_argnums=2)
def sharded_segment_sum(
    data: jax.Array, segment_ids: jax.Array, num_segments: int
) -> jax.Array:
    """
    Equivalent to `jax.ops.segment_sum`, but avoid data transfer among devices when
    `data` and `segment_ids` are both properly sharded.
    """
    ndevices = jax.device_count()
    num_segments = num_segments // ndevices
    data = data.reshape(ndevices, -1)
    segment_ids = segment_ids.reshape(ndevices, -1)
    segment_sum = lambda data, segment: jax.ops.segment_sum(data, segment, num_segments)
    output = jax.vmap(segment_sum)(data, segment_ids)
    return output.flatten()
