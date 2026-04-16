from typing import Sequence, Union
from numbers import Number
import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax.sharding import SingleDeviceSharding, PartitionSpec
from jax.experimental.multihost_utils import (
    global_array_to_host_local_array,
    host_local_array_to_global_array,
)
from .sharding import make_mesh, get_distributed_sharding, get_replicated_sharding


def is_sharded_array(array: Union[jax.Array, np.ndarray]) -> bool:
    """
    Whether the input array is sharded. The array is always considered not sharded
    if it's not a jax array.
    """
    if isinstance(array, jax.Array):
        return not isinstance(array.sharding, SingleDeviceSharding)
    else:
        return False


def to_distributed_array(array: Sequence) -> jax.Array:
    """
    Transform the array to be sharded across all devices in the first dimension.
    See `~quantax.utils.get_distributed_sharding` for the sharding.
    """
    array = jnp.asarray(array)
    array = jax.device_put(array, get_distributed_sharding())
    return array


def to_replicated_array(array: Sequence) -> jax.Array:
    """
    Transform the array to be replicated across all devices.
    See `~quantax.utils.get_replicated_sharding` for the sharding.
    """
    array = jnp.asarray(array)
    array = jax.device_put(array, get_replicated_sharding())
    return array


def global_to_local(array: jax.Array) -> jax.Array:
    """
    In multi-host jobs, use `jax.experimental.multihost_utils.global_array_to_host_local_array`
    to transform a sharded array to be local on each device.
    """
    if jax.process_count() > 1:
        global_mesh = make_mesh()
        distributed_pspecs = PartitionSpec("x")
        array = global_array_to_host_local_array(array, global_mesh, distributed_pspecs)
    return array


def local_to_global(array: Sequence) -> jax.Array:
    """
    In multi-host jobs, use `jax.experimental.multihost_utils.host_local_array_to_global_array`
    to transform local arrays to be sharded.
    """
    if jax.process_count() == 1:
        array = to_distributed_array(array)
    else:
        global_mesh = make_mesh()
        distributed_pspecs = PartitionSpec("x")
        array = host_local_array_to_global_array(array, global_mesh, distributed_pspecs)
        array = jnp.asarray(array)
    return array


def local_to_replicated(array: Sequence) -> jax.Array:
    """
    In multi-host jobs, use `jax.experimental.multihost_utils.host_local_array_to_global_array`
    to transform local arrays to be replicated on each device.
    """
    if jax.process_count() == 1:
        array = to_replicated_array(array)
    else:
        global_mesh = make_mesh()
        replicated_pspecs = PartitionSpec()
        array = host_local_array_to_global_array(array, global_mesh, replicated_pspecs)
        array = jnp.asarray(array)
    return array


def to_replicated_numpy(array: jax.Array) -> np.ndarray:
    """
    In multi-host jobs, use `jax.experimental.multihost_utils.global_array_to_host_local_array`
    to transform a sharded array to be replicated numpy arrays on each device.
    """
    if jax.process_count() > 1:
        array = to_replicated_array(array)
        global_mesh = make_mesh()
        replicated_pspecs = PartitionSpec()
        array = global_array_to_host_local_array(array, global_mesh, replicated_pspecs)
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


def array_set(array: jax.Array, inds: ArrayLike, array_set: jax.Array) -> jax.Array:
    """
    Equivalent to `array.at[inds].set(array_set)`, but significantly faster
    for complex-valued inputs.

    :param array:
        The original array.

    :param inds:
        The indices to be set.

    :param array_set:
        The values to be set.
    """
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        real = array.real.at[inds].set(array_set.real)
        imag = array.imag.at[inds].set(array_set.imag)
        return jax.lax.complex(real, imag)
    else:
        return array.at[inds].set(array_set)
