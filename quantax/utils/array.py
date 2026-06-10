import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax.sharding import SingleDeviceSharding
from jax.experimental.multihost_utils import (
    global_array_to_host_local_array,
    host_local_array_to_global_array,
)
from .sharding import (
    make_mesh,
    get_distributed_P,
    get_distributed_sharding,
    get_replicated_sharding,
)


def is_sharded_array(array: ArrayLike) -> bool:
    """
    Whether the input array is sharded across more than one device. Anything
    that is not a :class:`jax.Array` (e.g. a numpy array or a Python scalar) is
    always considered not sharded.
    """
    if isinstance(array, jax.Array):
        return not isinstance(array.sharding, SingleDeviceSharding)
    else:
        return False


def to_distributed_array(array: ArrayLike) -> jax.Array:
    """
    Place the array on all devices, sharded along its first dimension.
    See `~quantax.utils.get_distributed_sharding` for the sharding.

    .. note::
        This expects a global array. In multi-host jobs use
        `~quantax.utils.local_to_global` to assemble host-local arrays into a
        global one instead.
    """
    return jax.device_put(jnp.asarray(array), get_distributed_sharding())


def to_replicated_array(array: ArrayLike) -> jax.Array:
    """
    Place a full copy of the array on every device.
    See `~quantax.utils.get_replicated_sharding` for the sharding.
    """
    return jax.device_put(jnp.asarray(array), get_replicated_sharding())


def global_to_local(array: jax.Array) -> jax.Array:
    """
    Convert a distributed global array into the host-local array holding only
    this process's shards, using
    :func:`jax.experimental.multihost_utils.global_array_to_host_local_array`.

    In single-process jobs the array is already local and is returned unchanged.
    """
    if jax.process_count() > 1:
        array = global_array_to_host_local_array(
            array, make_mesh(), get_distributed_P()
        )
    return array


def local_to_global(array: ArrayLike) -> jax.Array:
    """
    Assemble the host-local arrays of all processes into a single global array
    sharded along its first dimension (see
    `~quantax.utils.get_distributed_sharding`).

    In single-process jobs this is equivalent to
    `~quantax.utils.to_distributed_array`; in multi-host jobs it uses
    :func:`jax.experimental.multihost_utils.host_local_array_to_global_array`.
    """
    if jax.process_count() == 1:
        array = to_distributed_array(array)
    else:
        array = host_local_array_to_global_array(
            array, make_mesh(), get_distributed_P()
        )
        array = jnp.asarray(array)
    return array


def local_to_replicated(array: ArrayLike) -> jax.Array:
    """
    Assemble identical host-local arrays into a global array replicated on
    every device (see `~quantax.utils.get_replicated_sharding`).

    In single-process jobs this is equivalent to
    `~quantax.utils.to_replicated_array`; in multi-host jobs it uses
    :func:`jax.experimental.multihost_utils.host_local_array_to_global_array`.
    Every process must supply the same local array.
    """
    if jax.process_count() == 1:
        array = to_replicated_array(array)
    else:
        array = host_local_array_to_global_array(array, make_mesh(), jax.P())
        array = jnp.asarray(array)
    return array


def to_replicated_numpy(array: jax.Array) -> np.ndarray:
    """
    Gather a (possibly distributed) array into a contiguous numpy array holding
    the full data, identical on every process.

    In multi-host jobs the array is first replicated and then brought to the
    host with
    :func:`jax.experimental.multihost_utils.global_array_to_host_local_array`.
    """
    if jax.process_count() > 1:
        array = to_replicated_array(array)
        array = global_array_to_host_local_array(array, make_mesh(), jax.P())
    return np.asarray(array, order="C")


def array_extend(
    array: jax.Array, multiple_of_num: int, axis: int = 0, padding_values: complex = 0
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
