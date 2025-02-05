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
from ..sites import TriangularB
from ..global_defs import get_lattice
import equinox as eqx
from typing import Optional
from jaxtyping import Key

def is_sharded_array(array: Union[jax.Array, np.ndarray]) -> bool:
    if isinstance(array, jax.Array):
        return not isinstance(array.sharding, SingleDeviceSharding)
    else:
        return False


@jax.jit
def to_global_array(array: Sequence) -> jax.Array:
    array = jnp.asarray(array)
    array = with_sharding_constraint(array, get_global_sharding())
    return array


@jax.jit
def to_replicate_array(array: Sequence) -> jax.Array:
    array = jnp.asarray(array)
    array = with_sharding_constraint(array, get_replicate_sharding())
    return array


def global_to_local(array: jax.Array) -> jax.Array:
    if jax.process_count() > 1:
        global_mesh = Mesh(jax.devices(), "x")
        global_pspecs = PartitionSpec("x")
        array = global_array_to_host_local_array(array, global_mesh, global_pspecs)
    return array


def local_to_global(array: Sequence) -> jax.Array:
    if jax.process_count() == 1:
        array = to_global_array(array)
    else:
        global_mesh = Mesh(jax.devices(), "x")
        global_pspecs = PartitionSpec("x")
        array = host_local_array_to_global_array(array, global_mesh, global_pspecs)
        array = jnp.asarray(array)
    return array


def local_to_replicate(array: Sequence) -> jax.Array:
    if jax.process_count() == 1:
        array = to_replicate_array(array)
    else:
        global_mesh = Mesh(jax.devices(), "x")
        replicate_pspecs = PartitionSpec()
        array = host_local_array_to_global_array(array, global_mesh, replicate_pspecs)
        array = jnp.asarray(array)
    return array


def to_replicate_numpy(array: jax.Array) -> np.ndarray:
    if jax.process_count() > 1:
        array = to_replicate_array(array)
        global_mesh = Mesh(jax.devices(), "x")
        replicate_pspecs = PartitionSpec()
        array = global_array_to_host_local_array(array, global_mesh, replicate_pspecs)
    return np.asarray(array, order="C")


def array_extend(
    array: jax.Array, multiple_of_num: int, axis: int = 0, padding_values: Number = 0
) -> jax.Array:
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
    An alternative for slow set of complex values in jax
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
    ndevices = jax.device_count()
    num_segments = num_segments // ndevices
    data = data.reshape(ndevices, -1)
    segment_ids = segment_ids.reshape(ndevices, -1)
    segment_sum = lambda data, segment: jax.ops.segment_sum(data, segment, num_segments)
    output = jax.vmap(segment_sum)(data, segment_ids)
    return output.flatten()

class Reshape_TriangularB(eqx.Module):
    """
    Reshape the TriangularB spins into the arrangement of Triangular for more efficient
    convolutions.
    """

    dtype: jnp.dtype = eqx.field(static=True)
    permutation: np.ndarray

    def __init__(self, dtype: jnp.dtype = jnp.float32):
        self.dtype = dtype
        lattice = get_lattice()
        if not isinstance(lattice, TriangularB):
            raise ValueError("The current lattice is not `TriangularB`.")

        permutation = np.arange(lattice.N, dtype=np.uint16)
        permutation = permutation.reshape(lattice.shape[1:])
        for i in range(permutation.shape[1]):
            permutation[:, i] = np.roll(permutation[:, i], shift=i)

        self.permutation = permutation

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        lattice = get_lattice()
        shape = lattice.shape
        if lattice.is_fermion:
            shape = (shape[0] * 2,) + shape[1:]
            x = x.reshape(2,-1)

        x = x[...,self.permutation]
        x = x.reshape(shape).astype(self.dtype)
        return x

class ReshapeTo_TriangularB(eqx.Module):
    """
    Reshape the Triangular spins back into the arrangement of TriangularB.
    """

    dtype: jnp.dtype = eqx.field(static=True)
    permutation: np.ndarray

    def __init__(self, dtype: jnp.dtype = jnp.float32):
        self.dtype = dtype
        lattice = get_lattice()
        if not isinstance(lattice, TriangularB):
            raise ValueError("The current lattice is not `TriangularB`.")

        permutation = np.arange(lattice.N, dtype=np.uint16)
        permutation = permutation.reshape(lattice.shape[1:])
        for i in range(permutation.shape[1]):
            permutation[:, i] = np.roll(permutation[:, i], shift=-i)

        self.permutation = permutation

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        x = x.reshape(x.shape[0], -1)
        x = x[:, self.permutation]
        x = x.reshape(x.shape[0], *get_lattice().shape)
        return x

def _triangularb_circularpad(x: jax.Array) -> jax.Array:
    pad_lower = jnp.roll(x[:, :, -1:], shift=-x.shape[2], axis=1)
    pad_upper = jnp.roll(x[:, :, :1], shift=x.shape[2], axis=1)
    x = jnp.concatenate([pad_lower, x, pad_upper], axis=2)
    x = jnp.pad(x, [(0, 0), (1, 1), (0, 0)], mode="wrap")
    return x
