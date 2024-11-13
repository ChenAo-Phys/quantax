from typing import Callable, Tuple, Union, Optional
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec
from jax.experimental.shard_map import shard_map
import equinox as eqx
from .array import array_extend


def _axes_to_specs(axes: Union[tuple, int]) -> PartitionSpec:
    if isinstance(axes, int):
        return PartitionSpec(*((None,) * axes), "x")

    specs = []
    for axis in axes:
        if axis is None:
            specs.append(None)
        else:
            new_specs = PartitionSpec(*((None,) * axis), "x")
            specs.append(new_specs)
    return tuple(specs)


def shard_vmap(
    f: Callable, in_axes: Union[tuple, int], out_axes: Union[tuple, int]
) -> Callable:
    """
    f -> jit(shard_map(vmap(f)))
    """
    f = jax.vmap(f, in_axes, out_axes)
    mesh = Mesh(jax.devices(), "x")
    in_specs = _axes_to_specs(in_axes)
    out_specs = _axes_to_specs(out_axes)
    f = shard_map(f, mesh, in_specs, out_specs)
    f = eqx.filter_jit(f)
    return f


def chunk_map(
    f: Callable, in_axes: Union[Tuple, int] = 0, chunk_size: Optional[int] = None
) -> Callable:
    """
    Convert a vmapped function to a function with chunked batches and parallel
    computation on all available machines. The arguments will be unchanged if the batch
    size on each machine is smaller than the chunk size, but it will be padded with 0
    if the batch size is larger than the chunk size and not a multiple of chunk size.

    :param f:
        The function to be converted. The arguments of f are assumed to be sharded.

    :param in_axes:
        The vmapped axes of f which are to be chunked. Currently only None and 0 are
        supported.

    :param chunk_size:
        The chunk size on each machine.
    """
    if any(axis not in (0, None) for axis in in_axes):
        raise NotImplementedError(
            "`chunk_parallel_map` with non-zero in_axes not implemented"
        )

    vmapped_idx = tuple(i for i, axis in enumerate(in_axes) if axis is not None)
    if len(vmapped_idx) == 0:
        return f  # fast return if there is no vmapped axis

    def chunked_f(*args):
        nbatch = args[vmapped_idx[0]].shape[0]
        ndevices = jax.device_count()

        if chunk_size is None or nbatch <= ndevices * chunk_size:
            return f(*args)  # fast return if chunk is not needed

        nbatch_per_device = (nbatch - 1) // ndevices + 1
        nchunks = (nbatch_per_device - 1) // chunk_size + 1
        args = list(args)
        for i in vmapped_idx:
            args[i] = args[i].reshape(ndevices, -1, *args[i].shape[1:])
            args[i] = array_extend(args[i], chunk_size, axis=1)

        outputs = None
        for n in range(nchunks):
            new_args = []
            for axis, arg in zip(in_axes, args):
                if axis is not None:
                    arg = arg[:, n * chunk_size : (n + 1) * chunk_size]
                    arg = arg.reshape(-1, *arg.shape[2:])
                new_args.append(arg)
            new_outputs = f(*new_args)

            is_output_tuple = isinstance(new_outputs, tuple)
            if not is_output_tuple:
                new_outputs = (new_outputs,)
            if outputs is None:
                outputs = [[new_out] for new_out in new_outputs]
            else:
                for out, new_out in zip(outputs, new_outputs):
                    out.append(new_out)

        for i, out in enumerate(outputs):
            out = [x.reshape(ndevices, -1, *x.shape[1:]) for x in out]
            out = jnp.concatenate(out, axis=1)[:, :nbatch_per_device]
            outputs[i] = out.reshape(-1, *out.shape[2:])
        if not is_output_tuple:
            outputs = outputs[0]

        return outputs

    return chunked_f

#temporary due to bug in jax
def complex_set(array, array_set, inds):
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        return jax.lax.complex(array.real.at[inds].set(array_set.real),array.imag.at[inds].set(array_set.imag))
    else:
        return array.at[inds].set(array_set)
  




