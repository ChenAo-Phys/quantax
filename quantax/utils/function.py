from typing import Callable, Tuple, Union, Optional
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from jax.sharding import Mesh, PartitionSpec
from jax.experimental.shard_map import shard_map
import equinox as eqx
from .array import array_extend
from .tree import filter_tree_map


def _moveaxis(
    tree: PyTree, source: Optional[int], destination: Optional[int]
) -> PyTree:
    if source is None or destination is None:
        return tree
    else:
        fn = lambda x: jnp.moveaxis(x, source, destination)
        return filter_tree_map(fn, tree)


def chunk_map(
    f: Callable,
    in_axes: Union[Tuple, int] = 0,
    out_axes: Union[Tuple, int] = 0,
    chunk_size: Optional[int] = None,
) -> Callable:
    """
    Convert a vmapped function to a function with chunked batches and parallel
    computation on all available machines. The arguments will be unchanged if the batch
    size on each machine is smaller than the chunk size, but it will be padded with 0
    if the batch size is larger than the chunk size and not a multiple of chunk size.

    :param f:
        The function to be converted. The arguments of f are assumed to be sharded.

    :param in_axes:
        The vmapped axes of f which are to be chunked.
        supported.

    :param out_axes:
        The vmapped axes of outputs.

    :param chunk_size:
        The chunk size on each machine.
    """

    def chunked_f(*args):
        _in_axes = (in_axes,) * len(args) if isinstance(in_axes, int) else in_axes

        vmapped_idx = tuple(i for i, axis in enumerate(in_axes) if axis is not None)
        if len(vmapped_idx) == 0:
            return f  # fast return if there is no vmapped axis

        args = [_moveaxis(arg, axis, 0) for axis, arg in zip(_in_axes, args)]

        nbatch = args[vmapped_idx[0]].shape[0]
        ndevices = jax.device_count()

        if chunk_size is None or nbatch <= ndevices * chunk_size:
            return f(*args)  # fast return if chunk is not needed

        nbatch_per_device = (nbatch - 1) // ndevices + 1
        nchunks = (nbatch_per_device - 1) // chunk_size + 1

        def fn_split(x):
            non_batch_shape = x.shape[1:]
            x = x.reshape(ndevices, -1, *non_batch_shape)
            x = array_extend(x, chunk_size, axis=1)
            x = x.reshape(ndevices, chunk_size, nchunks, *non_batch_shape)
            x = jnp.moveaxis(x, 2, 0)
            return x.reshape(nchunks, -1, *non_batch_shape)

        static_args = []
        dynamic_args = []
        for axis, arg in zip(_in_axes, args):
            if axis is None:
                static_args.append(arg)
                dynamic_args.append(None)
            else:
                dynamic, static = eqx.partition(arg, eqx.is_array)
                static_args.append(static)
                dynamic = jax.tree.map(fn_split, dynamic)
                dynamic_args.append(dynamic)

        def fn_scan(carry, dynamic_args):
            args = eqx.combine(dynamic_args, static_args)
            outputs = f(*args)
            return carry, outputs

        _, outputs = jax.lax.scan(fn_scan, None, dynamic_args, nchunks)

        is_tuple = isinstance(outputs, tuple)
        if not is_tuple:
            outputs = (outputs,)
        n_outputs = len(outputs)
        _out_axes = (out_axes,) * n_outputs if isinstance(out_axes, int) else out_axes

        outputs = [_moveaxis(out, axis, 0) for axis, out in zip(_out_axes, outputs)]

        def fn_combine(x):
            non_batch_shape = x.shape[2:]
            x = x.reshape(ndevices, -1, *non_batch_shape)
            x = x[:, :nbatch_per_device]
            x = x.reshape(-1, *non_batch_shape)
            return x

        new_outputs = []
        for axis, out in zip(_out_axes, outputs):
            if axis is None:
                new_outputs.append(out)
            else:
                out = _moveaxis(out, axis + 1, 0)
                out = filter_tree_map(fn_combine, out)
                out = _moveaxis(out, 0, axis)
                new_outputs.append(out)

        outputs = tuple(new_outputs)
        if not is_tuple:
            outputs = outputs[0]
        return outputs

    return chunked_f


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


def shard_chunk_vmap(
    f: Callable,
    in_axes: Union[tuple, int],
    out_axes: Union[tuple, int],
    chunk_size: Optional[int] = None,
) -> Callable:
    """
    f -> jit(chunk_map(shard_map(vmap(f))))
    """
    f = jax.vmap(f, in_axes, out_axes)
    mesh = Mesh(jax.devices(), "x")
    in_specs = _axes_to_specs(in_axes)
    out_specs = _axes_to_specs(out_axes)
    f = shard_map(f, mesh, in_specs, out_specs)
    f = chunk_map(f, in_axes, out_axes, chunk_size)
    f = eqx.filter_jit(f)
    return f
