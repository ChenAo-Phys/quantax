from typing import Callable, Tuple, Union, Optional, Sequence
from functools import partial
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
from jax.sharding import Mesh, PartitionSpec
from jax.experimental.shard_map import shard_map
import equinox as eqx
from .array import array_extend
from .tree import filter_tree_map


@eqx.filter_jit
def _chunk_args(
    args: tuple, in_axes: Union[Tuple, int], chunk_size: int
) -> Tuple[list, list, int]:
    if not isinstance(in_axes, Sequence):
        in_axes = (in_axes,) * len(args)

    ndevices = jax.device_count()

    for axis, arg in zip(in_axes, args):
        if axis is not None:
            dynamic = eqx.filter(arg, eqx.is_array)
            dynamic, treedef = jax.tree.flatten(dynamic)
            device_batch = dynamic[0].shape[axis] // ndevices
            break

    def fn_split(x: jax.Array, axis: int) -> jax.Array:
        before = x.shape[:axis]
        after = x.shape[axis + 1 :]
        x = x.reshape(*before, ndevices, -1, *after)
        if device_batch > chunk_size:
            x = array_extend(x, chunk_size, axis=axis + 1)
            x = x.reshape(*before, ndevices, chunk_size, -1, *after)
        else:
            x = x.reshape(*before, ndevices, device_batch, 1, *after)
        x = jnp.moveaxis(x, axis + 2, 0)
        x = x.reshape(x.shape[0], *before, -1, *after)
        return x

    dynamic_args = []
    static_args = []
    for axis, arg in zip(in_axes, args):
        if axis is None:
            dynamic_args.append(None)
            static_args.append(arg)
        else:
            dynamic, static = eqx.partition(arg, eqx.is_array)
            dynamic = jax.tree.map(lambda x: fn_split(x, axis), dynamic)
            dynamic_args.append(dynamic)
            static_args.append(static)

    return dynamic_args, static_args, device_batch


@partial(eqx.filter_jit, donate="all")
def _combine_outputs(
    outputs: PyTree, out_axes: Union[Tuple, int], device_batch: int
) -> PyTree:
    is_tuple = type(outputs) is tuple
    if not is_tuple:
        outputs = (outputs,)

    if not isinstance(out_axes, Sequence):
        out_axes = (out_axes,) * len(outputs)

    ndevices = jax.device_count()

    def fn_combine(x: jax.Array, axis: int) -> jax.Array:
        x = jnp.moveaxis(x, axis + 1, 0)  # an additional axis due to chunks
        non_batch_shape = x.shape[2:]
        x = x.reshape(ndevices, -1, *non_batch_shape)
        x = x[:, :device_batch]
        x = x.reshape(-1, *non_batch_shape)
        x = jnp.moveaxis(x, 0, axis)
        return x

    fn = lambda axis, out: filter_tree_map(lambda x: fn_combine(x, axis), out)
    outputs = tuple(
        out if axis is None else fn(axis, out) for axis, out in zip(out_axes, outputs)
    )

    if not is_tuple:
        outputs = outputs[0]
    return outputs


def chunk_map(
    f: Callable,
    in_axes: Union[Tuple, int, None] = 0,
    out_axes: Union[Tuple, int, None] = 0,
    chunk_size: Optional[int] = None,
    use_scan: bool = False,
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

    :param out_axes:
        The vmapped axes of outputs.

    :param chunk_size:
        The chunk size on each machine.

    :param use_scan:
        Whether to use `jax.lax.scan` in chunked function apply. The compilation will be
        accerlerated if `scan` is used, but the function must be jittable.
    """
    all_none = isinstance(in_axes, Sequence) and all(axis is None for axis in in_axes)
    if in_axes is None or all_none or chunk_size is None:
        return f  # fast return if chunk is not necessary

    any_none = isinstance(out_axes, Sequence) and any(axis is None for axis in out_axes)
    if out_axes is None or any_none:
        raise NotImplementedError("`chunk_map` with `out_axes=None` not implemented")

    def chunked_f(*args):
        dynamic_args, static_args, device_batch = _chunk_args(args, in_axes, chunk_size)

        if use_scan:
            fn_scan = lambda _, dynamic: (_, f(*eqx.combine(dynamic, static_args)))
            _, outputs = jax.lax.scan(fn_scan, None, dynamic_args)
        else:
            dynamic_args, treedef = jax.tree.flatten(dynamic_args)
            nchunks = dynamic_args[0].shape[0]
            outputs = []
            for i in range(nchunks):
                args = [arg[i] for arg in dynamic_args]
                args = jax.tree.unflatten(treedef, args)
                args = eqx.combine(args, static_args)
                outputs.append(f(*args))

            fn_concat = lambda *out: jnp.stack(out, axis=0)
            outputs = filter_tree_map(fn_concat, *outputs)

        return _combine_outputs(outputs, out_axes, device_batch)

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


def shmap(
    f: Callable, in_axes: Union[tuple, int, None], out_axes: Union[tuple, int, None]
) -> Callable:
    """
    f -> shard_map(f), sharded along the first dimension

    :param f:
        The function to be converted. The arguments of f will be sharded.

    :param in_axes:
        The sharded axes of f input arguments.

    :param out_axes:
        The sharded axes of f outputs.
    """

    mesh = Mesh(jax.devices(), "x")
    in_specs = _axes_to_specs(in_axes)
    out_specs = _axes_to_specs(out_axes)
    f = shard_map(f, mesh, in_specs, out_specs, check_rep=False)
    return f


def chunk_shard_vmap(
    f: Callable,
    in_axes: Union[tuple, int, None],
    out_axes: Union[tuple, int, None],
    chunk_size: Optional[int] = None,
    shard_axes: Union[tuple, int, None] = None,
) -> Callable:
    """
    f -> jit(chunk_map(shard_map(vmap(f))))

    :param f:
        The function to be converted. The arguments of f will be sharded.

    :param in_axes:
        The mapped axes of f input arguments.

    :param out_axes:
        The mapped axes of f outputs.
    """

    shard_axes = in_axes if shard_axes is None else shard_axes

    f = jax.vmap(f, in_axes, out_axes)
    f = shmap(f, shard_axes, out_axes)
    f = chunk_map(f, in_axes, out_axes, chunk_size, use_scan=True)
    f = eqx.filter_jit(f)
    return f
