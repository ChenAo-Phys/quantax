from collections.abc import Callable
from functools import partial
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
import equinox as eqx

import inspect as _inspect

try:  # stable jax.shard_map on recent jax; experimental path on older
    from jax import shard_map as _shard_map
except ImportError:  # pragma: no cover
    from jax.experimental.shard_map import shard_map as _shard_map

# the replication-check kwarg was renamed check_rep -> check_vma across versions
_SHARD_MAP_CHECK_KW = (
    "check_vma"
    if "check_vma" in _inspect.signature(_shard_map).parameters
    else "check_rep"
)

from .array import array_extend
from .tree import filter_tree_map
from .sharding import make_mesh, get_distributed_P


@eqx.filter_jit
def _get_device_batch(args: tuple, in_axes: int | tuple) -> int:
    if isinstance(in_axes, int):
        in_axes = (in_axes,) * len(args)

    ndevices = jax.device_count()
    device_batch = 0
    for axis, arg in zip(in_axes, args):
        if axis is not None:
            leaves = jax.tree.leaves(eqx.filter(arg, eqx.is_array))
            device_batch = leaves[0].shape[axis] // ndevices
            break

    return device_batch


@eqx.filter_jit
def _chunk_args(
    args: tuple, in_axes: int | tuple, device_batch: int, chunk_size: int
) -> tuple[list, list, int]:
    if isinstance(in_axes, int):
        in_axes = (in_axes,) * len(args)

    ndevices = jax.device_count()

    def fn_split(x: jax.Array, axis: int) -> jax.Array:
        # Chunks are contiguous on each device, so the inverse `_combine_outputs`
        # is a device-local reshape that doesn't copy the (possibly huge) outputs.
        before = x.shape[:axis]
        after = x.shape[axis + 1 :]
        x = x.reshape(*before, ndevices, -1, *after)
        x = array_extend(x, chunk_size, axis=axis + 1)
        x = x.reshape(*before, ndevices, -1, chunk_size, *after)
        x = jnp.moveaxis(x, axis + 1, 0)
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


@jax.jit
def _unstack_args(args: PyTree) -> PyTree:
    args, treedef = jax.tree.flatten(args)
    args = [jnp.unstack(arg) for arg in args]
    args = list(zip(*args))
    return [jax.tree.unflatten(treedef, arg) for arg in args]


@partial(eqx.filter_jit, donate="all")
def _combine_outputs(
    outputs: PyTree, out_axes: int | tuple, device_batch: int
) -> PyTree:
    is_tuple = type(outputs) is tuple
    if not is_tuple:
        outputs = (outputs,)

    if isinstance(out_axes, int):
        out_axes = (out_axes,) * len(outputs)

    ndevices = jax.device_count()

    def fn_combine(x: jax.Array, axis: int) -> jax.Array:
        before = x.shape[1 : axis + 1]
        after = x.shape[axis + 2 :]
        x = x.reshape(x.shape[0], *before, ndevices, -1, *after)
        x = jnp.moveaxis(x, 0, axis + 1)
        x = x.reshape(*before, ndevices, -1, *after)
        if x.shape[axis + 1] != device_batch:
            x = jax.lax.slice_in_dim(x, 0, device_batch, axis=axis + 1)
        x = x.reshape(*before, -1, *after)
        return x

    fn = lambda axis, out: filter_tree_map(lambda x: fn_combine(x, axis), out)
    outputs = tuple(
        out if axis is None else fn(axis, out) for axis, out in zip(out_axes, outputs)
    )

    if not is_tuple:
        outputs = outputs[0]
    return outputs


@partial(eqx.filter_jit, donate="all")
def _stack_outputs(outputs: PyTree, out_axes: int | tuple, device_batch: int) -> PyTree:
    fn_concat = lambda *out: jnp.stack(out, axis=0)
    outputs = filter_tree_map(fn_concat, *outputs)
    return _combine_outputs(outputs, out_axes, device_batch)


def chunk_map(
    f: Callable,
    in_axes: int | tuple | None = 0,
    out_axes: int | tuple | None = 0,
    chunk_size: int | None = None,
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
        accelerated if `scan` is used, but the function must be jittable.
    """
    all_none = isinstance(in_axes, tuple) and all(axis is None for axis in in_axes)
    if in_axes is None or all_none or chunk_size is None:
        return f  # fast return if chunk is not necessary

    any_none = isinstance(out_axes, tuple) and any(axis is None for axis in out_axes)
    if out_axes is None or any_none:
        raise NotImplementedError("`chunk_map` with `out_axes=None` not implemented")

    def chunked_f(*args):
        device_batch = _get_device_batch(args, in_axes)
        if device_batch <= chunk_size:
            return f(*args)

        dynamic_args, static_args, device_batch = _chunk_args(
            args, in_axes, device_batch, chunk_size
        )

        if use_scan:
            fn_scan = lambda _, dynamic: (_, f(*eqx.combine(dynamic, static_args)))
            _, outputs = jax.lax.scan(fn_scan, None, dynamic_args)
            return _combine_outputs(outputs, out_axes, device_batch)
        else:
            dynamic_args = _unstack_args(dynamic_args)
            outputs = [f(*eqx.combine(args, static_args)) for args in dynamic_args]
            return _stack_outputs(outputs, out_axes, device_batch)

    return chunked_f


def jit_chunk_vmap(
    f: Callable,
    in_axes: int | tuple | None = 0,
    out_axes: int | tuple | None = 0,
    chunk_size: int | None = None,
) -> Callable:
    """
    f -> jit(chunk_map(vmap(f), use_scan=True))

    :param f:
        The function to be converted. The arguments of f will be sharded.

    :param in_axes:
        The mapped axes of f input arguments.

    :param out_axes:
        The mapped axes of f outputs.

    :param chunk_size:
        The chunk size on each machine. If None, no chunking will be applied.
    """
    f = eqx.filter_vmap(f, in_axes=in_axes, out_axes=out_axes)
    f = chunk_map(f, in_axes, out_axes, chunk_size, use_scan=True)
    f = eqx.filter_jit(f)
    return f


def shard_chunk_vmap(f: Callable, chunk_size: int | None = None) -> Callable:
    """
    Like :func:`jit_chunk_vmap` for the ``in_axes=(None, 0), out_axes=0`` case
    (one replicated module argument, one batched array argument, one batched array
    output), but each device processes only its local batch shard inside
    ``jax.shard_map`` with a device-local ``lax.scan`` over ``chunk_size`` chunks.

    ``chunk_map`` reshapes the batch into a single globally-sharded axis of size
    ``ndevices * chunk_size`` per scan step. For per-sample reverse-mode ops that
    XLA lowers to a grouped convolution -- the convolution weight gradient -- the
    SPMD partitioner then all-gathers the activations across devices on every scan
    step, so the Jacobian cost grows with the device/node count (worst with a small
    ``chunk_size``). Running the scan inside ``shard_map`` keeps each device's work
    purely local, eliminating those all-gathers.

    :param f:
        The function ``f(module, x)`` to map; ``module`` is replicated, ``x`` is
        batched on axis 0.

    :param chunk_size:
        The per-device chunk size. If ``None`` no chunking is applied.
    """
    vmapped = eqx.filter_vmap(f, in_axes=(None, 0), out_axes=0)
    if chunk_size is None:
        return eqx.filter_jit(vmapped)

    @eqx.filter_jit
    def wrapped(module, x):
        device_batch = x.shape[0] // jax.device_count()
        if device_batch <= chunk_size:
            return vmapped(module, x)

        # Pass the module arrays THROUGH shard_map (replicated) rather than closing
        # over them, so inside the manual mesh they are manual-replicated (closed-over
        # auto-sharded params clash with shard_map's manual mesh).
        arrays, static = eqx.partition(module, eqx.is_array)

        def local(arrays, x_local):  # one device: replicated arrays + local x shard
            module = eqx.combine(arrays, static)
            db = x_local.shape[0]
            x_pad = array_extend(x_local, chunk_size, axis=0)
            n_chunks = x_pad.shape[0] // chunk_size
            x_chunks = x_pad.reshape(n_chunks, chunk_size, *x_local.shape[1:])
            scan_fn = lambda carry, xc: (carry, vmapped(module, xc))
            _, out = jax.lax.scan(scan_fn, None, x_chunks)
            out = out.reshape(-1, *out.shape[2:])
            return out[:db]

        spec = get_distributed_P()
        repl = jax.tree.map(lambda _: jax.P(), arrays)  # replicate every module array
        kw = {"mesh": make_mesh(), "in_specs": (repl, spec), "out_specs": spec}
        kw[_SHARD_MAP_CHECK_KW] = False
        return _shard_map(local, **kw)(arrays, x)

    return wrapped
