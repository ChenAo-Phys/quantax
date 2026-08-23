from collections.abc import Callable
from functools import partial, wraps
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
import equinox as eqx
from .array import array_extend
from .tree import filter_tree_map
from .sharding import make_mesh, get_distributed_P


@eqx.filter_jit
def _get_device_batch(args: tuple, in_axes: tuple, ndevices: int) -> int:
    device_batch = 0
    for axis, arg in zip(in_axes, args):
        if axis is not None:
            leaves = jax.tree.leaves(eqx.filter(arg, eqx.is_array))
            device_batch = leaves[0].shape[axis] // ndevices
            break

    return device_batch


@eqx.filter_jit
def _chunk_args(
    args: tuple, in_axes: tuple, chunk_size: int, ndevices: int
) -> tuple[list, list]:

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

    return dynamic_args, static_args


@jax.jit
def _unstack_args(args: PyTree) -> PyTree:
    args, treedef = jax.tree.flatten(args)
    args = [jnp.unstack(arg) for arg in args]
    args = list(zip(*args))
    return [jax.tree.unflatten(treedef, arg) for arg in args]


@partial(eqx.filter_jit, donate="all")
def _combine_outputs(
    outputs: PyTree, out_axes: int | tuple, chunk_size: int, device_batch: int
) -> PyTree:
    is_tuple = type(outputs) is tuple
    if not is_tuple:
        outputs = (outputs,)

    if isinstance(out_axes, int):
        out_axes = (out_axes,) * len(outputs)

    def fn_combine(x: jax.Array, axis: int) -> jax.Array:
        before = x.shape[1 : axis + 1]
        after = x.shape[axis + 2 :]
        nchunks = x.shape[0]
        x = x.reshape(nchunks, *before, -1, chunk_size, *after)
        x = jnp.moveaxis(x, 0, axis + 1)
        x = x.reshape(*before, -1, nchunks * chunk_size, *after)
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
def _stack_outputs(
    outputs: PyTree, out_axes: int | tuple, chunk_size: int, device_batch: int
) -> PyTree:
    fn_concat = lambda *out: jnp.stack(out, axis=0)
    outputs = filter_tree_map(fn_concat, *outputs)
    return _combine_outputs(outputs, out_axes, chunk_size, device_batch)


def _axes_to_specs(
    axes: tuple[int | None, ...] | int | None,
) -> jax.P | tuple[jax.P, ...]:
    if axes is None:
        return jax.P()  # replicated (no sharded axis)
    elif isinstance(axes, int):
        return get_distributed_P(axes)
    else:
        return tuple(jax.P() if a is None else get_distributed_P(a) for a in axes)


def shmap(
    f: Callable,
    in_axes: tuple | int | None,
    out_axes: tuple | int | None,
    mesh: jax.sharding.Mesh | None = None,
) -> Callable:
    """
    f -> shard_map(f), sharded along the first dimension

    :param f:
        The function to be converted. The arguments of f will be sharded.

    :param in_axes:
        The sharded axes of f input arguments.

    :param out_axes:
        The sharded axes of f outputs.

    :param mesh:
        The device mesh to shard over, default to `~quantax.utils.make_mesh`
        spanning all devices of the current run. Pass a compile-only mesh from
        `~quantax.utils.make_precompile_mesh` to stage the computation for a
        different topology without running on it.
    """

    if mesh is None:
        mesh = make_mesh()
    in_specs = _axes_to_specs(in_axes)
    out_specs = _axes_to_specs(out_axes)

    @wraps(f)
    def sharded_f(*args):
        # shard_map accepts only array leaves; close over eqx static (non-array)
        # leaves (dtypes, callables, hyperparams) instead of passing them through.
        dynamic, static = eqx.partition(args, eqx.is_array)
        body = lambda arrays: f(*eqx.combine(arrays, static))
        fn = jax.shard_map(
            body, mesh=mesh, in_specs=(in_specs,), out_specs=out_specs, check_vma=False
        )
        return fn(dynamic)

    # wraps() keeps __name__ for filter_jit's label; its __wrapped__ makes equinox
    # misread filter_vmap's static in_axes int as a lost parameter and warn.
    del sharded_f.__wrapped__

    return sharded_f


def chunk_map(
    f: Callable,
    in_axes: tuple | int | None = 0,
    out_axes: tuple | int | None = 0,
    chunk_size: int | None = None,
    use_scan: bool = False,
    shard_batch: bool = False,
    atleast_1chunk: bool = False,
    mesh: jax.sharding.Mesh | None = None,
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

    :param shard_batch:
        Whether to run `f` inside `jax.shard_map` so each device chunks and processes
        only its own batch shard (a device-local `lax.scan` inside one `shard_map`),
        instead of GSPMD partitioning a global batch. Needed only for per-sample
        reverse-mode ops where GSPMD inserts cross-device all-gathers -- the
        symmetry-vmapped Jacobian's conv weight-gradient -- so it defaults to False.
        Must stay False when `f` calls another `chunk_map`-wrapped function, because
        `shard_map` cannot be nested over the same mesh axes.

    :param atleast_1chunk:
        Whether to pad the batch with 0 to the chunk size when the batch size on each
        machine is smaller than the chunk size, so that `f` is always called with
        batch size `chunk_size` on each machine. The padded outputs are truncated.
        Default to False, in which case `f` is called with the original batch.

    :param mesh:
        The device mesh that determines the device count for chunking and the
        `shard_batch` sharding, default to `~quantax.utils.make_mesh` spanning all
        devices of the current run (resolved at trace time). Pass a compile-only
        mesh from `~quantax.utils.make_precompile_mesh` to stage the computation
        for a different topology without running on it.
    """
    all_none = isinstance(in_axes, tuple) and all(axis is None for axis in in_axes)
    if in_axes is None or all_none or chunk_size is None:
        if shard_batch:
            return shmap(f, in_axes, out_axes, mesh)
        else:
            return f  # fast return if chunk is not necessary

    any_none = isinstance(out_axes, tuple) and any(axis is None for axis in out_axes)
    if out_axes is None or any_none:
        raise NotImplementedError("`chunk_map` with `out_axes=None` not implemented")

    def chunked_f(*args):
        _in_axes = in_axes if isinstance(in_axes, tuple) else (in_axes,) * len(args)
        ndevices = jax.device_count() if mesh is None else mesh.size

        device_batch = _get_device_batch(args, _in_axes, ndevices)
        if device_batch == chunk_size or (
            device_batch < chunk_size and not atleast_1chunk
        ):
            if shard_batch:
                return shmap(f, _in_axes, out_axes, mesh)(*args)
            else:
                return f(*args)

        dynamic_args, static_args = _chunk_args(args, _in_axes, chunk_size, ndevices)

        def fn_loop(static_args, *dynamic_args):
            if use_scan:
                fn_scan = lambda _, args: (_, f(*eqx.combine(args, static_args)))
                _, outputs = jax.lax.scan(fn_scan, None, list(dynamic_args))
                outputs = _combine_outputs(outputs, out_axes, chunk_size, device_batch)
            else:
                dynamic_args = _unstack_args(list(dynamic_args))
                outputs = [f(*eqx.combine(args, static_args)) for args in dynamic_args]
                outputs = _stack_outputs(outputs, out_axes, chunk_size, device_batch)
            return outputs

        if shard_batch:
            shift = lambda a: a + 1 if a is not None else None
            in_axes_loop = (None,) + tuple(shift(axis) for axis in _in_axes)
            fn_loop = shmap(fn_loop, in_axes_loop, out_axes, mesh)

        out = fn_loop(static_args, *dynamic_args)
        return out

    return chunked_f


def jit_chunk_vmap(
    f: Callable,
    in_axes: int | tuple | None = 0,
    out_axes: int | tuple | None = 0,
    chunk_size: int | None = None,
    shard_batch: bool = False,
    mesh: jax.sharding.Mesh | None = None,
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

    :param shard_batch:
        Forwarded to :func:`chunk_map`; whether to run the vmapped `f` inside
        `jax.shard_map`. Defaults to False; set True only for the per-sample Jacobian.

    :param mesh:
        Forwarded to :func:`chunk_map`; the device mesh for chunking and sharding,
        default to all devices of the current run.
    """
    f = eqx.filter_vmap(f, in_axes=in_axes, out_axes=out_axes)
    f = chunk_map(
        f,
        in_axes,
        out_axes,
        chunk_size,
        use_scan=True,
        shard_batch=shard_batch,
        mesh=mesh,
    )
    f = eqx.filter_jit(f)
    return f
