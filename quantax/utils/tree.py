from typing import Tuple, Callable
from numbers import Number
from jaxtyping import PyTree
import jax
import jax.tree_util as jtu
import jax.flatten_util as jfu
import equinox as eqx
from .sharding import get_distribute_sharding
from .array import to_replicate_array, array_extend


def tree_fully_flatten(tree: PyTree) -> jax.Array:
    """Return the array given by `jax.flatten_util.ravel_pytree`."""
    array, unravel_fn = jfu.ravel_pytree(tree)
    return array


def filter_global(tree: PyTree) -> PyTree:
    """
    Transform the arrays in pytree to be sharded on all devices.
    See `~quantax.utils.get_global_sharding` for the sharding.
    """
    return eqx.filter_shard(tree, get_distribute_sharding())


def filter_replicate(tree: PyTree) -> PyTree:
    """
    Transform the arrays in pytree to be replicated on all devices.
    See `~quantax.utils.get_replicate_sharding` for the sharding.
    """
    vals, tree_def = jtu.tree_flatten(tree)
    new_vals = []
    for val in vals:
        if eqx.is_array(val):
            new_vals.append(to_replicate_array(val))
        else:
            new_vals.append(val)

    return jtu.tree_unflatten(tree_def, new_vals)


def filter_extend(
    tree: PyTree, multiple_of_num: int, axis: int = 0, padding_values: Number = 0
) -> PyTree:
    """
    The pytree version of `~quantax.utils.array_extend`.
    """
    vals, tree_def = jtu.tree_flatten(tree)
    new_vals = []
    for val in vals:
        if eqx.is_array(val):
            new_vals.append(array_extend(val, multiple_of_num, axis, padding_values))
        else:
            new_vals.append(val)

    return jtu.tree_unflatten(tree_def, new_vals)


def filter_tree_map(f: Callable, tree: PyTree, *rest: Tuple[PyTree]) -> PyTree:
    """
    The same as `jax.tree.map` but with filter, which means the map only applies to arrays.
    """
    f_filter = lambda x, *rest: f(x, *rest) if eqx.is_array(x) else x
    return jax.tree.map(f_filter, tree, *rest)


def tree_split_cpl(tree: PyTree) -> Tuple[PyTree, PyTree]:
    """
    Split a pytree potentially with complex values to two real pytrees, one for the real part
    and the other for the imaginary part.
    """
    get_real = lambda x: x.real if eqx.is_inexact_array(x) else x
    get_imag = lambda x: x.imag if eqx.is_inexact_array(x) else x
    tree_real = jtu.tree_map(get_real, tree)
    tree_imag = jtu.tree_map(get_imag, tree)
    return tree_real, tree_imag


def tree_combine_cpl(tree_real: PyTree, tree_imag: PyTree) -> PyTree:
    """
    Combine two real pytrees to a complex one. It's the inverse operation of
    `~quantax.utils.tree_split_cpl`.
    """
    get_cpl = lambda x, y: x + 1j * y if eqx.is_inexact_array(x) else x
    return jtu.tree_map(get_cpl, tree_real, tree_imag)


def apply_updates(model: PyTree, updates: PyTree) -> PyTree:
    """
    Similar to `equinox.apply_updates <https://docs.kidger.site/equinox/api/manipulation/#equinox.apply_updates>`_,
    but the original data type of the model is kept unchanged.
    """

    def fn(u, p):
        if u is None:
            return p
        elif eqx.is_array(u) and eqx.is_array(p):
            return p + u.astype(p.dtype)
        else:
            return p + u

    is_none = lambda x: x is None

    return jax.tree.map(fn, updates, model, is_leaf=is_none)
