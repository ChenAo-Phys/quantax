from typing import Tuple
from jaxtyping import PyTree
import jax
import jax.tree_util as jtu
import jax.flatten_util as jfu
import equinox as eqx
from .array import to_replicate_array


def tree_fully_flatten(tree: PyTree) -> jax.Array:
    array, unravel_fn = jfu.ravel_pytree(tree)
    return array


def filter_replicate(tree: PyTree) -> PyTree:
    vals, tree_def = jtu.tree_flatten(tree)
    new_vals = []
    for val in vals:
        if eqx.is_array(val):
            new_vals.append(to_replicate_array(val))
        else:
            new_vals.append(val)

    return jtu.tree_unflatten(tree_def, new_vals)


def tree_split_cpl(tree: PyTree) -> Tuple[PyTree, PyTree]:
    get_real = lambda x: x.real if eqx.is_inexact_array(x) else x
    get_imag = lambda x: x.imag if eqx.is_inexact_array(x) else x
    tree_real = jtu.tree_map(get_real, tree)
    tree_imag = jtu.tree_map(get_imag, tree)
    return tree_real, tree_imag


def tree_combine_cpl(tree_real: PyTree, tree_imag: PyTree) -> PyTree:
    get_cpl = lambda x, y: x + 1j * y if eqx.is_inexact_array(x) else x
    return jtu.tree_map(get_cpl, tree_real, tree_imag)
