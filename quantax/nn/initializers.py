from typing import Callable, Sequence, Union, Optional
from jaxtyping import Key
from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.nn import initializers
import equinox as eqx
from equinox.nn import Linear, Conv


variance_scaling = partial(
    initializers.variance_scaling, in_axis=1, out_axis=0, batch_axis=()
)


def _fix_init_axis(initializer: Callable) -> Callable:
    return initializer(in_axis=1, out_axis=0, batch_axis=())


lecun_normal = _fix_init_axis(initializers.lecun_normal)
lecun_uniform = _fix_init_axis(initializers.lecun_uniform)
glorot_normal = _fix_init_axis(initializers.glorot_normal)
glorot_uniform = _fix_init_axis(initializers.glorot_uniform)
he_normal = _fix_init_axis(initializers.he_normal)
he_uniform = _fix_init_axis(initializers.he_uniform)


def apply_lecun_normal(key: Key, net: Union[Linear, Conv]) -> Union[Linear, Conv]:
    """
    Apply the `Lecun normal initializer <https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.lecun_normal.html>`_.
    The bias is initialized to 0.

    :param key:
        The random key used in JAX for initializing parameters.

    :param net:
        The net to apply the initializer.

    :return:
        The net with properly initialized parameters.

    .. note::

        This function can only be applied to ``Linear`` or ``Conv`` layers in Equinox.

    .. note::

        The input ``net`` is not modified.
    """
    wkey, bkey = jr.split(key, 2)  # consistent with eqx keys
    weight = lecun_normal(wkey, net.weight.shape, net.weight.dtype)
    net = eqx.tree_at(lambda tree: tree.weight, net, weight)
    if net.use_bias:
        bias = jnp.zeros_like(net.bias)
        net = eqx.tree_at(lambda tree: tree.bias, net, bias)
    return net


def apply_glorot_normal(key: Key, net: Union[Linear, Conv]) -> Union[Linear, Conv]:
    """
    Apply the `Glorot normal initializer <https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.glorot_normal.html>`_.
    The bias is initialized to 0.

    :param key:
        The random key used in JAX for initializing parameters.

    :param net:
        The net to apply the initializer.

    :return:
        The net with properly initialized parameters.

    .. note::

        This function can only be applied to ``Linear`` or ``Conv`` layers in Equinox.

    .. note::

        The input ``net`` is not modified.
    """
    wkey, bkey = jr.split(key, 2)  # consistent with eqx keys
    weight = glorot_normal(wkey, net.weight.shape, net.weight.dtype)
    net = eqx.tree_at(lambda tree: tree.weight, net, weight)
    if net.use_bias:
        bias = jnp.zeros_like(net.bias)
        net = eqx.tree_at(lambda tree: tree.bias, net, bias)
    return net


def apply_he_normal(key: Key, net: Union[Linear, Conv]) -> Union[Linear, Conv]:
    """
    Apply the `He normal initializer <https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.he_normal.html#jax.nn.initializers.he_normal>`_.
    The bias is initialized to 0.

    :param key:
        The random key used in JAX for initializing parameters.

    :param net:
        The net to apply the initializer.

    :return:
        The net with properly initialized parameters.

    .. note::

        This function can only be applied to ``Linear`` or ``Conv`` layers in Equinox.

    .. note::

        The input ``net`` is not modified.
    """
    wkey, bkey = jr.split(key, 2)  # consistent with eqx keys
    weight = he_normal(wkey, net.weight.shape, net.weight.dtype)
    net = eqx.tree_at(lambda tree: tree.weight, net, weight)
    if net.use_bias:
        bias = jnp.zeros_like(net.bias)
        net = eqx.tree_at(lambda tree: tree.bias, net, bias)
    return net
