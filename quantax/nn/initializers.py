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
    Apply the `Lecun normal initializer <https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.lecun_normal.html>`_
    to the weights of the layer. The bias is initialized to 0.

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


def apply_he_normal(key: Key, net: Union[Linear, Conv]) -> Union[Linear, Conv]:
    """
    Apply the `He normal initializer <https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.he_normal.html#jax.nn.initializers.he_normal>`_
    to the weights of the layer. The bias is initialized to 0.

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


def value_pad(value: jax.Array) -> Callable:
    def init(key: Key, shape: Sequence, dtype: Optional[jnp.dtype] = None) -> jax.Array:
        if len(value.shape) != len(shape):
            raise ValueError("Only the value with the same dimension can be extended.")

        pad_width = []
        for l_kernel, l_value in zip(shape, value.shape):
            pad_left = (l_kernel - 1) // 2 - (l_value - 1) // 2
            pad_right = l_kernel // 2 - l_value // 2
            pad_width.append((pad_left, pad_right))

        # pad_width = [
        #     (0, l_kernel - l_value) for l_kernel, l_value in zip(shape, value.shape)
        # ]
        kernel = jnp.pad(value, pad_width)
        if dtype is not None:
            kernel = kernel.astype(dtype)
        return kernel

    return init
