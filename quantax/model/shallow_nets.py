from typing import Callable
import numpy as np
import jax.numpy as jnp
from equinox.nn import Linear, Conv
from ..nn import Sequential, apply_lecun_normal, ScaleFn, Prod, ReshapeConv
from ..global_defs import get_sites, get_lattice, get_subkeys


def SingleDense(
    features: int,
    actfn: Callable,
    use_bias: bool = True,
    holomorphic: bool = False,
    dtype: jnp.dtype = jnp.float32,
):
    r"""
    Network with one dense layer :math:`\psi(s) = \prod f(W s + b)`.

    :param features:
        The number of output features or hidden units.

    :param actfn:
        The activation function applied after the dense layer.

    :param use_bias:
        Whether to add on a bias.

    :param holomorphic:
        Whether the whole network is complex holomorphic.

    :param dtype:
        The data type of the parameters.
    """
    nsites = get_sites().nstates
    key = get_subkeys()
    linear = Linear(nsites, features, use_bias, dtype, key=key)
    linear = apply_lecun_normal(key, linear)
    layers = [linear, ScaleFn(actfn, features, dtype=dtype), Prod()]
    return Sequential(layers, holomorphic)


def RBM_Dense(features: int, use_bias: bool = True, dtype: jnp.dtype = jnp.float32):
    r"""
    The restricted Boltzmann machine with one dense layer 
    :math:`\psi(s) = \prod \cosh(W s + b)`.

    :param features:
        The number of output features or hidden units.

    :param use_bias:
        Whether to add on a bias.

    :param dtype:
        The data type of the parameters.
    """
    holomorphic = np.issubdtype(dtype, np.complexfloating)
    return SingleDense(features, jnp.cosh, use_bias, holomorphic, dtype)


def SingleConv(
    channels: int,
    actfn: Callable,
    use_bias: bool = True,
    holomorphic: bool = False,
    dtype: jnp.dtype = jnp.float32,
):
    r"""
    Network with one convolutional layer 
    :math:`\psi(s) = \prod f(\mathrm{Conv}(s))`.

    :param features:
        The number of channels in the convolutional network.

    :param actfn:
        The activation function applied after the convolutional layer.

    :param use_bias:
        Whether to add on a bias in the convolution.

    :param holomorphic:
        Whether the whole network is complex holomorphic.

    :param dtype:
        The data type of the parameters.
    """
    lattice = get_lattice()
    key = get_subkeys()
    conv = Conv(
        num_spatial_dims=lattice.ndim,
        in_channels=lattice.shape[0],
        out_channels=channels,
        kernel_size=lattice.shape[1:],
        padding="SAME",
        use_bias=use_bias,
        padding_mode="CIRCULAR",
        dtype=dtype,
        key=key,
    )
    conv = apply_lecun_normal(key, conv)
    scalefn = ScaleFn(actfn, features=channels * lattice.ncells, dtype=dtype)
    layers = [ReshapeConv(dtype), conv, scalefn, Prod()]
    return Sequential(layers, holomorphic)


def RBM_Conv(channels: int, use_bias: bool = True, dtype: jnp.dtype = jnp.float32):
    r"""
    The restricted Boltzmann machine with one convolutional layer 
    :math:`\psi(s) = \prod \cosh(\mathrm{Conv}(s))`.

    :param features:
        The number of channels in the convolutional network.

    :param use_bias:
        Whether to add on a bias in the convolution.

    :param dtype:
        The data type of the parameters.
    """
    holomorphic = np.issubdtype(dtype, np.complexfloating)
    return SingleConv(channels, jnp.cosh, use_bias, holomorphic, dtype)
