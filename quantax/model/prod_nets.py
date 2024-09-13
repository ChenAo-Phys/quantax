from typing import Callable, Optional
from jaxtyping import Key
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from equinox.nn import Conv
from ..nn import (
    Sequential,
    apply_he_normal,
    apply_lecun_normal,
    ScaleFn,
    Prod,
    Exp,
    ReshapeConv,
)
from ..global_defs import get_lattice, is_default_cpl, get_subkeys


class _ResBlock(eqx.Module):
    """Residual block"""

    conv1: Conv
    conv2: Conv
    nblock: int = eqx.field(static=True)

    def __init__(self, channels: int, kernel_size: int, nblock: int, dtype: jnp.dtype):
        lattice = get_lattice()

        def new_layer(is_first_layer: bool) -> Conv:
            if is_first_layer:
                in_channels = lattice.shape[0]
                if lattice.is_fermion:
                    in_channels *= 2
            else:
                in_channels = channels
            key = get_subkeys()
            conv = Conv(
                num_spatial_dims=lattice.ndim,
                in_channels=in_channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding="SAME",
                padding_mode="CIRCULAR",
                dtype=dtype,
                key=key,
            )
            return apply_he_normal(key, conv)

        self.conv1 = new_layer(nblock == 0)
        self.conv2 = new_layer(False)
        self.nblock = nblock

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None):
        residual = x.copy()

        for i, layer in enumerate([self.conv1, self.conv2]):
            if i == 0 and self.nblock == 0:
                x /= np.sqrt(2, dtype=x.dtype)
            else:
                mean = jnp.mean(x, axis=0, keepdims=True)
                var = jnp.var(x, axis=0, keepdims=True)
                x = (x - mean) / jnp.sqrt(var + 1e-6)
                x = jax.nn.gelu(x)
            x = layer(x)

        return x + residual


def ResProd(
    nblocks: int,
    channels: int,
    kernel_size: int,
    final_actfn: Callable,
    dtype: jnp.dtype = jnp.float32,
):
    """
    The convolutional residual network with a product in the end.
    This network still requires further tests.

    :param nblocks:
        The number of residual blocks. Each block contains two convolutional layers.

    :param channels:
        The number of channels. Each layer has the same amount of channels.

    :param kernel_size:
        The kernel size. Each layer has the same kernel size.

    :param final_actfn:
        The activation function in the last layer.

    :param dtype:
        The data type of the parameters.
    """
    if np.issubdtype(dtype, np.complexfloating):
        raise ValueError("`ResProd` doesn't support complex dtypes")
    blocks = [_ResBlock(channels, kernel_size, i, dtype) for i in range(nblocks)]
    out_features = channels * get_lattice().ncells
    scale_fn = ScaleFn(final_actfn, out_features, 1 / np.sqrt(nblocks), dtype)
    return Sequential(
        [ReshapeConv(dtype), *blocks, scale_fn, Prod()], holomorphic=False
    )


def SinhCosh(
    depth: int, channels: int, kernel_size: int, dtype: jnp.dtype = jnp.complex64
):
    lattice = get_lattice()

    def new_layer(is_first_layer: bool) -> Conv:
        in_channels = lattice.shape[0] if is_first_layer else channels
        key = get_subkeys()
        conv = Conv(
            num_spatial_dims=lattice.ndim,
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding="SAME",
            padding_mode="CIRCULAR",
            use_bias=False,
            dtype=dtype,
            key=key,
        )
        return apply_lecun_normal(key, conv)

    out_features = channels * lattice.ncells
    scale_fn = ScaleFn(jnp.cosh, out_features, dtype=dtype)
    scale = scale_fn.scale
    layers = [ReshapeConv(dtype), eqx.nn.Lambda(lambda x: x * scale)]

    for i in range(depth):
        layers.append(new_layer(is_first_layer=i == 0))
        if i < depth - 1:
            layers.append(eqx.nn.Lambda(jnp.sinh))

    layers.append(eqx.nn.Lambda(jnp.cosh))
    layers.append(Prod())
    return Sequential(layers, holomorphic=is_default_cpl())


def SchmittNet(
    depth: int, channels: int, kernel_size: int, dtype: jnp.dtype = jnp.complex64
):
    """
    CNN defined in `PRL 125, 100503 <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.100503>`_.

    :param depth:
        The depth of the network.

    :param channels:
        The number of channels. Each layer has the same amount of channels.

    :param kernel_size:
        The kernel size. Each layer has the same kernel size.

    :param dtype:
        The data type of the parameters.
    """
    lattice = get_lattice()
    fn_first = lambda z: z**2 / 2 - z**4 / 14 + z**6 / 45
    actfn_first = eqx.nn.Lambda(fn_first)
    actfn = eqx.nn.Lambda(lambda z: z - z**3 / 3 + z**5 * 2 / 15)

    fn = lambda z: jnp.exp(fn_first(z))
    scale_fn = ScaleFn(fn, channels * lattice.ncells, dtype=dtype)
    scale_layer = eqx.nn.Lambda(lambda x: x * scale_fn.scale)
    layers = [ReshapeConv(dtype), scale_layer]
    for i in range(depth):
        key = get_subkeys()
        conv = Conv(
            num_spatial_dims=lattice.ndim,
            in_channels=lattice.shape[0] if i == 0 else channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding="SAME",
            padding_mode="CIRCULAR",
            use_bias=False,
            dtype=dtype,
            key=key,
        )
        layers.append(apply_lecun_normal(key, conv))
        layers.append(actfn_first if i == 0 else actfn)
    layers.append(eqx.nn.Lambda(lambda z: jnp.sum(z)))
    layers.append(Exp())

    return Sequential(layers, holomorphic=is_default_cpl())
