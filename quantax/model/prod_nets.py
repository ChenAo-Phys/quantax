from typing import Callable, Optional
from jaxtyping import Key
import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
from equinox.nn import Conv
from .modules import Sequential
from .initializers import apply_he_normal, apply_lecun_normal
from .activation import ScaleFn, Prod, ExpSum
from .nqs_layers import ReshapeConv
from ..global_defs import get_lattice, get_params_dtype, is_default_cpl, get_subkeys


class _ResBlock(eqx.Module):
    """Residual block"""
    conv1: Conv
    conv2: Conv
    nblock: int = eqx.field(static=True)

    def __init__(self, channels: int, kernel_size: int, nblock: int):
        lattice = get_lattice()
        dtype = get_params_dtype()

        def new_layer(is_first_layer: bool) -> Conv:
            in_channels = lattice.shape[-1] if is_first_layer else channels
            key = get_subkeys()
            conv = Conv(
                num_spatial_dims=lattice.dim,
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
    depth: int, channels: int, kernel_size: int, final_actfn: Callable
):
    """
    Requires further tests...
    """
    if depth % 2:
        raise ValueError(f"'depth' should be a multiple of 2, got {depth}")
    num_blocks = depth // 2
    blocks = [_ResBlock(channels, kernel_size, i) for i in range(num_blocks)]
    out_features = channels * np.prod(get_lattice().shape[:-1])
    scale_fn = ScaleFn(final_actfn, out_features, scaling=1 / np.sqrt(num_blocks))
    return Sequential([ReshapeConv(), *blocks, scale_fn, Prod()], holomorphic=False)


def SinhCosh(depth: int, channels: int, kernel_size: int):
    def new_layer(is_first_layer: bool) -> Conv:
        lattice = get_lattice()
        in_channels = lattice.shape[-1] if is_first_layer else channels
        key = get_subkeys()
        conv = Conv(
            num_spatial_dims=lattice.dim,
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding="SAME",
            padding_mode="CIRCULAR",
            use_bias=False,
            dtype=get_params_dtype(),
            key=key,
        )
        return apply_lecun_normal(key, conv)

    out_features = channels * np.prod(get_lattice().shape[:-1])
    scale_fn = ScaleFn(jnp.cosh, out_features)
    scale = scale_fn.scale
    layers = [ReshapeConv(), eqx.nn.Lambda(lambda x: x * scale)]

    for i in range(depth):
        layers.append(new_layer(is_first_layer=i == 0))
        if i < depth - 1:
            layers.append(eqx.nn.Lambda(jnp.sinh))

    layers.append(eqx.nn.Lambda(jnp.cosh))
    layers.append(Prod())
    return Sequential(layers, holomorphic=is_default_cpl())


def SchmittNet(depth: int, channels: int, kernel_size: int):
    """CNN defined in Phys. Rev. Lett. 125, 100503"""
    lattice = get_lattice()
    fn_first = lambda z: z**2 / 2 - z**4 / 14 + z**6 / 45
    actfn_first = eqx.nn.Lambda(fn_first)
    actfn = eqx.nn.Lambda(lambda z: z - z**3 / 3 + z**5 * 2 / 15)

    fn = lambda z: jnp.exp(fn_first(z))
    scale_fn = ScaleFn(fn, channels * np.prod(lattice.shape[:-1]))
    scale_layer = eqx.nn.Lambda(lambda x: x * scale_fn.scale)
    layers = [ReshapeConv(), scale_layer]
    for i in range(depth):
        key = get_subkeys()
        conv = Conv(
            num_spatial_dims=lattice.dim,
            in_channels=lattice.shape[-1] if i == 0 else channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding="SAME",
            padding_mode="CIRCULAR",
            use_bias=False,
            dtype=get_params_dtype(),
            key=key,
        )
        layers.append(apply_lecun_normal(key, conv))
        layers.append(actfn_first if i == 0 else actfn)
    layers.append(ExpSum())

    return Sequential(layers, holomorphic=is_default_cpl())
