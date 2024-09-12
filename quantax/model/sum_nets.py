from typing import Optional, Union, Sequence
from jaxtyping import Key
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from equinox.nn import Conv
from ..nn import (
    Sequential,
    apply_he_normal,
    SinhShift,
    Exp,
    Scale,
    pair_cpl,
    ReshapeConv,
    ConvSymmetrize,
)
from ..symmetry import Symmetry, Identity
from ..global_defs import get_lattice, is_default_cpl, get_subkeys


class _ResBlock(eqx.Module):
    """Residual block"""

    conv1: Conv
    conv2: Conv
    nblock: int = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        nblock: int,
        total_blocks: int,
        dtype: jnp.dtype = jnp.float32,
    ):
        lattice = get_lattice()

        def new_layer(is_first_layer: bool, is_last_layer: bool) -> Conv:
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
                use_bias=not is_last_layer,
                padding_mode="CIRCULAR",
                dtype=dtype,
                key=key,
            )
            conv = apply_he_normal(key, conv)
            return conv

        self.conv1 = new_layer(nblock == 0, False)
        self.conv2 = new_layer(False, nblock == total_blocks - 1)
        self.nblock = nblock

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        residual = x.copy()
        x /= np.sqrt(self.nblock + 1, dtype=x.dtype)

        if self.nblock == 0:
            x /= np.sqrt(2, dtype=x.dtype)
        else:
            x = jax.nn.gelu(x)
        x = self.conv1(x)
        x = jax.nn.gelu(x)
        x = self.conv2(x)

        if x.shape[0] > residual.shape[0]:
            residual = jnp.repeat(residual, x.shape[0] // residual.shape[0], axis=0)
        return x + residual


def ResSum(
    nblocks: int,
    channels: int,
    kernel_size: Union[int, Sequence[int]],
    use_sinh: bool = False,
    trans_symm: Optional[Symmetry] = None,
    dtype: jnp.dtype = jnp.float32,
):
    """
    The convolutional residual network with a summation in the end.

    :param nblocks:
        The number of residual blocks. Each block contains two convolutional layers.

    :param channels:
        The number of channels. Each layer has the same amount of channels.

    :param kernel_size:
        The kernel size. Each layer has the same kernel size.

    :param use_sinh:
        Whether to use `~quantax.nn.SinhShift` as the activation function in the end.
        By default, ``use_sinh = False``, in which case the combination of 
        `~quantax.nn.pair_cpl` and `~quantax.nn.Exp` is used.

    :param trans_symm:
        The translation symmetry to be applied in the last layer, see `~quantax.nn.ConvSymmetrize`.
    
    :param dtype:
        The data type of the parameters.

    .. tip::
        This is the recommended architecture for deep neural quantum states.
    """
    if np.issubdtype(dtype, np.complexfloating):
        raise ValueError("`ResSum` doesn't support complex dtypes.")

    blocks = [
        _ResBlock(channels, kernel_size, i, nblocks, dtype) for i in range(nblocks)
    ]

    scale = Scale(1 / np.sqrt(nblocks))
    layers = [ReshapeConv(dtype), *blocks, scale]

    if is_default_cpl():
        cpl_layer = eqx.nn.Lambda(lambda x: pair_cpl(x))
        layers.append(cpl_layer)
    layers.append(SinhShift() if use_sinh else Exp())
    if trans_symm is not Identity():
        layers.append(ConvSymmetrize(trans_symm))

    return Sequential(layers, holomorphic=False)
