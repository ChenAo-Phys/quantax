from typing import Optional, Callable
from jaxtyping import Key
import numpy as np
import jax
import jax.numpy as jnp
from jax.nn import initializers
import jax.random as jr
from jax import lax
import equinox as eqx
from ..sites import Triangular, TriangularB
from ..symmetry import Symmetry
from ..nn import (
    lecun_normal,
    he_normal,
    SinhShift,
    Exp,
    Scale,
    pair_cpl,
    ReshapeConv,
    ConvSymmetrize,
    Sequential,
)
from ..global_defs import get_lattice, is_default_cpl, get_subkeys
from ..utils import Reshape_TriangularB, ReshapeTo_TriangularB, _triangularb_circularpad

class Triangular_Neighbor_Conv(eqx.Module):
    """Nearest neighbor convolution for the triangular lattice."""

    weight: jax.Array
    bias: Optional[jax.Array]
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    use_mask: bool = eqx.field(static=True)
    dtype: jnp.dtype = eqx.field(static=True)
    is_triangularB: bool = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_bias: bool = True,
        kernel_init: Callable = lecun_normal,
        bias_init: Callable = initializers.zeros,
        use_mask: bool = False,
        dtype: jnp.dtype = jnp.float32,
        *,
        key: Key,
        **kwargs,
    ):
        lattice = get_lattice()
        if isinstance(lattice, Triangular):
            self.is_triangularB = False
        elif isinstance(lattice, TriangularB):
            self.is_triangularB = True
        else:
            raise ValueError("The current lattice is not triangular.")

        super().__init__(**kwargs)
        wkey, bkey = jr.split(key, 2)
        if use_mask:
            kernel_shape = (out_channels, in_channels, 7)
        else:
            kernel_shape = (out_channels, in_channels, 3, 3)
        self.weight = kernel_init(wkey, kernel_shape, dtype)
        if use_bias:
            bias_shape = (out_channels, 1, 1)
            self.bias = bias_init(bkey, bias_shape, dtype)
        else:
            self.bias = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bias = use_bias
        self.use_mask = use_mask
        self.dtype = dtype

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        if x.ndim != 3:
            raise ValueError(f"Input needs to have rank 3, but has shape {x.shape}.")

        x = x.astype(self.weight.dtype)
        if self.is_triangularB:
            x = _triangularb_circularpad(x)
        else:
            x = jnp.pad(x, [(0, 0), (1, 1), (1, 1)], mode="wrap")
        x = jnp.expand_dims(x, axis=0)

        if self.use_mask:
            weight = jnp.pad(self.weight, [(0, 0), (0, 0), (1, 1)])
            weight = weight.reshape(self.out_channels, self.in_channels, 3, 3)
        else:
            weight = self.weight

        x = lax.conv_general_dilated(
            lhs=x, rhs=weight, window_strides=(1, 1), padding="VALID"
        )
        x = jnp.squeeze(x, axis=0)
        if self.use_bias:
            x = x + self.bias
        return x


class _ResBlock(eqx.Module):
    """Residual block"""

    conv1: Triangular_Neighbor_Conv
    conv2: Triangular_Neighbor_Conv
    nblock: int = eqx.field(static=True)

    def __init__(self, channels: int, nblock: int, total_blocks: int):
        def new_layer(is_first: bool, is_last: bool) -> Triangular_Neighbor_Conv:
            lattice = get_lattice()
            in_channels = lattice.shape[0] if is_first else channels
            return Triangular_Neighbor_Conv(
                in_channels=in_channels,
                out_channels=channels,
                use_bias=not is_last,
                kernel_init=he_normal,
                key=get_subkeys(),
            )

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
        return x + residual


def Triangular_ResSum(
    nblocks: int,
    channels: int,
    final_activation: Optional[Callable] = None,
    trans_symm: Optional[Symmetry] = None,
    dtype: jnp.dtype = jnp.float32,
):
    r"""
    The `~quantax.model.ResSum` equivalence for `~quantax.sites.Triangular` and
    `~quantax.sites.TriangularB` lattices. The kernel size is fixed as :math:`3\times3`.

    :param nblocks:
        The number of residual blocks. Each block contains two convolutional layers.

    :param channels:
        The number of channels. Each layer has the same amount of channels.

    :param final_activation:
        The activation function in the last layer.
        By default, `~quantax.nn.Exp` is used.

    :param trans_symm:
        The translation symmetry to be applied in the last layer, see `~quantax.nn.ConvSymmetrize`.

    :param dtype:
        The data type of the parameters.
    """
    lattice = get_lattice()
    if isinstance(lattice, Triangular):
        is_triangularB = False
    elif isinstance(lattice, TriangularB):
        is_triangularB = True
    else:
        raise ValueError("The current lattice is not triangular.")

    if np.issubdtype(dtype, np.complexfloating):
        raise ValueError(f"`ResSum` doesn't support complex dtypes.")

    blocks = [_ResBlock(channels, i, nblocks) for i in range(nblocks)]

    reshape = Reshape_TriangularB(dtype) if is_triangularB else ReshapeConv(dtype)
    scale = Scale(1 / np.sqrt(nblocks + 1))
    layers = [reshape, *blocks, scale]

    if is_default_cpl():
        layers.append(eqx.nn.Lambda(lambda x: pair_cpl(x)))

    if final_activation is None:
        final_activation = Exp()
    elif not isinstance(final_activation, eqx.Module):
        final_activation = eqx.nn.Lambda(final_activation)
    layers.append(final_activation)

    if is_triangularB:
        layers.append(ReshapeTo_TriangularB())
    layers.append(ConvSymmetrize(trans_symm))

    return Sequential(layers, holomorphic=False)
