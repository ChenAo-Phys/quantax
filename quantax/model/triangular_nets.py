from typing import Sequence, Callable
from jaxtyping import Key
import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
from jax.nn import initializers
import jax.random as jr
from jax import lax
import equinox as eqx
from ..sites import Triangular, TriangularB
from ..symmetry import Symmetry
from ..nn import (
    lecun_normal,
    exp_by_scale,
    pair_cpl,
    ConvSymmetrize,
    Sequential,
    ReshapeTo_TriangularB,
    triangularb_circularpad,
    Embedding,
)
from ..utils import PsiArray
from ..global_defs import get_lattice, is_default_cpl, get_subkeys


class Triangular_Neighbor_Conv(eqx.Module):
    """Nearest neighbor convolution for the triangular lattice."""

    weight: jax.Array
    bias: jax.Array | None
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    use_mask: bool = eqx.field(static=True)
    dtype: DTypeLike = eqx.field(static=True)
    is_triangularB: bool = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_bias: bool = True,
        kernel_init: Callable = lecun_normal,
        bias_init: Callable = initializers.zeros,
        use_mask: bool = False,
        dtype: DTypeLike = jnp.float32,
        *,
        key: Key,
        **kwargs,
    ):
        r"""
        :param in_channels:
            The number of input channels.

        :param out_channels:
            The number of output channels.

        :param use_bias:
            Whether to add a learnable bias, default to True.

        :param kernel_init:
            The initializer for the convolution kernel, default to `~quantax.nn.lecun_normal`.

        :param bias_init:
            The initializer for the bias, default to zeros.

        :param use_mask:
            If True, weights are only placed on the 7 sites of the triangular
            nearest-neighbor stencil (the center and its 6 neighbors) instead of
            the full :math:`3\times3` kernel, default to False.

        :param dtype:
            The data type of the parameters, default to float32.

        :param key:
            The random key for initializing parameters.
        """
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

    def __call__(self, x: jax.Array) -> jax.Array:
        if x.ndim != 3:
            raise ValueError(f"Input needs to have rank 3, but has shape {x.shape}.")

        x = x.astype(self.weight.dtype)
        if self.is_triangularB:
            x = triangularb_circularpad(x)
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
        if self.use_bias and self.bias is not None:
            x = x + self.bias
        return x


class _ResBlock(eqx.Module):
    """Residual block"""

    conv1: Triangular_Neighbor_Conv
    conv2: Triangular_Neighbor_Conv
    nblock: int = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        nblock: int,
        total_blocks: int,
        dtype: DTypeLike = jnp.float32,
    ):
        def new_layer(is_last: bool) -> Triangular_Neighbor_Conv:
            return Triangular_Neighbor_Conv(
                in_channels=channels,
                out_channels=channels,
                use_bias=not is_last,
                kernel_init=lecun_normal,
                dtype=dtype,
                key=get_subkeys(),
            )

        self.conv1 = new_layer(False)
        self.conv2 = new_layer(nblock == total_blocks - 1)
        self.nblock = nblock

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x.copy()
        x /= np.sqrt(self.nblock + 1, dtype=x.dtype)
        x = jax.nn.gelu(x)
        x = self.conv1(x)
        x = jax.nn.gelu(x)
        x = self.conv2(x)
        return x + residual


class Triangular_ResConv(Sequential):
    r"""
    The `~quantax.model.ResConv` equivalence for `~quantax.sites.Triangular` and
    `~quantax.sites.TriangularB` lattices. The kernel size is fixed as :math:`3\times3`.
    """

    nblocks: int
    channels: int
    sublattice: Sequence[int] | None
    final_activation: Callable[[jax.Array], PsiArray]
    trans_symm: Symmetry | None
    dtype: DTypeLike
    layers: tuple[Callable, ...]
    holomorphic: bool

    def __init__(
        self,
        nblocks: int,
        channels: int,
        sublattice: Sequence[int] | None = None,
        final_activation: Callable[[jax.Array], PsiArray] | None = None,
        trans_symm: Symmetry | None = None,
        dtype: DTypeLike = jnp.float32,
    ):
        r"""
        :param nblocks:
            The number of residual blocks. Each block contains two convolutional layers.

        :param channels:
            The number of channels. Each layer has the same amount of channels.

        :param sublattice:
            The sublattice size of the embedding, default to no sublattice.

        :param final_activation:
            The activation function in the last layer.
            By default, `~quantax.nn.exp_by_scale` is used.

        :param trans_symm:
            The translation symmetry to be applied in the last layer, see `~quantax.nn.ConvSymmetrize`.

        :param dtype:
            The data type of the parameters. Must be a real dtype.
        """
        lattice = get_lattice()
        if isinstance(lattice, Triangular):
            is_triangularB = False
        elif isinstance(lattice, TriangularB):
            is_triangularB = True
        else:
            raise ValueError("The current lattice is not triangular.")

        if np.issubdtype(dtype, np.complexfloating):
            raise ValueError("`Triangular_ResConv` doesn't support complex dtypes.")

        self.nblocks = nblocks
        self.channels = channels
        self.sublattice = sublattice
        if final_activation is None:
            final_activation = exp_by_scale
        self.final_activation = final_activation
        self.trans_symm = trans_symm
        self.dtype = dtype

        blocks = [_ResBlock(channels, i, nblocks, dtype) for i in range(nblocks)]

        layers: list[Callable] = [Embedding(channels, sublattice, dtype)]
        if is_triangularB:
            # Embedding lays the cells out on the natural grid; rearrange them into
            # the same Triangular arrangement `Reshape_TriangularB` used to produce,
            # so the convolutions and `triangularb_circularpad` see the right neighbors.
            layers.append(lambda x: get_lattice().to_neighbor_repr(x))
        layers += [*blocks, lambda x: x / jnp.sqrt(nblocks + 1)]

        if is_default_cpl():
            layers.append(eqx.nn.Lambda(lambda x: pair_cpl(x)))

        layers.append(final_activation)

        if is_triangularB:
            layers.append(ReshapeTo_TriangularB())
        layers.append(ConvSymmetrize(trans_symm))

        super().__init__(layers, holomorphic=False)
