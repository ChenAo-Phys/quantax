from typing import Sequence, Callable
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
import equinox as eqx
from ..nn import (
    Sequential,
    exp_by_scale,
    pair_cpl,
    Embedding,
    ConvSymmetrize,
    Conv,
)
from ..symmetry import Symmetry
from ..utils import PsiArray
from ..global_defs import get_lattice, get_subkeys


class _ConvBlock(eqx.Module):
    """Residual convolution block"""

    norm: Callable
    conv1: Conv
    conv2: Conv

    def __init__(
        self,
        i_block: int,
        channels: int,
        kernel_size: int | Sequence[int],
        use_rmsnorm: bool,
        use_final_bias: bool,
        dtype: DTypeLike = jnp.float32,
    ):
        lattice = get_lattice()

        if use_rmsnorm:
            shape = (channels, *lattice.shape[1:])
            self.norm = eqx.nn.RMSNorm(
                shape, use_weight=False, use_bias=False, dtype=dtype
            )
        else:
            self.norm = lambda x: x / jnp.sqrt(i_block + 1)

        self.conv1 = Conv(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dtype=dtype,
            key=get_subkeys(),
        )

        self.conv2 = Conv(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            use_bias=use_final_bias,
            dtype=dtype,
            key=get_subkeys(),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x.copy()
        x = self.norm(x)
        x = jax.nn.gelu(x)
        x = self.conv1(x)
        x = jax.nn.gelu(x)
        x = self.conv2(x)
        return x + residual


class ResConv(Sequential):
    """Deep convolutional residual network."""

    nblocks: int
    channels: int
    kernel_size: int | Sequence[int]
    final_activation: Callable[[jax.Array], PsiArray]
    trans_symm: Symmetry | None
    dtype: DTypeLike
    out_dtype: DTypeLike | None
    layers: tuple[Callable, ...]
    holomorphic: bool

    def __init__(
        self,
        nblocks: int,
        channels: int,
        kernel_size: int | Sequence[int],
        sublattice: Sequence[int] | None = None,
        use_rmsnorm: bool = False,
        use_final_bias: bool = False,
        final_activation: Callable[[jax.Array], PsiArray] | None = None,
        trans_symm: Symmetry | None = None,
        dtype: DTypeLike = jnp.float32,
        out_dtype: DTypeLike | None = None,
    ):
        """
        The convolutional residual network with a summation in the end.

        :param nblocks:
            The number of residual blocks. Each block contains two convolutional layers.

        :param channels:
            The number of channels. Each layer has the same amount of channels.

        :param kernel_size:
            The kernel size. Each layer has the same kernel size.

        :param sublattice:
            The sublattice size of the embedding, default to no sublattice.

        :param use_rmsnorm:
            Whether to use RMSNorm in each block. Default to False, in which case a manual
            renormalization is applied to rescale the block input by its initial variance.

        :param use_final_bias:
            Whether to add on a bias to the final layer. Default to False.

        :param final_activation:
            The activation function in the last layer.
            By default, `~quantax.nn.exp_by_scale` is used.

        :param trans_symm:
            The translation symmetry to be applied in the last layer, see `~quantax.nn.ConvSymmetrize`.

        :param dtype:
            The data type of the parameters. Must be a real dtype.

        :param out_dtype:
            The data type of the output wavefunction. By default, it is the same as ``dtype``.
            If ``out_dtype`` is complex, `~quantax.nn.pair_cpl` will be applied to the output
            of convolutional layers to make the final output complex.

        .. tip::
            This is the recommended architecture for deep NQS.
        """
        lattice = get_lattice()

        if jnp.issubdtype(dtype, jnp.complexfloating):
            raise ValueError("`ResConv` doesn't support complex dtypes.")

        self.nblocks = nblocks
        self.channels = channels
        self.kernel_size = kernel_size
        if final_activation is None:
            final_activation = exp_by_scale
        self.final_activation = final_activation
        self.trans_symm = trans_symm
        self.dtype = dtype
        self.out_dtype = out_dtype

        blocks = []
        for i in range(nblocks):
            _use_bias = use_final_bias or not i == nblocks - 1
            blocks.append(
                _ConvBlock(i, channels, kernel_size, use_rmsnorm, _use_bias, dtype)
            )

        def final_layer(x):
            x /= jnp.sqrt(nblocks + 1)
            if out_dtype is not None:
                if jnp.issubdtype(out_dtype, jnp.complexfloating):
                    x = pair_cpl(x)
                x = x.astype(out_dtype)
            x = final_activation(x)
            return x

        layers = [
            Embedding(channels, sublattice, dtype),
            lattice.to_neighbor_repr,
            *blocks,
            lattice.to_original_repr,
            final_layer,
            ConvSymmetrize(trans_symm),
        ]

        super().__init__(layers)
