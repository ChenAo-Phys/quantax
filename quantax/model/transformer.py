from typing import Optional, Union, Sequence, Callable
from jaxtyping import Key
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from ..nn import (
    apply_he_normal,
    apply_lecun_normal,
    Scale,
    ReshapeConv,
    Sequential,
    pair_cpl,
    Exp,
    ConvSymmetrize,
)
from ..symmetry import Symmetry
from ..global_defs import get_lattice, get_subkeys, is_default_cpl


class CNN_Block(eqx.Module):
    nlayer: int = eqx.field(static=True)
    conv: eqx.nn.Conv

    def __init__(
        self,
        nlayer: int,
        channels: int,
        kernel_size: Union[int, Sequence[int]],
        use_bias: bool = True,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.nlayer = nlayer
        key = get_subkeys()
        conv = eqx.nn.Conv(
            num_spatial_dims=get_lattice().ndim,
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            use_bias=use_bias,
            padding="SAME",
            padding_mode="CIRCULAR",
            dtype=dtype,
            key=key,
        )
        self.conv = apply_he_normal(key, conv)

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None):
        residual = x
        x /= np.sqrt(self.nlayer + 1, dtype=x.dtype)
        x = jax.nn.gelu(x)
        x = self.conv(x)
        return x + residual


@jax.jit
@jax.vmap
def get_irpe(R: jax.Array) -> jax.Array:
    """
    Compute image relative positional encoding from parameter matrix R
    output r[d, i, j] = R[d, x, y], in which x = xi - xj, y = yi - yj
    """
    arange = [np.arange(i) for i in R.shape]
    idx = jnp.stack(jnp.meshgrid(*arange, indexing="ij"), axis=-1)
    idx1 = jnp.expand_dims(idx, np.arange(R.ndim, 2 * R.ndim).tolist())
    idx2 = jnp.expand_dims(idx, np.arange(R.ndim).tolist())
    diff = (idx1 - idx2).reshape(R.size * R.size, R.ndim)
    diff = tuple(diff.T)
    return R[diff].reshape(R.size, R.size)


class CvT_Block(eqx.Module):
    nlayer: int = eqx.field(static=True)
    dk: int = eqx.field(static=True)
    heads: int = eqx.field(static=True)
    convQ: eqx.nn.Conv
    convK: eqx.nn.Conv
    convV: eqx.nn.Conv
    R: jax.Array
    W0: jax.Array

    def __init__(
        self,
        nlayer: int,
        channels: int,
        dk: int,
        heads: int,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.nlayer = nlayer
        self.dk = dk
        self.heads = heads
        lattice = get_lattice()
        ndim = lattice.ndim
        keys = get_subkeys(4)

        convQ = eqx.nn.Conv(
            ndim,
            channels,
            dk * heads,
            kernel_size=1,
            use_bias=False,
            dtype=dtype,
            key=keys[0],
        )
        self.convQ = apply_lecun_normal(keys[0], convQ)

        convK = eqx.nn.Conv(
            ndim,
            channels,
            dk * heads,
            kernel_size=1,
            use_bias=False,
            dtype=dtype,
            key=keys[1],
        )
        self.convK = apply_lecun_normal(keys[1], convK)

        convV = eqx.nn.Conv(
            ndim,
            channels,
            dk * heads,
            kernel_size=1,
            use_bias=False,
            dtype=dtype,
            key=keys[2],
        )
        self.convV = apply_lecun_normal(keys[2], convV)

        shape = np.array(lattice.shape[1:])  # // 2
        self.R = jnp.zeros((heads, *shape), dtype)

        initializer = jax.nn.initializers.lecun_normal(
            in_axis=1, out_axis=0, dtype=dtype
        )
        self.W0 = initializer(keys[3], (channels, dk * heads), dtype)

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        residual = x
        x /= np.sqrt(self.nlayer + 1, dtype=x.dtype)
        # x = self.layernorm(x)

        Q = self.convQ(x).reshape(self.heads, self.dk, -1)
        K = self.convK(x).reshape(self.heads, self.dk, -1)
        dot = jnp.einsum("hdi,hdj->hij", Q, K) + get_irpe(self.R)
        alpha = jax.nn.softmax(dot / jnp.sqrt(self.dk))

        V = self.convV(x).reshape(self.heads, self.dk, -1)
        attention = jnp.einsum("hij,hdj->hdi", alpha, V)
        attention = attention.reshape(-1, attention.shape[2])
        attention = self.W0 @ attention
        return attention.reshape(residual.shape) + residual


class IRFFN(eqx.Module):
    nlayer: int
    conv1: eqx.nn.Conv
    conv2: eqx.nn.Conv
    conv3: eqx.nn.Conv

    def __init__(
        self,
        nlayer: int,
        channels: int,
        kernel_size: Union[int, Sequence[int]],
        dtype: jnp.dtype = jnp.float32,
    ):
        self.nlayer = nlayer
        ndim = get_lattice().ndim
        keys = get_subkeys(3)

        conv1 = eqx.nn.Conv(
            ndim,
            channels,
            4 * channels,
            kernel_size=1,
            dtype=dtype,
            key=keys[0],
        )
        self.conv1 = apply_he_normal(keys[0], conv1)

        conv2 = eqx.nn.Conv(
            ndim,
            4 * channels,
            4 * channels,
            kernel_size=kernel_size,
            groups=4 * channels,
            padding="SAME",
            padding_mode="CIRCULAR",
            dtype=dtype,
            key=keys[1],
        )
        self.conv2 = apply_lecun_normal(keys[1], conv2)

        conv3 = eqx.nn.Conv(
            ndim,
            4 * channels,
            channels,
            kernel_size=1,
            dtype=dtype,
            key=keys[2],
        )
        self.conv3 = apply_he_normal(keys[2], conv3)

    def __call__(self, x, *, key=None):
        residual = x
        x /= np.sqrt(self.nlayer + 1, dtype=x.dtype)
        x = self.conv1(x)
        x = jax.nn.gelu(x)
        x = self.conv2(x)
        x = jax.nn.gelu(x)
        x = self.conv3(x)
        return x + residual


def ConvTransformer(
    nblocks: int,
    channels: int,
    kernel_size: Union[int, Sequence[int]],
    d: int,
    h: int,
    final_activation: Optional[Callable] = None,
    trans_symm: Optional[Symmetry] = None,
    dtype: jnp.dtype = jnp.float32,
):
    key = get_subkeys()
    conv_embed = eqx.nn.Conv(
        num_spatial_dims=get_lattice().ndim,
        in_channels=1,
        out_channels=channels,
        kernel_size=3,
        # stride=(2,2),  # this could be added to reduce transformer complexity
        padding="SAME",
        padding_mode="CIRCULAR",
        dtype=dtype,
        key=key,
    )
    conv_embed = apply_lecun_normal(key, conv_embed)
    layers = [ReshapeConv(), conv_embed]

    for i in range(nblocks):
        layers.append(CNN_Block(3 * i, channels, kernel_size, dtype))
        layers.append(CvT_Block(3 * i + 1, channels, d, h, dtype))
        layers.append(IRFFN(3 * i + 2, channels, kernel_size, dtype))

    layers.append(Scale(1 / np.sqrt(3 * nblocks + 1)))
    if is_default_cpl():
        layers.append(eqx.nn.Lambda(lambda x: pair_cpl(x)))

    if final_activation is None:
        final_activation = Exp()
    elif not isinstance(final_activation, eqx.Module):
        final_activation = eqx.nn.Lambda(final_activation)

    layers.append(final_activation)
    layers.append(ConvSymmetrize(trans_symm))

    return Sequential(layers, holomorphic=False)
