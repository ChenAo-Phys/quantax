from typing import Optional, Union, Sequence
from jaxtyping import Key
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from ..symmetry import Symmetry, Identity
from ..nn import (
    Depthwise_Separable_Conv,
    apply_he_normal,
    Scale,
    ReshapeConv,
    SinhShift,
    ConvSymmetrize,
    Sequential,
)
from ..global_defs import get_lattice, get_subkeys


@jax.jit
@jax.vmap
def _get_irpe(R: jax.Array) -> jax.Array:
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
    nblock: int = eqx.field(static=True)
    dk: int = eqx.field(static=True)
    heads: int = eqx.field(static=True)
    conv_embed: eqx.nn.Conv
    convQ: eqx.nn.Conv
    convK: Depthwise_Separable_Conv
    convV: Depthwise_Separable_Conv
    RQ: jax.Array
    RK: jax.Array
    RV: jax.Array
    W0: jax.Array

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dk: int,
        heads: int,
        nblock: int,
    ):
        self.nblock = nblock
        self.dk = dk
        self.heads = heads
        lattice = get_lattice()
        ndim = lattice.ndim
        dtype = jnp.float32
        keys = get_subkeys(5)

        in_channels = lattice.shape[0] if nblock == 0 else channels
        conv_embed = eqx.nn.Conv(
            ndim,
            in_channels,
            channels,
            kernel_size,
            padding="SAME",
            padding_mode="CIRCULAR",
            dtype=dtype,
            key=keys[0],
        )
        self.conv_embed = apply_he_normal(keys[0], conv_embed)

        self.convQ = eqx.nn.Conv(
            ndim,
            channels,
            dk * heads,
            kernel_size=1,
            use_bias=False,
            dtype=dtype,
            key=keys[1],
        )
        self.convK = Depthwise_Separable_Conv(
            ndim,
            channels,
            dk * heads,
            kernel_size,
            padding="SAME",
            groups=heads,
            use_bias=False,
            padding_mode="CIRCULAR",
            dtype=dtype,
            key=keys[2],
        )
        self.convV = Depthwise_Separable_Conv(
            ndim,
            channels,
            dk * heads,
            kernel_size,
            padding="SAME",
            groups=heads,
            use_bias=False,
            padding_mode="CIRCULAR",
            dtype=dtype,
            key=keys[3],
        )

        Rshape = (dk, *lattice.shape[1:])
        self.RQ = jnp.zeros(Rshape, dtype)
        self.RK = jnp.zeros(Rshape, dtype)
        self.RV = jnp.zeros(Rshape, dtype)

        initializer = jax.nn.initializers.lecun_normal(
            in_axis=1, out_axis=0, dtype=dtype
        )
        self.W0 = initializer(keys[4], (channels, dk * heads), dtype)

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        residual = x
        x /= np.sqrt(self.nblock + 1, dtype=x.dtype)
        if self.nblock == 0:
            x /= np.sqrt(2, dtype=x.dtype)
        else:
            x = jax.nn.gelu(x)
        x = self.conv_embed(x)
        x += residual

        residual = x
        x /= np.sqrt(2, dtype=x.dtype)

        Q = self.convQ(x).reshape(self.heads, self.dk, -1)
        K = self.convK(x).reshape(self.heads, self.dk, -1)
        dot = jnp.einsum("hdi,hdj->hij", Q, K)
        irpeK = jnp.einsum("hdi,dij->hij", Q, _get_irpe(self.RK))
        irpeQ = jnp.einsum("hdj,dij->hij", K, _get_irpe(self.RQ))
        alpha = jax.nn.softmax((dot + irpeK + irpeQ) / jnp.sqrt(self.dk))

        V = self.convV(x).reshape(self.heads, self.dk, -1)
        attention = jnp.einsum("hij,hdj->hdi", alpha, V)
        irpeV = jnp.einsum("hij,dij->hdi", alpha, _get_irpe(self.RV))
        attention = attention + irpeV

        attention = attention.reshape(-1, attention.shape[2])
        attention = self.W0 @ attention
        return attention.reshape(residual.shape) + residual


def CvT(
    n_layers: int,
    channels: int,
    kernel_size: Union[int, Sequence[int]],
    dk: int,
    heads: int,
    use_sinh: bool = False,
    trans_symm: Optional[Symmetry] = None,
):
    blocks = [CvT_Block(channels, kernel_size, dk, heads, i) for i in range(n_layers)]
    scale = Scale(1 / np.sqrt(2 * n_layers))
    layers = [ReshapeConv(), *blocks, scale]
    layers.append(SinhShift())
    if trans_symm is not Identity():
        layers.append(ConvSymmetrize(trans_symm))

    return Sequential(layers, holomorphic=False)
