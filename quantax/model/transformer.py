from typing import Callable, Optional, Tuple
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from ..global_defs import get_sites, get_subkeys
from ..nn import (
    lecun_normal,
    he_normal,
    glorot_normal,
    Sequential,
    pair_cpl,
    exp_by_scale,
)
from ..utils import PsiArray


class Embedding(eqx.Module):
    """Embedding layer."""

    E: jax.Array
    P: jax.Array

    def __init__(self, d: int, dtype=jnp.float32):
        """
        Initialize the embedding layer
        
        :param d: Dimension of the embedding
        :param dtype: Data type of the embedding
        """

        self.E = jr.normal(get_subkeys(), (4, d), dtype=dtype)
        self.P = jr.normal(get_subkeys(), (get_sites().Nsites, d), dtype=dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.reshape(-1, get_sites().Nsites)
        x = (2 * (x[0] < 0) + (x[1] < 0)).astype(jnp.uint8)
        out = self.E[x] + self.P
        return out.T


class MHSA(eqx.Module):
    layer_norm: eqx.nn.LayerNorm
    WQ: jax.Array
    WK: jax.Array
    WV: jax.Array
    W0: jax.Array

    def __init__(self, heads: int, d: int, dtype=jnp.float32):
        dH = d // heads
        lecun_init = jax.nn.initializers.lecun_normal(
            in_axis=1, out_axis=2, batch_axis=0, dtype=dtype
        )
        self.WQ = lecun_init(get_subkeys(), (heads, d, dH))
        self.WK = lecun_init(get_subkeys(), (heads, d, dH))
        self.WV = lecun_init(get_subkeys(), (heads, d, dH))
        self.W0 = lecun_normal(get_subkeys(), (d, d), dtype)
        N = get_sites().Nsites
        self.layer_norm = eqx.nn.LayerNorm((d, N), use_weight=False, use_bias=False)

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x

        x = self.layer_norm(x)
        Q = jnp.einsum("hcd,ci->hdi", self.WQ, x)
        K = jnp.einsum("hcd,ci->hdi", self.WK, x)
        dot = jnp.einsum("hdi,hdj->hij", Q, K)
        alpha = jax.nn.softmax(dot / jnp.sqrt(self.WK.shape[-1]))

        V = jnp.einsum("hcd,ci->hdi", self.WV, x)
        attention = jnp.einsum("hij,hdj->hdi", alpha, V)
        N = attention.shape[2]
        attention = attention.reshape(-1, N)
        attention = self.W0 @ attention
        return attention + residual


class FFN(eqx.Module):
    layer_norm: eqx.nn.LayerNorm
    W1: jax.Array
    b1: jax.Array
    W2: jax.Array
    b2: jax.Array

    def __init__(self, d: int, dtype=jnp.float32):
        N = get_sites().Nsites
        self.layer_norm = eqx.nn.LayerNorm((d, N), use_weight=False, use_bias=False)

        self.W1 = he_normal(get_subkeys(), (4 * d, d), dtype)
        self.b1 = jnp.zeros((4 * d, 1), dtype=dtype)

        self.W2 = glorot_normal(get_subkeys(), (d, 4 * d), dtype)
        self.b2 = jnp.zeros((d, 1), dtype=dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        N = get_sites().Nsites
        x = x.reshape(-1, N)
        residual = x
        x = self.layer_norm(x)

        x = self.W1 @ x + self.b1
        x = jax.nn.silu(x)
        x = self.W2 @ x + self.b2
        return x + residual


class Transformer(Sequential):
    nblocks: int
    d: int
    heads: int
    final_activation: Callable[[jax.Array], PsiArray]
    final_sum: bool
    dtype: jnp.dtype
    out_dtype: jnp.dtype
    layers: Tuple[Callable, ...]
    holomorphic: bool

    def __init__(
        self,
        nblocks: int,
        d: int,
        heads: int = 4,
        final_activation: Optional[Callable[[jax.Array], PsiArray]] = None,
        final_sum: bool = True,
        dtype: jnp.dtype = jnp.float32,
        out_dtype: Optional[jnp.dtype] = None,
    ):
        self.nblocks = nblocks
        self.d = d
        self.heads = heads
        if final_activation is None:
            final_activation = exp_by_scale
        self.final_activation = final_activation
        self.final_sum = final_sum
        self.dtype = dtype
        if out_dtype is None:
            out_dtype = dtype
        self.out_dtype = out_dtype

        layers = [Embedding(d, dtype)]
        for l in range(nblocks):
            layers.append(MHSA(heads, d, dtype))
            layers.append(FFN(d, dtype))

        def final_layer(x):
            x /= jnp.sqrt(nblocks + 1)
            if jnp.issubdtype(out_dtype, jnp.complexfloating):
                x = pair_cpl(x)
            x = x.astype(out_dtype)
            x = final_activation(x)
            if final_sum:
                x = x.sum()
            return x

        layers = [*layers, final_layer]
        super().__init__(layers)
