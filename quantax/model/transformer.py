from typing import Callable, Optional, Tuple, Sequence
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike
import equinox as eqx
from ..global_defs import get_sites, get_subkeys
from ..nn import (
    Embedding,
    lecun_normal,
    he_normal,
    Sequential,
    pair_cpl,
    exp_by_scale,
)
from ..utils import PsiArray


class MHSA(eqx.Module):
    norm: eqx.nn.RMSNorm
    WQ: jax.Array
    WK: jax.Array
    WV: jax.Array
    W0: jax.Array

    def __init__(self, heads: int, d: int, dtype: DTypeLike = jnp.float32):
        dH = d // heads
        lecun_init = jax.nn.initializers.lecun_normal(
            in_axis=1, out_axis=2, batch_axis=0, dtype=dtype
        )
        keyQ, keyK, keyV, key0 = get_subkeys(4)
        self.WQ = lecun_init(keyQ, (heads, d, dH))
        self.WK = lecun_init(keyK, (heads, d, dH))
        self.WV = lecun_init(keyV, (heads, d, dH))
        self.W0 = lecun_normal(key0, (d, d), dtype)
        N = get_sites().Nsites
        self.norm = eqx.nn.RMSNorm((d, N), use_weight=False, use_bias=False)

    def __call__(self, x: jax.Array) -> jax.Array:
        N = get_sites().Nsites
        x = x.reshape(-1, N)
        residual = x

        x = self.norm(x)
        Q = jnp.einsum("hcd,ci->hdi", self.WQ, x)
        K = jnp.einsum("hcd,ci->hdi", self.WK, x)
        dot = jnp.einsum("hdi,hdj->hij", Q, K)
        alpha = jax.nn.softmax(dot / jnp.sqrt(self.WK.shape[-1]))

        V = jnp.einsum("hcd,ci->hdi", self.WV, x)
        attention = jnp.einsum("hij,hdj->hdi", alpha, V)
        attention = attention.reshape(-1, N)
        attention = self.W0 @ attention
        return attention + residual


class FFN(eqx.Module):
    norm: eqx.nn.RMSNorm
    W: jax.Array
    b: jax.Array

    def __init__(self, d: int, dtype: DTypeLike = jnp.float32):
        N = get_sites().Nsites
        self.norm = eqx.nn.RMSNorm((d, N), use_weight=False, use_bias=False)
        self.W = he_normal(get_subkeys(), (d, d), dtype)
        self.b = jnp.zeros((d, 1), dtype=dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        N = get_sites().Nsites
        x = x.reshape(-1, N)
        residual = x
        x = self.norm(x)
        x = self.W @ x + self.b
        x = jax.nn.silu(x)
        return x + residual


class Transformer(Sequential):
    nblocks: int
    d: int
    heads: int
    final_activation: Callable[[jax.Array], PsiArray]
    final_sum: bool
    dtype: DTypeLike
    out_dtype: DTypeLike
    layers: Tuple[Callable, ...]
    holomorphic: bool

    def __init__(
        self,
        nblocks: int,
        d: int,
        heads: int = 4,
        sublattice: Optional[Sequence[int]] = None,
        final_activation: Optional[Callable[[jax.Array], PsiArray]] = None,
        final_sum: bool = True,
        dtype: DTypeLike = jnp.float32,
        out_dtype: Optional[DTypeLike] = None,
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

        layers = [Embedding(d, sublattice, dtype)]
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
