from typing import Callable, Optional, Tuple, Union
from jaxtyping import Key
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from ..nn import (
    Sequential,
    RefModel,
    apply_he_normal,
    apply_lecun_normal,
    Scale,
    Prod,
    ReshapeConv,
)
from ..global_defs import PARTICLE_TYPE, get_sites, get_lattice, get_subkeys


def _get_scale(
    fn: Callable, features: int, dtype: jnp.dtype = jnp.float32
) -> jax.Array:
    std0 = 0.1
    x = jr.normal(jr.key(0), (1000, features), dtype=dtype)

    def output_std_eq(scale):
        out = jnp.sum(jnp.log(jnp.abs(fn(x * scale))), axis=1)
        # target_std 0.1, 0.3, or pi/(2/sqrt3) (0.9)
        target_std = std0 * np.sqrt(get_sites().N)
        return (jnp.std(out) - target_std) ** 2

    test_arr = jnp.arange(0, 1, 0.01)
    out = jax.vmap(output_std_eq)(test_arr)
    arg = jnp.nanargmin(out)
    return jnp.asarray(test_arr[arg], dtype=dtype)


class SingleDense(Sequential, RefModel):
    layers: Tuple[eqx.Module]
    holomorphic: bool

    def __init__(
        self,
        features: int,
        actfn: Callable,
        use_bias: bool = True,
        holomorphic: bool = False,
        dtype: jnp.dtype = jnp.float32,
    ):
        N = get_sites().nstates
        key = get_subkeys()
        linear = eqx.nn.Linear(N, features, use_bias, dtype, key=key)
        linear = apply_lecun_normal(key, linear)
        scale = _get_scale(actfn, features, dtype)
        linear = eqx.tree_at(lambda tree: tree.weight, linear, linear.weight * scale)

        layers = [linear, eqx.nn.Lambda(lambda x: actfn(x)), Prod()]
        Sequential.__init__(self, layers, holomorphic)
        RefModel.__init__(self)

    @eqx.filter_jit
    def init_internal(self, x: jax.Array) -> jax.Array:
        return self.layers[0](x)

    def ref_forward(
        self,
        s: jax.Array,
        s_old: jax.Array,
        nflips: int,
        internal: jax.Array,
        return_update: bool = False,
    ) -> Union[jax.Array, Tuple[jax.Array, jax.Array]]:
        """
        Accelerated forward pass through local updates and internal quantities.
        This function is designed for sampling.

        :return:
            The evaluated wave function and the updated internal values.
        """
        idx_flips = jnp.argwhere(s != s_old, size=nflips).flatten()
        weight = self.layers[0].weight
        internal += 2 * weight[:, idx_flips] @ s[idx_flips]
        psi = self.layers[2](self.layers[1](internal))
        if return_update:
            return psi, internal
        else:
            return psi


def RBM_Dense(features: int, use_bias: bool = True, dtype: jnp.dtype = jnp.float32):
    r"""
    The restricted Boltzmann machine with one dense layer
    :math:`\psi(s) = \prod \cosh(W s + b)`.

    :param features:
        The number of output features or hidden units.

    :param use_bias:
        Whether to add on a bias.

    :param dtype:
        The data type of the parameters.
    """
    holomorphic = np.issubdtype(dtype, np.complexfloating)
    return SingleDense(features, jnp.cosh, use_bias, holomorphic, dtype)


def SingleConv(
    channels: int,
    actfn: Callable,
    use_bias: bool = True,
    holomorphic: bool = False,
    dtype: jnp.dtype = jnp.float32,
):
    r"""
    Network with one convolutional layer
    :math:`\psi(s) = \prod f(\mathrm{Conv}(s))`.

    :param features:
        The number of channels in the convolutional network.

    :param actfn:
        The activation function applied after the convolutional layer.

    :param use_bias:
        Whether to add on a bias in the convolution.

    :param holomorphic:
        Whether the whole network is complex holomorphic.

    :param dtype:
        The data type of the parameters.
    """
    lattice = get_lattice()
    key = get_subkeys()
    conv = eqx.nn.Conv(
        num_spatial_dims=lattice.ndim,
        in_channels=lattice.shape[0],
        out_channels=channels,
        kernel_size=lattice.shape[1:],
        padding="SAME",
        use_bias=use_bias,
        padding_mode="CIRCULAR",
        dtype=dtype,
        key=key,
    )
    conv = apply_lecun_normal(key, conv)
    scale = _get_scale(actfn, channels * lattice.ncells, dtype)
    conv = eqx.tree_at(lambda tree: tree.weight, conv, conv.weight * scale)
    layers = [ReshapeConv(dtype), conv, eqx.nn.Lambda(lambda x: actfn(x)), Prod()]
    return Sequential(layers, holomorphic)


def RBM_Conv(channels: int, use_bias: bool = True, dtype: jnp.dtype = jnp.float32):
    r"""
    The restricted Boltzmann machine with one convolutional layer
    :math:`\psi(s) = \prod \cosh(\mathrm{Conv}(s))`.

    :param features:
        The number of channels in the convolutional network.

    :param use_bias:
        Whether to add on a bias in the convolution.

    :param dtype:
        The data type of the parameters.
    """
    holomorphic = np.issubdtype(dtype, np.complexfloating)
    return SingleConv(channels, jnp.cosh, use_bias, holomorphic, dtype)


class _ResBlock(eqx.Module):
    """Residual block"""

    conv1: eqx.nn.Conv
    conv2: eqx.nn.Conv
    nblock: int = eqx.field(static=True)

    def __init__(self, channels: int, kernel_size: int, nblock: int, dtype: jnp.dtype):
        lattice = get_lattice()

        def new_layer(is_first_layer: bool) -> eqx.nn.Conv:
            if is_first_layer:
                in_channels = lattice.shape[0]
                if lattice.particle_type == PARTICLE_TYPE.spinful_fermion:
                    in_channels *= 2
            else:
                in_channels = channels
            key = get_subkeys()
            conv = eqx.nn.Conv(
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
    scale = _get_scale(final_actfn, out_features, dtype) / jnp.sqrt(nblocks + 1)
    actfn = eqx.nn.Lambda(lambda x: final_actfn(x * scale))
    return Sequential(
        [ReshapeConv(dtype), *blocks, actfn, Prod()], holomorphic=False
    )
