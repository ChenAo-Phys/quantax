from typing import Callable
import numpy as np
import jax.numpy as jnp
from equinox.nn import Linear, Conv
from ..nn import Sequential, apply_lecun_normal, ScaleFn, Prod, ReshapeConv
from ..global_defs import (
    get_sites,
    get_params_dtype,
    get_lattice,
    is_params_cpl,
    get_subkeys,
)


def SingleDense(
    features: int, actfn: Callable, use_bias: bool = True, holomorphic: bool = False
):
    nsites = get_sites().nsites
    key = get_subkeys()
    linear = Linear(nsites, features, use_bias, get_params_dtype(), key=key)
    linear = apply_lecun_normal(key, linear)
    layers = [linear, ScaleFn(actfn, features), Prod()]
    return Sequential(layers, holomorphic)


def RBM_Dense(features: int, use_bias: bool = True):
    return SingleDense(features, jnp.cosh, use_bias, holomorphic=is_params_cpl())


def SingleConv(
    channels: int, actfn: Callable, use_bias: bool = True, holomorphic: bool = False
):
    lattice = get_lattice()
    key = get_subkeys()
    conv = Conv(
        num_spatial_dims=lattice.dim,
        in_channels=lattice.shape[-1],
        out_channels=channels,
        kernel_size=lattice.shape[:-1],
        padding="SAME",
        use_bias=use_bias,
        padding_mode="CIRCULAR",
        dtype=get_params_dtype(),
        key=key,
    )
    conv = apply_lecun_normal(key, conv)
    out_features = channels * np.prod(lattice.shape[:-1])
    layers = [ReshapeConv(), conv, ScaleFn(actfn, out_features), Prod()]
    return Sequential(layers, holomorphic)


def RBM_Conv(channels: int, use_bias: bool = True):
    return SingleConv(channels, jnp.cosh, use_bias, holomorphic=is_params_cpl())
