from __future__ import annotations
from typing import Callable, Optional
from numpy.typing import ArrayLike
from jaxtyping import Key
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from .modules import NoGradLayer
from ..global_defs import (
    get_default_dtype,
    get_params_dtype,
    get_sites,
)


class Scale(NoGradLayer):
    scale: jax.Array

    def __init__(self, scale: ArrayLike):
        super().__init__()
        self.scale = jnp.asarray(scale)

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        return x * self.scale


class ScaleFn(NoGradLayer):
    fn: Callable
    scale: jax.Array

    def __init__(self, fn: Callable, features: int, scaling: ArrayLike = 1.0):
        super().__init__()
        self.fn = fn

        dtype = get_params_dtype()
        std0 = 0.1
        x = jr.normal(jr.key(0), (1000, features), dtype=dtype)

        def output_std_eq(scale):
            out = jnp.sum(jnp.log(jnp.abs(fn(x * scale))), axis=1)
            # target_std 0.1, 0.3, or pi/(2/sqrt3) (0.9)
            target_std = std0 * np.sqrt(get_sites().nsites)
            return (jnp.std(out) - target_std) ** 2

        test_arr = jnp.arange(0, 1, 0.01)
        out = jax.vmap(output_std_eq)(test_arr)
        arg = jnp.nanargmin(out)
        self.scale = scaling * test_arr[arg]

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        return self.fn(x * self.scale)


class Theta0Layer(NoGradLayer):
    """
    The activation layer with output f(x) * exp(θ₀), used to adjust the global 
    amplitude to avoid divergence.
    """
    theta0: jax.Array

    def __init__(self):
        super().__init__()
        self.theta0 = jnp.array(0, get_default_dtype())

    def rescale(self, maximum: jax.Array) -> Theta0Layer:
        theta0 = self.theta0 - jnp.log(maximum)
        return eqx.tree_at(lambda tree: tree.theta0, self, theta0)


class SinhShift(Theta0Layer):
    theta0: jax.Array

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        x = x.astype(get_default_dtype())
        sinhx = (jnp.exp(x + self.theta0) - jnp.exp(-x + self.theta0)) / 2
        return sinhx + jnp.exp(self.theta0)


class Prod(Theta0Layer):
    theta0: jax.Array

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        x = x.astype(get_default_dtype())
        x *= jnp.exp(self.theta0 / x.size)
        x = jnp.prod(x, axis=0)
        x = jnp.prod(x)
        return x


class ExpSum(Theta0Layer):
    theta0: jax.Array

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        x = x.astype(get_default_dtype())
        x = jnp.sum(x) + self.theta0
        return jnp.exp(x)


class Exp(Theta0Layer):
    theta0: jax.Array

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        return jnp.exp(x.astype(get_default_dtype()) + self.theta0)


@jax.jit
def crelu(x: jax.Array) -> jax.Array:
    """
    Deep Complex Networks https://arxiv.org/abs/1705.09792
    """
    return jax.nn.relu(x.real) + 1j * jax.nn.relu(x.imag)


@jax.jit
def cardioid(x: jax.Array) -> jax.Array:
    """
    f(z) = (1 + cos\phi) z / 2

    P. Virtue, S. X. Yu and M. Lustig, "Better than real: Complex-valued neural nets for
    MRI fingerprinting," 2017 IEEE International Conference on Image Processing (ICIP),
    Beijing, China, 2017, pp. 3953-3957, doi: 10.1109/ICIP.2017.8297024.
    """
    return 0.5 * (1 + jnp.cos(jnp.angle(x))) * x


@jax.jit
def pair_cpl(x: jax.Array) -> jax.Array:
    return x[: x.shape[0] // 2] + 1j * x[x.shape[0] // 2 :]
