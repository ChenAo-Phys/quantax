from __future__ import annotations
import jax
import jax.numpy as jnp
from ..utils import LogArray, ScaleArray
    

def sinhp1_by_scale(x: jax.Array) -> ScaleArray:
    r"""
    :math:`f(x) = \sinh(x) + 1`. Output is represented by `~quantax.utils.ScaleArray` 
    to avoid overflow.
    """
    xmax = jax.lax.stop_gradient(jnp.nanmax(jnp.abs(x)))
    significand = (jnp.exp(x - xmax) - jnp.exp(-x - xmax)) / 2 + jnp.exp(-xmax)
    return ScaleArray(significand, xmax)


def prod_by_log(x: jax.Array) -> ScaleArray:
    r"""
    :math:`f(x) = \prod x`. Output is represented by `~quantax.utils.LogArray` to 
    avoid overflow.
    """
    x = LogArray.from_value(x)
    return x.prod()
    

def exp_by_scale(x: jax.Array) -> ScaleArray:
    r"""
    :math:`f(x) = \exp(x)`. Output is represented by `~quantax.utils.ScaleArray` to 
    avoid overflow.
    """
    xmax = jax.lax.stop_gradient(jnp.nanmax(abs(x)))
    return ScaleArray(jnp.exp(x - xmax), xmax)


def exp_by_log(x: jax.Array) -> LogArray:
    r"""
    :math:`f(x) = \exp(x)`. Output is represented by `~quantax.utils.LogArray` to 
    avoid overflow.
    """
    if jnp.isrealobj(x):
        sign = jnp.ones_like(x)
    else:
        sign = jnp.exp(1j * x.imag)
    return LogArray(sign, x.real)


def crelu(x: jax.Array) -> jax.Array:
    r"""
    Complex relu activation function :math:`f(x) = \mathrm{ReLU(Re}x)` + i \mathrm{ReLU(Im}x)`.
    See `Deep Complex Networks <https://arxiv.org/abs/1705.09792>`_ for details
    """
    return jax.nn.relu(x.real) + 1j * jax.nn.relu(x.imag)


def cardioid(x: jax.Array) -> jax.Array:
    r"""
    f(z) = (1 + cos\phi) z / 2

    P. Virtue, S. X. Yu and M. Lustig, "Better than real: Complex-valued neural nets for
    MRI fingerprinting," 2017 IEEE International Conference on Image Processing (ICIP),
    Beijing, China, 2017, pp. 3953-3957, doi: 10.1109/ICIP.2017.8297024.
    """
    return 0.5 * (1 + jnp.cos(jnp.angle(x))) * x


def pair_cpl(x: jax.Array) -> jax.Array:
    r"""
    :math:`f(x) = x_1 + i x_2` , where :math:`x = (x_1, x_2)`.

    Originally proposed in `PRB 108, 054410 <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.108.054410>`_.
    """
    return jax.lax.complex(x[: x.shape[0] // 2], x[x.shape[0] // 2 :])
