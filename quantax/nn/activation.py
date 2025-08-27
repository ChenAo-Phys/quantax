from __future__ import annotations
from typing import Callable, Optional
from jaxtyping import Key
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from .modules import NoGradLayer
from ..global_defs import get_default_dtype, get_sites


class Scale(NoGradLayer):
    r"""Rescale the input :math:`f(x) = x * \mathrm{scale}`"""

    scale: jax.Array

    def __init__(self, scale: float):
        super().__init__()
        self.scale = jnp.asarray(scale)

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        return x * self.scale.astype(x.dtype)


class Theta0Layer(NoGradLayer):
    r"""
    The activation layer with output :math:`f(x) = g(x) * \exp(\theta_0)`.
    One can tune :math:`\theta_0` to adjust the norm of the output state and avoid
    possible overflow.
    """

    theta0: jax.Array

    def __init__(self):
        super().__init__()
        self.theta0 = jnp.array(0, get_default_dtype())

    def rescale(self, maximum: jax.Array) -> Theta0Layer:
        r"""
        Rescale the function output by adjusting :math:`\theta_0`.

        :param maximum:
            The maximum output m obtained from this activation function.
            :math:`\theta_0` is adjusted as :math:`\theta'_0 = \theta_0 - \log(m)`
            so that the maximum output is rescaled to 1.

        :return:
            The layer with adjusted :math:`\theta_0`.

        .. note::

            This method generates a new layer while doesn't modify the existing layer.
        """
        theta0 = self.theta0 - jnp.log(maximum)
        return eqx.tree_at(lambda tree: tree.theta0, self, theta0)


class SinhShift(Theta0Layer):
    r"""
    :math:`f(x) = (\sinh(x) + 1) \exp(\theta_0)`

    .. note::

        No matter which input data type is provided, the output data type is
        always given by `quantax.get_default_dtype`.
    """

    theta0: jax.Array

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        x = x.astype(get_default_dtype())
        sinhx = (jnp.exp(x + self.theta0) - jnp.exp(-x + self.theta0)) / 2
        return sinhx + jnp.exp(self.theta0)


class Prod(Theta0Layer):
    r"""
    :math:`f(x) = \exp(\theta_0) \prod x`

    .. note::

        No matter which input data type is provided, the output data type is
        always given by `quantax.get_default_dtype`.
    """

    theta0: jax.Array

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        x = x.astype(get_default_dtype())
        x *= jnp.exp(self.theta0 / x.size)
        x = jnp.prod(x, axis=0)
        x = jnp.prod(x)
        return x


class Exp(Theta0Layer):
    r"""
    :math:`f(x) = \exp(x + \theta_0)`

    .. note::

        No matter which input data type is provided, the output data type is
        always given by `quantax.get_default_dtype`.
    """

    theta0: jax.Array

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        return jnp.exp(x.astype(get_default_dtype()) + self.theta0)


@jax.jit
def crelu(x: jax.Array) -> jax.Array:
    r"""
    Complex relu activation function :math:`f(x) = \mathrm{ReLU(Re}x)` + i \mathrm{ReLU(Im}x)`.
    See `Deep Complex Networks <https://arxiv.org/abs/1705.09792>`_ for details
    """
    return jax.nn.relu(x.real) + 1j * jax.nn.relu(x.imag)


@jax.jit
def cardioid(x: jax.Array) -> jax.Array:
    r"""
    f(z) = (1 + cos\phi) z / 2

    P. Virtue, S. X. Yu and M. Lustig, "Better than real: Complex-valued neural nets for
    MRI fingerprinting," 2017 IEEE International Conference on Image Processing (ICIP),
    Beijing, China, 2017, pp. 3953-3957, doi: 10.1109/ICIP.2017.8297024.
    """
    return 0.5 * (1 + jnp.cos(jnp.angle(x))) * x


@jax.jit
def pair_cpl(x: jax.Array) -> jax.Array:
    """
    Make a real input complex by splitting it into two parts, one taken as the real part
    and the other the imaginary part.
    ``output = x[: x.shape[0] // 2] + 1j * x[x.shape[0] // 2 :]``.

    Originally proposed in `PRB 108, 054410 <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.108.054410>`_.
    """
    return jax.lax.complex(x[: x.shape[0] // 2], x[x.shape[0] // 2 :])
