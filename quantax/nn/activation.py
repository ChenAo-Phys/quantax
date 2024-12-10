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


class ScaleFn(NoGradLayer):
    r"""
    Apply a function to a rescaled input :math:`f(x) = fn(x * \mathrm{scale})`.
    The scale is automatically computed from the function to ensure that
    :math:`\sigma (\sum \log |f(x)|) = 0.1 \sqrt{N}`
    when :math:`\sigma(x) = 1` and the system has N sites.

    This is particularly helpful for the stability of networks when
    :math:`\psi = \prod f(x)`, for instance the RBM.

    .. note::

        No matter which input data type is provided, the output data type is
        always given by `quantax.get_default_dtype`.
    """

    fn: Callable
    scale: jax.Array

    def __init__(
        self,
        fn: Callable,
        features: int,
        scaling: float = 1.0,
        dtype: jnp.dtype = jnp.float32,
    ):
        r"""
        :param fn:
            The activation function to be applied.

        :param features:
            The size of input x, not considering the batch dimension.

        :param scaling:
            Additional scaling factor to apply on the input to rescale the inputs to
            :math:`\sigma(x) = 1`.

        :param dtype:
            The data type of inputs.
        """
        super().__init__()
        self.fn = fn

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
        self.scale = jnp.asarray(scaling * test_arr[arg], dtype=dtype)

    def __call__(self, x: jax.Array, *, key: Optional[Key] = None) -> jax.Array:
        return self.fn(x * self.scale)


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
    """
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
