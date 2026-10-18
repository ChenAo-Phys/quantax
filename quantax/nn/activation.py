from __future__ import annotations
import jax
import jax.numpy as jnp
from ..utils import LogArray


def sinhp1_by_log(x: jax.Array) -> LogArray:
    r"""
    :math:`f(x) = \sinh(x) + 1`. Output is represented by `~quantax.utils.LogArray`
    to avoid overflow. Only real inputs are supported.
    """
    if not jnp.isrealobj(x):
        raise TypeError(f"`sinhp1_by_log` only supports real inputs, got {x.dtype}.")
    # sinh(x) + 1 = sig * exp(m) with m = |x|, so that |sig| <= 1
    m = jax.lax.stop_gradient(jnp.abs(x))
    sig = (jnp.exp(x - m) - jnp.exp(-x - m)) / 2 + jnp.exp(-m)
    # sig can be exactly 0 at a root of sinh(x) + 1, where d log|sig| = dsig / sig is
    # infinite and turns the gradient into NaN. A tiny shift eps^2 keeps it finite,
    # and the gradient stays exact because sums weight d log|sig| by sig.
    tiny = float(jnp.finfo(sig.dtype).eps) ** 2
    y = LogArray.from_value(sig + jnp.where(sig == 0, tiny, 0.0))
    return LogArray(y.sign, y.logabs + m)


def prod_by_log(x: jax.Array) -> LogArray:
    r"""
    :math:`f(x) = \prod x`. Output is represented by `~quantax.utils.LogArray` to
    avoid overflow.
    """
    y = LogArray.from_value(x)
    return y.prod()


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
    Complex relu activation function
    :math:`f(x) = \mathrm{ReLU}(\mathrm{Re}\,x) + i\,\mathrm{ReLU}(\mathrm{Im}\,x)`.
    See `Deep Complex Networks <https://arxiv.org/abs/1705.09792>`_ for details
    """
    return jax.nn.relu(x.real) + 1j * jax.nn.relu(x.imag)


def cardioid(x: jax.Array) -> jax.Array:
    r"""
    :math:`f(z) = \frac{1}{2}(1 + \cos\phi)\,z`, where :math:`\phi = \arg z`.

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
