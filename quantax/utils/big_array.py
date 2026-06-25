r"""
Numerically stable array representations for neural quantum states.

Wavefunction amplitudes in many-body systems easily over- or underflow the
floating-point range, so this module provides two PyTree array types that keep
the magnitude in log / exponent space:

- :class:`LogArray` stores :math:`\text{value} = \text{sign} \cdot \exp(\text{logabs})`,
  where ``sign`` is a unit :math:`\pm 1` (or complex phase) and ``logabs`` is the
  real log-magnitude. Zero is encoded as ``sign=0, logabs=-inf``. This is best
  for products and powers, where the log-magnitudes simply add.
- :class:`ScaleArray` stores :math:`\text{value} = \text{significand} \cdot \exp(\text{exponent})`,
  where ``exponent`` is a real normalization factor shared across the magnitude.
  This is best for sums, where a common scale can be factored out (log-sum-exp).

Both types support the common arithmetic, reduction, and reshaping operations
while keeping the computation in the stable representation, and convert to a
dense JAX array via ``arr.value()`` or ``jnp.asarray(arr)``. See the warnings on
each class for caveats about JAX's partial support for custom arrays.

:data:`PsiArray` is the union of these two types with plain numpy / JAX arrays,
used throughout Quantax as the wavefunction-amplitude type.
"""

from __future__ import annotations
from collections.abc import Callable
from typing import ClassVar
from dataclasses import dataclass
from numpy.typing import ArrayLike, NDArray
from jax import Array
from jax.typing import DTypeLike
import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@jax.custom_jvp
def _phase(x: Array) -> Array:
    r"""
    Unit phase :math:`x / |x|` (with ``0`` mapped to ``0``). The value matches
    :func:`jax.numpy.sign` for both real and complex ``x``, but the gradient is
    correct for complex ``x``. ``jnp.sign`` has a *zero* JVP for complex inputs,
    which silently discards the phase gradient of complex wavefunctions; this
    helper restores it while keeping the zero (and finite) gradient on the real
    axis and at the origin.
    """
    absx = jnp.abs(x)
    return jnp.where(absx == 0, jnp.zeros_like(x), x / absx)


@_phase.defjvp
def _phase_jvp(primals: tuple[Array], tangents: tuple[Array]) -> tuple[Array, Array]:
    (x,), (dx,) = primals, tangents
    zeros = jnp.zeros_like(x)
    absx = jnp.abs(x)
    p = jnp.where(absx == 0, zeros, x / absx)
    if not jnp.iscomplexobj(x):
        return p, zeros

    # d(x/|x|) = dx/|x| - x Re(conj(x) dx) / |x|^3, which is 0 on the real axis.
    safe = jnp.where(absx == 0, 1, absx)
    d = dx / safe - x * jnp.real(jnp.conj(x) * dx) / safe**3
    dp = jnp.where(absx == 0, zeros, d)
    return p, dp


@jax.custom_jvp
def _addexp(x1: Array, x2: Array, b1: Array, b2: Array) -> tuple[Array, Array]:
    """
    Compute b1 * exp(x1) + b2 * exp(x2) and return two values (x, b) to represent the
    result b * exp(x).
    """
    xmax = jnp.maximum(x1, x2)
    r1 = jnp.where(x1 != xmax, jnp.exp(x1 - xmax), 1.0)
    r2 = jnp.where(x2 != xmax, jnp.exp(x2 - xmax), 1.0)
    b = b1 * r1 + b2 * r2
    return xmax, b


@_addexp.defjvp
def _addexp_jvp(
    primals: tuple[Array, Array, Array, Array],
    tangents: tuple[Array, Array, Array, Array],
) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
    x1, x2, b1, b2 = primals
    dx1, dx2, db1, db2 = tangents

    xmax = jnp.maximum(x1, x2)
    r1 = jnp.where(x1 != xmax, jnp.exp(x1 - xmax), 1.0)
    r2 = jnp.where(x2 != xmax, jnp.exp(x2 - xmax), 1.0)
    b = b1 * r1 + b2 * r2
    dx = jnp.zeros_like(xmax)
    db = db1 * r1 + db2 * r2 + dx1 * b1 * r1 + dx2 * b2 * r2
    return (xmax, b), (dx, db)


@jax.custom_jvp
def _sumexp(x: Array, b: Array) -> tuple[Array, Array]:
    r"""
    Compute :math:`\sum_i b_i \exp(x_i)` and return two values (x, b) to represent the
    result b * exp(x).
    """
    xmax = jnp.max(x)
    r = jnp.where(x != xmax, jnp.exp(x - xmax), 1.0)
    b = jnp.sum(b * r)
    return xmax, b


@_sumexp.defjvp
def _sumexp_jvp(
    primals: tuple[Array, Array], tangents: tuple[Array, Array]
) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
    xi, bi = primals
    dxi, dbi = tangents
    xmax = jnp.max(xi)
    ri = jnp.where(xi != xmax, jnp.exp(xi - xmax), 1.0)
    b = jnp.sum(bi * ri)
    dx = jnp.zeros_like(xmax)
    db = jnp.sum(ri * (bi * dxi + dbi))
    return (xmax, b), (dx, db)


def _get_reduction_size(
    shape: tuple[int, ...], axis: int | tuple[int, ...] | None
) -> int:
    if axis is None:
        axis = tuple(range(len(shape)))
    elif isinstance(axis, int):
        axis = (axis,)

    size = 1
    for ax in axis:
        size *= shape[ax]
    return size


def sumexp(
    x: Array,
    b: Array,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> tuple[Array, Array]:
    r"""
    Compute :math:`\sum_i b_i \exp(x_i)` over ``axis`` and return two values
    (x, b) to represent the result :math:`b \exp(x)`.

    :param x: The exponents :math:`x_i`.
    :param b: The coefficients :math:`b_i`.
    :param axis: Axis or axes to reduce over. ``None`` (default) reduces over all axes.
    :param keepdims: If ``True``, the reduced axes are kept with size 1.
    """
    if axis is None:
        axis = tuple(range(len(x.shape)))
    elif isinstance(axis, int):
        axis = (axis,)

    new_axis = tuple(range(len(axis)))
    x = jnp.moveaxis(x, axis, new_axis)
    b = jnp.moveaxis(b, axis, new_axis)
    reduction_size = _get_reduction_size(x.shape, new_axis)
    remaining_shape = x.shape[len(axis) :]
    x = x.reshape(reduction_size, -1)
    b = b.reshape(reduction_size, -1)
    x, b = jax.vmap(_sumexp, in_axes=1)(x, b)
    x = x.reshape(remaining_shape)
    b = b.reshape(remaining_shape)
    if keepdims:
        sorted_axes = sorted(axis)
        x = jnp.expand_dims(x, sorted_axes)
        b = jnp.expand_dims(b, sorted_axes)

    return x, b


def meanexp(
    x: Array,
    b: Array,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> tuple[Array, Array]:
    r"""
    Compute :math:`\left< b_i \exp(x_i) \right>` over ``axis`` and return two
    values (x, b) to represent the result :math:`b \exp(x)`.

    :param x: The exponents :math:`x_i`.
    :param b: The coefficients :math:`b_i`.
    :param axis: Axis or axes to reduce over. ``None`` (default) reduces over all axes.
    :param keepdims: If ``True``, the reduced axes are kept with size 1.
    """
    if axis is None:
        axis = tuple(range(len(x.shape)))
    elif isinstance(axis, int):
        axis = (axis,)

    new_axis = tuple(range(len(axis)))
    x = jnp.moveaxis(x, axis, new_axis)
    b = jnp.moveaxis(b, axis, new_axis)
    reduction_size = _get_reduction_size(x.shape, new_axis)
    remaining_shape = x.shape[len(axis) :]
    x = x.reshape(reduction_size, -1)
    b = b.reshape(reduction_size, -1)
    x, b = jax.vmap(_sumexp, in_axes=1)(x, b)
    x = x.reshape(remaining_shape)
    b = b.reshape(remaining_shape)
    x -= jnp.log(reduction_size)
    if keepdims:
        sorted_axes = sorted(axis)
        x = jnp.expand_dims(x, sorted_axes)
        b = jnp.expand_dims(b, sorted_axes)

    return x, b


@register_pytree_node_class
@dataclass
class LogArray:
    r"""
    Log-amplitude representation of JAX arrays: value = sign * exp(logabs)
    where `sign` is :math:`\pm 1` or a complex phase and `logabs` is real.
    Zero is encoded by sign=0, logabs=-inf.

    The array is a PyTree with two leaves: `sign` and `logabs`. To convert it to a
    dense JAX array, use `arr.value()` or `jnp.asarray(arr)`.

    .. warning::

        JAX doesn't have a full support for `customized arrays <https://docs.jax.dev/en/latest/jep/28661-jax-array-protocol.html>`_,
        so one should be careful when using ``LogArray``.
        Here we list several possible problems.

        1. Manipulations like ``jnp.fn(array)`` transform customized arrays to ``Array``.
        To avoid it, call ``array.fn()`` whenever possible.

        2. Computations like ``jax_array * customized_array`` always call
        ``jax_array.__mul__(customized_array)``, which returns a ``Array``.
        To avoid it, use ``customized_array * jax_array``.

    """

    sign: Array  # sign or phase
    logabs: Array  # real log-magnitude

    # Make Python/Numpy prefer our overloads when mixed types appear.
    __array_priority__: ClassVar[int] = 1000

    # Methods generated later
    __getitem__: ClassVar[Callable[..., LogArray]]
    choose: ClassVar[Callable[..., LogArray]]
    compress: ClassVar[Callable[..., LogArray]]
    copy: ClassVar[Callable[..., LogArray]]
    diagonal: ClassVar[Callable[..., LogArray]]
    flatten: ClassVar[Callable[..., LogArray]]
    ravel: ClassVar[Callable[..., LogArray]]
    repeat: ClassVar[Callable[..., LogArray]]
    reshape: ClassVar[Callable[..., LogArray]]
    squeeze: ClassVar[Callable[..., LogArray]]
    swapaxes: ClassVar[Callable[..., LogArray]]
    take: ClassVar[Callable[..., LogArray]]
    transpose: ClassVar[Callable[..., LogArray]]

    @staticmethod
    def from_value(x: ArrayLike) -> LogArray:
        """Create from a JAX array / Python scalar."""
        if isinstance(x, LogArray):
            return x

        if isinstance(x, ScaleArray):
            sign = _phase(x.significand)
            logabs = jnp.log(jnp.abs(x.significand)) + x.exponent
            return LogArray(sign, logabs)

        x = jnp.asarray(x)
        sign = _phase(x)
        logabs = jnp.log(jnp.abs(x))
        return LogArray(sign, logabs)

    # ---------- PyTree ----------
    def tree_flatten(self):
        children = (self.sign, self.logabs)
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        sign, logabs = children
        return cls(sign, logabs)

    # ---------- Basic properties ----------
    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the represented array."""
        return self.sign.shape

    @property
    def dtype(self) -> DTypeLike:
        """The data type of the represented array."""
        return jnp.promote_types(self.sign.dtype, self.logabs.dtype)

    @property
    def ndim(self) -> int:
        """The number of dimensions of the represented array."""
        return self.sign.ndim

    @property
    def size(self) -> int:
        """The total number of elements in the represented array."""
        return self.sign.size

    @property
    def nbytes(self) -> int:
        """The total number of bytes consumed by the represented array."""
        return self.sign.nbytes + self.logabs.nbytes

    @property
    def sharding(self) -> jax.sharding.Sharding:
        """The sharding of the represented array."""
        sign_sharding = self.sign.sharding
        logabs_sharding = self.logabs.sharding
        if not sign_sharding.is_equivalent_to(logabs_sharding, self.sign.ndim):
            raise ValueError(
                f"Sharding of LogArray is only defined when sign and logabs have the same sharding, "
                f"got {sign_sharding} and {logabs_sharding}"
            )
        return self.sign.sharding

    def value(self) -> Array:
        """Materialize the dense array value."""
        return self.sign * jnp.exp(self.logabs)

    # numpy / jax array conversions
    def __array__(self, dtype=None) -> NDArray:
        """Convert to a numpy array."""
        return np.asarray(self.value(), dtype)

    # JAX will prefer this to avoid dropping into host numpy during tracing (supported by recent JAX).
    def __jax_array__(self) -> Array:
        """Convert to a JAX array."""
        return self.value()

    # ---------- Unary ops ----------
    def __neg__(self) -> LogArray:
        """Negation of the represented value."""
        return LogArray(sign=-self.sign, logabs=self.logabs)

    @property
    def T(self) -> LogArray:
        """Transpose of the represented array."""
        return LogArray(self.sign.T, self.logabs.T)

    @property
    def mT(self) -> LogArray:
        """Matrix transpose of the represented array."""
        return LogArray(self.sign.mT, self.logabs.mT)

    def conj(self) -> LogArray:
        """Complex conjugate of the represented array."""
        return LogArray(sign=jnp.conj(self.sign), logabs=self.logabs)

    def abs(self) -> LogArray:
        """Absolute value of the represented array."""
        real_dtype = jnp.finfo(self.sign.dtype).dtype
        sign = jnp.ones_like(self.sign, dtype=real_dtype)
        return LogArray(sign=sign, logabs=self.logabs)

    def __abs__(self) -> LogArray:
        """Absolute value of the represented array."""
        return self.abs()

    @property
    def real(self) -> LogArray:
        """Real part of the represented array."""
        if jnp.iscomplexobj(self):
            sign = jnp.sign(self.sign.real)
            logabs = self.logabs + jnp.log(jnp.abs(self.sign.real))
            return LogArray(sign, logabs)
        else:
            return self

    @property
    def imag(self) -> LogArray:
        """Imaginary part of the represented array."""
        if jnp.iscomplexobj(self):
            sign = jnp.sign(self.sign.imag)
            logabs = self.logabs + jnp.log(jnp.abs(self.sign.imag))
            return LogArray(sign, logabs)
        else:
            return LogArray(
                jnp.zeros_like(self.sign), jnp.full_like(self.logabs, -jnp.inf)
            )

    def astype(self, dtype) -> LogArray:
        """Cast the represented array to given dtype."""
        real_dtype = jnp.finfo(dtype).dtype
        return LogArray(self.sign.astype(dtype), self.logabs.astype(real_dtype))

    # ---------- Binary ops ----------
    def __mul__(self, other: ArrayLike) -> LogArray:
        """Element-wise multiplication."""
        other = LogArray.from_value(other)
        sign = self.sign * other.sign
        logabs = self.logabs + other.logabs
        return LogArray(sign, logabs)

    def __rmul__(self, other: ArrayLike) -> LogArray:
        """Reversed element-wise multiplication."""
        return self.__mul__(other)

    def __truediv__(self, other: ArrayLike) -> LogArray:
        """Element-wise division."""
        other = LogArray.from_value(other)
        sign = self.sign / other.sign
        logabs = self.logabs - other.logabs
        return LogArray(sign, logabs)

    def __rtruediv__(self, other: ArrayLike) -> LogArray:
        """Reversed element-wise division."""
        other = LogArray.from_value(other)
        sign = other.sign / self.sign
        logabs = other.logabs - self.logabs
        return LogArray(sign, logabs)

    def __pow__(self, p: float | Array) -> LogArray:
        """Element-wise power."""
        p = jnp.asarray(p)
        sign = jnp.power(self.sign, p)
        logabs = self.logabs * p
        return LogArray(sign, logabs)

    def __add__(self, other: ArrayLike) -> LogArray:
        """Element-wise addition."""
        other = LogArray.from_value(other)
        x, b = _addexp(self.logabs, other.logabs, self.sign, other.sign)
        sign = jnp.sign(b)
        logabs = x + jnp.log(jnp.abs(b))
        return LogArray(sign, logabs)

    def __radd__(self, other: ArrayLike) -> LogArray:
        """Reversed element-wise addition."""
        return self.__add__(other)

    def __sub__(self, other: ArrayLike) -> LogArray:
        """Element-wise subtraction."""
        other = LogArray.from_value(other)
        return self.__add__(-other)

    def __rsub__(self, other: ArrayLike) -> LogArray:
        """Reversed element-wise subtraction."""
        return LogArray.from_value(other).__add__(-self)

    # ---------- Reductions ----------
    def sum(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> LogArray:
        """Sum of array elements over a given axis."""
        logabs, sign = sumexp(self.logabs, self.sign, axis=axis, keepdims=keepdims)
        logabs += jnp.log(jnp.abs(sign))
        sign = _phase(sign)
        return LogArray(sign, logabs)

    def mean(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> LogArray:
        """Mean of array elements over a given axis."""
        logabs, sign = meanexp(self.logabs, self.sign, axis=axis, keepdims=keepdims)
        logabs += jnp.log(jnp.abs(sign))
        sign = _phase(sign)
        return LogArray(sign, logabs)

    def prod(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> LogArray:
        """Product of array elements over a given axis."""
        sign = jnp.prod(self.sign, axis=axis, keepdims=keepdims)
        logabs = jnp.sum(self.logabs, axis=axis, keepdims=keepdims)
        return LogArray(sign, logabs)

    # ---------- Cumulative --------------

    # ---------- Representation ----------
    def __repr__(self) -> str:
        return f"LogArray(\n  sign={self.sign},\n  logabs={self.logabs}\n)"


@register_pytree_node_class
@dataclass
class ScaleArray:
    r"""
    Array representation with a scale: value = significand * exp(exponent),
    where exponent is a normalization factor.

    The array is a PyTree with two leaves: `significand` and `exponent`. To convert it to a
    dense JAX array, use `arr.value()` or `jnp.asarray(arr)`.

    .. note::
        The same value can be represented by different (significand, exponent) pairs.
        For example, (e, 0) and (1, 1) both represent the value e. We don't enforce
        a canonical form for better performance.

    .. warning::

        JAX doesn't have a full support for `customized arrays <https://docs.jax.dev/en/latest/jep/28661-jax-array-protocol.html>`_,
        so one should be careful when using ``ScaleArray``.
        Here we list several possible problems.

        1. Manipulations like ``jnp.fn(array)`` transform customized arrays to ``Array``.
        To avoid it, call ``array.fn()`` whenever possible.

        2. Computations like ``jax_array * customized_array`` always call
        ``jax_array.__mul__(customized_array)``, which returns a ``Array``.
        To avoid it, use ``customized_array * jax_array``.

    """

    significand: Array
    exponent: Array

    # Make Python/Numpy prefer our overloads when mixed types appear.
    __array_priority__: ClassVar[int] = 2000

    # Methods generated later
    __getitem__: ClassVar[Callable[..., ScaleArray]]
    choose: ClassVar[Callable[..., ScaleArray]]
    compress: ClassVar[Callable[..., ScaleArray]]
    copy: ClassVar[Callable[..., ScaleArray]]
    diagonal: ClassVar[Callable[..., ScaleArray]]
    flatten: ClassVar[Callable[..., ScaleArray]]
    ravel: ClassVar[Callable[..., ScaleArray]]
    repeat: ClassVar[Callable[..., ScaleArray]]
    reshape: ClassVar[Callable[..., ScaleArray]]
    squeeze: ClassVar[Callable[..., ScaleArray]]
    swapaxes: ClassVar[Callable[..., ScaleArray]]
    take: ClassVar[Callable[..., ScaleArray]]
    transpose: ClassVar[Callable[..., ScaleArray]]

    @staticmethod
    def from_value(x: ArrayLike) -> ScaleArray:
        """Create from a JAX array / Python scalar."""
        if isinstance(x, ScaleArray):
            return x

        if isinstance(x, LogArray):
            return ScaleArray(x.sign, x.logabs)

        x = jnp.asarray(x)
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            dtype = jnp.finfo(x.dtype).dtype
        else:
            dtype = x.dtype
        exponent = jnp.zeros(x.shape, dtype=dtype)
        return ScaleArray(significand=x, exponent=exponent)

    # ---------- PyTree ----------
    def tree_flatten(self):
        children = (self.significand, self.exponent)
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        significand, exponent = children
        return cls(significand, exponent)

    # ---------- Basic properties ----------
    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the represented array."""
        return self.significand.shape

    @property
    def dtype(self) -> DTypeLike:
        """The data type of the represented array."""
        return jnp.promote_types(self.significand.dtype, self.exponent.dtype)

    @property
    def ndim(self) -> int:
        """The number of dimensions of the represented array."""
        return self.significand.ndim

    @property
    def size(self) -> int:
        """The total number of elements in the represented array."""
        return self.significand.size

    @property
    def nbytes(self) -> int:
        """The total number of bytes consumed by the represented array."""
        return self.significand.nbytes + self.exponent.nbytes

    @property
    def sharding(self) -> jax.sharding.Sharding:
        """The sharding of the represented array."""
        return self.significand.sharding

    def value(self) -> Array:
        """Materialize the dense array value."""
        return self.significand * jnp.exp(self.exponent)

    # numpy / jax array conversions
    def __array__(self, dtype=None) -> NDArray:
        """Convert to a numpy array."""
        return np.asarray(self.value(), dtype)

    # JAX will prefer this to avoid dropping into host numpy during tracing (supported by recent JAX).
    def __jax_array__(self) -> Array:
        """Convert to a JAX array."""
        return self.value()

    # ---------- Unary ops ----------
    def __neg__(self) -> ScaleArray:
        """Negate the represented value."""
        return ScaleArray(-self.significand, self.exponent)

    @property
    def T(self) -> ScaleArray:
        """Transpose the represented array."""
        return ScaleArray(self.significand.T, self.exponent.T)

    @property
    def mT(self) -> ScaleArray:
        """Matrix transpose of the represented array."""
        return ScaleArray(self.significand.mT, self.exponent.mT)

    def conj(self) -> ScaleArray:
        """Complex conjugate of the represented value."""
        return ScaleArray(self.significand.conj(), self.exponent)

    def abs(self) -> ScaleArray:
        """Absolute value of the represented array."""
        return ScaleArray(jnp.abs(self.significand), self.exponent)

    def __abs__(self) -> ScaleArray:
        """Absolute value of the represented array."""
        return self.abs()

    @property
    def real(self) -> ScaleArray:
        """Real part of the represented array."""
        return ScaleArray(self.significand.real, self.exponent)

    @property
    def imag(self) -> ScaleArray:
        """Imaginary part of the represented array."""
        return ScaleArray(self.significand.imag, self.exponent)

    def astype(self, dtype) -> ScaleArray:
        """Cast the represented array to given dtype."""
        significant = self.significand.astype(dtype)
        exponent = self.exponent.astype(jnp.finfo(dtype).dtype)
        return ScaleArray(significant, exponent)

    # ---------- Binary ops ----------
    def __mul__(self, other: ArrayLike) -> ScaleArray:
        """Element-wise multiplication."""
        other = ScaleArray.from_value(other)
        significand = self.significand * other.significand
        exponent = self.exponent + other.exponent
        return ScaleArray(significand, exponent)

    def __rmul__(self, other: ArrayLike) -> ScaleArray:
        """Reversed element-wise multiplication."""
        return self.__mul__(other)

    def __truediv__(self, other: ArrayLike) -> ScaleArray:
        """Element-wise division."""
        other = ScaleArray.from_value(other)
        significand = self.significand / other.significand
        exponent = self.exponent - other.exponent
        return ScaleArray(significand, exponent)

    def __rtruediv__(self, other: ArrayLike) -> ScaleArray:
        """Reversed element-wise division."""
        other = ScaleArray.from_value(other)
        significand = other.significand / self.significand
        exponent = other.exponent - self.exponent
        return ScaleArray(significand, exponent)

    def __pow__(self, p: float | Array) -> ScaleArray:
        """Element-wise power."""
        p = jnp.asarray(p)
        significand = jnp.power(self.significand, p)
        exponent = self.exponent * p
        return ScaleArray(significand, exponent)

    def __add__(self, other: ArrayLike) -> ScaleArray:
        """Element-wise addition."""
        other = ScaleArray.from_value(other)
        exponent, significand = _addexp(
            self.exponent, other.exponent, self.significand, other.significand
        )
        return ScaleArray(significand, exponent)

    def __radd__(self, other: ArrayLike) -> ScaleArray:
        """Reversed element-wise addition."""
        return self.__add__(other)

    def __sub__(self, other: ArrayLike) -> ScaleArray:
        """Element-wise subtraction."""
        other = ScaleArray.from_value(other)
        return self.__add__(-other)

    def __rsub__(self, other: ArrayLike) -> ScaleArray:
        """Reversed element-wise subtraction."""
        return ScaleArray.from_value(other).__add__(-self)

    # ---------- Reductions ----------
    def sum(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> ScaleArray:
        """Sum of array elements over a given axis."""
        exponent = self.exponent
        significand = self.significand
        exponent, significand = sumexp(exponent, significand, axis, keepdims)
        return ScaleArray(significand, exponent)

    def mean(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> ScaleArray:
        """Mean of array elements over a given axis."""
        exponent = self.exponent
        significand = self.significand
        exponent, significand = meanexp(exponent, significand, axis, keepdims)
        return ScaleArray(significand, exponent)

    def prod(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> ScaleArray:
        """Product of array elements over a given axis."""
        sign = jnp.sign(self.significand)
        logabs = jnp.log(jnp.abs(self.significand))
        is_finite = jnp.isfinite(logabs)
        logabs = jnp.where(is_finite, logabs, 0.0)
        significand = jnp.where(is_finite, sign, self.significand)
        significand = jnp.prod(significand, axis=axis, keepdims=keepdims)
        exponent = self.exponent + logabs
        exponent = jnp.sum(exponent, axis=axis, keepdims=keepdims)
        return ScaleArray(significand, exponent)

    # ---------- Cumulative --------------

    # ---------- Representation ----------
    def __repr__(self) -> str:
        return f"ScaleArray(\n  significand={self.significand},\n  exponent={self.exponent}\n)"


PsiArray = NDArray | Array | LogArray | ScaleArray


_methods = (
    "__getitem__",
    "choose",
    "compress",
    "copy",
    "diagonal",
    "flatten",
    "ravel",
    "repeat",
    "reshape",
    "squeeze",
    "swapaxes",
    "take",
    "transpose",
)


def _make_log_method(name: str) -> Callable:
    def _method(self: LogArray, *args, **kwargs) -> LogArray:
        sign = getattr(self.sign, name)(*args, **kwargs)
        logabs = getattr(self.logabs, name)(*args, **kwargs)
        return LogArray(sign, logabs)

    _method.__name__ = name
    _method.__doc__ = f"Apply ``{name}`` to sign and logabs component-wise."
    return _method


for _name in _methods:
    setattr(LogArray, _name, _make_log_method(_name))


def _make_scale_method(name: str) -> Callable:
    def _method(self: ScaleArray, *args, **kwargs) -> ScaleArray:
        significand = getattr(self.significand, name)(*args, **kwargs)
        exponent = getattr(self.exponent, name)(*args, **kwargs)
        return ScaleArray(significand, exponent)

    _method.__name__ = name
    _method.__doc__ = f"Apply ``{name}`` to significand and exponent component-wise."
    return _method


for _name in _methods:
    setattr(ScaleArray, _name, _make_scale_method(_name))


def where(cond: ArrayLike, x: ArrayLike, y: ArrayLike) -> PsiArray:
    """
    Element-wise selection ``cond ? x : y`` that preserves the
    :class:`LogArray` / :class:`ScaleArray` representation.

    If either ``x`` or ``y`` is a :class:`ScaleArray` the result is a
    :class:`ScaleArray`; otherwise if either is a :class:`LogArray` the result is
    a :class:`LogArray`. In both cases the other operand is converted with
    ``from_value``. Falls back to :func:`jax.numpy.where` / :func:`numpy.where`
    for plain arrays.
    """
    if isinstance(x, ScaleArray) or isinstance(y, ScaleArray):
        cond = jnp.asarray(cond)
        x = ScaleArray.from_value(x)
        y = ScaleArray.from_value(y)
        exponent = jnp.where(cond, x.exponent, y.exponent)
        significand = jnp.where(cond, x.significand, y.significand)
        return ScaleArray(significand, exponent)
    elif isinstance(x, LogArray) or isinstance(y, LogArray):
        cond = jnp.asarray(cond)
        x = LogArray.from_value(x)
        y = LogArray.from_value(y)
        sign = jnp.where(cond, x.sign, y.sign)
        logabs = jnp.where(cond, x.logabs, y.logabs)
        return LogArray(sign, logabs)
    elif isinstance(x, Array) or isinstance(y, Array):
        cond = jnp.asarray(cond)
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        return jnp.where(cond, x, y)
    else:
        return np.where(cond, x, y)
