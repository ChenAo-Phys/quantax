from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Union, Callable
from jaxtyping import ArrayLike
import numpy as np
import jax
import jax.numpy as jnp
from jax import tree_util


_ArrayLike = Union[ArrayLike, "LogArray", "ScaleArray"]


@jax.custom_jvp
def _addexp(
    x1: jax.Array, x2: jax.Array, b1: jax.Array, b2: jax.Array
) -> Tuple[jax.Array, jax.Array]:
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
    primals: Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    tangents: Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
) -> Tuple[Tuple[jax.Array, jax.Array], Tuple[jax.Array, jax.Array]]:
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
def _sumexp(x: jax.Array, b: jax.Array) -> Tuple[jax.Array, jax.Array]:
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
    primals: Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    tangents: Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
) -> Tuple[Tuple[jax.Array, jax.Array], Tuple[jax.Array, jax.Array]]:
    xi, bi = primals
    dxi, dbi = tangents
    xmax = jnp.max(xi)
    ri = jnp.where(xi != xmax, jnp.exp(xi - xmax), 1.0)
    b = jnp.sum(bi * ri)
    dx = jnp.zeros_like(xmax)
    db = jnp.sum(ri * (bi * dxi + dbi))
    return (xmax, b), (dx, db)


def _get_reduction_size(
    shape: Tuple[int, ...], axis: Union[int, Tuple[int, ...], None]
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
    x: jax.Array,
    b: jax.Array,
    axis: Union[int, Tuple[int, ...], None] = None,
    keepdims: bool = False,
) -> Tuple[jax.Array, jax.Array]:
    r"""
    Compute :math:`\sum_i b_i \exp(x_i)` and return two values (x, b) to represent the
    result b * exp(x).
    """
    if axis is None:
        axis = tuple(range(len(x.shape)))
    elif isinstance(axis, int):
        axis = (axis,)

    x = jnp.moveaxis(x, axis, range(len(axis)))
    b = jnp.moveaxis(b, axis, range(len(axis)))
    reduction_size = _get_reduction_size(x.shape, axis)
    remaining_shape = x.shape[len(axis) :]
    x = x.reshape(reduction_size, -1)
    b = b.reshape(reduction_size, -1)
    x, b = jax.vmap(_sumexp, in_axes=1)(x, b)
    x = x.reshape(remaining_shape)
    b = b.reshape(remaining_shape)
    if keepdims:
        axis = sorted(axis)
        x = jnp.expand_dims(x, axis)
        b = jnp.expand_dims(b, axis)

    return x, b


def meanexp(
    x: jax.Array,
    b: jax.Array,
    axis: Union[int, Tuple[int, ...], None] = None,
    keepdims: bool = False,
) -> Tuple[jax.Array, jax.Array]:
    r"""
    Compute :math:`\left< b_i \exp(x_i) \right>` and return two values (x, b) to represent the
    result b * exp(x).
    """
    if axis is None:
        axis = tuple(range(len(x.shape)))
    elif isinstance(axis, int):
        axis = (axis,)

    x = jnp.moveaxis(x, axis, range(len(axis)))
    b = jnp.moveaxis(b, axis, range(len(axis)))
    reduction_size = _get_reduction_size(x.shape, axis)
    remaining_shape = x.shape[len(axis) :]
    x = x.reshape(reduction_size, -1)
    b = b.reshape(reduction_size, -1)
    x, b = jax.vmap(_sumexp, in_axes=1)(x, b)
    x = x.reshape(remaining_shape)
    b = b.reshape(remaining_shape)
    x -= jnp.log(reduction_size)
    if keepdims:
        axis = sorted(axis)
        x = jnp.expand_dims(x, axis)
        b = jnp.expand_dims(b, axis)

    return x, b


@tree_util.register_pytree_node_class
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

        1. Manipulations like ``jnp.fn(array)`` transform customized arrays to ``jax.Array``.
        To avoid it, call ``array.fn()`` whenever possible.

        2. Computations like ``jax_array * customized_array`` always call
        ``jax_array.__mul__(customized_array)``, which returns a ``jax.Array``.
        To avoid it, use ``customized_array * jax_array``.

    """

    sign: ArrayLike  # sign or phase
    logabs: ArrayLike  # real log-magnitude

    # Make Python/Numpy prefer our overloads when mixed types appear.
    __array_priority__ = 1000

    @staticmethod
    def from_value(x: _ArrayLike) -> LogArray:
        """Create from a JAX array / Python scalar."""
        if isinstance(x, LogArray):
            return x

        if isinstance(x, ScaleArray):
            sign = jnp.sign(x.significand)
            logabs = jnp.log(jnp.abs(x.significand)) + x.exponent
            return LogArray(sign, logabs)

        x = jnp.asarray(x)
        sign = jnp.sign(x)
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
    def shape(self) -> Tuple[int, ...]:
        """The shape of the represented array."""
        return self.sign.shape

    @property
    def dtype(self) -> jnp.dtype:
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

    def value(self) -> jax.Array:
        """Materialize the dense array value."""
        return self.sign * jnp.exp(self.logabs)

    # numpy / jax array conversions
    def __array__(self, dtype=None) -> np.ndarray:
        """Convert to a numpy array."""
        return np.asarray(self.value(), dtype)

    # JAX will prefer this to avoid dropping into host numpy during tracing (supported by recent JAX).
    def __jax_array__(self) -> jax.Array:
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
    def __mul__(self, other: _ArrayLike) -> LogArray:
        """Element-wise multiplication."""
        other = LogArray.from_value(other)
        sign = self.sign * other.sign
        logabs = self.logabs + other.logabs
        return LogArray(sign, logabs)

    def __rmul__(self, other: _ArrayLike) -> LogArray:
        """Reversed element-wise multiplication."""
        return self.__mul__(other)

    def __truediv__(self, other: _ArrayLike) -> LogArray:
        """Element-wise division."""
        other = LogArray.from_value(other)
        sign = self.sign / other.sign
        logabs = self.logabs - other.logabs
        return LogArray(sign, logabs)

    def __rtruediv__(self, other: _ArrayLike) -> LogArray:
        """Reversed element-wise division."""
        other = LogArray.from_value(other)
        sign = other.sign / self.sign
        logabs = other.logabs - self.logabs
        return LogArray(sign, logabs)

    def __pow__(self, p: Union[int, float, jax.Array]) -> LogArray:
        """Element-wise power."""
        p = jnp.asarray(p)
        sign = jnp.power(self.sign, p)
        logabs = self.logabs * p
        return LogArray(sign, logabs)

    def __add__(self, other: _ArrayLike) -> LogArray:
        """Element-wise addition."""
        other = LogArray.from_value(other)
        x, b = _addexp(self.logabs, other.logabs, self.sign, other.sign)
        sign = jnp.sign(b)
        logabs = x + jnp.log(jnp.abs(b))
        return LogArray(sign, logabs)

    def __radd__(self, other: _ArrayLike) -> LogArray:
        """Reversed element-wise addition."""
        return self.__add__(other)

    def __sub__(self, other: _ArrayLike) -> LogArray:
        """Element-wise subtraction."""
        return self.__add__(-other)

    def __rsub__(self, other: _ArrayLike) -> LogArray:
        """Reversed element-wise subtraction."""
        return LogArray.from_value(other).__add__(-self)

    # ---------- Reductions ----------
    def sum(
        self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False
    ) -> LogArray:
        """Sum of array elements over a given axis."""
        logabs, sign = sumexp(self.logabs, self.sign, axis=axis, keepdims=keepdims)
        logabs += jnp.log(jnp.abs(sign))
        sign = jnp.sign(sign)
        return LogArray(sign, logabs)

    def mean(
        self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False
    ) -> LogArray:
        """Mean of array elements over a given axis."""
        logabs, sign = meanexp(self.logabs, self.sign, axis=axis, keepdims=keepdims)
        logabs += jnp.log(jnp.abs(sign))
        sign = jnp.sign(sign)
        return LogArray(sign, logabs)

    def prod(
        self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False
    ) -> LogArray:
        """Product of array elements over a given axis."""
        sign = jnp.prod(self.sign, axis=axis, keepdims=keepdims)
        logabs = jnp.sum(self.logabs, axis=axis, keepdims=keepdims)
        return LogArray(sign, logabs)

    # ---------- Cumulative --------------

    # ---------- Representation ----------
    def __repr__(self) -> str:
        return f"LogArray(\n  sign={self.sign},\n  logabs={self.logabs}\n)"


@tree_util.register_pytree_node_class
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
        a canonical form for better performance, but the `normalize` method can be used
        to obtain a normalized ``ScaleArray`` where the maximum absolute value of the significand is 1.

    .. warning::

        JAX doesn't have a full support for `customized arrays <https://docs.jax.dev/en/latest/jep/28661-jax-array-protocol.html>`_,
        so one should be careful when using ``ScaleArray``.
        Here we list several possible problems.

        1. Manipulations like ``jnp.fn(array)`` transform customized arrays to ``jax.Array``.
        To avoid it, call ``array.fn()`` whenever possible.

        2. Computations like ``jax_array * customized_array`` always call
        ``jax_array.__mul__(customized_array)``, which returns a ``jax.Array``.
        To avoid it, use ``customized_array * jax_array``.

    """

    significand: ArrayLike
    exponent: ArrayLike

    # Make Python/Numpy prefer our overloads when mixed types appear.
    __array_priority__ = 2000

    def normalize(self) -> ScaleArray:
        """Return a normalized ``ScaleArray``."""
        max_significand = jax.lax.stop_gradient(jnp.max(jnp.abs(self.significand)))
        max_exponent = jax.lax.stop_gradient(jnp.max(self.exponent))
        exponent = max_exponent + jnp.log(max_significand)
        finite_exp = jnp.isfinite(exponent)
        exponent = jnp.where(finite_exp, exponent, max_exponent)
        exp_diff = jnp.where(self.exponent != exponent, self.exponent - exponent, 0.0)
        significand = self.significand * jnp.exp(exp_diff)
        return ScaleArray(significand, exponent)

    @staticmethod
    def from_value(x: _ArrayLike) -> ScaleArray:
        """Create from a JAX array / Python scalar."""
        if isinstance(x, ScaleArray):
            return x

        if isinstance(x, LogArray):
            return ScaleArray(x.sign, x.logabs)

        return ScaleArray(significand=x, exponent=0.0).normalize()

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
    def shape(self) -> Tuple[int, ...]:
        """The shape of the represented array."""
        return self.significand.shape

    @property
    def dtype(self) -> jnp.dtype:
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

    def value(self) -> jax.Array:
        """Materialize the dense array value."""
        return self.significand * jnp.exp(self.exponent)

    # numpy / jax array conversions
    def __array__(self, dtype=None) -> np.ndarray:
        """Convert to a numpy array."""
        return np.asarray(self.value(), dtype)

    # JAX will prefer this to avoid dropping into host numpy during tracing (supported by recent JAX).
    def __jax_array__(self) -> jax.Array:
        """Convert to a JAX array."""
        return self.value()

    # ---------- Unary ops ----------
    def __neg__(self) -> ScaleArray:
        """Negate the represented value."""
        return ScaleArray(-self.significand, self.exponent)

    @property
    def T(self) -> ScaleArray:
        """Transpose the represented array."""
        return ScaleArray(self.significand.T, self.exponent)

    @property
    def mT(self) -> ScaleArray:
        """Matrix transpose of the represented array."""
        return ScaleArray(self.significand.mT, self.exponent)

    def conj(self) -> ScaleArray:
        """Complex conjugate of the represented value."""
        return ScaleArray(self.significand.conj(), self.exponent)

    def abs(self) -> ScaleArray:
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
    def __mul__(self, other: _ArrayLike) -> ScaleArray:
        """Element-wise multiplication."""
        other = ScaleArray.from_value(other)
        significand = self.significand * other.significand
        exponent = self.exponent + other.exponent
        return ScaleArray(significand, exponent)

    def __rmul__(self, other: _ArrayLike) -> ScaleArray:
        """Reversed element-wise multiplication."""
        return self.__mul__(other)

    def __truediv__(self, other: _ArrayLike) -> ScaleArray:
        """Element-wise division."""
        other = ScaleArray.from_value(other)
        significand = self.significand / other.significand
        exponent = self.exponent - other.exponent
        return ScaleArray(significand, exponent)

    def __rtruediv__(self, other: _ArrayLike) -> ScaleArray:
        """Reversed element-wise division."""
        other = ScaleArray.from_value(other)
        significand = other.significand / self.significand
        exponent = other.exponent - self.exponent
        return ScaleArray(significand, exponent)

    def __pow__(self, p: Union[int, float, jax.Array]) -> ScaleArray:
        """Element-wise power."""
        p = jnp.asarray(p)
        significand = jnp.power(self.significand, p)
        exponent = self.exponent * p
        return ScaleArray(significand, exponent)

    def __add__(self, other: _ArrayLike) -> ScaleArray:
        """Element-wise addition."""
        other = ScaleArray.from_value(other)
        exponent, significand = _addexp(
            self.exponent, other.exponent, self.significand, other.significand
        )
        return ScaleArray(significand, exponent)

    def __radd__(self, other: _ArrayLike) -> ScaleArray:
        """Reversed element-wise addition."""
        return self.__add__(other)

    def __sub__(self, other: _ArrayLike) -> ScaleArray:
        """Element-wise subtraction."""
        return self.__add__(-other)

    def __rsub__(self, other: _ArrayLike) -> ScaleArray:
        """Reversed element-wise subtraction."""
        return ScaleArray.from_value(other).__add__(-self)

    # ---------- Reductions ----------
    def sum(
        self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False
    ) -> ScaleArray:
        """Sum of array elements over a given axis."""
        exponent = self.exponent
        significand = self.significand

        if exponent.ndim == 0:
            significand = jnp.sum(significand, axis=axis, keepdims=keepdims)
            return ScaleArray(significand, exponent)
        elif exponent.shape == significand.shape:
            exponent, significand = sumexp(exponent, significand, axis, keepdims)
            return ScaleArray(significand, exponent)
        else:
            raise ValueError(
                f"Cannot sum ScaleArray with significand shape {self.significand.shape} "
                f"and exponent shape {self.exponent.shape}"
            )

    def mean(
        self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False
    ) -> ScaleArray:
        """Mean of array elements over a given axis."""
        exponent = self.exponent
        significand = self.significand

        if exponent.ndim == 0:
            significand = jnp.mean(significand, axis=axis, keepdims=keepdims)
            return ScaleArray(significand, exponent)
        elif exponent.shape == significand.shape:
            exponent, significand = meanexp(exponent, significand, axis, keepdims)
            return ScaleArray(significand, exponent)
        else:
            raise ValueError(
                f"Cannot average ScaleArray with significand shape {self.significand.shape} "
                f"and exponent shape {self.exponent.shape}"
            )

    def prod(
        self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False
    ) -> ScaleArray:
        """Product of array elements over a given axis."""
        sign = jnp.sign(self.significand)
        logabs = jnp.log(jnp.abs(self.significand))

        if self.exponent.ndim == 0:
            sign = jnp.prod(sign, axis=axis, keepdims=keepdims)
            logabs = jnp.sum(logabs, axis=axis, keepdims=keepdims)
            max_logabs = jax.lax.stop_gradient(jnp.max(logabs))
            max_logabs = jnp.where(jnp.isfinite(max_logabs), max_logabs, 0.0)
            significand = sign * jnp.exp(logabs - max_logabs)
            size = _get_reduction_size(self.shape, axis)
            exponent = self.exponent * size + max_logabs
            return ScaleArray(significand, exponent)
        elif self.exponent.shape == self.significand.shape:
            is_finite = jnp.isfinite(logabs)
            logabs = jnp.where(is_finite, logabs, 0.0)
            significand = jnp.where(is_finite, sign, self.significand)
            significand = jnp.prod(significand, axis=axis, keepdims=keepdims)
            exponent = self.exponent + logabs
            exponent = jnp.sum(exponent, axis=axis, keepdims=keepdims)
            return ScaleArray(significand, exponent)
        else:
            raise ValueError(
                f"Cannot multiply ScaleArray with significand shape {self.significand.shape} "
                f"and exponent shape {self.exponent.shape}"
            )

    # ---------- Cumulative --------------

    # ---------- Representation ----------
    def __repr__(self) -> str:
        return f"ScaleArray(\n  significand={self.significand},\n  exponent={self.exponent}\n)"


PsiArray = Union[np.ndarray, jax.Array, LogArray, ScaleArray]


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
        if self.exponent.ndim == 0:
            exponent = self.exponent
        elif self.exponent.shape == self.significand.shape:
            exponent = getattr(self.exponent, name)(*args, **kwargs)
        else:
            raise ValueError(
                f"Cannot apply `{name}` to ScaleArray with significand shape {self.significand.shape} "
                f"and exponent shape {self.exponent.shape}"
            )
        return ScaleArray(significand, exponent)

    _method.__name__ = name
    _method.__doc__ = f"Apply ``{name}`` to significand and exponent if possible."
    return _method


for _name in _methods:
    setattr(ScaleArray, _name, _make_scale_method(_name))


def where(cond: ArrayLike, x: _ArrayLike, y: _ArrayLike) -> _ArrayLike:
    if isinstance(x, ScaleArray) or isinstance(y, ScaleArray):
        x = ScaleArray.from_value(x)
        y = ScaleArray.from_value(y)
        exponent = jnp.where(cond, x.exponent, y.exponent)
        significand = jnp.where(cond, x.significand, y.significand)
        max_exp = jnp.max(exponent)
        significand = significand * jnp.exp(exponent - max_exp)
        return ScaleArray(significand, max_exp)
    elif isinstance(x, LogArray) or isinstance(y, LogArray):
        x = LogArray.from_value(x)
        y = LogArray.from_value(y)
        sign = jnp.where(cond, x.sign, y.sign)
        logabs = jnp.where(cond, x.logabs, y.logabs)
        return LogArray(sign, logabs)
    elif isinstance(x, jax.Array) or isinstance(y, jax.Array):
        return jnp.where(cond, x, y)
    else:
        return np.where(cond, x, y)
