r"""
Numerically stable array representation for neural quantum states.

Wavefunction amplitudes in many-body systems easily over- or underflow the
floating-point range, so this module provides the PyTree array type
:class:`LogArray`, which stores
:math:`\text{value} = \text{sign} \cdot \exp(\text{logabs})`, where ``sign`` is a
unit :math:`\pm 1` (or complex phase) and ``logabs`` is the real log-magnitude.
Zero is encoded as ``sign=0, logabs=-inf``.

It supports the common arithmetic, reduction, and reshaping operations while
keeping the computation in the stable representation, and converts to a dense JAX
array via ``arr.value()`` or ``jnp.asarray(arr)``. See the warning on the class for
caveats about JAX's partial support for custom arrays.

:data:`PsiArray` is the union of :class:`LogArray` with plain numpy / JAX arrays,
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
    Unit phase :math:`x / |x|` (with ``0`` mapped to ``0``), equal to
    :func:`jax.numpy.sign` for real ``x``. For complex ``x`` the gradient is
    correct: ``jnp.sign`` has a *zero* JVP for complex inputs, which silently
    discards the phase gradient of complex wavefunctions. The gradient is zero for
    real ``x`` and at the origin.
    """
    if not jnp.iscomplexobj(x):
        return jnp.sign(x)
    absx = jnp.abs(x)
    return jnp.where(absx == 0, jnp.zeros_like(x), x / absx)


@_phase.defjvp
def _phase_jvp(primals: tuple[Array], tangents: tuple[Array]) -> tuple[Array, Array]:
    (x,), (dx,) = primals, tangents
    zeros = jnp.zeros_like(x)
    if not jnp.iscomplexobj(x):
        return jnp.sign(x), zeros

    absx = jnp.abs(x)
    p = jnp.where(absx == 0, zeros, x / absx)
    # d(x/|x|) = dx/|x| - x Re(conj(x) dx) / |x|^3
    safe = jnp.where(absx == 0, 1, absx)
    d = dx / safe - x * jnp.real(jnp.conj(x) * dx) / safe**3
    dp = jnp.where(absx == 0, zeros, d)
    return p, dp


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


def _exp_shifted(logabs: Array, logmax: Array) -> Array:
    r"""
    :math:`\exp(\text{logabs} - \text{logmax})` for ``logabs <= logmax``, with ``1``
    where ``logabs == logmax == +-inf`` instead of ``exp(nan)``.

    ``logmax`` must be a ``stop_gradient`` constant, so that the derivative of a sum
    :math:`\sum_i \text{sign}_i \exp(\text{logabs}_i - \text{logmax})` is exact,
    including at ties of the maximum.
    """
    tie_inf = jnp.isinf(logabs) & (logabs == logmax)
    return jnp.exp(jnp.where(tie_inf, 0.0, logabs - logmax))


@register_pytree_node_class
@dataclass
class LogArray:
    r"""
    Log-amplitude representation of JAX arrays: value = sign * exp(logabs)
    where ``sign`` is :math:`\pm 1` or a complex phase and ``logabs`` is real.
    Zero is encoded by sign=0, logabs=-inf.

    The array is a PyTree with two leaves: ``sign`` and ``logabs``. To convert it to a
    dense JAX array, use ``arr.value()`` or ``jnp.asarray(arr)``.

    .. warning::

        JAX doesn't have full support for `customized arrays <https://docs.jax.dev/en/latest/jep/28661-jax-array-protocol.html>`_,
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

        x = jnp.asarray(x)
        sign = _phase(x)
        logabs = jnp.log(jnp.abs(x))
        return LogArray(sign, logabs)

    # ---------- PyTree ----------
    def tree_flatten(self) -> tuple[tuple[jax.Array, jax.Array], None]:
        children = (self.sign, self.logabs)
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(
        cls, aux: None, children: tuple[jax.Array, jax.Array]
    ) -> LogArray:
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
    def __array__(self, dtype: DTypeLike | None = None) -> NDArray:
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
        # |sign| is exactly 1 for a valid sign or phase, so it is set to the constant 1
        # rather than computed. A non-finite sign (e.g. NaN from an invalid phase)
        # must be kept, otherwise ``abs`` would turn an invalid value into a finite
        # one that ``isnan`` / ``isfinite`` can no longer detect.
        sign = jnp.where(jnp.isfinite(self.sign), 1, jnp.abs(self.sign))
        return LogArray(sign=sign.astype(real_dtype), logabs=self.logabs)

    def __abs__(self) -> LogArray:
        """Absolute value of the represented array."""
        return self.abs()

    def isnan(self) -> Array:
        """True where the represented value is NaN."""
        return jnp.isnan(self.sign) | jnp.isnan(self.logabs)

    def isinf(self) -> Array:
        """True where the represented value is infinite (overflowed)."""
        return jnp.isinf(self.sign) | jnp.isposinf(self.logabs)

    def isfinite(self) -> Array:
        """True where the represented value is finite (logabs=-inf is zero, finite)."""
        finite_logabs = jnp.isfinite(self.logabs) | jnp.isneginf(self.logabs)
        return jnp.isfinite(self.sign) & finite_logabs

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

    def astype(self, dtype: DTypeLike) -> LogArray:
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
        logmax = jax.lax.stop_gradient(jnp.maximum(self.logabs, other.logabs))
        b = self.sign * _exp_shifted(self.logabs, logmax)
        b = b + other.sign * _exp_shifted(other.logabs, logmax)
        return LogArray(_phase(b), logmax + jnp.log(jnp.abs(b)))

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
        logmax = jnp.max(self.logabs, axis=axis, keepdims=True)
        logmax = jax.lax.stop_gradient(logmax)
        b = self.sign * _exp_shifted(self.logabs, logmax)
        b = jnp.sum(b, axis=axis, keepdims=keepdims)
        if not keepdims:
            logmax = jnp.squeeze(logmax, axis)
        return LogArray(_phase(b), logmax + jnp.log(jnp.abs(b)))

    def mean(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> LogArray:
        """Mean of array elements over a given axis."""
        out = self.sum(axis=axis, keepdims=keepdims)
        size = _get_reduction_size(self.shape, axis)
        return LogArray(out.sign, out.logabs - jnp.log(size))

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


PsiArray = NDArray | Array | LogArray
"""
Type alias of all array types that can hold wave function amplitudes, including the
ordinary numpy and jax arrays and the customized :class:`LogArray` for very large or
small numbers.
"""


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


def where(cond: ArrayLike, x: ArrayLike, y: ArrayLike) -> PsiArray:
    """
    Element-wise selection ``cond ? x : y`` that preserves the :class:`LogArray`
    representation. If either ``x`` or ``y`` is a :class:`LogArray`, the other
    operand is converted with ``from_value`` and the result is a :class:`LogArray`.
    Falls back to :func:`jax.numpy.where` / :func:`numpy.where` for plain arrays.
    """
    if isinstance(x, LogArray) or isinstance(y, LogArray):
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


def isnan(x: PsiArray) -> Array:
    """
    Element-wise NaN test that dispatches to :meth:`LogArray.isnan` and falls back
    to :func:`jax.numpy.isnan` for plain arrays.
    """
    if isinstance(x, LogArray):
        return x.isnan()
    return jnp.isnan(x)


def isinf(x: PsiArray) -> Array:
    """
    Element-wise infinity test that dispatches to :meth:`LogArray.isinf` and falls
    back to :func:`jax.numpy.isinf` for plain arrays.
    """
    if isinstance(x, LogArray):
        return x.isinf()
    return jnp.isinf(x)


def isfinite(x: PsiArray) -> Array:
    """
    Element-wise finiteness test that dispatches to :meth:`LogArray.isfinite`
    (where a zero entry counts as finite) and falls back to
    :func:`jax.numpy.isfinite` for plain arrays.
    """
    if isinstance(x, LogArray):
        return x.isfinite()
    return jnp.isfinite(x)
