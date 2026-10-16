from contextlib import nullcontext
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike


def is_low_precision(dtype: DTypeLike) -> bool:
    """
    Whether a forward pass with parameters in ``dtype`` runs in reduced precision,
    i.e. below single precision, or in single precision with reduced-precision
    matmuls like TF32. The latter is the default on GPU unless the matmul precision
    is set to full precision, e.g. "highest", "float32" or "F32_F32_F32", while CPU
    computes single-precision matmuls and convolutions in full precision regardless
    of the precision setting.
    """
    bits = jnp.finfo(dtype).bits
    if bits != 32 or jax.default_backend() == "cpu":
        return bits < 32
    precision = jax.config.jax_default_matmul_precision
    return precision not in ("highest", "float32", "F32_F32_F32", "F64_F64_F64")


def precision_context(dtype: DTypeLike | None):
    """
    Context manager that sets x64 to match the precision of ``dtype``, also
    disabling it for single precision so that python scalars inside lax primitives
    are not promoted to float64. ``None`` leaves the x64 setting unchanged.
    """
    if dtype is None:
        return nullcontext()
    return jax.enable_x64(jnp.finfo(dtype).bits == 64)
