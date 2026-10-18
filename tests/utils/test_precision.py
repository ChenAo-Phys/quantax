import jax
import jax.numpy as jnp
from quantax.utils import precision_context, is_low_precision


def test_precision_context(x64):
    # x64 is enabled by the fixture: single precision disables it inside the context
    # so that python scalars are not promoted to float64, double precision keeps it
    # and None leaves the setting unchanged
    for dtype, x64_inside in [
        (jnp.float32, False),
        (jnp.complex64, False),
        (jnp.float64, True),
        (None, True),
    ]:
        with precision_context(dtype):
            assert jax.config.jax_enable_x64 == x64_inside, dtype
            assert jnp.asarray(1.0).dtype == (
                jnp.float64 if x64_inside else jnp.float32
            )
        assert jax.config.jax_enable_x64

    # and enables x64 for double precision when it is off
    with jax.enable_x64(False):
        with precision_context(jnp.complex128):
            assert jax.config.jax_enable_x64
        with precision_context(None):
            assert not jax.config.jax_enable_x64


def test_is_low_precision(monkeypatch):
    # Below single precision, or single precision with reduced-precision matmuls,
    # which is the default on GPU. CPU computes float32 in full precision regardless
    # of the matmul precision setting.
    full = ["highest", "float32", "F32_F32_F32", "F64_F64_F64"]
    reduced = [None, "default", "high", "bfloat16", "tensorfloat32", "BF16_BF16_F32"]
    for backend in ("cpu", "gpu"):
        monkeypatch.setattr(jax, "default_backend", lambda: backend)
        for precision in full + reduced:
            with jax.default_matmul_precision(precision):
                assert is_low_precision(jnp.bfloat16) and is_low_precision(jnp.float16)
                assert not is_low_precision(jnp.float64)
                assert not is_low_precision(jnp.complex128)
                is_low = backend == "gpu" and precision in reduced
                assert is_low_precision(jnp.float32) == is_low
                assert is_low_precision(jnp.complex64) == is_low
