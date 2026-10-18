import numpy as np
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import pytest
import quantax as qtx
from quantax.nn import (
    variance_scaling,
    lecun_normal,
    lecun_uniform,
    glorot_normal,
    glorot_uniform,
    he_normal,
    he_uniform,
    apply_lecun_normal,
    apply_glorot_normal,
    apply_he_normal,
)

# The (out_channels, in_channels, kx, ky) weight layout of a 2D conv, with in_axis=1
# and out_axis=0 and the kernel as receptive field: fan_in = 30 * 6 and
# fan_out = 40 * 6. Swapping the axes changes the variance by 25%, and JAX's default
# axes (in=-2, out=-1) by much more. The same shape is used for the Conv layers in
# test_conv.py, so the random initializers compile once (keys are drawn from quantax
# as in the layers, since keys of another sharding would compile anew).
SHAPE = (40, 30, 3, 2)
FAN_IN, FAN_OUT = 30 * 6, 40 * 6


# ---------- module-level initializers: variance / fan convention ----------


@pytest.mark.parametrize(
    "init, expected_var",
    [
        (lecun_normal, 1.0 / FAN_IN),  # scale 1, fan_in
        (lecun_uniform, 1.0 / FAN_IN),
        (he_normal, 2.0 / FAN_IN),  # scale 2, fan_in
        (he_uniform, 2.0 / FAN_IN),
        (glorot_normal, 2.0 / (FAN_IN + FAN_OUT)),  # scale 1, fan_avg
        (glorot_uniform, 2.0 / (FAN_IN + FAN_OUT)),
        # the partial keeps the axis convention for a user-chosen mode
        (variance_scaling(1.0, "fan_out", "truncated_normal"), 1.0 / FAN_OUT),
    ],
    ids=[
        "lecun_normal",
        "lecun_uniform",
        "he_normal",
        "he_uniform",
        "glorot_normal",
        "glorot_uniform",
        "variance_scaling_fan_out",
    ],
)
def test_initializer_variance(init, expected_var):
    w = np.asarray(init(qtx.get_subkeys(), SHAPE, np.float32))
    assert w.shape == SHAPE and w.dtype == np.float32
    # The sample variance of 7200 entries has a relative std <= 0.017, so rtol 0.1
    # is a ~6 sigma bound, while a wrong fan is off by >= 25%.
    np.testing.assert_allclose(np.var(w), expected_var, rtol=0.1)
    # The support pins the distribution: uniform on +-sqrt(3 var), and the truncated
    # normal on +-2 standard deviations of the untruncated normal (JAX's stddev
    # correction 0.8796). 7200 draws come within 5% of the edge.
    uniform = init in (lecun_uniform, he_uniform, glorot_uniform)
    bound = np.sqrt((3.0 if uniform else (2 / 0.87962566) ** 2) * expected_var)
    assert 0.95 * bound < np.abs(w).max() <= bound * (1 + 1e-5)


# ---------- apply_* ----------


@pytest.mark.parametrize(
    "apply_fn, init",
    [
        (apply_lecun_normal, lecun_normal),
        (apply_glorot_normal, glorot_normal),
        (apply_he_normal, he_normal),
    ],
    ids=["lecun", "glorot", "he"],
)
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no_bias"])
def test_apply_initializer(apply_fn, init, use_bias):
    # apply_* only reads the shapes and dtypes of the layer, so an abstract layer
    # avoids compiling the random initialization of equinox
    key = qtx.get_subkeys()
    net = eqx.filter_eval_shape(
        eqx.nn.Conv2d, 30, 40, (3, 2), use_bias=use_bias, key=key
    )
    out = apply_fn(key, net)
    # The weight is drawn by the matching initializer from the first subkey, as
    # equinox layers split their key, and the bias (if any) is zeroed.
    expected = init(jr.split(key, 2)[0], SHAPE, net.weight.dtype)
    np.testing.assert_array_equal(np.asarray(out.weight), np.asarray(expected))
    if use_bias:
        assert out.bias.shape == net.bias.shape
        assert np.all(np.asarray(out.bias) == 0.0)
    else:
        assert out.bias is None

    # the parameters keep the dtype of the layer (abstract, so nothing compiles)
    net16 = eqx.filter_eval_shape(
        eqx.nn.Conv2d, 30, 40, (3, 2), use_bias=use_bias, dtype=jnp.float16, key=key
    )
    out16 = eqx.filter_eval_shape(apply_fn, key, net16)
    assert out16.weight.dtype == jnp.float16
    if use_bias:
        assert out16.bias.dtype == jnp.float16
