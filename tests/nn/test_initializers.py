import numpy as np
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import pytest
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

KEY = jr.PRNGKey(0)
# Non-square shape (out_features, in_features) so the fan-in convention is testable:
# in_axis=1 -> fan_in = 400. A wrong axis would give fan_in = 100 and the wrong variance.
FAN_OUT, FAN_IN = 100, 400
SHAPE = (FAN_OUT, FAN_IN)


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
    ],
)
def test_initializer_variance(init, expected_var):
    w = init(KEY, SHAPE, jnp.float32)
    assert w.shape == SHAPE
    assert w.dtype == jnp.float32
    # Sample variance over 40000 entries; loose tolerance for statistical noise.
    np.testing.assert_allclose(np.var(np.asarray(w)), expected_var, rtol=0.1)


def test_variance_scaling_partial():
    # variance_scaling is a partial with the axis convention baked in; supplying
    # scale/mode/distribution reproduces lecun_normal (scale 1, fan_in, normal).
    init = variance_scaling(1.0, "fan_in", "truncated_normal")
    w = init(KEY, SHAPE, jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(w), np.asarray(lecun_normal(KEY, SHAPE, jnp.float32))
    )


# ---------- apply_* on Linear and Conv ----------

APPLY_FNS = [apply_lecun_normal, apply_glorot_normal, apply_he_normal]


@pytest.mark.parametrize("apply_fn", APPLY_FNS)
def test_apply_linear(apply_fn):
    net = eqx.nn.Linear(FAN_IN, FAN_OUT, use_bias=True, key=KEY)
    out = apply_fn(KEY, net)
    assert out.weight.shape == net.weight.shape
    assert out.weight.dtype == net.weight.dtype
    # bias is zeroed
    assert np.all(np.asarray(out.bias) == 0.0)
    # the weight was actually re-initialized (not left at eqx's default)
    assert not np.allclose(np.asarray(out.weight), np.asarray(net.weight))


@pytest.mark.parametrize("apply_fn", APPLY_FNS)
def test_apply_conv(apply_fn):
    net = eqx.nn.Conv1d(3, 8, kernel_size=5, use_bias=True, key=KEY)
    out = apply_fn(KEY, net)
    assert out.weight.shape == net.weight.shape
    assert out.weight.dtype == net.weight.dtype
    assert np.all(np.asarray(out.bias) == 0.0)


@pytest.mark.parametrize("apply_fn", APPLY_FNS)
def test_apply_no_bias(apply_fn):
    net = eqx.nn.Linear(FAN_IN, FAN_OUT, use_bias=False, key=KEY)
    assert net.bias is None
    out = apply_fn(KEY, net)
    assert out.bias is None
    assert out.weight.shape == net.weight.shape


@pytest.mark.parametrize("apply_fn", APPLY_FNS)
def test_apply_does_not_mutate_input(apply_fn):
    net = eqx.nn.Linear(FAN_IN, FAN_OUT, use_bias=True, key=KEY)
    w_before = np.array(net.weight)
    b_before = np.array(net.bias)
    apply_fn(KEY, net)
    np.testing.assert_array_equal(np.asarray(net.weight), w_before)
    np.testing.assert_array_equal(np.asarray(net.bias), b_before)


@pytest.mark.parametrize("apply_fn", APPLY_FNS)
def test_apply_deterministic(apply_fn):
    net = eqx.nn.Linear(FAN_IN, FAN_OUT, key=KEY)
    out1 = apply_fn(KEY, net)
    out2 = apply_fn(KEY, net)
    np.testing.assert_array_equal(np.asarray(out1.weight), np.asarray(out2.weight))
    # different keys give different weights
    out3 = apply_fn(jr.PRNGKey(1), net)
    assert not np.allclose(np.asarray(out1.weight), np.asarray(out3.weight))
