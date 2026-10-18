import numpy as np
import jax.numpy as jnp
from quantax.utils import (
    filter_tree_map,
    tree_split_cpl,
    tree_combine_cpl,
    apply_updates,
)


def test_filter_tree_map():
    # maps only the array leaves of one or several trees, other leaves pass through
    a = {"x": jnp.arange(3.0), "label": "a"}
    b = {"x": jnp.ones(3), "label": "b"}
    out = filter_tree_map(lambda x, y: x + y, a, b)
    np.testing.assert_array_equal(out["x"], [1.0, 2.0, 3.0])
    assert out["label"] == "a"


def test_tree_split_combine_roundtrip():
    tree = {
        "a": jnp.array([1 + 2j, 3 + 4j]),
        "b": 5,
        "c": jnp.array([1.0, 2.0]),
        "n": jnp.array([1, 2]),
    }
    real, imag = tree_split_cpl(tree)
    np.testing.assert_array_equal(real["a"], [1.0, 3.0])
    np.testing.assert_array_equal(imag["a"], [2.0, 4.0])
    np.testing.assert_array_equal(imag["c"], [0.0, 0.0])
    combined = tree_combine_cpl(real, imag)
    np.testing.assert_array_equal(combined["a"], tree["a"])
    np.testing.assert_array_equal(combined["c"], tree["c"])
    # non-inexact leaves (python scalars, integer arrays) pass through unchanged
    assert combined["b"] == 5 and imag["b"] == 5
    np.testing.assert_array_equal(imag["n"], tree["n"])
    assert combined["n"].dtype == tree["n"].dtype
    np.testing.assert_array_equal(combined["n"], tree["n"])


def test_apply_updates():
    model = {
        "w": jnp.array([1.0, 2.0], dtype=jnp.float16),
        "b": jnp.array([0.0]),
        "frozen": jnp.array([1.0, 2.0]),
        "scalar": 1.5,
    }
    updates = {
        "w": jnp.array([0.5, 0.5], dtype=jnp.float32),
        "b": jnp.array([1.0]),
        "frozen": None,  # None leaves the parameter unchanged
        "scalar": 0.5,
    }
    out = apply_updates(model, updates)
    np.testing.assert_array_equal(out["w"], [1.5, 2.5])
    # the model's dtype is preserved despite the higher-precision update
    assert out["w"].dtype == jnp.float16
    np.testing.assert_array_equal(out["b"], [1.0])
    assert out["frozen"] is model["frozen"]
    assert out["scalar"] == 2.0
