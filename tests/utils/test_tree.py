import numpy as np
import jax
import jax.numpy as jnp
from quantax.utils import (
    tree_fully_flatten,
    filter_tree_map,
    tree_split_cpl,
    tree_combine_cpl,
    apply_updates,
)


def test_tree_fully_flatten():
    tree = {"a": jnp.arange(3.0), "b": jnp.ones((2, 2))}
    flat = tree_fully_flatten(tree)
    assert flat.shape == (7,)
    assert flat.ndim == 1


def test_filter_tree_map_skips_non_arrays():
    tree = {"a": jnp.arange(3.0), "b": "label"}
    out = filter_tree_map(lambda x: x + 1, tree)
    assert np.array_equal(np.asarray(out["a"]), np.array([1.0, 2.0, 3.0]))
    assert out["b"] == "label"  # non-array left untouched


def test_filter_tree_map_multiple_trees():
    a = {"x": jnp.arange(3.0)}
    b = {"x": jnp.ones(3)}
    out = filter_tree_map(lambda x, y: x + y, a, b)
    assert np.array_equal(np.asarray(out["x"]), np.array([1.0, 2.0, 3.0]))


def test_tree_split_combine_roundtrip():
    tree = {"a": jnp.array([1 + 2j, 3 + 4j]), "b": 5, "c": jnp.array([1.0, 2.0])}
    real, imag = tree_split_cpl(tree)
    assert np.allclose(np.asarray(real["a"]), [1.0, 3.0])
    assert np.allclose(np.asarray(imag["a"]), [2.0, 4.0])
    combined = tree_combine_cpl(real, imag)
    assert np.allclose(np.asarray(combined["a"]), np.asarray(tree["a"]))
    assert np.allclose(np.asarray(combined["c"]), np.asarray(tree["c"]))
    assert combined["b"] == 5  # non-array passthrough


def test_apply_updates():
    model = {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.0])}
    updates = {"w": jnp.array([0.5, 0.5]), "b": jnp.array([1.0])}
    out = apply_updates(model, updates)
    assert np.allclose(np.asarray(out["w"]), [1.5, 2.5])
    assert np.allclose(np.asarray(out["b"]), [1.0])


def test_apply_updates_none_keeps_param():
    model = {"w": jnp.array([1.0, 2.0])}
    updates = {"w": None}
    out = apply_updates(model, updates)
    assert np.allclose(np.asarray(out["w"]), [1.0, 2.0])


def test_apply_updates_preserves_dtype():
    model = {"w": jnp.array([1.0, 2.0], dtype=jnp.float16)}
    updates = {"w": jnp.array([1.0, 1.0], dtype=jnp.float32)}
    out = apply_updates(model, updates)
    # the model's dtype is preserved despite the higher-precision update
    assert out["w"].dtype == jnp.float16
