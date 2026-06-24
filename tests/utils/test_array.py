import numpy as np
import jax
import jax.numpy as jnp
from quantax.utils import (
    is_sharded_array,
    to_distributed_array,
    to_replicated_array,
    to_replicated_numpy,
    array_extend,
    array_set,
)


def test_is_sharded_array_non_jax():
    assert is_sharded_array(np.zeros(3)) is False
    assert is_sharded_array(5) is False
    assert is_sharded_array([1, 2, 3]) is False


def test_to_replicated_array_roundtrip():
    x = jnp.arange(8.0)
    out = to_replicated_array(x)
    assert isinstance(out, jax.Array)
    assert np.allclose(np.asarray(out), np.arange(8.0))


def test_to_distributed_array_roundtrip():
    x = jnp.arange(8.0).reshape(8, 1)
    out = to_distributed_array(x)
    assert isinstance(out, jax.Array)
    assert np.allclose(np.asarray(out), np.asarray(x))


def test_to_replicated_numpy():
    x = to_distributed_array(jnp.arange(8.0).reshape(8, 1))
    out = to_replicated_numpy(x)
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, np.arange(8.0).reshape(8, 1))


def test_array_extend_pads():
    out = array_extend(jnp.arange(5), 4)
    assert out.shape == (8,)
    assert np.array_equal(np.asarray(out), [0, 1, 2, 3, 4, 0, 0, 0])


def test_array_extend_no_op_when_multiple():
    x = jnp.arange(8)
    out = array_extend(x, 4)
    assert out is x  # fast return, no copy


def test_array_extend_axis_and_padding():
    x = jnp.ones((2, 3))
    out = array_extend(x, 4, axis=1, padding_values=7)
    assert out.shape == (2, 4)
    assert np.array_equal(np.asarray(out)[:, 3], [7, 7])


def test_array_set_real():
    a = jnp.zeros(4)
    out = array_set(a, jnp.array([1, 2]), jnp.array([5.0, 6.0]))
    assert np.array_equal(np.asarray(out), [0, 5, 6, 0])


def test_array_set_complex():
    a = jnp.zeros(4, dtype=jnp.complex64)
    out = array_set(a, jnp.array([1, 2]), jnp.array([1 + 2j, 3 - 1j]))
    assert np.allclose(np.asarray(out), [0, 1 + 2j, 3 - 1j, 0])
    assert jnp.iscomplexobj(out)
