import numpy as np
import jax
import jax.numpy as jnp
from quantax.utils import (
    is_sharded_array,
    to_distributed_array,
    to_replicated_array,
    global_to_local,
    local_to_global,
    local_to_replicated,
    to_replicated_numpy,
    get_distributed_sharding,
    get_replicated_sharding,
    array_extend,
    array_set,
)

NDEV = jax.device_count()


def test_distributed_and_replicated_arrays():
    x = np.arange(4.0 * NDEV).reshape(2 * NDEV, 2)
    dist_sharding, repl_sharding = get_distributed_sharding(), get_replicated_sharding()
    # in a single process, local and global arrays coincide
    dist = [to_distributed_array(x), local_to_global(x)]
    repl = [to_replicated_array(x), local_to_replicated(x)]
    assert all(a.sharding.is_equivalent_to(dist_sharding, 2) for a in dist)
    assert all(a.sharding.is_equivalent_to(repl_sharding, 2) for a in repl)
    assert global_to_local(dist[0]) is dist[0]
    # gathered back into one contiguous host array
    for arr in dist + repl:
        out = to_replicated_numpy(arr)
        assert isinstance(out, np.ndarray) and out.flags.c_contiguous
        np.testing.assert_array_equal(out, x)

    # only jax arrays placed on the mesh count as sharded, not host or
    # single-device arrays
    assert is_sharded_array(dist[0]) and is_sharded_array(repl[0])
    assert not is_sharded_array(jax.device_put(x, jax.devices()[0]))
    assert not any(is_sharded_array(a) for a in (x, 5, [1, 2, 3]))


def test_array_extend():
    # pads the axis up to the next multiple with padding_values
    out = array_extend(jnp.arange(5), 4)
    np.testing.assert_array_equal(out, [0, 1, 2, 3, 4, 0, 0, 0])
    out = array_extend(jnp.ones((2, 3)), 4, axis=1, padding_values=7)
    np.testing.assert_array_equal(out, [[1, 1, 1, 7], [1, 1, 1, 7]])
    # fast return without a copy when the size is already a multiple
    x = jnp.arange(8)
    assert array_extend(x, 4) is x


def test_array_set():
    # complex arrays are set through the real and imaginary parts separately
    inds = np.array([1, 2])
    a = np.arange(4.0, dtype=np.float32)
    for a, vals in [
        (a, np.array([5.0, 6.0])),
        (a * (1 - 1j), np.array([1 + 2j, 3 - 1j])),
    ]:
        vals = vals.astype(a.dtype)
        out = jax.jit(array_set)(a, inds, vals)
        expected = a.copy()
        expected[inds] = vals
        assert out.dtype == a.dtype
        np.testing.assert_array_equal(out, expected)
