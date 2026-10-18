import pytest
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from quantax.utils import LogArray, chunk_map, jit_chunk_vmap, to_distributed_array

NDEV = jax.device_count()


def _f(params, x):
    # per-sample function with an unbatched first argument holding a non-array leaf
    # (like the static fields of a model) and a pytree output with a LogArray leaf
    y = params["scale"] * jnp.sum(x ** params["power"])
    return {"y": y, "psi": LogArray.from_value(x[0])}, jnp.max(x)


PARAMS = {"scale": jnp.asarray(3.0), "power": 2}


def _batch(n):
    # mixed signs, no zeros
    x = np.arange(2 * n, dtype=np.float32).reshape(n, 2) - n + 0.5
    return to_distributed_array(x)


def _assert_matches_f(out, x):
    # numpy reference of the vmapped _f, which would compile op by op if run eagerly
    x = np.asarray(x)
    out, out_max = out
    np.testing.assert_allclose(out["y"], 3 * np.sum(x**2, axis=1), rtol=1e-6)
    assert isinstance(out["psi"], LogArray)
    np.testing.assert_array_equal(out["psi"].sign, np.sign(x[:, 0]))
    np.testing.assert_allclose(out["psi"].logabs, np.log(np.abs(x[:, 0])), rtol=1e-6)
    np.testing.assert_array_equal(out_max, np.max(x, axis=1))


# out_axes per output (a tuple) or shared by all outputs (an int)
@pytest.mark.parametrize("shard_batch, out_axes", [(False, (0, 0)), (True, 0)])
def test_jit_chunk_vmap_matches_vmap(shard_batch, out_axes):
    # 5 samples per device in chunks of 2: padded to 6, and the padding is cut off
    # again while the chunks are put back in the global sample order
    x = _batch(5 * NDEV)
    g = jit_chunk_vmap(_f, (None, 0), out_axes, chunk_size=2, shard_batch=shard_batch)
    _assert_matches_f(g(PARAMS, x), x)


def test_chunk_map_single_output():
    # a single (non-tuple) output, as in the Variational forward pass, comes back
    # as is and not wrapped in a tuple, with an int or a 1-tuple out_axes, for both
    # the scan and the loop path
    f = lambda x: jnp.sum(x, axis=1)
    x = _batch(5 * NDEV)
    for use_scan in [True, False]:
        for out_axes in [0, (0,)]:
            g = chunk_map(f, 0, out_axes, chunk_size=2, use_scan=use_scan)
            out = eqx.filter_eval_shape(g, x)  # traces without compiling
            assert isinstance(out, jax.ShapeDtypeStruct), (use_scan, out_axes)
            assert out.shape == (5 * NDEV,), (use_scan, out_axes)


def test_chunk_map_loop():
    # chunk_map without scan, as used in sampling and local energies: f is called
    # chunk by chunk with at most chunk_size samples per device at a time
    batch_sizes = []
    vf = jax.jit(jax.vmap(_f, in_axes=(None, 0)))

    def f(params, x):
        batch_sizes.append(x.shape[0])
        return vf(params, x)

    x = _batch(5 * NDEV)
    out = chunk_map(f, in_axes=(None, 0), chunk_size=2)(PARAMS, x)
    _assert_matches_f(out, x)
    assert batch_sizes == [2 * NDEV] * 3

    # a batch smaller than the chunk is passed through unchanged, unless
    # atleast_1chunk pads it to a full chunk; the padded outputs are cut off
    x = to_distributed_array(np.asarray(x)[:NDEV])
    for atleast_1chunk, size in [(True, 2 * NDEV), (False, NDEV)]:
        batch_sizes.clear()
        g = chunk_map(f, (None, 0), chunk_size=2, atleast_1chunk=atleast_1chunk)
        _assert_matches_f(g(PARAMS, x), x)
        assert batch_sizes == [size]


def test_chunk_map_shard_batch_traces_local_shards():
    # shard_batch runs f inside shard_map, so f sees the per-device batch instead of
    # the global one, both for chunked batches and for the unchunked fast paths
    shapes = []

    def f(params, x):
        shapes.append(x.shape[0])
        return jax.vmap(_f, in_axes=(None, 0))(params, x)

    x = _batch(4 * NDEV)
    for chunk_size, local in [(2, 2), (4, 4), (None, 4)]:
        for shard_batch, size in [(True, local), (False, local * NDEV)]:
            shapes.clear()
            kwargs = dict(chunk_size=chunk_size, use_scan=True, shard_batch=shard_batch)
            g = chunk_map(f, (None, 0), **kwargs)
            eqx.filter_eval_shape(g, PARAMS, x)  # traces f without compiling
            assert shapes == [size], (chunk_size, shard_batch)

    # in_axes=None replicates the inputs, so every device sees the whole batch
    shapes.clear()
    eqx.filter_eval_shape(chunk_map(f, None, None, shard_batch=True), PARAMS, x)
    assert shapes == [4 * NDEV]

    # the unchunked shard_map gives the vmapped result (the chunked one is checked
    # in test_jit_chunk_vmap_matches_vmap)
    out = eqx.filter_jit(chunk_map(f, (None, 0), shard_batch=True))(PARAMS, x)
    _assert_matches_f(out, x)


def test_chunk_map_fast_return_and_validation():
    f = lambda x: x
    # nothing to chunk: the original function is returned
    assert chunk_map(f, in_axes=None, chunk_size=2) is f
    assert chunk_map(f, in_axes=(None, None), chunk_size=2) is f
    assert chunk_map(f, chunk_size=None) is f
    # unbatched outputs can't be chunked
    with pytest.raises(NotImplementedError, match="out_axes=None"):
        chunk_map(f, out_axes=(0, None), chunk_size=2)
