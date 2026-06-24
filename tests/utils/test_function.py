import numpy as np
import jax
import jax.numpy as jnp
from quantax.sites import Chain
from quantax.utils import chunk_map, jit_chunk_vmap, to_distributed_array


def test_jit_chunk_vmap_matches_vmap():
    Chain(4)
    f = lambda x: jnp.sum(x**2)
    x = to_distributed_array(jnp.arange(20.0).reshape(20, 1))
    g = jit_chunk_vmap(f, in_axes=0, out_axes=0, chunk_size=3)
    out = g(x)
    ref = jax.vmap(f)(x)
    assert out.shape == (20,)
    assert np.allclose(np.asarray(out), np.asarray(ref))


def test_jit_chunk_vmap_no_chunk():
    Chain(4)
    f = lambda x: x * 2
    x = to_distributed_array(jnp.arange(12.0).reshape(12, 1))
    g = jit_chunk_vmap(f, in_axes=0, out_axes=0, chunk_size=None)
    out = g(x)
    assert np.allclose(np.asarray(out), np.asarray(x) * 2)


def test_chunk_map_fast_return_when_no_axes():
    f = lambda x: x
    # in_axes=None -> chunking unnecessary, original function returned
    assert chunk_map(f, in_axes=None) is f


def test_jit_chunk_vmap_fast_return_when_batch_fits_chunk():
    Chain(4)
    f = lambda x: jnp.sum(x**2)
    # device_batch (= batch // ndevices) <= chunk_size -> no chunking needed
    x = to_distributed_array(jnp.arange(12.0).reshape(12, 1))
    device_batch = 12 // jax.device_count()
    g = jit_chunk_vmap(f, in_axes=0, out_axes=0, chunk_size=device_batch)
    out = g(x)
    ref = jax.vmap(f)(x)
    assert out.shape == (12,)
    assert np.allclose(np.asarray(out), np.asarray(ref))


def test_jit_chunk_vmap_multi_output():
    Chain(4)
    f = lambda x: (jnp.sum(x), jnp.max(x))
    x = to_distributed_array(jnp.arange(16.0).reshape(16, 1))
    g = jit_chunk_vmap(f, in_axes=0, out_axes=(0, 0), chunk_size=3)
    s, m = g(x)
    ref_s, ref_m = jax.vmap(f)(x)
    assert np.allclose(np.asarray(s), np.asarray(ref_s))
    assert np.allclose(np.asarray(m), np.asarray(ref_m))
