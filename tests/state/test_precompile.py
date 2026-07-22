import os

import numpy as np
import jax
import jax.numpy as jnp
import pytest
import quantax as qtx
from quantax.sites import Chain
from quantax.state import Variational
from quantax.model import RBM_Dense
from quantax.utils import array_extend, make_precompile_mesh, to_distributed_array
from _dtype import use_dtype

# `Variational.precompile` ahead-of-time compiles the forward/backward jitted
# functions and stores the executables in the persistent compilation cache.
#
# Cross-topology precompilation (compile on one device setup, run on another)
# cannot be exercised inside one pytest process: the device set is fixed when
# the backend initializes (`--xla_force_host_platform_device_count`), and
# compile-only topologies (`make_precompile_mesh`) exist only for GPU. These
# tests therefore cover the same-setup path — precompile first, call later —
# plus the argument handling and the GPU-only guard. The cross-topology path
# is validated by the GPU scripts in the research project driving this feature.

# Big jitted functions covered by `precompile`; persistent-cache entries are
# files named "<module name>-<key hash>".
COVERED_PREFIXES = ("jit_batch_forward", "jit_chunked_f")


@pytest.fixture
def cache_dir(tmp_path):
    """
    Point the persistent compilation cache at a fresh directory and cache every
    compilation regardless of size. The cache object is initialized at most
    once per process and stays pinned to the first directory, so it must be
    reset around every test (as in jax's own cache tests); cleared jit caches
    force real compilations (otherwise executables cached in-process by earlier
    tests would be reused without writing cache entries). Restores the previous
    config afterwards.
    """
    from jax._src import compilation_cache as _cc

    old = (
        getattr(jax.config, "jax_compilation_cache_dir", None),
        getattr(jax.config, "jax_persistent_cache_min_compile_time_secs", 1.0),
        getattr(jax.config, "jax_persistent_cache_min_entry_size_bytes", 0),
    )
    path = tmp_path / "jax_cache"
    _cc.reset_cache()
    jax.clear_caches()
    jax.config.update("jax_compilation_cache_dir", str(path))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    yield path
    jax.config.update("jax_compilation_cache_dir", old[0])
    jax.config.update("jax_persistent_cache_min_compile_time_secs", old[1])
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", old[2])
    _cc.reset_cache()


def _make_model(features):
    # Each test uses a distinct `features` width: identical jaxprs would let a
    # later test silently reuse executables compiled (and cached in-process) by
    # an earlier one, skipping the persistent-cache writes these tests assert on.
    use_dtype(jnp.complex128)
    Chain(4, boundary=1)
    return RBM_Dense(features, dtype=jnp.complex128)


def _spins(n, nmodes, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.choice([-1, 1], size=(n, nmodes)).astype(np.int8))


def _shard(s):
    s = array_extend(jnp.asarray(s), jax.device_count())
    return to_distributed_array(s)


def _covered_entries(cache_path):
    if not os.path.isdir(cache_path):
        return set()
    return {
        name for name in os.listdir(cache_path) if name.startswith(COVERED_PREFIXES)
    }


BATCH = 2 * 4  # forward/backward batch: chunk size 2 on 4 test devices


def test_precompile_writes_cache_entries(cache_dir, x64):
    vs = Variational(_make_model(2), max_parallel=2)
    vs.precompile(forward_batches=BATCH, backward_batches=BATCH)
    entries = _covered_entries(cache_dir)
    assert any(name.startswith("jit_batch_forward") for name in entries)
    assert any(name.startswith("jit_chunked_f") for name in entries)


def test_precompile_then_call_does_not_recompile(cache_dir, x64):
    # After precompile, the covered functions must load from the cache instead
    # of compiling: no new cache entries with the covered module names.
    vs = Variational(_make_model(3), max_parallel=2)
    vs.precompile(forward_batches=BATCH, backward_batches=BATCH)
    before = _covered_entries(cache_dir)
    assert before

    spins = _spins(BATCH, vs.Nmodes)
    vs(spins)
    vs.fast_forward(_shard(spins))
    vs.jacobian(_shard(spins))
    assert _covered_entries(cache_dir) == before


def test_precompile_then_call_matches_reference(cache_dir, x64):
    # Values computed through precompiled executables must equal those of an
    # identical state that compiles normally.
    model = _make_model(4)
    vs_ref = Variational(model, max_parallel=2)
    vs = Variational(model, max_parallel=2)
    vs.precompile(forward_batches=BATCH, backward_batches=BATCH)

    spins = _spins(BATCH, vs.Nmodes)
    np.testing.assert_allclose(
        np.asarray(vs(spins)), np.asarray(vs_ref(spins)), rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(vs.fast_forward(_shard(spins))),
        np.asarray(vs_ref.fast_forward(_shard(spins))),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(vs.jacobian(_shard(spins))),
        np.asarray(vs_ref.jacobian(_shard(spins))),
        rtol=1e-6,
    )


def test_precompile_accepts_int_batches(cache_dir, x64):
    # A single int is wrapped as a one-element tuple: the int form must write
    # entries, and the tuple form must map to exactly the same cache keys.
    model = _make_model(5)
    vs = Variational(model, max_parallel=2)
    vs.precompile(forward_batches=BATCH, backward_batches=BATCH)
    entries_int = _covered_entries(cache_dir)
    assert entries_int
    vs2 = Variational(model, max_parallel=2)
    vs2.precompile(forward_batches=(BATCH,), backward_batches=(BATCH,))
    assert _covered_entries(cache_dir) == entries_int


def test_precompile_indivisible_batch_raises(cache_dir, x64):
    vs = Variational(_make_model(6), max_parallel=2)
    with pytest.raises(ValueError, match="not divisible"):
        vs.precompile(forward_batches=jax.device_count() + 1)


def test_precompile_without_cache_warns(x64):
    # Without a persistent cache the compiled executables cannot be reused.
    assert getattr(jax.config, "jax_compilation_cache_dir", None) is None
    vs = Variational(_make_model(7), max_parallel=2)
    with pytest.warns(UserWarning, match="persistent compilation cache"):
        vs.precompile()


def test_make_precompile_mesh_requires_gpu(x64):
    with pytest.raises(NotImplementedError, match="GPU"):
        make_precompile_mesh(2, 1)
