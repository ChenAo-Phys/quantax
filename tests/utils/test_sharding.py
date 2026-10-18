import hashlib
from types import SimpleNamespace
import numpy as np
import jax
import pytest
from jax.sharding import NamedSharding
from quantax.utils import (
    make_mesh,
    get_distributed_P,
    get_distributed_sharding,
    get_replicated_sharding,
    make_precompile_mesh,
    use_portable_compilation_cache,
)

NDEV = jax.device_count()


def test_mesh_and_sharding_layouts():
    mesh = make_mesh()
    # a (process, device) mesh spanning every device of the job
    assert mesh.axis_names == ("process", "device")
    assert mesh.devices.shape == (jax.process_count(), jax.local_device_count())
    assert set(mesh.devices.flat) == set(jax.devices())

    # distributed: the first axis is split evenly, one distinct block per device
    x = np.arange(2 * NDEV * 3).reshape(2 * NDEV, 3)
    dist = jax.device_put(x, get_distributed_sharding())
    blocks = sorted(np.asarray(s.data).tolist() for s in dist.addressable_shards)
    assert blocks == [x[2 * i : 2 * i + 2].tolist() for i in range(NDEV)]
    # get_distributed_P(axis) splits another axis instead
    y = jax.device_put(x.T, NamedSharding(mesh, get_distributed_P(1)))
    assert all(s.data.shape == (3, 2) for s in y.addressable_shards)
    # In a single process the "process" axis has size 1, so the blocks above are the
    # same with or without it; only the spec shows that multi-host runs also split
    # the samples over the processes instead of replicating them.
    assert get_distributed_sharding().spec == jax.P(("process", "device"))
    assert get_distributed_P(1) == jax.P(None, ("process", "device"))
    # replicated: a full copy on every device
    repl = jax.device_put(x, get_replicated_sharding())
    assert len(repl.addressable_shards) == NDEV
    for s in repl.addressable_shards:
        np.testing.assert_array_equal(s.data, x)


def test_make_precompile_mesh_requires_gpu(monkeypatch):
    # compile-only topologies exist only for GPU
    if jax.default_backend() == "gpu":
        pytest.skip("the guard only triggers off GPU")
    from jax._src import cache_key

    # a failed call must leave the persistent-cache key of the process unpatched;
    # the monkeypatch restores the jax internals in case it doesn't
    hash_fns = {}
    for name in ("_hash_accelerator_config", "_hash_platform"):
        hash_fns[name] = getattr(cache_key, name)
        monkeypatch.setattr(cache_key, name, hash_fns[name])
    with pytest.raises(NotImplementedError, match="GPU"):
        make_precompile_mesh(2, 1)
    for name, fn in hash_fns.items():
        assert getattr(cache_key, name) is fn, name


def test_portable_cache_key_hashes_device_models_and_count(monkeypatch):
    # A precompiling job and the real run see different topologies of the same
    # hardware, so the accelerator part of the persistent-cache key may depend only
    # on the device models and the device count.
    from jax._src import cache_key

    # undo the patch of the jax internals after the test
    hash_fn = cache_key._hash_accelerator_config
    monkeypatch.setattr(cache_key, "_hash_accelerator_config", hash_fn)
    use_portable_compilation_cache()

    def _key(devices):
        hash_obj = hashlib.sha256()
        devices = np.asarray(devices, dtype=object)
        cache_key._hash_accelerator_config(hash_obj, devices)
        return hash_obj.hexdigest()

    def _fake(*kinds):  # devices that carry nothing but their model
        return [SimpleNamespace(device_kind=kind) for kind in kinds]

    devices = np.array(jax.devices(), dtype=object)
    kind = devices[0].device_kind
    # device ids, order, layout and the other topology metadata don't enter the key
    assert _key(devices[::-1].reshape(NDEV, 1)) == _key(devices)
    assert _key(devices) == _key(_fake(*[kind] * NDEV))
    # the models of a mixed job are hashed as a sorted set, so neither their order
    # nor their multiplicity enters; 16 models make the reversed insertion order
    # reorder an unsorted set as well
    models = [f"model {i}" for i in range(16)]
    assert _key(_fake(*models, models[0])) == _key(_fake(*models[::-1], models[1]))
    # a different model or device count changes the key
    assert _key(_fake("A", "A")) != _key(_fake("A", "B"))
    assert _key(_fake("A")) != _key(_fake("A", "A"))
