import numpy as np
import jax
from jax.sharding import NamedSharding
from quantax.utils import (
    make_mesh,
    get_distributed_P,
    get_distributed_sharding,
    get_replicated_sharding,
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
