import jax
from jax.sharding import NamedSharding, Mesh
from quantax.utils import (
    make_mesh,
    get_distributed_P,
    get_distributed_sharding,
    get_replicated_sharding,
)


def test_make_mesh_size():
    mesh = make_mesh()
    assert isinstance(mesh, Mesh)
    # the mesh spans all devices in the job
    assert mesh.size == jax.device_count()
    assert set(mesh.axis_names) == {"process", "device"}


def test_distributed_P():
    P = get_distributed_P()
    assert P == jax.P(("process", "device"))


def test_distributed_sharding():
    sharding = get_distributed_sharding()
    assert isinstance(sharding, NamedSharding)
    assert sharding.spec == get_distributed_P()


def test_replicated_sharding():
    sharding = get_replicated_sharding()
    assert isinstance(sharding, NamedSharding)
    assert sharding.spec == jax.P()
