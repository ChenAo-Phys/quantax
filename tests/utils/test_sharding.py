import jax
import pytest
from jax.sharding import NamedSharding, Mesh
from quantax.utils import (
    make_mesh,
    get_distributed_P,
    get_distributed_sharding,
    get_replicated_sharding,
    make_precompile_mesh,
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


def test_make_precompile_mesh_requires_gpu():
    # compile-only topologies exist only for GPU
    if jax.default_backend() == "gpu":
        pytest.skip("the guard only triggers off GPU")
    with pytest.raises(NotImplementedError, match="GPU"):
        make_precompile_mesh(2, 1)
