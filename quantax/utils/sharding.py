import jax
from jax.sharding import NamedSharding, Mesh, AxisType


def make_mesh() -> Mesh:
    """
    Return a mesh that distributes arrays across all devices in `jax.devices()`.
    """
    shape = (jax.process_count(), jax.local_device_count())
    return jax.make_mesh(shape, ("process", "device"), (AxisType.Auto, AxisType.Auto))


def get_distributed_P() -> jax.P:
    """
    Return the PartitionSpec that distributes arrays across all devices in `jax.devices()`
    in the array's first dimension.
    """
    return jax.P(("process", "device"))


def get_distributed_sharding() -> NamedSharding:
    """
    Return the sharding that distributes arrays across all devices in
    `jax.devices()` in the array's first dimension.
    """
    mesh = make_mesh()
    return NamedSharding(mesh, get_distributed_P())


def get_replicated_sharding() -> NamedSharding:
    """
    Return the sharding that replicates arrays across all devices in `jax.devices()`.
    """
    mesh = make_mesh()
    return NamedSharding(mesh, jax.P())
