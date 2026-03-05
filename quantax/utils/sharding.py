import jax
from jax.sharding import NamedSharding, Mesh, PartitionSpec, AxisType


def make_mesh() -> Mesh:
    """
    Return a mesh that distributes arrays across all devices in `jax.devices()`.
    """
    return jax.make_mesh((jax.device_count(),), ("x",), (AxisType.Auto,))


def get_distributed_sharding() -> NamedSharding:
    """
    Return the sharding that distributes arrays across all devices in
    `jax.devices()` in the array's first dimension.
    """
    global_mesh = make_mesh()
    global_pspecs = PartitionSpec("x")
    return NamedSharding(global_mesh, global_pspecs)


def get_replicated_sharding() -> NamedSharding:
    """
    Return the sharding that replicates arrays across all devices in `jax.devices()`.
    """
    global_mesh = make_mesh()
    replicate_pspecs = PartitionSpec()
    return NamedSharding(global_mesh, replicate_pspecs)
