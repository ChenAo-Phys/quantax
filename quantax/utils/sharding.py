import jax
from jax.sharding import NamedSharding, Mesh, PartitionSpec


def get_distribute_sharding() -> NamedSharding:
    """
    Return the sharding that distributes arrays across all devices in
    `jax.devices()` in the array's first dimension.
    """
    global_mesh = Mesh(jax.devices(), "x")
    global_pspecs = PartitionSpec("x")
    return NamedSharding(global_mesh, global_pspecs)


def get_replicate_sharding() -> NamedSharding:
    """
    Return the sharding that replicates arrays across all devices in
    `jax.devices()`.
    """
    global_mesh = Mesh(jax.devices(), "x")
    replicate_pspecs = PartitionSpec()
    return NamedSharding(global_mesh, replicate_pspecs)
