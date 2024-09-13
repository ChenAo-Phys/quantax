import jax
from jax.sharding import NamedSharding, Mesh, PartitionSpec


def get_global_sharding():
    global_mesh = Mesh(jax.devices(), "x")
    global_pspecs = PartitionSpec("x")
    return NamedSharding(global_mesh, global_pspecs)


def get_replicate_sharding():
    global_mesh = Mesh(jax.devices(), "x")
    replicate_pspecs = PartitionSpec()
    return NamedSharding(global_mesh, replicate_pspecs)
