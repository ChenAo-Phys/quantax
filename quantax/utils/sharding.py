import jax
from jax.sharding import SingleDeviceSharding, NamedSharding, Mesh, PartitionSpec


local_sharding = SingleDeviceSharding(jax.devices()[0])


_global_mesh = Mesh(jax.devices(), 'x')
_global_pspecs = PartitionSpec('x')
global_sharding = NamedSharding(_global_mesh, _global_pspecs)


_replicate_pspecs = PartitionSpec()
replicate_sharding = NamedSharding(_global_mesh, _replicate_pspecs)