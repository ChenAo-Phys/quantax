"""
Helpers for building the JAX device mesh and the shardings Quantax uses to
distribute or replicate arrays across all available devices.

The mesh is two-dimensional, ``("process", "device")``, with one axis of size
`jax.process_count` and another of size :func:`jax.local_device_count`,
so that its size always matches the total number of devices in the job.
Distributed arrays are split along their first dimension over the flattened
mesh; replicated arrays are copied to every device.
"""

import numpy as np
import jax
from jax.sharding import NamedSharding, Mesh, AxisType


def make_mesh() -> Mesh:
    """
    Build the device mesh spanning all devices in :func:`jax.devices`.

    :return:
        A ``("process", "device")`` :class:`jax.sharding.Mesh` whose two axes
        have sizes `jax.process_count` and :func:`jax.local_device_count`,
        with both axes set to the automatic ``AxisType.Auto``.

    .. note::
        The mesh is built directly from :func:`jax.devices` reshaped to
        ``(process_count, local_device_count)`` rather than via
        :func:`jax.make_mesh`. ``jax.make_mesh`` rejects multi-slice topologies
        (raising on multi-host GPU since JAX 0.10), but ``jax.devices`` is
        already process-major, so the reshape reproduces the same
        ``("process", "device")`` layout while supporting multi-node runs.
    """
    devices = np.array(jax.devices()).reshape(
        jax.process_count(), jax.local_device_count()
    )
    return Mesh(
        devices, ("process", "device"), axis_types=(AxisType.Auto, AxisType.Auto)
    )


def use_portable_compilation_cache() -> None:
    """
    Make persistent-compilation-cache keys portable across device topologies of
    the same hardware, so executables written by
    `~quantax.state.Variational.precompile` under a compile-only mesh are found
    by real runs.

    Stock jax hashes the full device-topology fingerprint into every cache key.
    The fingerprint of a compile-only topology differs from that of the same
    devices in a real multi-node run (host layout metadata enters the
    fingerprint) even though the compiled executable is identical, so the real
    run would never find the precompiled entries. This replaces the topology
    component of the key with the device model(s) and the device count. All
    other components -- the computation, the compile options (including
    partition count and device assignment), the jaxlib version, the CUDA
    version, and the XLA flags -- are hashed as usual.

    Call this in **every** script that shares the cache across jobs: the
    precompiling job applies it automatically through
    :func:`make_precompile_mesh`, and the target run must call it explicitly
    before the first compilation.
    """
    from jax._src import cache_key as _cache_key

    def _hash_accelerator_config(hash_obj, accelerators) -> None:
        kinds = sorted({str(d.device_kind) for d in accelerators.flat})
        _cache_key._hash_string(hash_obj, ",".join(kinds))
        hash_obj.update(int(accelerators.size).to_bytes(8, byteorder="big"))

    _cache_key._hash_accelerator_config = _hash_accelerator_config


def make_precompile_mesh(num_processes: int, local_device_count: int) -> Mesh:
    """
    Build a compile-only ``("process", "device")`` mesh describing the topology of
    a **different** (typically larger) run, e.g. a multi-node job, without owning
    its devices. Ahead-of-time compilations staged on this mesh (see
    `~quantax.state.Variational.precompile`) are written to the persistent
    compilation cache with the same keys the real run computes, so the real run
    loads the executables instead of compiling them.

    :param num_processes:
        Number of processes of the target run (typically the number of nodes).

    :param local_device_count:
        Number of devices per process of the target run (typically GPUs per node).

    :return:
        A compile-only ``("process", "device")`` :class:`jax.sharding.Mesh` with
        shape ``(num_processes, local_device_count)``, mirroring
        :func:`make_mesh` of the target run.

    .. warning::

        This is only supported on GPU, and requires at least one visible GPU of
        the same model, CUDA version, jaxlib version, and XLA flags as the target
        run — cache keys include all of them, so in practice the precompiling job
        should run on one node of the same cluster with the same environment.

    .. note::

        Arrays cannot be created on a compile-only mesh; it can only be used to
        trace, lower, and compile. Executing a computation staged on it raises an
        error.
    """
    use_portable_compilation_cache()

    # The compile-only PJRT client reports the device name as platform_version,
    # while the real runtime client reports the CUDA version ("PJRT C API\ncuda
    # ..."). The string is hashed into the persistent-cache key, so entries
    # compiled here would never be found by the real run. Patch the hash to use
    # the real client's strings (idempotent; the real GPU client exists here).
    from jax.experimental import topologies
    from jax._src import cache_key as _cache_key

    backend = jax.local_devices()[0].client
    if backend.platform != "gpu":
        raise NotImplementedError(
            "`make_precompile_mesh` is only supported on GPU, got platform"
            f" '{backend.platform}'."
        )
    platform, platform_version = backend.platform, backend.platform_version

    def _hash_platform(hash_obj, backend) -> None:
        _cache_key._hash_string(hash_obj, platform)
        _cache_key._hash_string(hash_obj, platform_version)

    _cache_key._hash_platform = _hash_platform

    topology = topologies.get_topology_desc(
        platform="cuda", topology=f"1x{num_processes}x{local_device_count}"
    )
    devices = np.array(topology.devices).reshape(num_processes, local_device_count)
    return Mesh(
        devices, ("process", "device"), axis_types=(AxisType.Auto, AxisType.Auto)
    )


def get_distributed_P(axis: int = 0) -> jax.P:
    """
    The `jax.sharding.PartitionSpec` (``jax.P``) that distributes an
    array along the given axis over both mesh axes.
    """
    return jax.P(*((None,) * axis), ("process", "device"))


def get_distributed_sharding() -> NamedSharding:
    """
    The sharding that splits an array's first dimension evenly across all
    devices in :func:`jax.devices`.

    :return:
        A :class:`jax.sharding.NamedSharding` combining :func:`make_mesh` with
        :func:`get_distributed_P`.
    """
    mesh = make_mesh()
    return NamedSharding(mesh, get_distributed_P())


def get_replicated_sharding() -> NamedSharding:
    """
    The sharding that replicates an array on every device in
    :func:`jax.devices`.

    :return:
        A :class:`jax.sharding.NamedSharding` combining :func:`make_mesh` with
        an empty `jax.sharding.PartitionSpec`.
    """
    mesh = make_mesh()
    return NamedSharding(mesh, jax.P())
