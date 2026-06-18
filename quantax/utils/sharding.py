"""
Helpers for building the JAX device mesh and the shardings Quantax uses to
distribute or replicate arrays across all available devices.

The mesh is two-dimensional, ``("process", "device")``, with one axis of size
:func:`jax.process_count` and another of size :func:`jax.local_device_count`,
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
        have sizes :func:`jax.process_count` and :func:`jax.local_device_count`,
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


def get_distributed_P() -> jax.P:
    """
    The :class:`jax.sharding.PartitionSpec` (``jax.P``) that distributes an
    array along its first dimension over both mesh axes.

    :return:
        ``jax.P(("process", "device"))``.
    """
    return jax.P(("process", "device"))


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
        an empty :class:`jax.sharding.PartitionSpec`.
    """
    mesh = make_mesh()
    return NamedSharding(mesh, jax.P())
