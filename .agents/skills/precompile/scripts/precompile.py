"""Precompile the expensive jitted functions of a VMC run for a larger target topology.

Run this as a single process on one node of the same cluster, in the same environment
(jax/jaxlib, CUDA, XLA_FLAGS, GPU model) as the target run; one GPU is enough. It
writes the executables of

- the Metropolis sweep (`Metropolis._sweep_variational`, wave function included),
- the local-energy forward (`Variational._batch_forward`, used by `state(s)`),
- the Jacobian (`Variational._grad_vmap`, used by `state.jacobian`),

compiled for the target layout of TARGET_PROCESSES processes x
TARGET_DEVICES_PER_PROCESS devices, into the persistent compilation cache CACHE_DIR.
The target run (see target.py) loads them instead of compiling.

The CONFIG block must be identical to the target run: every argument of the
lattice, Hamiltonian-independent model and state settings, `max_parallel`, the
sampler class and its arguments, `nsamples`, default dtype, and matmul precision.
"""

import os

# persistent compilation cache on a filesystem shared with the target run
CACHE_DIR = "<shared path>/jax_cache"
# process layout of the target run, which enters the cache key, not only the total
# device count: (nodes, GPUs per node) for process_per_node.slurm, and
# (nodes * GPUs per node, 1) for process_per_gpu.slurm. E.g. 8 nodes of 4 GPUs:
TARGET_PROCESSES = 8
TARGET_DEVICES_PER_PROCESS = 4

import jax

# The cache must be configured before the first compilation, i.e. before
# importing quantax.
jax.config.update("jax_compilation_cache_dir", CACHE_DIR)
# The ahead-of-time compiles below reuse the autotuning results of the warm-up and
# are often shorter than the default 1 s threshold, under which nothing is written.
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

import numpy as np
import jax.numpy as jnp
import equinox as eqx
from equinox._jit import _preprocess  # equinox internals, checked with equinox 0.13
from jax.experimental import topologies
from jax.sharding import NamedSharding, Mesh, AxisType
from jax._src import cache_key as _cache_key  # jax internals of the cache key
import quantax as qtx
from quantax.utils import get_distributed_P, to_distributed_array


def use_portable_compilation_cache() -> None:
    """
    Make persistent-cache keys portable across device topologies of the same
    hardware (the same function is in target.py). Stock jax hashes the full topology
    fingerprint into every key. It differs between a compile-only topology and the
    same devices in a real multi-node run, which host layout metadata enters,
    although the compiled executable is identical. This hashes the device model(s)
    and the device count instead; the other components of the key (computation,
    compile options, jaxlib and CUDA versions, XLA flags) are hashed as usual.
    """

    def _hash_accelerator_config(hash_obj, accelerators) -> None:
        kinds = sorted({str(d.device_kind) for d in accelerators.flat})
        _cache_key._hash_string(hash_obj, ",".join(kinds))
        hash_obj.update(int(accelerators.size).to_bytes(8, byteorder="big"))

    _cache_key._hash_accelerator_config = _hash_accelerator_config


def make_precompile_mesh(num_processes: int, local_device_count: int) -> Mesh:
    """
    Compile-only ``("process", "device")`` mesh of ``num_processes`` processes x
    ``local_device_count`` devices, mirroring `quantax.utils.make_mesh` of the target
    run without owning its devices. Arrays can't be created on it; it can only be
    used to trace, lower and compile. Only supported on GPU, with the same GPU
    model, CUDA, jaxlib and XLA flags as the target run, since the cache keys hash
    all of them.
    """
    backend = jax.local_devices()[0].client
    if backend.platform != "gpu":
        raise NotImplementedError(
            f"Compile-only meshes are only supported on GPU, got '{backend.platform}'."
        )
    use_portable_compilation_cache()
    # The compile-only PJRT client reports the device name as platform_version,
    # while the real runtime client reports the CUDA version ("PJRT C API\ncuda
    # ..."). The string is hashed into the cache key, so entries compiled here would
    # never be found by the real run: hash the strings of the real client instead.
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


# compile-only mesh of the target run; also makes the cache keys topology-portable
mesh = make_precompile_mesh(TARGET_PROCESSES, TARGET_DEVICES_PER_PROCESS)

# ------------------------------- CONFIG ---------------------------------------
# Must be identical to the target run.
NSAMPLES = 4096  # global number of samples of the target run
MAX_PARALLEL = (4096, 64, 4096)  # (forward, backward, ref) chunk per device

qtx.sites.Square(8, Nparticles=(32, 32))
symm = qtx.symmetry.Rotation(np.pi / 2) @ qtx.symmetry.Flip()
symm = symm @ qtx.symmetry.SpinInverse()
model = qtx.model.ResConv(nblocks=4, channels=16, kernel_size=3)
state = qtx.state.Variational(model, symm=symm, max_parallel=MAX_PARALLEL)


def make_sampler(state, nsamples):
    # same sampler class and arguments as the target run, except `nsamples`
    # (scaled to the devices of this job) and no thermalization. MixSampler: split
    # `nsamples` among the components in the target's ratio, thermal_steps=0 for all
    return qtx.sampler.SpinExchange(state, nsamples, thermal_steps=0)


# ------------------------------------------------------------------------------


def aot_compile(jit_fn, args, shardings):
    """
    Lower and compile an equinox ``filter_jit`` wrapper for the target mesh.

    ``args`` are example arguments with the global shapes of the target run, and
    ``shardings`` gives, per argument, the target sharding of all its array leaves
    (None for unspecified, e.g. numpy scalars). The array leaves are replaced by
    `jax.ShapeDtypeStruct` templates after equinox splits the arguments, because
    ``filter_jit(...).lower`` treats ShapeDtypeStruct as a static leaf. Mirrors
    `_call` of equinox/_jit.py.
    """
    target = {}
    for arg, sharding in zip(args, shardings):
        for leaf in jax.tree.leaves(eqx.filter(arg, eqx.is_array)):
            target[id(leaf)] = sharding

    def to_struct(x):
        if eqx.is_array(x):
            return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=target[id(x)])
        return x

    info = (
        jit_fn._signature,
        jit_fn._dynamic_fun,
        jit_fn._static_fun,
        jit_fn.donate_first,
        jit_fn.donate_rest,
    )
    donate, nodonate, static = _preprocess(info, args, {}, return_static=True)
    donate = jax.tree.map(to_struct, donate)
    nodonate = jax.tree.map(to_struct, nodonate)
    jit_fn._cached.lower(donate, nodonate, static).compile()


def cache_entries():
    names = os.listdir(CACHE_DIR) if os.path.isdir(CACHE_DIR) else []
    return {n for n in names if n.endswith("-cache")}


entries_before = cache_entries()

ndev = jax.device_count()  # devices of this single-process job
if NSAMPLES % mesh.size != 0:
    raise ValueError(f"NSAMPLES={NSAMPLES} is not divisible by {mesh.size} devices.")
ns_dev = NSAMPLES // mesh.size  # samples per device in the target run
s0 = np.asarray(qtx.utils.rand_states()).reshape(1, -1)
rows = lambda n: np.repeat(s0, n, axis=0)

# Same number of chains per device as the target run. The sampler sweeps them in
# chunks of `forward_chunk` (`ref_chunk` for local updates) per device if there
# are more chains than that.
sampler = make_sampler(state, ns_dev * ndev)
# MixSampler: nsamples is the sum of the components'
assert sampler.nsamples == ns_dev * ndev, "make_sampler must give ns_dev * ndev chains"
chunk = state.ref_chunk if sampler.use_ref else state.forward_chunk
sweep_dev = ns_dev if chunk is None else min(ns_dev, chunk)
sweep_batch = sweep_dev * mesh.size  # global batch of one sweep call in the target

# 1. Warm-up on the real devices with the per-device batches of the target run.
# cuDNN/GEMM autotuning can't profile on compile-only devices, and an executable
# compiled with unresolved engine choices crashes at runtime. The autotuning
# results of the warm-up are stored in the per-fusion autotune cache next to the
# persistent cache (`jax_persistent_cache_enable_xla_caches`, enabled by default)
# and reused by the ahead-of-time compiles below.
if state.forward_chunk is not None:
    state(rows(state.forward_chunk * ndev))
state.jacobian(to_distributed_array(jnp.asarray(rows(ns_dev * ndev))))
sampler.sweep(1)

# 2. Rebind the jitted functions of the state to the target mesh, so that the
# trace-time chunking and shard_map resolve to the target topology. The state
# and sampler can't run computations afterwards.
state._init_forward(mesh)
state._init_backward(mesh)
# Drop the in-memory traces of the warm-up. If the target has as many devices as
# this job, the global shapes coincide and jit would reuse the sweep traced on the
# local devices ("Received incompatible devices").
jax.clear_caches()

# 3. Ahead-of-time compile for the target mesh.
replicated = NamedSharding(mesh, jax.P())
canonical = NamedSharding(mesh, get_distributed_P())
# The cache key contains the sharding of the arguments as spelled by its spec.
# Arrays put by `to_distributed_array` carry the canonical spec
# P(("process", "device")), while outputs of jitted functions (e.g. the spins of
# a sweep) drop the mesh axes of size 1. Compile both spellings if they differ.
dropped = jax.P(tuple(a for a in mesh.axis_names if mesh.shape[a] > 1))
distributed = [canonical]
if dropped != get_distributed_P():
    distributed.append(NamedSharding(mesh, dropped))

model = state.model
key = qtx.get_subkeys()
nsweeps = np.asarray(1, dtype=np.int32)  # traced, any value

# local energy forward, its input is always resharded to the canonical spec
if state.forward_chunk is not None:
    s = rows(state.forward_chunk * mesh.size)
    aot_compile(state._batch_forward, (model, s), (replicated, canonical))

for i, dist in enumerate(distributed):
    if i > 0:
        # The in-memory lowering cache treats both spellings as the same sharding
        # and would reuse the executables of the first one without writing new
        # entries, so it's cleared to lower the second spelling from scratch.
        jax.clear_caches()

    # Metropolis sweep, called by `sampler.sweep()`
    s = rows(sweep_batch)
    args = (sampler, key, nsweeps, s, model)
    shardings = (None, replicated, None, dist, replicated)
    aot_compile(type(sampler)._sweep_variational, args, shardings)

    # Jacobian of all samples, called by `StochasticQNGD.get_Obar` (SR, AdamSR, ...)
    s = rows(NSAMPLES)
    aot_compile(state._grad_vmap, (model, s), (replicated, dist))

PRECOMPILED = ("jit__sweep_variational", "jit_batch_forward", "jit_chunked_f")
new = sorted(n for n in cache_entries() - entries_before if n.startswith(PRECOMPILED))
# compare with the mesh and device kind printed by target.py
kind = mesh.devices.flat[0].device_kind
print(f"Target mesh {dict(mesh.shape)}, device kind: {kind}")
print("New cache entries (warm-up included):")
for name in new:
    print("  ", name[:60])
