"""A target VMC run loading the executables written by precompile.py.

Compared with an ordinary run, only two things are added: the persistent
compilation cache CACHE_DIR, configured before importing quantax, and
`use_portable_compilation_cache()`, called before the first compilation of the
precompiled functions. The CONFIG block must be identical to precompile.py.

Submitted with process_per_node.slurm or process_per_gpu.slurm of the hpc-usage
skill, matching TARGET_PROCESSES x TARGET_DEVICES_PER_PROCESS of precompile.py.
Process 0 prints the mesh and the device kind, which must match those printed by
precompile.py, and whether the precompiled functions are found in the cache.
"""

import os

CACHE_DIR = "<shared path>/jax_cache"  # same as precompile.py

import jax

# Skip JAX's CPU (gloo) collectives, unused by GPU runs: their per-rank hostname
# DNS lookup can fail transiently and kill a large multi-node job at startup.
jax.config.update("jax_cpu_collectives_implementation", None)

g = int(os.environ["SLURM_GPUS_ON_NODE"])  # GPUs of the node
t = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))  # tasks per node
lid = int(os.environ["SLURM_LOCALID"])
jax.distributed.initialize(
    num_processes=int(os.environ["SLURM_NTASKS"]),
    process_id=int(os.environ["SLURM_PROCID"]),
    local_device_ids=list(range(lid * g // t, (lid + 1) * g // t)),
)
if jax.process_index() == 0:
    print(
        f"processes: {jax.process_count()}, devices: {jax.device_count()}", flush=True
    )
jax.config.update("jax_compilation_cache_dir", CACHE_DIR)

import logging
import numpy as np
import quantax as qtx

qtx.utils.use_portable_compilation_cache()

if jax.process_index() == 0:
    # compare with the target mesh and device kind printed by precompile.py
    mesh_shape = dict(qtx.utils.make_mesh().shape)
    print(f"Mesh {mesh_shape}, device kind: {jax.devices()[0].device_kind}", flush=True)

PRECOMPILED = ("jit__sweep_variational", "jit_batch_forward", "jit_chunked_f")


class CacheReport(logging.Handler):
    """Print persistent-cache hits and misses of the precompiled functions."""

    def emit(self, record):
        msg = record.getMessage()
        if msg.startswith("Persistent compilation cache hit for"):
            result = "hit "
        elif msg.startswith("PERSISTENT COMPILATION CACHE MISS for"):
            result = "MISS"
        else:
            return
        name = msg.split("'")[1]
        if name.startswith(PRECOMPILED) and jax.process_index() == 0:
            print(f"cache {result} {name}", flush=True)


logger = logging.getLogger("jax._src.compiler")
logger.setLevel(logging.DEBUG)
logger.addHandler(CacheReport())

# ------------------------------- CONFIG ---------------------------------------
# Must be identical to precompile.py.
NSAMPLES = 4096
MAX_PARALLEL = (4096, 64, 4096)

qtx.sites.Square(8, Nparticles=(32, 32))
symm = qtx.symmetry.Rotation(np.pi / 2) @ qtx.symmetry.Flip()
symm = symm @ qtx.symmetry.SpinInverse()
model = qtx.model.ResConv(nblocks=4, channels=16, kernel_size=3)
state = qtx.state.Variational(model, symm=symm, max_parallel=MAX_PARALLEL)
sampler = qtx.sampler.SpinExchange(state, NSAMPLES)
# ------------------------------------------------------------------------------
assert sampler.nsamples == NSAMPLES  # MixSampler: the sum of the components'

H = qtx.operator.Heisenberg(J=[1.0, 0.5], n_neighbor=[1, 2], msr=True)
optimizer = qtx.optimizer.SR(state, H)

for i in range(10):
    samples = sampler.sweep()
    step = optimizer.get_step(samples)
    state.update(step * 1e-3)
    if jax.process_index() == 0:
        print(i, optimizer.energy, flush=True)
