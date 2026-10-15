"""A target VMC run loading the executables written by precompile.py.

Compared with an ordinary run, only two things are added: the persistent
compilation cache CACHE_DIR, configured before importing quantax, and
`use_portable_compilation_cache()`, called before the first compilation of the
precompiled functions. The CONFIG block must be identical to precompile.py.

Submitted with process_per_node.slurm of the hpc-usage skill. Process 0 prints
whether the precompiled functions are found in the cache.
"""

import os

CACHE_DIR = "<shared path>/jax_cache"  # same as precompile.py

import jax

gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
jax.distributed.initialize(
    num_processes=int(os.environ["SLURM_NTASKS"]),
    process_id=int(os.environ["SLURM_PROCID"]),
    local_device_ids=list(range(gpus_per_node)),
)
jax.config.update("jax_compilation_cache_dir", CACHE_DIR)

import logging
import numpy as np
import quantax as qtx

qtx.utils.use_portable_compilation_cache()

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

H = qtx.operator.Heisenberg(J=[1.0, 0.5], n_neighbor=[1, 2], msr=True)
optimizer = qtx.optimizer.SR(state, H)

for i in range(10):
    samples = sampler.sweep()
    step = optimizer.get_step(samples)
    state.update(step * 1e-3)
    if jax.process_index() == 0:
        print(i, optimizer.energy, flush=True)
