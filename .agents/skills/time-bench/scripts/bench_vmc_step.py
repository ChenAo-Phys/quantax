"""Time scaling of the 4 main parts of one VMC step vs the number of nodes.

Times sampling, local energy, Jacobian, and SR solver across VMC steps, and
reports the peak device memory. All four parts run on the same network in its
native dtype (fp32 with TF32 matmul by default). `nsamples` grows with the
device count (weak scaling), so rerun the script unchanged with the slurm job
set to each target node count and read the per-part means -- they show which
part dominates and how each scales as nodes are added.

Run the batch size scaling (bench_batch_size.py) first to fix `max_parallel`,
then adapt the lattice, network, Hamiltonian, sampler, `max_parallel`,
`NSAMPLES_PER_DEVICE`, and the optimizer class and solver here to the target
simulation. Launch under slurm with multi-node JAX (see the hpc-usage skill).
The initialization below works with one process per node or per GPU; check
that the printed `devices:` equals the allocated GPU count.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

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

from time import perf_counter
import numpy as np
from jax.experimental import multihost_utils
import quantax as qtx

# fp32 uses TF32 matmul (the GPU default, set explicitly here)
jax.config.update("jax_default_matmul_precision", "tensorfloat32")

lattice = qtx.sites.Square(10, Nparticles=(50, 50))
H = qtx.operator.Heisenberg(J=[1.0, 0.5], n_neighbor=[1, 2], msr=True)

NSAMPLES_PER_DEVICE = 2048  # from Step 1 of the time-bench skill
NSAMPLES = NSAMPLES_PER_DEVICE * jax.device_count()  # weak scaling
MAX_PARALLEL = (8192, 2048)  # (forward, backward) per device, from bench_batch_size.py
NREP = 10  # timed steps after 1 compile/warmup step

# Fixed work per solve: rtol=atol=0 stops lsmr only at maxiter, so in weak
# scaling the solver column measures per-iteration cost + communication, not the
# nsamples-dependent iteration count. For the solver share at production
# settings (Step 2), use the production solver instead and set LSMR_ITERS = None.
LSMR_ITERS = 100
SOLVER = qtx.optimizer.lsmr(rtol=0.0, atol=0.0, maxiter=LSMR_ITERS)

SYMM = qtx.symmetry.Identity()

# one network, in its native dtype, for sampling, energy, Jacobian, and solver
model = qtx.model.ResConv(nblocks=8, channels=32, kernel_size=3)
state = qtx.state.Variational(model, symm=SYMM, max_parallel=MAX_PARALLEL)
del model

sampler = qtx.sampler.SpinExchange(state, NSAMPLES, n_neighbor=[1, 2])
assert sampler.nsamples == NSAMPLES  # MixSampler sums its sub-samplers
# use the production optimizer class: e.g. AdamSR solves twice per step
optimizer = qtx.optimizer.SR(state, H, solver=SOLVER)
SOLVES_PER_STEP = 2 if isinstance(optimizer._updater, qtx.optimizer.AdamUpdater) else 1

if jax.process_index() == 0:
    print(
        f"nsamples: {NSAMPLES} ({NSAMPLES_PER_DEVICE}/device), "
        f"nparams: {state.nparams}, solves/step: {SOLVES_PER_STEP}",
        flush=True,
    )

times = {"sample": [], "Eloc": [], "jacobian": [], "solver": []}
for i in range(NREP + 1):
    t0 = perf_counter()
    samples = sampler.sweep()
    jax.block_until_ready(samples.spins)
    t1 = perf_counter()
    Ebar = jax.block_until_ready(optimizer.get_Ebar(samples))
    t2 = perf_counter()
    Obar = jax.block_until_ready(optimizer.get_Obar(samples))
    t3 = perf_counter()
    step, optimizer._buffers = optimizer.solve(Obar, Ebar, optimizer._buffers)
    jax.block_until_ready(step)
    t4 = perf_counter()
    del Ebar, Obar
    dt = (t1 - t0, t2 - t1, t3 - t2, t4 - t3)
    if i == 0:  # drop the compile/warmup step
        if jax.process_index() == 0:
            print(f"warmup done {t4 - t0:.1f} s", flush=True)
            print(f"{'step':>4}" + "".join(f"{k + '_s':>11}" for k in times))
        continue
    for key, ti in zip(times, dt):
        times[key].append(ti)
    if jax.process_index() == 0:
        print(f"{i:>4}" + "".join(f"{ti:>11.3f}" for ti in dt), flush=True)

# Later steps can recompile the local energy (a new connected-config size), so
# drop every step whose total exceeds 1.25x the median.
steps = np.array(list(times.values()))  # (4, NREP)
total = steps.sum(axis=0)
keep = total <= 1.25 * np.median(total)
means = steps[:, keep].mean(axis=1)

# peak memory over all devices of all processes, incl. warmup and compilation
local_peak = max(
    (d.memory_stats() or {}).get("peak_bytes_in_use", 0) for d in jax.local_devices()
)
peak = multihost_utils.process_allgather(np.float32(local_peak / 2**30)).max()

if jax.process_index() == 0:
    print("dropped steps:", (np.flatnonzero(~keep) + 1).tolist(), flush=True)
    result = {"nodes": jax.process_count() // t, "devices": jax.device_count()}
    result.update({key + "_s": ti for key, ti in zip(times, means)})
    result.update({"total_s": means.sum(), "peak_GiB": peak})
    if LSMR_ITERS is not None:  # ms per lsmr iteration, over all solves of a step
        ms_iter = means[3] / (LSMR_ITERS * SOLVES_PER_STEP) * 1e3
        result.update({"lsmr_iters": LSMR_ITERS, "ms/iter": ms_iter})
    print("".join(f"{key:>11}" for key in result), flush=True)
    print("".join(f"{v:>11.4g}" for v in result.values()), flush=True)
