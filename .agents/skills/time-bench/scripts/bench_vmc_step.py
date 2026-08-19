"""Time scaling of the 4 main parts of one VMC step vs the number of nodes.

Times sampling, local energy, Jacobian, and SR solver across VMC steps. The
fp16 network drives the Markov-chain sampling; the fp32 network (TF32 matmul)
drives the local energy, Jacobian, and the SR solver. Run this with the slurm
job set to the target node count and read the per-part means -- they show which
part dominates and how each scales as nodes are added.

Run the batch size scaling (bench_batch_size.py) first to fix `max_parallel`,
then adapt the lattice, network, Hamiltonian, sampler, and `max_parallel` here
to the target simulation. Launch under slurm with multi-node JAX (see the
hpc-usage skill); `jax.distributed.initialize` reads the SLURM_* variables.
"""

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import jax

gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
jax.distributed.initialize(
    num_processes=int(os.environ["SLURM_NTASKS"]),
    process_id=int(os.environ["SLURM_PROCID"]),
    local_device_ids=list(range(gpus_per_node)),
)

from time import perf_counter
import numpy as np
import jax.numpy as jnp
import quantax as qtx

# fp32 uses TF32 matmul (the GPU default, set explicitly here);
# this setting doesn't affect fp16
jax.config.update("jax_default_matmul_precision", "tensorfloat32")

lattice = qtx.sites.Square(10, Nparticles=(50, 50))
H = qtx.operator.Heisenberg(J=[1.0, 0.5], n_neighbor=[1, 2], msr=True)

NSAMPLES = 8192
MAX_PARALLEL = (8192, 2048)  # (forward, backward) per device, from bench_batch_size.py
NREP = 10  # timed steps after 1 compile/warmup step

SYMM = qtx.symmetry.Identity()

# fp32 network for energy/Jacobian/solver; fp16 copy for sampling
model = qtx.model.ResConv(nblocks=8, channels=32, kernel_size=3, dtype=jnp.float32)
state = qtx.state.Variational(model, symm=SYMM, max_parallel=MAX_PARALLEL)

model_fp16 = qtx.utils.filter_tree_map(lambda x: x.astype(jnp.float16), model)
state_fp16 = qtx.state.Variational(model_fp16, symm=SYMM)
del model, model_fp16

sampler = qtx.sampler.SpinExchange(state_fp16, NSAMPLES, n_neighbor=[1, 2])
optimizer = qtx.optimizer.SR(state, H, solver=qtx.optimizer.lsmr())

times = {"sample": [], "Eloc": [], "jacobian": [], "solver": []}
for i in range(NREP + 1):
    t0 = perf_counter()
    samples = sampler.sweep()
    samples = qtx.sampler.Samples(samples.spins)  # drop fp16 logpsi; recompute in fp32
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
    if i == 0:
        continue  # drop the compile/warmup step
    for key, t in zip(times, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)):
        times[key].append(t)
    if jax.process_index() == 0:
        print(i, t1 - t0, t2 - t1, t3 - t2, t4 - t3, flush=True)

if jax.process_index() == 0:
    nodes = jax.process_count()
    header = f"{'nodes':>6}" + "".join(
        f"{key + '_s':>11}" for key in ("sample", "Eloc", "jacobian", "solver", "total")
    )
    means = [np.mean(t) for t in times.values()]
    cols = "".join(f"{ti:>11.3f}" for ti in (*means, sum(means)))
    print(header, flush=True)
    print(f"{nodes:>6}" + cols, flush=True)
