"""Time scaling of the direct forward and backward pass vs batch size.

Sweeps the batch size of the network forward `state(s)` and the Jacobian
backward `state.jacobian(s)` for fp32 (TF32 matmul) and fp16 networks.
Adapt the lattice and network to the target simulation; the per-sample time
shows which batch size saturates the GPU.

Run on a single GPU. On the local machine, pick a free GPU with
`CUDA_VISIBLE_DEVICES=<id> python bench_batch_size.py`. On a cluster, request
only 1 GPU for the test.
"""

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

from time import perf_counter
import numpy as np
import jax
import jax.numpy as jnp
import quantax as qtx

lattice = qtx.sites.Square(10, Nparticles=(50, 50))

FORWARD_BATCHES = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
BACKWARD_BATCHES = [256, 512, 1024, 2048, 4096, 8192, 16384]
NREP = 5  # timed repetitions after 1 compile/warmup call


def bench(fn, batch_sizes):
    for B in batch_sizes:
        s = qtx.utils.rand_states(B)
        try:
            jax.block_until_ready(fn(s))  # compile/warmup
        except jax.errors.JaxRuntimeError:
            print(f"{B:>7}  OOM", flush=True)
            continue
        times = []
        for _ in range(NREP):
            t0 = perf_counter()
            jax.block_until_ready(fn(s))
            times.append(perf_counter() - t0)
        # min over reps, not mean: each rep runs identical work, so the spread is
        # one-sided noise from OS/GPU contention; the min is the cleanest estimate
        # of the true compute time.
        t_min = np.min(times)
        print(f"{B:>7} {t_min * 1e3:>9.2f} {t_min / B * 1e6:>12.3f}", flush=True)


for dtype in (jnp.float32, jnp.float16):
    model = qtx.model.ResConv(nblocks=8, channels=32, kernel_size=3, dtype=dtype)
    state = qtx.state.Variational(model)
    del model
    print(f"\n{dtype.__name__} forward")
    print(f"{'batch':>7} {'time_ms':>9} {'us/sample':>12}", flush=True)
    bench(state, FORWARD_BATCHES)
    if dtype == jnp.float32:  # fp16 backward is never used: Jacobian is cast to fp32
        print(f"\n{dtype.__name__} backward (Jacobian)")
        print(f"{'batch':>7} {'time_ms':>9} {'us/sample':>12}", flush=True)
        bench(state.jacobian, BACKWARD_BATCHES)
