"""Time scaling of the direct forward or backward pass vs batch size.

Sweeps the batch size of the network forward `state(s)` or the Jacobian
backward `state.jacobian(s)` for the network in its native dtype (fp32 with
TF32 matmul by default). Adapt the lattice and network to the target
simulation; the per-sample time shows which batch size saturates the GPU.

Run each scan in its own process on a single GPU:

    python bench_batch_size.py forward
    python bench_batch_size.py backward

Use an idle GPU of the same model as the target run, one benchmark process at a
time; on a cluster, request only 1 GPU. Shared or power-capped machines throttle
and can show +-15-30% run-to-run noise, so use them only for smoke tests.
Record the clocks and throttle reasons next to the table:

    nvidia-smi --format=csv --query-gpu=name,clocks.sm,power.draw,power.limit,clocks_throttle_reasons.active
"""

import os
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else None  # one scan per process
if MODE not in ("forward", "backward"):
    sys.exit("usage: python bench_batch_size.py forward|backward")

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

from time import perf_counter
import numpy as np
import jax
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
            # Larger batches OOM too, and a failed attempt can leave the allocator
            # full: re-check a borderline OOM size in a fresh process.
            print(f"{B:>7}  OOM", flush=True)
            break
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


model = qtx.model.ResConv(nblocks=8, channels=32, kernel_size=3)
state = qtx.state.Variational(model)
del model
fn, batch_sizes = {
    "forward": (state, FORWARD_BATCHES),
    "backward": (state.jacobian, BACKWARD_BATCHES),
}[MODE]
print(f"\n{MODE} on {jax.devices()[0].device_kind}, nparams: {state.nparams}")
print(f"{'batch':>7} {'time_ms':>9} {'us/sample':>12}", flush=True)
bench(fn, batch_sizes)
