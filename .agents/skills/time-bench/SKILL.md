---
name: time-bench
description: Perform time benchmarks to prepare for large-scale simulations. (1) Time cost of various batch sizes. (2) Time cost of 4 main parts in VMC -- sampling, local energy, Jacobian, and SR solver -- on various number of nodes.
---

# Batch size scaling

Batch size is set by `max_parallel` in `quantax.state.Variational` — the maximum batch **per device**. It strongly affects speed and stability:

- Too small: underuses GPU FLOPS.
- Too large: wasteful padding in local energy, and risks memory overflow (worst in the Jacobian).

The optimal batch is the saturation point, where a larger batch no longer lowers the cost per sample. It depends on the GPU, so benchmark on the same GPU the large run will use.

Scan the batch size without symmetry. With `symm` set in `Variational`, time and memory grow by ~`symm.nsymm`, so use `max_parallel // symm.nsymm`.

## Sample code

[scripts/bench_batch_size.py](scripts/bench_batch_size.py) times the forward pass `state(s)` and the Jacobian backward pass `state.jacobian(s)` across batch sizes. Adapt the lattice, network, and batch lists to the target run, then run on a single GPU.

## FLOPS utilization

Record the model FLOPS utilization (MFU) at the optimal forward batch for future reference, where `mfu = forward_flops / forward_time / device_flops`. The snippet below gives the forward FLOPs and is only reliable on **CPU**:

```python
import os
os.environ['JAX_PLATFORMS'] = 'cpu'

...

s = qtx.utils.rand_states(batch_size)
params, static = eqx.partition(model, eqx.is_array)
forward_vmap = jax.vmap(
    lambda params, s: eqx.combine(params, static)(s), in_axes=(None, 0)
)
compiled = jax.jit(forward_vmap).lower(params, s).compile()
analysis = compiled.cost_analysis()
print(analysis)
print(f"{analysis['flops']:.4e}")
```

# Per-part scaling of a VMC step

Do this after fixing the batch size and before large runs. One VMC step has 4 main parts:

- **Sampling** — `sampler.sweep()`: many small forward batches from Markov-chain updates; batch is `min(nsamples // ndevices, forward max_parallel)`. A time bottleneck. fp16 by default.
- **Local energy** — `optimizer.get_Ebar(samples)`: forward passes on all connected configs, padded to a multiple of the forward `max_parallel`. A time bottleneck. fp32 (TF32 matmul) by default.
- **Jacobian** — `optimizer.get_Obar(samples)`: backward passes chunked by the backward `max_parallel`. The main memory bottleneck. fp32/complex64 (TF32 matmul) by default. `state.jacobian` runs under `jax.shard_map` (device-local), so `jac_bwd` must stay **flat** across nodes; if it grows, GSPMD is all-gathering the per-sample conv weight-grad (symmetry vmap + small `backward_chunk`) — check the `all-gather` count in the compiled HLO.
- **SR solver** — `optimizer.solve(Obar, Ebar, optimizer._buffers)`: independent of `max_parallel`; can bottleneck both time and memory. Compare `quantax.optimizer.auto_shift_eig` (fp64/complex128 internally) and `quantax.optimizer.lsmr` (fp32/complex64 internally).

The first step is mostly XLA compile; the bench drops it as warmup. Large models take minutes — set a generous job time and confirm `warmup done` prints. If warmup never finishes, suspect a compile blow-up, not a hang.

Follow the steps below to pick hyperparameters and measure scaling.

## Step 1: nsamples per device

Balance two effects:

- Too few samples → low FLOPS utilization in sampling.
- Too many → memory overflow in the Jacobian.

Follow the following steps:

1. Calculate the largest nsamples per device that doesn't OOM in the Jacobian.
2. For the `auto_shift_eig` solver, divide by 4 for its double-precision internals.
3. If the nsamples per device still underuses FLOPS, raise the effective forward batch with a suitable `symm` in `Variational` (×`symm.nsymm`).

A caveat: In `MixSampler`, `nsamples` is the sum of all `nsamples` in sub-samplers.

## Step 2: Few-node benchmark

Set `nsamples = nsamples_per_device * jax.device_count()` and pick the fewest nodes with `nsamples >= 1024`. Run [scripts/bench_vmc_step.py](scripts/bench_vmc_step.py) there.

The SR solver should take under ~20% of the step. Adding `symm` scales sampling, local energy, and Jacobian by `symm.nsymm` but leaves the solver roughly fixed — so if the solver dominates, raise `symm` (and divide the `model` batch by `symm.nsymm` at the same time), then rerun.

## Step 3: Multi-node benchmark (weak scaling)

Keep `nsamples = nsamples_per_device * jax.device_count()` and rerun [scripts/bench_vmc_step.py](scripts/bench_vmc_step.py) on more nodes (more nodes = more samples).

In weak scaling, sampling, local energy, Jacobian, and peak memory should stay **flat** per device; only the **solver** grows (per-iteration all-reduce of the gradient). Judge scaling by the solver — a non-flat network part is a bug (see the Jacobian note above). The 1→2-node jump is a network-tier step (NVLink→InfiniBand), not steady state. Double the nodes each time; stop once a doubling adds more than ~30% time cost.

# Summarize

Generate `SUMMARY.md` during benchmarks and update it whenever new results become available. It should include the following contents.

- A table of batch-size scan
- Model FLOPS utilization (MFU)
- A table of per-part weak scaling
- All other important details for long-term memory
