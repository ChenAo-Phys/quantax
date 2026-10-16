---
name: time-bench
description: Perform time benchmarks to prepare for large-scale simulations. (1) Time cost of various batch sizes, and a per-kernel breakdown with Nsight Systems. (2) Time cost of 4 main parts in VMC -- sampling, local energy, Jacobian, and SR solver -- on various number of nodes.
---

# Batch size scaling

Batch size is set by `max_parallel` in `quantax.state.Variational` — the maximum number of input configurations per forward or backward pass **per device**. It strongly affects speed and stability:

- Too small: underuses GPU FLOPS.
- Too large: risks memory overflow (worst in the Jacobian), and pads every local-energy call up to a multiple of the chunk. The padding is minor for a Hamiltonian with many connected configurations per sample, but dominates for operators with only a few (see the `measure` skill).

The optimal batch is the saturation point, where a larger batch no longer lowers the cost per sample. It depends on the GPU, so benchmark on the same GPU model the large run will use, on an otherwise idle GPU, one benchmark at a time. Shared or power-capped machines throttle and can show ±15-30% run-to-run noise: use them only for smoke tests.

Scan the batch size without symmetry. With `symm` set in `Variational`, each input configuration is evaluated `symm.nsymm` times, so time and memory grow by ~`symm.nsymm`: use `max_parallel // symm.nsymm`.

## Sample code

[scripts/bench_batch_size.py](scripts/bench_batch_size.py) times the forward pass `state(s)` or the Jacobian backward pass `state.jacobian(s)` across batch sizes. Adapt the lattice, network, and batch lists to the target run, then run `python bench_batch_size.py forward` and `python bench_batch_size.py backward` as separate processes on a single GPU. Each scan stops at the first OOM. A failed attempt can leave the allocator full, so re-check a borderline OOM size in a fresh process before trusting it.

## FLOPS utilization

Record the model FLOPS utilization (MFU) at the optimal forward batch for future reference, where `mfu = forward_flops / forward_time / device_flops`. `device_flops` is the dense peak of the precision actually used, e.g. the TF32 tensor-core peak for fp32 models with TF32 matmul. MFU is meaningful only from an unthrottled GPU, so record the clocks and throttle reasons with it (step 4 of [Kernel-level profiling](#kernel-level-profiling)); correcting for a sagging clock doesn't rescue it. The snippet below gives the forward FLOPs and is only reliable on **CPU**:

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

## Kernel-level profiling

When MFU is low or a part is slower than expected, find out which GPU kernels dominate, e.g. custom Pallas kernels vs library GEMMs and convolutions, tensor-core (TF32) vs plain fp32 kernels, or many tiny XLA fusions. Use Nsight Systems (`nsys`). Avoid Nsight Compute (`ncu`): it needs access to GPU performance counters, which is often disabled for regular users on shared machines.

1. Open a capture window around the timed loop only, so that compilation and warmup are excluded. JAX has no public hook, so call `cudaProfilerStart/Stop` from the CUDA runtime that JAX has already loaded. Call `cudart()` after the first GPU computation, once the library is loaded.

    ```python
    import ctypes

    def cudart():
        with open("/proc/self/maps") as f:
            paths = {line.split()[-1] for line in f if "libcudart.so" in line}
        return ctypes.CDLL(paths.pop())

    jax.block_until_ready(state(s))  # compile outside the window
    rt = cudart()
    rt.cudaProfilerStart()
    for _ in range(10):
        psi = state(s)
    jax.block_until_ready(psi)
    rt.cudaProfilerStop()
    ```

2. Run under `nsys` with graph-node tracing and summarize the kernels:

    ```bash
    nsys profile --trace=cuda --cuda-graph-trace=node \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        -o prof python bench.py
    nsys stats --report cuda_gpu_kern_sum --format csv prof.nsys-rep
    ```

    XLA launches parts of the program as CUDA graphs (command buffers). Without `--cuda-graph-trace=node`, the kernels inside the graphs are silently missing from the table, which still looks plausible. In a test VMC step, 33 of 92 kernels were missing. For custom grouping, query the `prof.sqlite` written by `nsys stats`: table `CUPTI_ACTIVITY_KIND_KERNEL` (`start`, `end`, `registersPerThread`, ...) joined with `StringIds` on `shortName`.

3. Group the kernels by name. Pallas kernels appear under their Python function name. XLA-generated kernels are named after the fused op (`loop_*_fusion*`, `input_*_fusion*`, `triton_*`). Library kernels (cuBLAS, cuDNN) follow the library naming, which often encodes the precision (e.g. `tf32`, `sgemm`). Names vary across GPU architectures and library versions, so derive the groups from the actual report and keep an "other" bucket.

4. Read the times with care:
    - Tracing adds overhead to every launch, so wall time under `nsys` can be many times longer, but kernel durations stay usable. Compare the kernel-time total with the unprofiled wall time of the same loop. The gap is host work, launch overhead, and GPU idle time.
    - Kernel times depend on the clocks. Profile on an otherwise idle GPU, and record the clocks and the clock-event (throttle) reasons (`nvidia-smi -q -d CLOCK,PERFORMANCE`): power or thermal throttling and other processes on the same GPU change kernel times.

# Per-part scaling of a VMC step

Do this after fixing the batch size and before large runs. One VMC step has 4 main parts:

- **Sampling** — `sampler.sweep()`: many small forward batches from Markov-chain updates; batch is `min(nsamples // ndevices, chunk)`, where chunk is the forward `max_parallel`, or the ref chunk (third entry of `max_parallel`) for states with local updates (`use_ref`). A time bottleneck. Same dtype as the model (fp32, TF32 matmul, by default).
- **Local energy** — `optimizer.get_Ebar(samples)`: forward passes on all connected configs, padded to a multiple of the forward `max_parallel`. A time bottleneck. fp32 (TF32 matmul) by default.
- **Jacobian** — `optimizer.get_Obar(samples)`: backward passes chunked by the backward `max_parallel`. The main memory bottleneck. fp32/complex64 (TF32 matmul) by default. `state.jacobian` runs under `jax.shard_map` (device-local), so the `jacobian` time must stay **flat** across nodes; if it grows, GSPMD is all-gathering the per-sample conv weight-grad (symmetry vmap + small `backward_chunk`) — check the `all-gather` count in the compiled HLO.
- **SR solver** — `optimizer.solve(Obar, Ebar, optimizer._buffers)`: independent of `max_parallel`; can bottleneck both time and memory. Compare `quantax.optimizer.auto_shift_eig` (fp64/complex128 internally; the default when `solver=` is omitted) and `quantax.optimizer.lsmr` (fp32/complex64 internally). `auto_shift_eig` doesn't weak-scale in its MinSR branch (rows < nparams, where rows = nsamples, or 2 nsamples for non-holomorphic states stacked as [Re; Im]): every device holds the all-reduced Gram matrix, rows² × 8 B for real states, ×2 for complex (8.6 GB at 32768 real rows, 34 GB at 65536). For non-holomorphic states, its solve was measured to hold ~1.5x the full f64 (2 nsamples × nparams) matrix on every device, so more nodes never fix that OOM. Benchmark it at the production nsamples instead of extrapolating, and prefer `lsmr` at scale.

The first step is mostly XLA compile; the bench drops it as warmup and prints `warmup done`. Large models take minutes, so set a generous job time. Later steps can recompile too: `get_Ebar` passes the connected-config count, rounded up to a multiple of the forward chunk, as a static argument, so a step whose count lands on a new multiple re-jits the local energy. The bench drops every step whose total exceeds 1.25x the median and prints the `dropped steps`. A mid-run Eloc spike with flat sample and solver times is such a recompile, not network trouble; a larger forward chunk gives fewer distinct sizes.

If warmup never finishes on a multi-process run, suspect a compile blow-up or a wedge. A near-capacity OOM on one rank (Jacobian, solve, or the compile-time autotuner, which allocates operand-sized scratch buffers) leaves the other ranks waiting in NCCL, and the job stays RUNNING with no output. Check stderr for `RESOURCE_EXHAUSTED`, `Autotuning failed` or `Can't reduce memory use`, and dump the stacks (e.g. `py-spy dump`) before cancelling. Size with margin (Step 1). If only the solve fails while compiling, turn autotuning off for that jit alone:

```python
solve = jax.jit(
    lambda O, E, b: optimizer._updater.update(optimizer.solve_equation, O, E, b),
    donate_argnums=(0, 1, 2),
    compiler_options={"xla_gpu_autotune_level": 0},
)
step, optimizer._buffers = solve(Obar, Ebar, optimizer._buffers)
```

Never set `--xla_gpu_autotune_level=0` globally: it de-tunes the convolutions, slows sampling 10x or more, and invalidates the timing.

Follow the steps below to pick hyperparameters and measure scaling.

## Step 1: nsamples per device

Balance two effects:

- Too few samples → low FLOPS utilization in sampling.
- Too many → memory overflow in the Jacobian and the SR solve.

Follow the following steps:

1. Compute the per-device fp32 Jacobian J = (nsamples / ndevices) × nparams × 4 B (×2 for complex). The activations of one backward chunk come on top of it. The SR solve, not the Jacobian, usually sets the ceiling.
2. `lsmr`: budget ~2J + 2 GB. Beyond J, the solve itself needs only a few nparams-sized vectors (~1-2 GB at ~10^8 params), but some jax versions lower its matvec as a split-K GEMM that materializes a padded copy of the Jacobian unless nparams % 4096 == 0 or `XLA_FLAGS` contains `--xla_gpu_enable_triton_gemm=false` (set in the hpc-usage templates), and the compile-time autotuner can ask for a second J.
3. `auto_shift_eig`: its f64 internals cost ~4J on top of the resident J, plus the Gram matrix (rows², see the SR solver bullet above) and autotuner scratch. Keep J ≲ 0.15 × device memory (measured: the f64 solve ran at ~1/6 and OOMed at ~1/5 of the samples per device that J alone allows). Never pass `dtype=jnp.float32`: T = A A† squares the condition number. If f64 doesn't fit, use `lsmr`, more devices, or fewer samples.
4. Choose the backward `max_parallel` to divide nsamples / ndevices. Otherwise the last chunk is padded and the output sliced back with a full copy, ~2x the Jacobian peak. (A chunk ≥ nsamples / ndevices runs unchunked: no copy, but the activations of all samples at once.)
5. Verify the chosen size with a short run of [scripts/bench_vmc_step.py](scripts/bench_vmc_step.py) with the production optimizer and solver (set `SOLVER` and `LSMR_ITERS = None`) at the production `XLA_PYTHON_CLIENT_MEM_FRACTION` (it prints `peak_GiB`), and keep margin: a marginal OOM on one rank of a multi-process job can hang it as RUNNING instead of failing.
6. If the nsamples per device still underuses FLOPS, raise the effective forward batch with a suitable `symm` in `Variational` (×`symm.nsymm`).

A caveat: In `MixSampler`, `nsamples` is the sum of all `nsamples` in sub-samplers.

## Step 2: Few-node benchmark

Set `NSAMPLES_PER_DEVICE` in [scripts/bench_vmc_step.py](scripts/bench_vmc_step.py), which uses `nsamples = NSAMPLES_PER_DEVICE * jax.device_count()`, and pick the fewest nodes with `nsamples >= 1024`. Run it there, and check that the printed `devices:` equals the allocated GPU count.

Time the optimizer the run will use. `AdamSR` calls the solver twice per step, the second time with a diagonal preconditioner (measured ~2x the lsmr iterations at fixed rtol), so an `SR` bench underestimates its solver share ~2-3x. The solve can also lower differently and need different memory, so probe peak memory with the real optimizer too.

The SR solver should take under ~20% of the step; judge this in a second run with the production solver settings (set `SOLVER`, e.g. `lsmr()` with its default tolerance, and `LSMR_ITERS = None`), not the fixed-iteration solver of Step 3. Adding `symm` scales sampling, local energy, and Jacobian by `symm.nsymm` but leaves the solver roughly fixed — so if the solver dominates, raise `symm` (and divide `max_parallel` by `symm.nsymm` at the same time), then rerun.

## Step 3: Multi-node benchmark (weak scaling)

Rerun [scripts/bench_vmc_step.py](scripts/bench_vmc_step.py) with the fixed-work solver (the script default), changing only the node count: `nsamples` grows with the device count.

In weak scaling, sampling, local energy, Jacobian, and peak memory (with `lsmr`; see the solver note above) should stay **flat** per device; only the **solver** grows. Its time is iterations × (local matvecs + one all-reduce), and the lsmr iteration count to reach a tolerance itself grows with nsamples (measured ~nsamples^0.4 at fixed rtol, ~nsamples^0.75 at fixed accuracy, state-dependent; see the `lsmr` docstring for how its tolerance scales). So the bench runs a fixed-work `lsmr(rtol=0.0, atol=0.0, maxiter=LSMR_ITERS)`, which always takes `maxiter` iterations, and prints `ms/iter`, the solver time divided by all lsmr iterations of the step (`AdamSR` runs 2 solves of `LSMR_ITERS` each): apply the ~30% rule below to that. Estimate the production solve cost separately, by running the production solver once at the target nsamples.

Before reading the solver column as communication cost, confirm the inter-node NCCL transport: run once with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` and check that the log shows the fabric transport (e.g. `NET/IB`, or a libfabric/OFI plugin with GPUDirect RDMA), not `NET/Socket`. A silent TCP fallback makes each all-reduce ~7-25x slower.

Judge scaling by the solver — a non-flat network part is a bug (see the Jacobian note above). The 1→2-node jump is a network-tier step (NVLink → inter-node fabric, ~10x lower per-GPU bandwidth), not steady state. Double the nodes each time; stop once a doubling adds more than ~30% time cost.

# Summarize

Generate `SUMMARY.md` during benchmarks and update it whenever new results become available. It should include the following contents.

- A table of batch-size scan, with the GPU clocks and throttle reasons
- Model FLOPS utilization (MFU)
- A table of per-part weak scaling, with `peak_GiB`, the lsmr iterations per solve, and the dropped steps
- The software environment: quantax SHA and `pip list` (e.g. the `info_<timestamp>.txt` written by the hpc-usage submit scripts). To extend an old benchmark, rebuild that environment instead of using the live one.
- All other important details for long-term memory
