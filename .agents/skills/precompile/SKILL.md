---
name: precompile
description: Precompile the expensive XLA compilations of a large multi-node VMC run in a cheap single-node job, through the persistent compilation cache. Use when the compilation at the start of a multi-node job is a significant cost, e.g. before a large production run whose model, sampler, and topology are fixed.
---

# What it does

At the start of a run, every process traces and compiles the jitted functions before the first VMC step. For big networks the three expensive compilations take minutes, paid on every node of the job. Precompilation moves them to a single-node job: it compiles the executables for the **target** topology on a compile-only mesh and writes them into the persistent compilation cache. The target run loads them from the cache instead of compiling.

| Jitted function (cache entry name) | Called by | Global batch compiled |
| --- | --- | --- |
| `Metropolis._sweep_variational` (`jit__sweep_variational`) | `sampler.sweep()` | `min(nsamples // ndevices, chunk) * ndevices`, `chunk` = `forward_chunk` (`ref_chunk` for local updates) |
| `Variational._batch_forward` (`jit_batch_forward`) | `state(s)`: local energy, psi refill | `forward_chunk * ndevices` |
| `Variational._grad_vmap` (`jit_chunked_f`) | `state.jacobian` in `SR` / `QNGD` / `MinSR` | `nsamples` |

The sweep includes the wave function, and the local updates for states with `use_ref`. Not covered, compiled in the target run as usual:

- About a hundred small helper jits (each well below 1 s).
- The SR solver (~1 s).
- The local energy of states with local updates (`init_internal`, `segment_ref_forward`).
- Non-`Variational` states.
- The Jacobian batches of `TimeEvol`, which are `backward_chunk * ndevices`. Adapt the template batch in `scripts/precompile.py` if needed.

Fix `max_parallel` and `nsamples` with the `time-bench` skill first: any change of the configuration invalidates the precompiled executables.

# Requirements

The cache key of every executable hashes the following components. A difference in any of them is a cache miss, and the target run silently compiles as usual:

- **computation**: everything defining the computation must be identical in both jobs. This includes the lattice, the model and its dtype, `symm`, `max_parallel`, `nsamples`, the sampler class and its arguments, the default dtype, and `jax_default_matmul_precision`. The Hamiltonian and the optimizer don't enter the covered functions.
- **jaxlib version, CUDA version, XLA_FLAGS**: same environment and the same slurm template (it exports `XLA_FLAGS`).
- **GPU model**: e.g. A100 40GB and 80GB are different. Keep the same `--constraint`.
- **target device count**: the compile-only mesh fixes it.

Also:

- `CACHE_DIR` must be on a filesystem shared by both jobs. Configure it **before importing quantax**: the cache initializes at the first compilation and can't be enabled afterwards.
- The target run must call `qtx.utils.use_portable_compilation_cache()` before the first compilation of the precompiled functions. The precompiling job applies it through `make_precompile_mesh`.
- Only process 0 of a run writes cache entries, but all processes read them.

# Workflow

1. Copy [scripts/precompile.py](scripts/precompile.py) and [scripts/target.py](scripts/target.py) into the project. Fill in `CACHE_DIR` and the target topology (`TARGET_NODES`, `TARGET_GPUS_PER_NODE`). The `CONFIG` blocks must be identical; `make_sampler` in `precompile.py` only changes `nsamples` and drops the thermalization.
2. Submit `precompile.py` as a **single process on one node** of the same cluster, e.g. the `process_per_node.slurm` template of the `hpc-usage` skill with 1 node. One GPU is enough, and more local GPUs also work: the warm-up adapts its batches to the local device count. It prints the new cache entries of the three functions.
3. Submit the target run after the precompiling job has finished. Process 0 prints `cache hit` or `cache MISS` for the three functions. On a miss, see Troubleshooting.

# How it works

These are the pitfalls the scripts handle. Keep them in mind when adapting the scripts.

1. **Compile-only mesh**: `qtx.utils.make_precompile_mesh(P, D)` describes `P` processes × `D` devices without owning them. It also patches two components of the cache key that differ between compile-only and real devices: the topology fingerprint becomes the device model and count, and the platform version becomes that of the real GPU client.
2. **Autotuning warm-up**: cuDNN/GEMM autotuning can't profile on compile-only devices, and an executable compiled with unresolved engine choices crashes at runtime (`CUDNN_STATUS_BAD_PARAM`). The script first runs the forward, the Jacobian, and one sweep on the local devices, with the per-device batches of the target. The autotuning results are recorded in the per-fusion autotune cache inside `CACHE_DIR`, enabled by the default `jax_persistent_cache_enable_xla_caches`, and the ahead-of-time compiles reuse them.
3. **Rebinding to the target mesh**: `state._init_forward(mesh)` and `state._init_backward(mesh)` rebuild the jitted functions, so the trace-time chunking and `shard_map` resolve to the target topology. The sweep picks them up through `state._fulljit_forward`. The state and the sampler can't run computations afterwards, which is why precompilation is a dedicated job.
4. **Ahead-of-time lowering**: `filter_jit(...).lower` treats `jax.ShapeDtypeStruct` as a static leaf, so `aot_compile` goes through the equinox internals (`_preprocess`, `_cached.lower`). It replaces the array leaves by templates sharded on the target mesh. This was checked with equinox 0.13.
5. **Sharding spelling**: the computation hash contains the sharding of the arguments *as spelled*.
   - Arrays from `to_distributed_array` carry the canonical spec `P(("process", "device"))`. `state(s)` reshards its input to this spec.
   - Outputs of jitted functions drop the mesh axes of size 1: `P("device")` with one process, `P("process")` with one GPU per process. These include the spins of a sweep, which feed the next sweep and the Jacobian.
   - The script compiles both spellings when they differ. It calls `jax.clear_caches()` in between, because the in-memory lowering cache treats both spellings as the same sharding and would skip writing the second one.
   - Multi-node runs with ≥2 GPUs per process have a single spelling.
6. **Minimum compile time**: thanks to the warm-up, the ahead-of-time compiles are often shorter than the threshold `jax_persistent_cache_min_compile_time_secs`. The JAX default is 1 s, and job scripts may set it higher. Executables below the threshold are not written, so the script sets the threshold to 0.
7. **Candidates per round**: the number of proposal candidates of the sweep depends on the global batch. The script sets `sampler._n_candidates` to the value of the target.

# Troubleshooting a cache miss

Log the hash of every cache-key component in both jobs and compare the entries of the missing function:

```python
import logging

logger = logging.getLogger("jax._src.cache_key")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())
# prints "get_cache_key hash of serialized <component>: <hash>" per compilation
```

- `computation` differs: the configuration differs, or the sharding spelling of an argument does. To find the difference, wrap `jax._src.cache_key.get` to dump `module.operation.get_asm(large_elements_limit=20)` of the module in both jobs and diff the files.
- `backend version` or `accelerator_config` differs: different CUDA or GPU model, or the target didn't call `use_portable_compilation_cache()`.
- `XLA flags` differs: different `XLA_FLAGS`.
- `compile_options` differs: different device count or partitioning.
