---
name: hpc-usage
description: Submit JAX and Quantax jobs to slurm GPU clusters. Use when running simulations on HPC - writing slurm scripts, setting up multi-node JAX, submitting and monitoring jobs, and debugging hung multi-node jobs or broken cluster environments.
---

# CRITICAL: General rules

- On remote clusters, **NEVER** use `find /` or scan outside the project directory. The HPC filesystem has millions of files and these commands will hang forever.
- **NEVER** touch jobs submitted outside the current project, as they might be handled by other agents. Only cancel jobs you submitted yourself.
- **NEVER** `cp`, `stat` or read the data files (checkpoints, `.npz`, ...) of a RUNNING job on a Lustre scratch filesystem (e.g. NERSC `$SCRATCH`). A hung `cp` there held Lustre locks and blocked the job's next checkpoint write until the job had to be cancelled. Copy outputs once, with a single rsync, after the job's save has finished (check the gpu-out timestamp). Run analysis on the local machine or inside a compute allocation, not on shared login nodes. If a login-node process is stuck in D/I state on scratch, stop touching that path and report it to the site's help desk.
- Choose the **number of nodes** deliberately per job: too many wastes budget, too few delays the run or overflows GPU memory. Scale to the problem size, not the template default.
- Run a small-scale time benchmark before starting a big simulation (see the `time-bench` skill).
- If the compilation at the start of a big multi-node simulation is costly, precompile it in a single-node job (see the `precompile` skill).

# Job scripts

A brief introduction to job scripts is included in the HPC usage section of `tutorials/sharp_bits`. Here is a more detailed explanation.

There are 3 ways to submit JAX jobs to HPC clusters as listed below, each with a slurm template in `scripts/`. The templates take the python script as the first sbatch argument (`$1`), which is how the submit scripts below invoke them: `sbatch <template> <python_script>`.

The templates are probably not directly usable: fill in their `<placeholders>` and check the cluster documentation for required additions (see cluster-specific usage below). Do **NOT** modify the templates in `scripts/`. Copy them into the project directory and modify the copy.

## Option 1: Single-node, only one process

See `scripts/single_node.slurm` for the job script. Nothing else is needed in the Python script. The resource allocation might be tricky if the job doesn't occupy a whole node.

## Option 2: Multi-node, one process per GPU

See `scripts/process_per_gpu.slurm` for the job script, and initialize JAX as in [Distributed initialization](#distributed-initialization). Keep `--gpu-bind=none`: each rank picks its GPU by `SLURM_LOCALID` in the node's full GPU list (a bare `jax.distributed.initialize()` does the same). Any per-task binding (`--gpu-bind=map_gpu:...`/`single`, or the binding a bare `--gpus-per-task` implies) renumbers each rank's GPU to ordinal 0, and ranks 1..n-1 die with `invalid device ordinal`.

## Option 3: Multi-node, one process per node

This is the recommended mode for submitting jobs.

See `scripts/process_per_node.slurm` for the job script, and initialize JAX as in [Distributed initialization](#distributed-initialization). A bare `jax.distributed.initialize()` sets `local_device_ids=[SLURM_LOCALID]`, so every node silently uses only its first GPU.

## Distributed initialization

Run this at the top of the Python script, before any other JAX call. It is correct under both multi-process templates, as it relies on their `--ntasks-per-node` and `--gpu-bind=none`.

```python
import os
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
```

- Check the first stdout line: `devices` must equal the number of GPUs allocated. `SLURM_GPUS_ON_NODE` counts the node's GPUs, not the task's, so the form `local_device_ids=list(range(SLURM_GPUS_ON_NODE))` under one process per GPU makes every rank claim all GPUs of its node, and the job OOMs at the first allocation.
- The snippet trusts `SLURM_NTASKS_PER_NODE`. An `srun` inside a job script or `salloc` shell takes the job's `SLURM_NTASKS` as its `-n`, so a step asking for more tasks per node than the job was allocated silently runs fewer, while `SLURM_NTASKS_PER_NODE` still reports the request and `devices` comes out short. Request each layout when allocating (in `#SBATCH`, or in the hold of [Submission on a compute node](#submission-on-a-compute-node)), not only on the step.
- Drop the `jax_cpu_collectives_implementation` line for multi-process runs on the CPU backend: without gloo, every cross-process CPU computation fails with `Multiprocess computations aren't implemented on the CPU backend`.

## XLA flags

All templates (and `submit_compute.sh`) export the same `XLA_FLAGS`, explained in their comments. Unknown flags are fatal at startup, so check new ones against the cluster's jaxlib, e.g. `JAX_PLATFORMS=cpu XLA_FLAGS="..." python -c "import jax; jax.numpy.ones(1)"` on a login node.

- `--xla_gpu_nccl_termination_timeout_seconds=600` makes a stuck NCCL rendezvous fail after 10 min instead of hanging (see [Monitor jobs](#monitor-jobs)). Raise it for very large jobs: a healthy 1024-GPU clique init took ~9 min.
- Multi-GPU executables with XLA command buffers (CUDA graphs) can crash with `Failed to add memset node to a CUDA graph: CUDA_ERROR_INVALID_VALUE`, typically in the sampler's `jit_batch_forward`, because CUDA graphs cannot capture memsets on `cuda_async`-allocated memory. Append `--xla_gpu_enable_command_buffer=` to `XLA_FLAGS` whenever you set `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` or see this error. It has a cost: every while-loop condition becomes a host sync (a small 2-GPU run with the default allocator ran clean and ~9% faster with command buffers on). Single-GPU jobs are unaffected.
- An OOM at step >= 1 on a single huge allocation, while the total free memory is ample, is allocator fragmentation. Switch the allocator (BFC <-> `cuda_async`) before cutting samples, and retry in a fresh process: a process that has OOMed keeps its leftover buffers.

# Submit jobs

Pick the submit script below matching where the agent is running. As with the slurm templates, do **NOT** modify the submit scripts in `scripts/`: copy the script into the project directory and fill in the `<placeholders>` in its CONFIG block (or override them via environment variables; see each script's header comment for details).

All submit scripts take the same arguments:

```bash
submit_<where>.sh <python_script_path> <job_folder_path>
```

They create the job folder, copy in a timestamped python script (plus the slurm script when applicable), write an `info_<timestamp>.txt` recording the quantax version and `pip list` for reproducibility, then launch the job. The **last line of output** is the job ID (or PID on a compute node).

- The recorded SHA is the HEAD of the quantax clone. For a non-editable install it equals the installed code only if `pip install` was rerun after the last pull (compare `HEAD date` with `installed`). For an editable install, uncommitted edits show up as `-dirty` and in the change list.
- Only the python script (and slurm file) is copied, and the job runs inside the job folder. Input files the script opens by relative path (initial parameters, checkpoints, saved chains) must be copied into the job folder or referenced by absolute path; check that they exist on the cluster before submitting.
- Python puts the script's directory first on `sys.path`. Never put helper files named like stdlib or third-party modules (`bisect.py`, `random.py`, `json.py`, `types.py`, `copy.py`, `signal.py`, `inspect.py`, `jax.py`, `numpy.py`, ...) in a job folder: every run there breaks, e.g. `import jax` fails with `ImportError: cannot import name 'bisect' from 'bisect' (<job_folder>/bisect.py)`. Use prefixed names such as `<project>_diag.py`.

## Submission from the local machine

If the agent is running on a local machine, use `scripts/submit_remote.sh`. It creates everything remotely over ssh/scp and runs `sbatch` on the cluster; the job folder path is a remote path.

It needs an ssh alias with a persistent ControlMaster; `scripts/config` is a template for `~/.ssh/config` (with `ProxyJump`, the master belongs on the jump host's alias).

- On clusters that accept only password + 2FA (no publickey, so `IdentityFile` does nothing), the agent cannot authenticate itself. Ask the user to run `ssh <alias>` once in a terminal; `ControlPersist yes` then keeps the socket for later scripts. Check it with `ssh -O check <alias>`.
- An ssh connect timeout usually means the pinned login node is down: point `HostName` at another login node of the cluster, after which the user must log in interactively once more.
- When the master socket dies, every ssh/scp fails immediately (an authentication error or no output), which looks like a remote-side failure. Long transfer or retry loops must wait until `ssh -o BatchMode=yes <alias> true` succeeds instead of retrying immediately.

## Submission on a login node

If the agent is running on a cluster login node, use `scripts/submit_login.sh`. It runs `sbatch` directly.

## Submission on a compute node

If the agent is running on a compute node, it is already inside a slurm allocation, so there is no `sbatch` and no slurm template. `scripts/submit_compute.sh` launches the python script detached with `nohup` using the allocation's resources and prints the PID. The default launcher is `python -u` (option 1); for the multi-process layouts (options 2 and 3), set `RUN_CMD="srun --cpu-bind=none python -u"`.

To get an allocation for debugging or timing without the agent sitting on the node, hold one with `salloc --no-shell -N <nodes> --ntasks-per-node=<tasks> -t 01:00:00 <cluster args>` or `sbatch -N <nodes> --ntasks-per-node=<tasks> -t 01:00:00 <cluster args> --wrap 'sleep 3600'`, where `<tasks>` is the layout you will run (1 for option 3, the GPUs per node for option 2). Then run commands from the login node with `setsid nohup srun --jobid=<J> --overlap -N <nodes> --ntasks-per-node=<tasks> <cmd> > <log> 2>&1 < /dev/null &`. Pass `-N` and `--ntasks-per-node` to every step: the [Distributed initialization](#distributed-initialization) snippet reads `SLURM_NTASKS_PER_NODE`, which such a step gets only from that option. Without `< /dev/null`, `srun` forwards the caller's stdin to the first task. `scancel <J>` as soon as you are done, because the held node is charged and blocked until then.

# Monitor jobs

- Check queue status with `squeue -u $USER`. After submitting, verify the job starts and inspect the early output for crashes instead of assuming success.
- Output goes to `gpu-out.<job_id>` / `gpu-err.<job_id>` in the job folder (`gpu-out.<timestamp>` / `gpu-err.<timestamp>` for compute-node launches).
- A multi-process job can hang as RUNNING instead of failing: `squeue` keeps showing `R` and gpu-err may stay empty. Seen causes: an OOM (also a compile-time autotuner OOM) on some ranks while the others block in an NCCL rendezvous, an NCCL core/plugin version skew at the first clique init (see [NERSC Perlmutter](#nersc-perlmutter)), and two CUDA PJRT plugins splitting the ranks (see [Troubleshooting](#troubleshooting-broken-environments)). Judge liveness by progress lines in gpu-out (e.g. a per-step print) and their timestamps.
- If no step appears within ~15 min of when the first one is due, dump the stacks on every node BEFORE `scancel`, then fix and resubmit. With `py-spy` installed in the env (activate it first), from the login node: `srun --jobid=<id> --overlap -N <nodes> --ntasks-per-node=1 bash -c 'for p in $(pgrep -u $USER "^python"); do py-spy dump --pid $p; done' < /dev/null`. The templates' NCCL termination timeout catches rendezvous wedges, not every hang (a memory-pressure wedge slipped past it).
- For jobs that already left the queue, `sacct -X -j <job_id>` shows the final state (COMPLETED / FAILED / OOM / TIMEOUT).
- To stop a job: `scancel <job_id>` for submitted jobs, `kill <PID>` for compute-node launches. Only stop jobs you started.

# Troubleshooting broken environments

- **Ranks hang with 0 bytes of stdout, or a 2-rank job dies on a Triton MLIR parse error**, and stderr contains `ALREADY_EXISTS: PJRT_Api already exists for device type cuda`: two CUDA PJRT plugins are installed (e.g. a leftover jax-cuda12 plugin after a CUDA-13 upgrade), and each rank picks one at random. Uninstall the stale pair (`pip uninstall -y jax-cuda12-plugin jax-cuda12-pjrt`); `ls site-packages/jax_plugins` must show exactly one entry.
- **`jax.device_count()` equals the process count, or "cuDNN not found"**: JAX fell back to CPU. The `nvidia-{cudnn,nccl,nvshmem}-cu12`/`-cu13` twins share `site-packages/nvidia/<lib>/`, so uninstalling one deletes the other's files. After removing cu12 twins, run `pip install --force-reinstall --no-deps nvidia-cudnn-cu13==<v> nvidia-nccl-cu13==<v> nvidia-nvshmem-cu13==<v>` (the installed versions), then check that `nvidia/cudnn/lib/libcudnn.so.9` and `nvidia/nccl/lib/libnccl.so.2` exist.
- **The job dies at the first optimizer step with `ncclAlltoAll ... unhandled system error`** after sampling worked: the NCCL 2.27.x grouped send/recv bug (it reproduces on one node). Upgrade the pip NCCL (2.29-2.31 work), except where it must match a site plugin (NERSC).
- **`pip install <anything>` can silently upgrade jax/jaxlib** while leaving the CUDA plugin behind. Pin jax, jaxlib and the plugin in the same command (or use `--no-deps`), then check `pip list | grep -E '^jax|nccl'`. When downgrading CUDA wheels, pin the whole `nvidia-*` set.
- After fixing an env, clear the persistent compilation cache (`JAX_COMPILATION_CACHE_DIR`). After any jax/NCCL/dependency upgrade, rerun a known-good configuration before production: a dependency can add a few GiB and turn a near-capacity run into an OOM.
- **Startup dies with `gloo/transport/tcp/device.cc ... -3 vs 0`** on some ranks and `GetKeyValue() timed out ... cpu:local_topology` on the rest: a transient DNS failure (EAI_AGAIN) in JAX's CPU gloo setup. Resubmit; the `jax_cpu_collectives_implementation` line of [Distributed initialization](#distributed-initialization) avoids it.

# Cluster-specific usage

## NERSC Perlmutter

Include the following lines in the job script.

```bash
#SBATCH --constraint=gpu  # A100 40GB; for 80GB nodes use --constraint='gpu&hbm80g'
#SBATCH --qos=regular  # x1 (x0.75 at >= 128 nodes). premium: starts much sooner, x2 (x4 once 20% of the allocation went to premium), max 5 submitted jobs. debug: 30 min, x1, for smoke tests
#SBATCH --account=<account_name>

...

# Slingshot network: the nccl module supplies NERSC's aws-ofi-nccl (CXI) net plugin.
# Without it, NCCL silently falls back to TCP (~12-25x slower inter-node).
# JAX loads its own pip NCCL (nvidia-nccl-cuXX), so pick the module for jax's CUDA
# major and pin the pip NCCL to the module's version, e.g. for jax cuda13:
#   module load nccl/2.29.2-cu13  +  pip install --no-deps nvidia-nccl-cu13==2.29.2
# nccl/2.24.3 is the cuda12 pairing; it swaps cudatoolkit 13.2 for 12.9.
module load nccl/2.29.2-cu13  # check `module avail nccl` and `pip list | grep nccl`
export FI_MR_CACHE_MONITOR=userfaultfd  # FI_* and NCCL_CROSS_NIC: stability insurance, ~0 throughput effect
export FI_CXI_DEFAULT_CQ_SIZE=131072
export NCCL_CROSS_NIC=1
```

- A pip NCCL newer than the plugin's (2.31.2 against the 2.29.2 module) stochastically wedged the first multi-node clique init: the job stays RUNNING, the GPUs spin at 100%, and nothing is printed. Re-check `pip list | grep nccl` after any jax/pip change, because re-pulling the bundled NCCL brings the wedge back.
- To confirm the RDMA path once, run with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET`. Healthy logs show `Using network AWS Libfabric` and `via NET/AWS Libfabric/{0..3}/GDRDMA`; `NET/Socket` means a TCP fallback. A `NCCL WARN NCCL library version is too old` line (jaxlib built against a newer 2.29.x) is harmless with this pin. Do not leave `NCCL_DEBUG` on in production.
- Walltime is limited to 48 hours (debug 30 min, interactive 4 h). Each node has 4 A100 GPUs.
- Before a large job, estimate the charge = nodes x hours x QOS factor and check the balance with `iris` (the GPU allocation is the `<project>_g` row). A regular job with 127 nodes or fewer pays the full rate, so 128 nodes cost less per node-hour than 120.
- Job folders belong under `$SCRATCH`, not the home directory. `$SCRATCH` is purged: inputs staged there (initial parameters, checkpoints, saved chains) can disappear between runs, so check that they exist before submitting and keep master copies outside scratch.
- To hold a node for debugging or timing (see [Submission on a compute node](#submission-on-a-compute-node)): `salloc --no-shell -q interactive -C 'gpu&hbm80g' -N 1 --ntasks-per-node=<tasks> --gpus-per-node=4 -t 01:00:00 -A <account>`.

## Caltech Resnick

Instead of `#SBATCH --gpus-per-node=<count>`, use `#SBATCH --gres=gpu:<type>:<count>`. For option 2 the verified form is `--ntasks-per-node=<GPUs per node> --gres=gpu:<type>:<GPUs per node> --gpu-bind=none`.
Valid types: p100, v100, h100, nvidia_h200, nvidia_l40s, nvidia_b300_sxm6_ac.
Also include the following lines in the job script.

```bash
#SBATCH --partition=gpu
##SBATCH --reservation=<reservation_name>  # include if there is reservation
```

Jobs in a reservation are killed at its end time (`ResvOverRun=0`), and reservations have been extended silently. Re-check `scontrol show reservation <name>` close to the deadline before sizing `--time` or cutting a run short for the end date, and prefer a stop time the job reads at runtime over a hard-coded one.

There is no `$SCRATCH`; the templates then fall back to `$HOME/jax_cache` for the compilation cache — point `JAX_COMPILATION_CACHE_DIR` elsewhere if the home quota is tight.


## Flatiron Rusty

Include the following lines in the job script.

```bash
#SBATCH --partition=gpu      # or gpuxl, see below
#SBATCH --constraint=<type>  # gpu: h100 (4x SXM5 or 8x PCIe nodes), a100-80gb (4/node); gpuxl: h200 (4 or 8/node), h100 (4x SXM5)
#SBATCH --mem=<N>G           # not --mem=0 on multi-node jobs: it exceeds the per-user QOS memory cap (QOSMaxMemoryPerUser)
```

- The A100-40GB nodes are gone, so `-C a100-40gb` matches no node. `-C h100` matches both the 4-GPU SXM5 and the 8-GPU PCIe nodes, so `--gpus-per-node=8` lands on PCIe nodes only.
- The `gpu` QoS allows 7 days and at most 16 GPUs per user. The `gpuxl` QoS allows at most 3 days and needs at least 4 GPUs per job (smaller requests are rejected with `QOSMinGRES`), so request whole nodes. Queue times differ a lot between node types; gpuxl often starts sooner.
- JAX ships its own CUDA via pip wheels: do not `module load cuda/...` before jax, because it breaks the pip cuSPARSE.

Multi-GPU (sharded) jobs on Rusty H100 nodes have crashed at the first sharded op with `Failed to add memset node to a CUDA graph` (see [XLA flags](#xla-flags)), although a later small 2-GPU run there with the default allocator was clean. Until command buffers are re-tested on your model, disable them in multi-GPU Rusty jobs:

```bash
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_enable_command_buffer="
```
