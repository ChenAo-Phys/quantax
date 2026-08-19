---
name: hpc-usage
description: Submit JAX and Quantax jobs to slurm GPU clusters. Use when running simulations on HPC - writing slurm scripts, setting up multi-node JAX, and submitting jobs.
---

# CRITICAL: General rules

- On remote clusters, **NEVER** use `find /` or scan outside the project directory. The HPC filesystem has millions of files and these commands will hang forever.
- **NEVER** touch jobs submitted outside the current project, as they might be handled by other agents. Only cancel jobs you submitted yourself.
- Choose the **number of nodes** deliberately per job: too many wastes budget, too few delays the run or overflows GPU memory. Scale to the problem size, not the template default.
- Run a small-scale time benchmark before starting a big simulation (see the `time-bench` skill).

# Connection

Consider first where the agent is running - a local machine, an HPC login node, or an HPC compute node. If connection from the local machine to a remote cluster is needed, ask the user how to connect to the HPC cluster.

To avoid repeated two-factor authentication in ssh, consider using the template `scripts/config` to set up ControlMaster in the user's SSH configuration file. **NEVER** make any change to the SSH configuration without asking for the user's permission.

# Job scripts

A brief introduction to job scripts is included in the HPC usage section of `tutorials/sharp_bits`. Here is a more detailed explanation.

There are 3 ways to submit JAX jobs to HPC clusters as listed below, each with a slurm template in `scripts/`. The templates take the python script as the first sbatch argument (`$1`), which is how the submit scripts below invoke them: `sbatch <template> <python_script>`.

The templates are probably not directly usable: fill in their `<placeholders>` and check the cluster documentation for required additions (see cluster-specific usage below). Do **NOT** modify the templates in `scripts/`. Copy them into the project directory and modify the copy.

## Option 1: Single-node, only one process

See `scripts/single_node.slurm` for the job script. Nothing else is needed in the Python script. The resource allocation might be tricky if the job doesn't occupy a whole node.

## Option 2: Multi-node, one process per GPU

See `scripts/process_per_gpu.slurm` for the job script. In the Python script, call `jax.distributed.initialize()` right after importing JAX. This will automatically set up the distributed environment and allow JAX to use all available GPUs across nodes.

## Option 3: Multi-node, one process per node

This is the recommended mode for submitting jobs.

See `scripts/process_per_node.slurm` for the job script. The default arguments of `jax.distributed.initialize` often fail in this scenario. In the beginning of the Python script, run the initialization with the following arguments. This will allow JAX to use all available GPUs across nodes.

```python
import os
import jax

gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])

jax.distributed.initialize(
    num_processes=int(os.environ["SLURM_NTASKS"]),
    process_id=int(os.environ["SLURM_PROCID"]),
    local_device_ids=list(range(gpus_per_node)),
)
```

The `jaxmg` package can **NOT** be used under this mode.

# Submit jobs

Pick the submit script below matching where the agent is running. As with the slurm templates, do **NOT** modify the submit scripts in `scripts/`: copy the script into the project directory and fill in the `<placeholders>` in its CONFIG block (or override them via environment variables; see each script's header comment for details).

All submit scripts take the same arguments:

```bash
submit_<where>.sh <python_script_path> <job_folder_path>
```

They create the job folder, copy in a timestamped python script (plus the slurm script when applicable), write an `info.txt` recording the quantax SHA and `pip list` for reproducibility, then launch the job. The **last line of output** is the job ID (or PID on a compute node).

## Submission from the local machine

If the agent is running on a local machine, use `scripts/submit_remote.sh`. It creates everything remotely over ssh/scp and runs `sbatch` on the cluster; the job folder path is a remote path.

## Submission on a login node

If the agent is running on a cluster login node, use `scripts/submit_login.sh`. It runs `sbatch` directly.

## Submission on a compute node

If the agent is running on a compute node, it is already inside a slurm allocation, so there is no `sbatch` and no slurm template. `scripts/submit_compute.sh` launches the python script detached with `nohup` using the allocation's resources and prints the PID. The default launcher is `python -u` (option 1); for the multi-process layouts (options 2 and 3), set `RUN_CMD="srun --cpu-bind=none python -u"`.

# Monitor jobs

- Check queue status with `squeue -u $USER`. After submitting, verify the job starts and inspect the early output for crashes instead of assuming success.
- Output goes to `gpu-out.<job_id>` / `gpu-err.<job_id>` in the job folder (`gpu-out.<timestamp>` / `gpu-err.<timestamp>` for compute-node launches).
- For jobs that already left the queue, `sacct -X -j <job_id>` shows the final state (COMPLETED / FAILED / OOM / TIMEOUT).
- To stop a job: `scancel <job_id>` for submitted jobs, `kill <PID>` for compute-node launches. Only stop jobs you started.

# Cluster-specific usage

## NERSC Perlmutter

Include the following lines in the job script.

```bash
#SBATCH --constraint=gpu  # A100 40GB; for 80GB nodes use --constraint='gpu&hbm80g'
#SBATCH --qos=regular  # premium starts much sooner but charges 2x and allows at most 5 queued jobs
#SBATCH --account=<account_name>

...

# Slingshot network for NCCL: AWS OFI plugin + libfabric/CXI tunings.
# Without the module, NCCL silently falls back to TCP (~12-25x slower inter-node).
module load nccl/2.24.3
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
export NCCL_CROSS_NIC=1
```

Walltime is limited to 48 hours. Each node has 4 A100 GPUs. Job folders belong under `$SCRATCH`, not the home directory.

## Caltech Resnick

Instead of `#SBATCH --gpus-per-node=<count>`, use `#SBATCH --gres=gpu:<type>:<count>`.
Valid types: p100, v100, h100, nvidia_h200, nvidia_l40s, nvidia_b300_sxm6_ac.
Also include the following lines in the job script.

```bash
#SBATCH --partition=gpu
##SBATCH --reservation=<reservation_name>  # include if there is reservation
```

There is no `$SCRATCH`; the templates then fall back to `$HOME/jax_cache` for the compilation cache — point `JAX_COMPILATION_CACHE_DIR` elsewhere if the home quota is tight.


## Flatiron Rusty

Include the following lines in the job script.

```bash
#SBATCH --partition=gpu  # gpuxl might be available for more advanced GPUs
#SBATCH --constraint=<type>  # gpu: h100, a100-80gb, a100-40gb. gpuxl: h200
```

For any **multi-GPU (sharded)** job on Rusty H100 nodes, also disable XLA command
buffers, or the first sharded op crashes with `CUDA error: Failed to add memset node
to a CUDA graph: CUDA_ERROR_INVALID_VALUE`:

```bash
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_enable_command_buffer="
```

Single-GPU jobs are unaffected and don't need this flag.
