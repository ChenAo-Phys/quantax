#!/bin/bash -l
#SBATCH --nodes=<nodes>
#SBATCH --ntasks-per-node=<gpus_per_node>
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --exclusive
#SBATCH --mem=0  # all node memory; use --mem=<N>G where it exceeds a per-user memory cap
#SBATCH -c <cpus_per_node / gpus_per_node>
#SBATCH --output=gpu-out.%j
#SBATCH --error=gpu-err.%j
#SBATCH --time=<time_of_simulation>

source <path_to_quantax_env>/bin/activate

# JAX compilation cache
export JAX_COMPILATION_CACHE_DIR=${SCRATCH:-$HOME}/jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=10

# XLA flags (unknown flags are fatal: check new ones against the installed jaxlib)
# - shard_autotuning fails on some JAX versions.
# - The fusion/Triton-gemm autotuners benchmark kernels on freshly-allocated
#   operand-sized buffers, which OOMs when an op touches near-GPU-sized arrays
#   (e.g. a big Jacobian). Do NOT use a global --xla_gpu_autotune_level=0 instead:
#   it de-tunes convolutions and slowed sampling 10-70x. On jax >= 0.10 the GEMM
#   autotuner (incl. cuBLASLt) also profiles on fresh operand-sized buffers at
#   COMPILE time, which these flags don't stop (--xla_gpu_enable_cublaslt=false
#   was ignored on jax 0.10.2). If a near-capacity jit (e.g. the SR solve on a
#   large Jacobian) fails to compile with 'Autotuning failed' / 'Failed to
#   profile configs', disable autotuning for that jit only (verified for the
#   lsmr solve; expected but untested for the f64 cuBLASLt GEMM of *_eig solvers):
#     jax.jit(f, compiler_options={"xla_gpu_autotune_level": 0})
# - A stuck NCCL rendezvous aborts after 10 min instead of hanging; raise the
#   timeout for very large jobs (a healthy 1024-GPU clique init took ~9 min).
export XLA_FLAGS="--xla_gpu_shard_autotuning=false --xla_gpu_experimental_enable_fusion_autotuner=false --xla_gpu_enable_triton_gemm=false --xla_gpu_nccl_termination_timeout_seconds=600"

# The python script is passed as the first sbatch argument
srun --cpu-bind=cores python -u "$1"
