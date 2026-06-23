#!/bin/bash -l
#SBATCH --nodes=<nodes>
#SBATCH --ntasks-per-node=<gpus_per_node>
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -c <cpus_per_node / gpus_per_node>
#SBATCH --output=gpu-out.%j
#SBATCH --error=gpu-err.%j
#SBATCH --time=<time_of_simulation>

source <path_to_quantax_env>/bin/activate

# JAX compilation cache
export JAX_COMPILATION_CACHE_DIR=${SCRATCH:-$HOME}/jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=10

# shard_autotuning fails on some JAX versions. The fusion/Triton-gemm
# autotuners benchmark kernels on freshly-allocated operand-sized buffers,
# which OOMs when an op touches near-GPU-sized arrays (e.g. a big Jacobian).
# Do NOT use --xla_gpu_autotune_level=0 instead: it slows sampling ~10x.
export XLA_FLAGS="--xla_gpu_shard_autotuning=false --xla_gpu_experimental_enable_fusion_autotuner=false --xla_gpu_enable_triton_gemm=false"

# The python script is passed as the first sbatch argument
srun --cpu-bind=cores python -u "$1"
