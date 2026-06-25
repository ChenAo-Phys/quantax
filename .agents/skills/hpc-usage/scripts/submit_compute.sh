#!/bin/bash
# Run a job from a COMPUTE node the agent is already on. A compute node sits
# inside an existing slurm allocation, so there is no sbatch and no slurm
# template: the python script is launched directly (detached with nohup) using
# the allocation's resources. Project-agnostic.
#
# Usage: submit_compute.sh <python_script_path> <folder_path>
#   <python_script_path>  Path to the python script to run.
#   <folder_path>         Folder where the job is set up and run (created if
#                         missing). Output goes to gpu-out.<ts> / gpu-err.<ts>
#                         there.
#
# Example: submit_compute.sh ~/code/test.py $SCRATCH/myrun/test1
#   -> creates $SCRATCH/myrun/test1, copies the script in (timestamped), writes
#      info.txt there, and launches the job in the background; prints the PID.
#
# Before first use, fill in the <placeholders> in the CONFIG block below (or
# override them via environment variables). RUN_CMD is the launcher prepended
# to the python script:
#   - single process (option 1 in SKILL.md):    python -u           (default)
#   - one process per GPU/node (options 2, 3):  srun --cpu-bind=none python -u
#     (srun inside an allocation reuses that allocation; the process layout is
#     the one the allocation was created with.)
#
# Steps:
#   1. mkdir -p the job folder.
#   2. Copy the python script into it, timestamped.
#   3. Write info.txt (slurm job ID + quantax git SHA + python env pip list).
#   4. Launch detached with nohup and print the PID (last line of output).
#      Monitor with the gpu-out/gpu-err files; stop with kill <PID>.

set -uo pipefail

# ====================== CONFIG: fill in for your cluster ======================
QUANTAX_DIR=${QUANTAX_DIR:-<path_to_quantax_repo>}    # quantax repo (for the git SHA in info.txt)
PYTHON_ENV=${PYTHON_ENV:-<path_to_python_env>}        # venv root to activate
RUN_CMD=${RUN_CMD:-python -u}                         # launcher; see header for multi-process runs
# ==============================================================================

case "$QUANTAX_DIR$PYTHON_ENV" in *'<'*)
    echo "ERROR: fill in the <placeholders> in the CONFIG block of $0 first." >&2
    exit 1
esac

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <python_script_path> <folder_path>" >&2
    exit 1
fi
PY_SRC="$1"
JOB_FOLDER="$2"

[ -f "$PY_SRC" ] || { echo "Python script not found: $PY_SRC" >&2; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
TS_HUMAN=$(date '+%Y-%m-%dT%H:%M:%S%z')
py_base=$(basename "$PY_SRC")                     # test.py
PY_NAME="${py_base%.py}_${TS}.py"                 # test_<ts>.py
INFO_NAME="info_${TS}.txt"                        # info_<ts>.txt

echo "=== 1. Create job folder: $JOB_FOLDER ==="
mkdir -p "$JOB_FOLDER" || exit 1

echo "=== 2. Copy python script (timestamp $TS) ==="
cp "$PY_SRC" "$JOB_FOLDER/$PY_NAME" || exit 1
echo "  $PY_NAME"

echo "=== 3. Write info.txt (allocation + quantax SHA + pip list) ==="
SHA=$(git -C "$QUANTAX_DIR" rev-parse HEAD 2>/dev/null)
[ -n "$SHA" ] || echo "WARNING: could not read quantax SHA from $QUANTAX_DIR." >&2
{
    echo "started:       $TS_HUMAN"
    echo "python script: $PY_SRC -> $PY_NAME"
    echo "run command:   $RUN_CMD"
    echo "slurm job:     ${SLURM_JOB_ID:-<none>} on ${SLURM_JOB_NODELIST:-$(hostname)}"
    echo
    echo "=== quantax ($QUANTAX_DIR) ==="
    echo "SHA: $SHA"
    echo
    echo "=== pip list ($PYTHON_ENV) ==="
    "$PYTHON_ENV/bin/pip" list
} > "$JOB_FOLDER/$INFO_NAME"
echo "  wrote $JOB_FOLDER/$INFO_NAME"

echo
echo "=== 4. Launch ==="
source "$PYTHON_ENV/bin/activate"
# Same environment as the slurm templates beside this script
export JAX_COMPILATION_CACHE_DIR=${SCRATCH:-$HOME}/jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=10
# shard_autotuning fails on some JAX versions; the fusion/Triton-gemm
# autotuners OOM on near-GPU-sized arrays (see the slurm templates).
export XLA_FLAGS="--xla_gpu_shard_autotuning=false --xla_gpu_experimental_enable_fusion_autotuner=false --xla_gpu_enable_triton_gemm=false"
cd "$JOB_FOLDER"
nohup $RUN_CMD "$PY_NAME" > "gpu-out.$TS" 2> "gpu-err.$TS" </dev/null &
PID=$!
echo "Launched job in $JOB_FOLDER (output: gpu-out.$TS, errors: gpu-err.$TS)"
echo "$PID"
