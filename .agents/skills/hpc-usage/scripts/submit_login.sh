#!/bin/bash
# Submit a job to slurm from the cluster's LOGIN node (the agent runs on the
# cluster itself, so no ssh/scp is involved). Project-agnostic.
#
# Usage: submit_login.sh <python_script_path> <folder_path>
#   <python_script_path>  Path to the python script to run.
#   <folder_path>         Folder where the job is set up and run (created if
#                         missing).
#
# Example: submit_login.sh ~/code/test.py $SCRATCH/myrun/test1
#   -> creates $SCRATCH/myrun/test1, copies the script + slurm file in
#      (timestamped), writes info_<ts>.txt there, and sbatches the job.
#
# Before first use, fill in the <placeholders> in the CONFIG block below (or
# override them via environment variables). SLURM_TEMPLATE should point to a
# copy of one of the templates beside this script (single_node.slurm,
# process_per_gpu.slurm, process_per_node.slurm) with its <placeholders> filled
# in for the cluster. The templates take the python script as $1 -- this script
# submits via `sbatch <template> <python_script>`.
#
# Steps:
#   1. mkdir -p the job folder.
#   2. Copy the slurm template + python script into it, timestamped.
#   3. Write info_<ts>.txt (quantax git SHA/describe/status, install time,
#      python env pip list).
#   4. sbatch the job and print the job ID (last line of output).

set -uo pipefail

# ====================== CONFIG: fill in for your cluster ======================
QUANTAX_DIR=${QUANTAX_DIR:-<path_to_quantax_repo>}    # quantax repo (for the git SHA in info_<ts>.txt)
PYTHON_ENV=${PYTHON_ENV:-<path_to_python_env>}        # venv root (for the pip list in info_<ts>.txt)
SLURM_TEMPLATE=${SLURM_TEMPLATE:-<path_to_slurm_template>}  # slurm template taking the python script as $1
# ==============================================================================

case "$QUANTAX_DIR$PYTHON_ENV$SLURM_TEMPLATE" in *'<'*)
    echo "ERROR: fill in the <placeholders> in the CONFIG block of $0 first." >&2
    exit 1
esac

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <python_script_path> <folder_path>" >&2
    exit 1
fi
PY_SRC="$1"
JOB_FOLDER="$2"

[ -f "$PY_SRC" ]         || { echo "Python script not found: $PY_SRC" >&2; exit 1; }
[ -f "$SLURM_TEMPLATE" ] || { echo "Slurm template not found: $SLURM_TEMPLATE" >&2; exit 1; }
# Heuristic only (e.g. ~ vs an absolute path warns falsely): info_<ts>.txt
# records PYTHON_ENV, which should be the env the template activates.
grep -q "$PYTHON_ENV/bin/activate" "$SLURM_TEMPLATE" ||
    echo "WARNING: $SLURM_TEMPLATE may not source $PYTHON_ENV; info_<ts>.txt would record a different env." >&2

TS=$(date +%Y%m%d_%H%M%S)
TS_HUMAN=$(date '+%Y-%m-%dT%H:%M:%S%z')
slurm_base=$(basename "$SLURM_TEMPLATE")          # single_node.slurm
py_base=$(basename "$PY_SRC")                     # test.py
SLURM_NAME="${slurm_base%.slurm}_${TS}.slurm"     # single_node_<ts>.slurm
PY_NAME="${py_base%.py}_${TS}.py"                 # test_<ts>.py
INFO_NAME="info_${TS}.txt"                        # info_<ts>.txt

echo "=== 1. Create job folder: $JOB_FOLDER ==="
mkdir -p "$JOB_FOLDER" || exit 1

echo "=== 2. Copy slurm + python script (timestamp $TS) ==="
cp "$SLURM_TEMPLATE" "$JOB_FOLDER/$SLURM_NAME" || exit 1
cp "$PY_SRC"         "$JOB_FOLDER/$PY_NAME"    || exit 1
echo "  $SLURM_NAME"
echo "  $PY_NAME"

echo "=== 3. Write $INFO_NAME (quantax version + pip list) ==="
SHA=$(git -C "$QUANTAX_DIR" rev-parse HEAD 2>/dev/null)
[ -n "$SHA" ] || echo "WARNING: could not read quantax SHA from $QUANTAX_DIR." >&2
{
    echo "submitted:     $TS_HUMAN"
    echo "python script: $PY_SRC -> $PY_NAME"
    echo "slurm script:  $slurm_base -> $SLURM_NAME"
    echo
    echo "=== quantax ($QUANTAX_DIR) ==="
    echo "SHA:       $SHA"
    echo "describe:  $(git -C "$QUANTAX_DIR" describe --always --dirty 2>/dev/null)"
    echo "HEAD date: $(git -C "$QUANTAX_DIR" log -1 --format=%ci 2>/dev/null)"
    echo "installed: $(stat -c '%y %n' "$PYTHON_ENV"/lib/python3*/site-packages/quantax-*.dist-info 2>/dev/null)"
    echo "uncommitted changes:"
    git -C "$QUANTAX_DIR" status --short 2>/dev/null | head -50
    echo
    echo "=== pip list ($PYTHON_ENV) ==="
    "$PYTHON_ENV/bin/pip" list
} > "$JOB_FOLDER/$INFO_NAME"
echo "  wrote $JOB_FOLDER/$INFO_NAME"

echo
echo "=== 4. Submit ==="
JOBID=$(cd "$JOB_FOLDER" && sbatch --parsable "$SLURM_NAME" "$PY_NAME")
if [ -z "$JOBID" ]; then
    echo "ERROR: sbatch did not return a job ID." >&2
    exit 1
fi
echo "Submitted job in $JOB_FOLDER"
echo "$JOBID"
