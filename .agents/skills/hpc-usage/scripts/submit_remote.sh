#!/bin/bash
# Submit a job to a slurm GPU cluster from the LOCAL machine. Project-agnostic:
# it touches NO local files or git -- it creates everything REMOTELY over
# ssh/scp and submits.
#
# Usage: submit_remote.sh <python_script_path> <remote_folder_path>
#   <python_script_path>  LOCAL path to the python script to run.
#   <remote_folder_path>  Folder ON the cluster where the job is set up and run
#                         (absolute, or relative to the remote home dir).
#
# Example: submit_remote.sh ~/code/test.py scratch/myrun/test1
#   -> creates scratch/myrun/test1 on the cluster, scps the script + slurm file
#      in (timestamped), writes info.txt there, and sbatches the job.
#
# Before first use, fill in the <placeholders> in the CONFIG block below (or
# override them via environment variables). SLURM_TEMPLATE should point to a
# copy of one of the templates beside this script (single_node.slurm,
# process_per_gpu.slurm, process_per_node.slurm) with its <placeholders> filled
# in for the cluster. The templates take the python script as $1 -- this script
# submits via `sbatch <template> <python_script>`.
#
# Steps:
#   1. ssh: mkdir -p the remote folder.
#   2. scp the slurm template + python script into it, timestamped.
#   3. ssh: write info.txt REMOTELY (quantax git SHA + python env pip list).
#   4. ssh: sbatch the job and print the job ID (last line of output).
#
# ssh note: the sbatch job-ID is captured via a TEMP FILE (stdin from /dev/null)
# rather than `out=$(ssh ...)`. With ssh ControlMaster, the master keeps a copy
# of a command-substitution pipe fd that leaks into a later pipeline and
# deadlocks; a regular file avoids that.

set -uo pipefail

# ====================== CONFIG: fill in for your cluster ======================
REMOTE=${REMOTE:-<ssh_alias_of_cluster>}                   # ssh destination, e.g. an alias in ~/.ssh/config
QUANTAX_DIR=${QUANTAX_DIR:-<path_to_quantax_repo_on_cluster>}  # remote quantax repo (for the git SHA in info.txt)
PYTHON_ENV=${PYTHON_ENV:-<path_to_python_env_on_cluster>}  # remote venv root (for the pip list in info.txt)
SLURM_TEMPLATE=${SLURM_TEMPLATE:-<path_to_local_slurm_template>}  # LOCAL slurm template taking the python script as $1
# ==============================================================================

case "$REMOTE$QUANTAX_DIR$PYTHON_ENV$SLURM_TEMPLATE" in *'<'*)
    echo "ERROR: fill in the <placeholders> in the CONFIG block of $0 first." >&2
    exit 1
esac

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <python_script_path> <remote_folder_path>" >&2
    exit 1
fi
PY_SRC="$1"
REMOTE_FOLDER="$2"

[ -f "$PY_SRC" ]         || { echo "Python script not found: $PY_SRC" >&2; exit 1; }
[ -f "$SLURM_TEMPLATE" ] || { echo "Slurm template not found: $SLURM_TEMPLATE" >&2; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
TS_HUMAN=$(date '+%Y-%m-%dT%H:%M:%S%z')           # readable timestamp for info.txt (space-free: survives ssh arg flattening)
slurm_base=$(basename "$SLURM_TEMPLATE")          # single_node.slurm
py_base=$(basename "$PY_SRC")                     # test.py
SLURM_NAME="${slurm_base%.slurm}_${TS}.slurm"     # single_node_<ts>.slurm
PY_NAME="${py_base%.py}_${TS}.py"                 # test_<ts>.py
INFO_NAME="info_${TS}.txt"                        # info_<ts>.txt

echo "=== 1. Create remote folder on $REMOTE: $REMOTE_FOLDER ==="
ssh "$REMOTE" "mkdir -p \"$REMOTE_FOLDER\"" </dev/null || exit 1

echo "=== 2. Copy slurm + python script (timestamp $TS) ==="
scp -q "$SLURM_TEMPLATE" "$REMOTE:$REMOTE_FOLDER/$SLURM_NAME" || exit 1
scp -q "$PY_SRC"         "$REMOTE:$REMOTE_FOLDER/$PY_NAME"    || exit 1
echo "  $SLURM_NAME"
echo "  $PY_NAME"

echo "=== 3. Write info.txt remotely (quantax SHA + pip list) ==="
ssh "$REMOTE" bash -s -- \
    "$QUANTAX_DIR" "$PYTHON_ENV" "$REMOTE_FOLDER/$INFO_NAME" "$TS" "$PY_NAME" "$slurm_base" "$SLURM_NAME" "$TS_HUMAN" \
    </dev/null <<'EOF'
QDIR="$1"; QENV="$2"; INFO="$3"; TS="$4"; PY_NAME="$5"; SLURM_BASE="$6"; SLURM_NAME="$7"; TS_HUMAN="$8"
SHA=$(git -C "$QDIR" rev-parse HEAD 2>/dev/null)
[ -n "$SHA" ] || echo "WARNING: could not read quantax SHA from $QDIR." >&2
{
    echo "submitted:     $TS_HUMAN"
    echo "python script: -> $PY_NAME"
    echo "slurm script:  $SLURM_BASE -> $SLURM_NAME"
    echo
    echo "=== quantax ($QDIR) ==="
    echo "SHA: $SHA"
    echo
    echo "=== pip list ($QENV) ==="
    "$QENV/bin/pip" list
} > "$INFO"
EOF
echo "  wrote $REMOTE_FOLDER/$INFO_NAME"

echo
echo "=== 4. Submit on $REMOTE ==="
jobtmp=$(mktemp "${TMPDIR:-/tmp}/submit_job.XXXXXX")
ssh "$REMOTE" \
    "cd \"$REMOTE_FOLDER\" && sbatch --parsable \"$SLURM_NAME\" \"$PY_NAME\"" \
    </dev/null > "$jobtmp"
JOBID=$(tr -d '[:space:]' < "$jobtmp")
rm -f "$jobtmp"

if [ -z "$JOBID" ]; then
    echo "ERROR: sbatch did not return a job ID." >&2
    exit 1
fi
echo "Submitted job in $REMOTE_FOLDER"
echo "$JOBID"
