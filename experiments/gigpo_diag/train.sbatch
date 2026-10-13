#!/usr/bin/env bash
#SBATCH -A prjsnnn0000269
#SBATCH --job-name=ggp-app7b-train
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=17:50:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/ggp-appworld-7b-train-%j.out
export GIGPO_ENV=appworld
export GIGPO_VAL_BATCH=16
set -euo pipefail
module purge
module load python/3.12.5 cuda/12.6.2
source experiments/gigpo_diag/common.sh
source experiments/gigpo_diag/nvidia_job_env.sh
test -f "$GIGPO_VENV/.setup-complete"
source "$GIGPO_VENV/bin/activate"
python experiments/gigpo_diag/check_python_environment.py --pip-version 25.3
export GIGPO_CODE_COMMIT="$(git rev-parse HEAD)"
export GIGPO_RUN_ID="${GIGPO_RUN_ID:-${GIGPO_ENV}-qwen25-7b-gigpo-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)}"
export GIGPO_RUN_DIR="$ORION_ROOT/logs/gigpo-reproduction/$GIGPO_ENV/$GIGPO_RUN_ID"
mkdir -p "$GIGPO_RUN_DIR"
export GIGPO_DIAG_DIR="$GIGPO_RUN_DIR/diagnostics"
export SWANLAB_LOG_DIR="$GIGPO_RUN_DIR/swanlog"
export APPWORLD_ROOT="$GIGPO_RUN_DIR/appworld"
export APPWORLD_PORTS_FILE="$GIGPO_RUN_DIR/appworld_ports.ports"
export GIGPO_DATA_DIR="$ORION_ROOT/data/gigpo-reproduction/$GIGPO_ENV/placeholders"
export GIGPO_NUM_GPUS=2
export GIGPO_CPUS="${SLURM_CPUS_PER_TASK:-32}"
# Isolate third-party caches without replacing the shared user's HOME.
export XDG_CACHE_HOME="$ORION_ROOT/.cache/gigpo-$GIGPO_ENV"
mkdir -p "$XDG_CACHE_HOME"
on_exit() {
  local result=$?
  trap - EXIT
  python -m experiments.gigpo_diag.analyze "$GIGPO_RUN_DIR" --exit-code "$result" || true
  echo "GIGPO_TRAIN_END env=$GIGPO_ENV job=$SLURM_JOB_ID exit=$result output=$GIGPO_RUN_DIR"
  exit "$result"
}
trap on_exit EXIT
printf '%s\n' "bash experiments/gigpo_diag/launch_reference.sh vllm" > "$GIGPO_RUN_DIR/actual_command.txt"
cp "$GIGPO_VENV/gigpo-pip-freeze.txt" "$GIGPO_RUN_DIR/pip-freeze.txt"
cp experiments/gigpo_diag/CONFIG_DIFF.md "$GIGPO_RUN_DIR/CONFIG_DIFF.md"
git status --short --branch
python -c 'import sys,verl; print("TRAIN_RUNTIME",sys.executable,verl.__file__,flush=True)'
if [[ "$GIGPO_ENV" == scienceworld ]]; then
  python -m experiments.gigpo_diag.runtime science-reset-check
fi
# Native foreground training, full reference schedule, truthful exit status.
bash experiments/gigpo_diag/launch_reference.sh vllm 2>&1 | tee "$GIGPO_RUN_DIR/train.log"
