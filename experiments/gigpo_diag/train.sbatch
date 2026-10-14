#!/usr/bin/env bash
#SBATCH -A prjsnnn0000269
#SBATCH --job-name=ggp-app7-4g
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
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
export GIGPO_NUM_GPUS=4
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
export GIGPO_RESOURCE_PROFILE="$GIGPO_RUN_DIR/bounded_runtime_profile.json"
python -c 'import json,os; from pathlib import Path; from experiments.gigpo_diag.verify_existing_model import verify_existing_model; identity=verify_existing_model(os.environ["GIGPO_MODEL_PATH"]); (Path(os.environ["GIGPO_RUN_DIR"])/"model_identity.json").write_text(json.dumps(identity,indent=2)); print("EXISTING_7B_MODEL_VERIFIED_OFFLINE",identity["revision"],flush=True)'
# A separate short job validates real task execution and full optimizer resume;
# the long job is submitted with afterok and verifies the matching evidence gate.
if [[ "${GIGPO_PREFLIGHT_ONLY:-0}" != 1 ]]; then
  python -m experiments.gigpo_diag.bounded_acceptance verify
fi
timeout --signal=TERM --kill-after=30s 15m python -m experiments.gigpo_diag.appworld_resource_preflight
if [[ "${GIGPO_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  timeout --signal=TERM --kill-after=30s 5m python -m unittest \
    experiments.gigpo_diag.test_appworld_resource \
    experiments.gigpo_diag.test_appworld_resource_preflight \
    experiments.gigpo_diag.test_appworld_exception_cleanup \
    experiments.gigpo_diag.test_appworld_timeout \
    experiments.gigpo_diag.test_checkpoint_transaction
  timeout --signal=TERM --kill-after=30s 3m python -m unittest \
    experiments.gigpo_diag.test_appworld_failure_protocol \
    experiments.gigpo_diag.test_checkpoint_cpu_roundtrip
  timeout --signal=TERM --kill-after=30s 9m python -m experiments.gigpo_diag.check_appworld_live_protocol
  timeout --signal=TERM --kill-after=30s 40m python -m torch.distributed.run \
    --standalone --nproc_per_node=4 -m experiments.gigpo_diag.check_checkpoint_roundtrip \
    --model "$GIGPO_MODEL_PATH" --output "$GIGPO_RUN_DIR/checkpoint-roundtrip"
  python -m experiments.gigpo_diag.bounded_acceptance publish
  exit 0
fi
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
