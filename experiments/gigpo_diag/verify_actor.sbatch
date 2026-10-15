#!/usr/bin/env bash
# One real update with the production rollout and compiled actor path.
#SBATCH -A prjsnnn0000269
#SBATCH --job-name=ggp-app7-verify
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=01:15:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/ggp-appworld-7b-verify-%j.out
set -euo pipefail
export GIGPO_ENV=appworld GIGPO_VAL_BATCH=16
module purge
module load python/3.12.5 cuda/12.6.2
source experiments/gigpo_diag/common.sh
source experiments/gigpo_diag/nvidia_job_env.sh
test -f "$GIGPO_VENV/.setup-complete"
source "$GIGPO_VENV/bin/activate"
python experiments/gigpo_diag/check_python_environment.py --pip-version 25.3
export GIGPO_CODE_COMMIT="$(git rev-parse HEAD)"
export GIGPO_RUN_ID="appworld-7b-2gpu-actor-verify-${SLURM_JOB_ID}-$(date -u +%Y%m%dT%H%M%SZ)"
export GIGPO_RUN_DIR="$ORION_ROOT/logs/gigpo-reproduction/appworld/$GIGPO_RUN_ID"
mkdir -p "$GIGPO_RUN_DIR"
export GIGPO_DIAG_DIR="$GIGPO_RUN_DIR/diagnostics"
export APPWORLD_ROOT="$GIGPO_RUN_DIR/appworld"
export APPWORLD_PORTS_FILE="$GIGPO_RUN_DIR/appworld_ports.ports"
export GIGPO_DATA_DIR="$ORION_ROOT/data/gigpo-reproduction/appworld/placeholders"
export GIGPO_NUM_GPUS=2 GIGPO_CPUS="${SLURM_CPUS_PER_TASK:-32}"
export GIGPO_RESOURCE_PROFILE="$GIGPO_RUN_DIR/bounded_runtime_profile.json"
export GIGPO_ACTOR_VERIFY=1
export XDG_CACHE_HOME="$ORION_ROOT/.cache/gigpo-appworld"
mkdir -p "$XDG_CACHE_HOME"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HYDRA_FULL_ERROR=1 TORCHDYNAMO_VERBOSE=1
# Keep the existing compiler/cache settings. No eager fallback or cache deletion.
unset TORCH_COMPILE_DISABLE TORCHDYNAMO_DISABLE
unset GIGPO_PREFLIGHT_ONLY GIGPO_ACCEPTANCE_DIR GIGPO_RESUME_FROM
export GIGPO_RESUME_MODE=disable
on_exit() {
  local result=$?
  trap - EXIT
  set +e
  if [[ -d "$RAY_TMPDIR/session_latest/logs" ]]; then
    timeout --signal=TERM --kill-after=5s 60s tar -czf "$GIGPO_RUN_DIR/ray-logs.tar.gz" \
      -C "$RAY_TMPDIR/session_latest" logs
    echo "RAY_LOG_ARCHIVE_EXIT=$? path=$GIGPO_RUN_DIR/ray-logs.tar.gz"
  fi
  python -m experiments.gigpo_diag.analyze "$GIGPO_RUN_DIR" --exit-code "$result"
  echo "GIGPO_ACTOR_VERIFY_END job=$SLURM_JOB_ID exit=$result output=$GIGPO_RUN_DIR"
  exit "$result"
}
trap on_exit EXIT
echo "GIGPO_ACTOR_VERIFY_BEGIN output=$GIGPO_RUN_DIR"
python -m experiments.gigpo_diag.actor_update_evidence environment "$GIGPO_RUN_DIR"
cp experiments/gigpo_diag/verify_actor.sbatch "$GIGPO_RUN_DIR/verify_actor.sbatch"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$GIGPO_RUN_DIR/gpus.csv"
python -m experiments.gigpo_diag.runtime prepare-model
# This is a diagnostic job, not an acceptance publisher. Still calibrate and
# retain all environment protection before running any policy-generated code.
timeout --signal=TERM --kill-after=30s 15m python -m experiments.gigpo_diag.appworld_resource_preflight
# Match initial validation and the first real 16x8 rollout; stop after one update.
# Disable end-of-run eval/checkpoint and cloud logging only in this diagnostic.
args=(vllm trainer.total_training_steps=1 trainer.save_freq=-1 trainer.test_freq=-1
      'trainer.logger=[console]' actor_rollout_ref.actor.use_torch_compile=True)
printf '%q ' bash experiments/gigpo_diag/launch_reference.sh "${args[@]}" > "$GIGPO_RUN_DIR/actual_command.txt"
printf '\n' >> "$GIGPO_RUN_DIR/actual_command.txt"
timeout --signal=TERM --kill-after=30s 55m bash experiments/gigpo_diag/launch_reference.sh "${args[@]}" \
  2>&1 | tee "$GIGPO_RUN_DIR/train.log"
python -m experiments.gigpo_diag.actor_update_evidence verify "$GIGPO_RUN_DIR"
