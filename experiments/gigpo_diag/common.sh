#!/usr/bin/env bash
set -euo pipefail
export ORION_ROOT=/scratch/prjsnnn0000269/Students/shuming/cjk
: "${GIGPO_ENV:?set environment identity before sourcing common.sh}"
if [[ -n "${GIGPO_EXPECTED_COMMIT:-}" && "$(git rev-parse HEAD)" != "$GIGPO_EXPECTED_COMMIT" ]]; then
  echo 'Checkout changed after job submission; refusing a mixed-code run.' >&2
  exit 1
fi
export GIGPO_VENV="$ORION_ROOT/envs/gigpo-${GIGPO_ENV}-py312-v1"
export GIGPO_MODEL_PATH="$ORION_ROOT/data/models/Qwen2.5-1.5B-Instruct-989aa798"
export HF_HOME="$ORION_ROOT/.cache/huggingface"
export ALFWORLD_DATA="$ORION_ROOT/data/alfworld"
export GIGPO_APPWORLD_DATA="$ORION_ROOT/data/appworld-gigpo-v1"
export JAVA_HOME="$ORION_ROOT/runtimes/jdk-11.0.28+6"
export PATH="$JAVA_HOME/bin:$PATH"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PIP_CONFIG_FILE=/dev/null
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_NO_INDEX PIP_FIND_LINKS
unset RAY_ADDRESS WANDB_RUN_ID WANDB_RESUME
export WANDB_MODE=disabled
mkdir -p "$ORION_ROOT/logs/gigpo-reproduction" "$ORION_ROOT/tmp/gigpo-reproduction" "$ORION_ROOT/envs"
if [[ -f "$ORION_ROOT/dep-prep-credentials.env" ]]; then
  set -a
  source "$ORION_ROOT/dep-prep-credentials.env"
  set +a
fi
export SWANLAB_MODE="${SWANLAB_MODE:-cloud}"
export TMPDIR="$ORION_ROOT/tmp/gigpo-reproduction/${GIGPO_ENV}-${SLURM_JOB_ID:-local}"
mkdir -p "$TMPDIR"
# AF_UNIX socket paths are short and unique on each compute node.
export RAY_TMPDIR="/tmp/cjk-ggp-${SLURM_JOB_ID:-local}-${GIGPO_ENV:0:3}"
if [[ "$GIGPO_ENV" != alfworld ]]; then export VLLM_USE_V1=0; fi
if [[ "$GIGPO_ENV" == scienceworld ]]; then export JAVA_TOOL_OPTIONS='-Xmx512m -XX:ActiveProcessorCount=1'; fi
ulimit -c 0
