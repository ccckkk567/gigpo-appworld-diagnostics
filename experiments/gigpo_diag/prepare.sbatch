#!/usr/bin/env bash
#SBATCH -A prjsnnn0000269
#SBATCH --job-name=ggp-app-prepare
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/ggp-appworld-prepare-%j.out
export GIGPO_ENV=appworld
export GIGPO_VAL_BATCH=16
set -euo pipefail
module purge
module load python/3.12.5 cuda/12.6.2
source experiments/gigpo_diag/common.sh
nvidia-smi
trap 'echo "GIGPO_PREPARE_FAILED env=$GIGPO_ENV job=$SLURM_JOB_ID line=$LINENO" >&2' ERR
exec 9>"$GIGPO_VENV.install.lock"
flock -n 9 || { echo 'Another setup owns this project environment'; exit 1; }
BASE_PYTHON="$(command -v python3.12)"
"$BASE_PYTHON" -I -c 'import sys; assert sys.version_info[:2] == (3,12); print("BASE_PYTHON",sys.executable,sys.version,flush=True)'
if [[ ! -x "$GIGPO_VENV/bin/python" ]]; then "$BASE_PYTHON" -I -m venv "$GIGPO_VENV"; fi
source "$GIGPO_VENV/bin/activate"
python -I -c 'import sys; assert sys.version_info[:2] == (3,12) and sys.prefix != sys.base_prefix'
"$GIGPO_VENV/bin/python" -I -m ensurepip --upgrade
"$GIGPO_VENV/bin/python" -I experiments/gigpo_diag/check_python_environment.py
source experiments/gigpo_diag/pip_install.sh
rm -f "$GIGPO_VENV/.setup-complete"
gigpo_pip_install --upgrade --force-reinstall 'pip==25.3'
python experiments/gigpo_diag/check_python_environment.py --pip-version 25.3
gigpo_pip_install 'setuptools<81' wheel ninja packaging
export MAX_JOBS=4
gigpo_pip_install -r experiments/gigpo_diag/requirements-runtime.txt
gigpo_pip_install 'flash-attn==2.7.4.post1' --no-build-isolation -c experiments/gigpo_diag/constraints-runtime.txt
gigpo_pip_install -e . -c experiments/gigpo_diag/constraints-runtime.txt
if [[ "$GIGPO_ENV" == appworld ]]; then
  export APPWORLD_ROOT="$GIGPO_APPWORLD_DATA"
  mkdir -p "$APPWORLD_ROOT"
  gigpo_pip_install 'appworld @ git+https://github.com/StonyBrookNLP/appworld.git@42b5bcf3cd334fee33f0c37c02070a9f5807add5' -c experiments/gigpo_diag/constraints-runtime.txt
  appworld install
  appworld download data --root "$APPWORLD_ROOT"
  gigpo_pip_install 'antlr4-python3-runtime==4.9.3' -c experiments/gigpo_diag/constraints-runtime.txt
fi
"$GIGPO_VENV/bin/python" -I -m pip check
python -c 'import sys,verl,torch,vllm; print("RUNTIME",sys.executable,verl.__file__,torch.__version__,vllm.__version__,flush=True); assert torch.cuda.is_available()'
"$GIGPO_VENV/bin/python" -I -m pip freeze > "$GIGPO_VENV/gigpo-pip-freeze.txt"
# One shared immutable initialization; independent venvs and editable installations.
mkdir -p "$(dirname "$GIGPO_MODEL_PATH")"
exec 8>"$GIGPO_MODEL_PATH.download.lock"
flock 8
python -m experiments.gigpo_diag.runtime prepare-model
flock -u 8
if [[ "$GIGPO_ENV" == alfworld ]]; then
  test -d "$ALFWORLD_DATA/json_2.1.1/train"
  test -f "$ALFWORLD_DATA/logic/alfred.pddl"
fi
python -m examples.data_preprocess.prepare --mode text --train_data_size 16 --val_data_size "$GIGPO_VAL_BATCH" --local_dir "$ORION_ROOT/data/gigpo-reproduction/$GIGPO_ENV/placeholders"
date -u +%FT%TZ > "$GIGPO_VENV/.setup-complete"
echo "GIGPO_PREPARE_COMPLETE env=$GIGPO_ENV job=$SLURM_JOB_ID"
