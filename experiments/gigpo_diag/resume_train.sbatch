#!/usr/bin/env bash
#SBATCH -A prjsnnn0000269
#SBATCH --job-name=ggp-app7-resume
#SBATCH --partition=testqueue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=17:50:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/ggp-appworld-7b-resume-%j.out
set -euo pipefail
# This controller needs only stdlib. The pinned training entry loads modules
# and activates its own existing Python 3.12 environment after selection.
exec /scratch/prjsnnn0000269/Students/shuming/cjk/runtimes/bin/python3.10 \
  experiments/gigpo_diag/resume_chain.py run "$@"
