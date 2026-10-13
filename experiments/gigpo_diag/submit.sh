#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p /scratch/prjsnnn0000269/Students/shuming/cjk/logs
commit=$(git rev-parse HEAD)
prep=$(sbatch --parsable --export="ALL,GIGPO_EXPECTED_COMMIT=$commit" experiments/gigpo_diag/prepare.sbatch)
prep=${prep%%;*}
echo "prepare=$prep"
train=$(sbatch --parsable --export="ALL,GIGPO_EXPECTED_COMMIT=$commit" --dependency="afterok:$prep" --kill-on-invalid-dep=yes experiments/gigpo_diag/train.sbatch)
train=${train%%;*}
printf 'prepare=%s train=%s repo=%s commit=%s
' "$prep" "$train" "$PWD" "$(git rev-parse HEAD)"
mkdir -p /scratch/prjsnnn0000269/Students/shuming/cjk/logs/gigpo-reproduction/control
printf '%s	%s	%s	%s	queued
' "$prep" "$train" "$PWD" "$(git rev-parse HEAD)" >> /scratch/prjsnnn0000269/Students/shuming/cjk/logs/gigpo-reproduction/control/jobs.tsv
