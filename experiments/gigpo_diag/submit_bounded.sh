#!/usr/bin/env bash
# Run from the NEW isolated checkout; no scancel, pushes, installs or downloads.
set -euo pipefail
git diff --quiet
git diff --cached --quiet
commit=$(git rev-parse HEAD)
stamp=$(date -u +%Y%m%dT%H%M%SZ)-$$
root=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/gigpo-reproduction/appworld
acceptance="appworld-7b-4gpu-preflight-$stamp"
training="appworld-7b-4gpu-bounded-$stamp"
preflight=$(sbatch --parsable --time=01:15:00 --job-name=ggp-app-check \
  --mail-user=1909674001@qq.com --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90 \
  --export="ALL,GIGPO_EXPECTED_COMMIT=$commit,GIGPO_PREFLIGHT_ONLY=1,GIGPO_RUN_ID=$acceptance" \
  experiments/gigpo_diag/train.sbatch)
preflight=${preflight%%;*}
printf 'PREFLIGHT_JOB=%s RUN=%s/%s\n' "$preflight" "$root" "$acceptance"
full=$(sbatch --parsable --dependency="afterok:$preflight" --kill-on-invalid-dep=yes \
  --mail-user=1909674001@qq.com --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90 \
  --export="ALL,GIGPO_EXPECTED_COMMIT=$commit,GIGPO_PREFLIGHT_ONLY=0,GIGPO_RUN_ID=$training,GIGPO_ACCEPTANCE_DIR=$root/$acceptance" \
  experiments/gigpo_diag/train.sbatch)
printf 'TRAIN_JOB=%s RUN=%s/%s\n' "${full%%;*}" "$root" "$training"
