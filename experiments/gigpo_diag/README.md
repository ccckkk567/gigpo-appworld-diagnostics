# appworld GiGPO training diagnostics

Run from this project's code root. Keep branches and each editable venv separate.

```bash
bash experiments/gigpo_diag/submit.sh
```

This queues one GPU setup job and a two-GPU training job dependent only on this setup. The three projects have no cross-project dependencies. Queue: testqueue, account prjsnnn0000269. Logs: `/scratch/prjsnnn0000269/Students/shuming/cjk/logs/ggp-appworld-{prepare,train}-JOBID.out`.

The reference launcher calls this project's own `verl.trainer.main_ppo`, not frozen sampling. No main/master merge. Resolved config, source/model identity, package freeze, actual command and snapshots are saved per run. `CONFIG_DIFF.md` describes reference preservation and necessary adaptations.

During training, run the following in an allocated CPU job, or download the compact diagnostics to analyze locally:

```bash
python -m experiments.gigpo_diag.analyze /path/to/run
```

`reports/current.md`, `tables/rollout_metrics.csv`, PNG figures and native checkpoint index are derived only from observed records. An exit before any optimizer update is explicitly reported. No claim of training completion from submitted IDs.

Resume only your latest fully saved `global_step_N` directory: set `GIGPO_RESUME_MODE=resume_path` and `GIGPO_RESUME_FROM=/absolute/own/checkpoints/global_step_N` when submitting a new train job. A new output run keeps prior files. This completes the original schedule; it is not permission for extra seeds or training beyond it.

## Post-hoc SwanLab scalar backfill

AppWorld's training SDK 0.7.0 cannot write to the existing new-version project.
Modern SwanLab requires an incompatible protobuf version, so keep the training
venv unchanged. Local diagnostics continue independently of cloud availability.

Wait until the AppWorld job has ended before pulling this change in its HPC
checkout. Do not upgrade its venv or restart a healthy job for this issue.
Run the standalone utility with a separate modern SwanLab interpreter, such as
the existing `envs/dep-prep-py312-v2/bin/python` (previously used SDK 0.10.2).
Use an allocated CPU node if that module Python cannot run on the login node;
alternatively download the finished run's manifest and diagnostic metrics for
local upload. No GPU, model, training environment or new installation is needed.

Example, after the training job has ended and this code is available:

```bash
ORION_ROOT=/scratch/prjsnnn0000269/Students/shuming/cjk
RUN_DIR="$ORION_ROOT/logs/gigpo-reproduction/appworld/appworld-qwen25-15b-gigpo-160668-20261013T113324Z"
EXPORT_DIR="$ORION_ROOT/logs/gigpo-reproduction/swanlab-backfills/$(basename "$RUN_DIR")"
UPLOAD_PYTHON="$ORION_ROOT/envs/dep-prep-py312-v2/bin/python"
"$UPLOAD_PYTHON" -I experiments/gigpo_diag/backfill_swanlab.py "$RUN_DIR" --output "$EXPORT_DIR"
```

The default command prepares `snapshot.json` without importing SwanLab or using
the network. Review its source run ID, original steps and omitted-value count.
To upload that snapshot, load the existing credentials in this separate shell
and repeat with `--upload`:

```bash
set -a
source "$ORION_ROOT/dep-prep-credentials.env"
set +a
"$UPLOAD_PYTHON" -I experiments/gigpo_diag/backfill_swanlab.py "$RUN_DIR" --output "$EXPORT_DIR" --upload
```

The existing private project is used without changing its visibility. The run
name ends in `-backfill`; original run ID, commit and snapshot hash are in its
config. Only recorded finite scalar metrics are copied; unknowns are omitted,
never changed to zero. Raw trajectories, selected text and other diagnostic
streams stay in the source results. The utility does not change source files.
It rejects unfinished runs, mismatched identities and conflicting scalar values.
An existing completed upload receipt makes a repeat a no-op; inspect an uncertain
receipt and the cloud run before any manual retry. `sdk_finish_returned` does
not independently verify that every point is visible remotely. Local synthetic
tests cover ordering, input integrity, source preservation and upload failures;
they are not evidence of an HPC upload or a completed training update.
