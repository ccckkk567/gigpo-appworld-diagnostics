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
