# appworld: reference recipe and execution differences

Reference: `examples/gigpo_trainer/run_appworld.sh`, upstream commit in SOURCE.json. This run follows the locked source recipe, not an exact reproduction of a historical paper table.

- Initialization: Qwen/Qwen2.5-1.5B-Instruct + tokenizer revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Immutable named directory; full file hashes checked before loading. No old pilot/RL checkpoint, no SFT.
- Preserve train batch 16, group 8, val batch 16, native prompt/parser/history, native reward and invalid penalty 0.1, KL loss 0.01, learning rate 1e-6, gamma .95, step weight 1, native normalization, 150 epochs. Dynamic filtering disabled; exact anchors.
- Resource adaptation: 2 Slurm GPUs, TP from reference; actor and logprob microbatch 2. No hardcoded CUDA devices. Separate Python 3.12.5 venv; vLLM version from each source recipe, pinned dependencies in requirements-runtime.txt.
- Native checkpoint every 10 updates and at native final boundary; keep all native saved checkpoints, including optimizer/extra/tokenizer. Initial model is a pinned reference. A walltime interruption may lose work since the latest complete checkpoint; no claim of automatic final checkpoint on SIGKILL.
- Initial eval enabled, native eval every 5 updates. Native ALF/Sci holdout evaluated in aggregate; full holdout text excluded. Fixed train/dev study slices and prefix probes are not yet implemented and do not block native training.
- SwanLab uses the existing Tracking owner once; local process-sharded JSONL always saved. Failed network logging is nonfatal. Raw trace/model artifacts are not uploaded to GitHub or tracker.
- Diagnostics detached/read-only; explicit step identity before reordering. Native invalid penalty is captured before/after mutation; actual AE/AS/joint captured within each native core. Original sampled tokens remain the loss target. Local parameter samples and successful native weight-loading calls are logged separately from driver update count.
- ScienceWorld-only: parent samples original L0 pool with replacement, repeats explicit task/variation/reset seed, worker loads it; same task/observation/state-summary verified before training. Python sampler state has a checkpoint sidecar. Original easy preset and reference `10 * (done and score>0)` retained; no score-delta/milestone shaping. Strict uses Java completion and unrounded score>=1 only as a diagnostic. Two groups x two live resets precede training.
- AppWorld-only: native one-response code turn, train/dev split, max_interactions aligned to max_steps. `multi_episode_rollout=False`, success_coef=0 and prompt_type=vanilla; the latter two are not read by the AppWorld ordinary prompt/reward path (ALF/WebShop shaping is a separate branch). No claim that AppWorld's previous default success_coef caused shaping. 16*8 + 16*1 = 144 services derived from resolved config. Unique job/worker/reset experiments preserve native API traces. No global process cleanup. AppWorld library pinned to upstream Git commit 42b5bcf3cd334fee33f0c37c02070a9f5807add5 per Git-install recipe (PyPI 0.1.3.post1 requires pydantic1 and conflicts with vLLM).

Native schedules and environment RNG may not resume bitwise identically (ALFWorld/TextWorld and AppWorld worker state not checkpointed). ScienceWorld task sampler sidecar restores its newly introduced independent RNG. Unknown labels remain null. No actual training has run locally; only HPC results can establish updates or learning.

## Installation repair after first HPC attempt (2026-10-13)

Observed: ALFWorld job 160221 built/installed flash-attn, then pip rejected
`ray[default]` in the constraints file. AppWorld job 160225 failed resolution:
vLLM 0.8.5 requires OpenTelemetry SDK 1.26.x but Ray[default] 2.49.0 requires >=1.30.
ScienceWorld job 160223 received an unknown Content-Type for the datasets index;
the same datasets 3.6.0 was found in the other jobs. Module pip/setuptools also
leaked from /apps/spack into the activated venv in all logs.

Separate constraints-runtime.txt strips extras; requirements-runtime.txt retains
the requested extras. Clear module PYTHONPATH/PYTHONHOME in both setup and train,
invoke installation using the absolute venv Python with -I, and verify imported
pip + distribution metadata reside in that venv. Retry only recognized transient
transport/index errors, at most three attempts. Each project gets its own pip cache.
Existing venvs and installed flash-attn are reused. No directory/cache purge or
global pip installation. Training recipe, model, reward and algorithm are unchanged.

AppWorld dependency adaptation: Ray[default] 2.47.1 permits SDK 1.26.x and meets
vLLM's >=2.43,!=2.44.* requirement. SwanLab 0.7.0 accepts protobuf 4.x, required by
OpenTelemetry proto 1.26.0; SwanLab 0.10.2 instead requires protobuf >=6.31.1.
Keep vLLM 0.8.5 and torch 2.6.0. This is a declared runtime dependency adaptation,
not an exact recreation of the source README's Ray 2.49.0 installation sequence.

## AppWorld prepare job 160516: Pydantic internal API mismatch

The AppWorld package built/installed, but the first `appworld install` invocation
failed during imports with `match_type() takes 2 positional arguments but 3 were
given`. The pinned AppWorld source common/datetime.py replaces
GenerateSchema.match_type(self, obj); installed Pydantic 2.14.1 calls it with an
additional origin argument. Reproduced the identical TypeError locally with the
same fixed AppWorld source, Pydantic 2.14.1 and FastAPI 0.143.0 in a clean Python
3.12 environment. Dependency metadata resolution alone did not detect this
runtime monkey-patch incompatibility.

Pin Pydantic 2.12.5 and FastAPI 0.141.1 in both runtime requirements and constraints.
Pydantic selects its exact pydantic-core 2.41.5. These pins satisfy the unchanged
AppWorld source's requirements and the complete Linux/Python3.12 dependency
graph, including vLLM/Ray/SwanLab. No AppWorld source, datetime patch, evaluator,
reward or training configuration edits. AppWorld CLI --help now exits 0 locally
(UTF-8 output enabled on Windows), and its actual DateTime/nullable-field model,
FastAPI request/response and OpenAPI schema checks pass.

The same prepare job runs check_appworld_compat.py after installing AppWorld and
before unpacking its apps. It saves appworld-compat-check.json in the job TMPDIR;
failure blocks training. This is an installation check with no environment task
or model. The actual HPC app unpacking, data download, services and training are
still unverified. Existing venv and installed native libraries are reused. Retry
only this project's submit.sh after pulling, not bootstrap_all.sh.

AppWorld's native project dependencies also install decord 0.6.0, the same exact
PyPI wheel with an inconsistent cp36 internal tag already verified during the
ALFWorld 160510 incident. Reuse that version/byte/platform-guarded WHEEL+RECORD
repair with decord-only evidence; backup original metadata in this project's
venv, retain full pip check and require a native decord array round trip. The
TextWorld repair and ALFWorld environment check are not applied here. This avoids
a known subsequent metadata failure without changing library code. Native Linux
decord execution still awaits the HPC job; no local GPU/runtime success claimed.

## Shared data stack: ALFWorld prepare job 160563

ALFWorld's full pip check, native decord call, actual environment reset/look,
torch CUDA availability and pinned model preparation now passed. The final data
preparation import then failed: installed PyArrow 26.0.0 requires NumPy >=2.0,
while these reference runtimes pin NumPy 1.26.4. Dependency metadata resolution
and pip check alone did not detect this runtime restriction. Training has not
started in the supplied log.

Keep NumPy 1.26.4 and datasets 3.6.0, explicitly pin PyArrow 19.0.1 in requirements
and constraints. All three complete Linux/Python3.12 dependency graphs resolve.
Actual local Python3.12 checks passed datasets mapping, nested prompt/extra_info
Parquet write/read, pandas/pyarrow reads and NumPy conversion. Add the same small
check after pip check in the existing prepare job; outputs are dependency checks
under the job TMPDIR, not experimental rollouts. Do not change native prepare.py,
placeholder sizes, task distribution, reward or trainer. Existing environments,
model files and native libraries are reused. Other projects receive the same
version constraint to prevent this shared failure; their jobs are not restarted
or cancelled automatically. HPC data preparation/real training remain unverified.

## AppWorld prepare job 160576: missing Git LFS bundle contents

HPC APPWORLD_COMPAT_CHECK_PASSED confirms the earlier Pydantic fix. The next
native appworld install failed because the installed apps.bundle was an LFS
pointer. At the unchanged pinned upstream commit 42b5bcf3cd334fee33f0c37c02070a9f5807add5,
install_package requires apps.bundle and tests.bundle, totaling 398651 bytes.
Fetch only those two blobs from that commit's media.githubusercontent.com URLs;
check the exact pointer, size and SHA256 before replacement. Already materialized
matching files need no download. Guard the dedicated venv and package version,
preserve original pointers/RECORD and update installed RECORD hashes. Save the
verification report to the job TMPDIR. No upstream source or evaluator changes.

On Windows/Python 3.12.4 with the actual pinned AppWorld package, downloaded both
objects and ran unchanged appworld install successfully (apps and tests unpacked,
exit 0). A second helper invocation verified both without downloading. This is
local installation evidence, not HPC service, data-download or training evidence.
Set APPWORLD_CACHE under the user's own scratch root in both prepare and training;
the upstream default otherwise writes to the shared account's HOME cache.
Reuse the existing venv and retry only AppWorld. Preparation defaults to one hour.

## Prepare 160581 / 160583: multiprocessing AF_UNIX path length

ALFWorld and ScienceWorld passed dependency/CUDA/model checks, then both failed
in the native prepare.py datasets.map(num_proc=8) SyncManager listener bind:
OSError: AF_UNIX path too long. EOFError is the subsequent parent-side symptom.
The long scratch TMPDIR, not the separately shortened RAY_TMPDIR, determines
Python multiprocessing listener paths. No optimizer updates occurred.

Keep the scratch storage directory as GIGPO_JOB_TMPDIR. Give TMPDIR/TMP/TEMP a
short alias inside a unique mode-700 mktemp directory on the compute node;
the alias points to that job's original scratch directory. Bulk builds and
diagnostic files still persist under the original scratch path. This applies
to both preparation and training, whose dataset filtering can also spawn
workers. Do not change native num_proc, datasets, tasks or training recipe.
AppWorld gets the same preventative fix; no remote running job is modified.
Local Linux validation is recorded in IPC_PATH_VALIDATION.json. The actual
HPC retry remains necessary; no preparation/training success is claimed yet.

## Train 160623 / 160621: runtime startup failures after prepare passed

User reports all three preparation jobs passed. ALFWorld 160623 loaded native
train/eval environment pools and data, but the Worker CUDA visibility guard
rejected nonempty ROCR_VISIBLE_DEVICES together with HIP/CUDA_VISIBLE_DEVICES.
The log records zero updates and zero diagnostic episodes. It does not establish
which upstream process set ROCR; Slurm GPU flags can export multiple vendor masks
(https://slurm.schedmd.com/gres.conf.html). On these fixed NVIDIA jobs, record the
three masks then clear AMD-only ROCR/HIP before Python and job-local Ray start.
Require and preserve the exact Slurm CUDA_VISIBLE_DEVICES value. Keep Ray GPU
assignment and native Worker checks unchanged; no RAY_NOSET flags or GPU IDs added.
The same NVIDIA launcher adaptation is applied to AppWorld for the next retry.

AppWorld 160621 failed even earlier while importing the shared env_manager:
the module-level ALFWorld constants import executes ALFWorld package initializers,
which require the unrelated external alfworld package. Reproduced the identical
ModuleNotFoundError locally. Move this one import into its sole consuming ALFWorld
method, is_openable_receptacle. AppWorld's parser, prompt, reward and evaluator are
unchanged; the native ALFWorld method still uses the same constants when called.
No new dependencies or prepare rerun are required. Retry only the failed train
jobs with new run IDs; retain both failed-run reports. ScienceWorld is not changed.


## AppWorld 160641: non-blocking SwanLab validation logging

The retry passed imports and GPU initialization and recorded 16 initial dev
evaluation episodes, but completed zero optimizer updates. SwanLab init raised
ApiError (the old record retained only its type), and Tracking fell back to local
diagnostics. The separate validation logger still used the configured backend
list and called swanlab.log without an initialized run, aborting before training.
The available evidence does not distinguish authentication, network or server
causes of the original ApiError.

Validation now uses the successfully initialized backend dictionary. Its selected
samples are saved locally before any upload; SwanLab text construction/upload,
scalar upload and finish failures remain non-blocking. Init failures also record
the exception message with known environment credentials redacted and traceback
locations. No second run is initialized during validation. Full native validation
trajectories remain in checkpoints/val_trajectories/val_0.jsonl as before.
Snapshot tracking.py alongside the collector, trainer and diagnostics. No model,
reward, parser, task pool, sampling, evaluator, package or training schedule change.
Keep SwanLab 0.7.0 and rerun only train with a fresh run directory; no prepare.
Local regression tests simulate init/media/scalar/finish failures and successful
uploads. They do not establish HPC cloud connectivity or optimizer progress.

## AppWorld 160668: confirmed SwanLab project/SDK incompatibility

The training log continued through initial dev evaluation into the training
loop with local diagnostics. This does not by itself prove any optimizer update.
The tracker record shows SwanLab 0.7.0 POST /api/project returning HTTP 422.
Using the authorized key, login and GET of the existing private project both
returned 200. Reproducing the SDK's project request returned the explicit server
message: "新版本项目请使用新版 SDK（版本号 >= 0.9.0）". Supplying the existing
username and visibility returned the same message. No test run was created.

PyPI metadata confirms SwanLab 0.9.0/0.9.1/0.10.2 require protobuf >=6.31.1,<7,
whereas the current vLLM OpenTelemetry proto 1.26.0 requires protobuf >=3.19,<5.0.
The locked training resolution uses protobuf 4.25.9. Do not upgrade the active
training environment or substitute the algorithm's dependency stack for tracking.

Add only a standalone post-hoc scalar exporter, its tests and documentation.
It imports no trainer, model or environment. It requires a final manifest exit
code, preserves original global_step values and source run/commit identity,
hashes inputs, rejects conflicts/truncated input, and leaves unknowns omitted.
Snapshots and SDK logs go outside the original result directory. The cloud run
is explicitly suffixed -backfill; it is not a fresh experiment or new training.
Raw trajectories and text samples remain in the original local logs. Completed
upload receipts prevent accidental repeated upload; uncertain attempts require
inspection. SDK finish returning is recorded separately from cloud verification.

Run this utility with an independent modern SwanLab interpreter only after the
job ends (the existing pilot environment has used 0.10.2). Training scripts,
requirements, runtime logging and rewards are unchanged. Six synthetic local
tests passed; no HPC backfill or cloud restoration has been performed.

Metadata: https://pypi.org/pypi/swanlab/0.9.0/json and
https://pypi.org/pypi/opentelemetry-proto/1.26.0/json.
