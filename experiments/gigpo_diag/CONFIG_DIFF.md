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

## AppWorld 160668: LibCST validation exception before code execution

Job 160668 completed two native actor updates (driver policy updates 1 and 2),
with 272 completed episode records (16 initial dev + 2*128 train). It then failed
during the third rollout: /execute returned HTTP 500. The service traceback is
SafetyGuard.is_syntax_safe -> parse_imports -> libcst.parse_module ->
CSTValidationError: "The bare except: handler must be the last one."
Other pasted NameError and nonexistent-API errors returned HTTP 200 and are
ordinary model/environment feedback, not the cause of the training exit.

At locked in-context-explorer commit cd71379130155b2d75c9100753d1454d3cde6696,
AppWorldWorker.step directly calls env.execute without an exception handler,
and AppWorldEnvs.step propagates ray.get failures. projection.py checks tags,
not Python syntax. The upstream README installs AppWorld from unpinned Git;
the repository does not establish the authors' exact historical dependency
versions or whether they encountered this input. Our pinned AppWorld commit is
42b5bcf3cd334fee33f0c37c02070a9f5807add5. Its code_is_valid uses ast.parse, which
accepts the minimal offending structure, followed by uncaught LibCST validation.
The same exception was reproduced locally with the unmodified package.

The native service now starts via experiments.gigpo_diag.appworld_server. This
installs a process-local wrapper around SafetyGuard.is_syntax_safe before invoking
the original CLI. Only CSTValidationError is converted to (False, error message).
Native AppWorld.execute then performs its existing rejection path: no submitted
code executes, one interaction is consumed, original input and error are logged,
and feedback returns through HTTP 200. All normal checks and other exceptions
retain their original behavior. No package file is edited or upgraded; no broad
HTTP retry, task reset, success fallback or failure-to-zero conversion is added.
Projection/format penalty, reward, evaluator, GiGPO and batch/seed stay unchanged.
This is an explicitly recorded compatibility repair, not an existing upstream
feature. Snapshot the wrapper/services and record the installed LibCST version.

Local validation: five tests with real AppWorld 0.2.0.dev0, LibCST 1.9.0,
FastAPI 0.141.1 and Starlette 1.8.0. Real /execute endpoint and AppWorld.execute
run against a minimal synthetic world fixture (no task database/model): baseline
HTTP 500; patched rejection HTTP 200 with exact original input and one turn;
valid/forbidden/ordinary-invalid checks unchanged; unrelated errors still 500;
idempotent installation. This does not claim a new HPC rollout or training run.

Retain job 160668's failed-after-updates results. Its checkpoint schedule saves
every ten updates and it stopped at two; the normal retry starts a new run from
the pinned initial model, unless a real complete checkpoint is separately found.
No prepare rerun is needed. Other project checkouts and jobs are unchanged.


## AppWorld 160809: receive native code-timeout feedback before HTTP expiry

HPC run appworld-qwen25-15b-gigpo-160809-20261013T131208Z used 1755dc7,
including the prior LibCST rejection handler. It completed 16 initial dev
episodes, then failed in the first train rollout (zero actor updates) with
requests.ReadTimeout at 127.0.0.1:35695 after 100 seconds. Service manifest maps
this port to worker 118 / server PID 3955464. Its final execute logged HTTP 200.
The user's environment_io.md identifies task 07b42fd_1, interaction 22: the model
iterated filtered_artists while appending matching artists to the same list.
AppWorld recorded 'Exception: Execution timed out after 100 seconds.' The native
failure feedback existed but the HTTP client expired at the same 100s boundary.

Pinned AppWorld uses timeout_seconds for both server code execution and client
requests.post reads. Add a narrow AppWorldClient subclass that delegates to the
native helper, extending only execute HTTP requests by 60 seconds (100 -> 160
with this Linux recipe). Initialization still sends the original 100-second
execution limit. Other RPCs, no-timeout mode, parser, code input, error feedback,
interaction counts, evaluator/reward and GiGPO logic are unchanged. There are no
automatic execute retries, resets, invented observations, or swallowed transport
errors; a genuinely stalled service still aborts at the bounded client deadline.
The code may have made partial state changes before the native timeout; retain
AppWorld's original semantics rather than pretending to roll those back.

Each new run records execution and transport deadlines in environment provenance
and reset info, and snapshots the client and environment wrapper. The existing
process-local LibCST repair and nonfatal SwanLab fallback remain in place. This
requires only the repository update, not a dependency/environment reinstall.

Validation is local client compatibility checking, not a new HPC training run:
real native client with synthetic responses verifies 100s server initialization,
160s execute reads, original feedback/input/one-turn accounting, unchanged
success and no-timeout behavior, and fail-fast HTTP/connection/read errors with
one request only. A real loopback HTTP server uses scaled short timings to show
the old deadline fails while the new client receives the delayed error response;
it does not execute a 100-second infinite loop or task DBs. All 12 tests passed
(seven client checks plus five existing LibCST/FastAPI regressions). Further HPC updates remain
unverified until a new user-submitted training job runs.

Pinned source (native timeout and remote helper):
https://github.com/StonyBrookNLP/appworld/blob/42b5bcf3cd334fee33f0c37c02070a9f5807add5/src/appworld/environment.py


## AppWorld 160868: capture pending requests and slow native execution stages

Run appworld-qwen25-15b-gigpo-160868-20261013T135415Z used b27468a.
The first real actor update completed after 16 dev and 128 training episodes;
reference success remained zero and the second train rollout failed with an
execute HTTP read timeout at 160 seconds (worker 28, port 44483). The latest
saved native environment_io.md is interaction 28 of task e85d92a_1, reset 2:
an unexpected-indent rejection. The service log shows its subsequent save_logs
and task_completed requests returned 200. This is the last completed request,
not evidence of what the missing next request executed or where it stalled.
There is no completed checkpoint at the native ten-update save interval.

Do not extend deadlines again or claim a confirmed root cause. Add process-local
observation to the native service wrapper: flush original request code, task,
experiment, interaction counter and hash before execute; record entry/exit of
syntax checking, shell execution, state save and log save. Native methods still
run once with unchanged arguments, results and exceptions. At 110 seconds, a
faulthandler C watchdog writes one stack snapshot to a separate service file;
it does not stop execution or use SIGALRM. This distinguishes code execution
from parsing/post-execution I/O if the next run stalls again. Stack and journal
files are services/execute-PID.stacks.log and services/execute-PID.jsonl.

Both files are opened before AppWorld installs its execution safety guard.
Diagnostic output is written to those files rather than captured model output.
Linux elapsed times use CLOCK_MONOTONIC, outside AppWorld's frozen task time.
Diagnostic write errors are nonfatal. Existing 100s code / 160s HTTP timeouts,
LibCST rejection repair, rewards, prompts, task sampling, and optimizer remain
unchanged. This commit adds evidence capture, not a demonstrated timeout fix.
The underlying cause and recovery still require a bounded HPC rerun.

Validation: all 18 local AppWorld client/parser/execution-trace tests passed.
This includes the real native Python shell with SafetyGuard enabled, ensuring
file diagnostics do not enter captured model feedback. Scaled slow-call tests
produce a stack snapshot; these are not a reproduction of the HPC stall.

Reference comparison requested by the user: pengzj1/revisiting_opd at commit
3379dde59c323cedd30d400cc48dfffac71470f3 versus XiaofengLin7/in-context-explorer.
Both AppWorld worker files have identical Git blob
0984e9268decdabcd5a3f07b0cd70cf93c4452db and directly call env.execute; no new
execution-timeout recovery is present. Prompts and action projection also match
our upstream files (blobs 0d3c163ad23b8ab88fec6a7ce25171b6153df33c and
d6ccbd74aa3675ed4220c3fd6867a64a516be131). Its startup timeout is a readiness
check only, not an execution deadline. Its script uses a 3B model, group size 5,
max_response_length 20480, top_p 0.9, mean_std_norm and similarity grouping.
Those are distinct training settings, not demonstrated fixes for this stall.
Its service script defaults to 4*1+32=36 services while its training script
requires 16*5+128=208; defaults must be coordinated before using them.

Sources:
https://github.com/pengzj1/revisiting_opd/blob/3379dde59c323cedd30d400cc48dfffac71470f3/examples/gigpo_trainer/run_appworld.sh
https://github.com/pengzj1/revisiting_opd/blob/3379dde59c323cedd30d400cc48dfffac71470f3/agent_system/environments/env_package/appworld/envs.py
https://github.com/pengzj1/revisiting_opd/blob/3379dde59c323cedd30d400cc48dfffac71470f3/examples/env_server/start_appworld_server.sh


## Independent AppWorld 7B capacity experiment (full reference schedule)

The user requested Qwen2.5-7B-Instruct and then explicitly requested full training
instead of the proposed two-hour / two-update probe. Branch
codex/appworld-qwen25-7b-train keeps the full 150-epoch schedule, checkpoint
frequency 10, evaluation frequency 5, and initial/final dev evaluation. Slurm
requests two GPUs, 32 CPUs, 256G RAM and 17:50:00, within testqueue's 18h limit.
There is no two-update cutoff. The time limit does not guarantee completion of
all 150 epochs; native checkpoints allow later continuation.

Use the user's existing directory:
/scratch/prjsnnn0000269/Students/shuming/cjk/data/models/Qwen2.5-7B-Instruct-a09a35458c70
Pin model/tokenizer revision a09a35458c702b33eeacc393d103063234e8bc28.
No model download, dependency install or editable-install change is required.
Before services/model loading, verify the existing 11 official files offline:
four LFS weight SHA256s plus Git blob checksums for configuration/tokenizer files,
file sizes, shard set, and 7B architecture. The expected checksum manifest was
retrieved from the official pinned model API. Missing/corrupt/unexpected files
abort rather than download or silently trust the directory name. Verification
only reads the model directory; identity evidence is saved under the new run.

Keep prompts, response/prompt limits (1024/7167), seed, task sampling, 16*8
training batch, 16 dev evaluation batch, mean_norm, exact grouping, penalties,
gamma, step weight, learning rate and effective PPO minibatch unchanged. Memory
adaptations: actor/rollout-logprob/ref-logprob microbatch per GPU 2->1, vLLM
memory utilization 0.6->0.45. These preserve the intended objective but do not
establish numerical equivalence or guarantee 7B fits the allocated GPUs.
Generation and update speed can change. Report names and manifests identify 7B.

All prior AppWorld repairs and execution-stage diagnostics are included.
SwanLab's old-SDK cloud initialization limitation still uses the nonfatal local
fallback; the existing independent backfill utility remains available. The
capacity hypothesis remains unproven until new real trajectories/updates arrive.

Use a separate HPC checkout for this branch so running or queued 1.5B jobs
retain their original code. Existing 1.5B experiment branch and other projects
are unchanged by the 7B configuration.

Official pinned metadata and architecture:
https://huggingface.co/api/models/Qwen/Qwen2.5-7B-Instruct/revision/a09a35458c702b33eeacc393d103063234e8bc28?blobs=true
https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/a09a35458c702b33eeacc393d103063234e8bc28/config.json

Local verification: five offline integrity tests and all modified shell/Python syntax checks passed. The inherited 18 client/parser/execution-trace regressions passed before branching. No local 7B weights or GPU training were run; HPC capacity and learning remain unverified.
