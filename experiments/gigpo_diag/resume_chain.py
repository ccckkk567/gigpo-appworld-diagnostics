"""Append three Slurm jobs; select complete state at job start, never at submit.

The controller is separate from the immutable c1fb482 training checkout. It
does not change training, acceptance, checkpoint serialization or hyperparameters.
Only the explicitly recorded jobs in this chain may supply resume checkpoints.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid


ORION_ROOT = Path('/scratch/prjsnnn0000269/Students/shuming/cjk')
RUNS_ROOT = ORION_ROOT / 'logs/gigpo-reproduction/appworld'
TRAINING_COMMIT = 'c1fb4827301393c2f24f56a604a3fcb741e48f2f'
TRAINING_CHECKOUT = ORION_ROOT / 'code/gigpo-reproduction/appworld-7b-4gpu-train-c1fb482'
ACCEPTANCE_DIR = RUNS_ROOT / 'appworld-7b-4gpu-preflight-20261015T003620Z-3834802'
ACCEPTANCE_COMMIT = '58825a74e7de378070970581ae4d8d7b344c5235'
MODEL_PATH = ORION_ROOT / 'data/models/Qwen2.5-7B-Instruct-a09a35458c70'
HELPER_CHECKOUT = Path(__file__).resolve().parents[2]
COUNT = 3


def command(*args, cwd=None):
    return subprocess.check_output(list(args), cwd=cwd, text=True).strip()


def write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == 'posix':
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def numeric_job(value):
    if not re.fullmatch(r'[1-9][0-9]*', str(value)):
        raise ValueError(f'Expected an ordinary numeric Slurm job ID: {value}')
    return str(value)


def verify_checkout(path, expected):
    path = Path(path).resolve()
    if Path(command('git', 'rev-parse', '--show-toplevel', cwd=path)).resolve() != path:
        raise RuntimeError(f'Not a checkout root: {path}')
    if command('git', 'rev-parse', 'HEAD', cwd=path) != expected:
        raise RuntimeError(f'Checkout moved: {path}; expected {expected}')
    if command('git', 'status', '--porcelain', '--untracked-files=no', cwd=path):
        raise RuntimeError(f'Tracked modifications in {path}')


def slurm_fields(job):
    output = command('scontrol', 'show', 'job', '-o', numeric_job(job))
    # Paths in this project contain no whitespace. Values such as Reason may.
    return dict(re.findall(r'(?:^|\s)(\w+)=([^\s]+)', output))


def verify_parent(parent, fields):
    expected = {'JobId': parent, 'Account': 'prjsnnn0000269', 'Partition': 'testqueue',
                'WorkDir': str(TRAINING_CHECKOUT),
                'Command': str(TRAINING_CHECKOUT / 'experiments/gigpo_diag/train.sbatch'),
                'TimeLimit': '17:50:00', 'NumNodes': '1', 'NumCPUs': '32'}
    for key, value in expected.items():
        if fields.get(key) != value:
            raise RuntimeError(f'Parent {parent}: unexpected {key}={fields.get(key)}')
    if fields.get('JobState') not in ('PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING'):
        raise RuntimeError(f'Parent is no longer active: {fields.get("JobState")}')
    if fields.get('UserId', '').split('(')[0] != os.environ['USER']:
        raise RuntimeError('Parent job belongs to a different user')
    tres = dict(item.split('=', 1) for item in fields.get('ReqTRES', '').split(',') if '=' in item)
    if tres.get('gres/gpu') != '4' or tres.get('mem') not in ('320G', '327680M'):
        raise RuntimeError(f'Unexpected parent GPU/RAM request: {tres}')


@contextmanager
def submission_lock(path):
    import fcntl  # Slurm login host is Linux; no persistent stale lock directory.
    with Path(path).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def submit_jobs(chain, path, submit):
    """Durably record each submission; ambiguous outcomes never auto-resubmit."""
    for slot in range(1, COUNT + 1):
        if len(chain['jobs']) > slot:
            entry = chain['jobs'][slot]
            if entry['status'] != 'submitted':
                raise RuntimeError(f'Slot {slot} has an unresolved submission; inspect {path} and squeue before retrying')
            continue
        predecessor = numeric_job(chain['jobs'][slot - 1]['job_id'])
        entry = {'slot': slot, 'status': 'submitting', 'dependency': f'afterany:{predecessor}'}
        chain['jobs'].append(entry)
        write_json(path, chain)
        # CLI options override any inherited SBATCH_* settings.
        args = ['sbatch', '--parsable', '--account=prjsnnn0000269', '--partition=testqueue',
                '--nodes=1', '--ntasks=1', '--gres=gpu:4', '--cpus-per-task=32',
                '--mem=320G', '--time=17:50:00', '--no-requeue',
                f'--dependency={entry["dependency"]}', '--kill-on-invalid-dep=yes',
                f'--job-name=app7-{chain["parent_job"]}-r{slot}',
                f'--chdir={chain["helper_checkout"]}',
                f'--comment=gigpo-resume:{chain["parent_job"]}:{slot}',
                '--mail-user=1909674001@qq.com', '--mail-type=BEGIN,END,FAIL,TIME_LIMIT_90',
                f'--output={ORION_ROOT}/logs/ggp-appworld-7b-resume-%j.out', '--export=ALL',
                'experiments/gigpo_diag/resume_train.sbatch',
                '--chain', str(path), '--slot', str(slot)]
        # A failure here intentionally leaves "submitting" in the ledger:
        # timeout/lost reply can mean sbatch registered a job despite no output.
        reply = submit(*args, cwd=chain['helper_checkout'])
        entry.update(job_id=numeric_job(reply.split(';')[0]), status='submitted')
        write_json(path, chain)
        print(f'RESUME_SUBMITTED slot={slot} job={entry["job_id"]} dependency={entry["dependency"]}', flush=True)
    return [row['job_id'] for row in chain['jobs']]


def submit_chain(parent):
    parent = numeric_job(parent)
    helper_commit = command('git', 'rev-parse', 'HEAD', cwd=HELPER_CHECKOUT)
    verify_checkout(HELPER_CHECKOUT, helper_commit)
    verify_checkout(TRAINING_CHECKOUT, TRAINING_COMMIT)
    if not (ACCEPTANCE_DIR / 'acceptance.json').is_file():
        raise RuntimeError('Missing existing acceptance record')
    path = RUNS_ROOT / 'resume-chains' / f'from-{parent}' / 'chain.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = {'version': 1, 'parent_job': parent, 'resume_jobs': COUNT,
                'training_checkout': str(TRAINING_CHECKOUT), 'training_commit': TRAINING_COMMIT,
                'helper_checkout': str(HELPER_CHECKOUT), 'helper_commit': helper_commit,
                'acceptance_dir': str(ACCEPTANCE_DIR), 'runs_root': str(RUNS_ROOT)}
    with submission_lock(path.with_suffix('.lock')):
        if path.exists():
            chain = read_json(path)
            if any(chain.get(key) != value for key, value in identity.items()):
                raise RuntimeError(f'Existing chain has different configuration: {path}')
        else:
            verify_parent(parent, slurm_fields(parent))
            chain = {**identity, 'jobs': [{'slot': 0, 'job_id': parent, 'status': 'submitted'}]}
            write_json(path, chain)
        ids = submit_jobs(chain, path, command)
    print('CHAIN ' + ' -> '.join(ids), flush=True)
    print(f'CHAIN_RECORD {path}', flush=True)
    print(command('squeue', '-j', ','.join(ids), '-o', '%.18i %.24j %.10T %.30R'))


def load_transactions(checkout):
    # Import the exact stdlib validator without importing verl/torch on login
    # or before the job's ML environment has been activated.
    source = Path(checkout) / 'verl/utils/checkpoint/checkpoint_transaction.py'
    spec = importlib.util.spec_from_file_location('resume_checkpoint_transaction', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_run(run, job, commit):
    manifest = read_json(run / 'RUN_MANIFEST.json')
    required = {'env': 'appworld', 'job_id': job, 'code_commit': commit,
                'initial_checkpoint': str(MODEL_PATH), 'reference_schedule_epochs': 150,
                'seed': 0, 'resource_protocol': 'bounded_runtime_v1'}
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise RuntimeError(f'Incompatible source run {run}: {field}={manifest.get(field)!r}')
    if Path(manifest.get('checkpoint_dir', '')).resolve() != (run / 'checkpoints').resolve():
        raise RuntimeError(f'Unexpected checkpoint directory in {run}')
    if (run / 'actual_command.txt').read_text().strip() != 'bash experiments/gigpo_diag/launch_reference.sh vllm':
        raise RuntimeError(f'Source run is not the unchanged full training entry: {run}')
    targets = set()
    with (run / 'train.log').open(errors='replace') as stream:
        for line in stream:
            targets.update(int(value) for value in re.findall(r'Total training steps:\s*(\d+)', line))
    if len(targets) != 1 or min(targets) < 1:
        raise RuntimeError(f'Cannot establish native total training steps in {run}: {targets}')
    return targets.pop()


def select_checkpoint(runs_root, jobs, commit, tx):
    candidates, skipped, targets = [], [], set()
    for order, job in enumerate(jobs):
        for run in sorted(Path(runs_root).glob(f'appworld-qwen25-7b-gigpo-{numeric_job(job)}-*')):
            if run.is_symlink() or not run.is_dir():
                continue
            # Failed initialization without any published checkpoint is not a
            # source. A published but incompatible run must never be ignored.
            paths = list((run / 'checkpoints').glob('global_step_*'))
            if not paths:
                continue
            targets.add(validate_run(run, job, commit))
            for checkpoint in paths:
                match = re.fullmatch(r'global_step_([1-9][0-9]*)', checkpoint.name)
                if not match or checkpoint.is_symlink() or not checkpoint.is_dir():
                    continue
                try:
                    manifest = tx.validate_checkpoint(checkpoint)
                    if manifest['components'] != ['actor']:
                        raise ValueError('GiGPO checkpoint must contain only the actor role')
                    if manifest['global_step'] != int(match[1]):
                        raise ValueError('Directory/checkpoint step mismatch')
                    tx.validate_component(checkpoint / 'actor', world_size=4)
                except (OSError, ValueError, KeyError) as error:
                    skipped.append({'path': str(checkpoint), 'reason': str(error)})
                    continue
                candidates.append((manifest['global_step'], order, run.name, str(checkpoint)))
    if not candidates:
        raise RuntimeError(f'No complete four-GPU checkpoint for jobs {jobs}; refusing to restart from initial model. Rejected: {skipped}')
    if len(targets) != 1:
        raise RuntimeError(f'Inconsistent native schedules in chain: {targets}')
    step, _, _, checkpoint = max(candidates)
    target = targets.pop()
    if step > target:
        raise RuntimeError(f'Checkpoint update {step} exceeds the native target {target}')
    return {'status': 'already_complete' if step == target else 'resume',
            'checkpoint': checkpoint, 'global_step': step, 'total_training_steps': target,
            'source_jobs': jobs, 'skipped_checkpoints': skipped}


def training_environment(decision, acceptance, parent, slot, job):
    env = dict(os.environ)
    # A submitted shell may still contain variables from a verification job.
    checkpoint = decision['checkpoint']
    step = decision['global_step']
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = f'appworld-qwen25-7b-gigpo-{job}-{stamp}-chain{parent}-r{slot}-from{step}'
    # Never mix replayed updates with the discarded suffix of an earlier run.
    # These variables are supported by the existing SDK; no upgrade is needed.
    for key in ('SWANLAB_RUN_ID', 'SWANLAB_RUN_DIR', 'SWANLAB_RUN_PARALLEL', 'SWANLAB_LOAD'):
        env.pop(key, None)
    env.update(SWANLAB_RESUME='never', SWANLAB_GROUP=f'appworld-chain-{parent}',
               SWANLAB_TAGS=f'chain-{parent},resume-{slot},from-step-{step}',
               SWANLAB_DESCRIPTION=(f'Continuation {slot}/3 of Slurm job {parent}; current job {job}. '
                                    f'Restored full training state from {checkpoint}. '
                                    f'Metrics use absolute global steps, starting after {step}. '
                                    'Replayed updates are kept in this separate run; earlier run history is unchanged.'))
    env.update(GIGPO_EXPECTED_COMMIT=TRAINING_COMMIT, GIGPO_RUN_ID=run_id,
               GIGPO_PREFLIGHT_ONLY='0', GIGPO_RESUME_MODE='resume_path',
               GIGPO_RESUME_FROM=checkpoint, GIGPO_ACCEPTANCE_DIR=acceptance,
               GIGPO_ACCEPTANCE_REUSE_FROM=ACCEPTANCE_COMMIT, SLURM_EXPORT_ENV='ALL')
    return env


def run_resume(path, slot):
    path = Path(path).resolve()
    chain = read_json(path)
    job = numeric_job(os.environ['SLURM_JOB_ID'])
    evidence = path.parent / 'starts' / f'{job}-{time.time_ns()}.json'
    record = {'job_id': job, 'slot': slot, 'chain': str(path)}
    try:
        if chain['version'] != 1 or not 1 <= slot <= COUNT:
            raise RuntimeError('Unsupported chain or slot')
        if chain['training_commit'] != TRAINING_COMMIT or Path(chain['training_checkout']) != TRAINING_CHECKOUT:
            raise RuntimeError('Chain training checkout/commit differs from the pinned baseline')
        if Path(chain['runs_root']) != RUNS_ROOT or Path(chain['acceptance_dir']) != ACCEPTANCE_DIR:
            raise RuntimeError('Unexpected run/acceptance directory')
        if Path(chain['helper_checkout']).resolve() != HELPER_CHECKOUT:
            raise RuntimeError('Wrong helper checkout')
        verify_checkout(HELPER_CHECKOUT, chain['helper_commit'])
        verify_checkout(TRAINING_CHECKOUT, TRAINING_COMMIT)
        # The job can start between sbatch accepting it and returning its ID.
        current = chain['jobs'][slot]
        if current['status'] not in ('submitting', 'submitted') or current.get('job_id', job) != job:
            raise RuntimeError('Job ID does not match the recorded slot')
        previous = [numeric_job(row['job_id']) for row in chain['jobs'][:slot]]
        # Query the user's queue rather than -j: older completed IDs may have
        # aged out of slurmctld after several 17h50 allocations.
        queued = command('squeue', '-h', '-u', os.environ['USER'], '-o', '%i %T')
        active = '\n'.join(line for line in queued.splitlines() if line.split() and line.split()[0] in previous)
        if active:
            raise RuntimeError(f'Predecessor jobs are still active: {active}')
        tx = load_transactions(TRAINING_CHECKOUT)
        decision = select_checkpoint(RUNS_ROOT, previous + [job], TRAINING_COMMIT, tx)
        env = training_environment(decision, chain['acceptance_dir'], chain['parent_job'], slot, job)
        record.update(decision, training_commit=TRAINING_COMMIT,
                      helper_commit=chain['helper_commit'],
                      run_id=env['GIGPO_RUN_ID'], swanlab_policy='separate_run_absolute_steps',
                      swanlab_group=env['SWANLAB_GROUP'])
        write_json(evidence, record)
        print('RESUME_DECISION ' + json.dumps(record), flush=True)
        if decision['status'] == 'already_complete':
            print('TRAINING_ALREADY_COMPLETE: no model/environment processes started', flush=True)
            return
        os.chdir(TRAINING_CHECKOUT)
        os.execvpe('bash', ['bash', 'experiments/gigpo_diag/train.sbatch'],
                  env)
    except Exception as error:
        record.update(status='failed_before_training', error_type=type(error).__name__, error=str(error))
        write_json(evidence, record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    submit = commands.add_parser('submit')
    submit.add_argument('--parent-job', required=True)
    run = commands.add_parser('run')
    run.add_argument('--chain', type=Path, required=True)
    run.add_argument('--slot', type=int, required=True)
    args = parser.parse_args()
    if args.action == 'submit':
        submit_chain(args.parent_job)
    else:
        run_resume(args.chain, args.slot)


if __name__ == '__main__':
    main()
