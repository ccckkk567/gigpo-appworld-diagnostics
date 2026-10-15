"""Finite Linux preflight. Every child has hard AS and an external deadline.

Calibration uses legal native interactions before selecting the production limit.
The 60% environment envelope is a declared allocation policy, not a measurement
of trainer/checkpoint needs. Failure to fit stops the run; it never raises RAM.
"""
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.gigpo_diag.appworld_resource import atomic_json, proc_memory, cgroup_memory, PROTOCOL


def run_probe(directory, task, mode, limit, iterations=20, policy_limit=None):
    directory.mkdir(parents=True, exist_ok=False)
    root = directory / 'appworld'
    root.mkdir()
    (root / 'data').symlink_to(Path(os.environ['GIGPO_APPWORLD_DATA']) / 'data', target_is_directory=True)
    output = directory / 'events.jsonl'
    command = [sys.executable, '-m', 'experiments.gigpo_diag.appworld_service_probe',
               '--root', str(root), '--task-id', task, '--experiment', 'bounded-probe',
               '--as-bytes', str(limit), '--output', str(output), '--mode', mode,
               '--supervisor-pid', str(os.getpid()), '--iterations', str(iterations),
               '--soft-seconds', '100' if mode in ('healthy', 'retained-pressure') else '2']
    if policy_limit is not None:
        command.extend(['--policy-as-bytes', str(policy_limit)])
    peaks = {'rss_bytes': 0, 'vms_bytes': 0}
    start = time.monotonic()
    killed = False
    with (directory / 'stdout.log').open('wb') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        # Initialization has its own finite timeout. The dangerous portion gets
        # 10s, only after the child has durably announced that it entered it.
        danger_start = None
        try:
            while child.poll() is None:
                for key, value in proc_memory(child.pid).items():
                    peaks[key] = max(peaks[key], value or 0)
                events = read_events(output)
                if danger_start is None and any(e.get('event') == 'danger_started' for e in events):
                    danger_start = time.monotonic()
                danger_returned = any(e.get('event') == 'danger_returned' for e in events)
                if (time.monotonic() - start > 240 or
                        danger_start is not None and not danger_returned and time.monotonic() - danger_start > 10):
                    child.kill()
                    killed = True
                    break
                time.sleep(.05)
            child.wait(timeout=10)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)
    events = read_events(output)
    for e in events:
        peaks['rss_bytes'] = max(peaks['rss_bytes'], e.get('VmHWM_bytes', 0))
        peaks['vms_bytes'] = max(peaks['vms_bytes'], e.get('VmPeak_bytes', 0))
    expected_kill = mode == 'swallow-deadline'
    passed = (killed and danger_start is not None) if expected_kill else (
        child.returncode == 0 and any(e.get('event') == 'completed' and e.get('passed') for e in events))
    if not expected_kill:
        passed = passed and any(e.get('event') == 'evaluate_complete' for e in events)
    if mode == 'retained-pressure':
        passed = passed and any(e.get('event') == 'next_episode_complete' for e in events)
    result = {'mode': mode, 'task_id': task, 'limit_as_bytes': limit, 'external_kill': killed,
              'returncode': child.returncode, 'passed': bool(passed), 'sampled_peaks': peaks,
              'events': events, 'elapsed_seconds': time.monotonic() - start}
    atomic_json(directory / 'result.json', result)
    if not passed:
        raise RuntimeError(f'Bounded preflight failed: {directory}; inspect stdout and result')
    return result


def read_events(path):
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass  # A live writer's last partial line is not evidence.
    return events


def select_budgets(results, ceiling):
    """Keep measured cell headroom and reserve the remaining fixed envelope.

    A pressure probe still has to prove the reserve sufficient with live user
    objects. This calculation is not itself proof of evaluator correctness.
    """
    policy_peaks = []
    evaluation_deltas = []
    for result in results:
        events = result['events']
        before = next(e for e in events if e['event'] == 'evaluate_begin')
        after = next(e for e in events if e['event'] == 'evaluate_complete')
        policy_peaks.append(before['VmPeak_bytes'])
        evaluation_deltas.append(max(0, after['VmPeak_bytes'] - before['VmSize_bytes']))
    policy = math.ceil(max(policy_peaks) * 1.30 / 4096) * 4096
    hard = (ceiling // 4096) * 4096
    reserve = hard - policy
    required_reserve = math.ceil((max(evaluation_deltas) * 1.30 + 64 * 1024**2) / 4096) * 4096
    peak_vms = max(r['sampled_peaks']['vms_bytes'] for r in results)
    peak_rss = max(r['sampled_peaks']['rss_bytes'] for r in results)
    if min(peak_vms, peak_rss) <= 0 or reserve < required_reserve or peak_vms >= hard:
        raise RuntimeError(f'Native evaluation does not fit the fixed environment budget: '
                           f'policy={policy}, hard={hard}, reserve={reserve}, '
                           f'required_reserve={required_reserve}, measured_vms={peak_vms}. '
                           'No training started; RAM was not raised.')
    rss_watchdog = int(hard * .90)
    if peak_rss * 1.15 >= rss_watchdog:
        raise RuntimeError('Measured RSS leaves insufficient recovery headroom')
    return {'policy_as_bytes': policy, 'service_as_bytes': hard,
            'native_reserve_bytes': reserve, 'required_native_reserve_bytes': required_reserve,
            'measured_policy_peak_vms_bytes': max(policy_peaks),
            'measured_evaluation_delta_bytes': max(evaluation_deltas),
            'measured_peak_vms_bytes': peak_vms, 'measured_peak_rss_bytes': peak_rss,
            'service_rss_watchdog_bytes': rss_watchdog}


def main():
    if sys.platform != 'linux':
        raise RuntimeError('Native resource calibration requires Linux /proc and RLIMIT_AS')
    count = 16 * 8 + 16
    job_bytes = int(os.environ['SLURM_MEM_PER_NODE']) * 1024 * 1024
    if job_bytes != 320 * 1024**3:
        raise RuntimeError('This reviewed configuration requires exactly 320G per node')
    destination = Path(os.environ['GIGPO_RUN_DIR']) / 'resource_preflight'
    destination.mkdir(parents=True, exist_ok=False)
    envelope = int(job_bytes * .60)
    ceiling = envelope // count
    tasks = []
    for split in ('train', 'dev'):
        ids = (Path(os.environ['GIGPO_APPWORLD_DATA']) / 'data' / 'datasets' / (split + '.txt')).read_text().splitlines()
        ids = [s.strip().split(':', 1)[0] for s in ids if s.strip() and not s.lstrip().startswith('#')]
        tasks.extend((split, t) for t in ids[:2])
    if len(tasks) != 4:
        raise RuntimeError('Need two train and two dev tasks for measured calibration')
    # Regression from job 163354. This is extra calibration coverage, never a
    # filter or substitution in the actual training/dev datasets.
    tasks.append(('regression-163354', '6ea6792_1'))
    results = [run_probe(destination / f'baseline-{i}-{split}', task, 'healthy', ceiling)
               for i, (split, task) in enumerate(tasks)]
    budgets = select_budgets(results, ceiling)
    limit, policy = budgets['service_as_bytes'], budgets['policy_as_bytes']
    # Recheck legal operations under the selected production limit.
    for i, (split, task) in enumerate(tasks):
        run_probe(destination / f'limited-{i}-{split}', task, 'healthy', limit, policy_limit=policy)
        run_probe(destination / f'retained-{i}-{split}', task, 'retained-pressure', limit, policy_limit=policy)
    for mode in ('allocate', 'catch-memory', 'swallow-deadline'):
        # AppWorld's DB loader treats any path containing "memory" as an
        # in-memory connection, including task files below a disk root. Keep
        # the logical mode but avoid that substring in its physical directory.
        directory_name = 'catch-allocation' if mode == 'catch-memory' else mode
        run_probe(destination / directory_name, tasks[0][1], mode, limit, policy_limit=policy)
    versions = {p: importlib.metadata.version(p) for p in ('appworld', 'ipython', 'libcst')}
    profile = {'protocol': PROTOCOL, 'calibrated': True,
               'code_commit': os.environ['GIGPO_CODE_COMMIT'], 'packages': versions,
               'service_count': count, **budgets,
               'memory_budget_version': 'cell_with_native_reserve_v1',
               'native_evaluation_verified': True, 'retained_pressure_evaluation_verified': True,
               'hard_execute_seconds': 140, 'job_requested_bytes': job_bytes,
               'environment_envelope_bytes': envelope,
               'cgroup_before_training': cgroup_memory(), 'cases': tasks,
               'scope': 'sampled native interactions, save, evaluate and retained-pressure evaluation; not exhaustive task coverage'}
    atomic_json(os.environ['GIGPO_RESOURCE_PROFILE'], profile)
    print('BOUNDED_RUNTIME_PREFLIGHT_PASSED', json.dumps(profile), flush=True)


if __name__ == '__main__':
    main()
