"""HPC gate: eight real bounded AppWorld services, no Ray cluster or GPU.

The outer process limits the full gate to eight minutes.  Its child owns a new
process group, so timeout cleanup targets only this gate and its service children.
The actual AppWorldWorker is imported normally; no Ray initialization, GPU model
creation, or rollout collector is needed. All HTTP/control-plane calls are real.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiments.gigpo_diag.appworld_resource import (
    InfrastructureFailure, atomic_json, load_profile, read_json,
)


GROUP_SIZE = 8
GATE_TIMEOUT_SECONDS = 480
DANGER_CODE = ('protocol_before_stall = "old episode only"\n'
               'protocol_counter = 0\nwhile True:\n    try:\n        while True:\n'
               '            protocol_counter += 1\n    except:\n        pass')
TASK_STATE_CODE = 'print(json.dumps(apis.supervisor.show_active_task(), sort_keys=True))'


def require(condition, detail):
    if not condition:
        raise RuntimeError('AppWorld live protocol gate: ' + detail)


def events(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def record(event, **details):
    path = Path(os.environ['GIGPO_RUN_DIR']) / 'events.jsonl'
    row = {'event': event, 'clock': time.monotonic(), **details}
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()
    print(json.dumps(row, ensure_ascii=False), flush=True)


def load_native_worker():
    """Import the production Worker through its normal package path."""
    from agent_system.environments.env_package.appworld.envs import AppWorldWorker
    import ray
    import torch

    path = Path(__file__).resolve().parents[2] / 'agent_system/environments/env_package/appworld/envs.py'
    source = path.read_text(encoding='utf-8')
    require(not ray.is_initialized(), 'import unexpectedly initialized Ray')
    require(not torch.cuda.is_initialized(), 'import unexpectedly initialized CUDA')
    return AppWorldWorker, hashlib.sha256(source.encode('utf-8')).hexdigest()


def assert_healthy(result, expected=None):
    observation, reward, done, info = result
    require(reward == 0.0 and done is False, 'read-only step unexpectedly terminated or rewarded')
    require(info['resource_terminal'] is False, 'healthy peer became a resource terminal')
    require(not observation.startswith('Execution failed.'), 'healthy action failed: ' + observation[:500])
    if expected is not None:
        require(expected in observation, 'healthy output does not preserve expected state')
    return observation


def verify_service_limit(root, entry, profile):
    limit = read_json(root / f'limit-{entry["pid"]}.json', {})
    require(limit.get('pid') == entry['pid'], 'missing production launcher limit evidence')
    require(limit.get('rlimit_as') == [profile['service_as_bytes']] * 2,
            'service does not have the calibrated hard AS limit')


def run_child():
    require(sys.platform == 'linux', 'requires Linux resource limits')
    profile = load_profile()  # No dangerous dispatch without a calibrated profile.
    require(profile['hard_execute_seconds'] == 140, 'production hard deadline must remain 140s')
    require(profile.get('cases'), 'calibrated profile has no sampled task')
    task = profile['cases'][0][1]
    destination = Path(os.environ['GIGPO_RUN_DIR']).resolve()
    service_root = destination / 'services'
    service_root.mkdir(parents=True, exist_ok=True)
    from appworld import AppWorld, update_root
    from experiments.gigpo_diag import services

    update_root(os.environ['APPWORLD_ROOT'])
    require(AppWorld.init_defaults.timeout_seconds == 100, 'native soft timeout is not 100s')
    Worker, source_hash = load_native_worker()
    require(not services._owned and services._supervisor is None,
            'gate process unexpectedly already owns service processes')
    workers = []
    result_summary = None
    try:
        record('starting_services', task_id=task, group_size=GROUP_SIZE,
               profile_path=os.environ['GIGPO_RESOURCE_PROFILE'])
        with concurrent.futures.ThreadPoolExecutor(max_workers=GROUP_SIZE) as pool:
            entries = list(pool.map(lambda index: services._start_one(index, service_root),
                                    range(GROUP_SIZE)))
        for entry in entries:
            verify_service_limit(service_root, entry, profile)
        atomic_json(service_root / 'manifest.json', {'required_instances': GROUP_SIZE, 'services': entries})
        services._supervisor = services.ServiceSupervisor(entries, service_root, profile)
        workers = [Worker(index, 50, entry['port']) for index, entry in enumerate(entries)]
        # Remote client constructors share Python's task caches, so initialize
        # serially. The actual eight service processes are independent.
        resets = [worker.reset(task) for worker in workers]
        for _, info in resets:
            require(info['code_timeout_seconds'] == 100, 'worker changed native 100s deadline')
            require(info['execute_http_timeout_seconds'] == 160, 'worker changed HTTP 160s deadline')
        require(len({worker.service_pid for worker in workers}) == GROUP_SIZE, 'services share a PID')
        require(len({worker.experiment_name for worker in workers}) == GROUP_SIZE, 'episodes share outputs')
        require(len({observation for observation, _ in resets}) == 1, 'same task has different initial instructions')
        initial_states = []
        for worker in workers:
            result = worker.step('protocol_state = [1]\n' + TASK_STATE_CODE)
            initial_states.append(json.loads(assert_healthy(result)))
        require(all(state == initial_states[0] for state in initial_states), 'native initial task states disagree')
        before_identity = {'pid': workers[0].service_pid, 'experiment': workers[0].experiment_name,
                           'generation': workers[0].service_generation}
        danger_hash = hashlib.sha256(DANGER_CODE.encode('utf-8')).hexdigest()
        record('danger_dispatch', task_id=task, worker_slot=0, old_identity=before_identity,
               code_sha256=danger_hash, hard_execute_seconds=140, group_size=GROUP_SIZE)
        # These are the actual Worker.step methods and HTTP requests, not Ray or
        # RPC mocks. A production resource terminal must leave seven peers usable.
        with concurrent.futures.ThreadPoolExecutor(max_workers=GROUP_SIZE) as pool:
            futures = [pool.submit(worker.step, DANGER_CODE if slot == 0 else
                                   'protocol_state.append(2)\nprint(protocol_state)')
                       for slot, worker in enumerate(workers)]
            results = [future.result() for future in futures]
        terminal = results[0]
        require(sum(bool(result[2]) for result in results) == 1, 'expected exactly one terminated group member')
        require(terminal[1] == 0.0 and terminal[2] is True, 'resource terminal did not return zero/done')
        terminal_info = terminal[3]
        require(terminal_info['resource_terminal'] is True, 'missing resource-terminal classification')
        require(terminal_info['native_evaluator_success'] is None, 'killed service invented evaluator success')
        require(terminal_info['budgeted_task_success'] is False, 'killed service invented budgeted success')
        require(terminal_info['action_retried'] is False, 'terminal action was retried')
        require(terminal_info['resource_event']['reason'] == 'hard_execute_deadline',
                'bare-except case did not reach the production hard deadline')
        require(terminal_info['resource_event']['request']['code_sha256'] == danger_hash,
                'terminal evidence belongs to a different request')
        for result in results[1:]:
            assert_healthy(result, '[1, 2]')
        record('resource_terminal_verified', info=terminal_info, peers_alive=7)

        dead_worker = workers[0]
        dead_client = dead_worker.env
        original_rpc = dead_client._remote_environment_call
        rpc_calls = []

        def observe_rpc(*args, **kwargs):
            rpc_calls.append(str(args[0]) if args else '<no method>')
            return original_rpc(*args, **kwargs)

        dead_client._remote_environment_call = observe_rpc
        try:
            count_before = dead_worker.current_step_count
            sticky = dead_worker.step('raise RuntimeError("must never execute")')
        finally:
            dead_client._remote_environment_call = original_rpc
        require(not rpc_calls, 'sticky terminal performed another RPC')
        require(sticky[1:3] == (0.0, True) and sticky[3]['inactive'] is True,
                'sticky terminal did not preserve inactive zero-reward behavior')
        require(dead_worker.current_step_count == count_before, 'sticky step consumed another interaction')
        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
            continued = list(pool.map(lambda worker: worker.step('protocol_state.append(3)\nprint(protocol_state)'),
                                      workers[1:]))
        for result in continued:
            assert_healthy(result, '[1, 2, 3]')
        record('peers_and_sticky_verified', peers_next_cell_succeeded=7, sticky_rpc_calls=rpc_calls)

        dead_worker.reset(task)
        after_identity = {'pid': dead_worker.service_pid, 'experiment': dead_worker.experiment_name,
                          'generation': dead_worker.service_generation}
        require(after_identity['pid'] != before_identity['pid'], 'reset reused killed process')
        require(after_identity['experiment'] != before_identity['experiment'], 'reset reused episode outputs')
        require(after_identity['generation'] > before_identity['generation'], 'service generation did not advance')
        verify_service_limit(service_root, after_identity, profile)
        require(dead_worker.current_step_count == 0 and dead_worker.env.num_interactions == 0,
                'fresh episode already contains replayed interactions')
        missing_state = dead_worker.step('print(protocol_state)')
        require('NameError' in missing_state[0] and missing_state[2] is False,
                'fresh shell preserved old policy variables or replayed actions')
        require(missing_state[3]['resource_terminal'] is False, 'native NameError became resource terminal')
        new_task_state = json.loads(assert_healthy(dead_worker.step(TASK_STATE_CODE)))
        require(new_task_state == initial_states[0], 'reset did not restore initial native task state')
        old_executions = [row for row in events(service_root / f'execute-{before_identity["pid"]}.jsonl')
                          if row.get('event') == 'execute_begin' and row.get('code_sha256') == danger_hash]
        new_executions = [row for row in events(service_root / f'execute-{after_identity["pid"]}.jsonl')
                          if row.get('event') == 'execute_begin']
        require(len(old_executions) == 1, 'danger action was dispatched more than once')
        require(len(new_executions) == 2 and all(row.get('code_sha256') != danger_hash for row in new_executions),
                'new service contains unexpected replayed actions')
        record('fresh_reset_verified', old_identity=before_identity, new_identity=after_identity,
               fresh_executions=2, old_danger_executions=1)

        # Kill one of this process's idle service children. No matching resource
        # terminal exists; the next action must abort as unknown infrastructure.
        idle_worker = workers[1]
        idle_pid = idle_worker.service_pid
        process = next(process for process, _ in services._owned if process.pid == idle_pid)
        require(process.poll() is None, 'idle service already exited')
        require(read_json(service_root / f'request-{idle_pid}.json', {}).get('status') == 'complete',
                'selected service is not idle after a completed request')
        require(not (service_root / f'terminal-{idle_pid}.json').exists(),
                'idle service already has terminal evidence')
        process.kill()
        process.wait(timeout=5)
        record('owned_idle_service_killed', pid=idle_pid, worker_slot=1)
        try:
            idle_worker.step('print("must not become an ordinary failure reward")')
        except InfrastructureFailure as error:
            unknown_event = error.event
        else:
            raise RuntimeError('Idle service death did not raise InfrastructureFailure')
        require(idle_worker._service_dead is True, 'unknown service death was not recorded')
        require(not idle_worker._done, 'unknown service death was recycled as a normal terminal')
        record('infrastructure_failure_verified', resource_event=unknown_event)
        result_summary = {
            'passed': True, 'protocol': 'bounded_runtime_v1', 'group_size': GROUP_SIZE,
            'task_id': task, 'native_http_services': True, 'ray_cluster_started': False,
            'model_or_gpu_loaded': False, 'worker_source_sha256': source_hash,
            'worker_loading': 'normal production package import; real HTTP client and resource dependencies',
            'soft_seconds': 100, 'hard_seconds': 140, 'http_seconds': 160,
            'one_resource_terminal_zero_reward': True, 'native_evaluator_success_on_terminal': None,
            'peers_next_cell_succeeded': 7, 'sticky_rpc_calls': 0,
            'fresh_reset': {'old': before_identity, 'new': after_identity, 'replayed_actions': 0},
            'unknown_idle_exit_raised_infrastructure_failure': True,
            'logs': str(service_root), 'profile': os.environ['GIGPO_RESOURCE_PROFILE'],
        }
    finally:
        # workers.close uses only that worker's own endpoint and skips dead ones.
        for worker in workers:
            try:
                worker.close()
            except Exception as error:
                record('close_rpc_error', worker_slot=worker.worker_id, error_type=type(error).__name__)
        services.stop_services()
    require(result_summary is not None, 'gate did not complete')
    result_summary['owned_services_stopped'] = all(process.poll() is not None for process, _ in services._owned)
    require(result_summary['owned_services_stopped'], 'owned service survived gate cleanup')
    atomic_json(destination / 'LIVE_PROTOCOL_PASSED.json', result_summary)
    record('completed', **result_summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        run_child()
        return
    require(sys.platform == 'linux', 'requires Linux process groups and resource limits')
    profile_path = Path(os.environ['GIGPO_RESOURCE_PROFILE']).resolve(strict=True)
    load_profile()
    destination = Path(os.environ['GIGPO_RUN_DIR']).resolve() / 'live-protocol'
    destination.mkdir(parents=True, exist_ok=False)
    root = destination / 'appworld'
    root.mkdir()
    data = Path(os.environ['GIGPO_APPWORLD_DATA']).resolve() / 'data'
    require(data.is_dir(), 'native task data directory does not exist')
    (root / 'data').symlink_to(data, target_is_directory=True)
    environment = dict(os.environ)
    environment.update({
        'GIGPO_RESOURCE_PROFILE': str(profile_path), 'GIGPO_RUN_DIR': str(destination),
        'GIGPO_RUN_ID': os.environ['GIGPO_RUN_ID'] + '-live-protocol',
        'APPWORLD_ROOT': str(root), 'GIGPO_DIAG_DIR': str(destination / 'diagnostics'),
        'APPWORLD_PORTS_FILE': str(destination / 'ports'),
    })
    timed_out = False
    with (destination / 'gate.log').open('wb') as log:
        child = subprocess.Popen([sys.executable, '-m', __package__ + '.check_appworld_live_protocol', '--child'],
                                 env=environment, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            child.wait(timeout=GATE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            # Only this new session/process group and its own service children.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=10)
        finally:
            if child.poll() != 0:
                # Also clean children if the gate process itself crashed/OOMed
                # before its finally block. They remain in its unique group.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                if child.poll() is None:
                    child.wait(timeout=10)
    result = read_json(destination / 'LIVE_PROTOCOL_PASSED.json', {})
    if child.returncode or timed_out or result.get('passed') is not True:
        atomic_json(destination / 'LIVE_PROTOCOL_FAILED.json',
                    {'passed': False, 'returncode': child.returncode,
                     'external_timeout': timed_out, 'log': str(destination / 'gate.log')})
        raise RuntimeError(f'Live protocol gate failed; inspect {destination}')
    print('APPWORLD_LIVE_PROTOCOL_PASSED', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
