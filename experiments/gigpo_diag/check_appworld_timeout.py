"""Bounded CPU-only native-shell preflight; no task DB, model, or GPU needed."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import types


def require(condition, detail):
    if not condition:
        raise RuntimeError('AppWorld timeout preflight: ' + detail)


def probe(pause=None):
    from appworld import AppWorld
    from appworld.common.safety_guard import SafetyGuard
    from IPython.core.interactiveshell import InteractiveShell
    from experiments.gigpo_diag.appworld_timeout import install_cell_deadline

    install_cell_deadline()
    world = AppWorld.__new__(AppWorld)
    world.remote_environment_url = None
    world.remote_mcp_url = None
    world.raise_on_unsafe_syntax = True
    world.raise_on_unsafe_execution = True
    world.safety_guard = SafetyGuard()
    world.requester = types.SimpleNamespace(reset_request_count=lambda: None)
    world.timeout_seconds = 1  # Probe only; training keeps native 100 seconds.
    world.num_interactions = 0
    world.num_sub_interactions = 0
    world.max_interactions = 50
    world.environment_io = []
    world.output_db_home_path_on_disk = 'synthetic-no-database'
    saves = []
    world._save_state = lambda directory: saves.append('state')
    world.save_logs = lambda: saves.append('logs')
    world.shell = InteractiveShell()
    calls = []
    event = threading.Event()
    if pause is None:
        pause = event.wait

    def failing_api():
        calls.append(1)
        # Keep the signal in the try-protected call, without a busy loop or
        # captured-output growth. SafetyGuard replaces time.sleep, not this wait.
        pause(.01)
        raise ValueError('synthetic invalid API request')

    world.shell.user_ns['failing_api'] = failing_api
    code = 'while True:\n    try:\n        failing_api()\n    except Exception:\n        pass'
    handler = signal.getsignal(signal.SIGALRM)
    try:
        feedback = world.execute(code)
        require(feedback == 'Execution failed. Traceback:\nExecution timed out after 1 seconds.', feedback)
        require(world.num_interactions == 1, 'timeout did not consume exactly one interaction')
        require(world.environment_io == [{'number': '1', 'input': code, 'output': feedback}],
                'original action/feedback was not recorded')
        require(saves == ['state', 'logs'], 'native save path was not completed')
        require(bool(calls), 'retry loop did not execute')
        require(signal.getsignal(signal.SIGALRM) == handler, 'signal handler was not restored')
        require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), 'timer was not canceled')
        require(world.execute('print(6 * 7)') == '42\n', 'next cell did not succeed')
        require(world.num_interactions == 2, 'next interaction count was incorrect')
        require(saves == ['state', 'logs', 'state', 'logs'], 'next cell was not saved')
    finally:
        world.safety_guard.disable()
    return {'status': 'passed', 'native_shell': True, 'safety_guard': True,
            'probe_timeout_seconds': 1, 'training_timeout_changed': False,
            'task_databases_used': False, 'timeout_turns': 1,
            'next_cell_succeeded': True}


def main():
    if sys.argv[1:] == ['--child']:
        print(json.dumps(probe()), flush=True)
        return
    # A broken alarm implementation must not leave a preflight hanging either.
    result = subprocess.run([sys.executable, '-m', __package__ + '.check_appworld_timeout', '--child'],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError('AppWorld timeout preflight failed:\n' + (result.stdout + result.stderr)[-6000:])
    evidence = json.loads(result.stdout.strip().splitlines()[-1])
    if evidence.get('status') != 'passed':
        raise RuntimeError(f'Unexpected timeout preflight result: {evidence}')
    if os.environ.get('GIGPO_RUN_DIR'):
        (Path(os.environ['GIGPO_RUN_DIR']) / 'appworld_timeout_preflight.json').write_text(
            json.dumps(evidence, indent=2), encoding='utf-8')
    print('APPWORLD_TIMEOUT_PREFLIGHT_PASSED ' + json.dumps(evidence), flush=True)


if __name__ == '__main__':
    main()
