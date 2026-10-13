"""Observe native service execution without changing its result or deadlines."""
import atexit
import faulthandler
import functools
import hashlib
import json
import os
from pathlib import Path
import time


def clock():
    # AppWorld freezes datetime and time.monotonic. Linux clock_gettime remains
    # a real elapsed clock; perf_counter is only the Windows test fallback.
    if hasattr(time, 'clock_gettime'):
        return time.clock_gettime(time.CLOCK_MONOTONIC)
    return time.perf_counter()


class ExecutionTrace:
    def __init__(self, directory, stack_after_seconds):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.journal = (directory / f'execute-{os.getpid()}.jsonl').open('a', encoding='utf-8')
        self.stacks = (directory / f'execute-{os.getpid()}.stacks.log').open('a', encoding='utf-8')
        self.stack_after_seconds = stack_after_seconds
        self.serial = 0
        self.active = None
        self.available = True

    def write(self, event, **fields):
        try:
            if self.available:
                row = {'event': event, 'pid': os.getpid(), 'elapsed_clock': clock(),
                       'request': self.active, **fields}
                self.journal.write(json.dumps(row, ensure_ascii=False) + '\n')
                self.journal.flush()
        except Exception:
            self.available = False  # Diagnostics must not replace native feedback/errors.

    def arm_stack(self):
        try:
            # Native C watchdog: it can dump Python stacks while the server's
            # event loop is blocked. It does not send SIGALRM or stop execution.
            faulthandler.dump_traceback_later(self.stack_after_seconds, file=self.stacks,
                                            repeat=False, exit=False)
            return True
        except Exception:
            return False

    def cancel_stack(self):
        try:
            faulthandler.cancel_dump_traceback_later()
        except Exception:
            pass

    def close(self):
        self.cancel_stack()
        self.journal.close()
        self.stacks.close()


def install_execution_trace(directory=None, stack_after_seconds=110):
    from appworld import AppWorld
    from appworld.common.safety_guard import SafetyGuard

    if directory is None:
        run_dir = os.environ.get('GIGPO_RUN_DIR')
        if not run_dir:
            return None
        directory = Path(run_dir) / 'services'
    installed = getattr(AppWorld.execute, '_gigpo_execution_trace', None)
    if installed is not None:
        return installed
    try:
        trace = ExecutionTrace(directory, stack_after_seconds)
    except Exception as exc:
        print(f'APPWORLD_EXECUTION_TRACE unavailable: {type(exc).__name__}', flush=True)
        return None

    def wrap_phase(cls, name, label):
        original = getattr(cls, name)

        @functools.wraps(original)
        def observed(*args, **kwargs):
            if trace.active is None:
                return original(*args, **kwargs)
            trace.write('phase_begin', phase=label)
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                trace.write('phase_end', phase=label, status='raised', error_type=type(exc).__name__)
                raise
            trace.write('phase_end', phase=label, status='returned')
            return result

        setattr(cls, name, observed)

    wrap_phase(SafetyGuard, 'is_syntax_safe', 'syntax_check')
    wrap_phase(AppWorld, '_shell_run_cell', 'shell_run_cell')
    wrap_phase(AppWorld, '_save_state', 'save_state')
    wrap_phase(AppWorld, 'save_logs', 'save_logs')
    original = AppWorld.execute

    @functools.wraps(original)
    def execute(self, code):
        if self.remote_environment_url:
            return original(self, code)
        trace.serial += 1
        trace.active = {'serial': trace.serial, 'task_id': self.task_id,
                        'experiment_name': getattr(self, 'experiment_name', None),
                        'interactions_before': self.num_interactions}
        trace.write('execute_begin', code=code,
                    code_sha256=hashlib.sha256(code.encode('utf-8')).hexdigest() if isinstance(code, str) else None)
        armed = trace.arm_stack()
        try:
            result = original(self, code)
        except BaseException as exc:
            trace.write('execute_end', status='raised', error_type=type(exc).__name__)
            raise
        else:
            trace.write('execute_end', status='returned', interactions_after=self.num_interactions)
            return result
        finally:
            if armed:
                trace.cancel_stack()
            trace.active = None

    execute._gigpo_execution_trace = trace
    AppWorld.execute = execute
    atexit.register(trace.close)
    print(f'APPWORLD_EXECUTION_TRACE enabled stack_after_seconds={stack_after_seconds}', flush=True)
    return trace
