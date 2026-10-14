"""One externally supervised, hard-AS-limited native AppWorld CPU probe.

The parent must launch this as a subprocess with a wall-clock timeout and kill
the process if it does not exit.  Danger modes are not importable auto-tests and
never run before RLIMIT_AS has been applied and the supervising PID checked.
Each case needs its own root/experiment/output.  No model or GPU is loaded.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time


MODES = ('healthy', 'allocate', 'catch-memory', 'swallow-deadline')


def memory_snapshot():
    values = {}
    status = Path('/proc/self/status')
    if status.exists():
        for line in status.read_text().splitlines():
            key, _, value = line.partition(':')
            if key in ('VmRSS', 'VmSize', 'VmHWM', 'VmPeak'):
                values[key + '_bytes'] = int(value.split()[0]) * 1024
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--as-bytes', required=True, type=int)
    parser.add_argument('--output', required=True, help='Append-only JSONL evidence path')
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--supervisor-pid', required=True, type=int)
    parser.add_argument('--soft-seconds', type=int)
    parser.add_argument('--iterations', default=20, type=int)
    args = parser.parse_args(argv)
    if args.as_bytes <= 0 or not 1 <= args.iterations <= 100:
        parser.error('positive AS bytes and 1..100 iterations are required')
    if args.soft_seconds is None:
        args.soft_seconds = 100 if args.mode == 'healthy' else 2
    if args.soft_seconds <= 0:
        parser.error('soft-seconds must be positive')
    if not args.experiment or any(character in args.experiment for character in '/\\'):
        parser.error('experiment must be one directory name')
    return args


def set_hard_limit(args):
    if sys.platform != 'linux':
        raise RuntimeError('This probe requires Linux RLIMIT_AS and /proc')
    if os.getppid() != args.supervisor_pid or args.supervisor_pid <= 1:
        raise RuntimeError('The specified external supervisor is not the parent process')
    import resource

    _, old_hard = resource.getrlimit(resource.RLIMIT_AS)
    if old_hard != resource.RLIM_INFINITY and args.as_bytes > old_hard:
        raise RuntimeError('Probe AS limit exceeds inherited hard limit')
    resource.setrlimit(resource.RLIMIT_AS, (args.as_bytes, args.as_bytes))
    if resource.getrlimit(resource.RLIMIT_AS) != (args.as_bytes, args.as_bytes):
        raise RuntimeError('RLIMIT_AS was not installed exactly')


def ensure_root(args):
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    data = root / 'data'
    if not data.exists():
        source_value = os.environ.get('GIGPO_APPWORLD_DATA')
        if not source_value:
            raise RuntimeError('root/data missing and GIGPO_APPWORLD_DATA is unset')
        source = Path(source_value).resolve() / 'data'
        if not source.is_dir():
            raise RuntimeError('GIGPO_APPWORLD_DATA is not a directory')
        data.symlink_to(source, target_is_directory=True)
    os.environ['APPWORLD_ROOT'] = str(root)
    return root


def emit(args, event, **details):
    # AppWorld freezes datetime/time for task reproducibility. clock_gettime_ns
    # stays an OS observation clock rather than a task-world timestamp.
    clocks = ({'wall_time_ns': time.clock_gettime_ns(time.CLOCK_REALTIME),
               'monotonic_ns': time.clock_gettime_ns(time.CLOCK_MONOTONIC)}
              if sys.platform == 'linux' else {})
    record = {'event': event, 'mode': args.mode, 'pid': os.getpid(),
              **clocks,
              'as_bytes': args.as_bytes, 'soft_seconds': args.soft_seconds,
              'task_id': args.task_id, **memory_snapshot(), **details}
    line = json.dumps(record, ensure_ascii=True, separators=(',', ':'))
    with Path(args.output).open('a', encoding='utf-8') as stream:
        stream.write(line + '\n')
        stream.flush()
    print(line, flush=True)


def require(condition, detail):
    if not condition:
        raise RuntimeError(detail)


def execute_checked(world, code, expected=None, failure=False):
    before = world.num_interactions
    message = world.execute(code)
    require(isinstance(message, str), 'native feedback is not text')
    require(world.num_interactions == before + 1, 'action did not consume exactly one turn')
    require(world.environment_io[-1]['input'] == code.strip(), 'original input was not saved')
    require(world.environment_io[-1]['output'] == message.rstrip(), 'feedback was not saved')
    require(message.startswith('Execution failed.') == failure,
            'unexpected native feedback category: ' + message[:400])
    if expected is not None:
        require(expected in message, 'expected feedback not found: ' + message[:400])
    return message


def run_probe(args):
    set_hard_limit(args)  # Must precede every AppWorld/IPython import.
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    emit(args, 'begin', passed=False, supervisor_pid=args.supervisor_pid)
    root = ensure_root(args)
    require(not (root / 'experiments' / 'outputs' / args.experiment).exists(),
            'Probe experiment already exists; use a fresh isolated name')
    from appworld import AppWorld, update_root
    from experiments.gigpo_diag.appworld_server import install_cst_rejection_handler
    from experiments.gigpo_diag.appworld_timeout import install_cell_deadline
    from experiments.gigpo_diag.appworld_exception_cleanup import install_exception_cleanup

    update_root(str(root))
    install_cst_rejection_handler()
    install_cell_deadline()
    install_exception_cleanup()
    emit(args, 'world_begin', passed=False)
    world = AppWorld(task_id=args.task_id, experiment_name=args.experiment,
                     max_interactions=args.iterations + 20, timeout_seconds=args.soft_seconds)
    # This root/experiment is exclusive to the probe.  Constructor and every
    # execute use the actual task DB, state persistence and environment/API logs.
    logs = Path(world.output_logs_directory)
    emit(args, 'ready', passed=False, num_interactions=world.num_interactions,
         logs_directory=str(logs))
    try:
        execute_checked(world, 'probe_state = {"values": [1]}\n'
                        'def probe_add(value):\n    probe_state["values"].append(value)\n'
                        'probe_task = apis.supervisor.show_active_task()\n'
                        'print("NATIVE_API_OK", isinstance(probe_task, dict))',
                        'NATIVE_API_OK True')
        emit(args, 'normal_api_complete', passed=False,
             num_interactions=world.num_interactions)
        if args.mode == 'healthy':
            for index in range(args.iterations):
                code = ('def probe_small_error():\n'
                        '    local_payload = bytearray(65536)\n'
                        '    probe_add(2)\n'
                        '    raise ValueError("bounded native probe error")\n'
                        'probe_small_error()')
                execute_checked(world, code, 'ValueError: bounded native probe error', failure=True)
                emit(args, 'small_error_complete', iteration=index + 1, passed=False,
                     num_interactions=world.num_interactions)
            execute_checked(world, 'probe_add(3)\nprint(probe_state["values"])',
                            str([1] + [2] * args.iterations + [3]))
        else:
            # Pure-Python requests avoid remote side effects.  Payloads are local
            # so a recoverable failure can actually be released by cleanup.
            allocation = ('def probe_allocate():\n    local_queue = []\n'
                          '    while True:\n        local_queue.append(bytearray(1048576))\n'
                          'probe_allocate()')
            retry_allocation = ('def probe_allocate():\n    local_queue = []\n'
                                '    while True:\n        try:\n'
                                '            local_queue.append(bytearray(1048576))\n'
                                '        except Exception:\n            pass\nprobe_allocate()')
            swallowed = ('probe_counter = 0\nwhile True:\n    try:\n        while True:\n'
                         '            probe_counter += 1\n    except:\n        pass')
            code = {'allocate': allocation, 'catch-memory': retry_allocation,
                    'swallow-deadline': swallowed}[args.mode]
            # Parent may only count an external timeout after this marker as a
            # swallowed-deadline case; import/initialization stalls are failures.
            emit(args, 'danger_started', passed=False, num_interactions=world.num_interactions)
            feedback = execute_checked(world, code, failure=True)
            if args.mode == 'swallow-deadline':
                raise RuntimeError('bare-except probe unexpectedly returned; no hard termination evidence')
            require('MemoryError' in feedback or 'Execution timed out' in feedback,
                    'allocation did not end through the expected native resource feedback')
            emit(args, 'danger_returned', passed=False,
                 feedback_type='MemoryError' if 'MemoryError' in feedback else 'deadline',
                 num_interactions=world.num_interactions)
            execute_checked(world, 'probe_add(3)\nprint(probe_state["values"])', '[1, 3]')
        world.save_logs()
        require((logs / 'environment_io.md').is_file(), 'native environment log missing')
        require((logs / 'api_calls.jsonl').is_file(), 'native API call log missing')
        require(bool(world.shell.user_ns['probe_task']), 'real supervisor API returned empty state')
        turns = world.num_interactions
        world.close()
        world = None
        emit(args, 'completed', passed=True, num_interactions=turns,
             native_database=True, native_safety_guard=True,
             native_state_and_logs=True, next_cell_succeeded=True)
    finally:
        if world is not None:
            world.close()


def main(argv=None):
    args = parse_args(argv)
    try:
        run_probe(args)
    except BaseException as error:
        # Under true exhaustion even this bounded report can fail; parent must
        # treat missing completed evidence/nonzero exit as failure, never pass.
        try:
            emit(args, 'failed', passed=False, error_type=type(error).__name__,
                 detail=str(error)[:800])
        except BaseException:
            pass
        raise


if __name__ == '__main__':
    main()
