"""Shared, stdlib-only bounded_runtime_v1 control plane. Never replay code."""
import hashlib
import json
import os
from pathlib import Path
import time

PROTOCOL = 'bounded_runtime_v1'


class ResourceTerminal(Exception):
    def __init__(self, event):
        self.event = event
        super().__init__(event.get('reason', 'declared resource terminal'))

    def __reduce__(self):
        return type(self), (self.event,)


class InfrastructureFailure(Exception):
    def __init__(self, event):
        self.event = event
        super().__init__(event.get('reason', 'unknown infrastructure failure'))

    def __reduce__(self):
        return type(self), (self.event,)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def service_root():
    return Path(os.environ['GIGPO_RUN_DIR']) / 'services'


def wait_service(worker_id, timeout=180, previous_pid=None):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        entry = read_json(service_root() / f'slot-{worker_id}.json', {})
        if entry.get('ready') and entry.get('pid') != previous_pid:
            return entry
        if entry.get('fatal'):
            raise InfrastructureFailure(entry)
        time.sleep(.1)
    raise InfrastructureFailure({'reason': 'service readiness deadline', 'worker': worker_id})


def request_identity(world, code):
    return {'task_id': world.task_id, 'experiment_name': world.experiment_name,
            'interactions_before': world.num_interactions,
            'code_sha256': hashlib.sha256(code.encode('utf-8')).hexdigest()}


def matches(event, identity):
    request = event.get('request', {})
    return all(request.get(k) == v for k, v in identity.items())


def resolve_failed_execute(world, code, error, timeout=3):
    """Return an exact completed response or raise a proven terminal/unknown.

    The connection may close just before the supervisor publishes its event.
    Wait briefly for evidence, never retry the HTTP request.
    """
    from urllib.parse import urlparse
    port = urlparse(world.remote_environment_url).port
    root = service_root()
    identity = request_identity(world, code)
    end = time.monotonic() + timeout
    while True:
        endpoint = read_json(root / f'port-{port}.json', {})
        pid = getattr(world, 'resource_service_pid', None) or endpoint.get('pid')
        if pid:
            ack = read_json(root / f'request-{pid}.json', {})
            terminal = read_json(root / f'terminal-{pid}.json', {})
            # Hard termination takes precedence over an ack published in a race.
            if matches(terminal, identity) and terminal.get('protocol') == PROTOCOL:
                if terminal.get('kind') == 'resource_terminal':
                    raise ResourceTerminal(terminal)
                raise InfrastructureFailure(terminal)
            if matches(ack, identity) and ack.get('status') == 'complete' and isinstance(ack.get('output'), str):
                return ack['output']
        if time.monotonic() >= end:
            raise InfrastructureFailure({'kind': 'infra_unknown', 'reason': type(error).__name__,
                                         'request': identity, 'port': port, 'pid': pid}) from error
        time.sleep(.05)


def proc_memory(pid):
    values = {'rss_bytes': None, 'vms_bytes': None}
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmSize:')):
                key = 'rss_bytes' if line.startswith('VmRSS:') else 'vms_bytes'
                values[key] = int(line.split()[1]) * 1024
    except OSError:
        pass
    return values


def cgroup_memory():
    """Read v2 ancestor chain: a leaf can omit the Slurm job's finite limit."""
    try:
        name = next(s.split(':', 2)[2] for s in Path('/proc/self/cgroup').read_text().splitlines() if s.startswith('0::'))
        root = Path('/sys/fs/cgroup') / name.lstrip('/')
        ancestors = []
        while root.is_relative_to('/sys/fs/cgroup'):
            result = {'path': str(root)}
            for key in ('memory.current', 'memory.peak', 'memory.max', 'memory.events', 'memory.stat'):
                file = root / key
                if file.exists():
                    value = file.read_text().strip()
                    result[key] = int(value) if value.isdigit() else value
            ancestors.append(result)
            if root == Path('/sys/fs/cgroup'):
                break
            root = root.parent
        return {'ancestors': ancestors}
    except (OSError, StopIteration):
        return {'ancestors': [], 'unavailable': True}


def load_profile():
    profile = read_json(os.environ['GIGPO_RESOURCE_PROFILE'])
    if not profile or profile.get('protocol') != PROTOCOL or not profile.get('calibrated'):
        raise RuntimeError('Missing calibrated bounded_runtime_v1 profile')
    if not (100 < profile['hard_execute_seconds'] < 160):
        raise RuntimeError('hard deadline must be between native 100s and HTTP 160s')
    if profile['service_as_bytes'] <= profile['measured_peak_vms_bytes']:
        raise RuntimeError('Service address-space budget has no measured headroom')
    if not (profile['measured_peak_rss_bytes'] < profile['service_rss_watchdog_bytes'] < profile['service_as_bytes']):
        raise RuntimeError('Service RSS watchdog must leave measured recovery headroom')
    if profile['code_commit'] != os.environ['GIGPO_CODE_COMMIT']:
        raise RuntimeError('Calibrated resource profile belongs to another code revision')
    if (profile.get('memory_budget_version') != 'cell_with_native_reserve_v1' or
            not profile.get('native_evaluation_verified') or
            not profile.get('retained_pressure_evaluation_verified')):
        raise RuntimeError('Resource profile lacks native and retained-pressure evaluation coverage')
    if not (0 < profile['policy_as_bytes'] < profile['service_as_bytes'] and
            profile['native_reserve_bytes'] == profile['service_as_bytes'] - profile['policy_as_bytes'] and
            profile['service_as_bytes'] * profile['service_count'] <= profile['environment_envelope_bytes']):
        raise RuntimeError('Invalid native reserve or environment envelope')
    return profile
