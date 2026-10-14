"""AppWorld subprocesses owned exclusively by the current training job."""
import atexit
import concurrent.futures
import json
import os
import signal
import socket
import subprocess
import sys
import time
import threading
import urllib.request
from pathlib import Path

_owned=[]
_supervisor = None
_stopping = threading.Event()


def stop_services():
    _stopping.set()
    if _supervisor is not None:
        _supervisor.stop()
    for process, log in _owned:
        if process.poll() is None:
            process.terminate()
    # One grace period for the whole owned pool, not 144 serial 15s waits.
    deadline = time.monotonic() + 15
    while any(p.poll() is None for p, _ in _owned) and time.monotonic() < deadline:
        time.sleep(.1)
    for process, log in _owned:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        log.close()


def required_instances(config):
    # Native manager uses one eval sample/environment (val_kwargs.n must be 1).
    eval_n=int(config.actor_rollout_ref.rollout.val_kwargs.n)
    if eval_n != 1:
        raise ValueError('Native AppWorld manager supports eval group=1; refusing inconsistent count')
    return int(config.data.train_batch_size)*int(config.env.rollout.n)+int(config.data.val_batch_size)*eval_n


def _start_one(index, root, generation=0):
    from experiments.gigpo_diag.appworld_resource import atomic_json
    for attempt in range(3):
        if _stopping.is_set():
            raise RuntimeError('Service pool is shutting down')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            port=sock.getsockname()[1]
        path=root/f'server-{index}-g{generation}-{attempt}.log'
        log=path.open('ab')
        module = 'experiments.gigpo_diag.appworld_service_launcher' if os.environ.get('GIGPO_RESOURCE_PROFILE') else 'experiments.gigpo_diag.appworld_server'
        process=subprocess.Popen([sys.executable,'-m',module,'serve','environment','--port',str(port),
                                  '--root',os.environ['APPWORLD_ROOT']], stdout=log, stderr=subprocess.STDOUT)
        _owned.append((process,log))
        (root/f'server-{index}-{attempt}.pid.json').write_text(json.dumps({'pid':process.pid,'port':port,'index':index,'job':os.environ.get('SLURM_JOB_ID')}))
        deadline=time.monotonic()+180
        while not _stopping.is_set() and time.monotonic()<deadline and process.poll() is None:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/openapi.json', timeout=3) as response:
                    payload=json.load(response)
                if 'paths' in payload and process.poll() is None:
                    entry = {'index':index,'port':port,'pid':process.pid,'log':str(path),'ready':True,'generation':generation}
                    atomic_json(root/f'port-{port}.json', entry)
                    atomic_json(root/f'slot-{index}.json', entry)
                    return entry
            except Exception:
                time.sleep(1)
        if process.poll() is None:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    raise RuntimeError(f'AppWorld service {index} failed after 3 attempts; see {root}')


class ServiceSupervisor:
    """Runs outside all execution processes; no policy code or DB recovery here."""
    def __init__(self, entries, root, profile):
        self.entries = {e['index']: e for e in entries}
        self.root, self.profile = root, profile
        self.stopped = threading.Event()
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.pending = {}
        from experiments.gigpo_diag.appworld_resource import cgroup_memory
        job = os.environ.get('SLURM_JOB_ID')
        self.job_cgroup = None
        if job:
            for row in cgroup_memory().get('ancestors', []):
                name = Path(row['path']).name
                if name in (f'job_{job}', f'job_{job}.scope', f'job{job}') and isinstance(row.get('memory.max'), int):
                    self.job_cgroup = row
                    break
        self.thread = threading.Thread(target=self.run, name='appworld-supervisor', daemon=True)
        self.thread.start()

    def stop(self):
        self.stopped.set()
        self.thread.join(timeout=5)
        self.pool.shutdown(wait=True, cancel_futures=True)

    def run(self):
        from experiments.gigpo_diag.appworld_resource import atomic_json, read_json, proc_memory, cgroup_memory, PROTOCOL
        last_sample = 0
        with (self.root/'resource_usage.jsonl').open('a', encoding='utf-8') as stream:
            while not self.stopped.wait(.5):
                try:
                    if self.job_cgroup:
                        current = int((Path(self.job_cgroup['path'])/'memory.current').read_text())
                        if current > self.job_cgroup['memory.max'] * .90:
                            # Aggregate job pressure has no honest single-action
                            # attribution. Stop owned services, then fail the batch
                            # as infrastructure, never assign resource reward0.
                            raise RuntimeError(f'job_cgroup_pressure_abort current={current} max={self.job_cgroup["memory.max"]}')
                    rows = []
                    for index, future in list(self.pending.items()):
                        if not future.done():
                            continue
                        del self.pending[index]
                        try:
                            self.entries[index] = future.result()
                        except Exception as exc:
                            atomic_json(self.root/f'slot-{index}.json', {'fatal':True,'reason':type(exc).__name__})
                            self.entries.pop(index, None)
                    processes = {p.pid:p for p, _ in _owned}
                    for index, entry in list(self.entries.items()):
                        if index in self.pending:
                            continue
                        pid = entry['pid']
                        process = processes[pid]
                        active = read_json(self.root/f'request-{pid}.json', {})
                        memory = proc_memory(pid)
                        reason = None
                        if process.poll() is None and active.get('status') == 'active':
                            if time.monotonic() - active['started'] > self.profile['hard_execute_seconds']:
                                reason = 'hard_execute_deadline'
                            elif (memory['rss_bytes'] or 0) > self.profile['service_rss_watchdog_bytes']:
                                reason = 'declared_service_memory_budget'
                        if reason:
                            # Evidence written before kill; exact request identity is mandatory.
                            atomic_json(self.root/f'terminal-{pid}.json',
                                {'protocol':PROTOCOL,'kind':'resource_terminal','reason':reason,
                                 'request':active['request'],'pid':pid,'generation':entry['generation'],
                                 'memory':memory,'clock':time.monotonic()})
                            process.kill()
                            process.wait(timeout=5)
                        if process.poll() is not None:
                            terminal = read_json(self.root/f'terminal-{pid}.json')
                            if not terminal:
                                atomic_json(self.root/f'terminal-{pid}.json',
                                    {'protocol':PROTOCOL,'kind':'infra_unknown','reason':'unexpected_service_exit',
                                     'request':active.get('request',{}),'pid':pid,'exit_code':process.returncode})
                            atomic_json(self.root/f'slot-{index}.json', {**entry,'ready':False})
                            # Fresh service is for the next reset, never a replay of the old action.
                            if terminal and terminal.get('kind') == 'resource_terminal' and entry['generation'] < 20:
                                self.pending[index] = self.pool.submit(_start_one,index,self.root,entry['generation']+1)
                            else:
                                atomic_json(self.root/f'slot-{index}.json', {**entry,'ready':False,'fatal':True,'reason':'service exited or restart budget exhausted'})
                                self.entries.pop(index, None)
                        rows.append({'index':index,'pid':pid,'generation':entry['generation'],
                                     'request':active.get('request'),'status':active.get('status'),**memory})
                    if time.monotonic() - last_sample >= 5:
                        last_sample = time.monotonic()
                        snapshot = {'clock':last_sample,'services':rows,'cgroup':cgroup_memory(),
                                    'job_pressure_guard_path': self.job_cgroup['path'] if self.job_cgroup else None}
                        stream.write(json.dumps(snapshot)+'\n')
                        stream.flush()
                        atomic_json(self.root/'resource_latest.json', snapshot)
                except Exception as exc:
                    # A dead supervisor must not leave unbounded services silently running.
                    _stopping.set()
                    atomic_json(self.root/'supervisor_failed.json', {'kind':'infra_unknown','reason':str(exc),'error_type':type(exc).__name__})
                    for index, entry in self.entries.items():
                        atomic_json(self.root/f'slot-{index}.json', {**entry,'ready':False,'fatal':True,'reason':'supervisor aborted'})
                    for process, _ in _owned:
                        if process.poll() is None:
                            process.kill()
                    return


def start_services(config):
    global _supervisor
    _stopping.clear()
    root=Path(os.environ['GIGPO_RUN_DIR'])/'services'
    root.mkdir(parents=True, exist_ok=True)
    root_data=Path(os.environ['APPWORLD_ROOT'])
    root_data.mkdir(parents=True, exist_ok=True)
    data_target=root_data/'data'
    if not data_target.exists():
        data_target.symlink_to(Path(os.environ['GIGPO_APPWORLD_DATA'])/'data', target_is_directory=True)
    count=required_instances(config)
    profile = None
    if os.environ.get('GIGPO_RESOURCE_PROFILE'):
        from experiments.gigpo_diag.appworld_resource import load_profile
        profile = load_profile()
        if profile['service_count'] != count:
            raise RuntimeError('Calibrated service count does not match effective configuration')
    atexit.register(stop_services)
    # Only this process's children are retained/cleaned. No global kill or Ray stop.
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            entries=list(pool.map(lambda i:_start_one(i,root),range(count)))
        port_file=Path(os.environ['APPWORLD_PORTS_FILE'])
        port_file.write_text(''.join(str(e['port'])+'\n' for e in entries))
        (root/'manifest.json').write_text(json.dumps({'train_batch':config.data.train_batch_size,
            'train_group':config.env.rollout.n,'eval_batch':config.data.val_batch_size,'eval_group':1,
            'required_instances':count,'services':entries},indent=2))
        if profile:
            _supervisor = ServiceSupervisor(entries, root, profile)
        print('APPWORLD_SERVICES_READY',count,port_file,flush=True)
    except BaseException:
        stop_services()
        raise
