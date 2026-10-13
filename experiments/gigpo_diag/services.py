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
import urllib.request
from pathlib import Path

_owned=[]


def stop_services():
    for process, log in _owned:
        if process.poll() is None:
            process.terminate()
    for process, log in _owned:
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        log.close()


def required_instances(config):
    # Native manager uses one eval sample/environment (val_kwargs.n must be 1).
    eval_n=int(config.actor_rollout_ref.rollout.val_kwargs.n)
    if eval_n != 1:
        raise ValueError('Native AppWorld manager supports eval group=1; refusing inconsistent count')
    return int(config.data.train_batch_size)*int(config.env.rollout.n)+int(config.data.val_batch_size)*eval_n


def _start_one(index, root):
    for attempt in range(3):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            port=sock.getsockname()[1]
        path=root/f'server-{index}-{attempt}.log'
        log=path.open('ab')
        process=subprocess.Popen([sys.executable,'-m','appworld.cli','serve','environment','--port',str(port),
                                  '--root',os.environ['APPWORLD_ROOT']], stdout=log, stderr=subprocess.STDOUT)
        _owned.append((process,log))
        (root/f'server-{index}-{attempt}.pid.json').write_text(json.dumps({'pid':process.pid,'port':port,'index':index,'job':os.environ.get('SLURM_JOB_ID')}))
        deadline=time.monotonic()+180
        while time.monotonic()<deadline and process.poll() is None:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{port}/openapi.json', timeout=3) as response:
                    payload=json.load(response)
                if 'paths' in payload and process.poll() is None:
                    return {'index':index,'port':port,'pid':process.pid,'log':str(path),'ready':True}
            except Exception:
                time.sleep(1)
        if process.poll() is None:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    raise RuntimeError(f'AppWorld service {index} failed after 3 attempts; see {root}')


def start_services(config):
    root=Path(os.environ['GIGPO_RUN_DIR'])/'services'
    root.mkdir(parents=True, exist_ok=True)
    root_data=Path(os.environ['APPWORLD_ROOT'])
    root_data.mkdir(parents=True, exist_ok=True)
    data_target=root_data/'data'
    if not data_target.exists():
        data_target.symlink_to(Path(os.environ['GIGPO_APPWORLD_DATA'])/'data', target_is_directory=True)
    count=required_instances(config)
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
        print('APPWORLD_SERVICES_READY',count,port_file,flush=True)
    except BaseException:
        stop_services()
        raise
