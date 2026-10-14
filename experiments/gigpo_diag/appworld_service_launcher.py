"""Set a measured service-only hard AS limit before importing AppWorld.

No preexec_fn: this launcher starts as a new interpreter from threaded parents.
"""
import json
import os
import resource
import sys
from pathlib import Path
from experiments.gigpo_diag.appworld_resource import atomic_json, load_profile, proc_memory


def main():
    profile = load_profile()
    limit = int(profile['service_as_bytes'])
    current = proc_memory(os.getpid())['vms_bytes']
    if current is None or current >= limit:
        raise RuntimeError('Launcher baseline does not fit service AS budget')
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    atomic_json(Path(os.environ['GIGPO_RUN_DIR'])/'services'/f'limit-{os.getpid()}.json',
                {'pid': os.getpid(), 'profile': profile, 'rlimit_as': resource.getrlimit(resource.RLIMIT_AS)})
    from experiments.gigpo_diag.appworld_server import main as serve
    serve()


if __name__ == '__main__':
    main()
