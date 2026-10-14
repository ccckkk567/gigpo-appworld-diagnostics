"""Control evidence tests; Linux adds real hard-AS child + external kill tests."""
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from experiments.gigpo_diag import appworld_resource as resource


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.directory = self.root / 'services'
        self.directory.mkdir()
        self.environment = patch.dict(os.environ, {'GIGPO_RUN_DIR': str(self.root)})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.world = types.SimpleNamespace(task_id='task', experiment_name='episode',
            num_interactions=2, remote_environment_url='http://127.0.0.1:23456', resource_service_pid=11)
        self.request = resource.request_identity(self.world, 'side_effect()')

    def test_only_exact_completed_ack_recovers_lost_response(self):
        resource.atomic_json(self.directory/'request-11.json',
            {'request':self.request, 'status':'complete', 'output':'original feedback'})
        self.assertEqual(resource.resolve_failed_execute(self.world,'side_effect()',OSError(),0), 'original feedback')
        with self.assertRaises(resource.InfrastructureFailure):
            resource.resolve_failed_execute(self.world,'different_action()',OSError(),0)

    def test_terminal_wins_ack_and_fixed_pid_wins_reused_port(self):
        resource.atomic_json(self.directory/'port-23456.json', {'pid':99})
        resource.atomic_json(self.directory/'request-11.json', {'request':self.request, 'status':'complete', 'output':'late'})
        event = {'request':self.request, 'protocol':resource.PROTOCOL, 'kind':'resource_terminal','reason':'hard_execute_deadline'}
        resource.atomic_json(self.directory/'terminal-11.json', event)
        with self.assertRaises(resource.ResourceTerminal) as caught:
            resource.resolve_failed_execute(self.world,'side_effect()',OSError(),0)
        self.assertEqual(caught.exception.event, event)

    def test_unknown_exit_is_never_a_task_failure(self):
        event = {'request':self.request,'protocol':resource.PROTOCOL,'kind':'infra_unknown','reason':'exit'}
        resource.atomic_json(self.directory/'terminal-11.json',event)
        with self.assertRaises(resource.InfrastructureFailure):
            resource.resolve_failed_execute(self.world,'side_effect()',OSError(),0)

    def test_exceptions_survive_ray_pickle(self):
        for cls in (resource.ResourceTerminal, resource.InfrastructureFailure):
            exception = cls({'request':self.request, 'reason':'test'})
            self.assertEqual(pickle.loads(pickle.dumps(exception)).event, exception.event)

    def test_reset_waits_for_new_pid(self):
        resource.atomic_json(self.directory/'slot-1.json', {'pid':11,'ready':True})
        with self.assertRaises(resource.InfrastructureFailure):
            resource.wait_service(1, .01, previous_pid=11)
        self.assertEqual(resource.wait_service(1, .1)['pid'], 11)


@unittest.skipUnless(sys.platform == 'linux', 'requires Linux hard AS and /proc')
class LinuxSupervisorTests(unittest.TestCase):
    def test_limits_deadline_peer_isolation_and_job_pressure_abort(self):
        from experiments.gigpo_diag import services
        # AS is measured in the fresh child plus 16MiB test headroom. No policy
        # code runs in this test runner and no child starts without its hard cap.
        child_code = '''import json,os,resource,sys,time
from pathlib import Path
vms=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmSize:')))*1024
limit=vms+16*1024*1024
resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
if sys.argv[2]=='allocate':
    try:
        blocks=[]
        while True: blocks.append(bytearray(1048576))
    except MemoryError:
        del blocks
        Path(sys.argv[1]+'/memory-error').write_text(str(limit))
request={'task_id':'test','experiment_name':'once','interactions_before':0,'code_sha256':'test'}
Path(sys.argv[1]+'/request-'+str(os.getpid())+'.json').write_text(json.dumps({'status':'active' if sys.argv[2]=='allocate' else 'complete','started':time.monotonic(),'request':request}))
while True:
    try: time.sleep(.02)
    except: pass
'''
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            children = []
            monitor = None
            try:
                for mode in ('allocate','healthy'):
                    p = subprocess.Popen([sys.executable,'-c',child_code,td,mode], stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                    children.append(p)
                end = time.monotonic()+5
                while not all((root/f'request-{p.pid}.json').exists() for p in children):
                    if time.monotonic()>end:
                        self.fail('Child did not reach limited execution')
                    time.sleep(.02)
                self.assertTrue((root/'memory-error').exists())
                entries = [dict(index=i,pid=p.pid,ready=True,generation=20) for i,p in enumerate(children)]
                with patch.object(services,'_owned', [(p,None) for p in children]):
                    monitor = services.ServiceSupervisor(entries,root,{'hard_execute_seconds':.25,'service_rss_watchdog_bytes':10**9})
                    children[0].wait(timeout=5)
                    monitor.stop()
                    monitor = None
                terminal=resource.read_json(root/f'terminal-{children[0].pid}.json')
                self.assertEqual(terminal['kind'],'resource_terminal')
                self.assertEqual(terminal['reason'],'hard_execute_deadline')
                self.assertIsNone(children[1].poll(), 'healthy service was affected')
                # Identified aggregate pressure is an infrastructure abort;
                # it must never fabricate a terminal reward for this idle peer.
                job_dir = root/'job_12345'
                job_dir.mkdir()
                (job_dir/'memory.current').write_text('91')
                with patch.dict(os.environ, {'SLURM_JOB_ID':'12345'}), \
                     patch.object(resource,'cgroup_memory', return_value={'ancestors':[{'path':str(job_dir),'memory.max':100}]}), \
                     patch.object(services,'_owned',[(p,None) for p in children]):
                    monitor=services.ServiceSupervisor([entries[1]],root,{'hard_execute_seconds':.25,'service_rss_watchdog_bytes':10**9})
                    children[1].wait(timeout=5)
                    monitor.stop()
                    monitor=None
                self.assertEqual(resource.read_json(root/'supervisor_failed.json')['kind'], 'infra_unknown')
                self.assertFalse((root/f'terminal-{children[1].pid}.json').exists())
            finally:
                services._stopping.clear()
                if monitor:
                    monitor.stop()
                for p in children:
                    if p.poll() is None:
                        p.kill()
                    p.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
