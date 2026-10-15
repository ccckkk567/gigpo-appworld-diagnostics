"""Limit ownership regressions; Linux child tests use real AS and SQLite backup."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

from experiments.gigpo_diag.appworld_memory_budget import install_cell_memory_budget
from experiments.gigpo_diag.appworld_resource_preflight import select_budgets


class BudgetTests(unittest.TestCase):
    def test_restores_reserve_after_success_exception_and_base_exception(self):
        for failure in (None, MemoryError('allocation'), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                limits = [200, 200]
                resource = types.SimpleNamespace(RLIMIT_AS=9,
                    getrlimit=lambda _: tuple(limits),
                    setrlimit=lambda _, pair: limits.__setitem__(slice(None), pair))

                class World:
                    def _shell_run_cell(self, code):
                        self.calls += 1
                        self.assert_limits = tuple(limits)
                        if failure is not None:
                            raise failure
                        return code

                with patch.dict(sys.modules, {'resource': resource,
                        'appworld': types.SimpleNamespace(AppWorld=World)}), \
                        contextlib.redirect_stdout(io.StringIO()):
                    install_cell_memory_budget(100, 200)
                    owner = World._shell_run_cell
                    install_cell_memory_budget(100, 200)
                    self.assertIs(owner, World._shell_run_cell)
                    world = World()
                    world.calls = 0
                    if failure is None:
                        self.assertEqual(world._shell_run_cell('same code'), 'same code')
                    else:
                        with self.assertRaises(type(failure)) as caught:
                            world._shell_run_cell('same code')
                        self.assertIs(caught.exception, failure)
                    self.assertEqual(world.calls, 1)
                    self.assertEqual(world.assert_limits, (100, 200))
                    self.assertEqual(limits, [200, 200])
                    limits[:] = [150, 200]
                    with self.assertRaisesRegex(RuntimeError, 'outside'):
                        world._shell_run_cell('must not execute')
                    self.assertEqual(world.calls, 1)

    def test_calibration_requires_evaluation_and_preserves_envelope(self):
        mib = 1024**2
        result = {'sampled_peaks': {'vms_bytes': 900*mib, 'rss_bytes': 650*mib},
                  'events': [{'event': 'evaluate_begin', 'VmPeak_bytes': 800*mib, 'VmSize_bytes': 780*mib},
                             {'event': 'evaluate_complete', 'VmPeak_bytes': 900*mib}]}
        budgets = select_budgets([result], 1365*mib)
        self.assertEqual(budgets['policy_as_bytes'], 1040*mib)
        self.assertEqual(budgets['service_as_bytes'], 1365*mib)
        self.assertGreater(budgets['native_reserve_bytes'], budgets['required_native_reserve_bytes'])
        with self.assertRaisesRegex(RuntimeError, 'fixed environment budget'):
            select_budgets([result], 1100*mib)
        with self.assertRaises(StopIteration):
            select_budgets([{**result, 'events': []}], 1365*mib)

    @unittest.skipUnless(sys.platform == 'linux', 'real Linux RLIMIT_AS required')
    def test_live_objects_fill_policy_budget_but_native_sqlite_backup_survives(self):
        # No AppWorld task data is available here. Exercise the actual kernel
        # mechanism and SQLite copy in a fresh externally bounded child. HPC
        # preflight separately requires the real AppWorld evaluator to pass.
        code = r'''
import json, os, resource, sqlite3, sys, types
from pathlib import Path
from experiments.gigpo_diag.appworld_memory_budget import install_cell_memory_budget
mib = 1024**2
baseline = int(next(s.split()[1] for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmSize:'))) * 1024
policy, hard = baseline + 32*mib, baseline + 96*mib
resource.setrlimit(resource.RLIMIT_AS, (hard, hard))
class World:
    def __init__(self): self.ns = {}
    def _shell_run_cell(self, code):
        try: exec(code, self.ns)
        except MemoryError: return 'MemoryError'
        return 'ok'
sys.modules['appworld'] = types.SimpleNamespace(AppWorld=World)
install_cell_memory_budget(policy, hard)
w = World()
assert w._shell_run_cell('blocks = []\nwhile True: blocks.append(bytearray(1048576))') == 'MemoryError'
blocks = len(w.ns['blocks'])
assert blocks > 20 and resource.getrlimit(resource.RLIMIT_AS) == (hard, hard)
source = sqlite3.connect(':memory:')
source.execute('create table payload (x blob)')
source.execute('insert into payload values (zeroblob(?))', (12*mib,))
source.commit()
destination = sqlite3.connect(':memory:')
source.backup(destination)
assert destination.execute('select length(x) from payload').fetchone()[0] == 12*mib
assert len(w.ns['blocks']) == blocks
source.close(); destination.close()
assert w._shell_run_cell('blocks.clear()') == 'ok'
assert w._shell_run_cell('answer = 42') == 'ok' and w.ns['answer'] == 42
assert resource.getrlimit(resource.RLIMIT_AS) == (hard, hard)
print(json.dumps({'passed': True, 'retained_blocks': blocks}))
'''
        result = subprocess.run([sys.executable, '-c', code], capture_output=True,
                                text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(json.loads(result.stdout.splitlines()[-1])['passed'])


if __name__ == '__main__':
    unittest.main()
