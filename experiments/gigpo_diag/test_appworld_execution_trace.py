"""Native rejection plus scaled server stalls; no task DBs or model rollouts."""
import atexit
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest import mock

from appworld import AppWorld
from appworld.common.safety_guard import SafetyGuard

spec = importlib.util.spec_from_file_location('execution_trace_patch', Path(__file__).with_name('appworld_execution_trace.py'))
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


class AppWorldExecutionTraceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.originals = [(AppWorld, name, getattr(AppWorld, name))
                          for name in ('execute', '_shell_run_cell', '_save_state', 'save_logs')]
        self.originals.append((SafetyGuard, 'is_syntax_safe', SafetyGuard.is_syntax_safe))
        self.addCleanup(self.restore)
        self.trace = None

    def restore(self):
        for cls, name, original in self.originals:
            setattr(cls, name, original)
        if self.trace is not None:
            atexit.unregister(self.trace.close)
            self.trace.close()

    def install(self, delay=110):
        self.trace = repair.install_execution_trace(self.directory, stack_after_seconds=delay)

    def rows(self):
        return [json.loads(line) for p in self.directory.glob('*.jsonl') for line in p.read_text().splitlines()]

    def world(self):
        world = AppWorld.__new__(AppWorld)
        world.task_id = 'synthetic-task'
        world.remote_environment_url = None
        world.experiment_name = 'synthetic-w28-r2'
        world.num_interactions = 28
        return world

    def test_native_rejection_preserves_feedback_input_and_turn(self):
        world = self.world()
        world.raise_on_unsafe_syntax = True
        world.safety_guard = SafetyGuard()
        world.num_sub_interactions = 0
        world.environment_io = []
        world.save_logs = mock.Mock()
        self.install()
        with mock.patch.object(repair.faulthandler, 'dump_traceback_later') as arm:
            result = world.execute('import subprocess')
        self.assertTrue(result.startswith('Execution failed. Traceback:'))
        self.assertEqual(world.num_interactions, 29)
        self.assertEqual(world.environment_io[0]['input'], 'import subprocess')
        self.assertEqual(world.environment_io[0]['output'], result)
        world.save_logs.assert_called_once()
        arm.assert_called_once_with(110, file=self.trace.stacks, repeat=False, exit=False)
        self.assertEqual(self.rows()[0]['code'], 'import subprocess')
        self.assertEqual(self.rows()[-1]['status'], 'returned')

    def test_native_shell_safety_guard_does_not_capture_diagnostic_output(self):
        from IPython.core.interactiveshell import InteractiveShell
        world = self.world()
        world.raise_on_unsafe_syntax = True
        world.raise_on_unsafe_execution = True
        world.remote_mcp_url = None
        world.safety_guard = SafetyGuard()
        self.addCleanup(world.safety_guard.disable)
        world.requester = mock.Mock()
        world.max_interactions = 50
        world.num_sub_interactions = 0
        world.environment_io = []
        world.timeout_seconds = None  # Native Windows shell test mode.
        world.shell = InteractiveShell()
        world.output_db_home_path_on_disk = str(self.directory / 'synthetic-db')
        AppWorld._save_state = lambda self, directory: None
        AppWorld.save_logs = lambda self: None
        self.install()
        self.assertEqual(world.execute('print("native output")'), 'native output\n')
        self.assertEqual(world.num_interactions, 29)
        rows = self.rows()
        phases = {r['phase'] for r in rows if r['event'] == 'phase_begin'}
        self.assertEqual(phases, {'syntax_check', 'shell_run_cell', 'save_state', 'save_logs'})
        self.assertTrue(self.trace.available)

    def test_slow_native_call_dumps_stack_and_records_active_phase(self):
        def slow_shell(self, code):
            time.sleep(.2)
            return 'native result'
        def execute(self, code):
            return self._shell_run_cell(code)
        AppWorld._shell_run_cell = slow_shell
        AppWorld.execute = execute
        self.install(delay=.05)
        world = self.world()
        self.assertEqual(world.execute('synthetic pending code'), 'native result')
        rows = self.rows()
        self.assertEqual(rows[0]['request']['interactions_before'], 28)
        self.assertEqual(rows[0]['code'], 'synthetic pending code')
        self.assertEqual(rows[1]['phase'], 'shell_run_cell')
        self.assertEqual(rows[-1]['event'], 'execute_end')
        self.trace.stacks.flush()
        stack = next(self.directory.glob('*.stacks.log')).read_text()
        self.assertIn('slow_shell', stack)
        self.assertIn('Timeout', stack)
        self.assertEqual(world.num_interactions, 28)  # No turns invented by tracing.

    def test_native_exception_is_propagated_without_retry(self):
        failure = RuntimeError('synthetic infrastructure failure')
        original = mock.Mock(side_effect=failure)
        AppWorld.execute = lambda self, code: original(self, code)
        self.install()
        world = self.world()
        with self.assertRaises(RuntimeError) as caught:
            world.execute('x += 1')
        self.assertIs(caught.exception, failure)
        original.assert_called_once_with(world, 'x += 1')
        self.assertEqual(self.rows()[-1]['error_type'], 'RuntimeError')
        self.assertEqual(world.num_interactions, 28)

    def test_diagnostic_write_failure_does_not_replace_native_result(self):
        result = object()
        AppWorld.execute = lambda self, code: result
        self.install()
        self.trace.journal.close()
        self.assertIs(self.world().execute('x = 1'), result)
        self.assertFalse(self.trace.available)

    def test_install_is_idempotent_and_remote_client_is_untouched(self):
        original = mock.Mock(return_value='remote native result')
        AppWorld.execute = lambda self, code: original(self, code)
        self.install()
        self.assertIs(repair.install_execution_trace(self.directory), self.trace)
        world = self.world()
        world.remote_environment_url = 'http://synthetic'
        self.assertEqual(world.execute('x = 1'), 'remote native result')
        original.assert_called_once_with(world, 'x = 1')
        self.assertEqual(self.rows(), [])


if __name__ == '__main__':
    unittest.main()
