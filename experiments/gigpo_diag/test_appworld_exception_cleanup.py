"""Real pinned AppWorld.execute/SafetyGuard/IPython; synthetic persistence only.

No benchmark task DB, real API server, GPU, or actual memory exhaustion is used.
Payloads are 64 KiB.  MemoryError is injected; Windows deadline delivery uses a
deterministic alarm double while the real shell and native timeout path run.
"""
import gc
import sys
import types
import unittest
import weakref
from unittest import mock

from appworld import AppWorld
from appworld.common.safety_guard import SafetyGuard
from IPython.terminal.embed import InteractiveShellEmbed
from traitlets.config.loader import Config

from experiments.gigpo_diag import appworld_exception_cleanup as repair
from experiments.gigpo_diag import appworld_timeout
from experiments.gigpo_diag.test_appworld_timeout import Alarm


class Payload:
    def __init__(self):
        self.data = bytearray(64 * 1024)


def native_world():
    world = AppWorld.__new__(AppWorld)
    world.remote_environment_url = None
    world.remote_mcp_url = None
    world.raise_on_unsafe_syntax = True
    world.raise_on_unsafe_execution = True
    world.safety_guard = SafetyGuard()
    world.timeout_seconds = None
    world.num_interactions = 0
    world.num_sub_interactions = 0
    world.max_interactions = 100
    world.environment_io = []
    world.output_db_home_path_on_disk = 'synthetic-no-database'
    world.test_events = []
    world.requester = types.SimpleNamespace(
        reset_request_count=lambda: world.test_events.append('request_count_reset'))
    world._save_state = lambda directory: world.test_events.append('state')
    world.save_logs = lambda: world.test_events.append('logs')
    config = Config()
    config.HistoryManager.enabled = False
    original_excepthook = sys.excepthook
    world.shell = InteractiveShellEmbed(config=config)
    sys.excepthook = original_excepthook
    world.shell.ast_node_interactivity = 'none'
    return world


class ExceptionCleanupTests(unittest.TestCase):
    def setUp(self):
        self.native_execute = AppWorld.execute
        self.native_run_cell = AppWorld._shell_run_cell
        self.addCleanup(setattr, AppWorld, 'execute', self.native_execute)
        self.addCleanup(setattr, AppWorld, '_shell_run_cell', self.native_run_cell)
        repair.install_exception_cleanup()
        self.world = native_world()
        self.addCleanup(self.world.safety_guard.disable)
        self.addCleanup(repair.clear_completed_execution_exceptions, self.world.shell, {})
        self.refs = []

        def make_payload():
            payload = Payload()
            self.refs.append(weakref.ref(payload))
            return payload

        self.world.shell.user_ns['make_payload'] = make_payload

    def assert_one_native_turn(self, start, code, feedback):
        self.assertEqual(self.world.num_interactions, start + 1)
        self.assertEqual(self.world.environment_io[-1],
                         {'number': str(start + 1), 'input': code,
                          'output': feedback.rstrip()})
        self.assertEqual(self.world.test_events[-4:],
                         ['request_count_reset', 'state', 'logs', 'request_count_reset'])

    def assert_framework_clear(self):
        self.assertIsNone(self.world.shell.last_execution_result)
        self.assertIsNone(getattr(self.world.shell.InteractiveTB, 'tb', None))
        self.assertIsNone(self.world.shell.SyntaxTB.last_syntax_error)
        for name in repair._SYS_NAMES:
            self.assertIsNone(getattr(sys, name, None), name)

    def test_real_native_ordinary_and_memory_error_cleanup_after_persistence(self):
        snapshots = []

        def save_logs():
            result = self.world.shell.last_execution_result
            snapshots.append((isinstance(result.error_in_exec, BaseException),
                              self.refs[-1]() is not None,
                              self.world.num_interactions,
                              self.world.environment_io[-1]['output']))
            self.world.test_events.append('logs')

        self.world.save_logs = save_logs
        for error_name in ('ValueError', 'MemoryError'):
            code = ('def fail():\n    payload = make_payload()\n'
                    f'    raise {error_name}("bounded test failure")\nfail()')
            start = self.world.num_interactions
            feedback = self.world.execute(code)
            self.assertIn(error_name, feedback)
            self.assert_one_native_turn(start, code, feedback)
            self.assertEqual(snapshots[-1][:3], (True, True, start + 1))
            self.assertEqual(snapshots[-1][3], feedback.rstrip())
            self.assert_framework_clear()
            gc.collect()
            self.assertTrue(all(ref() is None for ref in self.refs))

    def test_twenty_small_errors_release_payloads_and_preserve_user_state(self):
        setup = ('state = {"values": [1]}\n'
                 'def add_value(value):\n    state["values"].append(value)\n'
                 'def fail():\n    payload = make_payload()\n'
                 '    state["values"].append(2)\n    raise ValueError("small failure")')
        self.world.execute(setup)
        for _ in range(20):
            start = self.world.num_interactions
            feedback = self.world.execute('fail()')
            self.assert_one_native_turn(start, 'fail()', feedback)
            gc.collect()
            self.assertTrue(all(ref() is None for ref in self.refs))
        self.assertEqual(len(self.refs), 20)
        self.assertEqual(self.world.num_interactions, 21)
        self.assertEqual(self.world.test_events.count('state'), 21)
        self.assertEqual(self.world.test_events.count('logs'), 21)
        self.assertEqual(self.world.test_events.count('request_count_reset'), 42)
        self.assertEqual(self.world.execute('add_value(3)\nprint(state["values"])'),
                         str([1] + [2] * 20 + [3]) + '\n')
        self.assertEqual(self.world.num_interactions, 22)

    def test_intentionally_retained_exception_and_its_locals_are_not_mutated(self):
        code = ('saved_errors = []\n'
                'def fail_and_retain():\n    payload = make_payload()\n'
                '    try:\n        raise ValueError("intentionally retained")\n'
                '    except Exception as error:\n        saved_errors.append(error)\n'
                '        raise\nfail_and_retain()')
        feedback = self.world.execute(code)
        self.assertIn('intentionally retained', feedback)
        self.assert_framework_clear()
        gc.collect()
        self.assertIsNotNone(self.refs[-1]())
        kept = self.world.shell.user_ns['saved_errors'][0]
        self.assertIsInstance(kept, ValueError)
        self.assertIsNotNone(kept.__traceback__)
        traceback = kept.__traceback__
        found = False
        while traceback is not None:
            if traceback.tb_frame.f_code.co_name == 'fail_and_retain':
                self.assertIs(traceback.tb_frame.f_locals['payload'], self.refs[-1]())
                found = True
            traceback = traceback.tb_next
        self.assertTrue(found)
        # Only the policy deliberately dropping its own reference releases this.
        del kept, traceback
        self.assertEqual(self.world.execute('saved_errors.clear()\nprint("continued")'),
                         'continued\n')
        gc.collect()
        self.assertIsNone(self.refs[-1]())

    def test_private_deadline_uses_native_feedback_and_cleans_exception(self):
        alarm = Alarm()
        appworld_timeout.install_cell_deadline()
        self.world.timeout_seconds = 100

        def fire_deadline():
            alarm.handler(14, None)

        self.world.shell.user_ns['fire_deadline'] = fire_deadline
        traceback_present_during_save = []

        def save_state(directory):
            traceback_present_during_save.append(
                getattr(self.world.shell.InteractiveTB, 'tb', None) is not None)
            self.world.test_events.append('state')

        self.world._save_state = save_state
        code = ('def fail():\n    payload = make_payload()\n'
                '    fire_deadline()\nfail()')
        with mock.patch.object(appworld_timeout, 'signal', alarm):
            feedback = self.world.execute(code)
            self.assertEqual(feedback,
                             'Execution failed. Traceback:\nExecution timed out after 100 seconds.')
            self.assert_one_native_turn(0, code, feedback)
            self.assertEqual(traceback_present_during_save, [True])
            self.assert_framework_clear()
            gc.collect()
            self.assertIsNone(self.refs[-1]())
            self.assertEqual(self.world.execute('x = 42\nprint(x)'), '42\n')
        self.assertEqual(alarm.armed, 0)
        self.assertIs(alarm.handler, alarm.original)

    def test_unrepaired_native_error_retains_payload_until_all_owners_dropped(self):
        code = ('def fail():\n    payload = make_payload()\n'
                '    raise ValueError("unrepaired owner baseline")\nfail()')
        feedback = self.native_execute(self.world, code)
        self.assertIn('unrepaired owner baseline', feedback)
        gc.collect()
        self.assertIsNotNone(self.refs[-1]())
        repair.clear_completed_execution_exceptions(self.world.shell, {})
        gc.collect()
        self.assertIsNone(self.refs[-1]())

    def test_syntax_cache_is_cleared_through_native_execute(self):
        # ast.parse accepts top-level return; IPython compilation rejects it.
        code = 'return 42'
        feedback = self.world.execute(code)
        self.assertIn('SyntaxError', feedback)
        self.assert_one_native_turn(0, code, feedback)
        self.assert_framework_clear()

    def test_success_preserves_result_and_unrelated_sys_exception(self):
        unrelated = ValueError('outside this service execution')
        for name, value in [('last_type', ValueError), ('last_value', unrelated),
                            ('last_traceback', None), ('last_exc', unrelated)]:
            patch = mock.patch.object(sys, name, value, create=True)
            patch.start()
            self.addCleanup(patch.stop)
        feedback = self.world.execute('state = {"value": 42}\nprint(state["value"])')
        self.assertEqual(feedback, '42\n')
        self.assertTrue(self.world.shell.last_execution_result.success)
        self.assertIs(sys.last_value, unrelated)
        self.assertIs(sys.last_exc, unrelated)

    def test_mixed_sys_cache_preserves_unrelated_exception_by_identity(self):
        unrelated = RuntimeError('outside owner updated during persistence')
        with mock.patch.object(sys, 'last_exc', None, create=True):
            def save_logs():
                sys.last_exc = unrelated
                self.world.test_events.append('logs')

            self.world.save_logs = save_logs
            feedback = self.world.execute('raise ValueError("policy failure")')
            self.assertIn('ValueError', feedback)
            self.assertIsNone(sys.last_value)
            self.assertIs(sys.last_exc, unrelated)

    def test_persistence_error_is_not_hidden_or_retried(self):
        saved = []

        def failing_save(directory):
            saved.append(directory)
            raise OSError('synthetic state persistence error')

        self.world._save_state = failing_save
        with self.assertRaisesRegex(OSError, 'synthetic state persistence error'):
            self.world.execute('value = 7\nraise ValueError("policy failure")')
        self.assertEqual(len(saved), 1)
        self.assertEqual(self.world.num_interactions, 1)
        self.assertEqual(self.world.shell.user_ns['value'], 7)
        self.assertIsNotNone(self.world.shell.last_execution_result.error_in_exec)
        self.assertNotIn('logs', self.world.test_events)

    def test_install_is_idempotent_and_reentrant_execution_fails_before_action(self):
        installed = AppWorld.execute
        repair.install_exception_cleanup()
        self.assertIs(AppWorld.execute, installed)
        repair._EXECUTION_LOCK.acquire()
        try:
            with self.assertRaisesRegex(RuntimeError, 'Concurrent local AppWorld'):
                self.world.execute('should_not_exist = True')
        finally:
            repair._EXECUTION_LOCK.release()
        self.assertNotIn('should_not_exist', self.world.shell.user_ns)
        self.assertEqual(self.world.num_interactions, 0)


if __name__ == '__main__':
    unittest.main()
