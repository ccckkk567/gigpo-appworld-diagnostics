"""Native AppWorld/IPython/SafetyGuard integration; deterministic alarm on Windows."""
import signal
import types
import unittest
from unittest import mock

from appworld import AppWorld
from appworld.common.safety_guard import SafetyGuard
from IPython.core.interactiveshell import InteractiveShell
from experiments.gigpo_diag import appworld_timeout as repair
from experiments.gigpo_diag import check_appworld_timeout as preflight


class Alarm:
    SIGALRM = 14
    ITIMER_REAL = 0

    def __init__(self):
        self.handler = self.original = object()
        self.armed = 0

    def getsignal(self, sig):
        return self.handler

    def signal(self, sig, handler):
        previous = self.handler
        self.handler = handler
        return previous

    def alarm(self, seconds):
        self.armed = seconds

    def getitimer(self, timer):
        return (float(self.armed), 0.0)


class DeadlineTests(unittest.TestCase):
    def setUp(self):
        self.original = AppWorld._shell_run_cell
        self.addCleanup(setattr, AppWorld, '_shell_run_cell', self.original)
        self.alarm = Alarm()
        patch = mock.patch.object(repair, 'signal', self.alarm)
        patch.start()
        self.addCleanup(patch.stop)

    def test_real_native_timeout_path_and_next_cell(self):
        attempts = []
        def wait(timeout=None):
            attempts.append(1)
            if len(attempts) == 3:
                self.alarm.handler(14, None)
        with mock.patch.object(preflight, 'signal', self.alarm):
            result = preflight.probe(pause=wait)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(len(attempts), 3)
        self.assertIs(self.alarm.handler, self.alarm.original)
        self.assertEqual(self.alarm.armed, 0)

    def test_success_and_ordinary_runtime_error_results_preserved(self):
        shell = InteractiveShell()
        for code in ('x = 6 * 7', 'raise ValueError("ordinary policy error")'):
            result = repair.run_cell_with_deadline(shell, code, 100)
            self.assertIsNotNone(result)
            if code.startswith('raise'):
                self.assertIsInstance(result.error_in_exec, ValueError)
            else:
                self.assertTrue(result.success)
        self.assertEqual(shell.user_ns['x'], 42)
        self.assertIs(self.alarm.handler, self.alarm.original)

    def test_deadline_escaping_shell_enters_native_none_path(self):
        shell = types.SimpleNamespace(run_cell=lambda **kw: self.alarm.handler(14, None))
        self.assertIsNone(repair.run_cell_with_deadline(shell, 'original code', 100))
        self.assertEqual(self.alarm.armed, 0)

    def test_unrelated_infrastructure_error_not_swallowed(self):
        failure = RuntimeError('synthetic infrastructure error')
        shell = types.SimpleNamespace(run_cell=mock.Mock(side_effect=failure))
        with self.assertRaises(RuntimeError) as caught:
            repair.run_cell_with_deadline(shell, 'original code', 100)
        self.assertIs(caught.exception, failure)
        shell.run_cell.assert_called_once_with(raw_cell='original code')
        self.assertIs(self.alarm.handler, self.alarm.original)
        self.assertEqual(self.alarm.armed, 0)

    def test_existing_timer_rejected_without_changing_it(self):
        self.alarm.armed = 7
        shell = types.SimpleNamespace(run_cell=mock.Mock())
        with self.assertRaisesRegex(RuntimeError, 'existing real timer'):
            repair.run_cell_with_deadline(shell, 'original code', 100)
        shell.run_cell.assert_not_called()
        self.assertEqual(self.alarm.armed, 7)
        self.assertIs(self.alarm.handler, self.alarm.original)

    def test_install_idempotent_and_no_deadline_mode_delegates(self):
        native = mock.Mock(return_value='native without deadline')
        AppWorld._shell_run_cell = native
        repair.install_cell_deadline()
        installed = AppWorld._shell_run_cell
        repair.install_cell_deadline()
        self.assertIs(AppWorld._shell_run_cell, installed)
        world = types.SimpleNamespace(_maybe_raise_remote_environment_error=mock.Mock(),
                                      shell=object(), timeout_seconds=None)
        self.assertEqual(installed(world, 'unchanged'), 'native without deadline')
        native.assert_called_once_with(world, 'unchanged')


if __name__ == '__main__':
    unittest.main()
