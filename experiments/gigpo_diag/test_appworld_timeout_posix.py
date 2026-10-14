"""Real Linux alarms and IPython. Run with an outer process timeout."""
import ast
import os
from pathlib import Path
import signal
import sys
import threading
import time
import unittest

from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from experiments.gigpo_diag.appworld_timeout import run_cell_with_deadline


@unittest.skipUnless(hasattr(signal, 'SIGALRM'), 'POSIX signal integration')
class PosixDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.handler = signal.getsignal(signal.SIGALRM)
        self.addCleanup(signal.signal, signal.SIGALRM, self.handler)
        self.addCleanup(signal.alarm, 0)
        self.shell = InteractiveShell()
        self.last_api_call = None
        event = threading.Event()
        def bad_api():
            self.last_api_call = time.clock_gettime(time.CLOCK_MONOTONIC)
            event.wait(.01)
            raise ValueError('synthetic invalid API call')
        self.shell.user_ns.update(bad_api=bad_api, clock=time.clock_gettime,
                                  clock_id=time.CLOCK_MONOTONIC)

    def test_real_alarm_interrupts_retry_loop_and_restores_handler(self):
        code = 'while True:\n    try:\n        bad_api()\n    except Exception:\n        pass'
        start = time.clock_gettime(time.CLOCK_MONOTONIC)
        with capture_output():
            result = run_cell_with_deadline(self.shell, code, 1)
        elapsed = time.clock_gettime(time.CLOCK_MONOTONIC) - start
        self.assertIsNone(result)
        self.assertGreaterEqual(elapsed, .9)
        # First IPython traceback formatting can be slow on WSL-mounted files.
        # Check when policy calls stop separately from formatting/return time.
        self.assertLess(self.last_api_call - start, 1.2)
        self.assertLess(elapsed, 15)
        self.assertIs(signal.getsignal(signal.SIGALRM), self.handler)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0.0, 0.0))
        self.assertTrue(run_cell_with_deadline(self.shell, 'answer = 42', 1).success)
        self.assertEqual(self.shell.user_ns['answer'], 42)

    @unittest.skipUnless(os.environ.get('APPWORLD_NATIVE_TIME_SOURCE'), 'pinned native source required')
    def test_pinned_native_exception_alarm_is_swallowed(self):
        # Execute the original TimeoutError and timeout_call definitions directly
        # from the pinned source, avoiding unrelated platform binary imports.
        tree = ast.parse(Path(os.environ['APPWORLD_NATIVE_TIME_SOURCE']).read_text())
        nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                 and n.name in ('TimeoutError', 'timeout_call')]
        self.assertEqual(len(nodes), 2)
        namespace = {'signal': signal, 'sys': sys}
        source = 'from __future__ import annotations\n' + '\n'.join(ast.unparse(n) for n in nodes)
        exec(compile(source, '<pinned appworld.common.time>', 'exec'), namespace)
        self.shell.user_ns['stop_at'] = time.clock_gettime(time.CLOCK_MONOTONIC) + 1.5
        code = ('swallowed = 0\nwhile clock(clock_id) < stop_at:\n    try:\n        bad_api()\n'
                '    except Exception as exc:\n        if "timed out" in str(exc):\n            swallowed += 1')
        with capture_output():
            result = namespace['timeout_call'](self.shell.run_cell, timeout_seconds=1, raw_cell=code)
        self.assertTrue(result.success)
        self.assertEqual(self.shell.user_ns['swallowed'], 1)


if __name__ == '__main__':
    unittest.main()
