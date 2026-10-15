"""Native AppWorld/IPython ordering with a fake limit, no benchmark data/GPU."""
import contextlib
import io
import sys
import types
import unittest
from unittest.mock import patch

from appworld import AppWorld
from experiments.gigpo_diag.appworld_memory_budget import install_cell_memory_budget
from experiments.gigpo_diag.test_appworld_exception_cleanup import native_world


class NativeOrderingTests(unittest.TestCase):
    def test_reserve_restored_before_native_error_format_and_persistence(self):
        original = AppWorld._shell_run_cell
        self.addCleanup(setattr, AppWorld, '_shell_run_cell', original)
        limits = [200, 200]
        resource = types.SimpleNamespace(RLIMIT_AS=9, getrlimit=lambda _: tuple(limits),
            setrlimit=lambda _, pair: limits.__setitem__(slice(None), pair))
        world = native_world()
        observed = []
        world.shell.user_ns['observe_budget'] = lambda: observed.append(tuple(limits))
        world._save_state = lambda _: self.assertEqual(limits, [200, 200])
        world.save_logs = lambda: self.assertEqual(limits, [200, 200])
        with patch.dict(sys.modules, {'resource': resource}), contextlib.redirect_stdout(io.StringIO()):
            install_cell_memory_budget(100, 200)
            ok = world.execute('persistent = [42]\nobserve_budget()')
            failed = world.execute('observe_budget()\nraise MemoryError("fixture pressure")')
            following = world.execute('observe_budget()\npersistent.append(43)')
        self.assertFalse(ok.startswith('Execution failed.'))
        self.assertIn('MemoryError: fixture pressure', failed)
        self.assertFalse(following.startswith('Execution failed.'))
        self.assertEqual(observed, [(100, 200)] * 3)
        self.assertEqual(world.shell.user_ns['persistent'], [42, 43])
        self.assertEqual(world.num_interactions, 3)
        self.assertEqual(limits, [200, 200])


if __name__ == '__main__':
    unittest.main()
