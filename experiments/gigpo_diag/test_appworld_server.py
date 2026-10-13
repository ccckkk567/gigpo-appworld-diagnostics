"""Real pinned AppWorld/LibCST/FastAPI, synthetic input; no benchmark rollouts."""
import importlib.util
import types
import unittest
from pathlib import Path
from unittest import mock

import appworld  # Activate its Pydantic compatibility before importing FastAPI.
from appworld.common.safety_guard import SafetyGuard
from appworld.environment import AppWorld
from appworld.serve import environment as server
from fastapi.testclient import TestClient
from libcst import CSTValidationError

spec = importlib.util.spec_from_file_location('appworld_server_patch', Path(__file__).with_name('appworld_server.py'))
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)

BAD_CODE = 'try:\n    pass\nexcept:\n    pass\nexcept ValueError:\n    pass\n'


class AppWorldParserTests(unittest.TestCase):
    def setUp(self):
        self.original = SafetyGuard.is_syntax_safe
        self.addCleanup(setattr, SafetyGuard, 'is_syntax_safe', self.original)
        # Only the pre-execution rejection path needs these native world fields.
        # No synthetic result is described as a real task/evaluator outcome.
        self.world = types.SimpleNamespace(task_id='synthetic-task', remote_environment_url=None,
                                          raise_on_unsafe_syntax=True, safety_guard=SafetyGuard(),
                                          num_interactions=0, num_sub_interactions=0,
                                          environment_io=[], save_logs=mock.Mock())
        self.world.execute = types.MethodType(AppWorld.execute, self.world)
        self.world.interaction_number = types.MethodType(AppWorld.interaction_number, self.world)
        self.world_patch = mock.patch.object(server, 'world', self.world)
        self.world_patch.start()
        self.addCleanup(self.world_patch.stop)

    def request(self, code):
        with TestClient(server.app, raise_server_exceptions=False) as client:
            return client.post('/execute', json={'task_id': self.world.task_id, 'code': code})

    def test_reproduce_original_exception_and_http_500(self):
        with self.assertRaisesRegex(CSTValidationError, 'bare except'):
            self.world.safety_guard.is_syntax_safe(BAD_CODE)
        self.assertEqual(self.request(BAD_CODE).status_code, 500)
        self.assertEqual(self.world.num_interactions, 0)

    def test_native_rejection_consumes_one_turn_and_preserves_original_input(self):
        repair.install_cst_rejection_handler()
        response = self.request(BAD_CODE)
        self.assertEqual(response.status_code, 200)
        output = response.json()['output']
        self.assertTrue(output.startswith('Execution failed. Traceback:\nCSTValidationError:'))
        self.assertIn('bare except', output)
        self.assertEqual(self.world.num_interactions, 1)
        self.assertEqual(self.world.environment_io, [{'number': '1', 'input': BAD_CODE, 'output': output}])
        self.world.save_logs.assert_called_once()

    def test_valid_forbidden_and_ordinary_syntax_checks_unchanged(self):
        samples = ['x = 1\nprint(x)', 'import subprocess', 'x = (',
                   'try:\n    pass\nexcept ValueError:\n    pass\nexcept:\n    pass\n']
        baseline = [self.world.safety_guard.is_syntax_safe(code) for code in samples]
        repair.install_cst_rejection_handler()
        self.assertEqual([self.world.safety_guard.is_syntax_safe(code) for code in samples], baseline)
        self.assertFalse(self.world.safety_guard.is_syntax_safe('import subprocess')[0])

    def test_other_server_errors_still_propagate(self):
        with mock.patch.object(SafetyGuard, 'is_syntax_safe', side_effect=RuntimeError('synthetic infrastructure failure')):
            repair.install_cst_rejection_handler()
            self.assertEqual(self.request('x = 1').status_code, 500)
            self.assertEqual(self.world.num_interactions, 0)
            self.world.save_logs.assert_not_called()

    def test_install_idempotent(self):
        repair.install_cst_rejection_handler()
        first = SafetyGuard.is_syntax_safe
        repair.install_cst_rejection_handler()
        self.assertIs(SafetyGuard.is_syntax_safe, first)


if __name__ == '__main__':
    unittest.main()
