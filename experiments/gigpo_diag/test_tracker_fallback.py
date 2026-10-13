"""Synthetic tracker failures, never reported as HPC training results."""
import ast
import gzip
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy  # Load before patch.dict restores sys.modules between fixtures.

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('tracking_fixture', ROOT/'verl/utils/tracking.py')
tracking = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tracking
spec.loader.exec_module(tracking)

# Exercise the actual trainer call site without importing GPU workers.
tree = ast.parse((ROOT/'verl/trainer/ppo/ray_trainer.py').read_text(encoding='utf-8'))
trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RayPPOTrainer')
method = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == '_maybe_log_val_generations')
namespace = {}
exec(compile(ast.Module(body=[method], type_ignores=[]), '<trainer-method>', 'exec'), namespace)


class TrackerFallbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = mock.patch.dict(os.environ, {'GIGPO_DIAG_DIR': self.tmp.name,
                                               'SWANLAB_API_KEY': 'synthetic-test-secret'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.swan = types.ModuleType('swanlab')
        self.swan.init = mock.Mock(return_value=types.SimpleNamespace(url='https://example.test/run'))
        self.swan.Text = mock.Mock(side_effect=lambda text, caption: (text, caption))
        self.swan.log = mock.Mock()
        self.swan.finish = mock.Mock()
        self.modules = mock.patch.dict(sys.modules, {'swanlab': self.swan})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.logger = None

    def tearDown(self):
        if self.logger is not None:
            self.logger.logger.clear()

    def records(self, kind):
        rows = []
        for path in Path(self.tmp.name).glob(f'{kind}.pid-*.jsonl.gz'):
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                rows.extend(json.loads(line) for line in stream)
        return rows

    def initialize(self):
        self.logger = tracking.Tracking('test-project', 'test-run', ['swanlab'])

    def validate(self):
        instance = types.SimpleNamespace(
            config=types.SimpleNamespace(trainer=types.SimpleNamespace(log_val_generations=1, logger=['swanlab'])),
            tracking=self.logger, validation_generations_logger=tracking.ValidationGenerationsLogger(),
            global_steps=0)
        namespace['_maybe_log_val_generations'](instance, ['prompt'], ['response'], [0.25])

    def test_init_failure_skips_cloud_but_preserves_samples_and_metrics(self):
        self.swan.init.side_effect = RuntimeError('failed synthetic-test-secret Bearer bearer-secret')
        self.initialize()
        self.validate()
        self.logger.log({'val/score': 0.25}, step=0)
        self.swan.Text.assert_not_called()
        self.swan.log.assert_not_called()
        self.assertEqual(self.records('validation_samples')[0]['output'], 'response')
        self.assertEqual(self.records('metrics')[0]['metrics']['val/score'], 0.25)
        failure = self.records('tracker')[0]
        self.assertEqual(failure['phase'], 'init')
        self.assertNotIn('synthetic-test-secret', failure['error_message'])
        self.assertNotIn('bearer-secret', failure['error_message'])
        self.assertTrue(failure['traceback_frames'])
        self.swan.init.assert_called_once()

    def test_validation_upload_failure_keeps_local_samples_before_attempt(self):
        self.initialize()
        def fail_upload(*args, **kwargs):
            self.assertEqual(self.records('validation_samples')[0]['score'], 0.25)
            raise ConnectionError('synthetic upload failure')
        self.swan.log.side_effect = fail_upload
        self.validate()
        self.assertEqual(self.records('tracker')[-1]['phase'], 'validation')
        self.swan.init.assert_called_once()

    def test_media_construction_failure_does_not_abort(self):
        self.initialize()
        self.swan.Text.side_effect = ValueError('synthetic media error')
        self.validate()
        self.swan.log.assert_not_called()
        self.assertEqual(len(self.records('validation_samples')), 1)
        self.assertEqual(self.records('tracker')[-1]['error_type'], 'ValueError')

    def test_initialized_backend_uploads_once_without_reinitializing(self):
        self.initialize()
        self.validate()
        self.swan.log.assert_called_once()
        self.assertIn('val/generations', self.swan.log.call_args.args[0])
        self.assertEqual(self.swan.log.call_args.kwargs['step'], 0)
        self.swan.init.assert_called_once()
        self.assertEqual(len(self.records('validation_samples')), 1)

    def test_scalar_and_finish_failure_do_not_abort(self):
        self.initialize()
        self.swan.log.side_effect = ConnectionError('synthetic scalar failure')
        self.swan.finish.side_effect = ConnectionError('synthetic finish failure')
        self.logger.log({'train/value': 1}, step=1)
        self.logger.__del__()
        self.assertEqual(self.records('metrics')[0]['global_step'], 1)
        self.assertEqual([r.get('phase') for r in self.records('tracker')][1:], ['metrics', 'finish'])

    def test_disabled_diagnostics_preserves_successful_cloud_logging(self):
        with mock.patch.dict(os.environ, {'GIGPO_DIAG_DIR': ''}):
            self.initialize()
            self.validate()
        self.swan.log.assert_called_once()
        self.assertEqual(self.records('validation_samples'), [])


if __name__ == '__main__':
    unittest.main()
