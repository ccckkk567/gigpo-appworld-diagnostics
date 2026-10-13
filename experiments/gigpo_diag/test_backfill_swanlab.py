"""Synthetic fixtures only; no cloud writes and no claimed training results."""
import gzip
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location('backfill', Path(__file__).with_name('backfill_swanlab.py'))
backfill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backfill)


class BackfillTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'source'
        (self.root / 'diagnostics').mkdir(parents=True)
        self.output = Path(self.tmp.name) / 'export'
        self.manifest = {'run_id': 'synthetic-run', 'code_commit': 'abc', 'exit_code': 1,
                         'training_updates': 0, 'policy_updated': False}
        self.write_manifest()
        self.rows = [dict(self.manifest, global_step=5, metrics={'loss': .2, 'unknown': None}),
                     dict(self.manifest, global_step=0, metrics={'val/success': 0})]
        self.write_rows()

    def write_manifest(self):
        (self.root / 'RUN_MANIFEST.json').write_text(json.dumps(self.manifest), encoding='utf-8')

    def write_rows(self):
        with gzip.open(self.root / 'diagnostics/metrics.pid-1.jsonl.gz', 'wt', encoding='utf-8') as f:
            for row in self.rows:
                f.write(json.dumps(row) + '\n')

    def test_order_unknown_and_source_preserved(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        payload = backfill.read_snapshot(self.root)
        self.assertEqual([p['step'] for p in payload['points']], [0, 5])
        self.assertEqual(payload['points'][1]['metrics'], {'loss': .2})
        self.assertEqual(payload['source']['training_updates'], 0)
        backfill.persist_snapshot(payload, self.output)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_running_run_rejected(self):
        self.manifest['exit_code'] = None
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'wait'):
            backfill.read_snapshot(self.root)

    def test_mixed_source_and_conflicts_rejected(self):
        self.rows.append(dict(self.manifest, global_step=5, metrics={'loss': .8}))
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            backfill.read_snapshot(self.root)
        self.rows[-1]['run_id'] = 'another-run'
        self.write_rows()
        with self.assertRaisesRegex(ValueError, 'Mixed'):
            backfill.read_snapshot(self.root)

    def test_partial_gzip_rejected(self):
        path = self.root / 'diagnostics/metrics.pid-1.jsonl.gz'
        path.write_bytes(path.read_bytes()[:-5])
        with self.assertRaises(EOFError):
            backfill.read_snapshot(self.root)

    def test_upload_preserves_steps_and_does_not_repeat_completed_upload(self):
        payload = backfill.read_snapshot(self.root)
        sha = backfill.persist_snapshot(payload, self.output)
        sdk = types.SimpleNamespace(init=mock.Mock(return_value=types.SimpleNamespace(url='https://example.test')),
                                    log=mock.Mock(), finish=mock.Mock())
        with mock.patch.dict('sys.modules', {'swanlab': sdk}), mock.patch.object(
                backfill.importlib.metadata, 'version', return_value='0.10.2'):
            receipt = backfill.upload(payload, self.output, sha, 'test-project', 'test-workspace')
            backfill.upload(payload, self.output, sha, 'test-project', 'test-workspace')
        sdk.init.assert_called_once()
        self.assertEqual([c.kwargs['step'] for c in sdk.log.call_args_list], [0, 5])
        self.assertEqual(receipt['state'], 'sdk_finish_returned')
        self.assertFalse(receipt['cloud_contents_verified'])

    def test_upload_failure_keeps_snapshot_and_receipt(self):
        payload = backfill.read_snapshot(self.root)
        sha = backfill.persist_snapshot(payload, self.output)
        sdk = types.SimpleNamespace(init=mock.Mock(return_value=types.SimpleNamespace(url='https://example.test')),
                                    log=mock.Mock(side_effect=ConnectionError('synthetic')), finish=mock.Mock())
        with mock.patch.dict('sys.modules', {'swanlab': sdk}), mock.patch.object(
                backfill.importlib.metadata, 'version', return_value='0.10.2'):
            with self.assertRaises(ConnectionError):
                backfill.upload(payload, self.output, sha, 'test-project', 'test-workspace')
            with self.assertRaisesRegex(RuntimeError, 'uncertain'):
                backfill.upload(payload, self.output, sha, 'test-project', 'test-workspace')
        sdk.init.assert_called_once()
        self.assertTrue((self.output / 'snapshot.json').exists())


if __name__ == '__main__':
    unittest.main()
