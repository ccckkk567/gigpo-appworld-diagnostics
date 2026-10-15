"""Regression for real AppWorld DB path handling; no tasks/GPU/allocation loop."""
import contextlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from appworld.apps.lib.models.db import apply_db_changes

from experiments.gigpo_diag import appworld_resource_preflight as preflight


class ProbePathTests(unittest.TestCase):
    def test_generated_probe_paths_load_native_db_changes(self):
        with tempfile.TemporaryDirectory(prefix='appworld-probe-') as td:
            root = Path(td)
            datasets = root / 'data' / 'datasets'
            datasets.mkdir(parents=True)
            for split in ('train', 'dev'):
                (datasets / (split + '.txt')).write_text('task-a\ntask-b\n')
            profile = root / 'profile.json'
            calls = []

            def probe(directory, task, mode, limit, **kwargs):
                # Use main's actual directory selection and the installed DB
                # loader. Only task execution and its measurements are fixtures.
                changes = directory / 'appworld' / 'data' / 'tasks' / task / 'dbs' / 'spotify.jsonl'
                changes.parent.mkdir(parents=True)
                changes.write_text(json.dumps(['CREATE TABLE probe (value TEXT)', [], False]) + '\n'
                                   + json.dumps(['INSERT INTO probe VALUES (?)', [mode], False]) + '\n',
                                   encoding='utf-8')
                with contextlib.closing(sqlite3.connect(':memory:')) as connection:
                    apply_db_changes(connection, str(changes))
                    self.assertEqual(connection.execute('SELECT value FROM probe').fetchone(), (mode,))
                calls.append((directory.name, mode))
                return {'sampled_peaks': {'vms_bytes': 200 * 1024**2, 'rss_bytes': 100 * 1024**2},
                        'events': [{'event': 'evaluate_begin', 'VmPeak_bytes': 150 * 1024**2,
                                    'VmSize_bytes': 140 * 1024**2},
                                   {'event': 'evaluate_complete', 'VmPeak_bytes': 200 * 1024**2}]}

            env = {'SLURM_MEM_PER_NODE': str(320 * 1024),
                   'GIGPO_RUN_DIR': str(root / 'run'), 'GIGPO_APPWORLD_DATA': str(root),
                   'GIGPO_CODE_COMMIT': 'test', 'GIGPO_RESOURCE_PROFILE': str(profile)}
            with patch.dict(os.environ, env), patch.object(preflight.sys, 'platform', 'linux'), \
                    patch.object(preflight, 'run_probe', side_effect=probe), \
                    patch.object(preflight, 'cgroup_memory', return_value={}), \
                    patch.object(preflight.importlib.metadata, 'version', return_value='fixture'), \
                    contextlib.redirect_stdout(io.StringIO()):
                preflight.main()
            self.assertEqual(len(calls), 18)
            self.assertEqual([mode for _, mode in calls[-3:]],
                             ['allocate', 'catch-memory', 'swallow-deadline'])
            self.assertEqual(sum(mode == 'healthy' for _, mode in calls), 10)
            self.assertEqual(sum(mode == 'retained-pressure' for _, mode in calls), 5)
            self.assertTrue(json.loads(profile.read_text())['calibrated'])


if __name__ == '__main__':
    unittest.main()
