import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.gigpo_diag.acceptance_reuse import (
    SOURCE_COMMIT, REVIEWED_FILES, OTHER_REVIEWED_PATHS, validate_changes)
from experiments.gigpo_diag.bounded_acceptance import verify_compatible


class AcceptanceReuseTests(unittest.TestCase):
    def setUp(self):
        self.current = {'code_commit': 'current', 'model_path': 'existing-7b',
                        'packages': {'torch': '2.6.0', 'triton': '3.2.0'},
                        'python_version': '3.12.5', 'world_size': 4,
                        'job_requested_mib': 327680}
        self.accepted = {**copy.deepcopy(self.current), 'passed': True,
                         'code_commit': SOURCE_COMMIT}

    def test_exact_commit_needs_no_reuse(self):
        self.accepted['code_commit'] = 'current'
        self.assertIsNone(verify_compatible(self.accepted, self.current))

    def test_changed_commit_requires_explicit_matching_source(self):
        for source in (None, '', 'wrong'):
            with self.subTest(source=source), self.assertRaises(RuntimeError):
                verify_compatible(self.accepted, self.current, source)

    def test_changed_runtime_identity_cannot_be_overridden(self):
        for field, value in [('model_path', 'other-model'), ('packages', {'torch': 'other'}),
                             ('python_version', '3.12.6'), ('world_size', 2),
                             ('job_requested_mib', 262144)]:
            current = {**self.current, field: value}
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                verify_compatible(self.accepted, current, SOURCE_COMMIT)

    def test_failed_acceptance_is_never_reused(self):
        self.accepted['passed'] = False
        with self.assertRaises(RuntimeError):
            verify_compatible(self.accepted, self.current, SOURCE_COMMIT)

    def test_explicit_reuse_requires_code_review_check(self):
        with patch('experiments.gigpo_diag.acceptance_reuse.reviewed_changes') as check:
            check.return_value = {'source_commit': SOURCE_COMMIT}
            self.assertEqual(verify_compatible(self.accepted, self.current, SOURCE_COMMIT),
                             check.return_value)
            check.assert_called_once_with(SOURCE_COMMIT, 'current')
            check.side_effect = RuntimeError('Unreviewed change')
            with self.assertRaises(RuntimeError):
                verify_compatible(self.accepted, self.current, SOURCE_COMMIT)

    def test_resource_repair_cannot_reuse_diagnostic_only_acceptance(self):
        repo = Path(__file__).resolve().parents[2]
        with self.assertRaisesRegex(RuntimeError, 'unreviewed content'):
            validate_changes(['experiments/gigpo_diag/train.sbatch'],
                             lambda path: (repo / path).read_text(encoding='utf-8').encode())

    def test_unreviewed_algorithm_or_configuration_changes_fail(self):
        for path in ['verl/trainer/ppo/core_algos.py', 'experiments/gigpo_diag/launch_reference.sh',
                     'experiments/gigpo_diag/appworld_resource.py', 'unknown.py']:
            with self.subTest(path=path), self.assertRaises(RuntimeError):
                validate_changes([path], lambda _: b'changed')

    def test_observer_or_native_worker_modification_fails(self):
        for path in REVIEWED_FILES:
            with self.subTest(path=path), self.assertRaises(RuntimeError):
                validate_changes([path], lambda _: b'altered after review')


if __name__ == '__main__':
    unittest.main()
