"""Linux shell fixtures: exercise full job control flow without ML or Slurm."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.name == 'posix', 'Requires Linux bash')
class FullTrainingShellTests(unittest.TestCase):
    def run_job(self, failure):
        source = Path(__file__).with_name('train.sbatch').read_text()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / 'experiments/gigpo_diag'
            scripts.mkdir(parents=True)
            venv = root / 'venv'
            (venv / 'bin').mkdir(parents=True)
            for name in ['.setup-complete', 'bin/activate', 'gigpo-pip-freeze.txt']:
                (venv / name).touch()
            (scripts / 'CONFIG_DIFF.md').touch()
            (scripts / 'nvidia_job_env.sh').write_text(':\n')
            (scripts / 'common.sh').write_text('''
export ORION_ROOT="$FIXTURE_ROOT" GIGPO_VENV="$FIXTURE_ROOT/venv"
export RAY_TMPDIR="$FIXTURE_ROOT/ray"
''')
            (scripts / 'launch_reference.sh').write_text('''
printf '%s\\n' "$@" > "$FIXTURE_ROOT/overrides.txt"
printf '%s\\n' "$GIGPO_NUM_GPUS $GIGPO_ACTOR_VERIFY" > "$FIXTURE_ROOT/actual-config.txt"
if [[ "$FAILURE" == launch* ]]; then exit 42; fi
echo fixture-full-training
''')
            logs = root / 'ray/session_latest/logs'
            logs.mkdir(parents=True)
            (logs / 'worker.err').write_text('original compiler error\n')
            script = scripts / 'train.sbatch'
            script.write_text(source)
            shim = root / 'shims.sh'
            shim.write_text('''
module() { :; }
git() { echo fixture-commit; }
nvidia-smi() { echo fixture-gpu; }
timeout() { while [[ "$1" == --* ]]; do shift; done; shift; "$@"; }
tar() { if [[ "$FAILURE" == *archive ]]; then return 49; fi; command tar "$@"; }
python() {
  printf '%s\\n' "$*" >> "$FIXTURE_ROOT/calls.txt"
  if [[ "$*" == *bounded_acceptance* && "$FAILURE" == acceptance ]]; then return 41; fi
  if [[ "$*" == *appworld_resource_preflight* && "$FAILURE" == preflight ]]; then return 43; fi
}
export -f module git nvidia-smi timeout tar python
''')
            env = {**os.environ, 'BASH_ENV': str(shim), 'FIXTURE_ROOT': str(root),
                   'FAILURE': failure, 'SLURM_JOB_ID': '12345', 'SLURM_CPUS_PER_TASK': '32',
                   'GIGPO_PREFLIGHT_ONLY': '0', 'GIGPO_RUN_ID': ''}
            result = subprocess.run(['bash', str(script)], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=20)
            calls = (root / 'calls.txt').read_text()
            runs = list((root / 'logs/gigpo-reproduction/appworld').iterdir())
            self.assertEqual(len(runs), 1)
            self.assertIn(f'exit={result.returncode}', result.stdout)
            self.assertIn('bounded_acceptance verify', calls)
            self.assertNotIn('bounded_acceptance publish', calls)
            self.assertNotIn('actor_update_evidence verify', calls)
            if failure in ('acceptance', 'preflight'):
                self.assertFalse((root / 'overrides.txt').exists())
            else:
                self.assertEqual((root / 'overrides.txt').read_text().splitlines(), ['vllm'])
                self.assertEqual((root / 'actual-config.txt').read_text().strip(), '4 1')
            if failure.endswith('archive'):
                self.assertIn('RAY_LOG_ARCHIVE_EXIT=49', result.stdout)
            else:
                self.assertTrue((runs[0] / 'ray-logs.tar.gz').is_file(), result.stderr)
            return result

    def test_full_schedule_and_four_workers(self):
        self.assertEqual(self.run_job('').returncode, 0)

    def test_missing_acceptance_prevents_training(self):
        self.assertEqual(self.run_job('acceptance').returncode, 41)

    def test_failed_node_resource_calibration_prevents_training(self):
        self.assertEqual(self.run_job('preflight').returncode, 43)

    def test_train_failure_survives_tee_and_archive(self):
        self.assertEqual(self.run_job('launch').returncode, 42)

    def test_failed_archive_does_not_mask_train_failure(self):
        self.assertEqual(self.run_job('launch-archive').returncode, 42)

    def test_failed_archive_does_not_turn_success_into_training_failure(self):
        self.assertEqual(self.run_job('archive').returncode, 0)


if __name__ == '__main__':
    unittest.main()
