"""Linux shell control-flow checks; all HPC/ML commands are local fixtures."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.name == "posix", "Requires Linux bash")
class VerifyJobShellTests(unittest.TestCase):
    def run_job(self, failure):
        source = Path(__file__).with_name("verify_actor.sbatch").read_text()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / "experiments/gigpo_diag"
            scripts.mkdir(parents=True)
            env_root = root / "venv"
            (env_root / "bin").mkdir(parents=True)
            (env_root / ".setup-complete").touch()
            (env_root / "bin/activate").touch()
            logs = root / "ray/session_latest/logs"
            logs.mkdir(parents=True)
            (logs / "worker-fixture.err").write_text("original compiler error\n")
            script = scripts / "verify_actor.sbatch"
            script.write_text(source)
            (scripts / "nvidia_job_env.sh").write_text(":\n")
            (scripts / "common.sh").write_text('''
export ORION_ROOT="$FIXTURE_ROOT" GIGPO_VENV="$FIXTURE_ROOT/venv"
export RAY_TMPDIR="$FIXTURE_ROOT/ray"
''')
            (scripts / "launch_reference.sh").write_text('''
printf '%s\\n' "$@" > "$FIXTURE_ROOT/overrides.txt"
printf '%s\\n' "$GIGPO_NUM_GPUS" > "$FIXTURE_ROOT/gpu-count.txt"
if [[ "$FAILURE" == launch ]]; then exit 42; fi
echo fixture-update-returned
''')
            shim = root / "shims.sh"
            shim.write_text('''
module() { :; }
git() { echo fixture-commit; }
nvidia-smi() { echo fixture-gpu; }
timeout() { while [[ "$1" == --* ]]; do shift; done; shift; "$@"; }
python() {
  printf '%s\\n' "$*" >> "$FIXTURE_ROOT/calls.txt"
  if [[ "$*" == *appworld_resource_preflight* && "$FAILURE" == preflight ]]; then return 43; fi
  if [[ "${2-}" == experiments.gigpo_diag.actor_update_evidence && "${3-}" == verify && "$FAILURE" == verifier ]]; then return 44; fi
}
export -f module git nvidia-smi python timeout
''')
            env = {**os.environ, "BASH_ENV": str(shim), "FIXTURE_ROOT": str(root),
                   "FAILURE": failure, "SLURM_JOB_ID": "12345", "SLURM_CPUS_PER_TASK": "32"}
            result = subprocess.run(["bash", str(script)], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=20)
            calls = (root / "calls.txt").read_text()
            runs = list((root / "logs/gigpo-reproduction/appworld").iterdir())
            self.assertEqual(len(runs), 1)
            self.assertTrue((runs[0] / "ray-logs.tar.gz").is_file(), result.stderr)
            self.assertIn(f"exit={result.returncode}", result.stdout)
            if failure != "preflight":
                overrides = (root / "overrides.txt").read_text().splitlines()
                self.assertIn("trainer.total_training_steps=1", overrides)
                self.assertIn("actor_rollout_ref.actor.use_torch_compile=True", overrides)
                self.assertEqual((root / "gpu-count.txt").read_text().strip(), "2")
            return result, calls

    def test_success_requires_post_update_verifier(self):
        result, calls = self.run_job("")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("actor_update_evidence verify", calls)

    def test_pipeline_failure_is_not_hidden_by_tee_or_exit_trap(self):
        result, calls = self.run_job("launch")
        self.assertEqual(result.returncode, 42, result.stderr)
        self.assertNotIn("actor_update_evidence verify", calls)

    def test_environment_preflight_failure_prevents_rollout(self):
        result, calls = self.run_job("preflight")
        self.assertEqual(result.returncode, 43, result.stderr)
        self.assertNotIn("actor_update_evidence verify", calls)

    def test_post_update_evidence_failure_fails_job(self):
        result, _ = self.run_job("verifier")
        self.assertEqual(result.returncode, 44, result.stderr)


if __name__ == "__main__":
    unittest.main()
