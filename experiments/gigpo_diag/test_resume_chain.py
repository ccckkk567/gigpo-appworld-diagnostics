"""No scheduler/network/GPU use: real checkpoint manifests plus Slurm fixtures."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.gigpo_diag import resume_chain as rc
from experiments.gigpo_diag.test_checkpoint_transaction import populate


class CheckpointFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tx = rc.load_transactions(rc.HELPER_CHECKOUT)

    def run_dir(self, job='100', suffix='20261015T120000Z', target=150):
        run = self.root / f'appworld-qwen25-7b-gigpo-{job}-{suffix}'
        run.mkdir(exist_ok=True)
        rc.write_json(run / 'RUN_MANIFEST.json', {
            'env': 'appworld', 'job_id': job, 'code_commit': rc.TRAINING_COMMIT,
            'initial_checkpoint': str(rc.MODEL_PATH), 'reference_schedule_epochs': 150,
            'seed': 0, 'resource_protocol': 'bounded_runtime_v1',
            'checkpoint_dir': str(run / 'checkpoints')})
        (run / 'actual_command.txt').write_text('bash experiments/gigpo_diag/launch_reference.sh vllm\n')
        (run / 'train.log').write_text(f'(TaskRunner pid=40) Total training steps: {target}\n')
        return run

    def checkpoint(self, run, step, world_size=4):
        stage, final = self.tx.begin_checkpoint(run / 'checkpoints', step)
        populate(stage, step, world_size=world_size)
        self.tx.publish_checkpoint(stage, final, step)
        return final

    def select(self, jobs=('100', '101')):
        return rc.select_checkpoint(self.root, list(jobs), rc.TRAINING_COMMIT, self.tx)


class SelectionTests(CheckpointFixture):
    def test_latest_complete_selected_at_start_across_chain_only(self):
        first = self.run_dir()
        self.checkpoint(first, 10)
        second = self.run_dir('101')
        self.checkpoint(second, 20)
        # A valid later checkpoint is published after our first selection.
        self.assertEqual(self.select()['global_step'], 20)
        newest = self.checkpoint(second, 30)
        self.checkpoint(self.run_dir('999'), 140)
        pending, _ = self.tx.begin_checkpoint(second / 'checkpoints', 40)
        populate(pending, 40, world_size=4)
        (second / 'checkpoints/latest_checkpointed_iteration.txt').write_text('20\n')
        result = self.select()
        self.assertEqual(result['checkpoint'], str(newest))
        self.assertEqual(result['global_step'], 30)
        self.assertEqual(result['status'], 'resume')

    def test_truncated_rank_and_wrong_world_size_are_not_usable(self):
        run = self.run_dir()
        valid = self.checkpoint(run, 10)
        broken = self.checkpoint(run, 20)
        (broken / 'actor/optim_world_size_4_rank_3.pt').write_bytes(b'broken')
        self.checkpoint(run, 30, world_size=2)
        result = self.select()
        self.assertEqual(result['checkpoint'], str(valid))
        self.assertEqual(len(result['skipped_checkpoints']), 2)

    def test_missing_checkpoint_never_restarts_from_model(self):
        run = self.run_dir()
        pending, _ = self.tx.begin_checkpoint(run / 'checkpoints', 10)
        populate(pending, 10, world_size=4)
        with self.assertRaisesRegex(RuntimeError, 'refusing to restart'):
            self.select()

    def test_wrong_model_commit_or_job_refused(self):
        run = self.run_dir()
        self.checkpoint(run, 10)
        manifest = rc.read_json(run / 'RUN_MANIFEST.json')
        for key in ('code_commit', 'initial_checkpoint', 'job_id', 'reference_schedule_epochs'):
            with self.subTest(key=key):
                rc.write_json(run / 'RUN_MANIFEST.json', {**manifest, key: 'wrong'})
                with self.assertRaisesRegex(RuntimeError, 'Incompatible'):
                    self.select()

    def test_native_schedule_and_final_checkpoint(self):
        run = self.run_dir(target=150)
        self.checkpoint(run, 150)
        self.assertEqual(self.select()['status'], 'already_complete')
        self.checkpoint(run, 160)
        with self.assertRaisesRegex(RuntimeError, 'exceeds'):
            self.select()

    def test_conflicting_or_missing_native_schedule_refused(self):
        first = self.run_dir()
        self.checkpoint(first, 10)
        other = self.run_dir('101', target=200)
        self.checkpoint(other, 20)
        with self.assertRaisesRegex(RuntimeError, 'Inconsistent'):
            self.select()
        (other / 'train.log').write_text('No native initialization completed\n')
        with self.assertRaisesRegex(RuntimeError, 'Cannot establish'):
            self.select()


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'chain.json'
        self.chain = {'parent_job': '163354', 'helper_checkout': '/helper',
                      'jobs': [{'slot': 0, 'job_id': '163354', 'status': 'submitted'}]}

    def test_three_serial_dependencies_and_idempotent_reentry(self):
        calls = []
        def submit(*args, **kwargs):
            calls.append(args)
            # Intent was durable before sbatch could succeed.
            self.assertEqual(rc.read_json(self.path)['jobs'][-1]['status'], 'submitting')
            self.assertIn('--gres=gpu:4', args)
            self.assertIn('--mem=320G', args)
            self.assertIn('--time=17:50:00', args)
            self.assertIn('--no-requeue', args)
            return str(200000 + len(calls)) + ';cluster'
        jobs = rc.submit_jobs(self.chain, self.path, submit)
        self.assertEqual(jobs, ['163354', '200001', '200002', '200003'])
        for call, previous in zip(calls, jobs):
            self.assertIn(f'--dependency=afterany:{previous}', call)
        rc.submit_jobs(rc.read_json(self.path), self.path, submit)
        self.assertEqual(len(calls), 3)

    def test_partial_known_submission_continues_without_duplicate(self):
        self.chain['jobs'].append({'slot': 1, 'status': 'submitted', 'job_id': '200001'})
        calls = []
        def submit(*args, **kwargs):
            calls.append(args)
            return str(200001 + len(calls))
        rc.submit_jobs(self.chain, self.path, submit)
        self.assertEqual(len(calls), 2)
        self.assertIn('--dependency=afterany:200001', calls[0])

    def test_ambiguous_sbatch_outcome_never_duplicates(self):
        with self.assertRaises(TimeoutError):
            rc.submit_jobs(self.chain, self.path, lambda *a, **k: (_ for _ in ()).throw(TimeoutError()))
        with self.assertRaisesRegex(RuntimeError, 'unresolved submission'):
            rc.submit_jobs(rc.read_json(self.path), self.path, lambda *a, **k: self.fail('Must not resubmit'))

    def test_parent_resource_and_workdir_checks(self):
        fields = {'JobId': '163354', 'Account': 'prjsnnn0000269', 'Partition': 'testqueue',
                  'WorkDir': str(rc.TRAINING_CHECKOUT),
                  'Command': str(rc.TRAINING_CHECKOUT / 'experiments/gigpo_diag/train.sbatch'),
                  'TimeLimit': '17:50:00', 'NumNodes': '1', 'NumCPUs': '32',
                  'JobState': 'PENDING', 'UserId': 'linz3(173020)',
                  'ReqTRES': 'cpu=32,mem=320G,node=1,billing=40,gres/gpu=4'}
        with patch.dict(os.environ, USER='linz3'):
            rc.verify_parent('163354', fields)
            for key, value in [('ReqTRES', 'mem=256G,gres/gpu=2'), ('WorkDir', '/other'),
                               ('UserId', 'another(1)'), ('JobState', 'COMPLETED')]:
                with self.subTest(key=key), self.assertRaises(RuntimeError):
                    rc.verify_parent('163354', {**fields, key: value})


class RunTests(CheckpointFixture):
    def setUp(self):
        super().setUp()
        self.chain_path = self.root / 'resume-chains/from-100/chain.json'
        rc.write_json(self.chain_path, {
            'version': 1, 'parent_job': '100', 'training_commit': rc.TRAINING_COMMIT,
            'training_checkout': str(rc.TRAINING_CHECKOUT), 'helper_checkout': str(rc.HELPER_CHECKOUT),
            'helper_commit': 'helper-version', 'runs_root': str(self.root),
            'acceptance_dir': str(rc.ACCEPTANCE_DIR),
            'jobs': [{'slot': 0, 'job_id': '100', 'status': 'submitted'},
                     {'slot': 1, 'job_id': '101', 'status': 'submitted'}]})

    def start(self, queue=''):
        with patch.object(rc, 'RUNS_ROOT', self.root), patch.object(rc, 'verify_checkout'), \
             patch.object(rc, 'command', return_value=queue) as command, \
             patch.object(rc, 'load_transactions', return_value=self.tx), \
             patch.object(rc.os, 'chdir'), patch.object(rc.os, 'execvpe') as execute, \
             patch.dict(os.environ, SLURM_JOB_ID='101', USER='linz3'):
            rc.run_resume(self.chain_path, 1)
            command.assert_called_once_with('squeue', '-h', '-u', 'linz3', '-o', '%i %T')
            return execute

    def test_runtime_executes_pinned_training_with_full_resume(self):
        checkpoint = self.checkpoint(self.run_dir(), 10)
        execute = self.start(queue='999 RUNNING')
        args = execute.call_args.args
        self.assertEqual(args[:2], ('bash', ['bash', 'experiments/gigpo_diag/train.sbatch']))
        env = args[2]
        self.assertEqual(env['GIGPO_RESUME_MODE'], 'resume_path')
        self.assertEqual(env['GIGPO_RESUME_FROM'], str(checkpoint))
        self.assertEqual(env['GIGPO_EXPECTED_COMMIT'], rc.TRAINING_COMMIT)
        evidence = next(self.chain_path.parent.glob('starts/*.json'))
        self.assertEqual(rc.read_json(evidence)['global_step'], 10)

    def test_completion_does_not_launch_model(self):
        self.checkpoint(self.run_dir(), 150)
        self.start().assert_not_called()

    def test_missing_checkpoint_or_active_predecessor_does_not_launch(self):
        with self.assertRaisesRegex(RuntimeError, 'No complete'):
            self.start()
        self.checkpoint(self.run_dir(), 10)
        with self.assertRaisesRegex(RuntimeError, 'still active'):
            self.start(queue='100 RUNNING')

    def test_swanlab_separate_run_and_lineage_without_step_rewriting(self):
        old = {'SWANLAB_RUN_ID': 'old-cloud-id', 'SWANLAB_RESUME': 'must',
               'SWANLAB_RUN_DIR': 'old-dir', 'SWANLAB_RUN_PARALLEL': 'shared',
               'GIGPO_RUN_ID': 'old-training-run', 'GIGPO_PREFLIGHT_ONLY': '1',
               'GIGPO_RESUME_MODE': 'disable'}
        decision = {'checkpoint': '/source/global_step_10', 'global_step': 10}
        with patch.dict(os.environ, old):
            env = rc.training_environment(decision, '/acceptance', '100', 1, '101')
        self.assertEqual(env['SWANLAB_RESUME'], 'never')
        self.assertNotIn('SWANLAB_RUN_ID', env)
        self.assertNotIn('SWANLAB_RUN_DIR', env)
        self.assertNotIn('SWANLAB_RUN_PARALLEL', env)
        self.assertEqual(env['SWANLAB_GROUP'], 'appworld-chain-100')
        self.assertIn('resume-1,from-step-10', env['SWANLAB_TAGS'])
        self.assertIn('/source/global_step_10', env['SWANLAB_DESCRIPTION'])
        self.assertTrue(env['GIGPO_RUN_ID'].startswith('appworld-qwen25-7b-gigpo-101-'))
        self.assertTrue(env['GIGPO_RUN_ID'].endswith('-chain100-r1-from10'))
        self.assertEqual(env['GIGPO_PREFLIGHT_ONLY'], '0')


if __name__ == '__main__':
    unittest.main()
