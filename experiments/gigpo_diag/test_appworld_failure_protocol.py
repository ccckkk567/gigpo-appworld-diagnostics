"""CPU-only worker/collector-boundary fixtures; Ray and AppWorld RPCs are stubbed.

No model, real task database, kernel resource limit, or Ray cluster is exercised
here. The service integration preflight must separately verify those boundaries.
"""
import importlib.util
import ast
import json
import os
import pickle
from pathlib import Path
import sys
import tempfile
import types
import unittest
import uuid
from unittest import mock

import numpy as np
from experiments.gigpo_diag import diagnostics
from experiments.gigpo_diag import appworld_resource as real_resource
from experiments.gigpo_diag.appworld_resource import ResourceTerminal, InfrastructureFailure


REPO = Path(__file__).resolve().parents[2]


class Future:
    def __init__(self, method, args):
        self.method, self.args = method, args

    def get(self):
        return self.method(*self.args)


class Actor:
    def __init__(self, instance):
        self.instance = instance

    def __getattr__(self, name):
        method = getattr(self.instance, name)
        return types.SimpleNamespace(remote=lambda *args: Future(method, args))


class Client:
    def __init__(self, **kwargs):
        self.task_id = kwargs['task_id']
        self.experiment_name = kwargs['experiment_name']
        self.remote_environment_url = kwargs['remote_environment_url']
        self.timeout_seconds = 100
        self.num_interactions = 0
        self.task = types.SimpleNamespace(instruction='Task instruction', supervisor={'first_name': 'Synthetic'})
        self.execute = mock.Mock(return_value='native successful output')
        self.task_completed = mock.Mock(return_value=False)
        self.evaluate = mock.Mock(return_value=types.SimpleNamespace(success=True))
        self.save_logs = mock.Mock()
        self.close = mock.Mock()


class TensorRows(dict):
    def __len__(self):
        return len(next(iter(self.values()))) if dict.__len__(self) else 0


class CollectorBatch:
    """Minimal DataProto carrier for exercising unchanged collector methods."""
    def __init__(self, tensors=None, non_tensors=None):
        self.batch = TensorRows(tensors or {})
        self.non_tensor_batch = non_tensors or {}
        self.meta_info = {}

    def copy(self):
        return CollectorBatch({k: v.copy() for k, v in self.batch.items()},
                              {k: v.copy() for k, v in self.non_tensor_batch.items()})

    def pop(self, batch_keys, non_tensor_batch_keys):
        return CollectorBatch({k: self.batch.pop(k) for k in batch_keys},
                              {k: self.non_tensor_batch.pop(k) for k in non_tensor_batch_keys})

    def union(self, other):
        self.batch.update(other.batch)
        self.non_tensor_batch.update(other.non_tensor_batch)
        return self

    @classmethod
    def from_single_dict(cls, data):
        return data


def load_collector_methods():
    """Compile the actual loop/filter method bodies, not a rewritten mask loop.

    Only import-time GPU dependencies and data/model carriers are replaced.
    """
    path = REPO / 'agent_system/multi_turn_rollout/rollout_loop.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TrajectoryCollector')
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in ('vanilla_multi_turn_loop', 'gather_rollout_data')]
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + methods,
                      type_ignores=[])
    namespace = {
        'np': np, 'uuid': uuid, 'DataProto': CollectorBatch,
        'diag': types.SimpleNamespace(enabled=lambda: False, verify_groups=diagnostics.verify_groups,
                                      begin_rollout=lambda *args: None, step=lambda *args: None,
                                      end_rollout=lambda *args: None),
        'pad_dataproto_to_divisor': lambda batch, divisor: (batch, 0),
        'unpad_dataproto': lambda batch, pad_size: batch,
        'torch_to_numpy': lambda value, is_object=False: np.asarray(value, dtype=object if is_object else None),
        'to_list_of_dict': lambda batch: [{k: v[i] for k, v in {**batch.batch, **batch.non_tensor_batch}.items()}
                                         for i in range(len(batch.batch))],
        'collate_fn': lambda rows: rows,
    }
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    return namespace


def load_worker_module():
    ray = types.ModuleType('ray')
    ray.is_initialized = lambda: True
    ray.get = lambda futures: [future.get() for future in futures]
    ray.remote = lambda **resources: lambda cls: types.SimpleNamespace(
        remote=lambda **kwargs: Actor(cls(**kwargs)))
    ray.kill = mock.Mock()
    appworld = types.ModuleType('appworld')
    appworld.load_task_ids = lambda split: ['task-a', 'task-b', 'task-c']
    client = types.ModuleType('experiments.gigpo_diag.appworld_client')
    client.AppWorldClient, client.EXECUTE_HTTP_GRACE_SECONDS = Client, 60
    resource = types.ModuleType('experiments.gigpo_diag.appworld_resource')
    resource.ResourceTerminal, resource.InfrastructureFailure = ResourceTerminal, InfrastructureFailure
    resource.wait_service = mock.Mock(side_effect=lambda worker_id, **kwargs: {
        'port': 20000 + worker_id, 'pid': 1000 + worker_id, 'generation': 0,
    })
    name = 'appworld_worker_protocol_fixture'
    path = REPO / 'agent_system/environments/env_package/appworld/envs.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {
        'ray': ray, 'appworld': appworld,
        client.__name__: client, resource.__name__: resource,
    }):
        spec.loader.exec_module(module)
    return module


class FailureProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = mock.patch.dict(os.environ, {
            'GIGPO_RESOURCE_PROFILE': str(Path(self.temp.name) / 'profile.json'),
            'GIGPO_RUN_DIR': self.temp.name, 'GIGPO_DIAG_DIR': self.temp.name,
            'GIGPO_RUN_ID': 'synthetic-bounded-test', 'APPWORLD_ROOT': self.temp.name,
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.module = load_worker_module()
        self.sleep = mock.patch.object(self.module.time, 'sleep')
        self.sleep.start()
        self.addCleanup(self.sleep.stop)
        rng = np.random.get_state()
        self.addCleanup(np.random.set_state, rng)

    def wrapper(self):
        ports = Path(self.temp.name) / 'ports'
        ports.write_text('\n'.join(str(20000 + i) for i in range(8)), encoding='utf-8')
        wrapper = self.module.AppWorldEnvs('train', 50, 0, 1, 8, 0, {}, str(ports))
        wrapper.reset()
        return wrapper

    def context(self, wrapper, step=0, active=None):
        wrapper.set_rollout_context(
            ['group-a'] * 8, [f'traj-{i}' for i in range(8)], step,
            [f'<code>action-{i}-{step}</code>' for i in range(8)],
            [True] * 8 if active is None else active,
        )

    def test_one_resource_terminal_in_eight_member_group_preserves_peers_and_masks(self):
        # Exercise every possible failed slot in an eight-member GiGPO group.
        for terminal_slot in range(8):
            with self.subTest(terminal_slot=terminal_slot):
                wrapper = self.wrapper()
                worker = wrapper.workers[terminal_slot].instance
                dead_client = worker.env
                dead_client.execute.side_effect = ResourceTerminal({
                    'reason': 'service_memory_budget', 'request_id': f'request-{terminal_slot}',
                })
                self.context(wrapper)
                observations, rewards, dones, infos = wrapper.step([f'action-{i}-0' for i in range(8)])
                self.assertEqual(sum(dones), 1)
                self.assertTrue(dones[terminal_slot])
                self.assertEqual(rewards, [0.0] * 8)
                self.assertIn('bounded_runtime_v1', observations[terminal_slot])
                self.assertIsNone(infos[terminal_slot]['native_evaluator_success'])
                self.assertFalse(infos[terminal_slot]['budgeted_task_success'])
                self.assertEqual(infos[terminal_slot]['step_count'], 1)
                dead_client.task_completed.assert_not_called()
                dead_client.evaluate.assert_not_called()
                dead_client.save_logs.assert_not_called()
                active_masks = np.logical_not(dones)
                self.context(wrapper, 1, active_masks)
                _, rewards2, dones2, infos2 = wrapper.step([f'action-{i}-1' for i in range(8)])
                dead_client.execute.assert_called_once_with(f'action-{terminal_slot}-0')
                self.assertEqual(worker.current_step_count, 1)
                self.assertEqual(rewards2[terminal_slot], 0)
                self.assertTrue(dones2[terminal_slot])
                self.assertTrue(infos2[terminal_slot]['inactive'])
                # Native collector stores only active rows: 8 first + 7 second.
                self.assertEqual(8 + int(active_masks.sum()), 15)
                for slot in range(8):
                    if slot != terminal_slot:
                        self.assertEqual(wrapper.workers[slot].instance.env.execute.call_count, 2)

    def test_unknown_failure_persists_all_identities_and_aborts_before_advantages(self):
        wrapper = self.wrapper()
        self.context(wrapper)
        wrapper.step([f'previous-action-{i}' for i in range(8)])
        error = InfrastructureFailure({'reason': 'unexplained_disconnect', 'request_id': 'unknown-request'})
        wrapper.workers[3].instance.env.execute.side_effect = error
        self.context(wrapper, 1)
        reached_advantages = False
        with self.assertRaises(InfrastructureFailure):
            wrapper.step([f'failed-step-action-{i}' for i in range(8)])
            reached_advantages = True
        self.assertFalse(reached_advantages)
        rows = [json.loads(line) for line in wrapper._evidence_path.read_text(encoding='utf-8').splitlines()]
        abort = rows[-1]
        self.assertEqual(abort['event'], 'infrastructure_abort')
        self.assertEqual(abort['decision'], 'abort_entire_rollout_before_advantages')
        self.assertEqual(abort['context']['episode_group_uids'], ['group-a'] * 8)
        self.assertEqual(abort['context']['trajectory_uids'], [f'traj-{i}' for i in range(8)])
        self.assertEqual(len(abort['context']['raw_actions']), 8)
        self.assertEqual(len(abort['task_ids']), 8)
        self.assertEqual(rows[-2]['executed_actions'], [f'failed-step-action-{i}' for i in range(8)])
        previous = next(row for row in rows if row['event'] == 'step_results')
        self.assertEqual(previous['results'][0][0], 'native successful output')
        self.assertFalse(abort['retry'])
        with self.assertRaisesRegex(RuntimeError, 'refusing replay'):
            wrapper.step(['do-not-replay'] * 8)
        with self.assertRaisesRegex(RuntimeError, 'refusing replay'):
            wrapper.reset()

    def test_healthy_memoryerror_and_timeout_feedback_keep_native_path(self):
        worker = self.module.AppWorldWorker(2, 50, 20002)
        worker.reset('task-a')
        for feedback in ('Execution failed. Traceback: MemoryError',
                         'Execution failed. Traceback: Exception: Execution timed out after 100 seconds.'):
            worker.env.execute.return_value = feedback
            observation, reward, done, info = worker.step('native-error-code')
            self.assertEqual(observation, feedback)
            self.assertEqual(reward, 0)
            self.assertFalse(done)
            self.assertFalse(info['resource_terminal'])
        self.assertEqual(worker.env.execute.call_count, 2)
        self.assertEqual(worker.env.task_completed.call_count, 2)
        self.assertEqual(worker.env.save_logs.call_count, 2)

    def test_normal_success_is_sticky_and_reward_is_paid_once(self):
        worker = self.module.AppWorldWorker(0, 50, 20000)
        worker.reset('task-a')
        worker.env.task_completed.return_value = True
        first = worker.step('complete-task')
        second = worker.step('must-never-execute')
        self.assertEqual(first[1:3], (10.0, True))
        self.assertEqual(second[1:3], (0.0, True))
        worker.env.execute.assert_called_once_with('complete-task')
        worker.env.evaluate.assert_called_once()
        self.assertEqual(worker.current_step_count, 1)

    def test_resource_terminal_reset_uses_new_service_and_fresh_episode(self):
        worker = self.module.AppWorldWorker(2, 50, 20002)
        worker.reset('task-a')
        original = worker.env
        original.execute.side_effect = ResourceTerminal({'reason': 'final_deadline'})
        worker.step('side_effect_then_stall')
        self.module.wait_service.side_effect = None
        self.module.wait_service.return_value = {'port': 21002, 'pid': 2002, 'generation': 1}
        observation, info = worker.reset('task-b')
        original.close.assert_not_called()
        self.module.wait_service.assert_called_with(2, timeout=180, previous_pid=1002)
        self.assertIsNot(worker.env, original)
        self.assertEqual(worker.experiment_name, 'synthetic-bounded-test-w2-r2')
        self.assertEqual(info['task_id'], 'task-b')
        self.assertEqual(info['service_identity']['generation'], 1)
        self.assertEqual(worker.current_step_count, 0)
        self.assertFalse(worker._done)
        worker.step('new-episode-action')
        worker.env.execute.assert_called_once_with('new-episode-action')

    def test_stale_service_after_resource_terminal_is_rejected(self):
        worker = self.module.AppWorldWorker(2, 50, 20002)
        worker.reset('task-a')
        worker.env.execute.side_effect = ResourceTerminal({'reason': 'final_deadline'})
        worker.step('stall')
        with self.assertRaises(InfrastructureFailure):
            worker.reset('task-b')

    def test_bounded_save_logs_failure_aborts_instead_of_returning_healthy_feedback(self):
        wrapper = self.wrapper()
        self.context(wrapper)
        worker = wrapper.workers[4].instance
        worker.env.save_logs.side_effect = ConnectionError('synthetic closed service')
        with self.assertRaises(InfrastructureFailure) as caught:
            wrapper.step(['side-effect-action'] * 8)
        event = caught.exception.event
        self.assertEqual(event['reason'], 'save_logs_failed')
        self.assertEqual(event['execution_feedback'], 'native successful output')
        self.assertEqual(event['pid'], 1004)
        self.assertEqual(event['step_count'], 1)
        self.assertTrue(worker._service_dead)
        worker.env.execute.assert_called_once_with('side-effect-action')
        worker.close()
        worker.env.close.assert_not_called()
        evidence = [json.loads(line) for line in wrapper._evidence_path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(evidence[-1]['event'], 'infrastructure_abort')
        self.assertEqual(evidence[-1]['resource_event'], event)

    def test_unbounded_save_logs_failure_retains_legacy_diagnostic_behavior(self):
        with mock.patch.dict(os.environ, {'GIGPO_RESOURCE_PROFILE': ''}):
            worker = self.module.AppWorldWorker(4, 50, 20004)
            worker.reset('task-a')
            worker.env.save_logs.side_effect = ConnectionError('synthetic closed service')
            output, reward, done, info = worker.step('legacy-action')
            self.assertEqual(output, 'native successful output')
            self.assertFalse(done)
            self.assertEqual(reward, 0)
            self.assertEqual(info['api_trace_unavailable'], 'ConnectionError')
            self.assertFalse(worker._service_dead)

    def test_other_rpc_failures_mark_service_unknown_and_abort_without_replay(self):
        for method in ('execute', 'task_completed', 'evaluate'):
            with self.subTest(method=method):
                worker = self.module.AppWorldWorker(4, 50, 20004)
                worker.reset('task-a')
                if method == 'evaluate':
                    worker.env.task_completed.return_value = True
                getattr(worker.env, method).side_effect = ConnectionError('synthetic RPC failure')
                with self.assertRaises(InfrastructureFailure) as caught:
                    worker.step('one-side-effect-action')
                expected = 'execute_or_native_post_execute' if method == 'execute' else method
                self.assertEqual(caught.exception.event['method'], expected)
                self.assertTrue(worker._service_dead)
                worker.env.execute.assert_called_once_with('one-side-effect-action')
                worker.env.save_logs.assert_not_called()
                worker.close()
                worker.env.close.assert_not_called()

    def test_fixed_service_pid_resolves_original_terminal_after_port_reuse_and_pickle(self):
        worker = self.module.AppWorldWorker(2, 50, 20002)
        worker.reset('task-a')
        code = 'side_effect_then_stall'
        identity = real_resource.request_identity(worker.env, code)
        root = Path(self.temp.name) / 'services'
        event = {'protocol': 'bounded_runtime_v1', 'kind': 'resource_terminal',
                 'reason': 'final_deadline', 'pid': 1002, 'request': identity}
        real_resource.atomic_json(root / 'port-20002.json', {'pid': 2002})
        real_resource.atomic_json(root / 'terminal-1002.json', event)
        # A reused port must not redirect the old client's acknowledgement.
        real_resource.atomic_json(root / 'terminal-2002.json', {
            **event, 'pid': 2002, 'kind': 'infra_unknown', 'reason': 'wrong-service',
        })

        def execute(action):
            try:
                return real_resource.resolve_failed_execute(worker.env, action, ConnectionError('closed'), timeout=0)
            except ResourceTerminal as exc:
                # Exercise actual control exception serialization, as remote
                # failures require, without pretending this is a Ray cluster.
                raise pickle.loads(pickle.dumps(exc))

        worker.env.execute.side_effect = execute
        _, reward, done, info = worker.step(code)
        self.assertTrue(done)
        self.assertEqual(reward, 0)
        self.assertEqual(info['resource_event'], event)
        self.assertEqual(worker.env.resource_service_pid, 1002)

    def test_new_pid_terminal_cannot_cover_missing_old_pid_evidence(self):
        worker = self.module.AppWorldWorker(2, 50, 20002)
        worker.reset('task-a')
        code = 'side_effect_then_disconnect'
        identity = real_resource.request_identity(worker.env, code)
        root = Path(self.temp.name) / 'services'
        real_resource.atomic_json(root / 'port-20002.json', {'pid': 2002})
        real_resource.atomic_json(root / 'terminal-2002.json', {
            'protocol': 'bounded_runtime_v1', 'kind': 'resource_terminal',
            'reason': 'wrong-service', 'pid': 2002, 'request': identity,
        })
        worker.env.execute.side_effect = lambda action: real_resource.resolve_failed_execute(
            worker.env, action, ConnectionError('closed'), timeout=0)
        with self.assertRaises(InfrastructureFailure) as caught:
            worker.step(code)
        restored = pickle.loads(pickle.dumps(caught.exception))
        self.assertEqual(restored.event, caught.exception.event)
        self.assertEqual(restored.event['pid'], 1002)
        self.assertTrue(worker._service_dead)
        worker.env.task_completed.assert_not_called()

    def test_sampler_state_round_trip_reproduces_native_next_task_sample(self):
        wrapper = self.wrapper()
        state = json.loads(json.dumps(wrapper.sampler_state()))
        expected = np.random.choice(wrapper.task_ids, wrapper.env_num, replace=False).tolist()
        np.random.choice(wrapper.task_ids, 3, replace=False)
        wrapper.restore_sampler_state(state)
        actual = np.random.choice(wrapper.task_ids, wrapper.env_num, replace=False).tolist()
        self.assertEqual(actual, expected)
        self.assertEqual(wrapper.reset_batch_serial, state['reset_batch_serial'])
        with self.assertRaises(ValueError):
            wrapper.restore_sampler_state({'schema': 'unknown'})

    def run_native_collector(self, unknown=False):
        wrapper = self.wrapper()
        methods = load_collector_methods()

        class Manager:
            envs = wrapper

            def reset(self, kwargs=None):
                observations, infos = self.envs.reset()
                for slot, actor in enumerate(self.envs.workers):
                    actor.instance.env.task_completed.side_effect = [False, True]
                    if slot == 3:
                        exception = InfrastructureFailure if unknown else ResourceTerminal
                        actor.instance.env.execute.side_effect = exception({'reason': 'synthetic-budget-or-disconnect'})
                return {'text': observations, 'image': None, 'anchor': observations}, infos

            def step(self, raw_actions):
                observations, rewards, dones, infos = self.envs.step(raw_actions)
                for info in infos:
                    info['is_action_valid'] = True
                return {'text': observations, 'image': None, 'anchor': observations}, np.array(rewards), np.array(dones), infos

            def success_evaluator(self, total_infos, total_batch_list, **kwargs):
                successes = []
                for infos, rows in zip(total_infos, total_batch_list):
                    successes.append(next(float(info['won']) for info, row in reversed(list(zip(infos, rows)))
                                          if row['active_masks']))
                return {'success_rate': np.array(successes)}

        class ActorGroup:
            world_size = 1

            def generate_sequences(self, batch):
                return CollectorBatch({'responses': np.ones((8, 1), dtype=np.int64)})

        collector = types.SimpleNamespace(
            config=types.SimpleNamespace(
                env=types.SimpleNamespace(max_steps=4, rollout=types.SimpleNamespace(n=8), env_name='appworld'),
            ),
            tokenizer=types.SimpleNamespace(batch_decode=lambda *args, **kwargs: ['<code>original_action</code>'] * 8),
            preprocess_batch=lambda gen_batch, obs: gen_batch.copy(),
        )
        batch = CollectorBatch(
            {key: np.zeros((8, 1), dtype=np.int64) for key in ('input_ids', 'attention_mask', 'position_ids')},
            {'raw_prompt_ids': np.zeros((8, 1), dtype=object)},
        )
        result = methods['vanilla_multi_turn_loop'](collector, batch, ActorGroup(), Manager())
        rows = methods['gather_rollout_data'](collector, *result)
        return wrapper, result, rows

    def test_actual_collector_filters_inactive_rows_without_losing_group_member(self):
        wrapper, result, rows = self.run_native_collector()
        total, rewards, lengths, success, trajectory_ids, _ = result
        self.assertEqual(len(rows), 15)
        self.assertEqual(len({row['uid'] for row in rows}), 1)
        self.assertEqual(len({row['traj_uid'] for row in rows}), 8)
        self.assertTrue(all(row['active_masks'] for row in rows))
        self.assertEqual([row['active_masks'] for row in total[3]], [True, False])
        self.assertEqual(lengths.tolist(), [2, 2, 2, 1, 2, 2, 2, 2])
        self.assertEqual(rewards.tolist(), [10, 10, 10, 0, 10, 10, 10, 10])
        self.assertEqual(success['success_rate'].tolist(), [1, 1, 1, 0, 1, 1, 1, 1])
        self.assertEqual(wrapper.workers[3].instance.env.execute.call_count, 1)
        evidence = [json.loads(line) for line in wrapper._evidence_path.read_text(encoding='utf-8').splitlines()]
        dispatched = [row for row in evidence if row['event'] == 'step_dispatch']
        self.assertEqual(dispatched[-1]['context']['trajectory_uids'], trajectory_ids.tolist())
        self.assertEqual(dispatched[-1]['context']['active_masks'], [True, True, True, False, True, True, True, True])

    def test_actual_collector_unknown_failure_never_returns_training_rows(self):
        with self.assertRaises(InfrastructureFailure):
            self.run_native_collector(unknown=True)
        paths = list(Path(self.temp.name).glob('appworld_protocol-*.jsonl'))
        rows = [json.loads(line) for line in paths[0].read_text(encoding='utf-8').splitlines()]
        self.assertEqual(rows[-1]['event'], 'infrastructure_abort')
        self.assertEqual(len(set(rows[-1]['context']['trajectory_uids'])), 8)


if __name__ == '__main__':
    unittest.main()
