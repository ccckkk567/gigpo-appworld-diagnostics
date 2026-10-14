# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import numpy as np
import ray
import time
from pathlib import Path

from appworld import load_task_ids
from experiments.gigpo_diag.appworld_client import AppWorldClient, EXECUTE_HTTP_GRACE_SECONDS
from experiments.gigpo_diag.diagnostics import plain
from experiments.gigpo_diag.appworld_resource import (
    InfrastructureFailure, ResourceTerminal, wait_service,
)


RESOURCE_PROTOCOL = "bounded_runtime_v1"


def bounded_runtime_enabled():
    return bool(os.environ.get("GIGPO_RESOURCE_PROFILE"))

def load_available_ports(port_file="appworld_ports.ports"):
    """
    Load available port list from file
    """
    port_file = os.environ.get('APPWORLD_PORTS_FILE', port_file)
    if not os.path.exists(port_file):
        raise FileNotFoundError(f"Port file {port_file} does not exist. Please run the service startup script first.")
    
    ports = []
    with open(port_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line.isdigit():
                ports.append(int(line))
    
    if not ports:
        raise ValueError(f"No valid ports found in port file {port_file}.")
    
    return ports

class AppWorldWorker:
    """
    Ray Actor that holds an instance of AppWorld and operates the environment
    based on method calls from the main process.
    """
    def __init__(self, worker_id, max_interactions, port):
        self.env = None
        self.current_step_count = 0
        self.reset_serial = 0
        self.max_interactions = max_interactions
        self.worker_id = worker_id
        self.service_pid = None
        self.service_generation = None
        self._service_dead = False
        self._done = False
        self._terminal_observation = None
        self._terminal_info = None
        
        self.url = f"http://127.0.0.1:{port}"

    def reset(self, task_id):
        """Reset the environment with a new task."""
        if self.env is not None and not self._service_dead:
            self.env.close()
            time.sleep(2)

        if bounded_runtime_enabled():
            endpoint = wait_service(
                self.worker_id, timeout=180,
                previous_pid=self.service_pid if self._service_dead else None,
            )
            if self._service_dead and endpoint['pid'] == self.service_pid:
                raise InfrastructureFailure({
                    "reason": "supervisor_returned_terminated_service",
                    "worker_id": self.worker_id, "pid": self.service_pid,
                })
            self.url = f"http://127.0.0.1:{endpoint['port']}"
            self.service_pid = endpoint['pid']
            self.service_generation = endpoint['generation']

        self.current_step_count = 0
        self._done = False
        self._terminal_observation = None
        self._terminal_info = None

        self.reset_serial += 1
        self.task_id = task_id
        self.experiment_name = f"{os.environ['GIGPO_RUN_ID']}-w{self.worker_id}-r{self.reset_serial}"
        self.env = AppWorldClient(
            task_id=task_id,
            experiment_name=self.experiment_name,
            remote_environment_url=self.url,
            max_interactions=self.max_interactions,
        )
        self.env.resource_worker_id = self.worker_id
        self.env.resource_service_pid = self.service_pid
        self.env.service_pid = self.service_pid
        self.env.service_generation = self.service_generation
        self._service_dead = False

        obs = self.env.task.instruction
        info = {
            "task_id": task_id,
            "service_identity": {"url": self.url, "experiment": self.experiment_name, "worker": self.worker_id,
                                 "pid": self.service_pid, "generation": self.service_generation},
            "code_timeout_seconds": self.env.timeout_seconds,
            "execute_http_timeout_seconds": (self.env.timeout_seconds + EXECUTE_HTTP_GRACE_SECONDS
                                             if self.env.timeout_seconds is not None else None),
            "initial_state_hash": None,
            "initialization_semantics": "native fresh task databases and default deterministic AppWorld seed; instruction consistency checked",
            "supervisor": dict(self.env.task.supervisor),
            "resource_protocol": RESOURCE_PROTOCOL if bounded_runtime_enabled() else None,
        }
        return obs, info

    def step(self, action):
        """Execute one step in the environment."""
        if self.env is None:
            raise RuntimeError("Environment not reset before step. Please call reset() first.")
        if self._done:
            # The collector still polls all slots. Its existing active mask
            # excludes these rows; never re-execute or pay terminal reward twice.
            return self._terminal_observation, 0.0, True, {**self._terminal_info, "inactive": True}

        self.current_step_count += 1

        try:
            obs = self.env.execute(action)
        except ResourceTerminal as exc:
            if not bounded_runtime_enabled():
                raise
            self._service_dead = True
            reason = exc.event.get('reason', 'declared_resource_budget_exceeded')
            obs = f"Episode terminated by {RESOURCE_PROTOCOL}: {reason}. The action was not retried."
            info = {
                "won": False, "budgeted_task_success": False,
                "native_evaluator_success": None, "step_count": self.current_step_count,
                "task_id": self.task_id, "protocol": RESOURCE_PROTOCOL,
                "termination": "resource_terminal", "resource_terminal": True,
                "resource_event": exc.event, "action_retried": False,
                "service_identity": {"url": self.url, "experiment": self.experiment_name,
                                     "worker": self.worker_id, "pid": self.service_pid,
                                     "generation": self.service_generation},
            }
            self._done = True
            self._terminal_observation, self._terminal_info = obs, info
            # A killed service cannot honestly supply evaluator or log results.
            return obs, 0.0, True, info
        except InfrastructureFailure:
            # Never recycle an unknown trajectory as ordinary failure reward.
            self._service_dead = True
            raise
        except Exception as exc:
            if bounded_runtime_enabled():
                # Native AppWorld.execute also saves logs after its execute
                # RPC. A failure there has no resource-terminal evidence.
                self._raise_infrastructure_failure('execute_or_native_post_execute', exc)
            raise

        try:
            done = self.env.task_completed() or (self.current_step_count >= self.max_interactions)
        except Exception as exc:
            if bounded_runtime_enabled():
                self._raise_infrastructure_failure('task_completed', exc, obs)
            raise

        if done:
            try:
                is_success = self.env.evaluate().success
            except Exception as exc:
                if bounded_runtime_enabled():
                    self._raise_infrastructure_failure('evaluate', exc, obs)
                raise

            reward = 10.0 if is_success else 0.0
            info = {"won": is_success, "step_count": self.current_step_count}
        else:
            reward = 0.0
            info = {"won": False, "step_count": self.current_step_count}

        info['native_evaluator_success'] = bool(is_success) if done else None
        info['budgeted_task_success'] = bool(info['won'])
        info['resource_terminal'] = False
        info['protocol'] = RESOURCE_PROTOCOL if bounded_runtime_enabled() else None

        # Native persistent API log, kept under this unique episode experiment.
        # save_logs does not execute policy actions or change evaluator/reward.
        info['task_id'] = self.task_id
        info['termination'] = 'task_success' if info['won'] else ('budget' if self.current_step_count >= self.max_interactions else 'complete_task')
        info['api_trace_ref'] = os.path.join(os.environ['APPWORLD_ROOT'], 'experiments', 'outputs', self.experiment_name, 'tasks', self.task_id, 'logs', 'api_calls.jsonl')
        try:
            self.env.save_logs()
        except Exception as exc:
            if bounded_runtime_enabled():
                self._raise_infrastructure_failure('save_logs', exc, obs)
            info['api_trace_unavailable'] = type(exc).__name__
        if done:
            self._done = True
            self._terminal_observation, self._terminal_info = obs, info
        return obs, reward, done, info

    def _raise_infrastructure_failure(self, method, exc, execution_feedback=None):
        self._service_dead = True
        raise InfrastructureFailure({
            "kind": "infra_unknown", "protocol": RESOURCE_PROTOCOL,
            "reason": f"{method}_failed", "method": method,
            "error_type": type(exc).__name__, "error": str(exc),
            "worker_id": self.worker_id, "pid": self.service_pid,
            "generation": self.service_generation, "task_id": self.task_id,
            "experiment_name": self.experiment_name,
            "step_count": self.current_step_count,
            "execution_feedback": execution_feedback, "action_retried": False,
        }) from exc

    def close(self):
        """Close the environment."""
        if self.env is not None and not self._service_dead:
            self.env.close()


class AppWorldEnvs:
    """
    A Ray-based distributed wrapper for AppWorld.
    - Creates multiple Ray actors, each holding a separate AppWorld instance.
    - Implements Gym-style interfaces such as step() / reset() / close().
    """
    def __init__(self, 
                 dataset_name,
                 max_interactions,
                 seed,
                 env_num,
                 group_n,
                 start_server_id,
                 resources_per_worker,
                 port_file="appworld_ports.ports"
                 ):
        super().__init__()

        self.dataset_name = dataset_name
        self.max_interactions = max_interactions
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self.start_server_id = start_server_id
        self.reset_batch_serial = 0
        self._aborted = False
        self._rollout_context = None
        self.current_task_ids = None
        self._evidence_path = None
        if bounded_runtime_enabled():
            directory = os.environ.get('GIGPO_DIAG_DIR')
            if not directory:
                directory = str(Path(os.environ['GIGPO_RUN_DIR']) / 'diagnostics')
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            self._evidence_path = directory / f'appworld_protocol-{dataset_name}-{start_server_id}.pid-{os.getpid()}.jsonl'
        self.task_ids = load_task_ids(dataset_name)
   
        if self.env_num > len(self.task_ids):
            raise ValueError(f"Env_num ({self.env_num}) exceeds available task_ids in '{self.dataset_name}' ({len(self.task_ids)}). Please reducing env_num to {len(self.task_ids)}.")
            
        all_ports = load_available_ports(port_file)

        self.available_ports = all_ports[start_server_id:start_server_id + self.num_processes]

        # Check if we have enough ports
        if len(self.available_ports) < self.num_processes:
            raise ValueError(
                f"Need {self.num_processes} ports, but only {len(self.available_ports)} available ports. "
                f"Please ensure enough service instances are started."
            )

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        # Create Ray actors (workers)
        env_worker = ray.remote(**resources_per_worker)(AppWorldWorker)
        self.workers = []
        for i in range(self.num_processes):
            port = self.available_ports[i]
            worker = env_worker.remote(
                worker_id=start_server_id + i,
                max_interactions=self.max_interactions,
                port=port
            )
            self.workers.append(worker)

    def _record_protocol(self, event, **payload):
        """Required evidence, not best-effort diagnostics, for bounded runs."""
        if self._evidence_path is None:
            return
        row = {
            "schema": RESOURCE_PROTOCOL, "event": event, "time_ns": time.time_ns(),
            "run_id": os.environ.get('GIGPO_RUN_ID'), "dataset": self.dataset_name,
            "reset_batch_serial": self.reset_batch_serial, "group_n": self.group_n,
            "worker_slots": list(range(self.start_server_id, self.start_server_id + self.num_processes)),
            "task_ids": self.current_task_ids,
            **payload,
        }
        with self._evidence_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(plain(row), ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            if event == 'infrastructure_abort':
                os.fsync(stream.fileno())

    def set_rollout_context(self, groups, trajectories, step_index, raw_actions, active_masks):
        """Receive identity before dispatch; do not alter the collector masks."""
        if not bounded_runtime_enabled():
            return
        if self._aborted:
            raise RuntimeError('AppWorld rollout batch aborted; refusing replay or resampling')
        if any(len(items) != self.num_processes for items in (groups, trajectories, raw_actions, active_masks)):
            raise ValueError('Rollout identity length must match all worker slots')
        self._rollout_context = {
            "episode_group_uids": [str(x) for x in groups],
            "trajectory_uids": [str(x) for x in trajectories],
            "step_index": int(step_index), "raw_actions": list(raw_actions),
            "active_masks": [bool(x) for x in active_masks],
        }

    def _abort_batch(self, phase, exc):
        self._aborted = True
        self._record_protocol(
            'infrastructure_abort', phase=phase, context=self._rollout_context,
            error_type=type(exc).__name__, error=str(exc),
            resource_event=getattr(exc, 'event', None),
            decision='abort_entire_rollout_before_advantages',
            current_dispatch_feedback='unknown_for_uncollected_results',
            retry=False, resample=False,
        )

    def sampler_state(self):
        """Snapshot native global NumPy sampling; no extra RNG draws."""
        algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
        return {
            "schema": "appworld-numpy-global-v1",
            "reset_batch_serial": self.reset_batch_serial,
            "rng_state": {"bit_generator": algorithm, "keys": keys.tolist(),
                          "position": int(position), "has_gauss": int(has_gauss),
                          "cached_gaussian": float(cached_gaussian)},
        }

    def restore_sampler_state(self, state):
        if state.get('schema') != 'appworld-numpy-global-v1':
            raise ValueError('Unsupported AppWorld sampler state schema')
        rng = state['rng_state']
        np.random.set_state((rng['bit_generator'], np.asarray(rng['keys'], dtype=np.uint32),
                             int(rng['position']), int(rng['has_gauss']), float(rng['cached_gaussian'])))
        self.reset_batch_serial = int(state['reset_batch_serial'])

    def step(self, actions):
        """
        actions: Must be a list with length equal to self.num_processes, 
        each sent to the corresponding worker.
        
        Return format follows Gym's step() convention:
            observations, rewards, dones, infos
        """
        assert len(actions) == self.num_processes, "The length of actions must match the number of processes."
        if self._aborted:
            raise RuntimeError('AppWorld rollout batch aborted; refusing replay or resampling')
        self._record_protocol('step_dispatch', context=self._rollout_context, executed_actions=list(actions))

        # Send step commands to all workers
        futures = []
        try:
            for i, worker in enumerate(self.workers):
                future = worker.step.remote(actions[i])
                futures.append(future)
            # Any unknown failure aborts the entire batch before it can return
            # samples to the collector/advantage path. No half-group fallback.
            results = ray.get(futures)
        except Exception as exc:
            self._abort_batch('step', exc)
            raise
        self._record_protocol('step_results', context=self._rollout_context, results=results)
        
        obs_list = []
        reward_list = []
        done_list = []
        info_list = []

        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        """
        Reset all worker environments simultaneously, 
        returning each environment's initial observation and info.
        """
        if self._aborted:
            raise RuntimeError('AppWorld rollout batch aborted; refusing replay or resampling')
        self.reset_batch_serial += 1
        self._rollout_context = None
        # randomly select self.env_num task_id from self.task_ids
        task_id = np.random.choice(self.task_ids, self.env_num, replace=False)
        # repeat task_id group_n times
        task_id = np.repeat(task_id, self.group_n).tolist()
        self.current_task_ids = task_id
        self._record_protocol('reset_dispatch')

        # Send reset commands to all workers
        futures = []
        try:
            for i, worker in enumerate(self.workers):
                future = worker.reset.remote(task_id[i])
                futures.append(future)
            results = ray.get(futures)
        except Exception as exc:
            self._abort_batch('reset', exc)
            raise
        self._record_protocol('reset_results', results=results)
        
        obs_list = []
        info_list = []

        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    def close(self):
        """Close all workers."""
        # Send close commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.close.remote()
            futures.append(future)
        
        # Wait for all workers to close
        ray.get(futures)
        
        # Shutdown Ray actors
        for worker in self.workers:
            ray.kill(worker)

    def render(self):
        """Implement this if visualization is needed."""
        pass

def build_appworld_envs(dataset_name="train",
                        max_interactions=50,
                        seed=0,
                        env_num=1, 
                        group_n=1,
                        start_server_id=0,
                        resources_per_worker={"num_cpus": 0.1},
                        ):

    return AppWorldEnvs(
        dataset_name=dataset_name,
        max_interactions=max_interactions,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        start_server_id=start_server_id,
        resources_per_worker=resources_per_worker
    )
