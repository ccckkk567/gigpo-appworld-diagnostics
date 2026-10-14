"""CPU/stdlib tests for publication, retention, and real manager object lifetimes.

The FSDP context/collectives and torch serialization are fixture adapters here;
the separate four-GPU smoke exercises actual torch/FSDP and Adam states.
"""
import ast
from contextlib import nullcontext
import gc
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import tempfile
import types
import unittest
from unittest import mock
import warnings
import weakref

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tx = load_file("checkpoint_tx", "verl/utils/checkpoint/checkpoint_transaction.py")
memory = load_file("checkpoint_memory", "verl/utils/checkpoint/checkpoint_memory.py")


def populate(stage, step=1, world_size=2):
    actor = stage / "actor"
    actor.mkdir()
    for rank in range(world_size):
        for prefix in ("model", "optim", "extra_state"):
            (actor / f"{prefix}_world_size_{world_size}_rank_{rank}.pt").write_bytes(b"serialized-state")
    (actor / "config.json").write_text("{}")
    tx.seal_component(actor, world_size, step)
    for name in ("data.pt", "driver_state.pt"):
        (stage / name).write_bytes(b"driver-state")
    (stage / "environment_sampler.json").write_text("{}")


class Transactions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def complete(self, step):
        stage, final = tx.begin_checkpoint(self.root, step)
        populate(stage, step)
        tx.publish_checkpoint(stage, final, step)
        return final

    def test_publish_then_rotate_keeps_two_complete_and_ignores_partial(self):
        first, second = self.complete(10), self.complete(20)
        pending, _ = tx.begin_checkpoint(self.root, 30)
        populate(pending, 30)
        (pending / "actor/model_world_size_2_rank_1.pt").write_bytes(b"broken")
        with self.assertRaises(ValueError):
            tx.publish_checkpoint(pending, self.root / "global_step_30", 30)
        self.assertEqual((self.root / "latest_checkpointed_iteration.txt").read_text().strip(), "20")
        self.assertTrue(first.exists() and second.exists())
        third = self.complete(40)
        tx.rotate_complete_checkpoints(self.root, 2)
        self.assertFalse(first.exists())
        self.assertTrue(second.exists() and third.exists() and pending.exists())

    def test_missing_data_prevents_publication_and_latest_change(self):
        self.complete(1)
        stage, final = tx.begin_checkpoint(self.root, 2)
        populate(stage, 2)
        (stage / "data.pt").unlink()
        with self.assertRaises(ValueError):
            tx.publish_checkpoint(stage, final, 2)
        self.assertFalse(final.exists())
        self.assertEqual((self.root / "latest_checkpointed_iteration.txt").read_text().strip(), "1")

    def test_publish_crash_before_latest_keeps_old_latest_and_all_complete(self):
        first = self.complete(1)
        stage, final = tx.begin_checkpoint(self.root, 2)
        populate(stage, 2)
        original = tx.atomic_text
        def fail_latest(path, text):
            if Path(path).name == "latest_checkpointed_iteration.txt":
                raise OSError("injected storage error")
            original(path, text)
        with mock.patch.object(tx, "atomic_text", side_effect=fail_latest):
            with self.assertRaises(OSError):
                tx.publish_checkpoint(stage, final, 2)
        self.assertTrue(first.exists())
        self.assertEqual(tx.validate_checkpoint(final)["global_step"], 2)
        self.assertEqual((self.root / "latest_checkpointed_iteration.txt").read_text().strip(), "1")

    def test_no_overwrite_and_wrong_world_size_rejected(self):
        complete = self.complete(1)
        with self.assertRaises(FileExistsError):
            tx.begin_checkpoint(self.root, 1)
        with self.assertRaises(ValueError):
            tx.validate_component(complete / "actor", 4)

    def test_missing_rank_and_manifest_escape_rejected(self):
        stage, _ = tx.begin_checkpoint(self.root, 1)
        populate(stage)
        (stage / "actor/optim_world_size_2_rank_1.pt").unlink()
        with self.assertRaises(ValueError):
            tx.validate_component(stage / "actor")
        with self.assertRaises(ValueError):
            tx._validate_files(stage, {"../outside": 1})


class State(dict):
    pass


class ManagerLifecycle(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "actor"
        self.refs, self.events, self.stages = {}, [], []
        owner = self
        class Base:
            def __init__(self, model, optimizer, lr_scheduler=None, processing_class=None, checkpoint_contents=None):
                self.model, self.optimizer, self.lr_scheduler = model, optimizer, lr_scheduler
                self.processing_class, self.checkpoint_contents = processing_class, checkpoint_contents
                self.rank, self.world_size = 0, 1
            @staticmethod
            def local_mkdir(path):
                Path(path).mkdir(parents=True, exist_ok=True)
                return str(path)
            @staticmethod
            def get_rng_state():
                return {"fixture": 93}
            def load_rng_state(self, value):
                owner.events.append(("rng", value))
        class Component:
            def __init__(self, kind):
                self.kind = kind
                self.config = self
            def state_dict(self):
                if self.kind == "optimizer":
                    owner.assertIsNone(owner.refs["model"]())
                state = State(kind=self.kind, payload=[1, 2, 3])
                owner.refs[self.kind] = weakref.ref(state)
                owner.events.append(("extract", self.kind))
                return state
            def load_state_dict(self, state):
                if self.kind == "optimizer":
                    owner.assertIsNone(owner.refs["loaded_model"]())
                owner.events.append(("restore", self.kind))
                owner.refs["loaded_" + self.kind] = weakref.ref(state)
            def can_generate(self):
                return False
            def save_pretrained(self, path):
                (Path(path) / (self.kind + ".json")).write_text("{}")
        def save(value, stream):
            pickle.dump(value, stream)
        def load(path, **kwargs):
            with open(path, "rb") as stream:
                return pickle.load(stream)
        torch = types.SimpleNamespace(save=save, load=load, distributed=types.SimpleNamespace(
            barrier=lambda: None, all_gather_object=lambda values, value: values.__setitem__(0, value)))
        tree = ast.parse((ROOT / "verl/utils/checkpoint/fsdp_checkpoint_manager.py").read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        tree.body.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
        scope = {"os": os, "gc": gc, "warnings": warnings, "torch": torch, "BaseCheckpointManager": Base,
                 "is_cuda_available": False, "copy_to_local": str, "validate_component": tx.validate_component,
                 "seal_component": tx.seal_component, "fsdp_version": lambda _: 2,
                 "get_fsdp_state_ctx": lambda *args: nullcontext(),
                 "StateDictType": types.SimpleNamespace(SHARDED_STATE_DICT=1),
                 "ShardedStateDictConfig": lambda **kwargs: None, "ShardedOptimStateDictConfig": lambda **kwargs: None,
                 "record_checkpoint_stage": lambda stage, *args: self.stages.append(stage)}
        exec(compile(ast.fix_missing_locations(tree), str(ROOT / "verl/utils/checkpoint/fsdp_checkpoint_manager.py"), "exec"), scope)
        self.manager = scope["FSDPCheckpointManager"](Component("model"), Component("optimizer"), Component("scheduler"), processing_class=Component("processor"))

    def test_save_load_releases_model_before_optimizer_and_restores_extra(self):
        self.manager.save_checkpoint(str(self.path), global_step=10)
        self.assertIsNone(self.refs["model"]())
        self.assertIsNone(self.refs["optimizer"]())
        self.manager.load_checkpoint(str(self.path))
        self.assertIsNone(self.refs["loaded_model"]())
        self.assertIsNone(self.refs["loaded_optimizer"]())
        self.assertIn(("restore", "scheduler"), self.events)
        self.assertIn(("rng", {"fixture": 93}), self.events)
        self.assertLess(self.stages.index("save_model_released"), self.stages.index("save_optimizer_extracted"))
        self.assertLess(self.stages.index("load_model_released"), self.stages.index("load_optimizer_loaded"))

    def test_failed_optimizer_does_not_seal_or_delete_previous(self):
        old = self.path.parent / "previous"
        old.mkdir()
        (old / "keep").write_text("recoverable")
        self.manager.previous_saved_paths = [str(old)]
        original = self.manager._save_file
        def fail_optimizer(value, path):
            if "optim_" in str(path):
                raise OSError("injected ENOSPC")
            return original(value, path)
        self.manager._save_file = fail_optimizer
        with self.assertRaisesRegex(RuntimeError, "optimizer"):
            self.manager.save_checkpoint(str(self.path), global_step=20, max_ckpt_to_keep=1)
        self.assertTrue((old / "keep").exists())
        self.assertFalse((self.path / tx.COMPONENT_MANIFEST).exists())


class DriverTransactions(unittest.TestCase):
    def test_trainer_restores_step_epoch_cursor_sampler_and_retains_two(self):
        class Config(dict):
            __getattr__ = dict.__getitem__
        class Dataloader:
            cursor = 8
            def state_dict(self):
                return {"cursor": self.cursor}
            def load_state_dict(self, state):
                self.cursor = state["cursor"]
        class Sampler:
            serial = 5
            def sampler_state(self):
                return {"serial": self.serial}
            def restore_sampler_state(self, state):
                self.serial = state["serial"]
        class Worker:
            def save_checkpoint(self, path, remote, step, max_ckpt_to_keep=None):
                self.last_keep = max_ckpt_to_keep
                actor = Path(path)
                actor.mkdir()
                for prefix in ("model", "optim", "extra_state"):
                    (actor / f"{prefix}_world_size_1_rank_0.pt").write_bytes(b"state")
                tx.seal_component(actor, 1, step)
            def load_checkpoint(self, path, del_local_after_load=False):
                tx.validate_component(path, 1)
        def load(path, **kwargs):
            with open(path, "rb") as stream:
                return pickle.load(stream)
        rng = {"torch": b"tensor-rng", "numpy": [1, 2, 3]}
        torch = types.SimpleNamespace(save=pickle.dump, load=load, get_rng_state=lambda: rng["torch"],
                                      set_rng_state=lambda state: rng.__setitem__("torch", state))
        numpy = types.SimpleNamespace(random=types.SimpleNamespace(get_state=lambda: rng["numpy"],
                                      set_state=lambda state: rng.__setitem__("numpy", state)))
        tree = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RayPPOTrainer")
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in ("_save_checkpoint", "_load_checkpoint")]
        cls.bases = []
        scope = {"os": os, "json": json, "torch": torch, "np": numpy, "random": random,
                 "diag": types.SimpleNamespace(checkpoint=lambda *args: None),
                 "find_latest_ckpt_path": lambda root: str(Path(root) / ("global_step_" + (Path(root) / "latest_checkpointed_iteration.txt").read_text().strip()))}
        for name in ("atomic_text", "begin_checkpoint", "publish_checkpoint", "rotate_complete_checkpoints", "validate_checkpoint"):
            scope[name] = getattr(tx, name)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), "trainer_checkpoint_methods", "exec"), scope)
        with tempfile.TemporaryDirectory() as directory:
            trainer = scope["RayPPOTrainer"]()
            trainer.config = types.SimpleNamespace(trainer=Config(
                default_hdfs_dir=None, default_local_dir=directory, max_actor_ckpt_to_keep=2,
                resume_mode="auto", del_local_ckpt_after_load=False))
            trainer.use_critic = False
            trainer.actor_rollout_wg = Worker()
            trainer.train_dataloader = Dataloader()
            trainer.envs = types.SimpleNamespace(envs=Sampler())
            trainer.val_envs = types.SimpleNamespace(envs=Sampler())
            trainer._checkpoint_epoch = 3
            for step in (10, 20, 30):
                trainer.global_steps = step
                trainer._save_checkpoint()
            self.assertFalse((Path(directory) / "global_step_10").exists())
            self.assertIsNone(trainer.actor_rollout_wg.last_keep)
            trainer.train_dataloader.cursor = 999
            trainer.envs.envs.serial = 999
            rng.update(torch=b"changed", numpy=[9])
            trainer._load_checkpoint()
            self.assertEqual(trainer.global_steps, 30)
            self.assertEqual(trainer._resume_epoch, 3)
            self.assertEqual(trainer.train_dataloader.cursor, 8)
            self.assertEqual(trainer.envs.envs.serial, 5)
            self.assertEqual(rng, {"torch": b"tensor-rng", "numpy": [1, 2, 3]})


if __name__ == "__main__":
    unittest.main()
