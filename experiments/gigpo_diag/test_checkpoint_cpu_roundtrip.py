"""Real CPU torch/Adam roundtrip; FSDP context is mocked.

Four-rank CUDA sharding is covered by check_checkpoint_roundtrip.py on HPC.
"""
from contextlib import nullcontext
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from verl.utils.checkpoint import checkpoint_manager as base
from verl.utils.checkpoint import fsdp_checkpoint_manager as implementation


class Metadata:
    def save_pretrained(self, path):
        (Path(path) / "config.json").write_text(json.dumps({"fixture": "cpu-linear"}))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
        self.drop = torch.nn.Dropout(0.2)
        self.config = Metadata()

    def can_generate(self):
        return False

    def forward(self, value):
        return self.linear(self.drop(value))


class RealCpuRoundtrip(unittest.TestCase):
    def test_initialized_adam_scheduler_rng_and_next_update(self):
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(implementation, "get_fsdp_state_ctx", side_effect=lambda *args: nullcontext()), \
             mock.patch.object(implementation, "fsdp_version", return_value=2), \
             mock.patch.object(implementation, "is_cuda_available", False), \
             mock.patch.object(base, "is_cuda_available", False), \
             mock.patch.object(base, "is_npu_available", False), \
             mock.patch("torch.distributed.get_rank", return_value=0), \
             mock.patch("torch.distributed.get_world_size", return_value=1), \
             mock.patch("torch.distributed.barrier"), \
             mock.patch("torch.distributed.all_gather_object", side_effect=lambda rows, value: rows.__setitem__(0, value)):
            torch.manual_seed(42)
            np.random.seed(42)
            random.seed(42)
            model = Model()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            batches = list(torch.arange(48, dtype=torch.float32).reshape(6, 2, 4))
            def update(batch):
                optimizer.zero_grad(set_to_none=True)
                model(batch).square().mean().backward()
                optimizer.step()
                scheduler.step()
            update(batches[0])
            self.assertTrue(optimizer.state)
            manager = implementation.FSDPCheckpointManager(model, optimizer, scheduler, processing_class=Metadata())
            manager.save_checkpoint(directory, global_step=1)
            original_rng = (torch.rand(4), np.random.random(), random.random())
            next_batch = batches[1]
            update(next_batch)
            expected_model = {key: value.detach().clone() for key, value in model.state_dict().items()}
            expected_optimizer = {key: {name: value.clone() if torch.is_tensor(value) else value for name, value in state.items()}
                                  for key, state in optimizer.state_dict()["state"].items()}
            expected_scheduler = scheduler.state_dict()
            manager.load_checkpoint(directory)
            resumed_rng = (torch.rand(4), np.random.random(), random.random())
            self.assertTrue(torch.equal(original_rng[0], resumed_rng[0]))
            self.assertEqual(original_rng[1:], resumed_rng[1:])
            resumed_batch = batches[1]
            update(resumed_batch)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, expected_model[key]), key)
            for key, state in optimizer.state_dict()["state"].items():
                for name, value in state.items():
                    if torch.is_tensor(value):
                        self.assertTrue(torch.equal(value, expected_optimizer[key][name]), (key, name))
                    else:
                        self.assertEqual(value, expected_optimizer[key][name])
            self.assertEqual(scheduler.state_dict(), expected_scheduler)


if __name__ == "__main__":
    unittest.main()
