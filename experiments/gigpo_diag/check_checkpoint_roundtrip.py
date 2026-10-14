"""Actual four-GPU FSDP/Adam checkpoint test, using already downloaded weights.

Run inside a NEW allocation, before services/Ray are started:
  timeout --signal=TERM --kill-after=30s 40m python -m torch.distributed.run \
    --standalone --nproc_per_node=4 -m experiments.gigpo_diag.check_checkpoint_roundtrip \
    --model "$GIGPO_MODEL_PATH" --output "$GIGPO_RUN_DIR/checkpoint-roundtrip"

This tests the production sharded save/load manager and optimizer continuation.
It is not a GiGPO score measurement, nor a proof of the complete job's peak RAM.
No downloads, existing checkpoints, jobs, or training hyperparameters are changed.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.checkpoint.checkpoint_transaction import (
    atomic_text, begin_checkpoint, publish_checkpoint, validate_checkpoint,
)
from verl.utils.fsdp_utils import get_fsdp_wrap_policy


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def tensor_digest(tensor):
    # Bounded host transfers, rather than another full model/optimizer CPU copy.
    digest = hashlib.sha256()
    flat = tensor.detach().reshape(-1)
    for offset in range(0, flat.numel(), 262144):
        digest.update(flat[offset:offset + 262144].contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def training_digest(model, optimizer, scheduler):
    parameters = [tensor_digest(parameter) for parameter in model.parameters()]
    states = []
    for state in optimizer.state.values():
        states.append({key: tensor_digest(value) if torch.is_tensor(value) else value for key, value in state.items()})
    return {"parameters": parameters, "optimizer": states, "scheduler": scheduler.state_dict()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Existing local model directory; never downloaded")
    parser.add_argument("--output", required=True, help="New, empty evidence directory")
    args = parser.parse_args()
    require(Path(args.model).is_dir(), "The existing local model directory is required")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    rank, world_size = dist.get_rank(), dist.get_world_size()
    require(world_size == 4, "This acceptance test requires exactly four GPUs")
    root = Path(args.output).resolve()
    if rank == 0:
        require(not root.exists(), f"Evidence path must be new: {root}")
        root.mkdir(parents=True)
    dist.barrier()
    os.environ["GIGPO_DIAG_DIR"] = str(root / "diagnostics")
    torch.manual_seed(71)
    np.random.seed(71 + rank)
    random.seed(71 + rank)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    module = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.float32, attn_implementation="eager",
    )
    module.config.use_cache = False
    module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = FSDP(module, auto_wrap_policy=get_fsdp_wrap_policy(module), device_id=local_rank,
                 use_orig_params=False, sync_module_states=True, sharding_strategy=ShardingStrategy.FULL_SHARD,
                 mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    manager = FSDPCheckpointManager(model, optimizer, scheduler, processing_class=tokenizer)
    dataset = torch.arange(16 * 32, dtype=torch.long).reshape(16, 32) % min(module.config.vocab_size, 1000)
    loader = StatefulDataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    iterator = iter(loader)

    def update(batch):
        # Actual gradients and Adam moments, not initialization-only serialization.
        optimizer.zero_grad(set_to_none=True)
        tokens = batch.cuda()
        loss = model(input_ids=tokens, labels=tokens).loss
        require(bool(torch.isfinite(loss)), "Non-finite smoke-test loss")
        loss.backward()
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        return float(loss.detach())

    first_loss = update(next(iterator))
    require(bool(optimizer.state), "Optimizer state was not populated by the first update")
    saved_digest = training_digest(model, optimizer, scheduler)
    objects = [None, None]
    if rank == 0:
        stage, final = begin_checkpoint(root, 1)
        objects = [str(stage), str(final)]
    dist.broadcast_object_list(objects)
    stage, final = map(Path, objects)
    manager.save_checkpoint(str(stage / "actor"), global_step=1)
    if rank == 0:
        for name, state in (("data.pt", loader.state_dict()), ("driver_state.pt", {"global_step": 1, "epoch": 0, "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate()})):
            with (stage / name).open("wb") as stream:
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
        atomic_text(stage / "environment_sampler.json", "{}\n")
        publish_checkpoint(stage, final, 1)
    dist.barrier()
    validate_checkpoint(final)
    expected_random = (torch.rand(4), torch.rand(4, device="cuda"), np.random.random(), random.random())
    next_batch = next(iterator)
    second_loss = update(next_batch)
    expected_digest = training_digest(model, optimizer, scheduler)

    # Mutated model and Adam moments are overwritten by the first-update snapshot.
    manager.load_checkpoint(str(final / "actor"))
    require(training_digest(model, optimizer, scheduler) == saved_digest, "Model/optimizer/scheduler state mismatch immediately after restore")
    actual_random = (torch.rand(4), torch.rand(4, device="cuda"), np.random.random(), random.random())
    require(torch.equal(expected_random[0], actual_random[0]) and torch.equal(expected_random[1], actual_random[1])
            and expected_random[2:] == actual_random[2:], "CPU/CUDA/NumPy/Python RNG state mismatch")
    saved_driver_rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    loader.load_state_dict(torch.load(final / "data.pt", weights_only=False))
    resumed_iterator = iter(loader)
    torch.set_rng_state(saved_driver_rng[0])
    np.random.set_state(saved_driver_rng[1])
    random.setstate(saved_driver_rng[2])
    resumed_batch = next(resumed_iterator)
    require(torch.equal(next_batch, resumed_batch), "Dataloader resumed at a different batch")
    resumed_loss = update(resumed_batch)
    require(training_digest(model, optimizer, scheduler) == expected_digest, "Continued optimizer update differs after checkpoint roundtrip")
    row = {"rank": rank, "world_size": world_size, "model": args.model, "checkpoint": str(final),
           "torch": torch.__version__, "first_loss": first_loss, "second_loss": second_loss,
           "resumed_loss": resumed_loss, "optimizer_initialized": True, "model_optimizer_scheduler_match": True,
           "cpu_cuda_numpy_python_rng_match": True, "dataloader_match": True, "next_update_match": True}
    atomic_text(root / f"roundtrip-rank-{rank}.json", json.dumps(row, indent=2) + "\n")
    dist.barrier()
    if rank == 0:
        rows = [json.loads((root / f"roundtrip-rank-{i}.json").read_text()) for i in range(world_size)]
        atomic_text(root / "CHECKPOINT_ROUNDTRIP_PASSED.json", json.dumps({"ranks": rows}, indent=2) + "\n")
        print("CHECKPOINT_ROUNDTRIP_PASSED", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
