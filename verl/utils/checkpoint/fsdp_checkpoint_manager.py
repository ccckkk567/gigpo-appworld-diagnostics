# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import gc
import os
import warnings
from typing import Optional, Union

import torch
import torch.distributed
from accelerate import init_empty_weights
from torch.distributed.fsdp import FullStateDictConfig, ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig, PreTrainedTokenizer, ProcessorMixin

from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import fsdp_version, get_fsdp_state_ctx

from .checkpoint_manager import BaseCheckpointManager
from .checkpoint_memory import record_checkpoint_stage
from .checkpoint_transaction import seal_component, validate_component


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    Manage FSDP checkpointing in SPMD training.

    - Saves/loads per-rank sharded model & optimizer states
    - Persists full lr_scheduler and RNG state
    - Stores HF tokenizer/processor and model/config for unified restore

    Args:
        model (FSDP): Wrapped model instance.
        optimizer (Optimizer): Training optimizer.
        lr_scheduler (LRScheduler): Learning-rate scheduler.
        processing_class (PreTrainedTokenizer or ProcessorMixin, optional):
            Pre-/post-processing artifact handler.
        checkpoint_contents (list[str], optional):
            Components to include; must contain 'model', 'optimizer', 'extra'.
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin] = None,
        checkpoint_contents: Optional[list] = None,
        **kwargs,
    ):
        if checkpoint_contents is None:
            checkpoint_contents = ["model", "optimizer", "extra"]
        if processing_class is None:
            assert "tokenizer" in kwargs, "tokenizer or processor must be provided"
            warnings.warn("`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2)
            processing_class = kwargs.pop("tokenizer")
        assert "model" in checkpoint_contents and "optimizer" in checkpoint_contents and "extra" in checkpoint_contents, f"FSDPCheckpointManager must include ['model', 'optimizer', 'extra'], got {checkpoint_contents}"

        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_contents=checkpoint_contents,
        )

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        """
        Load an FSDP checkpoint for this rank.

        Downloads and loads:
          - model and optimizer shards
          - extra state dict (scheduler + RNG)

        Args:
            local_path: Directory with per-rank checkpoint files.
            hdfs_path: Unused (for API compatibility).
            del_local_after_load: Remove local files after loading.
        """
        if local_path is None:
            return

        # every rank download its own checkpoint
        remote_model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        remote_optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
        remote_extra_state_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
        print(f"[rank-{self.rank}]: Loading from {remote_model_path} and {remote_optim_path} and {remote_extra_state_path}")
        local_model_path = copy_to_local(remote_model_path)
        local_optim_path = copy_to_local(remote_optim_path)
        local_extra_state_path = copy_to_local(remote_extra_state_path)

        self._coordinated_stage("load_validate", lambda: validate_component(local_path, self.world_size))
        record_checkpoint_stage("load_begin", self.rank, local_path)
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            model_state_dict = None
            def read_model():
                nonlocal model_state_dict
                model_state_dict = torch.load(local_model_path, weights_only=False)
            self._coordinated_stage("load_model_read", read_model)
            record_checkpoint_stage("load_model_loaded", self.rank, local_path, model_state_dict)
            self._coordinated_stage("load_model_apply", lambda: self.model.load_state_dict(model_state_dict))
            del model_state_dict
            gc.collect()
            record_checkpoint_stage("load_model_released", self.rank, local_path)
            optimizer_state_dict = None
            def read_optimizer():
                nonlocal optimizer_state_dict
                optimizer_state_dict = torch.load(local_optim_path, weights_only=False)
            self._coordinated_stage("load_optimizer_read", read_optimizer)
            record_checkpoint_stage("load_optimizer_loaded", self.rank, local_path, optimizer_state_dict)
            if self.optimizer is not None:
                self._coordinated_stage("load_optimizer_apply", lambda: self.optimizer.load_state_dict(optimizer_state_dict))
            del optimizer_state_dict
            gc.collect()
            record_checkpoint_stage("load_optimizer_released", self.rank, local_path)
        extra_state_dict = None
        def read_extra():
            nonlocal extra_state_dict
            extra_state_dict = torch.load(local_extra_state_path, weights_only=False)
        self._coordinated_stage("load_extra_read", read_extra)
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(extra_state_dict["lr_scheduler"])
        if "rng" not in extra_state_dict:
            raise ValueError("Complete resume requires per-rank RNG state")
        self.load_rng_state(extra_state_dict["rng"])
        del extra_state_dict
        gc.collect()
        record_checkpoint_stage("load_complete", self.rank, local_path)
        # Never delete a verified complete checkpoint during load. A failed later
        # rank/driver restore must leave the source fully recoverable.
        if del_local_after_load:
            warnings.warn("del_local_after_load ignored for complete checkpoint safety", RuntimeWarning)

    def _save_file(self, value, path):
        # Close and fsync before the component/whole-run completion manifest.
        with open(path, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())

    def _coordinated_stage(self, name, action):
        # All ranks execute the same stage. In particular, model state_dict is
        # never moved into a rank-0-only branch because it can use collectives.
        error = None
        try:
            action()
        except Exception as exc:
            error = f"rank={self.rank} stage={name} {type(exc).__name__}: {exc}"
        errors = [None] * self.world_size
        torch.distributed.all_gather_object(errors, error)
        if any(errors):
            raise RuntimeError("Checkpoint stage failed: " + "; ".join(x for x in errors if x))

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        """
        Save an FSDP checkpoint for this rank.

        Writes:
          - model & optimizer shard files
          - extra state dict (scheduler + RNG)
          - HF tokenizer/processor and model/config on rank 0
          - optional full HF model under 'huggingface/' if requested

        Whole-checkpoint retention is performed by the trainer after publication.

        Args:
            local_path: Target directory for checkpoint files.
            hdfs_path: Unused (for API compatibility).
            global_step: Current training step (used for bookkeeping).
            max_ckpt_to_keep: Number of recent checkpoints to retain.
        """
        if local_path is None:
            return

        # Retention is owned by the trainer after whole-checkpoint publication.
        # No previous recoverable checkpoint is removed on this path.
        if max_ckpt_to_keep is not None:
            warnings.warn("Component retention deferred to atomic trainer publication", RuntimeWarning)
        local_path = self.local_mkdir(local_path)
        torch.distributed.barrier()
        record_checkpoint_stage("save_begin", self.rank, local_path)
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                def save_model():
                    state = self.model.state_dict()
                    record_checkpoint_stage("save_model_extracted", self.rank, local_path, state)
                    self._save_file(state, os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt"))
                    record_checkpoint_stage("save_model_serialized", self.rank, local_path)
                    del state
                    gc.collect()
                    record_checkpoint_stage("save_model_released", self.rank, local_path)
                self._coordinated_stage("model", save_model)
                def save_optimizer():
                    state = self.optimizer.state_dict() if self.optimizer is not None else None
                    record_checkpoint_stage("save_optimizer_extracted", self.rank, local_path, state)
                    self._save_file(state, os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt"))
                    record_checkpoint_stage("save_optimizer_serialized", self.rank, local_path)
                    del state
                    gc.collect()
                    record_checkpoint_stage("save_optimizer_released", self.rank, local_path)
                self._coordinated_stage("optimizer", save_optimizer)
                def save_extra():
                    state = {"lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
                             "rng": self.get_rng_state()}
                    self._save_file(state, os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt"))
                    del state
                    record_checkpoint_stage("save_extra_serialized", self.rank, local_path)
                self._coordinated_stage("extra", save_extra)

        model_config = generation_config = None
        def save_metadata():
            nonlocal model_config, generation_config
            if self.rank != 0:
                return
            if fsdp_version(self.model) == 1:
                unwrap_model = self.model._fsdp_wrapped_module
            else:
                unwrap_model = self.model

            model_config = unwrap_model.config
            if unwrap_model.can_generate() and hasattr(model_config, "name_or_path") and model_config.name_or_path:
                # Some model's name_or_path is empty if not initialized from pretrained,
                # in this cases, we don't save generation config.
                generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
                generation_config.save_pretrained(local_path)
            else:
                generation_config = None

            model_config.save_pretrained(local_path)
            self.processing_class.save_pretrained(local_path)

        self._coordinated_stage("metadata", save_metadata)

        # wait for everyone to dump to local
        torch.distributed.barrier()

        if "hf_model" in self.checkpoint_contents:
            hf_local_path = os.path.join(local_path, "huggingface")
            os.makedirs(hf_local_path, exist_ok=True)

            # Only rank 0 will save hf model and,
            # offload to cpu to save LLMs which may be too large to fit in one GPU
            state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with get_fsdp_state_ctx(self.model, StateDictType.FULL_STATE_DICT, state_dict_config, None):
                state_dict = self.model.state_dict()

            if self.rank == 0:
                if "ForTokenClassification" in model_config.architectures[0]:
                    from transformers import AutoModelForTokenClassification

                    auto_model_cls = AutoModelForTokenClassification
                elif "ForCausalLM" in model_config.architectures[0]:
                    from transformers import AutoModelForCausalLM

                    auto_model_cls = AutoModelForCausalLM
                elif "ForConditionalGeneration" in model_config.architectures[0]:
                    from transformers import AutoModelForVision2Seq

                    auto_model_cls = AutoModelForVision2Seq
                else:
                    raise NotImplementedError(f"Unknown architecture {model_config['architectures']}")

                with init_empty_weights():
                    save_model = auto_model_cls.from_config(model_config, torch_dtype=torch.bfloat16)
                save_model.to_empty(device="cpu")

                if save_model.can_generate():
                    if generation_config is not None:
                        save_model.generation_config = generation_config
                    else:
                        print(f"Warning: {self.__class__.__name__}.save_checkpoint: Generation config file not found in, using a generation config created from the model config when saving hf_model.")

                save_model.save_pretrained(hf_local_path, state_dict=state_dict)
                self.processing_class.save_pretrained(hf_local_path)
                del state_dict
                del save_model

            # wait for rank0 to dump hf_model to local
            torch.distributed.barrier()

        record_checkpoint_stage("save_barrier", self.rank, local_path)
        torch.distributed.barrier()
        self._coordinated_stage("seal", lambda: seal_component(local_path, self.world_size, global_step) if self.rank == 0 else None)
        self.previous_global_step = global_step
        record_checkpoint_stage("save_complete", self.rank, local_path)
