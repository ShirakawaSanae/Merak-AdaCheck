# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: Swli (lucasleesw9@gmail.com), TXacs (txacs1993@gmail.com), Yck(eyichenke@gmail.com)
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
import hashlib
import json
import logging
import math
import os
import socket
import time
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import AdamW, Adam
try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor
from torch.utils.tensorboard import SummaryWriter
from transformers.optimization import Adafactor, get_scheduler
from transformers.trainer_pt_utils import get_parameter_names

from . import get_logger
from .core import PipelineEngine, PipelineModule, mpu
from .core.checkpoint import (
    CheckpointLoader,
    CheckpointSaver,
    rotate_checkpoints,
    RcharaCheckpointLoader,
    RcharaCheckpointSaver,
)
from .core.fx.tracer.utils import _generate_dummy_input
from .core.zero import configure_zero_optimizer
from .merak_args import MerakArguments, manual_set_args, mergeargs
from .utils import BaseParams, MegatronPretrainingRandomSampler, WorkerInitObj


def _validate_moe_adacheck_grouping(*, enabled, zero1_enabled, layout, zero_stage):
    if not enabled:
        return
    if not zero1_enabled or zero_stage != 1:
        raise ValueError("MoE AdaCheck grouping requires UCP ZeRO-1 publication or restore")
    if layout is None or layout.ep <= 1 or layout.dp <= 1 or layout.tp != 1:
        raise ValueError("MoE AdaCheck grouping currently requires EP>1, replica DP>1, and TP1")


def _zero1_tensor_coordinates(module, rank):
    layout = module._merak_ep_layout
    coordinates = layout.coordinates(rank)
    tensor_group = module._grid.get_slice_parallel_group()
    tensor_ranks = tuple(sorted(dist.get_process_group_ranks(tensor_group)))
    expected_ranks = tuple(
        sorted(
            layout.group_ranks(
                "tp",
                pp=coordinates["pp"],
                ep=coordinates["ep"],
                dp=coordinates["dp"],
            )
        )
    )
    if tensor_ranks != expected_ranks:
        raise ValueError("UCP tensor group differs from the MoE layout")
    return coordinates["tp"], layout.tp


def _data_parallel_coordinates(model):
    rank = getattr(model, "_merak_sampler_data_parallel_rank", None)
    size = getattr(model, "_merak_sampler_data_parallel_size", None)
    if rank is not None and size is not None:
        return rank, size
    layout = getattr(model, "_merak_ep_layout", None)
    if layout is None:
        return mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
    coordinates = layout.coordinates(dist.get_rank())
    return coordinates["dp"], layout.dp


class MerakTrainer:
    def __init__(
        self,
        model: nn.Module,
        args: MerakArguments,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        data_collator: Optional[Callable] = None,
        leaf_modules: Tuple[nn.Module] = (),
        post_pipeline_init: Optional[Callable[[nn.Module], None]] = None,
    ):
        """
        MerakTrainer is a simple training and eval loop for PyTorch,
        optimized for Merak.

        Args:
        -   model ([`PreTrainedModel`] or `torch.nn.Module`, *optional*):
                The model to train, evaluate or use for predictions.
                If not provided, a `model_init` must be passed.
        -   args ([`TrainingArguments` and `MerakArguments`], *optional*):
                The arguments to tweak for training. Will default to a basic
                instance of [`TrainingArguments` and `MerakArguments`] with the
                `output_dir` set to a directory named *tmp_trainer* in
                the current directory if not provided.
        -   train_dataset (`torch.utils.data.Dataset` or `torch.utils.data.
                IterableDataset`, *optional*):
                The dataset to use for training. If it is an `datasets.
                Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed.

                Note that if it's a `torch.utils.data.IterableDataset` with
                some randomization and you are training in
                a distributed fashion, your iterable dataset should either use
                a internal attribute `generator` that
                is a `torch.Generator` for the randomization that must be
                identical on all processes (and the Trainer
                will manually set the seed of this `generator` at each epoch)
                or have a `set_epoch()` method that
                internally sets the seed of the RNGs used.
        -   eval_dataset (`torch.utils.data.Dataset`, *optional*):
                The dataset to use for evaluation. If it is an `datasets.
                Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed.
        -   optimizer (`torch.optim.optimizer`, *optional*):
        -   data_collator (`Callable`, *optional*):
                The function to use to form a batch from a list of elements of
                `train_dataset` or `eval_dataset`.
        -   leaf_modules ([`nn.Module`], *optional*):
                List of nn.Module can not be trace, add to leaf modules

        """

        self.model = model
        self.args = args
        if hasattr(model, "_merak_pipeline_input_to_shard") and not self.args.input_names:
            self.args.input_names = ["input_ids"]
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.collate_fn = data_collator
        self.leaf_modules = leaf_modules

        self.optimizer = None
        self.lr_scheduler = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.engine = None
        self.input_to_stage_dic = None
        self._merak_custom_pipeline = hasattr(model, "_merak_pipeline_layers")
        self.dummy_inputs = args.input_names
        self._already_load = False

        self.logger = get_logger("simple")
        if hasattr(self.model, "config"):
            self.model_config = self.model.config
            mergeargs(self.args, self.model_config)
        else:
            mergeargs(self.args, None)
        manual_set_args(self.args)

        self.peft_config = None

        self.tuning_params = BaseParams(self.args)

        if dist.get_rank() == 0:
            self.summary_writer = self.get_summary_writer()
        else:
            self.summary_writer = None

        # init pipeline module
        self.loss_fn = (
            self.get_loss_fn()
            if not self.args.parallel_vocab
            else self.get_vocab_loss_fn()
        )
        phase = getattr(self.model, "_merak_moe_phase_marker", None)
        if phase is not None:
            phase("trainer_pipeline_module_pre")
        self.create_pipeline_module()
        if post_pipeline_init is not None:
            post_pipeline_init(self.model)
        if phase is not None:
            phase("trainer_pipeline_module_post")
        self._configure_checkpoints()

        # init training parameters
        self.create_dataloader()
        if phase is not None:
            phase("trainer_dataloader_post")
        self.tuning_params.train(self.train_dataset)
        if self.eval_dataset is not None:
            self.tuning_params.eval(self.eval_dataset)
        self.get_dummy_inputs()
        if phase is not None:
            phase("trainer_dummy_inputs_post")

    def _configure_checkpoints(self):
        if not self.args.ada_checkpoint:
            self._checkpoint_saver = CheckpointSaver()
            self._checkpoint_loader = CheckpointLoader()
        else:
            self._checkpoint_saver = RcharaCheckpointSaver()
            self._checkpoint_loader = RcharaCheckpointLoader()

        self._save_checkpoint = self._checkpoint_saver.save_checkpoint
        self._rotate_checkpoints = rotate_checkpoints
        self._load_checkpoint = self._checkpoint_loader.load_checkpoint

    def create_pipeline_module(self):
        phase = getattr(self.model, "_merak_moe_phase_marker", None)
        if phase is not None:
            phase("trainer_create_pipeline_pre")
        if self._merak_custom_pipeline or self.args.trace_method in ["fx", "dynamo"]:
            self.model = PipelineModule(
                model=self.model,
                args=self.args,
                loss_fn=self.loss_fn,
                leaf_modules=self.leaf_modules,
                dummy_inputs=self.dummy_inputs,
            )

            self.input_to_stage_dic = self.model.input_to_stage_dic
            if phase is not None:
                phase("trainer_pipeline_module_constructed")
        else:
            dummy_input = _generate_dummy_input(self.args, self.model)
            self.input_to_stage_dic = {0: list(dummy_input.keys())}
            self.args.no_tie_modules = True
            setattr(self.model, "input_to_stage_dic", self.input_to_stage_dic)
            setattr(self.model, "_grid", self.args.get_communication_grid())
            assert (
                mpu.get_pipe_parallel_world_size() * mpu.get_model_parallel_world_size()
                == 1
            )
        dist.barrier()
        if phase is not None:
            phase("trainer_pipeline_barrier_post")

        if mpu.get_data_parallel_rank() == 0 and mpu.get_model_parallel_rank() == 0:
            if hasattr(self.model, "stage_id"):
                for i in range(mpu.get_pipe_parallel_world_size()):
                    if i == self.model.stage_id:
                        self.logger.debug(
                            [dist.get_rank(), self.model.stage_id, self.model]
                        )
            else:
                self.logger.debug([dist.get_rank(), self.model])

    def get_train_sampler(self) -> MegatronPretrainingRandomSampler:
        data_rank, data_size = _data_parallel_coordinates(self.model)
        return MegatronPretrainingRandomSampler(
            total_samples=len(self.train_dataset),
            # Set random seed according to be consumed examples,
            # but currently not supported
            consumed_samples=0,
            micro_batch_size=self.args.per_device_train_batch_size,
            data_parallel_rank=data_rank,
            data_parallel_size=data_size,
        )

    def create_dataloader(self) -> torch.utils.data.DataLoader:
        if self.train_dataset is None and self.eval_dataset is None:
            raise ValueError(
                "MerakTrainer: training requires a train_dataset/eval_dataset."
            )
        if self.train_dataset is not None:
            train_dataset = self.train_dataset
            data_rank, data_size = _data_parallel_coordinates(self.model)

            worker_init = WorkerInitObj(self.args.seed + data_rank)
            train_sampler = self.get_train_sampler()
            train_generator = torch.Generator().manual_seed(
                self.args.seed + data_rank
            )

            self.train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                generator=train_generator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                collate_fn=self.collate_fn,
                worker_init_fn=(
                    worker_init if self.args.dataloader_num_workers > 0 else None
                ),
            )
            if mpu.get_pipe_parallel_rank() not in self.input_to_stage_dic.keys():
                self.train_dataloader = range(len(self.train_dataloader))

        if self.eval_dataset is not None:

            eval_dataset = self.eval_dataset

            self.eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                collate_fn=self.collate_fn,
            )
            if mpu.get_pipe_parallel_rank() not in self.input_to_stage_dic.keys():
                self.eval_dataloader = range(len(self.eval_dataloader))

    def create_optimizer(self, model: PipelineModule) -> torch.optim.Optimizer:

        decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n in decay_parameters and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n not in decay_parameters and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        if self.args.pytorch_tp:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for p in model.parameters()
                        if isinstance(p, DTensor) and p.requires_grad
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for p in model.parameters()
                        if not isinstance(p, DTensor) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]
        optimizer_cls = Adafactor if self.args.adafactor else AdamW
        if self.args.adafactor:
            optimizer_cls = Adafactor
            optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
        else:
            optimizer_cls = torch.optim.AdamW
            optimizer_kwargs = {
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }
            if (
                os.environ.get("CKPT_INCREMENTAL_ENABLE", "0") == "1"
                or os.environ.get("CKPT_OBJECT_SEMANTIC_ENABLE", "0") == "1"
                or os.environ.get("CKPT_OBJECT_RECOVERY_FACTORY") is not None
                or os.environ.get("MERAK_FORCE_SCALAR_ADAMW", "0") == "1"
            ):
                optimizer_kwargs.update(foreach=False, fused=False)
        optimizer_kwargs["lr"] = self.args.learning_rate
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def create_lr_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler:

        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.get_warmup_steps(self.tuning_params.max_steps),
            num_training_steps=self.tuning_params.max_steps,
            scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
        )
        return self.lr_scheduler

    def get_dummy_inputs(self):
        if self._merak_custom_pipeline:
            return
        if self.dummy_inputs is None:
            self.iter_dataloader = iter(
                self.train_dataloader
                if self.train_dataloader is not None
                else self.eval_dataloader
            )
            one_batch = next(self.iter_dataloader)
            try:
                if not isinstance(one_batch[-1][0], torch.Tensor):
                    one_batch = one_batch[:-2]
                else:
                    one_batch.pop(-1)
            except:
                pass
            self.dummy_inputs = one_batch
            del self.iter_dataloader, self.train_dataloader
            gc.collect()
            self.create_dataloader()

    def get_loss_fn(self) -> Callable:
        moe_causal_lm = getattr(self.model_config, "model_type", None) in {
            "qwen2_moe",
            "deepseek_v2",
        }
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        def loss_fn(
            outputs: Union[torch.Tensor, tuple], labels: Union[torch.Tensor, tuple]
        ) -> torch.Tensor:
            if hasattr(outputs, "logits"):
                outputs = outputs.logits
            elif isinstance(outputs, Mapping):
                outputs = outputs["logits"]
            elif isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            if isinstance(labels, Mapping):
                labels = labels["labels"]
            elif isinstance(labels, (tuple, list)):
                labels = labels[0]

            if moe_causal_lm:
                outputs = outputs[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
            return criterion(outputs.reshape(-1, outputs.size(-1)), labels.reshape(-1))

        return loss_fn

    def get_vocab_loss_fn(self) -> Callable:
        def vocab_loss_fn(
            outputs: Union[torch.Tensor, tuple], labels: Union[torch.Tensor, tuple]
        ) -> torch.Tensor:
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if isinstance(labels, tuple):
                labels = labels[0]
            losses = mpu.vocab_parallel_cross_entropy(outputs, labels)
            # loss mask??
            loss = torch.mean(losses.view(-1))
            return loss

        return vocab_loss_fn

    def prepare_data(self, data: Union[tuple, list, dict]) -> Union[tuple, list]:
        if self._merak_custom_pipeline:
            if hasattr(data, "data"):
                data = data.data
            if not isinstance(data, Mapping) or not {"input_ids", "labels"}.issubset(data):
                raise ValueError("MoE pipeline batches require input_ids and labels")
            return data["input_ids"], data["labels"]
        if not isinstance(data, (tuple, list)):
            if hasattr(data, "data"):
                data = data.data
            if isinstance(data, dict):
                if (
                    mpu.get_pipe_parallel_world_size() == 1
                    and mpu.get_model_parallel_world_size() == 1
                    and self.args.trace_method == "None"
                ):
                    return data
                inputs_list = []
                for key, val in self.input_to_stage_dic.items():
                    for d in list(data.keys()):
                        for i in val:
                            if d in i:
                                inputs_list.append(data.pop(d))
                                break
                inputs_list += list(data.values())
                return tuple(inputs_list)
            else:
                raise NotImplementedError("only support data in tuple, list or dict")
        else:
            return data

    def get_summary_writer(self, name: str = "MerakJobName") -> SummaryWriter:
        base_dir = self.args.output_dir
        date_format = "%b%d_%H-%M-%S_"
        date = datetime.now().strftime(date_format)
        hostname = socket.gethostname()
        log_dir = os.path.join(base_dir, "runs", date + hostname)

        os.makedirs(log_dir, exist_ok=True)

        return SummaryWriter(log_dir=log_dir)

    def init_engine(self, eval: bool = False):
        self.engine = PipelineEngine(
            self.model,
            self.args,
            optimizer=self.create_optimizer if not eval else None,
            lr_scheduler=self.create_lr_scheduler if not eval else None,
            tuning_params=self.tuning_params,
            loss_fn=self.loss_fn,
        )

    def train(self):
        self.init_engine()
        self.engine.batch_fn = self.prepare_data
        prepare_remote = getattr(
            self._checkpoint_saver, "prepare_remote_checkpointing", None
        )
        if prepare_remote is not None:
            prepare_remote(self.args)

        object_factory = os.environ.get("CKPT_OBJECT_RECOVERY_FACTORY")
        object_restore_epoch = os.environ.get("CKPT_OBJECT_RESTORE_EPOCH")
        zero1_publish_setting = os.environ.get("CKPT_ZERO1_OBJECT_PUBLISH_STEPS")
        zero1_restore_epoch = os.environ.get("CKPT_ZERO1_OBJECT_RESTORE_EPOCH")
        if zero1_publish_setting is not None and zero1_restore_epoch is not None:
            raise ValueError("UCP ZeRO-1 publication and restore require separate runs")
        zero1_publish_steps = set()
        if zero1_publish_setting is not None:
            try:
                values = [int(value) for value in zero1_publish_setting.split(",")]
            except ValueError as error:
                raise ValueError("CKPT_ZERO1_OBJECT_PUBLISH_STEPS must be comma-separated steps") from error
            if not values or any(value < 1 for value in values) or len(values) != len(set(values)):
                raise ValueError("UCP ZeRO-1 publication steps must be unique positive integers")
            zero1_publish_steps = set(values)
        zero1_enabled = zero1_publish_setting is not None or zero1_restore_epoch is not None
        record_step_metrics = os.environ.get("CKPT_RECORD_STEP_METRICS", "0") == "1"
        object_semantic_setting = os.environ.get("CKPT_OBJECT_SEMANTIC_ENABLE", "0")
        if object_semantic_setting not in {"0", "1"}:
            raise ValueError("CKPT_OBJECT_SEMANTIC_ENABLE must be 0 or 1")
        object_semantic_enabled = object_semantic_setting == "1"
        incremental_enabled = os.environ.get("CKPT_INCREMENTAL_ENABLE", "0") == "1"
        moe_detector_audit_enabled = os.environ.get("CKPT_MOE_ADACHECK_AUDIT", "0")
        if moe_detector_audit_enabled not in {"0", "1"}:
            raise ValueError("CKPT_MOE_ADACHECK_AUDIT must be 0 or 1")
        moe_detector_audit_enabled = moe_detector_audit_enabled == "1"
        moe_detector_grouping = os.environ.get("CKPT_MOE_ADACHECK_GROUPING", "0")
        if moe_detector_grouping not in {"0", "1"}:
            raise ValueError("CKPT_MOE_ADACHECK_GROUPING must be 0 or 1")
        expert_parallel = getattr(self.engine.module, "_merak_ep_layout", None)
        ep_zero1 = (
            expert_parallel is not None
            and expert_parallel.ep > 1
            and getattr(self.args, "zero_stage", None) == 1
        )
        _validate_moe_adacheck_grouping(
            enabled=moe_detector_grouping == "1",
            zero1_enabled=zero1_enabled,
            layout=expert_parallel,
            zero_stage=getattr(self.args, "zero_stage", None),
        )
        if moe_detector_grouping == "1":
            moe_detector_audit_enabled = True
        if zero1_enabled and (
            not bool(getattr(self.engine.module, "_merak_role_aware_zero1", False))
            or self.args.zero_stage != 1
            or self.args.ada_checkpoint
            or self.args.resume_from_checkpoint
            or self.args.save
            or object_factory is not None
            or incremental_enabled
            or (ep_zero1 and object_semantic_enabled)
            or os.environ.get("CKPT_REMOTE_RESTORE_EPOCH") is not None
            or os.environ.get("MERAK_ENABLE_CHECKPOINT_SYNC", "0") == "1"
        ):
            raise ValueError("UCP ZeRO-1 requires a fresh role-aware ZeRO-1 job without other checkpoint modes")
        if (object_factory is None) != (object_restore_epoch is None):
            raise ValueError(
                "object recovery requires CKPT_OBJECT_RECOVERY_FACTORY and "
                "CKPT_OBJECT_RESTORE_EPOCH"
            )
        if object_factory is not None and (
            incremental_enabled
            or object_semantic_enabled
            or os.environ.get("CKPT_INCREMENTAL_RESTORE_EPOCH") is not None
            or os.environ.get("MERAK_ENABLE_CHECKPOINT_SYNC", "0") == "1"
            or os.environ.get("CKPT_REMOTE_RESTORE_EPOCH") is not None
            or self.args.resume_from_checkpoint
        ):
            raise ValueError("object recovery is mutually exclusive with other restore modes")
        if object_semantic_enabled and (
            incremental_enabled
            or os.environ.get("CKPT_INCREMENTAL_RESTORE_EPOCH") is not None
            or os.environ.get("MERAK_ENABLE_CHECKPOINT_SYNC", "0") == "1"
            or os.environ.get("CKPT_REMOTE_RESTORE_EPOCH") is not None
            or self.args.resume_from_checkpoint
        ):
            raise ValueError(
                "SemanticFixed object training is mutually exclusive with other checkpoint modes"
            )

        incremental_controller = None
        if incremental_enabled:
            if os.environ.get("MERAK_ENABLE_CHECKPOINT_SYNC", "0") == "1":
                raise ValueError(
                    "incremental checkpointing and CheckpointSync are mutually exclusive"
                )
            from .core.checkpoint.merak_incremental import (
                build_merak_incremental_controller,
            )

            incremental_controller = build_merak_incremental_controller(
                self.engine.module,
                self.engine.optimizer,
                self.engine.lr_scheduler,
                args=self.args,
                sampler=getattr(self.train_dataloader, "batch_sampler", None),
                epoch_provider=lambda: self.tuning_params.global_steps,
            )
        elif object_semantic_enabled:
            from .core.checkpoint.object_semantic_training import (
                build_semantic_fixed_training_controller,
            )

            incremental_controller = build_semantic_fixed_training_controller(
                self.engine.module,
                self.engine.optimizer,
                self.engine.lr_scheduler,
                args=self.args,
                sampler=getattr(self.train_dataloader, "batch_sampler", None),
                epoch_provider=lambda: self.tuning_params.global_steps,
            )

        object_controller = None
        if object_factory is not None:
            from .core.checkpoint.object_recovery_controller import (
                load_object_recovery_controller,
            )

            object_controller = load_object_recovery_controller(
                object_factory,
                module=self.engine.module,
                optimizer=self.engine.optimizer,
                scheduler=self.engine.lr_scheduler,
                sampler=getattr(self.train_dataloader, "batch_sampler", None),
                args=self.args,
            )

        zero1_controller = None
        zero1_clients = None
        zero1_rank_hosts = None
        zero1_rank_metadata = None
        zero1_capture_rank = None
        zero1_publish_distributed = None
        zero1_write_locator = None
        zero1_locator = None
        zero1_completed_steps = set()
        zero1_group = None
        zero1_tensor_coordinates = None
        zero1_last_restore_seconds = None
        if zero1_enabled:
            from .core.checkpoint.object_semantic_runtime import create_world_gloo_control_group
            from .core.checkpoint.replica_service import ReplicaServiceClient
            from .core.checkpoint.zero1_recovery_controller import (
                build_zero1_object_restore_controller,
                capture_zero1_rank,
                zero1_rank_metadata,
            )
            from .core.checkpoint.zero1_service import (
                publish_distributed_zero1_epoch,
                write_zero1_locator,
            )
            if moe_detector_audit_enabled and not ep_zero1:
                raise ValueError("MoE AdaCheck capture requires EP ZeRO-1")

            if not hasattr(self.engine.module, "_merak_ep_layout"):
                raise ValueError("UCP ZeRO-1 requires the validated expert-parallel layout")
            if os.environ.get("CKPT_ZERO1_ARM", "adacheck_fixed") not in {
                "adacheck_fixed", "host_global_fixed", "m1"
            }:
                raise ValueError("CKPT_ZERO1_ARM must select a supported placement arm")
            try:
                endpoints = json.loads(os.environ["CKPT_REPLICA_ENDPOINTS"])
                zero1_locator = Path(os.environ["CKPT_ZERO1_LOCATOR"])
                node_rank = int(os.environ["NODE_RANK"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("UCP ZeRO-1 endpoint, locator, or NODE_RANK is missing") from error
            if (
                not isinstance(endpoints, list) or len(endpoints) != 2
                or any(not isinstance(item, str) or not item for item in endpoints)
                or node_rank not in (0, 1)
                or dist.get_world_size() != self.engine.module._merak_ep_layout.world_size
            ):
                raise ValueError("UCP ZeRO-1 requires two services and a matching two-host EP layout")
            ep_coordinates = self.engine.module._merak_ep_layout.coordinates(dist.get_rank())
            zero1_tensor_coordinates = (
                ep_coordinates["tp"], self.engine.module._merak_ep_layout.tp
            )
            if _zero1_tensor_coordinates(self.engine.module, dist.get_rank()) != (
                *zero1_tensor_coordinates,
            ):
                raise ValueError("UCP tensor coordinates differ from the MoE layout")
            for parameter in self.engine.module.parameters():
                partitioned = hasattr(parameter, "partition_dim") or getattr(
                    parameter, "partition_stride", None
                ) is not None or isinstance(
                    getattr(parameter, "__dict__", {}).get("stride"), int
                )
                if zero1_tensor_coordinates[1] == 1 and partitioned:
                    raise ValueError("TP1 UCP parameter carries tensor-partition metadata")
                if zero1_tensor_coordinates[1] > 1 and partitioned and not bool(
                    getattr(parameter, "model_parallel", False)
                ):
                    raise ValueError("TP-partitioned UCP parameter lacks its model-parallel marker")
            zero1_group = getattr(
                self.engine.module, "_merak_zero1_control_group", None
            )
            if zero1_group is None:
                timeout_seconds = os.environ.get("CKPT_MOE_CONTROL_TIMEOUT_SECONDS")
                if timeout_seconds is None:
                    zero1_group = create_world_gloo_control_group()
                else:
                    zero1_group = dist.new_group(
                        ranks=list(range(dist.get_world_size())),
                        backend="gloo",
                        timeout=timedelta(seconds=float(timeout_seconds)),
                    )
            if dist.get_backend(zero1_group) != "gloo" or (
                dist.get_world_size(zero1_group) != dist.get_world_size()
            ):
                raise ValueError("UCP ZeRO-1 control group must span the Gloo world")
            if os.environ.get("CKPT_MOE_TRACE_PHASES") == "1":
                self.logger.info(
                    f"MOE_PHASE rank={dist.get_rank()} phase=ucp_control_group_ready",
                    ranks=[dist.get_rank()],
                )
            host_rows = [None] * dist.get_world_size()
            dist.all_gather_object(host_rows, node_rank, group=zero1_group)
            zero1_rank_hosts = dict(enumerate(host_rows))
            if set(host_rows) != {0, 1} or any(host_rows.count(host) != dist.get_world_size() // 2 for host in (0, 1)):
                raise ValueError("UCP ZeRO-1 requires equal rank counts on NODE_RANK 0 and 1")
            timeout = float(os.environ.get("CKPT_OBJECT_SERVICE_TIMEOUT_SECONDS", "600"))
            surviving_host = (
                int(os.environ["CKPT_OBJECT_SURVIVING_HOST"])
                if zero1_restore_epoch is not None else None
            )
            zero1_clients = {
                host: ReplicaServiceClient.from_endpoint(endpoints[host], timeout_seconds=timeout)
                for host in ((surviving_host,) if surviving_host is not None else (0, 1))
            }
            zero1_capture_rank = capture_zero1_rank
            zero1_publish_distributed = publish_distributed_zero1_epoch
            zero1_write_locator = write_zero1_locator
            zero1_controller = build_zero1_object_restore_controller(
                self.engine.module,
                self.engine.optimizer,
                self.engine.lr_scheduler,
                sampler=getattr(self.train_dataloader, "batch_sampler", None),
                args=self.args,
                control_group=zero1_group,
                endpoints=endpoints,
                rank_hosts=zero1_rank_hosts,
            ) if zero1_restore_epoch is not None else None
            restore_aliases = (
                zero1_controller.model_aliases if zero1_controller is not None else {}
            )
            zero1_rank_metadata = zero1_rank_metadata(
                self.engine.module,
                self.engine.optimizer,
                rank=dist.get_rank(),
                host=node_rank,
                layout=self.engine.module._merak_ep_layout,
                model_aliases=restore_aliases,
            )

        recover = None
        if os.environ.get("MERAK_ENABLE_CHECKPOINT_SYNC", "0") == "1":
            from process_monitor.procguard.dist_monitor import get_monitor
            from .CheckpointSync.optim import OptimizerWrapper
            from .CheckpointSync.recover import Recover

            monitor = get_monitor()
            recover = Recover(monitor, self.engine.module, self.engine.optimizer, self.logger)
            self.engine.optimizer = OptimizerWrapper(recover, optim=self.engine.optimizer)
            self.logger.warning("CheckpointSync recovery is enabled by MERAK_ENABLE_CHECKPOINT_SYNC=1")

        incremental_restore_epoch = os.environ.get("CKPT_INCREMENTAL_RESTORE_EPOCH")
        remote_restore_epoch = os.environ.get("CKPT_REMOTE_RESTORE_EPOCH")
        if incremental_restore_epoch is not None and incremental_controller is None:
            raise ValueError("incremental restore requires CKPT_INCREMENTAL_ENABLE=1")
        if incremental_restore_epoch is not None and remote_restore_epoch is not None:
            raise ValueError("only one remote restore mode may be selected")
        if remote_restore_epoch is not None and self.args.resume_from_checkpoint:
            raise ValueError(
                "CKPT_REMOTE_RESTORE_EPOCH and resume_from_checkpoint are mutually exclusive"
            )
        if object_controller is not None:
            iteration = object_controller.restore(int(object_restore_epoch))
            epochs_trained = self.tuning_params.resume(iteration)
            dist.barrier()
            self.logger.warning(f"Resumed from object checkpoint epoch {iteration}")
        elif zero1_controller is not None:
            restore_started = time.perf_counter()
            iteration = zero1_controller.restore(int(zero1_restore_epoch))
            zero1_last_restore_seconds = time.perf_counter() - restore_started
            epochs_trained = self.tuning_params.resume(iteration)
            dist.barrier()
            self.logger.warning(f"Resumed from UCP ZeRO-1 object epoch {iteration}")
            if dist.get_rank() == 0:
                metrics_path = Path(self.args.output_dir) / "zero1-object-metrics.jsonl"
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                with metrics_path.open("a") as stream:
                    stream.write(json.dumps({
                        "event": "fresh_restore",
                        "epoch": iteration,
                        "restore_seconds": zero1_last_restore_seconds,
                        "scope": "restore controller plus service reads, state apply and runtime restore; excludes process startup",
                    }, sort_keys=True) + "\n")
        elif incremental_restore_epoch is not None:
            if self.args.resume_from_checkpoint:
                raise ValueError(
                    "CKPT_INCREMENTAL_RESTORE_EPOCH and resume_from_checkpoint are mutually exclusive"
                )
            iteration = incremental_controller.restore(
                base_epoch=int(os.environ.get("CKPT_INCREMENTAL_BASE_EPOCH", "0")),
                final_epoch=int(incremental_restore_epoch),
            )
            epochs_trained = self.tuning_params.resume(iteration)
            self.logger.warning(
                f"Resumed from incremental checkpoint epoch {iteration}"
            )
        elif remote_restore_epoch is not None:
            iteration = self.restore_remote_checkpoint(int(remote_restore_epoch))
            epochs_trained = self.tuning_params.resume(iteration)
            if hasattr(self.train_dataloader, "batch_sampler"):
                self.train_dataloader.batch_sampler.consumed_samples = (
                    iteration * self.tuning_params.total_batch_size
                )
            self.logger.warning(
                f"Resumed from committed remote checkpoint epoch {iteration}"
            )
        elif self.args.resume_from_checkpoint:
            iteration = self.load_checkpoint(self.args.resume_from_checkpoint)
            epochs_trained = self.tuning_params.resume(iteration)
            if hasattr(self.train_dataloader, "batch_sampler"):
                self.train_dataloader.batch_sampler.consumed_samples = (
                    iteration * self.tuning_params.total_batch_size
                )
        elif zero1_restore_epoch is not None:
            raise RuntimeError("UCP ZeRO-1 restore controller was not initialized")
        else:
            iteration = 0
            epochs_trained = 0

        if self.args.lora_config:
            self.peft_config = self.engine.module._add_lora_layers(self.model_config)
            self.engine._configure_optimizer(force=True)
            _ = self.load_checkpoint(
                self.args.resume_from_checkpoint, peft_config=self.peft_config
            )

        if incremental_controller is not None and incremental_restore_epoch is None:
            incremental_controller.start(iteration)
            dist.barrier()

        self.tuning_params.logging()

        for epoch in range(epochs_trained, math.ceil(self.args.num_train_epochs)):
            if dist.get_rank() == 0:
                self.logger.info(
                    f"\nEpoch: {epoch+1}/{math.ceil(self.args.num_train_epochs)}",
                    ranks=[0],
                )
            # if hasattr(self.train_dataloader.batch_sampler, 'set_epoch'):
            #     self.train_dataloader.batch_sampler.set_epoch(epoch)
            for step in range(self.tuning_params.num_steps_per_epoch):
                step_wall_started = time.perf_counter()
                self.tuning_params.step = step + 1
                self.tuning_params.global_steps += 1
                checkpoint_exposed_started = None

                if recover is not None:
                    recover.start_recover()
                train_compute_started_ns = time.monotonic_ns()
                loss = self.engine.train_batch(self.train_dataloader)
                train_compute_finished_ns = time.monotonic_ns()
                step_metrics = {"step": self.tuning_params.global_steps}
                if zero1_enabled:
                    optimizer = self.engine.optimizer
                    if optimizer.global_step != self.tuning_params.global_steps:
                        raise RuntimeError("UCP ZeRO-1 capture is not at an optimizer-step boundary")
                    should_publish = torch.tensor(
                        int(self.tuning_params.global_steps in zero1_publish_steps),
                        dtype=torch.int32,
                        device=self.engine.device,
                    )
                    dist.all_reduce(should_publish, op=dist.ReduceOp.MIN)
                    if should_publish.item():
                        checkpoint_exposed_started = time.perf_counter()
                        capture_started = time.perf_counter()
                        sampler = getattr(self.train_dataloader, "batch_sampler", None)
                        detector_audit = None
                        model_aliases = {}
                        if moe_detector_audit_enabled:
                            from .core.moe.semantic_adapter import (
                                capture_moe_adacheck_groups,
                                moe_detector_audit_report,
                            )

                            detector_groups = capture_moe_adacheck_groups(
                                self.engine.module,
                                host_by_rank=zero1_rank_hosts,
                            )
                            detector_audit = moe_detector_audit_report(detector_groups)
                            if moe_detector_grouping == "1":
                                model_aliases = detector_groups.model_aliases().get(
                                    dist.get_rank(), {}
                                )
                        snapshot = list(zero1_capture_rank(
                            self.engine.module,
                            optimizer,
                            self.engine.lr_scheduler,
                            rank=dist.get_rank(),
                            host=zero1_rank_hosts[dist.get_rank()],
                            epoch=optimizer.global_step,
                            sampler=sampler,
                            storage_hosts=(0, 1),
                            tensor_parallel_rank=zero1_tensor_coordinates[0],
                            tensor_parallel_world_size=zero1_tensor_coordinates[1],
                            detector_audit=detector_audit,
                            model_aliases=model_aliases,
                        ))
                        capture_seconds = time.perf_counter() - capture_started
                        publish_started = time.perf_counter()
                        from .core.checkpoint.zero1_recovery_controller import zero1_rank_metadata as build_rank_metadata
                        rank_metadata = build_rank_metadata(
                            self.engine.module,
                            optimizer,
                            rank=dist.get_rank(),
                            host=zero1_rank_hosts[dist.get_rank()],
                            layout=self.engine.module._merak_ep_layout,
                            model_aliases=model_aliases,
                        )
                        sealed = zero1_publish_distributed(
                            zero1_clients,
                            epoch=optimizer.global_step,
                            captured_snapshot=snapshot,
                            rank_hosts=zero1_rank_hosts,
                            arm=os.environ.get("CKPT_ZERO1_ARM", "adacheck_fixed"),
                            group=zero1_group,
                            rank_metadata=rank_metadata,
                            detector_audit=detector_audit,
                            model_aliases=model_aliases,
                        )
                        observation = getattr(sealed, "publication_observation", None)
                        if observation is None:
                            raise RuntimeError("UCP ZeRO-1 publication metrics are missing")
                        publish_seconds = time.perf_counter() - publish_started
                        step_metrics["committed_checkpoint_epoch"] = sealed.epoch
                        step_metrics["checkpoint_capture_seconds"] = capture_seconds
                        step_metrics["checkpoint_publish_seconds"] = publish_seconds
                        if dist.get_rank() == 0:
                            zero1_write_locator(zero1_locator, sealed)
                        dist.barrier(group=zero1_group)
                        step_metrics["checkpoint_exposed_stall_seconds"] = (
                            time.perf_counter() - checkpoint_exposed_started
                        )
                        zero1_completed_steps.add(sealed.epoch)
                        if dist.get_rank() == 0:
                            metrics_path = Path(self.args.output_dir) / "zero1-object-metrics.jsonl"
                            metrics_path.parent.mkdir(parents=True, exist_ok=True)
                            with metrics_path.open("a") as stream:
                                stream.write(json.dumps({
                                    "epoch": sealed.epoch,
                                    "arm": sealed.arm,
                                    "capture_seconds": capture_seconds,
                                    "publish_seconds": publish_seconds,
                                    "remote_payload_bytes": sealed.remote_payload_bytes,
                                    "remote_wire_bytes": observation.remote_wire_bytes,
                                    "remote_logical_bytes": observation.remote_logical_bytes,
                                    "remote_edge_payload_bytes": observation.remote_edge_payload_bytes,
                                    "remote_control_payload_bytes": observation.remote_control_payload_bytes,
                                    "remote_control_wire_bytes": observation.remote_control_wire_bytes,
                                    "remote_control_logical_bytes": observation.remote_control_logical_bytes,
                                    "remote_data_logical_bytes": observation.remote_logical_bytes - observation.remote_control_logical_bytes,
                                    "maximum_remote_put_ms": observation.maximum_remote_put_ms,
                                    "maximum_remote_data_put_to_commit_ack_ms": observation.maximum_remote_edge_ack_ms,
                                    "control_checksum": sealed.control_checksum,
                                    "detector_audit": sealed.detector_audit,
                                    "model_aliases_applied": sealed.model_aliases_applied,
                                    "placement_policy_scope": (
                                        "canonical model fragments follow per-host placement; optimizer and runtime roots remain rank-local"
                                        if any(sealed.model_aliases)
                                        else "complete rank-local UCP placement; detector audit is recorded but not consumed by placement"
                                        if sealed.detector_audit is not None
                                        else "complete rank-local UCP placement"
                                    ),
                                    "transfer_size_scope": "cross-host serialized UCP object snapshot; full state bundles and control, not incremental gradient checkpointing",
                                    "max_edge_scope": "maximum client-observed remote data-bundle PUT-through-commit-ACK interval; control excluded; not pure data-plane time or physical NIC time",
                                    **({"committed_inventory": observation.committed_inventory}
                                       if observation.committed_inventory is not None else {}),
                                }, sort_keys=True) + "\n")
                            self.logger.info(
                                f"UCP ZeRO-1 epoch {sealed.epoch} sealed: "
                                f"remote_bytes={sealed.remote_payload_bytes} "
                                f"max_remote_put_ms={sealed.maximum_remote_put_ms:.3f}"
                            )
                if incremental_controller is not None and incremental_restore_epoch is None:
                    incremental_controller.record_profile_interval(
                        "training_compute",
                        train_compute_started_ns,
                        train_compute_finished_ns,
                        epoch=self.tuning_params.global_steps,
                    )
                    incremental_controller.poll(self.tuning_params.global_steps)
                step_wall_time = time.perf_counter() - step_wall_started
                if zero1_enabled or record_step_metrics:
                    step_metrics["step_wall_seconds"] = step_wall_time
                    step_metrics_path = (
                        Path(self.args.output_dir) / "zero1-step-metrics"
                        / f"rank_{dist.get_rank():05d}.jsonl"
                    )
                    step_metrics_path.parent.mkdir(parents=True, exist_ok=True)
                    with step_metrics_path.open("a") as stream:
                        stream.write(json.dumps(step_metrics, sort_keys=True) + "\n")
                if (
                    dist.get_rank() == 0
                    and self.tuning_params.global_steps % self.args.logging_steps == 0
                ):
                    self.logger.info(
                        f"step wall: {self.tuning_params.global_steps} "
                        f"train-plus-checkpoint-poll time (s): {step_wall_time:0.3f}",
                        ranks=[0],
                    )

                if self.summary_writer is not None:
                    self.summary_events = [
                        (
                            f"Train/loss",
                            loss.mean().item(),
                            self.tuning_params.global_steps,
                        ),
                        (
                            f"Train/learning_rate",
                            self.engine.optimizer.param_groups[0]["lr"].real,
                            self.tuning_params.global_steps,
                        ),
                    ]
                    for event in self.summary_events:  # write_summary_events
                        self.summary_writer.add_scalar(event[0], event[1], event[2])
                if (
                    self.args.output_dir
                    and self.args.save
                    and self.tuning_params.global_steps % self.args.save_steps == 0
                ):
                    self.save_checkpoint()
                    self._rotate_checkpoints(self.args, self.args.output_dir)

                if self.tuning_params.should_break():
                    break

            if self.args.output_dir and self.args.save:
                self.save_checkpoint()
                self._rotate_checkpoints(self.args, self.args.output_dir)
            if self.args.do_eval:
                self.evaluation()

        finalize_remote = getattr(
            self._checkpoint_saver, "finalize_remote_checkpointing", None
        )
        if finalize_remote is not None:
            finalize_remote()
        if incremental_controller is not None and incremental_restore_epoch is None:
            summary = incremental_controller.finalize(
                self.tuning_params.global_steps
            )
            recovery_valid = torch.tensor(
                int(summary.get("recovery_valid", summary["valid"])),
                device=torch.cuda.current_device(),
            )
            cadence_valid = torch.tensor(
                int(summary.get("cadence_valid", summary["valid"])),
                device=torch.cuda.current_device(),
            )
            dist.all_reduce(recovery_valid, op=dist.ReduceOp.MIN)
            dist.all_reduce(cadence_valid, op=dist.ReduceOp.MIN)
            strict_incremental = (
                os.environ.get("CKPT_INCREMENTAL_REQUIRE_1C1S", "1") == "1"
            )
            if not recovery_valid.item():
                raise RuntimeError(
                    "Merak checkpoint recovery contract was not satisfied"
                )
            if not cadence_valid.item() and strict_incremental:
                raise RuntimeError(
                    "Merak checkpoint lag contract was not satisfied"
                )
        if zero1_publish_steps != zero1_completed_steps:
            missing = sorted(zero1_publish_steps - zero1_completed_steps)
            raise RuntimeError(f"UCP ZeRO-1 publication steps were not sealed: {missing}")

    def evaluation(self):
        assert (
            self.eval_dataloader is not None
        ), "The eval_dataloader is None, Please check eval_dataset"
        if self.engine is None:
            self.init_engine(eval=True)

        if self.args.resume_from_checkpoint and not self._already_load:
            iteration = self.load_checkpoint(self.args.resume_from_checkpoint)

        # eval_iterator = iter(eval_dataloader)
        self.logger.info("\nStart Evaluation...", ranks=[0])
        self.tuning_params.logging(eval=True)
        for step in range(self.tuning_params.eval_steps):
            self.tuning_params.eval_step += 1
            self.tuning_params.global_eval_steps += 1

            loss, logits, labels = self.engine.eval_batch(
                batch_fn=self.prepare_data, data_iter=self.eval_dataloader
            )

        if self.summary_writer is not None:
            self.summary_events = [
                (
                    f"Eval/loss",
                    loss.mean().item(),
                    self.tuning_params.global_eval_steps,
                ),
            ]
            for event in self.summary_events:  # write_summary_events
                self.summary_writer.add_scalar(event[0], event[1], event[2])
        dist.barrier()

    def save_checkpoint(self, best_model: Any = None):
        kwargs = {
            "best_model": best_model,
            "args": self.args,
            "peft_config": self.peft_config,
        }
        self._save_checkpoint(
            self.tuning_params.global_steps,
            self.engine.module,
            self.engine.optimizer,
            self.engine.lr_scheduler,
            **kwargs,
        )

    def restore_remote_checkpoint(self, epoch: int) -> int:
        if not self.args.ada_checkpoint:
            raise RuntimeError("remote restore requires ada_checkpoint")
        self._checkpoint_saver.args = self.args
        state_dict = self._checkpoint_saver.restore_remote_checkpoint(epoch)
        return self._checkpoint_loader.load_remote_state(
            state_dict,
            self.engine.module,
            self.engine.optimizer,
            self.engine.lr_scheduler,
            self.args,
            verbose=True,
        )

    def load_checkpoint(
        self, resume_from_checkpoint: str, peft_config: Optional[Callable] = None
    ) -> int:
        if os.path.exists(resume_from_checkpoint):
            load_result, state_dict, opt_state_dict = self._load_checkpoint(
                self.engine.module,
                self.engine.optimizer,
                self.engine.lr_scheduler,
                self.args,
                peft_config,
                verbose=True,
            )

            if os.path.isfile(self.args.zero_reshard):
                # load consolidate opt state dict
                opt_state_dict = torch.load(
                    self.args.zero_reshard, map_location="cpu", weights_only=False
                )
                opt_state_dict["param_groups"] = self.engine.optimizer.param_groups
                self.engine.optimizer.load_state_dict(opt_state_dict)

                # reshard zero optimizer with new dp axies
                dynamic_loss_args = (
                    self.engine.amp_engine.dynamic_loss_args()
                    if self.args.fp16 or self.args.bf16
                    else None
                )
                self.engine.optimizer = configure_zero_optimizer(
                    self.engine.optimizer, self.args, dynamic_loss_args
                )

                # load loss scaler
                self.engine.optimizer.load_state_dict(opt_state_dict)

            del state_dict, opt_state_dict
            self._already_load = True
            return load_result

        else:
            raise ValueError(
                f"Cannot find checkpoint files in {self.args.resume_from_checkpoint}"
            )
