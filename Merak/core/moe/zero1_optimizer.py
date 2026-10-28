"""Opt-in role-aware ZeRO-1 for expert-parallel models."""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
import torch.distributed as dist

from Merak import get_logger
from Merak.core.zero.stage1 import ZERO_SUPPORTED_OPTIMIZERS, ZeroOptimizer_Stage1
from Merak.core.zero.utils import DynamicLossScaler, LossScaler, get_group_alignment_padding


def _is_expert(parameter: torch.Tensor) -> bool:
    marked = bool(getattr(parameter, "_merak_expert_parallel", False))
    if marked != hasattr(parameter, "_merak_expert_id"):
        raise ValueError("expert parameters need both Merak EP markers")
    return marked


class RoleAwareZeroOptimizerStage1(ZeroOptimizer_Stage1):
    """ZeRO-1 with shared/router and expert parameters on distinct groups.

    Shared and router parameters shard over ``shared_process_group`` (EP x DP).
    Expert parameters shard only over ``expert_process_group`` (true DP peers
    with the same EP coordinate and therefore the same expert IDs).
    """

    def __init__(
        self,
        args: Any,
        init_optimizer: torch.optim.Optimizer,
        *,
        model: torch.nn.Module,
        shared_process_group: object,
        expert_process_group: object,
        static_loss_scale: float = 1.0,
        dynamic_loss_scale: bool = False,
        dynamic_loss_args: dict[str, Any] | None = None,
        allgather_size: int = 500_000_000,
        clip_grad: float = 0.0,
        global_process_group: object | None = None,
        max_elements_per_comm: int = 500_000_000,
        postscale_gradients: bool = True,
        gradient_predivide_factor: float = 1.0,
        gradient_average: bool = True,
        communication_data_type: torch.dtype = torch.float16,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("role-aware ZeRO-1 requires torch.distributed")
        if type(init_optimizer) not in ZERO_SUPPORTED_OPTIMIZERS:
            raise TypeError("role-aware ZeRO-1 requires a supported fresh optimizer")
        if init_optimizer.state:
            raise ValueError("role-aware ZeRO-1 must wrap a fresh optimizer")

        self.args = args
        self.optimizer = init_optimizer
        self.global_step = 0
        self.flatten = torch._utils._flatten_dense_tensors
        self.unflatten = torch._utils._unflatten_dense_tensors
        self.clip_grad = float(clip_grad)
        self.verbose = False
        self.logger = get_logger("detailed")
        self.postscale_gradients = postscale_gradients
        self.gradient_predivide_factor = gradient_predivide_factor
        self.gradient_average = gradient_average
        self.communication_data_type = communication_data_type
        self.allgather_size = allgather_size
        self.max_elements_per_comm = max_elements_per_comm
        self.is_gradient_accumulation_boundary = True
        self.dp_process_group = shared_process_group
        self.global_process_group = (
            dist.group.WORLD if global_process_group is None else global_process_group
        )

        original_groups = list(init_optimizer.param_groups)
        trainable_parameters = [
            parameter.device
            for group in original_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError("role-aware ZeRO-1 needs trainable parameters on this stage")
        parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
        if any(id(parameter) not in parameter_names for group in original_groups for parameter in group["params"]):
            raise ValueError("role-aware ZeRO-1 parameter lacks a canonical model name")
        physical_groups = []
        self.dp_process_groups = []
        self.group_roles = []
        for source_index, group in enumerate(original_groups):
            options = {key: value for key, value in group.items() if key != "params"}
            for role, process_group in (
                ("shared", shared_process_group),
                ("expert", expert_process_group),
            ):
                params = [parameter for parameter in group["params"] if _is_expert(parameter) == (role == "expert")]
                if not params:
                    continue
                physical_groups.append(
                    {
                        **options,
                        "params": params,
                        "_merak_zero1_role": role,
                        "_merak_source_group": source_index,
                    }
                )
                self.dp_process_groups.append(process_group)
                self.group_roles.append(role)
        init_optimizer.param_groups = physical_groups

        self.fp16_groups = []
        self.fp16_groups_flat = []
        self.parallel_sub_partitioned_fp16_groups = []
        self.parallel_comm_sub_partitioned_fp16_groups = []
        self.local_sub_partitions_of_fp32_groups = []
        self.params_not_local = []
        self.params_in_rank_sub_partitions = []
        self.params_in_rank_sub_partitions_offsets = []
        self.sub_partition_sizes = []
        self.num_comm_intervals_per_group = []
        self.group_paddings = []
        self.partition_counts = []
        self.local_ranks = []
        self.max_elems_per_comm = []
        self.zero1_shard_ranges = []
        self._reduced_gradients = None
        self._pre_optimizer_step_callback: Callable[
            [int, torch.optim.Optimizer, float], None
        ] | None = None
        self.default_device = trainable_parameters[0]

        for group_index, (param_group, process_group) in enumerate(
            zip(physical_groups, self.dp_process_groups)
        ):
            params = param_group["params"]
            self._validate_replicas(
                params, process_group, self.group_roles[group_index], parameter_names
            )
            world_size = dist.get_world_size(group=process_group)
            local_rank = dist.get_rank(group=process_group)
            self.partition_counts.append(world_size)
            self.local_ranks.append(local_rank)
            self.fp16_groups.append(params)
            max_elements = self.best_max_elems_per_comm(
                sum(parameter.numel() for parameter in params),
                int(max_elements_per_comm),
                world_size,
                self.logger,
            )
            self.max_elems_per_comm.append(max_elements)
            flat = self.flatten_dense_tensors_sub_partition_aligned(
                params, world_size, max_elements, process_group
            )
            self.fp16_groups_flat.append(flat)
            for parameter, view in zip(params, self.unflatten(flat, params)):
                parameter.data = view.data
            comm, by_rank, intervals, size, count = self.get_data_parallel_sub_partitions(
                flat, max_elements, world_size, process_group
            )
            self.parallel_comm_sub_partitioned_fp16_groups.append(comm)
            self.parallel_sub_partitioned_fp16_groups.append(by_rank)
            self.sub_partition_sizes.append(size)
            self.num_comm_intervals_per_group.append(count)
            parameter_ranges = []
            parameter_start = 0
            for parameter in params:
                parameter_ranges.append((parameter, parameter_start, parameter.numel()))
                parameter_start += parameter.numel()
            shard_ranges = []
            for chunk_index, interval in enumerate(intervals[local_rank]):
                chunk_start, chunk_stop = interval
                cursor = 0
                chunk_ranges = []
                for parameter, parameter_start, parameter_size in parameter_ranges:
                    start = max(chunk_start, parameter_start)
                    stop = min(chunk_stop, parameter_start + parameter_size)
                    if start < stop:
                        length = stop - start
                        chunk_ranges.append((parameter, start - parameter_start, length, cursor))
                        cursor += length
                shard_ranges.append(tuple(chunk_ranges))
            self.zero1_shard_ranges.append(tuple(shard_ranges))
            masters = []
            for partition in by_rank[local_rank]:
                master = partition.clone().float().detach()
                master.requires_grad = True
                masters.append(master)
            self.local_sub_partitions_of_fp32_groups.append(masters)
            self.group_paddings.append(
                get_group_alignment_padding(params, size, count * world_size)
            )
            param_group["params"] = masters
            local, offsets, remote = self.get_all_sub_partition_info(
                params, intervals, local_rank, world_size
            )
            self.params_in_rank_sub_partitions.append(local)
            self.params_in_rank_sub_partitions_offsets.append(offsets)
            self.params_not_local.append(remote)

        if dynamic_loss_scale:
            values = dynamic_loss_args or {}
            self.loss_scaler = DynamicLossScaler(
                init_scale=values.get("INITIAL_LOSS_SCALE", 2**32),
                scale_window=values.get("SCALE_WINDOW", 1000),
                min_scale=values.get("MIN_LOSS_SCALE", 1),
            )
            self.dynamic_loss_scale = True
        else:
            self.loss_scaler = LossScaler(scale=static_loss_scale)
            self.dynamic_loss_scale = False
            self.cur_iter = 0
        self.overflow = False

    def backward(self, loss, retain_graph=False):
        self.loss_scaler.backward(loss.float(), retain_graph=retain_graph)

    @staticmethod
    def _validate_replicas(params, process_group, role: str, parameter_names) -> None:
        local = tuple(
            (
                parameter_names.get(id(parameter)),
                int(getattr(parameter, "_merak_expert_id", -1)),
                tuple(parameter.shape),
                str(parameter.dtype),
                bool(parameter.requires_grad),
            )
            for parameter in params
        )
        gathered = [None] * dist.get_world_size(group=process_group)
        dist.all_gather_object(gathered, local, group=process_group)
        if any(value != local for value in gathered):
            raise ValueError(f"{role} ZeRO-1 peers do not own matching parameters")

    @staticmethod
    def _reduce_scatter(output, inputs, process_group) -> None:
        if str(dist.get_backend(process_group)).lower() == "gloo":
            merged = torch.cat(inputs)
            dist.all_reduce(merged, group=process_group)
            rank = dist.get_rank(group=process_group)
            output.copy_(merged.narrow(0, rank * output.numel(), output.numel()))
        else:
            dist.reduce_scatter(output, inputs, group=process_group)

    def reduce_gradients(self, pipeline_parallel: bool = False) -> None:
        del pipeline_parallel
        reduced_groups = []
        for group_index in range(len(self.fp16_groups)):
            process_group = self.dp_process_groups[group_index]
            world_size = self.partition_counts[group_index]
            local_rank = self.local_ranks[group_index]
            all_partitions = []
            for rank in range(world_size):
                partitions, _ = self.get_flat_sub_partitions(
                    self.params_in_rank_sub_partitions[group_index][rank],
                    self.params_in_rank_sub_partitions_offsets[group_index][rank],
                    self.sub_partition_sizes[group_index],
                    self.communication_data_type,
                    self.default_device,
                    self.num_comm_intervals_per_group[group_index],
                )
                all_partitions.append(partitions)
            reduced = []
            for comm_index in range(self.num_comm_intervals_per_group[group_index]):
                inputs = [all_partitions[rank][comm_index] for rank in range(world_size)]
                if self.postscale_gradients and self.gradient_predivide_factor != 1.0:
                    for partition in inputs:
                        partition.mul_(1.0 / self.gradient_predivide_factor)
                elif not self.postscale_gradients:
                    for partition in inputs:
                        partition.div_(world_size)
                self._reduce_scatter(inputs[local_rank], inputs, process_group)
                if self.postscale_gradients and self.gradient_average:
                    inputs[local_rank].mul_(self.gradient_predivide_factor / world_size)
                reduced.append(inputs[local_rank])
            reduced_groups.append(reduced)
        self._reduced_gradients = reduced_groups

    def _has_overflow(self) -> bool:
        overflow = any(
            not bool(torch.isfinite(gradient).all().item())
            for group in self._reduced_gradients
            for gradient in group
        )
        value = torch.tensor([overflow], dtype=torch.uint8, device=self.default_device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX, group=self.global_process_group)
        return bool(value.item())

    def set_pre_optimizer_step_callback(
        self,
        callback: Callable[[int, torch.optim.Optimizer, float], None] | None,
    ) -> None:
        """Observe reduced local ZeRO shards before the AdamW update."""
        if callback is not None and not callable(callback):
            raise TypeError("pre-optimizer-step callback must be callable")
        self._pre_optimizer_step_callback = callback

    def step(self, closure=None):
        if closure is not None:
            raise ValueError("role-aware ZeRO-1 does not support closures")
        if self._reduced_gradients is None:
            raise RuntimeError("reduce_gradients() must run before step()")
        self.global_step += 1
        backward_scale = float(self.loss_scale)
        self.overflow = self._has_overflow()
        self.loss_scaler.update_scale(self.overflow)
        if self.overflow:
            self.zero_grad()
            self._reduced_gradients = None
            return True

        local_gradients = []
        for group_index in range(len(self.fp16_groups)):
            if not self.fp16_groups[group_index]:
                continue
            local_rank = self.local_ranks[group_index]
            self.free_grad_in_param_list(self.params_not_local[group_index])
            gradients = [
                gradient.float() for gradient in self._reduced_gradients[group_index]
            ]
            for master, gradient in zip(
                self.local_sub_partitions_of_fp32_groups[group_index], gradients
            ):
                master.grad = gradient
            self.free_grad_in_param_list(
                self.params_in_rank_sub_partitions[group_index][local_rank]
            )
            local_gradients.extend(gradients)

        squared = torch.zeros(1, dtype=torch.float32, device=self.default_device)
        for gradient in local_gradients:
            squared.add_(gradient.float().square().sum())
        dist.all_reduce(squared, group=self.global_process_group)
        total_norm = math.sqrt(squared.item())
        combined_scale = backward_scale
        if self.clip_grad > 0:
            clip = ((total_norm / combined_scale) + 1e-6) / self.clip_grad
            if clip > 1:
                combined_scale *= clip
        for gradient in local_gradients:
            gradient.mul_(1.0 / combined_scale)

        if self._pre_optimizer_step_callback is not None:
            self._pre_optimizer_step_callback(
                self.global_step, self.optimizer, float(1.0 / combined_scale)
            )

        self.optimizer.step()
        for group_index, (partitions, masters, process_group) in enumerate(
            zip(
                self.parallel_sub_partitioned_fp16_groups,
                self.local_sub_partitions_of_fp32_groups,
                self.dp_process_groups,
            )
        ):
            if not self.fp16_groups[group_index]:
                continue
            local_rank = self.local_ranks[group_index]
            for destination, source in zip(partitions[local_rank], masters):
                destination.copy_(source.data)
            for sub_partitions in self.parallel_comm_sub_partitioned_fp16_groups[group_index]:
                dist.all_gather(sub_partitions, sub_partitions[local_rank], group=process_group)
            for parameter, view in zip(
                self.fp16_groups[group_index],
                self.unflatten(self.fp16_groups_flat[group_index], self.fp16_groups[group_index]),
            ):
                parameter.data = view.data
        for group in self.local_sub_partitions_of_fp32_groups:
            for parameter in group:
                parameter.grad = None
        self._reduced_gradients = None
        return False

    def state_dict(self):
        state = self._rigid_state_dict()
        state["group_roles"] = tuple(self.group_roles)
        return state

    @property
    def step_count(self):
        return self.global_step

    def load_state_dict(self, state_dict, load_optimizer_states=True, **kwargs):
        if kwargs:
            raise TypeError(f"unsupported ZeRO-1 load options: {sorted(kwargs)}")
        expected = {
            "global_step",
            "loss_scaler",
            "dynamic_loss_scale",
            "overflow",
            "base_optimizer_state",
            "local_sub_partitions_of_fp32_groups",
            "group_roles",
        }
        if self.args.ada_checkpoint and state_dict.get("global_step") in self.args.check_steps:
            expected.update({"params_in_partition", "first_offset", "partition_size"})
        if set(state_dict) != expected or tuple(state_dict["group_roles"]) != tuple(
            self.group_roles
        ):
            raise ValueError("role-aware ZeRO-1 checkpoint layout differs")
        if len(state_dict["local_sub_partitions_of_fp32_groups"]) != len(
            self.local_sub_partitions_of_fp32_groups
        ):
            raise ValueError("role-aware ZeRO-1 local group count differs")
        for saved_group, live_group in zip(
            state_dict["local_sub_partitions_of_fp32_groups"],
            self.local_sub_partitions_of_fp32_groups,
        ):
            if len(saved_group) != len(live_group):
                raise ValueError("role-aware ZeRO-1 local chunk count differs")
            for saved, live in zip(saved_group, live_group):
                if saved.shape != live.shape or saved.dtype != live.dtype:
                    raise ValueError("role-aware ZeRO-1 local chunk shape differs")
        saved_groups = state_dict["base_optimizer_state"].get("param_groups", ())
        if len(saved_groups) != len(self.optimizer.param_groups) or any(
            any(
                key in saved and saved[key] != live[key]
                for key in ("_merak_zero1_role", "_merak_source_group", "weight_decay", "betas", "eps")
            )
            for saved, live in zip(saved_groups, self.optimizer.param_groups)
        ):
            raise ValueError("role-aware ZeRO-1 optimizer group configuration differs")
        if type(state_dict["global_step"]) is not int or state_dict["global_step"] < 0:
            raise ValueError("role-aware ZeRO-1 global step is invalid")
        self._rigid_load_state_dict(state_dict, load_optimizer_states=load_optimizer_states)

    def zero_grad(self, set_grads_to_None=True):
        super().zero_grad(set_grads_to_None=set_grads_to_None)

    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

    def _get_loss_scale(self):
        return self.loss_scaler.loss_scale

    def _set_loss_scale(self, value):
        self.loss_scaler.cur_scale = value

    loss_scale = property(_get_loss_scale, _set_loss_scale)
    cur_scale = property(_get_loss_scale, _set_loss_scale)


def configure_role_aware_zero1(
    optimizer: torch.optim.Optimizer,
    args: Any,
    *,
    model: torch.nn.Module,
    shared_process_group: object,
        expert_process_group: object,
        global_process_group: object | None = None,
        dynamic_loss_scale_args: dict[str, Any] | None = None,
    static_loss_scale: float = 1.0,
    dynamic_loss_scale: bool = False,
) -> RoleAwareZeroOptimizerStage1:
    """Build the opt-in EP-aware ZeRO-1 wrapper from Merak arguments."""
    return RoleAwareZeroOptimizerStage1(
        args,
        optimizer,
        model=model,
        shared_process_group=shared_process_group,
        expert_process_group=expert_process_group,
        global_process_group=global_process_group,
        static_loss_scale=static_loss_scale,
        dynamic_loss_scale=dynamic_loss_scale,
        dynamic_loss_args=dynamic_loss_scale_args,
        clip_grad=args.max_grad_norm,
        allgather_size=int(args.zero_allgather_bucket_size),
        max_elements_per_comm=int(args.zero_reduce_bucket_size),
        postscale_gradients=not args.prescale_gradients,
        gradient_predivide_factor=args.gradient_predivide_factor,
        communication_data_type=(
            torch.float16 if args.fp16 else torch.bfloat16 if args.bf16 else torch.float32
        ),
    )


__all__ = ["RoleAwareZeroOptimizerStage1", "configure_role_aware_zero1"]
