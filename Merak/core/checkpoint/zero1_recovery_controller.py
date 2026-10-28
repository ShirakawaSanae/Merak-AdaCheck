"""Independent complete-state restore for the opt-in role-aware ZeRO-1 path."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.distributed as dist

from .merak_incremental import capture_runtime_state, restore_runtime_state
from .object_recovery import OpaqueRecord, ObjectFragment, RestoredRankState
from .object_recovery_controller import _RNG_KEYS
from .replica_service import ReplicaServiceClient
from .zero1_adapter import (
    _buffer_object_id,
    _model_object_id,
    apply_zero1_fragments,
    apply_zero1_model_tensors,
    iter_zero1_shards,
    persistent_model_buffers,
)


def _zero1_tensor_coordinates(module: Any, rank: int) -> tuple[int, int]:
    layout = getattr(module, "_merak_ep_layout", None)
    if layout is None or layout.world_size <= rank:
        raise ValueError("UCP ZeRO-1 requires a compatible expert-parallel layout")
    coordinates = layout.coordinates(rank)
    pipeline_grid = getattr(module, "_grid", None)
    if pipeline_grid is not None:
        tensor_group = pipeline_grid.get_slice_parallel_group()
        tensor_ranks = tuple(sorted(dist.get_process_group_ranks(tensor_group)))
        expected = tuple(sorted(layout.group_ranks(
            "tp", pp=coordinates["pp"], ep=coordinates["ep"], dp=coordinates["dp"]
        )))
        if tensor_ranks != expected:
            raise ValueError("Merak tensor group differs from the UCP EP layout")
    return coordinates["tp"], layout.tp


def _validate_zero1_tensor_parameters(module: Any, world_size: int) -> None:
    for parameter in module.parameters():
        partition_dim = hasattr(parameter, "partition_dim")
        partition_stride = getattr(parameter, "partition_stride", None)
        legacy_stride = getattr(parameter, "__dict__", {}).get("stride")
        partitioned = partition_dim or partition_stride is not None or isinstance(
            legacy_stride, int
        )
        if world_size == 1:
            if partitioned:
                raise ValueError("TP1 UCP parameters carry tensor-partition metadata")
        elif partitioned and not bool(getattr(parameter, "model_parallel", False)):
            raise ValueError("TP-partitioned UCP parameter lacks its model-parallel marker")


def capture_zero1_opaque(
    module: Any, optimizer: Any, scheduler: Any, *, epoch: int, sampler: Any
) -> dict[str, Any]:
    """Snapshot the rank-local roots that cannot be inferred from tensor shards."""
    iter_zero1_shards(module, optimizer)
    if getattr(optimizer, "step_count", optimizer.global_step) != epoch:
        raise ValueError("ZeRO-1 optimizer epoch differs from capture epoch")
    runtime = capture_runtime_state(epoch, sampler)
    groups = tuple(
        {key: copy.deepcopy(value) for key, value in group.items() if key != "params"}
        for group in optimizer.optimizer.param_groups
    )
    return {
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "rng": {key: runtime[key] for key in _RNG_KEYS},
        "sampler": runtime.get("sampler", {}),
        "progress": {
            "iteration": epoch,
            "optimizer": {
                "global_step": getattr(optimizer, "step_count", optimizer.global_step),
                "overflow": optimizer.overflow,
                "dynamic_loss_scale": optimizer.dynamic_loss_scale,
                "loss_scaler_type": type(optimizer.loss_scaler).__name__,
                "loss_scaler": copy.deepcopy(vars(optimizer.loss_scaler)),
                "groups": groups,
            },
        },
    }


def capture_zero1_rank(
    module: Any,
    optimizer: Any,
    scheduler: Any,
    *,
    rank: int,
    host: int,
    epoch: int,
    sampler: Any,
    storage_hosts: tuple[int, ...],
    tensor_parallel_rank: int = 0,
    tensor_parallel_world_size: int = 1,
    detector_audit: Mapping[str, Any] | None = None,
    optimizer_manifest: tuple[Any, ...] | None = None,
    detector_evidence: Mapping[str, torch.Tensor] | None = None,
    model_aliases: Mapping[str, str] | None = None,
) -> tuple[tuple[ObjectFragment, ...], tuple[OpaqueRecord, ...]]:
    """Capture complete named CPU objects and exact rank-local training roots."""
    from .object_recovery_adapter import capture_opaque_records
    from .zero1_adapter import (
        capture_zero1_fragments,
        capture_zero1_model_fragments,
        capture_zero1_step_fragments,
    )

    opaque = capture_zero1_opaque(module, optimizer, scheduler, epoch=epoch, sampler=sampler)
    if detector_audit is not None:
        if detector_audit.get("placement_applied") is not False:
            raise ValueError("MoE detector audit cannot claim placement application")
    fragments = (
        *capture_zero1_model_fragments(
            module, optimizer, rank=rank, storage_host=host, epoch=epoch,
            tensor_parallel_rank=tensor_parallel_rank,
            tensor_parallel_world_size=tensor_parallel_world_size,
            detector_evidence=detector_evidence,
            model_aliases=model_aliases,
        ),
        *capture_zero1_fragments(
            module, optimizer, rank=rank, storage_host=host, epoch=epoch,
            manifest=optimizer_manifest,
        ),
        *capture_zero1_step_fragments(
            module, optimizer, rank=rank, storage_host=host, epoch=epoch,
            manifest=optimizer_manifest,
        ),
    )
    return fragments, capture_opaque_records(
        epoch, opaque, owner_rank=rank, storage_hosts=storage_hosts
    )


def zero1_target_layout(
    module: Any, optimizer: Any, *, rank: int,
    tensor_parallel_rank: int = 0, tensor_parallel_world_size: int = 1,
    optimizer_manifest: tuple[Any, ...] | None = None,
    model_aliases: Mapping[str, str] | None = None,
):
    """Describe a target logical rank's required tensors and exact local roots."""
    from .object_recovery import OpaqueRequest, TargetLayout
    from .zero1_adapter import zero1_model_requests, zero1_optimizer_requests

    ep_layout = getattr(module, "_merak_ep_layout", None)
    if ep_layout is not None and ep_layout.world_size > rank:
        expected_rank, expected_world_size = _zero1_tensor_coordinates(module, rank)
        if (tensor_parallel_rank, tensor_parallel_world_size) != (
            expected_rank, expected_world_size
        ):
            raise ValueError("UCP tensor coordinates differ from the EP layout")
        _validate_zero1_tensor_parameters(module, expected_world_size)

    return TargetLayout(
        zero1_model_requests(
            module, rank=rank, tensor_parallel_rank=tensor_parallel_rank,
            tensor_parallel_world_size=tensor_parallel_world_size,
            model_aliases=model_aliases,
        )
        + zero1_optimizer_requests(
            module, optimizer, rank=rank, manifest=optimizer_manifest
        ),
        tuple(
            OpaqueRequest(rank, rank, key)
            for key in ("scheduler", "rng", "sampler", "progress")
        ),
    )


def apply_live_zero1_state(
    state: RestoredRankState,
    module: Any,
    optimizer: Any,
    scheduler: Any,
    *,
    sampler: Any,
    expected_iteration: int,
    runtime_restore: Callable[[Mapping[str, Any]], int] | None = None,
    manifest: tuple[Any, ...] | None = None,
    model_aliases: Mapping[str, str] | None = None,
) -> int:
    """Reject incomplete rank state before restoring ZeRO-1 and training roots."""
    if not isinstance(state, RestoredRankState) or set(state.opaque) != {
        "scheduler", "rng", "sampler", "progress"
    }:
        raise ValueError("ZeRO-1 restore requires all four opaque roots")
    scheduler_state, rng, sampler_state, progress = (
        state.opaque[key] for key in ("scheduler", "rng", "sampler", "progress")
    )
    if not all(isinstance(value, Mapping) for value in (
        scheduler_state, rng, sampler_state, progress
    )) or set(rng) != _RNG_KEYS:
        raise ValueError("ZeRO-1 scheduler, RNG, sampler, or progress is incomplete")
    if set(progress) != {"iteration", "optimizer"} or progress["iteration"] != expected_iteration:
        raise ValueError("ZeRO-1 restore epoch differs from requested epoch")
    metadata = progress["optimizer"]
    if not isinstance(metadata, Mapping) or set(metadata) != {
        "global_step", "overflow", "dynamic_loss_scale", "loss_scaler_type",
        "loss_scaler", "groups",
    } or metadata["global_step"] != expected_iteration:
        raise ValueError("ZeRO-1 optimizer progress is incomplete or stale")
    if (metadata["dynamic_loss_scale"] != optimizer.dynamic_loss_scale
        or metadata["loss_scaler_type"] != type(optimizer.loss_scaler).__name__
        or not isinstance(metadata["loss_scaler"], Mapping)
        or set(metadata["loss_scaler"]) != set(vars(optimizer.loss_scaler))
        or not isinstance(metadata["overflow"], bool)):
        raise ValueError("ZeRO-1 loss scaler is incompatible")
    if type(metadata["global_step"]) is not int or type(expected_iteration) is not int:
        raise ValueError("ZeRO-1 optimizer epoch must be integral")
    if set(scheduler_state) != set(scheduler.state_dict()):
        raise ValueError("ZeRO-1 scheduler state is incompatible")
    if not isinstance(sampler_state, Mapping) or (sampler is None and sampler_state) or any(
        key not in {"epoch", "consumed_samples"} or not hasattr(sampler, key)
        for key in sampler_state
    ):
        raise ValueError("ZeRO-1 sampler state is incompatible")
    local_groups = optimizer.optimizer.param_groups
    groups = metadata["groups"]
    if not isinstance(groups, (tuple, list)) or len(groups) != len(local_groups):
        raise ValueError("ZeRO-1 optimizer parameter groups differ")
    for saved, live in zip(groups, local_groups):
        if not isinstance(saved, Mapping) or set(saved) != set(live) - {"params"}:
            raise ValueError("ZeRO-1 optimizer parameter group topology differs")
        if (any(saved[key] != live[key] for key in (
                "_merak_zero1_role", "_merak_source_group"
            ))
            or any(
                saved[key] != live[key] for key in ("weight_decay", "betas", "eps", "amsgrad", "maximize")
                if key in saved
            )):
            raise ValueError("ZeRO-1 optimizer parameter group topology differs")
    model_tensors = {key: value for key, value in state.tensors.items()
                     if key.endswith(("::model", "::model.buffer"))}
    optimizer_tensors = {key: value for key, value in state.tensors.items()
                         if key.endswith(("::optimizer.master", "::optimizer.exp_avg", "::optimizer.exp_avg_sq"))}
    steps = {key: value for key, value in state.tensors.items()
             if key.endswith("::optimizer.step")}
    target_rank = dist.get_rank() if dist.is_initialized() else 0
    if len(state.tensors) != len(model_tensors) + len(optimizer_tensors) + len(steps):
        raise ValueError("ZeRO-1 restore contains unsupported tensor roles")
    aliases = dict(model_aliases or {})
    named_parameters = dict(module.named_parameters())
    model_ids = {
        name: _model_object_id(name, module, target_rank, parameter)
        for name, parameter in named_parameters.items()
    }
    if set(aliases) - set(named_parameters):
        raise ValueError("MoE model alias does not match a live parameter")
    live_tensors = {
        **{
            f"{_model_object_id(name, module, target_rank, parameter)}::model": parameter
            for name, parameter in named_parameters.items()
        },
        **{
            f"{_buffer_object_id(name, module, target_rank)}::model.buffer": value
            for name, value in persistent_model_buffers(module)
        },
    }
    if set(live_tensors) != set(model_tensors) or any(
        not isinstance(model_tensors[key], torch.Tensor)
        or model_tensors[key].shape != value.shape
        or model_tensors[key].dtype != value.dtype
        for key, value in live_tensors.items()
    ):
        raise ValueError(
            "ZeRO-1 model tensors are incomplete or incompatible: "
            f"missing={sorted(set(live_tensors) - set(model_tensors))}, "
            f"extra={sorted(set(model_tensors) - set(live_tensors))}, "
            f"state={sorted(model_tensors)}"
        )
    # The ZeRO adapter preflights all shard and moment ranges before mutation.
    apply_zero1_fragments(
        module, optimizer, optimizer_tensors, steps=steps,
        rank=target_rank,
        manifest=manifest or (
            getattr(module, "_merak_semantic_zero1_manifest", None)
            if bool(getattr(module, "_merak_semantic_zero1", False))
            else None
        ),
    )
    apply_zero1_model_tensors(
        module, model_tensors,
        rank=target_rank,
        model_aliases=aliases,
    )
    for saved, live in zip(groups, local_groups):
        live.update(copy.deepcopy(saved))
    scheduler.load_state_dict(copy.deepcopy(dict(scheduler_state)))
    vars(optimizer.loss_scaler).update(copy.deepcopy(dict(metadata["loss_scaler"])))
    optimizer.global_step = metadata["global_step"]
    optimizer.overflow = metadata["overflow"]
    runtime = {**rng, "sampler": sampler_state, "iteration": expected_iteration}
    restored = (runtime_restore or (lambda value: restore_runtime_state(value, sampler)))(runtime)
    if restored != expected_iteration:
        raise ValueError("ZeRO-1 runtime restored a different iteration")
    return restored


class LiveZero1ObjectRecoveryController:
    """Restore one same-layout role-aware ZeRO-1 rank from sealed UCP objects."""

    def __init__(
        self,
        module: Any,
        optimizer: Any,
        scheduler: Any,
        *,
        sampler: Any,
        tensor_parallel_rank: int,
        tensor_parallel_world_size: int,
        state_loader: Callable[[int], RestoredRankState],
        optimizer_manifest: tuple[Any, ...] | None = None,
        model_aliases: Mapping[str, str] | None = None,
        serial_group: object | None = None,
        serial_position: tuple[int, int] | None = None,
    ) -> None:
        self.module = module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sampler = sampler
        self.tensor_parallel_rank = tensor_parallel_rank
        self.tensor_parallel_world_size = tensor_parallel_world_size
        self.state_loader = state_loader
        self.optimizer_manifest = optimizer_manifest
        self.model_aliases = dict(model_aliases or {})
        self.serial_group = serial_group
        self.serial_position = serial_position

    def restore(self, epoch: int) -> int:
        if self.serial_group is not None:
            rank = dist.get_rank(self.serial_group)
            position, turns = self.serial_position or (rank, dist.get_world_size(self.serial_group))
            for _ in range(position):
                dist.barrier(group=self.serial_group)
        iteration = self._restore(epoch)
        if self.serial_group is not None:
            for _ in range(turns - position):
                dist.barrier(group=self.serial_group)
        return iteration

    def _restore(self, epoch: int) -> int:
        return apply_live_zero1_state(
            self.state_loader(epoch),
            self.module,
            self.optimizer,
            self.scheduler,
            sampler=self.sampler,
            expected_iteration=epoch,
            manifest=self.optimizer_manifest,
            model_aliases=self.model_aliases,
        )


def build_zero1_object_restore_controller(
    module: Any,
    optimizer: Any,
    scheduler: Any,
    *,
    sampler: Any,
    args: Any,
    control_group: object | None = None,
    endpoints: list[str] | None = None,
    rank_hosts: Mapping[int, int] | None = None,
    optimizer_manifest: tuple[Any, ...] | None = None,
) -> LiveZero1ObjectRecoveryController:
    """Build a same-layout role-aware ZeRO-1 restore from sealed replica services."""
    import os
    from pathlib import Path

    from .zero1_service import load_zero1_locator, restore_zero1_from_host
    if not bool(getattr(module, "_merak_role_aware_zero1", False)):
        raise ValueError("UCP ZeRO-1 restore requires the role-aware EP optimizer")
    if getattr(args, "zero_stage", None) != 1 or args.resume_from_checkpoint:
        raise ValueError("UCP ZeRO-1 restore requires a fresh ZeRO-1 job")
    try:
        endpoints = endpoints or __import__("json").loads(os.environ["CKPT_REPLICA_ENDPOINTS"])
        surviving_host = int(os.environ["CKPT_OBJECT_SURVIVING_HOST"])
        locator_path = Path(os.environ["CKPT_ZERO1_LOCATOR"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("UCP ZeRO-1 restore environment is incomplete") from error
    if (
        not isinstance(endpoints, list)
        or len(endpoints) != 2
        or any(not isinstance(endpoint, str) or not endpoint for endpoint in endpoints)
        or surviving_host not in (0, 1)
    ):
        raise ValueError("UCP ZeRO-1 service endpoints or survivor are invalid")
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    if world_size % 2:
        raise ValueError("UCP ZeRO-1 requires an even two-host rank layout")
    locator = load_zero1_locator(locator_path)
    client = ReplicaServiceClient.from_endpoint(
        endpoints[surviving_host],
        timeout_seconds=float(os.environ.get("CKPT_OBJECT_SERVICE_TIMEOUT_SECONDS", "600")),
    )
    from .matched import CONTROL_SOURCE_RANK

    envelope, _ = client.get(
        CONTROL_SOURCE_RANK,
        locator.epoch,
        control_checksum=locator.control_checksum,
        expected_manifest=locator.control_evidence[surviving_host],
    )
    from .catalog import deserialize_checkpoint, validate_checkpoint_objects

    control = deserialize_checkpoint(envelope.payload, envelope.manifest.payload_checksum)
    from .global_commit import manifest_evidence
    if manifest_evidence(envelope.manifest) != locator.control_evidence[surviving_host]:
        raise ValueError("UCP ZeRO-1 control manifest differs from sealed evidence")
    validate_checkpoint_objects(control, envelope.manifest.objects)
    if (
        control.get("kind") != "zero1_object_control"
        or control.get("epoch") != locator.epoch
        or control.get("data_evidence") != locator.data_evidence
        or tuple(control.get("model_aliases", ())) != locator.model_aliases
        or control.get("model_aliases_applied") is not bool(
            locator.model_aliases and any(locator.model_aliases)
        )
    ):
        raise ValueError("UCP ZeRO-1 sealed control differs from its locator")
    rank_metadata = control.get("rank_metadata")
    if (
        not isinstance(rank_metadata, (tuple, list))
        or len(rank_metadata) != world_size
        or any(not isinstance(row, Mapping) for row in rank_metadata)
    ):
        raise ValueError("UCP ZeRO-1 sealed rank metadata is incomplete")
    aliases_by_rank = locator.model_aliases or tuple({} for _ in range(world_size))
    if len(aliases_by_rank) != world_size:
        raise ValueError("UCP ZeRO-1 model aliases do not cover the target world")
    for source_rank, row in enumerate(rank_metadata):
        expected_alias_hash = hashlib.sha256(
            json.dumps(dict(aliases_by_rank[source_rank]), sort_keys=True).encode()
        ).hexdigest()
        if row.get("model_aliases_hash") != expected_alias_hash:
            raise ValueError("sealed ZeRO-1 model aliases differ from rank metadata")
    model_aliases = dict(aliases_by_rank[rank])
    if set(model_aliases) - {name for name, _ in module.named_parameters()}:
        raise ValueError("sealed MoE model alias does not match the target module")
    if control_group is None:
        from .object_semantic_runtime import create_world_gloo_control_group
        control_group = create_world_gloo_control_group()
    rank_hosts = dict(rank_hosts or {
        rank: int(os.environ.get("NODE_RANK", -1))
        for rank in range(world_size)
    })
    if (
        tuple(sorted(rank_hosts)) != tuple(range(world_size))
        or set(rank_hosts.values()) != {0, 1}
        or any(list(rank_hosts.values()).count(host) != world_size // 2 for host in (0, 1))
    ):
        raise ValueError("UCP ZeRO-1 target host map is incomplete")
    expert_layout = getattr(module, "_merak_ep_layout", None)
    if expert_layout is None or expert_layout.world_size != world_size:
        raise ValueError("UCP ZeRO-1 target EP layout is missing or incompatible")
    tensor_parallel_rank, tensor_parallel_world_size = _zero1_tensor_coordinates(
        module, rank
    )

    from .zero1_recovery_controller import zero1_target_layout
    zero1_target_layout(
        module,
        optimizer,
        rank=rank,
        tensor_parallel_rank=tensor_parallel_rank,
        tensor_parallel_world_size=tensor_parallel_world_size,
        optimizer_manifest=optimizer_manifest,
        model_aliases=model_aliases,
    )

    report = zero1_rank_metadata(
        module, optimizer, rank=rank, host=rank_hosts[rank], layout=expert_layout,
        optimizer_manifest=optimizer_manifest, model_aliases=model_aliases,
    )
    reports = [None] * world_size
    torch.distributed.all_gather_object(reports, report, group=control_group)
    if [item["rank"] for item in reports] != list(range(world_size)):
        raise ValueError("UCP ZeRO-1 target topology manifest is incomplete")
    expected = tuple(reports)

    def load(epoch: int) -> RestoredRankState:
        if epoch != locator.epoch:
            raise ValueError("requested UCP epoch differs from the sealed locator")

        layout = zero1_target_layout(
            module,
            optimizer,
            rank=rank,
            tensor_parallel_rank=tensor_parallel_rank,
            tensor_parallel_world_size=tensor_parallel_world_size,
            optimizer_manifest=optimizer_manifest,
            model_aliases=model_aliases,
        )
        state = restore_zero1_from_host(
            client,
            host=surviving_host,
            sealed=locator,
            rank=rank,
            layout=layout,
            expected_rank_metadata_rows=expected,
            target_only=os.environ.get("CKPT_ZERO1_TARGET_ONLY_RESTORE", "0") == "1",
        )
        return state

    return LiveZero1ObjectRecoveryController(
        module,
        optimizer,
        scheduler,
        sampler=sampler,
        tensor_parallel_rank=tensor_parallel_rank,
        tensor_parallel_world_size=tensor_parallel_world_size,
        state_loader=load,
        optimizer_manifest=optimizer_manifest,
        model_aliases=model_aliases,
        serial_group=control_group if os.environ.get("CKPT_ZERO1_SERIAL_RESTORE", "0") in {"1", "host"} else None,
        serial_position=(
            sorted(r for r, host in rank_hosts.items() if host == rank_hosts[rank]).index(rank),
            world_size // 2,
        ) if os.environ.get("CKPT_ZERO1_SERIAL_RESTORE", "0") == "host" else None,
    )


def zero1_rank_metadata(
    module: Any, optimizer: Any, *, rank: int, host: int, layout: Any,
    optimizer_manifest: tuple[Any, ...] | None = None,
    model_aliases: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    from ..moe.expert_parallel import ExpertParallelLayout

    if not isinstance(layout, ExpertParallelLayout) or layout.world_size <= rank:
        raise ValueError("UCP ZeRO-1 rank metadata requires an ExpertParallelLayout")
    coordinates = layout.coordinates(rank)
    expert_ids = sorted({
        int(parameter._merak_expert_id)
        for parameter in module.parameters()
        if bool(getattr(parameter, "_merak_expert_parallel", False))
        and parameter.numel() > 0
    })
    groups = {
        "shared": tuple(dist.get_process_group_ranks(module._merak_zero1_shared_process_group)),
        "expert": tuple(dist.get_process_group_ranks(module._merak_zero1_expert_process_group)),
        "dispatch": layout.group_ranks(
            "ep",
            pp=coordinates["pp"],
            dp=coordinates["dp"],
            tp=coordinates["tp"],
        ),
    }
    target = zero1_target_layout(
        module,
        optimizer,
        rank=rank,
        tensor_parallel_rank=coordinates["tp"],
        tensor_parallel_world_size=layout.tp,
        optimizer_manifest=optimizer_manifest,
        model_aliases=model_aliases,
    )
    topology = {
        "world_size": layout.world_size,
        "shape": (layout.pp, layout.ep, layout.dp, layout.tp),
        "order": layout.order,
    }
    topology_hash = hashlib.sha256(
        json.dumps(topology, sort_keys=True).encode()
    ).hexdigest()
    result = {
        "rank": rank,
        "host": host,
        "coordinates": coordinates,
        "optimizer_roles": tuple(optimizer.group_roles),
        "groups": groups,
        "expert_ids": expert_ids,
        "layout_hash": target.layout_hash,
        "model_aliases_hash": hashlib.sha256(
            json.dumps(dict(model_aliases or {}), sort_keys=True).encode()
        ).hexdigest(),
        "topology_hash": topology_hash,
    }
    return result
