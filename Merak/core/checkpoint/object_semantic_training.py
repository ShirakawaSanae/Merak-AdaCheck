"""Opt-in construction and lifecycle wrapper for SemanticFixed object training."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.distributed as dist

from .. import mpu
from ..amp.amp_optimizer import HalfPrecisionOptimizer
from .adacheck_semantic import (
    build_gradient_semantic_plan,
    semantic_binding_from_detector_config,
)
from .catalog import checkpoint_state_checksum
from .communication_profile import CommunicationProfile
from .merak_incremental import (
    _cpu_snapshot,
    capture_runtime_state,
    half_precision_parameter_ids,
)
from .object_detector_mapping import map_detector_catalog_to_gradient_shards
from .object_semantic_base import commit_distributed_object_semantic_base
from .object_semantic_deduplicated_base import (
    commit_distributed_deduplicated_object_semantic_base,
)
from .object_semantic_distributed import gather_active_gradient_plan
from .object_semantic_plan import RankBinding
from .object_semantic_runtime import (
    SEMANTIC_ARMS,
    SemanticFixedRuntimeController,
    create_world_gloo_control_group,
)
from .replica_service import ReplicaServiceClient
from .incremental_checkpoint import RecoveryBinding


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _integer(name: str, *, minimum: int = 1, default: int | None = None) -> int:
    try:
        value = int(_required(name) if default is None else os.environ.get(name, default))
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _service_timeout_seconds() -> int:
    name = "CKPT_OBJECT_SERVICE_TIMEOUT_SECONDS"
    try:
        value = int(os.environ.get(name, "60"))
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _endpoints() -> tuple[str, str]:
    try:
        value = json.loads(_required("CKPT_REPLICA_ENDPOINTS"))
    except json.JSONDecodeError as error:
        raise ValueError("CKPT_REPLICA_ENDPOINTS must be a JSON string list") from error
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(endpoint, str) or not endpoint for endpoint in value)
    ):
        raise ValueError("CKPT_REPLICA_ENDPOINTS must contain two endpoints")
    return value[0], value[1]


def _catalog_path(config_path: Path) -> Path:
    try:
        config = json.loads(config_path.read_text())
        summary = Path(config["frozen_catalog"]["summary_path"])
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("CKPT_OBJECT_DETECTOR_CONFIG is unreadable") from error
    if not summary.is_absolute():
        summary = config_path.parent / summary
    catalog = summary.parent / "object_catalog.jsonl"
    if not catalog.is_file():
        raise ValueError("detector frozen summary lacks object_catalog.jsonl")
    return catalog


def _model_argument(config_path: Path) -> str:
    try:
        value = json.loads(config_path.read_text())["model_argument"]
    except (KeyError, OSError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("CKPT_OBJECT_DETECTOR_CONFIG lacks model_argument") from error
    if not isinstance(value, str) or not value:
        raise ValueError("CKPT_OBJECT_DETECTOR_CONFIG model_argument is invalid")
    return value


def _summary_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _summary_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _summary_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summary_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class SemanticFixedTrainingController:
    """One base commit followed by the existing async SemanticFixed runtime."""

    def __init__(
        self,
        *,
        module: torch.nn.Module,
        optimizer: HalfPrecisionOptimizer,
        scheduler: Any,
        sampler: Any,
        epoch_provider: Callable[[], int],
        runtime_kwargs: Mapping[str, Any],
        binding: RecoveryBinding,
        rank_bindings: Sequence[RankBinding],
        parameter_ids: Sequence[Sequence[str]],
        semantic_binding: Any,
        semantic_plan: Any,
        detector_evidence: Mapping[str, Any],
        clients: Mapping[int, ReplicaServiceClient],
        control_group: object,
        root: Path,
        run_id: str,
        arm: str,
        profile: CommunicationProfile,
        base_upload_parallelism: int = 4,
        base_mode: str = "rank_local_v1",
    ) -> None:
        if type(base_upload_parallelism) is not int or base_upload_parallelism < 1:
            raise ValueError("SemanticFixed base upload parallelism must be positive")
        if base_mode not in {"rank_local_v1", "dp_deduplicated_v1"}:
            raise ValueError("SemanticFixed base mode is unsupported")
        self.module = module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sampler = sampler
        self.epoch_provider = epoch_provider
        self.runtime_kwargs = dict(runtime_kwargs)
        self.mapping_hash = str(self.runtime_kwargs.pop("mapping_hash"))
        self.binding = binding
        self.rank_bindings = tuple(rank_bindings)
        self.parameter_ids = tuple(tuple(group) for group in parameter_ids)
        self.semantic_binding = semantic_binding
        self.semantic_plan = semantic_plan
        self.detector_evidence = dict(detector_evidence)
        self.clients = clients
        self.control_group = control_group
        self.root = root
        self.run_id = run_id
        self.arm = arm
        self.profile = profile
        self.base_upload_parallelism = base_upload_parallelism
        self.base_mode = base_mode
        self.base_epoch: int | None = None
        self.base_id: str | None = None
        self.base_result: Any = None
        self.runtime: SemanticFixedRuntimeController | None = None

    def _encoder_evidence(self) -> dict[str, Any]:
        actions = self.semantic_plan.actions_for_rank(self.binding.rank)
        return {
            "schema_version": 1,
            "kind": "adacheck_semantic_fixed",
            "plan_hash": self.semantic_plan.plan_hash,
            "trainable_action": "gradient_bf16",
            "gradient_dtype": "torch.bfloat16",
            "gradient_snapshot_point": "raw_bf16_pre_unscale",
            "gradient_transform": "upcast_fp32_times_exact_step_scale",
            "metadata_action": "metadata_exact",
            "protection_action": "object_semantic_runtime",
            "remote_trainable_payload": "semantic_gradient_bf16",
            "peer_equivalence": "post_allreduce_dp_invariant",
            "object_partition": "fqn_global_slice_pp_tp",
            "dp_equivalence_basis": "post_allreduce_training_invariant",
            "integrity": "semantic_tensor_sha256_plus_serialized_payload_sha256",
            "rank_action_count": len(actions),
            "global_action_count": len(self.semantic_plan.actions),
        }

    def _base_state(self, epoch: int) -> dict[str, Any]:
        state = {
            "schema_version": 1,
            "kind": "merak_incremental_base",
            "iteration": int(epoch),
            "model": self.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict(),
            "lr_scheduler_type": (
                f"{type(self.scheduler).__module__}.{type(self.scheduler).__qualname__}"
            ),
            "parameter_ids": [list(group) for group in self.parameter_ids],
            "binding": asdict(self.binding),
            "run_id": self.run_id,
            "runtime_state": capture_runtime_state(epoch, self.sampler),
            "adacheck_semantic": self.semantic_binding.to_dict(),
            "adacheck_semantic_encoder": self._encoder_evidence(),
            "adacheck_detector_evidence": self.detector_evidence,
        }
        return _cpu_snapshot(state)

    def record_profile_interval(
        self, name: str, start_ns: int, end_ns: int, **fields: Any
    ) -> None:
        self.profile.record_interval(name, start_ns, end_ns, **fields)

    def start(self, epoch: int) -> None:
        if self.runtime is not None:
            raise RuntimeError("SemanticFixed training controller is already started")
        if self.optimizer.global_step != epoch:
            raise ValueError("SemanticFixed optimizer and base epochs disagree")
        started = time.monotonic_ns()
        if self.base_mode == "dp_deduplicated_v1":
            self.base_result = commit_distributed_deduplicated_object_semantic_base(
                self.clients,
                run_id=self.run_id,
                epoch=epoch,
                local_base_factory=lambda: self._base_state(epoch),
                rank_bindings=self.rank_bindings,
                locator_path=self.root / "base" / f"epoch-{epoch:08d}.json",
                group=self.control_group,
                max_parallel_uploads=self.base_upload_parallelism,
            )
            self.base_id = self.base_result.local_base_id
        else:
            base = self._base_state(epoch)
            self.base_id = f"sha256:{checkpoint_state_checksum(base)}"
            self.base_result = commit_distributed_object_semantic_base(
                self.clients,
                run_id=self.run_id,
                epoch=epoch,
                local_base=base,
                rank_bindings=self.rank_bindings,
                locator_path=self.root / "base" / f"epoch-{epoch:08d}.json",
                group=self.control_group,
                max_parallel_uploads=self.base_upload_parallelism,
            )
        self.record_profile_interval(
            "checkpoint_object_semantic_base",
            started,
            time.monotonic_ns(),
            epoch=epoch,
            arm=self.arm,
        )
        self.base_epoch = int(epoch)
        self.runtime = SemanticFixedRuntimeController(
            self.optimizer,
            base_id=self.base_id,
            epoch_provider=self.epoch_provider,
            runtime_capture=lambda value: capture_runtime_state(value, self.sampler),
            scheduler=self.scheduler,
            profile=self.profile.record_interval,
            **self.runtime_kwargs,
        )
        self.runtime.start(epoch)

    def poll(self, current_epoch: int | None = None) -> tuple[Any, ...]:
        if self.runtime is None:
            return ()
        return self.runtime.poll(current_epoch)

    def finalize(self, current_epoch: int | None = None) -> dict[str, Any]:
        if self.runtime is None:
            return {}
        summary = self.runtime.finalize(current_epoch)
        summary.update(
            {
                "run_id": self.run_id,
                "rank": self.binding.rank,
                "arm": self.arm,
                "base_epoch": self.base_epoch,
                "base_id": self.base_id,
                "base_mode": self.base_mode,
                "base_result": _summary_value(self.base_result),
                "plan_hash": self.runtime.plan.plan_hash,
                "mapping_hash": self.mapping_hash,
            }
        )
        path = self.root / "metrics" / f"rank_{self.binding.rank:05d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(_summary_value(summary), indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
        self.profile.close()
        return summary


@dataclass(frozen=True)
class SemanticFixedStaticContext:
    """Detector-bound state that a fresh source or restore rank must rebuild."""

    arm: str
    binding: RecoveryBinding
    rank_bindings: tuple[RankBinding, ...]
    parameter_ids: tuple[tuple[str, ...], ...]
    semantic_binding: Any
    semantic_plan: Any
    detector_evidence: Mapping[str, Any]
    plan: Any
    clients: Mapping[int, ReplicaServiceClient]
    control_group: object
    root: Path
    run_id: str
    mapping_hash: str


def build_semantic_fixed_static_context(
    module: torch.nn.Module,
    optimizer: HalfPrecisionOptimizer,
    args: Any,
    *,
    arm: str | None = None,
) -> SemanticFixedStaticContext:
    """Rebuild the immutable detector, topology, and active-object plan."""
    arm = _required("CKPT_EXPERIMENT_ARM") if arm is None else arm
    if arm not in SEMANTIC_ARMS:
        raise ValueError("CKPT_EXPERIMENT_ARM is not a SemanticFixed arm")
    ranks_per_node = _integer("CKPT_RANKS_PER_NODE")
    if ranks_per_node != 8:
        raise ValueError("SemanticFixed training requires eight ranks per host")
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("SemanticFixed training requires torch.distributed")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != 16:
        raise ValueError("SemanticFixed training requires two eight-rank hosts")
    endpoints = _endpoints()
    run_id = _required("CKPT_OBJECT_RUN_ID")
    root = Path(_required("CKPT_OBJECT_ROOT"))
    config_path = Path(_required("CKPT_OBJECT_DETECTOR_CONFIG"))
    catalog_path = _catalog_path(config_path)

    control_group = create_world_gloo_control_group()
    binding = RecoveryBinding(
        rank=rank,
        world_size=world_size,
        data_parallel_rank=mpu.get_data_parallel_rank(),
        pipeline_parallel_rank=mpu.get_pipe_parallel_rank(),
        tensor_parallel_rank=mpu.get_model_parallel_rank(),
    )
    topology: list[Any] = [None] * world_size
    dist.all_gather_object(topology, asdict(binding), group=control_group)
    if [item.get("rank") if isinstance(item, Mapping) else None for item in topology] != list(range(world_size)):
        raise ValueError("SemanticFixed rank topology is incomplete")
    rank_bindings = tuple(
        RankBinding(
            rank=item["rank"],
            pipeline_rank=item["pipeline_parallel_rank"],
            tensor_rank=item["tensor_parallel_rank"],
            data_rank=item["data_parallel_rank"],
            host=item["rank"] // ranks_per_node,
        )
        for item in topology
    )
    pairs = tuple((index, ranks_per_node + index) for index in range(ranks_per_node))
    semantic_binding, detector_evidence = semantic_binding_from_detector_config(
        config_path,
        expected_model=_model_argument(config_path),
        world_size=world_size,
        ranks_per_host=ranks_per_node,
        rank_bindings=topology,
        pairs=pairs,
    )
    parameter_ids = half_precision_parameter_ids(module, optimizer)
    parameter_topologies: list[Any] = [None] * world_size
    dist.all_gather_object(parameter_topologies, parameter_ids, group=control_group)
    semantic_plan = build_gradient_semantic_plan(
        semantic_binding,
        world_size=world_size,
        pairs=pairs,
        parameter_ids_by_rank=parameter_topologies,
    )
    local_mapping = map_detector_catalog_to_gradient_shards(
        catalog_path,
        expected_catalog_hash=semantic_binding.catalog_hash,
        world_size=world_size,
        binding=rank_bindings[rank],
        parameter_ids=parameter_ids,
        parameter_groups=optimizer.fp16_groups,
        tensor_parallel_rank=binding.tensor_parallel_rank,
        tensor_parallel_world_size=mpu.get_model_parallel_world_size(),
        encoded_dtype=torch.bfloat16,
    )
    plan = gather_active_gradient_plan(
        rank_bindings[rank],
        local_mapping.shards,
        arm=arm,
        detector_catalog_hash=semantic_binding.catalog_hash,
        semantic_plan_hash=semantic_plan.plan_hash,
        fixed_pairs=pairs,
        group=control_group,
    )
    return SemanticFixedStaticContext(
        arm=arm,
        binding=binding,
        rank_bindings=rank_bindings,
        parameter_ids=tuple(tuple(group) for group in parameter_ids),
        semantic_binding=semantic_binding,
        semantic_plan=semantic_plan,
        detector_evidence=dict(detector_evidence),
        plan=plan,
        clients={
            host: ReplicaServiceClient.from_endpoint(
                endpoint, timeout_seconds=_service_timeout_seconds()
            )
            for host, endpoint in enumerate(endpoints)
        },
        control_group=control_group,
        root=root,
        run_id=run_id,
        mapping_hash=local_mapping.mapping_hash,
    )


def build_semantic_fixed_training_controller(
    module: torch.nn.Module,
    optimizer: HalfPrecisionOptimizer,
    scheduler: Any,
    args: Any,
    sampler: Any,
    epoch_provider: Callable[[], int],
) -> SemanticFixedTrainingController:
    """Build the independent runtime for the trainer lifecycle adapter."""
    if os.environ.get("CKPT_OBJECT_SEMANTIC_ENABLE") != "1":
        raise ValueError("CKPT_OBJECT_SEMANTIC_ENABLE=1 is required")
    if not isinstance(optimizer, HalfPrecisionOptimizer) or not bool(getattr(args, "bf16", False)):
        raise ValueError("SemanticFixed training requires the BF16 HalfPrecisionOptimizer")
    if getattr(args, "zero_stage", None) is not None:
        raise ValueError("SemanticFixed training does not support ZeRO")
    if getattr(args, "dataloader_num_workers", None) != 0:
        raise ValueError("SemanticFixed training requires dataloader_num_workers=0")
    if type(optimizer.optimizer) is not torch.optim.AdamW:
        raise ValueError("SemanticFixed training requires torch.optim.AdamW")
    for group in optimizer.optimizer.param_groups:
        if group.get("foreach") is not False or group.get("fused") not in (None, False):
            raise ValueError("SemanticFixed training requires non-fused AdamW foreach=False")
        if any(group.get(name, False) for name in ("amsgrad", "capturable", "differentiable")):
            raise ValueError("SemanticFixed training rejects unsupported AdamW semantics")
    if any(
        parameter.dtype is not torch.bfloat16
        for group in optimizer.fp16_groups
        for parameter in group
    ):
        raise ValueError("SemanticFixed training requires BF16 optimizer parameters")

    arm = _required("CKPT_EXPERIMENT_ARM")
    if arm not in SEMANTIC_ARMS:
        raise ValueError("CKPT_EXPERIMENT_ARM is not a SemanticFixed arm")
    max_inflight = _integer("CKPT_OBJECT_MAX_INFLIGHT")
    base_upload_parallelism = _integer(
        "CKPT_OBJECT_BASE_UPLOAD_PARALLELISM", default=4
    )
    base_mode = os.environ.get("CKPT_OBJECT_BASE_MODE", "rank_local_v1")
    if base_mode not in {"rank_local_v1", "dp_deduplicated_v1"}:
        raise ValueError("CKPT_OBJECT_BASE_MODE is unsupported")
    context = build_semantic_fixed_static_context(module, optimizer, args, arm=arm)
    profile = CommunicationProfile(
        enabled=True,
        jsonl_path=context.root / "profiles" / f"rank_{context.binding.rank:05d}.jsonl",
    )
    return SemanticFixedTrainingController(
        module=module,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        epoch_provider=epoch_provider,
        runtime_kwargs={
            "arm": context.arm,
            "plan": context.plan,
            "binding": context.binding,
            "parameter_ids": context.parameter_ids,
            "control_group": context.control_group,
            "clients": context.clients,
            "run_id": context.run_id,
            "locator_root": context.root / "updates",
            "max_inflight": max_inflight,
            "mapping_hash": context.mapping_hash,
        },
        binding=context.binding,
        rank_bindings=context.rank_bindings,
        parameter_ids=context.parameter_ids,
        semantic_binding=context.semantic_binding,
        semantic_plan=context.semantic_plan,
        detector_evidence=context.detector_evidence,
        clients=context.clients,
        control_group=context.control_group,
        root=context.root,
        run_id=context.run_id,
        arm=context.arm,
        profile=profile,
        base_upload_parallelism=base_upload_parallelism,
        base_mode=base_mode,
    )
