"""Service-backed ObjectRecovery for the same logical rank and layout only."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
import numpy as np

from .catalog import deserialize_checkpoint, validate_checkpoint_objects
from .matched import CONTROL_SOURCE_RANK
from .object_protection_runtime import (
    ProtectedObjectRuntime,
    restore_protected_from_services,
)
from .object_recovery import ObjectRuntimeIdentity
from .opportunity import RankGrouping
from .peer_delta_runtime import (
    PeerDeltaProtectionPlan,
    PeerShardAssignment,
)
from .replica_service import ReplicaServiceClient
from .rchara_checkpoint import RcharaCheckpointLoader


_ARMS = {"fixed": "adacheck_object_fixed", "m1": "m1_object"}
_STATE_ROOTS = frozenset(
    (
        "model",
        "optimizer",
        "lr_scheduler",
        "iteration",
        "random_rng_state",
        "np_rng_state",
        "torch_rng_state",
        "cuda_rng_state",
        "rng_tracker_states",
        "sampler",
    )
)
_OPTIMIZER_ROOTS = frozenset(
    (
        "dynamic_loss_scale",
        "cur_scale",
        "cur_iter",
        "global_step",
        "optimizer_state_dict",
        "fp32_groups",
    )
)
_DYNAMIC_OPTIMIZER_ROOTS = frozenset(
    ("last_overflow_iter", "scale_factor", "scale_window")
)


def _digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _protection_from_dict(value: Mapping[str, Any]) -> PeerDeltaProtectionPlan:
    grouping_value = value.get("grouping")
    if not isinstance(grouping_value, Mapping):
        raise ValueError("object service metadata has no grouping")
    grouping = RankGrouping(
        str(grouping_value["policy"]),
        tuple(tuple(int(rank) for rank in group) for group in grouping_value["groups"]),
        (
            None
            if grouping_value.get("seed") is None
            else int(grouping_value["seed"])
        ),
    )
    assignments = tuple(
        PeerShardAssignment(
            source_rank=int(item["source_rank"]),
            target_host=int(item["target_host"]),
            peer_rank=int(item["peer_rank"]),
            kind=str(item["kind"]),
            value_ids=tuple(str(value_id) for value_id in item["value_ids"]),
            logical_bytes=int(item["logical_bytes"]),
        )
        for item in value["assignments"]
    )
    required = value["required_value_ids_by_host"]
    if not isinstance(required, Mapping):
        raise ValueError("object service metadata has invalid host requirements")
    plan = PeerDeltaProtectionPlan(
        catalog_hash=str(value["catalog_hash"]),
        grouping=grouping,
        availability_scope=str(value["availability_scope"]),
        world_size=int(value["world_size"]),
        ranks_per_host=int(value["ranks_per_host"]),
        assignments=assignments,
        required_value_ids_by_host=tuple(
            (int(host), tuple(str(value_id) for value_id in value_ids))
            for host, value_ids in sorted(required.items(), key=lambda item: int(item[0]))
        ),
        planned_failure_sets=tuple(
            tuple(int(host) for host in failed)
            for failed in value["planned_failure_sets"]
        ),
        pair_local_transfer_bytes=int(value["pair_local_transfer_bytes"]),
        remote_transfer_bytes=int(value["remote_transfer_bytes"]),
        maximum_remote_publisher_bytes=int(value["maximum_remote_publisher_bytes"]),
        publisher_policy=str(value["publisher_policy"]),
    )
    if plan.to_dict() != dict(value):
        raise ValueError("object service protection plan hash or fields changed")
    return plan


@dataclass(frozen=True)
class SameLayoutServiceMetadata:
    """Hashes and placement only; never checkpoint tensor payloads."""

    epoch: int
    arm: str
    protection: PeerDeltaProtectionPlan
    semantic_plan_hash: str
    encoder_name: str
    source_layout_hash: str
    target_layout_hash: str
    recovery_plan_hash: str

    @classmethod
    def from_runtime(cls, runtime: ProtectedObjectRuntime) -> "SameLayoutServiceMetadata":
        value = cls(
            epoch=runtime.recovery.runtime.epoch,
            arm=runtime.recovery.arm,
            protection=runtime.protection,
            semantic_plan_hash=runtime.recovery.runtime.semantic_plan_hash,
            encoder_name=runtime.recovery.runtime.encoder_name,
            source_layout_hash=runtime.recovery.runtime.source_layout_hash,
            target_layout_hash=runtime.recovery.runtime.target_layout_hash,
            recovery_plan_hash=runtime.recovery.plan_hash,
        )
        value.validate()
        return value

    def validate(self) -> None:
        if self.epoch < 0 or self.arm not in set(_ARMS.values()):
            raise ValueError("object service metadata has an invalid epoch or arm")
        allowed_publishers = (
            {"least_load"}
            if self.arm == "m1_object"
            else {"rank_local", "first_eligible"}
        )
        if self.protection.publisher_policy not in allowed_publishers:
            raise ValueError("object service arm and publisher policy disagree")
        if tuple(
            tuple(sorted(set(int(host) for host in failed)))
            for failed in self.failure_sets
        ) != self.failure_sets:
            raise ValueError("object service metadata has a non-canonical failure contract")
        if not all(
            isinstance(value, str) and value
            for value in (
                self.semantic_plan_hash,
                self.encoder_name,
                self.source_layout_hash,
                self.target_layout_hash,
                self.recovery_plan_hash,
            )
        ):
            raise ValueError("object service metadata has an incomplete identity")
        if self.source_layout_hash != self.target_layout_hash:
            raise ValueError("object service metadata requires identical source and target layouts")

    @property
    def failure_sets(self) -> tuple[tuple[int, ...], ...]:
        return self.protection.planned_failure_sets

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "scope": "same_logical_rank_and_layout_only",
            "control": {"source_rank": CONTROL_SOURCE_RANK, "epoch": self.epoch},
            "arm": self.arm,
            "protection_plan": self.protection.to_dict(),
            "semantic_plan_hash": self.semantic_plan_hash,
            "encoder_name": self.encoder_name,
            "source_layout_hash": self.source_layout_hash,
            "target_layout_hash": self.target_layout_hash,
            "recovery_plan_hash": self.recovery_plan_hash,
        }

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        body = self._body()
        return {**body, "metadata_hash": _digest(body)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SameLayoutServiceMetadata":
        expected_keys = {
            "schema_version",
            "scope",
            "control",
            "arm",
            "protection_plan",
            "semantic_plan_hash",
            "encoder_name",
            "source_layout_hash",
            "target_layout_hash",
            "recovery_plan_hash",
            "metadata_hash",
        }
        if set(value) != expected_keys:
            raise ValueError("object service metadata has unknown or missing keys")
        body = {key: item for key, item in value.items() if key != "metadata_hash"}
        if (
            value["schema_version"] != 1
            or value["scope"] != "same_logical_rank_and_layout_only"
            or value["metadata_hash"] != _digest(body)
        ):
            raise ValueError("object service metadata schema or checksum changed")
        control = value["control"]
        if not isinstance(control, Mapping) or set(control) != {"source_rank", "epoch"}:
            raise ValueError("object service control locator is invalid")
        if int(control["source_rank"]) != CONTROL_SOURCE_RANK:
            raise ValueError("object service control source is invalid")
        metadata = cls(
            epoch=int(control["epoch"]),
            arm=str(value["arm"]),
            protection=_protection_from_dict(value["protection_plan"]),
            semantic_plan_hash=str(value["semantic_plan_hash"]),
            encoder_name=str(value["encoder_name"]),
            source_layout_hash=str(value["source_layout_hash"]),
            target_layout_hash=str(value["target_layout_hash"]),
            recovery_plan_hash=str(value["recovery_plan_hash"]),
        )
        metadata.validate()
        return metadata


def write_same_layout_service_metadata(
    path: str | os.PathLike[str], runtime: ProtectedObjectRuntime
) -> None:
    """Persist only the sealed same-layout plan and hashes, never tensor payloads."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(
            SameLayoutServiceMetadata.from_runtime(runtime).to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    os.replace(temporary, destination)


def load_same_layout_service_metadata(
    path: str | os.PathLike[str],
) -> SameLayoutServiceMetadata:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, Mapping):
        raise ValueError("object service metadata must be a JSON object")
    return SameLayoutServiceMetadata.from_dict(value)


def _validate_surviving_controls(
    metadata: SameLayoutServiceMetadata,
    clients: Mapping[int, ReplicaServiceClient],
) -> None:
    expected_plan = metadata.protection.to_dict()
    for client in clients.values():
        envelope, _ = client.get(CONTROL_SOURCE_RANK, metadata.epoch)
        if (
            envelope.manifest.epoch != metadata.epoch
            or envelope.manifest.source_rank != CONTROL_SOURCE_RANK
        ):
            raise ValueError("surviving service returned the wrong control epoch")
        control = deserialize_checkpoint(
            envelope.payload, envelope.manifest.payload_checksum
        )
        validate_checkpoint_objects(control, envelope.manifest.objects)
        if (
            control.get("schema_version") != 3
            or control.get("kind") != "peer_delta_control"
            or control.get("catalog_hash") != metadata.protection.catalog_hash
            or control.get("plan") != expected_plan
        ):
            raise ValueError("surviving service control differs from sealed metadata")


def _validate_rank_state(
    state: Any,
    *,
    module: Any,
    optimizer: Any,
    scheduler: Any,
    sampler: Any,
    args: Any,
    epoch: int,
) -> Mapping[str, Any]:
    if not isinstance(state, Mapping) or not _STATE_ROOTS <= set(state):
        raise ValueError("same-layout object restore has incomplete rank-state roots")
    if int(state["iteration"]) != epoch:
        raise ValueError("same-layout object restore iteration differs from its epoch")
    for key in ("model", "optimizer", "lr_scheduler", "sampler"):
        if not isinstance(state[key], Mapping):
            raise ValueError(f"same-layout object restore {key} root is invalid")
    live_model = module.state_dict()
    if set(state["model"]) != set(live_model):
        raise ValueError("same-layout object restore model keys changed")
    for name, value in state["model"].items():
        live = live_model[name]
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != tuple(live.shape)
            or value.dtype != live.dtype
        ):
            raise ValueError("same-layout object restore model tensor changed")
    optimizer_keys = set(state["optimizer"])
    required_optimizer = set(_OPTIMIZER_ROOTS)
    if bool(state["optimizer"].get("dynamic_loss_scale", False)):
        required_optimizer.update(_DYNAMIC_OPTIMIZER_ROOTS)
    if not required_optimizer <= optimizer_keys <= (
        required_optimizer | {"fp16_groups"}
    ):
        raise ValueError("same-layout object restore optimizer keys changed")
    restored_groups = state["optimizer"]["fp32_groups"]
    if len(restored_groups) != len(optimizer.fp32_groups):
        raise ValueError("same-layout object restore optimizer groups changed")
    for restored_group, live_group in zip(restored_groups, optimizer.fp32_groups):
        if len(restored_group) != len(live_group):
            raise ValueError("same-layout object restore optimizer groups changed")
        for restored, live in zip(restored_group, live_group):
            if (
                not isinstance(restored, torch.Tensor)
                or tuple(restored.shape) != tuple(live.shape)
                or restored.dtype != live.dtype
            ):
                raise ValueError("same-layout object restore optimizer tensor changed")
    inner = getattr(optimizer, "optimizer", None)
    inner_state = state["optimizer"]["optimizer_state_dict"]
    if not isinstance(inner, (torch.optim.Adam, torch.optim.AdamW)) or not isinstance(
        inner_state, Mapping
    ):
        raise ValueError("same-layout object restore requires Adam or AdamW")
    saved_param_groups = inner_state.get("param_groups")
    saved_slots = inner_state.get("state")
    if (
        not isinstance(saved_param_groups, list)
        or not isinstance(saved_slots, Mapping)
        or len(saved_param_groups) != len(inner.param_groups)
    ):
        raise ValueError("same-layout object restore Adam groups changed")
    for saved_group, live_group, live_masters in zip(
        saved_param_groups, inner.param_groups, optimizer.fp32_groups
    ):
        saved_parameters = saved_group.get("params")
        if (
            live_group.get("amsgrad", False)
            or not isinstance(saved_parameters, list)
            or len(saved_parameters) != len(live_masters)
        ):
            raise ValueError("same-layout object restore Adam configuration changed")
        for parameter_id, live in zip(saved_parameters, live_masters):
            slots = saved_slots.get(parameter_id)
            if not isinstance(slots, Mapping) or set(slots) != {
                "step",
                "exp_avg",
                "exp_avg_sq",
            }:
                raise ValueError("same-layout object restore Adam slots changed")
            if (
                not isinstance(slots["step"], torch.Tensor)
                or slots["step"].numel() != 1
                or slots["step"].dtype != torch.float32
            ):
                raise ValueError("same-layout object restore Adam step changed")
            for name in ("exp_avg", "exp_avg_sq"):
                value = slots[name]
                if (
                    not isinstance(value, torch.Tensor)
                    or tuple(value.shape) != tuple(live.shape)
                    or value.dtype != live.dtype
                ):
                    raise ValueError("same-layout object restore Adam moment changed")
    if set(state["lr_scheduler"]) != set(scheduler.state_dict()):
        raise ValueError("same-layout object restore scheduler keys changed")
    random.Random().setstate(state["random_rng_state"])
    np.random.RandomState().set_state(state["np_rng_state"])
    torch.Generator().set_state(state["torch_rng_state"])
    if (
        not isinstance(state["cuda_rng_state"], torch.Tensor)
        or state["cuda_rng_state"].dtype != torch.uint8
        or not isinstance(state["rng_tracker_states"], Mapping)
        or not state["rng_tracker_states"]
        or not all(
            isinstance(value, torch.Tensor) and value.dtype == torch.uint8
            for value in state["rng_tracker_states"].values()
        )
    ):
        raise ValueError("same-layout object restore RNG tracker state is empty")
    if sampler is None and state["sampler"]:
        raise ValueError("same-layout object restore has no live sampler")
    for name in state["sampler"]:
        if name not in {"consumed_samples", "epoch"} or not hasattr(sampler, name):
            raise ValueError(f"unsupported sampler state field: {name}")
        int(state["sampler"][name])
    if args.finetune or args.no_load_optim or args.no_load_rng:
        raise ValueError("same-layout object restore requires optimizer and RNG loading")
    return state


@dataclass(frozen=True)
class _MetadataExpectedRecovery:
    arm: str
    runtime: ObjectRuntimeIdentity
    plan_hash: str


@dataclass(frozen=True)
class _MetadataExpectedRuntime:
    protection: PeerDeltaProtectionPlan
    recovery: _MetadataExpectedRecovery


class SameLayoutObjectServiceController:
    """Fetch a sealed service checkpoint and restore the same rank/layout."""

    def __init__(
        self,
        metadata: SameLayoutServiceMetadata,
        endpoints: Sequence[str],
        failed_hosts: Sequence[int],
        *,
        rank: int,
        module: Any,
        optimizer: Any,
        scheduler: Any,
        sampler: Any,
        args: Any,
    ) -> None:
        metadata.validate()
        failed = tuple(sorted(set(int(host) for host in failed_hosts)))
        if tuple(int(host) for host in failed_hosts) != failed:
            raise ValueError("failed hosts must be unique and sorted")
        if failed not in metadata.failure_sets:
            raise ValueError("failed hosts are outside the sealed failure contract")
        if len(endpoints) != metadata.protection.host_count or not all(endpoints):
            raise ValueError("service endpoints do not match the sealed host topology")
        if not 0 <= rank < metadata.protection.world_size:
            raise ValueError("restore rank is outside the sealed world")
        self.metadata = metadata
        self.failed_hosts = failed
        self.clients = {
            host: ReplicaServiceClient.from_endpoint(endpoint)
            for host, endpoint in enumerate(endpoints)
            if host not in failed
        }
        if not self.clients:
            raise ValueError("object restore requires a surviving service")
        self.rank = rank
        self.module = module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sampler = sampler
        self.args = args

    def restore(self, epoch: int) -> int:
        if epoch != self.metadata.epoch:
            raise ValueError("requested object epoch differs from sealed metadata")
        _validate_surviving_controls(self.metadata, self.clients)
        expected = _MetadataExpectedRuntime(
            self.metadata.protection,
            _MetadataExpectedRecovery(
                arm=self.metadata.arm,
                runtime=ObjectRuntimeIdentity(
                    detector_catalog_hash=self.metadata.protection.catalog_hash,
                    semantic_plan_hash=self.metadata.semantic_plan_hash,
                    encoder_name=self.metadata.encoder_name,
                    source_layout_hash=self.metadata.source_layout_hash,
                    target_layout_hash=self.metadata.target_layout_hash,
                    epoch=epoch,
                    failure_sets=self.metadata.failure_sets,
                ),
                plan_hash=self.metadata.recovery_plan_hash,
            ),
        )
        restored = restore_protected_from_services(
            expected,
            self.clients,
            epoch=epoch,
            failed_hosts=self.failed_hosts,
        )
        if restored.plan_hash != self.metadata.recovery_plan_hash:
            raise ValueError("service payload differs from the sealed ObjectRecovery plan")
        if self.rank not in restored.rank_states:
            raise ValueError("service payload has no state for the current rank")
        state = _validate_rank_state(
            restored.rank_states[self.rank],
            module=self.module,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            sampler=self.sampler,
            args=self.args,
            epoch=epoch,
        )
        iteration = RcharaCheckpointLoader().load_remote_state(
            state,
            self.module,
            self.optimizer,
            self.scheduler,
            self.args,
            strict=True,
        )
        for name, value in state["sampler"].items():
            setattr(self.sampler, name, int(value))
        return iteration


def build_same_layout_service_controller(
    *, module: Any, optimizer: Any, scheduler: Any, sampler: Any, args: Any
) -> SameLayoutObjectServiceController:
    """Trainer factory for a same-rank/layout service restore, not FQN resharding."""
    required = (
        "CKPT_OBJECT_RECOVERY_METADATA",
        "CKPT_OBJECT_RECOVERY_ARM",
        "CKPT_REPLICA_ENDPOINTS",
        "CKPT_OBJECT_FAILED_HOSTS",
    )
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise ValueError(f"object service factory requires environment keys: {missing}")
    arm_name = os.environ["CKPT_OBJECT_RECOVERY_ARM"]
    if arm_name not in _ARMS:
        raise ValueError("CKPT_OBJECT_RECOVERY_ARM must be fixed or m1")
    metadata = load_same_layout_service_metadata(
        os.environ["CKPT_OBJECT_RECOVERY_METADATA"]
    )
    if metadata.arm != _ARMS[arm_name]:
        raise ValueError("requested object arm differs from sealed metadata")
    endpoints = json.loads(os.environ["CKPT_REPLICA_ENDPOINTS"])
    failed_hosts = json.loads(os.environ["CKPT_OBJECT_FAILED_HOSTS"])
    if not isinstance(endpoints, list) or not all(
        isinstance(endpoint, str) and endpoint for endpoint in endpoints
    ):
        raise ValueError("CKPT_REPLICA_ENDPOINTS must be a JSON string list")
    if not isinstance(failed_hosts, list) or not all(
        isinstance(host, int) and not isinstance(host, bool) for host in failed_hosts
    ):
        raise ValueError("CKPT_OBJECT_FAILED_HOSTS must be a JSON integer list")
    return SameLayoutObjectServiceController(
        metadata,
        endpoints,
        failed_hosts,
        rank=dist.get_rank(),
        module=module,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        args=args,
    )
