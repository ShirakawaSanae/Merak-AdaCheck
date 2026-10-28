"""Bridge detector content classes to failure-domain object recovery."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import torch

from .catalog import deserialize_checkpoint, tensor_checksum, validate_checkpoint_objects
from .matched import CONTROL_SOURCE_RANK, reduced_catalog_hash
from .object_recovery import (
    ObjectAddress,
    ObjectCatalog,
    ObjectFragment,
    ObjectRecoveryPlan,
    ObjectRuntimeIdentity,
    OpaqueRecord,
    OpaqueRequest,
    SliceRequest,
    TargetLayout,
    assert_fair_object_ab,
    build_object_recovery_plan,
    restore_object_layout,
)
from .peer_delta_runtime import (
    PeerDeltaProtectionPlan,
    restore_peer_delta_from_services,
)
from .reduced import ReducedCheckpoint
from .replica_service import ReplicaServiceClient


@dataclass(frozen=True)
class ProtectedObjectRuntime:
    reduced: ReducedCheckpoint
    protection: PeerDeltaProtectionPlan
    catalog: ObjectCatalog
    layout: TargetLayout
    recovery: ObjectRecoveryPlan

    @property
    def checkpoint_transfer_bytes(self) -> int:
        return self.protection.remote_transfer_bytes

    @property
    def maximum_publisher_bytes(self) -> int:
        return self.protection.maximum_remote_publisher_bytes


@dataclass(frozen=True)
class ObjectServiceRestore:
    rank_states: Mapping[int, Any]
    wire_bytes: int
    plan_hash: str


def _address(item: Any) -> ObjectAddress:
    shape = (1,) if not item.shape else item.shape
    return ObjectAddress(item.value_id, item.role, shape, item.dtype)


def _target_layout(reduced: ReducedCheckpoint) -> TargetLayout:
    slices = []
    for item in reduced.objects:
        address = _address(item)
        shape = (1,) if not item.shape else item.shape
        offsets = (0,) * len(shape)
        for rank in item.source_ranks:
            slices.append(SliceRequest(rank, address, offsets, shape))
    return TargetLayout(
        tuple(sorted(slices, key=lambda value: value.key)),
        tuple(OpaqueRequest(rank, rank, "template") for rank in sorted(reduced.templates)),
    )


def _protected_catalog(
    reduced: ReducedCheckpoint,
    protection: PeerDeltaProtectionPlan,
    *,
    epoch: int,
    payload_hosts: Sequence[int] | None = None,
) -> ObjectCatalog:
    if reduced_catalog_hash(reduced) != protection.catalog_hash:
        raise ValueError("protection plan does not match the detector catalog")
    available = (
        set(range(protection.host_count))
        if payload_hosts is None
        else {int(host) for host in payload_hosts}
    )
    if not available <= set(range(protection.host_count)):
        raise ValueError("payload host is outside the protection topology")
    objects = {item.value_id: item for item in reduced.objects}
    prototypes = {
        value_id: ObjectFragment.from_tensor(
            _address(item),
            epoch=epoch,
            offsets=(0,) * max(1, len(item.shape)),
            tensor=(
                reduced.tensors[value_id].reshape(1)
                if not item.shape
                else reduced.tensors[value_id]
            ),
            source_rank=min(item.source_ranks),
            storage_host=0,
            base_epoch=epoch,
            clone=False,
        )
        for value_id, item in objects.items()
    }
    fragments = []
    for assignment in protection.assignments:
        for value_id in assignment.value_ids:
            prototype = prototypes[value_id]
            fragments.append(
                replace(
                    prototype,
                    tensor=(
                        prototype.tensor
                        if assignment.target_host in available
                        else None
                    ),
                    source_rank=assignment.source_rank,
                    storage_host=assignment.target_host,
                )
            )
    opaque = tuple(
        OpaqueRecord.from_value(
            owner_rank=rank,
            key="template",
            epoch=epoch,
            value=template,
            storage_hosts=range(protection.host_count),
        )
        for rank, template in sorted(reduced.templates.items())
    )
    catalog = ObjectCatalog(epoch, tuple(fragments), opaque)
    catalog.validate()
    return catalog


def build_protected_object_runtime(
    reduced: ReducedCheckpoint,
    protection: PeerDeltaProtectionPlan,
    *,
    arm: str,
    epoch: int,
    semantic_plan_hash: str,
    encoder_name: str,
    source_layout_hash: str,
    payload_hosts: Sequence[int] | None = None,
) -> ProtectedObjectRuntime:
    """Seal one common detector/encoder/layout contract around a placement."""
    reduced.validate()
    layout = _target_layout(reduced)
    catalog = _protected_catalog(
        reduced, protection, epoch=epoch, payload_hosts=payload_hosts
    )
    identity = ObjectRuntimeIdentity(
        detector_catalog_hash=protection.catalog_hash,
        semantic_plan_hash=semantic_plan_hash,
        encoder_name=encoder_name,
        source_layout_hash=source_layout_hash,
        target_layout_hash=layout.layout_hash,
        epoch=epoch,
        failure_sets=protection.planned_failure_sets,
    )
    recovery = build_object_recovery_plan(
        catalog,
        layout,
        arm=arm,
        runtime=identity,
    )
    return ProtectedObjectRuntime(reduced, protection, catalog, layout, recovery)


def assert_fair_protected_ab(
    fixed: ProtectedObjectRuntime, m1: ProtectedObjectRuntime
) -> None:
    if (
        fixed.protection.catalog_hash != m1.protection.catalog_hash
        or fixed.protection.availability_scope != m1.protection.availability_scope
        or fixed.protection.world_size != m1.protection.world_size
        or fixed.protection.ranks_per_host != m1.protection.ranks_per_host
        or fixed.protection.planned_failure_sets != m1.protection.planned_failure_sets
    ):
        raise ValueError("protected A/B changes detector, topology, or failure contract")
    assert_fair_object_ab(fixed.recovery, m1.recovery)


def restore_protected_rank_states(
    runtime: ProtectedObjectRuntime, *, failed_hosts: Sequence[int]
) -> dict[int, Any]:
    restored = restore_object_layout(
        runtime.catalog, runtime.recovery, runtime.layout, failed_hosts=failed_hosts
    )
    tensors: dict[str, torch.Tensor] = {}
    templates = {}
    objects = {item.value_id: item for item in runtime.reduced.objects}
    for rank, state in restored.items():
        templates[rank] = state.opaque["template"]
        for value_id, tensor in state.tensors.items():
            item = objects[value_id]
            value = tensor.reshape(()) if not item.shape else tensor
            if (
                tuple(value.shape) != item.shape
                or str(value.dtype) != item.dtype
                or tensor_checksum(value) != item.checksum
            ):
                raise ValueError(f"restored object metadata conflicts: {value_id}")
            previous = tensors.setdefault(value_id, value)
            if not torch.equal(previous, value):
                raise ValueError(f"restored content class conflicts: {value_id}")
    rebuilt = ReducedCheckpoint(
        templates=templates,
        tensors=tensors,
        objects=runtime.reduced.objects,
        full_logical_bytes=runtime.reduced.full_logical_bytes,
    )
    rebuilt.validate()
    return {rank: rebuilt.reconstruct_rank(rank) for rank in sorted(templates)}


def restore_protected_from_services(
    expected: ProtectedObjectRuntime,
    clients: Mapping[int, ReplicaServiceClient] | Sequence[ReplicaServiceClient],
    *,
    epoch: int,
    failed_hosts: Sequence[int],
) -> ObjectServiceRestore:
    """Verify the sealed service plan, then restore using survivor payloads only."""
    indexed = (
        dict(clients)
        if isinstance(clients, Mapping)
        else {host: client for host, client in enumerate(clients)}
    )
    failed = {int(host) for host in failed_hosts}
    verification_wire = 0
    for host, client in sorted(indexed.items()):
        if host in failed:
            continue
        envelope, metrics = client.get(CONTROL_SOURCE_RANK, epoch)
        verification_wire += metrics.wire_bytes
        control = deserialize_checkpoint(
            envelope.payload, envelope.manifest.payload_checksum
        )
        validate_checkpoint_objects(control, envelope.manifest.objects)
        if (
            control.get("catalog_hash") != expected.protection.catalog_hash
            or control.get("plan", {}).get("plan_hash")
            != expected.protection.plan_hash
        ):
            raise ValueError("service control differs from the expected object plan")
    reduced, restore_wire = restore_peer_delta_from_services(
        indexed, epoch=epoch, failed_host_indices=tuple(sorted(failed))
    )
    runtime = build_protected_object_runtime(
        reduced,
        expected.protection,
        arm=expected.recovery.arm,
        epoch=epoch,
        semantic_plan_hash=expected.recovery.runtime.semantic_plan_hash,
        encoder_name=expected.recovery.runtime.encoder_name,
        source_layout_hash=expected.recovery.runtime.source_layout_hash,
        payload_hosts=tuple(sorted(set(indexed) - failed)),
    )
    if runtime.recovery.plan_hash != expected.recovery.plan_hash:
        raise ValueError("rebuilt recovery plan differs from the sealed plan")
    return ObjectServiceRestore(
        restore_protected_rank_states(runtime, failed_hosts=tuple(sorted(failed))),
        verification_wire + restore_wire,
        runtime.recovery.plan_hash,
    )
