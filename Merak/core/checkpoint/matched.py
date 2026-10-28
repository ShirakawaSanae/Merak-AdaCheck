"""Matched two-host placement, service commit, and reconstruction."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import torch

from .catalog import (
    SerializedCheckpoint,
    deserialize_checkpoint,
    serialize_checkpoint,
    tensor_checksum,
    validate_checkpoint_objects,
)
from .manifest import CheckpointManifest, create_prepared_manifest
from .opportunity import (
    RankGrouping,
    asymmetric_plan_from_reduced,
    evaluate_grouping,
)
from .reduced import ReducedCheckpoint, ReducedObject
from .replica_service import ReplicaServiceClient, ServiceTransferMetrics
from .transport import ReplicaEnvelope


CONTROL_SOURCE_RANK = -1


def reduced_catalog_hash(reduced: ReducedCheckpoint) -> str:
    reduced.validate()
    value = {
        "full_logical_bytes": reduced.full_logical_bytes,
        "template_ranks": sorted(int(rank) for rank in reduced.templates),
        "objects": [
            {
                "value_id": item.value_id,
                "role": item.role,
                "dtype": item.dtype,
                "shape": list(item.shape),
                "logical_bytes": item.logical_bytes,
                "checksum": item.checksum,
                "source_ranks": list(item.source_ranks),
                "requirements": [requirement.__dict__ for requirement in item.requirements],
            }
            for item in reduced.objects
        ],
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ObjectTransfer:
    value_id: str
    source_rank: int
    source_host: int
    target_rank: int
    target_host: int
    logical_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class PublisherAssignment:
    host_index: int
    publisher_rank: int
    local_value_ids: tuple[str, ...]
    received_value_ids: tuple[str, ...]

    @property
    def value_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.local_value_ids + self.received_value_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "host_index": self.host_index,
            "publisher_rank": self.publisher_rank,
            "local_value_ids": list(self.local_value_ids),
            "received_value_ids": list(self.received_value_ids),
            "value_ids": list(self.value_ids),
        }


@dataclass(frozen=True)
class TwoHostProtectionPlan:
    catalog_hash: str
    grouping: RankGrouping
    world_size: int
    ranks_per_host: int
    assignments: tuple[PublisherAssignment, ...]
    transfers: tuple[ObjectTransfer, ...]

    def assignment_for_rank(self, rank: int) -> PublisherAssignment:
        for assignment in self.assignments:
            if assignment.publisher_rank == rank:
                return assignment
        raise KeyError(f"rank {rank} has no publisher assignment")

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": 1,
            "catalog_hash": self.catalog_hash,
            "grouping": self.grouping.to_dict(),
            "world_size": self.world_size,
            "ranks_per_host": self.ranks_per_host,
            "assignments": [item.to_dict() for item in self.assignments],
            "transfers": [item.to_dict() for item in self.transfers],
        }
        value["plan_hash"] = hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return value

    @property
    def plan_hash(self) -> str:
        return str(self.to_dict()["plan_hash"])


def build_two_host_protection_plan(
    reduced: ReducedCheckpoint,
    grouping: RankGrouping,
    *,
    ranks_per_host: int,
) -> TwoHostProtectionPlan:
    opportunity = asymmetric_plan_from_reduced(
        reduced,
        name="detector-built-reduced-checkpoint",
        ranks_per_host=ranks_per_host,
    )
    if opportunity.host_count != 2:
        raise ValueError("matched protection currently requires exactly two hosts")
    evaluation = evaluate_grouping(opportunity, grouping)
    peer = {
        rank: other
        for first, second in grouping.groups
        for rank, other in ((first, second), (second, first))
    }
    local: dict[int, list[str]] = {rank: [] for rank in range(opportunity.world_size)}
    received: dict[int, list[str]] = {
        rank: [] for rank in range(opportunity.world_size)
    }
    transfers_by_value = {item.object_id: item for item in evaluation.transfers}
    transfers = []

    for item in reduced.objects:
        holders_by_host = {
            host: tuple(
                rank
                for rank in item.source_ranks
                if opportunity.host_for_rank(rank) == host
            )
            for host in range(2)
        }
        for host in range(2):
            holders = holders_by_host[host]
            if holders:
                local[min(holders)].append(item.value_id)
                continue
            transfer = transfers_by_value.get(item.value_id)
            if transfer is None:
                raise ValueError(f"object {item.value_id} lacks a transfer to host {host}")
            if transfer.target_host != host or peer[transfer.source_rank] != transfer.target_rank:
                raise ValueError(f"object {item.value_id} transfer disagrees with grouping")
            received[transfer.target_rank].append(item.value_id)
            transfers.append(
                ObjectTransfer(
                    value_id=item.value_id,
                    source_rank=transfer.source_rank,
                    source_host=transfer.source_host,
                    target_rank=transfer.target_rank,
                    target_host=transfer.target_host,
                    logical_bytes=transfer.logical_bytes,
                )
            )

    assignments = tuple(
        PublisherAssignment(
            host_index=rank // ranks_per_host,
            publisher_rank=rank,
            local_value_ids=tuple(sorted(local[rank])),
            received_value_ids=tuple(sorted(received[rank])),
        )
        for rank in range(opportunity.world_size)
    )
    expected = {item.value_id for item in reduced.objects}
    for host in range(2):
        placed = [
            value_id
            for assignment in assignments
            if assignment.host_index == host
            for value_id in assignment.value_ids
        ]
        if len(placed) != len(set(placed)) or set(placed) != expected:
            raise ValueError(f"host {host} does not receive exactly one copy of every object")
    return TwoHostProtectionPlan(
        catalog_hash=reduced_catalog_hash(reduced),
        grouping=grouping,
        world_size=opportunity.world_size,
        ranks_per_host=ranks_per_host,
        assignments=assignments,
        transfers=tuple(sorted(transfers, key=lambda item: item.value_id)),
    )


@dataclass(frozen=True)
class DataShardRecord:
    host_index: int
    publisher_rank: int
    manifest_id: str
    payload_checksum: str
    value_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "host_index": self.host_index,
            "publisher_rank": self.publisher_rank,
            "manifest_id": self.manifest_id,
            "payload_checksum": self.payload_checksum,
            "value_ids": list(self.value_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DataShardRecord":
        return cls(
            host_index=int(value["host_index"]),
            publisher_rank=int(value["publisher_rank"]),
            manifest_id=str(value["manifest_id"]),
            payload_checksum=str(value["payload_checksum"]),
            value_ids=tuple(str(item) for item in value["value_ids"]),
        )


def build_data_shard(
    reduced: ReducedCheckpoint,
    plan: TwoHostProtectionPlan,
    rank: int,
    tensors: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    if reduced_catalog_hash(reduced) != plan.catalog_hash:
        raise ValueError("placement plan does not match the reduced checkpoint catalog")
    assignment = plan.assignment_for_rank(rank)
    source = tensors if tensors is not None else reduced.tensors
    values = {}
    objects = {item.value_id: item for item in reduced.objects}
    for value_id in assignment.value_ids:
        tensor = source.get(value_id)
        if tensor is None:
            raise ValueError(f"publisher rank {rank} is missing tensor {value_id}")
        if tensor_checksum(tensor) != objects[value_id].checksum:
            raise ValueError(f"publisher rank {rank} has corrupt tensor {value_id}")
        values[value_id] = tensor.detach().cpu().contiguous().clone()
    return {
        "schema_version": 1,
        "catalog_hash": plan.catalog_hash,
        "plan_hash": plan.plan_hash,
        "host_index": assignment.host_index,
        "publisher_rank": rank,
        "tensors": values,
    }


def _prepared_for_service(
    *,
    epoch: int,
    source_rank: int,
    source_host: str,
    target_rank: int,
    target_host: str,
    group_ranks: tuple[int, ...],
    serialized: SerializedCheckpoint,
    replica_count: int,
    tolerated_failures: int,
) -> ReplicaEnvelope:
    manifest = create_prepared_manifest(
        epoch=epoch,
        source_rank=source_rank,
        source_host=source_host,
        target_rank=target_rank,
        target_host=target_host,
        group_ranks=group_ranks,
        serialized=serialized,
        replica_count=replica_count,
        tolerated_failures=tolerated_failures,
    )
    return ReplicaEnvelope(manifest, serialized.payload)


def commit_data_shard(
    client: ReplicaServiceClient,
    *,
    epoch: int,
    shard: Mapping[str, Any],
    group_ranks: tuple[int, int],
    commit: bool = True,
) -> tuple[DataShardRecord | None, tuple[ServiceTransferMetrics, ...]]:
    rank = int(shard["publisher_rank"])
    serialized = serialize_checkpoint(shard)
    envelope = _prepared_for_service(
        epoch=epoch,
        source_rank=rank,
        source_host=f"host-{int(shard['host_index'])}",
        target_rank=rank,
        target_host=client.endpoint,
        group_ranks=group_ranks,
        serialized=serialized,
        replica_count=1,
        tolerated_failures=0,
    )
    metrics = [client.put_prepared(envelope)]
    if not commit:
        return None, tuple(metrics)
    committed = replace(
        envelope.manifest, state="committed", committed_time_ns=time.time_ns()
    )
    metrics.append(client.commit(committed))
    value_ids = tuple(sorted(str(item) for item in shard["tensors"]))
    return (
        DataShardRecord(
            host_index=int(shard["host_index"]),
            publisher_rank=rank,
            manifest_id=committed.manifest_id,
            payload_checksum=committed.payload_checksum,
            value_ids=value_ids,
        ),
        tuple(metrics),
    )


def build_control_state(
    reduced: ReducedCheckpoint,
    plan: TwoHostProtectionPlan,
    records: Sequence[DataShardRecord],
) -> dict[str, Any]:
    by_host = {
        host: tuple(
            sorted(
                (item for item in records if item.host_index == host),
                key=lambda item: item.publisher_rank,
            )
        )
        for host in range(2)
    }
    expected = {item.value_id for item in reduced.objects}
    for host, host_records in by_host.items():
        actual = [value_id for record in host_records for value_id in record.value_ids]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError(f"refusing to commit incomplete host {host} shard set")
    return {
        "schema_version": 1,
        "catalog_hash": plan.catalog_hash,
        "plan": plan.to_dict(),
        "templates": dict(reduced.templates),
        "objects": tuple(reduced.objects),
        "full_logical_bytes": reduced.full_logical_bytes,
        "host_records": {
            host: [record.to_dict() for record in host_records]
            for host, host_records in by_host.items()
        },
    }


def commit_control(
    clients: Sequence[ReplicaServiceClient],
    *,
    epoch: int,
    control_state: Mapping[str, Any],
) -> tuple[CheckpointManifest, tuple[ServiceTransferMetrics, ...]]:
    if len(clients) < 2 or len({client.endpoint for client in clients}) != len(clients):
        raise ValueError("control commit requires at least two distinct replica services")
    serialized = serialize_checkpoint(control_state)
    envelope = _prepared_for_service(
        epoch=epoch,
        source_rank=CONTROL_SOURCE_RANK,
        source_host="coordinator",
        target_rank=-2,
        target_host=f"{len(clients)}-host-services",
        group_ranks=(-2, CONTROL_SOURCE_RANK),
        serialized=serialized,
        replica_count=len(clients),
        tolerated_failures=1,
    )
    metrics = [client.put_prepared(envelope) for client in clients]
    committed = replace(
        envelope.manifest, state="committed", committed_time_ns=time.time_ns()
    )
    metrics.extend(client.commit(committed) for client in clients)
    return committed, tuple(metrics)


def restore_reduced_from_service(
    client: ReplicaServiceClient, *, epoch: int, host_index: int
) -> tuple[ReducedCheckpoint, int]:
    control_envelope, control_metrics = client.get(CONTROL_SOURCE_RANK, epoch)
    control = deserialize_checkpoint(
        control_envelope.payload, control_envelope.manifest.payload_checksum
    )
    validate_checkpoint_objects(control, control_envelope.manifest.objects)
    if int(control.get("schema_version", -1)) != 1:
        raise ValueError("unsupported matched control schema")
    records_value = control.get("host_records", {}).get(host_index)
    if records_value is None:
        records_value = control.get("host_records", {}).get(str(host_index))
    if not isinstance(records_value, list):
        raise ValueError(f"control record has no shard index for host {host_index}")
    records = tuple(DataShardRecord.from_dict(item) for item in records_value)
    tensors: dict[str, torch.Tensor] = {}
    wire_bytes = control_metrics.wire_bytes
    for record in records:
        envelope, metrics = client.get(record.publisher_rank, epoch)
        wire_bytes += metrics.wire_bytes
        if envelope.manifest.manifest_id != record.manifest_id:
            raise ValueError("data shard manifest does not match the control record")
        if envelope.manifest.payload_checksum != record.payload_checksum:
            raise ValueError("data shard checksum does not match the control record")
        shard = deserialize_checkpoint(
            envelope.payload, envelope.manifest.payload_checksum
        )
        validate_checkpoint_objects(shard, envelope.manifest.objects)
        if shard.get("catalog_hash") != control.get("catalog_hash"):
            raise ValueError("data shard catalog does not match the control record")
        values = shard.get("tensors")
        if not isinstance(values, Mapping) or tuple(sorted(values)) != record.value_ids:
            raise ValueError("data shard object list does not match the control record")
        for value_id, tensor in values.items():
            if value_id in tensors:
                raise ValueError(f"duplicate restored tensor {value_id}")
            tensors[str(value_id)] = tensor
    objects = control.get("objects")
    if not isinstance(objects, (list, tuple)) or not all(
        isinstance(item, ReducedObject) for item in objects
    ):
        raise ValueError("control record has invalid reduced objects")
    reduced = ReducedCheckpoint(
        templates=control["templates"],
        tensors=tensors,
        objects=tuple(objects),
        full_logical_bytes=int(control["full_logical_bytes"]),
    )
    reduced.validate()
    if reduced_catalog_hash(reduced) != control.get("catalog_hash"):
        raise ValueError("restored reduced catalog hash changed")
    return reduced, wire_bytes
