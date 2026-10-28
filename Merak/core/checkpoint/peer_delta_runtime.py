"""Executable pair-local base/delta placement on the replica service."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .catalog import deserialize_checkpoint, tensor_checksum, validate_checkpoint_objects
from .matched import DataShardRecord, reduced_catalog_hash
from .opportunity import RankGrouping
from .reduced import ReducedCheckpoint, ReducedObject
from .replica_service import ReplicaServiceClient


@dataclass(frozen=True)
class PeerShardAssignment:
    source_rank: int
    target_host: int
    peer_rank: int
    kind: str
    value_ids: tuple[str, ...]
    logical_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_rank": self.source_rank,
            "target_host": self.target_host,
            "peer_rank": self.peer_rank,
            "kind": self.kind,
            "value_ids": list(self.value_ids),
            "logical_bytes": self.logical_bytes,
        }


@dataclass(frozen=True)
class PeerDeltaProtectionPlan:
    catalog_hash: str
    grouping: RankGrouping
    availability_scope: str
    world_size: int
    ranks_per_host: int
    assignments: tuple[PeerShardAssignment, ...]
    required_value_ids_by_host: tuple[tuple[int, tuple[str, ...]], ...]
    planned_failure_sets: tuple[tuple[int, ...], ...]
    pair_local_transfer_bytes: int
    remote_transfer_bytes: int
    maximum_remote_publisher_bytes: int = 0
    publisher_policy: str = "rank_local"

    @property
    def host_count(self) -> int:
        return self.world_size // self.ranks_per_host

    @property
    def host_global_control_bytes(self) -> int:
        """Backward-compatible name used by the completed two-host artifacts."""
        return self.remote_transfer_bytes

    @property
    def host_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            sorted(
                {
                    tuple(
                        sorted(
                            (
                                first // self.ranks_per_host,
                                second // self.ranks_per_host,
                            )
                        )
                    )
                    for first, second in self.grouping.groups
                }
            )
        )

    def peer_rank(self, rank: int) -> int:
        for first, second in self.grouping.groups:
            if rank == first:
                return second
            if rank == second:
                return first
        raise KeyError(f"rank {rank} is absent from the peer grouping")

    def peer_host(self, host: int) -> int:
        for first, second in self.host_pairs:
            if host == first:
                return second
            if host == second:
                return first
        raise KeyError(f"host {host} is absent from the peer grouping")

    def assignment(self, source_rank: int, target_host: int) -> PeerShardAssignment:
        for assignment in self.assignments:
            if (
                assignment.source_rank == source_rank
                and assignment.target_host == target_host
            ):
                return assignment
        raise KeyError(f"rank {source_rank} has no shard for host {target_host}")

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": 3,
            "catalog_hash": self.catalog_hash,
            "grouping": self.grouping.to_dict(),
            "availability_scope": self.availability_scope,
            "world_size": self.world_size,
            "ranks_per_host": self.ranks_per_host,
            "host_count": self.host_count,
            "host_pairs": [list(item) for item in self.host_pairs],
            "assignments": [item.to_dict() for item in self.assignments],
            "required_value_ids_by_host": {
                str(host): list(value_ids)
                for host, value_ids in self.required_value_ids_by_host
            },
            "planned_failure_sets": [
                list(failed) for failed in self.planned_failure_sets
            ],
            "pair_local_transfer_bytes": self.pair_local_transfer_bytes,
            "remote_transfer_bytes": self.remote_transfer_bytes,
            "host_global_control_bytes": self.host_global_control_bytes,
            "maximum_remote_publisher_bytes": self.maximum_remote_publisher_bytes,
            "publisher_policy": self.publisher_policy,
            "planned_tolerated_host_failures": max(
                (len(failed) for failed in self.planned_failure_sets), default=0
            ),
            "runtime_failure_sets_validated": [],
        }
        identity = {
            "catalog_hash": self.catalog_hash,
            "availability_scope": self.availability_scope,
            "world_size": self.world_size,
            "ranks_per_host": self.ranks_per_host,
            "groups": self.grouping.groups,
            "assignments": [item.to_dict() for item in self.assignments],
            "required_value_ids_by_host": self.required_value_ids_by_host,
            "planned_failure_sets": self.planned_failure_sets,
            "publisher_policy": self.publisher_policy,
        }
        value["plan_hash"] = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return value

    @property
    def plan_hash(self) -> str:
        return str(self.to_dict()["plan_hash"])


def build_peer_delta_protection_plan(
    reduced: ReducedCheckpoint,
    grouping: RankGrouping,
    *,
    ranks_per_host: int,
    availability_scope: str = "peer",
    balance_publishers: bool = True,
) -> PeerDeltaProtectionPlan:
    if availability_scope not in {"peer", "host"}:
        raise ValueError("availability_scope must be peer or host")
    catalog_hash = reduced_catalog_hash(reduced)
    world_size = len(reduced.templates)
    if world_size % ranks_per_host:
        raise ValueError("world_size must be divisible by ranks_per_host")
    flattened = [rank for group in grouping.groups for rank in group]
    if sorted(flattened) != list(range(world_size)):
        raise ValueError("peer grouping must contain every rank exactly once")
    peer = {
        rank: other
        for first, second in grouping.groups
        for rank, other in ((first, second), (second, first))
    }
    host_partner: dict[int, int] = {}
    for first, second in grouping.groups:
        first_host = first // ranks_per_host
        second_host = second // ranks_per_host
        if first_host == second_host:
            raise ValueError("peer grouping must cross physical hosts")
        for host, partner in ((first_host, second_host), (second_host, first_host)):
            previous = host_partner.setdefault(host, partner)
            if previous != partner:
                raise ValueError(f"host {host} is paired with multiple hosts")
    host_count = world_size // ranks_per_host
    if set(host_partner) != set(range(host_count)):
        raise ValueError("peer grouping must cover every host")

    rank_values = {rank: set() for rank in range(world_size)}
    object_bytes = {}
    object_holders = {}
    for item in reduced.objects:
        holders = {requirement.rank for requirement in item.requirements}
        object_bytes[item.value_id] = item.logical_bytes
        object_holders[item.value_id] = holders
        for rank in holders:
            rank_values[rank].add(item.value_id)

    def byte_sum(value_ids: Sequence[str]) -> int:
        return sum(object_bytes[value_id] for value_id in value_ids)

    assignments: list[PeerShardAssignment] = []
    if availability_scope == "peer":
        for source_rank in range(world_size):
            source_host = source_rank // ranks_per_host
            peer_rank = peer[source_rank]
            delta_ids = tuple(sorted(rank_values[source_rank] - rank_values[peer_rank]))
            base_ids = tuple(sorted(rank_values[source_rank]))
            assignments.extend(
                (
                    PeerShardAssignment(
                        source_rank=source_rank,
                        target_host=source_host,
                        peer_rank=peer_rank,
                        kind="base",
                        value_ids=base_ids,
                        logical_bytes=byte_sum(base_ids),
                    ),
                    PeerShardAssignment(
                        source_rank=source_rank,
                        target_host=peer_rank // ranks_per_host,
                        peer_rank=peer_rank,
                        kind="delta",
                        value_ids=delta_ids,
                        logical_bytes=byte_sum(delta_ids),
                    ),
                )
            )
    else:
        value_ids_by_assignment = {
            (rank, target): []
            for rank in range(world_size)
            for target in (
                rank // ranks_per_host,
                host_partner[rank // ranks_per_host],
            )
        }
        remote_publisher_loads = {rank: 0 for rank in range(world_size)}
        for first_host, second_host in sorted(
            {tuple(sorted((host, partner))) for host, partner in host_partner.items()}
        ):
            pair_ranks = set(
                range(first_host * ranks_per_host, (first_host + 1) * ranks_per_host)
            ) | set(
                range(second_host * ranks_per_host, (second_host + 1) * ranks_per_host)
            )
            for item in sorted(
                reduced.objects,
                key=lambda value: (-value.logical_bytes, value.value_id),
            ):
                holders = object_holders[item.value_id]
                if not holders & pair_ranks:
                    continue
                for target_host in (first_host, second_host):
                    local_holders = sorted(
                        rank
                        for rank in holders
                        if rank // ranks_per_host == target_host
                    )
                    eligible = local_holders or sorted(holders & pair_ranks)
                    is_remote = not local_holders
                    publisher = eligible[0]
                    if balance_publishers and is_remote:
                        publisher = min(
                            eligible,
                            key=lambda rank: (remote_publisher_loads[rank], rank),
                        )
                    value_ids_by_assignment[(publisher, target_host)].append(
                        item.value_id
                    )
                    if is_remote:
                        remote_publisher_loads[publisher] += item.logical_bytes
        for source_rank in range(world_size):
            source_host = source_rank // ranks_per_host
            for target_host in (source_host, host_partner[source_host]):
                value_ids = tuple(
                    sorted(value_ids_by_assignment[(source_rank, target_host)])
                )
                assignments.append(
                    PeerShardAssignment(
                        source_rank=source_rank,
                        target_host=target_host,
                        peer_rank=peer[source_rank],
                        kind=("base" if source_host == target_host else "delta"),
                        value_ids=value_ids,
                        logical_bytes=byte_sum(value_ids),
                    )
                )
    required_value_ids_by_host = []
    for target_host in range(host_count):
        pair_hosts = {target_host, host_partner[target_host]}
        expected = {
            value_id
            for rank, value_ids in rank_values.items()
            if rank // ranks_per_host in pair_hosts
            for value_id in value_ids
        }
        placed = {
            value_id
            for assignment in assignments
            if assignment.target_host == target_host
            for value_id in assignment.value_ids
        }
        if placed != expected:
            raise ValueError(f"peer delta plan leaves host {target_host} incomplete")
        required_value_ids_by_host.append((target_host, tuple(sorted(expected))))
    planned_failure_sets = ((),) + tuple((host,) for host in range(host_count))
    all_value_ids = set(object_bytes)
    required_by_host = dict(required_value_ids_by_host)
    for failed in planned_failure_sets:
        surviving = {
            value_id
            for host, value_ids in required_by_host.items()
            if host not in failed
            for value_id in value_ids
        }
        if surviving != all_value_ids:
            raise ValueError(f"peer delta plan does not tolerate host failures {failed}")
    remote_transfer_bytes = sum(
        assignment.logical_bytes
        for assignment in assignments
        if assignment.source_rank // ranks_per_host != assignment.target_host
    )
    remote_publisher_loads = {
        rank: sum(
            assignment.logical_bytes
            for assignment in assignments
            if assignment.source_rank == rank
            and assignment.source_rank // ranks_per_host != assignment.target_host
        )
        for rank in range(world_size)
    }
    return PeerDeltaProtectionPlan(
        catalog_hash=catalog_hash,
        grouping=grouping,
        availability_scope=availability_scope,
        world_size=world_size,
        ranks_per_host=ranks_per_host,
        assignments=tuple(assignments),
        required_value_ids_by_host=tuple(required_value_ids_by_host),
        planned_failure_sets=planned_failure_sets,
        pair_local_transfer_bytes=remote_transfer_bytes,
        remote_transfer_bytes=remote_transfer_bytes,
        maximum_remote_publisher_bytes=max(remote_publisher_loads.values(), default=0),
        publisher_policy=(
            "rank_local"
            if availability_scope == "peer"
            else ("least_load" if balance_publishers else "first_eligible")
        ),
    )


def build_peer_delta_shard(
    reduced: ReducedCheckpoint,
    plan: PeerDeltaProtectionPlan,
    *,
    source_rank: int,
    target_host: int,
    tensors: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    if reduced_catalog_hash(reduced) != plan.catalog_hash:
        raise ValueError("peer plan does not match the reduced catalog")
    assignment = plan.assignment(source_rank, target_host)
    source = reduced.tensors if tensors is None else tensors
    objects = {item.value_id: item for item in reduced.objects}
    values = {}
    for value_id in assignment.value_ids:
        tensor = source.get(value_id)
        if tensor is None or tensor_checksum(tensor) != objects[value_id].checksum:
            raise ValueError(
                f"rank {source_rank} has missing or corrupt tensor {value_id}"
            )
        values[value_id] = tensor.detach().cpu().contiguous().clone()
    return {
        "schema_version": 1,
        "catalog_hash": plan.catalog_hash,
        "plan_hash": plan.plan_hash,
        "source_rank": source_rank,
        "target_host": target_host,
        "peer_rank": assignment.peer_rank,
        "kind": assignment.kind,
        "tensors": values,
    }


def build_peer_delta_control_state(
    reduced: ReducedCheckpoint,
    plan: PeerDeltaProtectionPlan,
    records: Sequence[DataShardRecord],
) -> dict[str, Any]:
    indexed = {(item.publisher_rank, item.host_index): item for item in records}
    if len(indexed) != len(records):
        raise ValueError("duplicate peer shard record")
    for assignment in plan.assignments:
        record = indexed.get((assignment.source_rank, assignment.target_host))
        if record is None or record.value_ids != assignment.value_ids:
            raise ValueError(
                f"missing or inconsistent {assignment.kind} record for "
                f"rank {assignment.source_rank} on host {assignment.target_host}"
            )
    if len(records) != len(plan.assignments):
        raise ValueError("peer control contains unexpected shard records")
    required_by_host = dict(plan.required_value_ids_by_host)
    by_host = {}
    for host in range(plan.host_count):
        host_records = tuple(
            sorted(
                (item for item in records if item.host_index == host),
                key=lambda item: item.publisher_rank,
            )
        )
        placed = {
            value_id for item in host_records for value_id in item.value_ids
        }
        expected = set(required_by_host[host])
        if placed != expected:
            raise ValueError(f"refusing to commit incomplete peer host {host}")
        if plan.availability_scope == "host":
            placed_list = [
                value_id for item in host_records for value_id in item.value_ids
            ]
            if len(placed_list) != len(set(placed_list)):
                raise ValueError(f"host-global plan duplicates objects on host {host}")
        by_host[host] = host_records
    all_value_ids = {item.value_id for item in reduced.objects}
    for failed in plan.planned_failure_sets:
        surviving = {
            value_id
            for host, host_records in by_host.items()
            if host not in failed
            for record in host_records
            for value_id in record.value_ids
        }
        if surviving != all_value_ids:
            raise ValueError(f"failure set {failed} leaves peer control incomplete")
    return {
        "schema_version": 3,
        "kind": "peer_delta_control",
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


def _load_peer_control(
    client: ReplicaServiceClient, epoch: int
) -> tuple[Mapping[str, Any], int]:
    from .matched import CONTROL_SOURCE_RANK

    envelope, metrics = client.get(CONTROL_SOURCE_RANK, epoch)
    control = deserialize_checkpoint(envelope.payload, envelope.manifest.payload_checksum)
    validate_checkpoint_objects(control, envelope.manifest.objects)
    if control.get("schema_version") not in {2, 3} or control.get("kind") != "peer_delta_control":
        raise ValueError("not a peer delta control record")
    return control, metrics.wire_bytes


def _load_peer_host_tensors(
    client: ReplicaServiceClient,
    *,
    epoch: int,
    host_index: int,
    control: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], int]:
    records_value = control.get("host_records", {}).get(host_index)
    if records_value is None:
        records_value = control.get("host_records", {}).get(str(host_index))
    if not isinstance(records_value, list):
        raise ValueError(f"control record has no shard index for host {host_index}")
    records = tuple(DataShardRecord.from_dict(item) for item in records_value)
    objects_value = control.get("objects")
    if not isinstance(objects_value, (list, tuple)) or not all(
        isinstance(item, ReducedObject) for item in objects_value
    ):
        raise ValueError("peer control has invalid reduced objects")
    objects = {item.value_id: item for item in objects_value}
    tensors: dict[str, torch.Tensor] = {}
    wire_bytes = 0
    for record in records:
        envelope, metrics = client.get(record.publisher_rank, epoch)
        wire_bytes += metrics.wire_bytes
        if (
            envelope.manifest.manifest_id != record.manifest_id
            or envelope.manifest.payload_checksum != record.payload_checksum
        ):
            raise ValueError("peer shard does not match the control record")
        shard = deserialize_checkpoint(
            envelope.payload, envelope.manifest.payload_checksum
        )
        validate_checkpoint_objects(shard, envelope.manifest.objects)
        if (
            shard.get("catalog_hash") != control.get("catalog_hash")
            or shard.get("plan_hash") != control.get("plan", {}).get("plan_hash")
        ):
            raise ValueError("peer shard catalog or plan changed")
        values = shard.get("tensors")
        if not isinstance(values, Mapping) or tuple(sorted(values)) != record.value_ids:
            raise ValueError("peer shard object list changed")
        for value_id, tensor in values.items():
            expected = objects.get(value_id)
            if expected is None or tensor_checksum(tensor) != expected.checksum:
                raise ValueError(f"corrupt restored peer tensor {value_id}")
            previous = tensors.setdefault(str(value_id), tensor)
            if tensor_checksum(previous) != expected.checksum:
                raise ValueError(f"conflicting duplicate peer tensor {value_id}")
    return tensors, wire_bytes


def restore_peer_delta_from_services(
    clients: Mapping[int, ReplicaServiceClient] | Sequence[ReplicaServiceClient],
    *,
    epoch: int,
    failed_host_indices: Sequence[int] = (),
) -> tuple[ReducedCheckpoint, int]:
    """Merge pair-complete shards from the surviving failure domains."""
    indexed = (
        dict(clients)
        if isinstance(clients, Mapping)
        else {host: client for host, client in enumerate(clients)}
    )
    failed = {int(host) for host in failed_host_indices}
    survivors = {host: client for host, client in indexed.items() if host not in failed}
    if not survivors:
        raise ValueError("restore requires at least one surviving service")
    control = None
    tensors: dict[str, torch.Tensor] = {}
    wire_bytes = 0
    for host, client in sorted(survivors.items()):
        candidate, control_wire = _load_peer_control(client, epoch)
        wire_bytes += control_wire
        if control is None:
            control = candidate
            plan_value = control.get("plan", {})
            planned_failure_sets = plan_value.get("planned_failure_sets")
            if planned_failure_sets is not None:
                host_count = int(plan_value["host_count"])
                if not failed <= set(range(host_count)):
                    raise ValueError(
                        "failed peer service is outside the planned topology"
                    )
                unavailable = tuple(
                    sorted(set(range(host_count)) - set(survivors))
                )
                declared = {
                    tuple(sorted(int(host) for host in failure_set))
                    for failure_set in planned_failure_sets
                }
                if unavailable not in declared:
                    raise ValueError(
                        f"undeclared peer service failure set {unavailable}"
                    )
        elif (
            candidate.get("catalog_hash") != control.get("catalog_hash")
            or candidate.get("plan", {}).get("plan_hash")
            != control.get("plan", {}).get("plan_hash")
        ):
            raise ValueError("surviving services disagree on catalog or plan")
        host_tensors, host_wire = _load_peer_host_tensors(
            client, epoch=epoch, host_index=host, control=candidate
        )
        wire_bytes += host_wire
        for value_id, tensor in host_tensors.items():
            previous = tensors.setdefault(value_id, tensor)
            if tensor_checksum(previous) != tensor_checksum(tensor):
                raise ValueError(f"surviving services disagree on tensor {value_id}")
    assert control is not None
    objects_value = control.get("objects")
    if not isinstance(objects_value, (list, tuple)) or not all(
        isinstance(item, ReducedObject) for item in objects_value
    ):
        raise ValueError("peer control has invalid reduced objects")
    reduced = ReducedCheckpoint(
        templates=control["templates"],
        tensors=tensors,
        objects=tuple(objects_value),
        full_logical_bytes=int(control["full_logical_bytes"]),
    )
    reduced.validate()
    if reduced_catalog_hash(reduced) != control.get("catalog_hash"):
        raise ValueError("restored peer delta catalog hash changed")
    return reduced, wire_bytes


def restore_peer_delta_from_service(
    client: ReplicaServiceClient,
    *,
    epoch: int,
    host_index: int,
) -> tuple[ReducedCheckpoint, int]:
    """Backward-compatible single-service restore for a two-host plan."""
    return restore_peer_delta_from_services(
        {host_index: client}, epoch=epoch
    )
