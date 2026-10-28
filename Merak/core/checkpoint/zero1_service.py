"""Sealed two-host CPU transport for independent ZeRO-1 object snapshots.

Place exact equal-version fragments using fixed pairs, host pooling, or M1;
this remains a synchronous CPU gate, not an online training encoder.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from . import object_recovery as recovery
from .catalog import (
    deserialize_checkpoint, iter_checkpoint_objects, serialize_checkpoint,
    validate_checkpoint_objects,
)
from .global_commit import manifest_evidence
from .matched import CONTROL_SOURCE_RANK
from .manifest import create_prepared_manifest
from .object_recovery import (
    ObjectCatalog, ObjectFragment, ObjectRuntimeIdentity, OpaqueRecord,
    RestoredRankState, TargetLayout, build_object_recovery_plan,
    restore_object_layout,
)
from .object_update_transport import commit_object_update_state, seal_object_update_host
from .replica_service import ReplicaServiceClient
from .transport import ReplicaEnvelope


@dataclass(frozen=True)
class SealedZero1Epoch:
    epoch: int
    control_checksum: str
    control_evidence: tuple[dict[str, Any], dict[str, Any]]
    data_evidence: tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]
    remote_payload_bytes: int
    maximum_remote_put_ms: float
    arm: str = "adacheck_fixed"
    detector_audit: Mapping[str, Any] | None = None
    model_aliases: tuple[Mapping[str, str], ...] = ()
    model_aliases_applied: bool = False


@dataclass(frozen=True)
class Zero1PublicationObservation:
    """Client-side payload and PUT-through-commit observations for one publication."""

    epoch: int
    remote_payload_bytes: int
    remote_wire_bytes: int
    remote_logical_bytes: int
    remote_edge_payload_bytes: int
    remote_control_logical_bytes: int
    remote_control_payload_bytes: int
    remote_control_wire_bytes: int
    maximum_remote_put_ms: float
    maximum_remote_edge_ack_ms: float
    committed_inventory: Mapping[str, Any] | None = None


def _commit_zero1_bundle(
    client: ReplicaServiceClient,
    state: Mapping[str, Any],
    **kwargs: Any,
) -> tuple[Any, int, float, float, int, int]:
    logical_bytes = sum(
        item.tensor.numel() * item.tensor.element_size()
        for item in state.get("fragments", ())
        if item.tensor is not None
    )
    logical_bytes += sum(
        sum(obj.logical_bytes for obj in iter_checkpoint_objects(item.value))
        for item in state.get("opaque", ())
    )
    serialized = serialize_checkpoint(state, state_is_snapshot=True)
    manifest = create_prepared_manifest(
        epoch=int(state["epoch"]),
        source_rank=kwargs["source_rank"],
        source_host=f"host-{kwargs['source_host']}",
        target_rank=kwargs["target_rank"],
        target_host=client.endpoint,
        group_ranks=kwargs["group_ranks"],
        serialized=serialized,
    )
    edge_started = time.perf_counter()
    put_started = time.perf_counter()
    put_metrics = client.put_prepared(ReplicaEnvelope(manifest, serialized.payload))
    put_ms = (time.perf_counter() - put_started) * 1000.0
    committed = replace(
        manifest, state="committed", committed_time_ns=time.time_ns()
    )
    commit_metrics = client.commit(committed)
    return (
        committed,
        put_metrics.wire_bytes + commit_metrics.wire_bytes,
        put_ms,
        (time.perf_counter() - edge_started) * 1000.0,
        logical_bytes,
        len(serialized.payload),
    )


def write_zero1_locator(path: str | os.PathLike[str], sealed: SealedZero1Epoch) -> None:
    """Expose an epoch to fresh jobs only after both replica services are sealed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(asdict(sealed), sort_keys=True) + "\n")
    temporary.replace(destination)


def load_zero1_locator(path: str | os.PathLike[str]) -> SealedZero1Epoch:
    value = json.loads(Path(path).read_text())
    fields = set(SealedZero1Epoch.__dataclass_fields__)
    if set(value) not in (
        fields,
        fields - {"model_aliases_applied"},
        fields - {"detector_audit", "model_aliases_applied"},
        fields - {"detector_audit", "model_aliases", "model_aliases_applied"},
    ):
        raise ValueError("ZeRO-1 locator schema is incomplete")
    return SealedZero1Epoch(
        epoch=value["epoch"],
        control_checksum=value["control_checksum"],
        control_evidence=tuple(value["control_evidence"]),
        data_evidence=tuple(tuple(row) for row in value["data_evidence"]),
        remote_payload_bytes=value["remote_payload_bytes"],
        maximum_remote_put_ms=value["maximum_remote_put_ms"],
        arm=value["arm"],
        detector_audit=value.get("detector_audit"),
        model_aliases=tuple(value.get("model_aliases", ())),
        model_aliases_applied=value.get("model_aliases_applied", False),
    )


def _descriptor_catalog_hash(
    epoch: int, fragments: Sequence[ObjectFragment], opaque: Sequence[OpaqueRecord],
) -> str:
    """Match ObjectCatalog.content_hash without moving tensors or rank-local RNG."""
    if any(item.epoch != epoch for item in (*fragments, *opaque)):
        raise ValueError("ZeRO-1 descriptor epoch differs from the publication epoch")
    keys = {}
    unique = {}
    for item in fragments:
        if item.base_epoch != epoch or item.dependency_epoch is not None:
            raise ValueError("ZeRO-1 descriptors require materialized final objects")
        key = (item.address.key, item.offsets, item.lengths)
        if key in keys and keys[key] != item.version_id:
            raise ValueError("ZeRO-1 replicas of one slice disagree")
        keys[key] = item.version_id
        unique[key] = item
    by_address = {}
    for item in unique.values():
        by_address.setdefault(item.address.key, []).append(item)
    for siblings in by_address.values():
        if any(
            recovery._rectangles_overlap(left, right)
            for index, left in enumerate(siblings) for right in siblings[index + 1:]
        ) or sum(recovery._volume(item.lengths) for item in siblings) != recovery._volume(
            siblings[0].address.global_shape
        ):
            raise ValueError("ZeRO-1 descriptor catalog has overlapping or missing slices")
    if len({(item.owner_rank, item.key) for item in opaque}) != len(opaque):
        raise ValueError("ZeRO-1 descriptor catalog has duplicate rank-local roots")
    return recovery._digest({
        "epoch": epoch,
        "fragments": sorted({
            (item.address.key, item.epoch, item.offsets, item.lengths,
             item.shard_id, item.version_id, item.checksum,
             item.base_epoch, item.dependency_epoch)
            for item in fragments
        }),
        "opaque": [
            (item.owner_rank, item.key, item.epoch, item.checksum)
            for item in sorted(opaque, key=lambda item: (item.owner_rank, item.key))
        ],
    })


def _fragment_bytes(item: ObjectFragment) -> int:
    if not item.address.dtype.startswith("torch."):
        raise ValueError("ZeRO-1 descriptor has an unsupported dtype")
    size = item.tensor.element_size() if item.tensor is not None else torch.empty(
        (), dtype=getattr(torch, item.address.dtype.removeprefix("torch."))
    ).element_size()
    return recovery._volume(item.lengths) * size


def _descriptor(item: ObjectFragment) -> ObjectFragment:
    """Remove tensors from the control plane without serializing their bytes."""
    return replace(item, tensor=None)


def _zero1_committed_inventory(
    clients: Mapping[int, ReplicaServiceClient], *, epoch: int, arm: str,
    rank_hosts: Mapping[int, int], planned: Sequence[Sequence[ObjectFragment]],
    opaque_metadata: Sequence[OpaqueRecord], evidence: Sequence[Sequence[Mapping[str, Any]]],
    host_hashes: Sequence[str], control_checksum: str,
) -> dict[str, Any]:
    hosts = []
    publisher_logical_bytes: dict[str, int] = {}
    publisher_payload_bytes: dict[str, int] = {}
    role_bytes = {name: 0 for name in ("model", "optimizer", "runtime")}
    planned_sets = [{
        (item.shard_id, item.version_id, item.source_rank)
        for item in planned[host]
    } for host in (0, 1)]
    for host in (0, 1):
        fragments = []
        opaque = []
        bundles = []
        remote_logical_bytes = 0
        remote_payload_bytes = 0
        for source_rank, source_host in sorted(rank_hosts.items()):
            expected = evidence[host][source_rank]
            envelope, _ = clients[host].get(
                source_rank, epoch, control_checksum=control_checksum,
                expected_manifest=expected,
            )
            bundle = deserialize_checkpoint(
                envelope.payload, envelope.manifest.payload_checksum
            )
            if (
                bundle.get("kind") != "zero1_object_bundle"
                or bundle.get("epoch") != epoch
                or bundle.get("source_rank") != source_rank
                or bundle.get("source_host") != source_host
            ):
                raise ValueError("committed ZeRO-1 bundle identity differs from its sealed evidence")
            bundle_fragments = tuple(bundle.get("fragments", ()))
            bundle_opaque = tuple(bundle.get("opaque", ()))
            bundle_logical_bytes = sum(
                item.tensor.numel() * item.tensor.element_size()
                for item in bundle_fragments
            ) + sum(
                obj.logical_bytes
                for item in bundle_opaque
                for obj in iter_checkpoint_objects(item.value)
            )
            if any(
                not isinstance(item, ObjectFragment)
                or item.source_rank != source_rank or item.storage_host != host
                for item in bundle_fragments
            ) or any(
                not isinstance(item, OpaqueRecord)
                or item.owner_rank != source_rank or host not in item.storage_hosts
                for item in bundle_opaque
            ):
                raise ValueError("committed ZeRO-1 bundle has invalid object ownership")
            if {item.key for item in bundle_opaque} != {
                "scheduler", "rng", "sampler", "progress"
            }:
                raise ValueError("committed ZeRO-1 bundle lacks runtime roots")
            expected_items = tuple(
                item for item in planned[host] if item.source_rank == source_rank
            )
            identity = lambda item: (
                item.address.key, item.epoch, item.offsets, item.lengths,
                item.source_rank, item.storage_host, item.checksum, item.shard_id,
                item.version_id, item.base_epoch, item.dependency_epoch,
            )
            if sorted(map(identity, bundle_fragments), key=repr) != sorted(
                map(identity, expected_items), key=repr
            ):
                raise ValueError("planned and committed ZeRO-1 object sets differ")
            for item in bundle_fragments:
                item.validate()
                logical_bytes = item.tensor.numel() * item.tensor.element_size()
                role = "model" if item.address.role.startswith("model") else "optimizer"
                role_bytes[role] += logical_bytes
                publisher_logical_bytes[str(source_rank)] = (
                    publisher_logical_bytes.get(str(source_rank), 0) + logical_bytes
                )
                fragments.append(item)
            for item in bundle_opaque:
                item.validate()
                logical_bytes = sum(
                    obj.logical_bytes for obj in iter_checkpoint_objects(item.value)
                )
                role_bytes["runtime"] += logical_bytes
                publisher_logical_bytes[str(source_rank)] = (
                    publisher_logical_bytes.get(str(source_rank), 0) + logical_bytes
                )
                opaque.append(item)
            publisher_payload_bytes[str(source_rank)] = (
                publisher_payload_bytes.get(str(source_rank), 0)
                + envelope.manifest.payload_bytes
            )
            if source_host != host:
                remote_logical_bytes += bundle_logical_bytes
                remote_payload_bytes += envelope.manifest.payload_bytes
            bundles.append({
                "source_rank": source_rank,
                "manifest_id": envelope.manifest.manifest_id,
                "payload_checksum": envelope.manifest.payload_checksum,
                "payload_bytes": envelope.manifest.payload_bytes,
                "manifest_logical_bytes": envelope.manifest.logical_bytes,
                "logical_bytes": bundle_logical_bytes,
                "fragments": [
                    {
                        "semantic_id": item.address.semantic_id,
                        "role": item.address.role,
                        "shape": list(item.address.global_shape),
                        "dtype": item.address.dtype,
                        "offsets": list(item.offsets),
                        "lengths": list(item.lengths),
                        "source_rank": item.source_rank,
                        "storage_host": item.storage_host,
                        "logical_bytes": item.tensor.numel() * item.tensor.element_size(),
                        "checksum": item.checksum,
                        "shard_id": item.shard_id,
                        "version_id": item.version_id,
                    }
                    for item in bundle_fragments
                ],
                "opaque": [
                    {
                        "owner_rank": item.owner_rank,
                        "key": item.key,
                        "checksum": item.checksum,
                        "logical_bytes": sum(
                            obj.logical_bytes for obj in iter_checkpoint_objects(item.value)
                        ),
                    }
                    for item in bundle_opaque
                ],
            })
        catalog = ObjectCatalog(epoch, tuple(fragments), tuple(opaque))
        catalog.validate()
        content_hash = catalog.content_hash
        if content_hash != host_hashes[host]:
            raise ValueError("committed ZeRO-1 catalog hash differs from planned host hash")
        hosts.append({
            "host": host,
            "content_hash": content_hash,
            "bundles": bundles,
            "fragment_count": len(fragments),
            "opaque_count": len(opaque),
            "remote_logical_bytes": remote_logical_bytes,
            "remote_payload_bytes": remote_payload_bytes,
        })
    return {
        "schema_version": 1,
        "epoch": epoch,
        "arm": arm,
        "planned_equals_committed": True,
        "host_catalog_hashes_match": True,
        "hosts": hosts,
        "planned_shard_counts": [len(value) for value in planned_sets],
        "role_logical_bytes_per_host": role_bytes,
        "remote_data_logical_bytes": sum(item["remote_logical_bytes"] for item in hosts),
        "remote_data_payload_bytes": sum(item["remote_payload_bytes"] for item in hosts),
        "publisher_logical_bytes": publisher_logical_bytes,
        "maximum_publisher_logical_bytes": max(publisher_logical_bytes.values()),
        "publisher_payload_bytes": publisher_payload_bytes,
        "maximum_publisher_payload_bytes": max(publisher_payload_bytes.values()),
    }


def _validate_zero1_model_aliases(
    model_aliases: Sequence[Mapping[str, str]],
    fragments: Sequence[ObjectFragment],
    rank_hosts: Mapping[int, int],
) -> tuple[Mapping[str, str], ...]:
    if len(model_aliases) != len(rank_hosts) or any(
        not isinstance(row, Mapping) for row in model_aliases
    ):
        raise ValueError("ZeRO-1 model alias manifest is incomplete")
    aliases_by_canonical: dict[str, list[tuple[int, str]]] = {}
    for rank, row in enumerate(model_aliases):
        if len(set(row.values())) != len(row) or any(
            not isinstance(name, str) or not name
            or not isinstance(canonical, str)
            or not canonical.startswith("moe/model/")
            for name, canonical in row.items()
        ):
            raise ValueError("ZeRO-1 rank has an invalid or duplicate model alias")
        for name, canonical in row.items():
            aliases_by_canonical.setdefault(canonical, []).append((rank, name))

    fragments_by_owner_name = {}
    for rank, row in enumerate(model_aliases):
        for name, canonical in row.items():
            matches = [
                item for item in fragments
                if item.source_rank == rank
                and item.address.semantic_id == f"{canonical}::model"
                and item.address.role == "model"
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"MoE alias must bind one model fragment per rank/FQN: rank={rank}, name={name!r}"
                )
            fragments_by_owner_name[rank, name] = matches[0]

    for canonical, owners in aliases_by_canonical.items():
        owner_ranks = {rank for rank, _ in owners}
        if len(owner_ranks) < 2:
            raise ValueError(f"MoE canonical model alias lacks replicas: {canonical}")
        rows = [fragments_by_owner_name[rank, name] for rank, name in owners]
        if len({(item.address.global_shape, item.address.dtype, item.checksum, item.version_id)
                for item in rows}) != 1:
            raise ValueError(f"MoE canonical model replicas conflict: {canonical}")
    aliased_fragments = {
        (rank, fragments_by_owner_name[rank, name].shard_id)
        for rank, row in enumerate(model_aliases) for name in row
    }
    if len(aliased_fragments) != sum(map(len, model_aliases)):
        raise ValueError("ZeRO-1 model alias FQNs do not bind distinct source fragments")
    return tuple(dict(row) for row in model_aliases)


def select_zero1_publications(
    fragments: Sequence[ObjectFragment],
    *,
    rank_hosts: Mapping[int, int],
    arm: str,
) -> tuple[tuple[ObjectFragment, ...], tuple[ObjectFragment, ...]]:
    """Choose per-host publishers; never merge unequal versions or missing shards."""
    if arm not in {"adacheck_fixed", "host_global_fixed", "m1"}:
        raise ValueError("unsupported ZeRO-1 placement arm")
    if set(rank_hosts.values()) != {0, 1} or any(
        item.source_rank not in rank_hosts
        or item.storage_host != rank_hosts[item.source_rank]
        for item in fragments
    ):
        raise ValueError("ZeRO-1 placement lacks a complete source mapping")
    chosen: list[list[ObjectFragment]] = [[], []]
    loads = {rank: 0 for rank in rank_hosts}
    resident = {host: tuple(sorted(rank for rank, value in rank_hosts.items() if value == host)) for host in (0, 1)}
    fixed_pairs = tuple(zip(resident[0], resident[1]))
    if arm == "adacheck_fixed" and len(resident[0]) != len(resident[1]):
        raise ValueError("fixed ZeRO-1 placement requires one cross-host peer per rank")
    classes: dict[tuple[Any, ...], list[ObjectFragment]] = {}
    for item in fragments:
        classes.setdefault((item.address.key, item.shard_id, item.version_id), []).append(item)
    for _, holders in sorted(classes.items(), key=lambda pair: (
        -_fragment_bytes(pair[1][0]), str(pair[0])
    )):
        for host in (0, 1):
            candidates = holders
            if arm == "adacheck_fixed":
                # Each disjoint cross-host fixed pair protects its own local copy.
                # Redundancy shared across pairs cannot be removed by this policy.
                for pair in fixed_pairs:
                    available = [item for item in candidates if item.source_rank in pair]
                    if available:
                        local = [item for item in available if rank_hosts[item.source_rank] == host]
                        source = min(local or available, key=lambda item: item.source_rank)
                        chosen[host].append(replace(source, storage_host=host))
                if not any(
                    item.shard_id == holders[0].shard_id
                    and item.version_id == holders[0].version_id
                    for item in chosen[host]
                ):
                    raise ValueError("fixed ZeRO-1 pairs cannot cover a required object")
                continue
            if arm == "host_global_fixed":
                local = [item for item in candidates if rank_hosts[item.source_rank] == host]
                source = min(local or candidates, key=lambda item: item.source_rank)
            else:
                local = [item for item in candidates if rank_hosts[item.source_rank] == host]
                source = min(local or candidates, key=lambda item: (loads[item.source_rank], item.source_rank))
                loads[source.source_rank] += _fragment_bytes(source)
            chosen[host].append(replace(source, storage_host=host))
    return tuple(chosen[0]), tuple(chosen[1])


def publish_distributed_zero1_epoch(
    clients: Mapping[int, ReplicaServiceClient], *, epoch: int,
    fragments: Sequence[ObjectFragment] = (), opaque: Sequence[OpaqueRecord] = (),
    rank_hosts: Mapping[int, int], arm: str, group: object,
    rank_metadata: Mapping[str, Any] | None = None,
    detector_audit: Mapping[str, Any] | None = None,
    model_aliases: Mapping[str, str] | None = None,
    captured_snapshot: list | None = None,
) -> SealedZero1Epoch:
    """Plan with small descriptors; upload objects from their owning ranks."""
    if captured_snapshot is not None:
        if fragments or opaque:
            raise ValueError("provide either an owned snapshot or fragment/opaque sequences")
        fragments, opaque = captured_snapshot
        captured_snapshot.clear()
    if (group is None or not dist.is_initialized() or dist.get_backend(group) != "gloo"
        or dist.get_world_size(group) != len(rank_hosts)
        or tuple(sorted(rank_hosts)) != tuple(range(len(rank_hosts)))
        or set(rank_hosts.values()) != {0, 1} or type(epoch) is not int
        or epoch < 0):
        raise ValueError("distributed ZeRO-1 publication needs a world-size Gloo group")
    rank = dist.get_rank(group)
    if rank != dist.get_rank() or rank not in rank_hosts or set(clients) != {0, 1}:
        raise ValueError("ZeRO-1 publication rank or service layout differs")
    local = tuple(fragments)
    roots = tuple(opaque)
    if any(
        item.source_rank != rank or item.tensor is None or item.tensor.device.type != "cpu"
        or item.epoch != epoch for item in local
    ):
        raise ValueError("ZeRO-1 rank published a foreign or incomplete fragment")
    if len(roots) != 4 or {item.key for item in roots} != {"scheduler", "rng", "sampler", "progress"} or any(
        item.owner_rank != rank or item.epoch != epoch or item.storage_hosts != (0, 1)
        for item in roots
    ):
        raise ValueError("ZeRO-1 rank lacks its four local training roots")
    progress = next(item.value for item in roots if item.key == "progress")
    if "detector_audit" in progress:
        raise ValueError("detector audit metadata must remain outside rank restore state")
    if not local or any(item.storage_host != rank_hosts[rank] for item in local):
        raise ValueError("ZeRO-1 rank lacks complete local CPU tensors")
    local_rank_metadata = dict(rank_metadata or {})
    local_model_aliases = dict(model_aliases or {})
    if any(
        not isinstance(name, str) or not name or not isinstance(value, str)
        or not value.startswith("moe/model/")
        for name, value in local_model_aliases.items()
    ):
        raise ValueError("MoE model alias manifest is invalid")
    local_detector_audit = dict(detector_audit or {})
    detector_audit_enabled = bool(local_detector_audit)
    if detector_audit_enabled and (
        local_detector_audit.get("placement_applied") is not False
        or local_detector_audit.get("schema_version") != 1
        or not local_detector_audit.get("evidence_sha256")
        or local_detector_audit.get("coverage")
        != "positive detector proposals only; singleton negatives are not exported"
    ):
        raise ValueError("MoE detector audit metadata is invalid")
    if (
        not local_rank_metadata
        or local_rank_metadata.get("rank") != rank
        or local_rank_metadata.get("host") != rank_hosts[rank]
        or not local_rank_metadata.get("layout_hash")
        or not local_rank_metadata.get("topology_hash")
        or not isinstance(local_rank_metadata.get("coordinates"), Mapping)
        or not local_rank_metadata.get("groups")
        or "expert_ids" not in local_rank_metadata
    ):
        raise ValueError("ZeRO-1 rank topology metadata is incomplete")
    descriptors: list[Any] = [None] * len(rank_hosts)
    dist.all_gather_object(
        descriptors,
        (
            tuple(_descriptor(item) for item in local),
            tuple(replace(item, value=None) for item in roots),
            local_rank_metadata,
            local_detector_audit,
            local_model_aliases,
        ),
        group=group,
    )
    if any(not isinstance(row, tuple) or len(row) != 5 for row in descriptors):
        raise ValueError("ZeRO-1 rank metadata is incomplete")
    metadata = tuple(item for row in descriptors for item in row[0])
    opaque_metadata = tuple(item for row in descriptors for item in row[1])
    rank_metadata_rows = tuple(row[2] for row in descriptors)
    detector_audit_rows = tuple(row[3] for row in descriptors)
    model_alias_rows = tuple(row[4] for row in descriptors)
    model_alias_rows = _validate_zero1_model_aliases(
        model_alias_rows, metadata, rank_hosts
    )
    detector_audit_report = None
    if detector_audit_enabled:
        if any(not isinstance(row, Mapping) or not row for row in detector_audit_rows):
            raise ValueError("ZeRO-1 ranks disagree on detector audit enablement")
        if any(dict(row) != dict(detector_audit_rows[0]) for row in detector_audit_rows[1:]):
            raise ValueError("MoE detector audit differs across ranks")
        detector_audit_report = dict(detector_audit_rows[0])
    elif any(detector_audit_rows):
        raise ValueError("ZeRO-1 ranks disagree on detector audit enablement")
    if any(
        not isinstance(row, Mapping)
        or row.get("rank") != rank
        or row.get("host") != rank_hosts[rank]
        or not row.get("layout_hash")
        or not row.get("topology_hash")
        or not isinstance(row.get("coordinates"), Mapping)
        or not isinstance(row.get("groups"), Mapping)
        or set(row["groups"]) != {"shared", "expert", "dispatch"}
        or not isinstance(row.get("optimizer_roles"), (tuple, list))
        or "expert_ids" not in row
        for rank, row in enumerate(rank_metadata_rows)
    ):
        raise ValueError("ZeRO-1 source topology manifest is incomplete")
    if {item.source_rank for item in metadata} != set(rank_hosts) or {
        item.owner_rank for item in opaque_metadata
    } != set(rank_hosts):
        raise ValueError("distributed ZeRO-1 catalog lacks rank coverage")
    # The catalog's semantic hash does not need tensor/opaque values; the
    # individual committed bundles retain checksummed data for restore.
    if any(
        item.source_rank != source_rank or item.storage_host != rank_hosts[source_rank]
        for source_rank, row in enumerate(descriptors) for item in row[0]
    ) or any(
        item.owner_rank != source_rank or item.storage_hosts != (0, 1)
        for source_rank, row in enumerate(descriptors) for item in row[1]
    ):
        raise ValueError("ZeRO-1 gathered descriptors disagree with source ranks")
    source_hash = _descriptor_catalog_hash(epoch, metadata, opaque_metadata)
    planned = select_zero1_publications(metadata, rank_hosts=rank_hosts, arm=arm)
    host_hashes = tuple(_descriptor_catalog_hash(epoch, planned[host], opaque_metadata) for host in (0, 1))
    for host in (0, 1):
        if {(item.shard_id, item.version_id) for item in planned[host]} != {
            (item.shard_id, item.version_id) for item in metadata
        }:
            raise ValueError("ZeRO-1 planner breaks surviving host coverage")
    ranks = tuple(range(len(rank_hosts)))
    targets = {host: min(r for r in ranks if rank_hosts[r] == host) for host in (0, 1)}
    local_evidence = None
    bundle = None
    schedule = os.environ.get("CKPT_ZERO1_SERIAL_PUBLISH", "0")
    serial_publish = schedule in {"1", "host"}
    position, turns = rank, len(ranks)
    if schedule == "host":
        position = sorted(r for r in ranks if rank_hosts[r] == rank_hosts[rank]).index(rank)
        turns = max(sum(value == host for value in rank_hosts.values()) for host in (0, 1))
    # Each barrier advances a turn; host scheduling admits one publisher per host.
    if serial_publish:
        for _ in range(position):
            dist.barrier(group=group)
    try:
        local_evidence = []
        for host in (0, 1):
            selected_ids = {
                (item.shard_id, item.version_id)
                for item in planned[host] if item.source_rank == rank
            }
            bundle = {
                "kind": "zero1_object_bundle", "epoch": epoch, "source_rank": rank,
                "source_host": rank_hosts[rank],
                "fragments": tuple(
                    replace(item, storage_host=host) for item in local
                    if (item.shard_id, item.version_id) in selected_ids
                ),
                "opaque": roots,
            }
            manifest, wire_bytes, put_ms, edge_ms, logical_bytes, payload_bytes = _commit_zero1_bundle(
                clients[host], bundle, source_rank=rank, source_host=rank_hosts[rank],
                target_host=host, target_rank=targets[host], group_ranks=ranks,
            )
            local_evidence.append((
                manifest_evidence(manifest), manifest.payload_bytes,
                wire_bytes, put_ms, edge_ms, logical_bytes, payload_bytes,
            ))
    except Exception as error:
        local_evidence = f"rank {rank}: {type(error).__name__}: {error}"
    # The trainer transfers ownership, so completed publishers can release CPU payloads.
    del bundle, fragments, opaque, local, roots
    if serial_publish:
        for _ in range(turns - position):
            dist.barrier(group=group)
    gathered: list[Any] = [None] * len(rank_hosts)
    dist.all_gather_object(gathered, local_evidence, group=group)
    failures = [
        item for item in gathered
        if not isinstance(item, list) or len(item) != 2
        or any(not isinstance(row, tuple) or len(row) != 7 for row in item)
    ]
    if failures:
        raise RuntimeError(f"distributed ZeRO-1 upload failed: {failures[0]}")
    evidence = tuple(tuple(row[host][0] for row in gathered) for host in (0, 1))
    remote_bytes = sum(
        gathered[source][host][1]
        for source in ranks for host in (0, 1) if host != rank_hosts[source]
    )
    remote_edge_payload_bytes = sum(
        gathered[source][host][6]
        for source in ranks for host in (0, 1) if host != rank_hosts[source]
    )
    remote_logical_bytes = sum(
        gathered[source][host][5]
        for source in ranks for host in (0, 1) if host != rank_hosts[source]
    )
    remote_wire_bytes = sum(
        gathered[source][host][2]
        for source in ranks for host in (0, 1) if host != rank_hosts[source]
    )
    max_put = max(
        gathered[source][host][3]
        for source in ranks for host in (0, 1) if host != rank_hosts[source]
    )
    max_edge_ack = max(
        gathered[source][host][4]
        for source in ranks for host in (0, 1) if host != rank_hosts[source]
    )
    remote_control_payload_bytes = 0
    remote_control_wire_bytes = 0
    result = None
    observation = None
    if rank == 0:
        try:
            control = {
                "kind": "zero1_object_control", "epoch": epoch, "arm": arm,
                "rank_hosts": tuple(sorted(rank_hosts.items())),
                "rank_metadata": rank_metadata_rows,
                "detector_audit": detector_audit_report,
                "model_aliases": model_alias_rows,
                "model_aliases_applied": any(model_alias_rows),
                "source_content_hash": source_hash,
                "host_content_hashes": host_hashes,
                "data_evidence": evidence,
            }
            controls = []
            for host in (0, 1):
                manifest, wire_bytes, put_ms, edge_ms, logical_bytes, payload_bytes = _commit_zero1_bundle(
                    clients[host], control, source_rank=CONTROL_SOURCE_RANK,
                    source_host=0, target_host=host, target_rank=targets[host],
                    group_ranks=ranks,
                )
                controls.append(manifest)
                if host == 1:
                    remote_bytes += manifest.payload_bytes
                    remote_wire_bytes += wire_bytes
                    remote_logical_bytes += logical_bytes
                    remote_control_logical_bytes = logical_bytes
                    max_put = max(max_put, put_ms)
                    remote_control_wire_bytes = wire_bytes
                    remote_control_payload_bytes = payload_bytes
            checksum = controls[0].payload_checksum
            if controls[1].payload_checksum != checksum:
                raise ValueError("ZeRO-1 services disagree on epoch control")
            controls_evidence = tuple(manifest_evidence(item) for item in controls)
            for host in (0, 1):
                seal_object_update_host(
                    clients[host], epoch=epoch, control_checksum=checksum,
                    control_evidence=controls_evidence[host],
                    bundle_evidence=evidence[host],
                )
            inventory = None
            if os.environ.get("CKPT_ZERO1_EXPORT_INVENTORY") == "1":
                inventory = _zero1_committed_inventory(
                    clients, epoch=epoch, arm=arm, rank_hosts=rank_hosts,
                    planned=planned, opaque_metadata=opaque_metadata,
                    evidence=evidence, host_hashes=host_hashes,
                    control_checksum=checksum,
                )
            result = (True, SealedZero1Epoch(
                epoch, checksum, controls_evidence, evidence, remote_bytes, max_put, arm,
                detector_audit_report, model_alias_rows, any(model_alias_rows),
            ), Zero1PublicationObservation(
                epoch=epoch,
                remote_payload_bytes=remote_bytes,
                remote_wire_bytes=remote_wire_bytes,
                remote_logical_bytes=remote_logical_bytes,
                remote_edge_payload_bytes=remote_edge_payload_bytes,
                remote_control_payload_bytes=remote_control_payload_bytes,
                remote_control_wire_bytes=remote_control_wire_bytes,
                remote_control_logical_bytes=remote_control_logical_bytes,
                maximum_remote_put_ms=max_put,
                maximum_remote_edge_ack_ms=max_edge_ack,
                committed_inventory=inventory,
            ))
        except Exception as error:
            result = (False, f"{type(error).__name__}: {error}", None)
    broadcast = [result]
    dist.broadcast_object_list(broadcast, src=0, group=group)
    success, value, observation = broadcast[0]
    if not success:
        raise RuntimeError(f"distributed ZeRO-1 epoch could not be sealed: {value}")
    object.__setattr__(value, "publication_observation", observation)
    return value


def publish_zero1_epoch(
    clients: Mapping[int, ReplicaServiceClient],
    *,
    epoch: int,
    fragments: Sequence[ObjectFragment],
    opaque: Sequence[OpaqueRecord],
    rank_hosts: Mapping[int, int],
    arm: str = "adacheck_fixed",
    rank_metadata: Sequence[Mapping[str, Any]] | None = None,
    detector_audit: Mapping[str, Any] | None = None,
    model_aliases: Sequence[Mapping[str, str]] | None = None,
) -> SealedZero1Epoch:
    """Commit a complete materialized epoch; advertise only after both seals."""
    if set(clients) != {0, 1} or not rank_hosts or set(rank_hosts.values()) != {0, 1}:
        raise ValueError("ZeRO-1 publication requires exactly two populated hosts")
    if type(epoch) is not int or epoch < 0:
        raise ValueError("ZeRO-1 publication requires a non-negative integral epoch")
    ranks = tuple(sorted(rank_hosts))
    if ranks != tuple(range(len(ranks))) or any(
        type(host) is not int for host in rank_hosts.values()
    ):
        raise ValueError("ZeRO-1 publication requires contiguous logical ranks")
    source = ObjectCatalog(epoch, tuple(fragments), tuple(opaque))
    source.validate()
    if {item.source_rank for item in fragments} != set(ranks) or {
        item.owner_rank for item in opaque
    } != set(ranks) or any(
        item.storage_host != rank_hosts[item.source_rank] for item in fragments
    ) or any(item.storage_hosts != (0, 1) for item in opaque) or any(
        item.tensor is None or item.tensor.device.type != "cpu"
        for item in fragments
    ):
        raise ValueError("ZeRO-1 publication has incomplete or misplaced rank state")
    if detector_audit is not None and (
        detector_audit.get("placement_applied") is not False
        or detector_audit.get("schema_version") != 1
        or not detector_audit.get("evidence_sha256")
        or detector_audit.get("coverage")
        != "positive detector proposals only; singleton negatives are not exported"
    ):
        raise ValueError("MoE detector audit metadata is invalid")
    aliases_by_rank = _validate_zero1_model_aliases(
        tuple(model_aliases or ({} for _ in ranks)), fragments, rank_hosts
    )
    selected = select_zero1_publications(fragments, rank_hosts=rank_hosts, arm=arm)
    for host in (0, 1):
        stored = ObjectCatalog(epoch, selected[host], tuple(
            replace(item, storage_hosts=(host,)) for item in opaque
        ))
        stored.validate()
        if {(item.shard_id, item.version_id) for item in stored.fragments} != {
            (item.shard_id, item.version_id) for item in source.fragments
        }:
            raise ValueError("ZeRO-1 placement breaks a host's recovery coverage")
    target_ranks = {host: min(rank for rank in ranks if rank_hosts[rank] == host) for host in clients}
    manifests: dict[int, list[Any]] = {0: [], 1: []}
    remote_bytes = 0
    maximum_put_ms = 0.0
    for rank in ranks:
        owned_by_host = tuple(
            tuple(item for item in selected[host] if item.source_rank == rank)
            for host in (0, 1)
        )
        roots = tuple(item for item in opaque if item.owner_rank == rank)
        if {item.key for item in roots} != {"scheduler", "rng", "sampler", "progress"}:
            raise ValueError("ZeRO-1 rank lacks complete opaque roots")
        for host in (0, 1):
            payload = {
                "kind": "zero1_object_bundle", "epoch": epoch, "source_rank": rank,
                "source_host": rank_hosts[rank],
                "fragments": owned_by_host[host],
                "opaque": roots,
            }
            manifest, _, put_ms = commit_object_update_state(
                clients[host], payload,
                source_rank=rank, source_host=rank_hosts[rank], target_host=host,
                target_rank=target_ranks[host], group_ranks=ranks,
            )
            manifests[host].append(manifest)
            if host != rank_hosts[rank]:
                remote_bytes += manifest.payload_bytes
                maximum_put_ms = max(maximum_put_ms, put_ms)
    data_evidence = tuple(
        tuple(manifest_evidence(item) for item in manifests[host]) for host in (0, 1)
    )
    control = {
        "kind": "zero1_object_control", "epoch": epoch,
        "arm": arm,
        "detector_audit": None if detector_audit is None else dict(detector_audit),
        "rank_hosts": tuple(sorted(rank_hosts.items())),
        "rank_metadata": tuple(dict(item) for item in (rank_metadata or ())),
        "source_content_hash": source.content_hash,
        "model_aliases": aliases_by_rank,
        "model_aliases_applied": any(aliases_by_rank),
        "host_content_hashes": tuple(
            ObjectCatalog(epoch, selected[host], tuple(
                replace(item, storage_hosts=(host,)) for item in opaque
            )).content_hash for host in (0, 1)
        ),
        "data_evidence": data_evidence,
    }
    controls = []
    for host in (0, 1):
        manifest, _, put_ms = commit_object_update_state(
            clients[host], control, source_rank=CONTROL_SOURCE_RANK,
            source_host=0, target_host=host, target_rank=target_ranks[host],
            group_ranks=ranks,
        )
        controls.append(manifest)
        if host != 0:
            remote_bytes += manifest.payload_bytes
            maximum_put_ms = max(maximum_put_ms, put_ms)
    checksum = controls[0].payload_checksum
    if controls[1].payload_checksum != checksum:
        raise ValueError("ZeRO-1 services disagree on epoch control")
    control_evidence = tuple(manifest_evidence(item) for item in controls)
    for host in (0, 1):
        seal_object_update_host(
            clients[host], epoch=epoch, control_checksum=checksum,
            control_evidence=control_evidence[host],
            bundle_evidence=data_evidence[host],
        )
    return SealedZero1Epoch(
        epoch, checksum, control_evidence, data_evidence, remote_bytes,
        maximum_put_ms, arm, detector_audit, aliases_by_rank, any(aliases_by_rank),
    )


def restore_zero1_from_host(
    client: ReplicaServiceClient,
    *,
    host: int,
    sealed: SealedZero1Epoch,
    rank: int,
    layout: TargetLayout,
    arm: str = "m1_object",
    expected_rank_metadata: Mapping[str, Any] | None = None,
    expected_rank_metadata_rows: Sequence[Mapping[str, Any]] | None = None,
    target_only: bool = False,
) -> RestoredRankState:
    """Read only a sealed surviving service and check full object coverage."""
    if host not in (0, 1) or rank not in range(len(sealed.data_evidence[host])):
        raise ValueError("ZeRO-1 restore host or rank is outside the sealed epoch")
    envelope, _ = client.get(
        CONTROL_SOURCE_RANK, sealed.epoch,
        control_checksum=sealed.control_checksum,
        expected_manifest=sealed.control_evidence[host],
    )
    if manifest_evidence(envelope.manifest) != sealed.control_evidence[host]:
        raise ValueError("ZeRO-1 control manifest differs from sealed evidence")
    control = deserialize_checkpoint(envelope.payload, envelope.manifest.payload_checksum)
    validate_checkpoint_objects(control, envelope.manifest.objects)
    if (
        control.get("kind") != "zero1_object_control"
        or control.get("epoch") != sealed.epoch
        or control.get("data_evidence") != sealed.data_evidence
        or control.get("arm") != sealed.arm
        or control.get("detector_audit") != sealed.detector_audit
        or tuple(control.get("model_aliases", ())) != sealed.model_aliases
        or control.get("model_aliases_applied") is not bool(
            sealed.model_aliases and any(sealed.model_aliases)
        )
        or not isinstance(control.get("rank_hosts"), (tuple, list))
        or set(dict(control["rank_hosts"]).values()) != {0, 1}
        or tuple(item[0] for item in control.get("rank_hosts", ()))
        != tuple(range(len(sealed.data_evidence[host])))
    ):
        raise ValueError("ZeRO-1 sealed control is incomplete or differs from locator")
    rank_metadata = control.get("rank_metadata")
    if (
        not isinstance(rank_metadata, (tuple, list))
        or len(rank_metadata) != len(sealed.data_evidence[host])
        or any(
            not isinstance(row, Mapping)
            or row.get("rank") != source_rank
            or row.get("host") != dict(control["rank_hosts"])[source_rank]
            or not row.get("layout_hash")
            or not row.get("topology_hash")
            for source_rank, row in enumerate(rank_metadata)
        )
    ):
        raise ValueError("ZeRO-1 sealed source topology manifest is incomplete")
    if expected_rank_metadata is not None:
        if dict(rank_metadata[rank]) != dict(expected_rank_metadata):
            raise ValueError("ZeRO-1 target rank topology differs from its sealed source")
    if expected_rank_metadata_rows is not None:
        if (
            len(expected_rank_metadata_rows) != len(rank_metadata)
            or tuple(dict(row) for row in expected_rank_metadata_rows)
            != tuple(dict(row) for row in rank_metadata)
        ):
            raise ValueError("ZeRO-1 target topology differs from its sealed source manifest")
    fragments = []
    opaque = []
    requested_addresses = {request.source.key for request in layout.slices}
    requested_roots = {(request.source_rank, request.key) for request in layout.opaque}
    source_ranks = list(range(len(sealed.data_evidence[host])))
    if target_only:
        coordinates = rank_metadata[rank].get("coordinates", {})
        # Pooled model objects usually live with a peer in the same PP/TP stage.
        # Keep every other source as a fallback for layouts with wider dependencies.
        source_ranks.sort(key=lambda source: (
            source != rank,
            any(rank_metadata[source].get("coordinates", {}).get(axis)
                != coordinates.get(axis) for axis in ("pp", "tp")),
            source,
        ))
    data_payload_bytes = 0
    bundles_read = 0
    for source_rank in source_ranks:
        evidence = sealed.data_evidence[host][source_rank]
        envelope, _ = client.get(
            source_rank, sealed.epoch,
            control_checksum=sealed.control_checksum,
            expected_manifest=evidence,
        )
        if manifest_evidence(envelope.manifest) != evidence:
            raise ValueError("ZeRO-1 data manifest differs from sealed evidence")
        data_payload_bytes += len(envelope.payload)
        bundles_read += 1
        bundle = deserialize_checkpoint(envelope.payload, envelope.manifest.payload_checksum)
        validate_checkpoint_objects(bundle, envelope.manifest.objects)
        if (
            bundle.get("kind") != "zero1_object_bundle"
            or bundle.get("epoch") != sealed.epoch
            or bundle.get("source_rank") != source_rank
            or bundle.get("source_host") != dict(control["rank_hosts"])[source_rank]
        ):
            raise ValueError("ZeRO-1 bundle disagrees with sealed control")
        if not isinstance(bundle.get("fragments"), (tuple, list)) or not isinstance(
            bundle.get("opaque"), (tuple, list)
        ):
            raise ValueError("ZeRO-1 bundle lacks object lists")
        for item in bundle["fragments"]:
            if not isinstance(item, ObjectFragment) or item.source_rank != source_rank or item.storage_host != host:
                raise ValueError("ZeRO-1 fragment has an invalid source or host")
            # Check all payloads, but retain only tensors needed by this target layout.
            item.validate()
            if item.address.key in requested_addresses:
                fragments.append(item)
            elif not target_only:
                fragments.append(_descriptor(item))
        for item in bundle["opaque"]:
            if not isinstance(item, OpaqueRecord) or item.owner_rank != source_rank or host not in item.storage_hosts:
                raise ValueError("ZeRO-1 opaque root has an invalid source or host")
            if not target_only or (item.owner_rank, item.key) in requested_roots:
                opaque.append(replace(item, storage_hosts=(host,)))
        del bundle, envelope
        if target_only and requested_addresses <= {
            item.address.key for item in fragments
            if item.offsets == (0,) * len(item.lengths)
            and item.lengths == item.address.global_shape
        } and requested_roots <= {(item.owner_rank, item.key) for item in opaque}:
            break
    catalog = ObjectCatalog(sealed.epoch, tuple(fragments), tuple(opaque))
    planning_catalog = ObjectCatalog(
        sealed.epoch, tuple(_descriptor(item) for item in fragments), tuple(opaque)
    )
    content_hash = planning_catalog.content_hash
    if not target_only and content_hash != control["host_content_hashes"][host]:
        raise ValueError("ZeRO-1 surviving service has a different content catalog")
    runtime = ObjectRuntimeIdentity(
        content_hash, "zero1_service_snapshot", "zero1_full_cpu",
        "zero1_source_layout", layout.layout_hash, sealed.epoch,
        ((),),
    )
    plan = build_object_recovery_plan(planning_catalog, layout, arm=arm, runtime=runtime)
    restored = restore_object_layout(catalog, plan, layout, failed_hosts=())[rank]
    if target_only:
        print(f"ZERO1_TARGET_RESTORE rank={rank} epoch={sealed.epoch} "
              f"bundles={bundles_read} data_payload_bytes={data_payload_bytes}", flush=True)
    return restored
