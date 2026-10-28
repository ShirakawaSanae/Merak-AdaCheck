"""DP-deduplicated sealed bases for the independent SemanticFixed runtime."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from os import PathLike
from typing import Any, Callable, Mapping, Sequence

import torch.distributed as dist

from .catalog import (
    checkpoint_state_checksum,
    deserialize_checkpoint,
    validate_checkpoint_objects,
)
from .global_commit import manifest_evidence
from .matched import CONTROL_SOURCE_RANK
from .object_semantic_base import (
    ObjectSemanticBaseLocator,
    ObjectSemanticBaseTransferMetrics,
    RestoredObjectSemanticBase,
    _base_metadata,
    _binding,
    _cpu,
    _layout,
    _layout_from_json,
    _layout_json,
    _require_group,
    _topology,
    write_object_semantic_base_locator,
)
from .object_semantic_plan import RankBinding
from .object_update_transport import commit_object_update_state, seal_object_update_host
from .replica_service import ReplicaServiceClient


_SHARED_STATE_KEYS = frozenset(("model", "optimizer"))
_MANIFEST_KEYS = frozenset(
    ("manifest_id", "epoch", "source_rank", "target_rank", "group_ranks", "payload_checksum")
)
_SHARED_BUNDLE_KEYS = frozenset(
    (
        "schema_version", "kind", "run_id", "epoch", "representative_rank",
        "storage_source_rank", "source_host", "class_key", "shared_id", "snapshot_shared",
    )
)
_OVERLAY_BUNDLE_KEYS = frozenset(
    (
        "schema_version", "kind", "run_id", "epoch", "source_rank", "source_host",
        "class_key", "shared_id", "shared_source_rank", "base_id", "binding",
        "parameter_ids", "snapshot_overlay",
    )
)
_CONTROL_KEYS = frozenset(
    (
        "schema_version", "kind", "run_id", "epoch", "world_size", "rank_bindings",
        "shared_by_host", "overlays_by_host",
    )
)


@dataclass(frozen=True)
class PublishedDeduplicatedObjectSemanticBase:
    locator: ObjectSemanticBaseLocator
    metrics: ObjectSemanticBaseTransferMetrics
    local_base_id: str


def _content_id(value: object) -> str:
    return f"sha256:{checkpoint_state_checksum(value)}"


def _valid_content_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _class_key(binding: RankBinding) -> tuple[int, int]:
    return binding.pipeline_rank, binding.tensor_rank


def _manifest_only(value: Mapping[str, Any]) -> dict[str, Any]:
    if not _MANIFEST_KEYS.issubset(value):
        raise ValueError("deduplicated base manifest evidence is incomplete")
    return {name: value[name] for name in _MANIFEST_KEYS}


def _upload_local(
    *,
    rank: int,
    epoch: int,
    run_id: str,
    base_factory: Callable[[], Mapping[str, Any]],
    layout: Sequence[RankBinding],
    clients: Mapping[int, ReplicaServiceClient],
) -> dict[str, Any]:
    world_size, binding = len(layout), layout[rank]
    snapshot = dict(base_factory())
    if not _cpu(snapshot):
        raise ValueError("full training base snapshot must be CPU-only")
    recovery_binding, topology = _base_metadata(
        snapshot, epoch=epoch, rank=rank, world_size=world_size
    )
    if (
        recovery_binding.data_parallel_rank != binding.data_rank
        or recovery_binding.pipeline_parallel_rank != binding.pipeline_rank
        or recovery_binding.tensor_parallel_rank != binding.tensor_rank
    ):
        raise ValueError("full training base binding differs from the logical layout")

    shared = {name: snapshot[name] for name in _SHARED_STATE_KEYS}
    overlay = {name: value for name, value in snapshot.items() if name not in _SHARED_STATE_KEYS}
    shared_id, base_id = _content_id(shared), _content_id(snapshot)
    key = _class_key(binding)
    representative = min(
        item.rank for item in layout if _class_key(item) == key
    )
    shared_source_rank = world_size + representative
    target_ranks = {
        host: min(item.rank for item in layout if item.host == host) for host in (0, 1)
    }
    manifests = {}
    remote_payload = remote_wire = 0
    maximum_put_ms = 0.0
    for host in (0, 1):
        overlay_bundle = {
            "schema_version": 1,
            "kind": "object_semantic_deduplicated_base_overlay",
            "run_id": run_id,
            "epoch": epoch,
            "source_rank": rank,
            "source_host": binding.host,
            "class_key": list(key),
            "shared_id": shared_id,
            "shared_source_rank": shared_source_rank,
            "base_id": base_id,
            "binding": dict(snapshot["binding"]),
            "parameter_ids": [list(item) for item in topology],
            "snapshot_overlay": overlay,
        }
        overlay_manifest, overlay_wire, overlay_ms = commit_object_update_state(
            clients[host],
            overlay_bundle,
            source_rank=rank,
            source_host=binding.host,
            target_host=host,
            target_rank=target_ranks[host],
            group_ranks=tuple(range(world_size)),
        )
        shared_manifest = None
        shared_wire = 0
        shared_ms = 0.0
        if rank == representative:
            shared_bundle = {
                "schema_version": 1,
                "kind": "object_semantic_deduplicated_base_shared",
                "run_id": run_id,
                "epoch": epoch,
                "representative_rank": rank,
                "storage_source_rank": shared_source_rank,
                "source_host": binding.host,
                "class_key": list(key),
                "shared_id": shared_id,
                "snapshot_shared": shared,
            }
            committed, shared_wire, shared_ms = commit_object_update_state(
                clients[host],
                shared_bundle,
                source_rank=shared_source_rank,
                source_host=binding.host,
                target_host=host,
                target_rank=target_ranks[host],
                group_ranks=tuple(range(world_size)),
            )
            shared_manifest = manifest_evidence(committed)
        manifests[host] = {
            "overlay": manifest_evidence(overlay_manifest),
            "shared": shared_manifest,
        }
        if host != binding.host:
            remote_payload += overlay_manifest.payload_bytes
            if shared_manifest is not None:
                remote_payload += committed.payload_bytes
            remote_wire += overlay_wire + shared_wire
            maximum_put_ms = max(maximum_put_ms, overlay_ms, shared_ms)
    return {
        "rank": rank,
        "base_id": base_id,
        "shared_id": shared_id,
        "shared_source_rank": shared_source_rank,
        "class_key": list(key),
        "binding": dict(snapshot["binding"]),
        "parameter_ids": [list(item) for item in topology],
        "manifests_by_host": [[host, manifests[host]] for host in (0, 1)],
        "metrics": {
            "remote_data_payload_bytes": remote_payload,
            "remote_wire_bytes": remote_wire,
            "maximum_remote_put_ms": maximum_put_ms,
        },
    }


def _control_and_evidence(
    *, run_id: str, epoch: int, layout: Sequence[RankBinding], gathered: Sequence[object]
) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]], dict[str, Any], dict[int, str]]:
    world_size = len(layout)
    rows = {}
    totals = {
        "remote_data_payload_bytes": 0,
        "remote_wire_bytes": 0,
        "maximum_remote_put_ms": 0.0,
    }
    for value in gathered:
        expected = {
            "rank", "base_id", "shared_id", "shared_source_rank", "class_key",
            "binding", "parameter_ids", "manifests_by_host", "metrics",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("gathered deduplicated base evidence schema changed")
        rank = value["rank"]
        if type(rank) is not int or rank in rows or not 0 <= rank < world_size:
            raise ValueError("gathered deduplicated base rank is invalid")
        binding = _binding(value["binding"])
        topology = _topology(value["parameter_ids"])
        item = layout[rank]
        key = _class_key(item)
        representative = min(member.rank for member in layout if _class_key(member) == key)
        if (
            binding.rank != rank
            or binding.world_size != world_size
            or (binding.data_parallel_rank, binding.pipeline_parallel_rank, binding.tensor_parallel_rank)
            != (item.data_rank, item.pipeline_rank, item.tensor_rank)
            or tuple(value["class_key"]) != key
            or value["shared_source_rank"] != world_size + representative
            or not _valid_content_id(value["base_id"])
            or not _valid_content_id(value["shared_id"])
        ):
            raise ValueError("gathered deduplicated base identity changed")
        manifests = dict(value["manifests_by_host"])
        if set(manifests) != {0, 1}:
            raise ValueError("gathered deduplicated base lacks a host manifest")
        target_ranks = {
            host: min(member.rank for member in layout if member.host == host)
            for host in (0, 1)
        }
        for host in (0, 1):
            pair = manifests[host]
            if not isinstance(pair, Mapping) or set(pair) != {"overlay", "shared"}:
                raise ValueError("gathered deduplicated base manifest schema changed")
            overlay_manifest = pair["overlay"]
            shared_manifest = pair["shared"]
            if (
                not isinstance(overlay_manifest, Mapping)
                or int(overlay_manifest.get("source_rank", -1)) != rank
                or int(overlay_manifest.get("epoch", -1)) != epoch
                or int(overlay_manifest.get("target_rank", -1)) != target_ranks[host]
                or tuple(overlay_manifest.get("group_ranks", ())) != tuple(range(world_size))
                or ((rank == representative) != isinstance(shared_manifest, Mapping))
            ):
                raise ValueError("gathered deduplicated base manifest changed")
            if isinstance(shared_manifest, Mapping) and (
                int(shared_manifest.get("source_rank", -1)) != world_size + representative
                or int(shared_manifest.get("epoch", -1)) != epoch
                or int(shared_manifest.get("target_rank", -1)) != target_ranks[host]
                or tuple(shared_manifest.get("group_ranks", ())) != tuple(range(world_size))
            ):
                raise ValueError("gathered deduplicated shared manifest changed")
        metrics = value["metrics"]
        if not isinstance(metrics, Mapping) or set(metrics) != set(totals):
            raise ValueError("gathered deduplicated base metrics changed")
        for name in ("remote_data_payload_bytes", "remote_wire_bytes"):
            if type(metrics[name]) is not int or metrics[name] < 0:
                raise ValueError("gathered deduplicated base metrics are invalid")
            totals[name] += metrics[name]
        if not isinstance(metrics["maximum_remote_put_ms"], float):
            raise ValueError("gathered deduplicated base metrics are invalid")
        totals["maximum_remote_put_ms"] = max(
            totals["maximum_remote_put_ms"], metrics["maximum_remote_put_ms"]
        )
        rows[rank] = (value, topology, manifests)
    if set(rows) != set(range(world_size)):
        raise ValueError("gathered deduplicated base lacks a rank")

    classes = {}
    for rank, (value, topology, _) in rows.items():
        classes.setdefault(tuple(value["class_key"]), []).append((rank, value, topology))
    for members in classes.values():
        if len({value["shared_id"] for _, value, _ in members}) != 1:
            raise ValueError("data-parallel base replicas diverged")
        if len({topology for _, _, topology in members}) != 1:
            raise ValueError("data-parallel base parameter topology diverged")

    seal_evidence: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    overlays_by_host = {0: [], 1: []}
    shared_by_host = {0: [], 1: []}
    for host in (0, 1):
        for rank in range(world_size):
            value, _, manifests = rows[rank]
            manifest = dict(manifests[host]["overlay"])
            seal_evidence[host].append(manifest)
            overlays_by_host[host].append({
                **manifest,
                **{name: copy.deepcopy(value[name]) for name in (
                    "rank", "base_id", "shared_id", "shared_source_rank", "class_key",
                    "binding", "parameter_ids",
                )},
            })
        for key in sorted(classes):
            representative = min(rank for rank, _, _ in classes[key])
            value, _, manifests = rows[representative]
            manifest = dict(manifests[host]["shared"])
            seal_evidence[host].append(manifest)
            shared_by_host[host].append({
                **manifest,
                "representative_rank": representative,
                "shared_source_rank": value["shared_source_rank"],
                "shared_id": value["shared_id"],
                "class_key": list(key),
            })
    control = {
        "schema_version": 1,
        "kind": "object_semantic_deduplicated_base_control",
        "run_id": run_id,
        "epoch": epoch,
        "world_size": world_size,
        "rank_bindings": _layout_json(layout),
        "shared_by_host": [[host, shared_by_host[host]] for host in (0, 1)],
        "overlays_by_host": [[host, overlays_by_host[host]] for host in (0, 1)],
    }
    return control, seal_evidence, totals, {
        rank: str(rows[rank][0]["base_id"]) for rank in range(world_size)
    }


def _publish(
    *,
    clients: Mapping[int, ReplicaServiceClient],
    run_id: str,
    epoch: int,
    layout: Sequence[RankBinding],
    locator_path: str | PathLike[str],
    gathered: Sequence[object],
) -> dict[str, Any]:
    control, evidence, totals, base_ids = _control_and_evidence(
        run_id=run_id, epoch=epoch, layout=layout, gathered=gathered
    )
    target_ranks = {
        host: min(item.rank for item in layout if item.host == host) for host in (0, 1)
    }
    controls = {}
    for host in (0, 1):
        manifest, wire_bytes, put_ms = commit_object_update_state(
            clients[host],
            control,
            source_rank=CONTROL_SOURCE_RANK,
            source_host=0,
            target_host=host,
            target_rank=target_ranks[host],
            group_ranks=tuple(range(len(layout))),
        )
        controls[host] = manifest
        if host != 0:
            totals["remote_wire_bytes"] += wire_bytes
            totals["maximum_remote_put_ms"] = max(
                totals["maximum_remote_put_ms"], put_ms
            )
            totals["remote_control_payload_bytes"] = manifest.payload_bytes
    checksum = controls[0].payload_checksum
    if any(manifest.payload_checksum != checksum for manifest in controls.values()):
        raise ValueError("two hosts committed different deduplicated base controls")
    for host in (0, 1):
        seal_object_update_host(
            clients[host],
            epoch=epoch,
            control_checksum=checksum,
            control_evidence=manifest_evidence(controls[host]),
            bundle_evidence=evidence[host],
        )
    locator = ObjectSemanticBaseLocator.create(
        run_id=run_id,
        epoch=epoch,
        layout=layout,
        control_payload_checksum=checksum,
        control_manifests_by_host=controls,
    )
    write_object_semantic_base_locator(locator_path, locator)
    return {
        "locator": locator.to_dict(),
        "base_ids": [[rank, base_ids[rank]] for rank in range(len(layout))],
        "metrics": {
            "remote_data_payload_bytes": totals["remote_data_payload_bytes"],
            "remote_control_payload_bytes": totals.get("remote_control_payload_bytes", 0),
            "remote_wire_bytes": totals["remote_wire_bytes"],
            "maximum_remote_put_ms": totals["maximum_remote_put_ms"],
        },
    }


def commit_distributed_deduplicated_object_semantic_base(
    clients: Mapping[int, ReplicaServiceClient],
    *,
    run_id: str,
    epoch: int,
    local_base_factory: Callable[[], Mapping[str, Any]],
    rank_bindings: Sequence[RankBinding],
    locator_path: str | PathLike[str],
    group: object | None,
    max_parallel_uploads: int = 2,
) -> PublishedDeduplicatedObjectSemanticBase:
    """Publish one heavy base per PP/TP class plus one small overlay per rank."""
    rank, world_size = _require_group(group, rank_bindings)
    layout = _layout(rank_bindings, world_size)
    if (
        set(clients) != {0, 1}
        or not run_id
        or type(epoch) is not int
        or epoch < 0
        or not callable(local_base_factory)
        or type(max_parallel_uploads) is not int
        or max_parallel_uploads < 1
    ):
        raise ValueError("deduplicated base commit inputs are invalid")
    gathered = []
    for start in range(0, world_size, max_parallel_uploads):
        stop = min(start + max_parallel_uploads, world_size)
        local: object = None
        if start <= rank < stop:
            try:
                local = {
                    "ok": True,
                    "evidence": _upload_local(
                        rank=rank,
                        epoch=epoch,
                        run_id=run_id,
                        base_factory=local_base_factory,
                        layout=layout,
                        clients=clients,
                    ),
                }
            except Exception as error:
                local = {
                    "ok": False,
                    "error": f"rank {rank} {type(error).__name__}: {error}",
                }
        window: list[object] = [None] * world_size
        dist.all_gather_object(window, local, group=group)
        for source_rank in range(start, stop):
            value = window[source_rank]
            if not isinstance(value, Mapping) or not value.get("ok"):
                error = (
                    value.get("error", "rank base commit failed")
                    if isinstance(value, Mapping)
                    else "rank base commit failed"
                )
                raise RuntimeError(str(error))
            gathered.append(value["evidence"])

    response: object = None
    if rank == 0:
        try:
            response = {
                "ok": True,
                "published": _publish(
                    clients=clients,
                    run_id=run_id,
                    epoch=epoch,
                    layout=layout,
                    locator_path=locator_path,
                    gathered=gathered,
                ),
            }
        except Exception as error:
            response = {"ok": False, "error": f"rank 0 {type(error).__name__}: {error}"}
    broadcast = [response]
    dist.broadcast_object_list(broadcast, src=0, group=group)
    value = broadcast[0]
    if not isinstance(value, Mapping) or not value.get("ok"):
        error = value.get("error", "deduplicated base publication failed") if isinstance(value, Mapping) else "deduplicated base publication failed"
        raise RuntimeError(str(error))
    published = value.get("published")
    if not isinstance(published, Mapping) or set(published) != {"locator", "base_ids", "metrics"}:
        raise RuntimeError("deduplicated base publication schema changed")
    locator = ObjectSemanticBaseLocator.from_dict(published["locator"])
    base_ids = dict(published["base_ids"])
    metrics = published["metrics"]
    if set(base_ids) != set(range(world_size)) or not isinstance(metrics, Mapping):
        raise RuntimeError("deduplicated base publication is incomplete")
    return PublishedDeduplicatedObjectSemanticBase(
        locator,
        ObjectSemanticBaseTransferMetrics(
            int(metrics["remote_data_payload_bytes"]),
            int(metrics["remote_control_payload_bytes"]),
            int(metrics["remote_wire_bytes"]),
            float(metrics["maximum_remote_put_ms"]),
        ),
        str(base_ids[rank]),
    )


def restore_deduplicated_object_semantic_base(
    client: ReplicaServiceClient,
    *,
    surviving_host: int,
    locator: ObjectSemanticBaseLocator,
    logical_rank: int,
    expected_rank_bindings: Sequence[RankBinding],
    expected_parameter_ids: Sequence[Sequence[str]],
    expected_base_id: str | None = None,
) -> RestoredObjectSemanticBase:
    """Restore one rank from a shared PP/TP base and its rank-local overlay."""
    locator.validate()
    expected_layout = _layout(expected_rank_bindings, locator.world_size)
    topology = _topology(expected_parameter_ids)
    if (
        surviving_host not in (0, 1)
        or not 0 <= logical_rank < locator.world_size
        or locator.layout != expected_layout
    ):
        raise ValueError("deduplicated base recovery refuses a different logical layout")
    control_evidence = dict(locator.control_manifests_by_host)[surviving_host]
    started = time.perf_counter()
    envelope, metrics = client.get(
        CONTROL_SOURCE_RANK,
        locator.epoch,
        control_checksum=locator.control_payload_checksum,
        expected_manifest=control_evidence,
    )
    times = [(time.perf_counter() - started) * 1000.0]
    payload_bytes, wire_bytes = len(envelope.payload), metrics.wire_bytes
    control = deserialize_checkpoint(envelope.payload, locator.control_payload_checksum)
    validate_checkpoint_objects(control, envelope.manifest.objects)
    if (
        not isinstance(control, Mapping)
        or set(control) != _CONTROL_KEYS
        or control.get("schema_version") != 1
        or control.get("kind") != "object_semantic_deduplicated_base_control"
        or control.get("run_id") != locator.run_id
        or control.get("epoch") != locator.epoch
        or control.get("world_size") != locator.world_size
        or _layout_from_json(control.get("rank_bindings"), locator.world_size) != expected_layout
    ):
        raise ValueError("deduplicated base control identity or layout changed")
    overlays = dict(control["overlays_by_host"])
    shared = dict(control["shared_by_host"])
    if set(overlays) != {0, 1} or set(shared) != {0, 1}:
        raise ValueError("deduplicated base control lacks two-host evidence")
    overlay_rows = overlays[surviving_host]
    shared_rows = shared[surviving_host]
    if (
        not isinstance(overlay_rows, list)
        or len(overlay_rows) != locator.world_size
        or {item.get("rank") for item in overlay_rows if isinstance(item, Mapping)}
        != set(range(locator.world_size))
        or not isinstance(shared_rows, list)
    ):
        raise ValueError("deduplicated base control is incomplete")
    selected_overlay = [
        item for item in overlay_rows
        if isinstance(item, Mapping) and item.get("rank") == logical_rank
    ]
    if len(selected_overlay) != 1:
        raise ValueError("deduplicated base control lacks the target rank")
    overlay_evidence = dict(selected_overlay[0])
    expected_key = _class_key(expected_layout[logical_rank])
    shared_source_rank = overlay_evidence.get("shared_source_rank")
    selected_shared = [
        item for item in shared_rows
        if isinstance(item, Mapping) and item.get("shared_source_rank") == shared_source_rank
    ]
    if (
        len(selected_shared) != 1
        or tuple(overlay_evidence.get("class_key", ())) != expected_key
        or _binding(overlay_evidence.get("binding"))
        != _binding({
            "rank": logical_rank,
            "world_size": locator.world_size,
            "data_parallel_rank": expected_layout[logical_rank].data_rank,
            "pipeline_parallel_rank": expected_layout[logical_rank].pipeline_rank,
            "tensor_parallel_rank": expected_layout[logical_rank].tensor_rank,
        })
        or _topology(overlay_evidence.get("parameter_ids")) != topology
    ):
        raise ValueError("deduplicated base control mapping changed")
    shared_evidence = dict(selected_shared[0])
    if (
        shared_evidence.get("shared_id") != overlay_evidence.get("shared_id")
        or tuple(shared_evidence.get("class_key", ())) != expected_key
    ):
        raise ValueError("deduplicated base shared reference changed")

    bundles = []
    for source_rank, evidence in (
        (int(shared_source_rank), shared_evidence),
        (logical_rank, overlay_evidence),
    ):
        started = time.perf_counter()
        bundle_envelope, bundle_metrics = client.get(
            source_rank,
            locator.epoch,
            control_checksum=locator.control_payload_checksum,
            expected_manifest=_manifest_only(evidence),
        )
        times.append((time.perf_counter() - started) * 1000.0)
        payload_bytes += len(bundle_envelope.payload)
        wire_bytes += bundle_metrics.wire_bytes
        bundle = deserialize_checkpoint(
            bundle_envelope.payload, bundle_envelope.manifest.payload_checksum
        )
        validate_checkpoint_objects(bundle, bundle_envelope.manifest.objects)
        bundles.append(bundle)
    shared_bundle, overlay_bundle = bundles
    if (
        not isinstance(shared_bundle, Mapping)
        or set(shared_bundle) != _SHARED_BUNDLE_KEYS
        or shared_bundle.get("schema_version") != 1
        or shared_bundle.get("kind") != "object_semantic_deduplicated_base_shared"
        or shared_bundle.get("run_id") != locator.run_id
        or shared_bundle.get("epoch") != locator.epoch
        or shared_bundle.get("storage_source_rank") != shared_source_rank
        or shared_bundle.get("shared_id") != shared_evidence.get("shared_id")
        or tuple(shared_bundle.get("class_key", ())) != expected_key
        or not isinstance(shared_bundle.get("snapshot_shared"), Mapping)
        or set(shared_bundle["snapshot_shared"]) != _SHARED_STATE_KEYS
        or not _cpu(shared_bundle["snapshot_shared"])
        or _content_id(shared_bundle["snapshot_shared"]) != shared_bundle["shared_id"]
    ):
        raise ValueError("deduplicated shared base bundle changed")
    if (
        not isinstance(overlay_bundle, Mapping)
        or set(overlay_bundle) != _OVERLAY_BUNDLE_KEYS
        or overlay_bundle.get("schema_version") != 1
        or overlay_bundle.get("kind") != "object_semantic_deduplicated_base_overlay"
        or overlay_bundle.get("run_id") != locator.run_id
        or overlay_bundle.get("epoch") != locator.epoch
        or overlay_bundle.get("source_rank") != logical_rank
        or overlay_bundle.get("source_host") != expected_layout[logical_rank].host
        or overlay_bundle.get("shared_id") != shared_bundle.get("shared_id")
        or overlay_bundle.get("shared_source_rank") != shared_source_rank
        or tuple(overlay_bundle.get("class_key", ())) != expected_key
        or not isinstance(overlay_bundle.get("snapshot_overlay"), Mapping)
        or _SHARED_STATE_KEYS.intersection(overlay_bundle["snapshot_overlay"])
        or not _cpu(overlay_bundle["snapshot_overlay"])
    ):
        raise ValueError("deduplicated rank overlay bundle changed")
    state = {
        **copy.deepcopy(dict(shared_bundle["snapshot_shared"])),
        **copy.deepcopy(dict(overlay_bundle["snapshot_overlay"])),
    }
    binding, restored_topology = _base_metadata(
        state,
        epoch=locator.epoch,
        rank=logical_rank,
        world_size=locator.world_size,
    )
    base_id = _content_id(state)
    if (
        binding != _binding(overlay_bundle["binding"])
        or restored_topology != _topology(overlay_bundle["parameter_ids"])
        or restored_topology != topology
        or overlay_bundle.get("base_id") != base_id
        or overlay_evidence.get("base_id") != base_id
        or (expected_base_id is not None and expected_base_id != base_id)
    ):
        raise ValueError("deduplicated base id, binding, or topology changed")
    return RestoredObjectSemanticBase(
        state, base_id, payload_bytes, wire_bytes, max(times)
    )
