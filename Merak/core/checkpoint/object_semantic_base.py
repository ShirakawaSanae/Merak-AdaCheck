"""Sealed two-host full bases for fresh SemanticFixed recovery."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist

from .catalog import (
    checkpoint_state_checksum,
    deserialize_checkpoint,
    validate_checkpoint_objects,
)
from .global_commit import manifest_evidence
from .incremental_checkpoint import RecoveryBinding
from .matched import CONTROL_SOURCE_RANK
from .object_semantic_plan import RankBinding
from .object_update_transport import commit_object_update_state, seal_object_update_host
from .replica_service import ReplicaServiceClient


_SNAPSHOT_KEYS = frozenset((
    "schema_version", "kind", "run_id", "epoch", "source_rank", "source_host",
    "binding", "parameter_ids", "base_id", "snapshot",
))
_CONTROL_KEYS = frozenset((
    "schema_version", "kind", "run_id", "epoch", "world_size", "rank_bindings",
    "bases_by_host",
))
_RUNTIME_KEYS = frozenset((
    "iteration", "random_rng_state", "np_rng_state", "torch_rng_state",
    "cuda_rng_state", "rng_tracker_states", "sampler",
))
_BINDING_KEYS = frozenset((
    "rank", "world_size", "data_parallel_rank", "pipeline_parallel_rank",
    "tensor_parallel_rank",
))


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _cpu(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return value.device.type == "cpu"
    if isinstance(value, Mapping):
        return all(_cpu(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_cpu(item) for item in value)
    return True


def _binding(value: object) -> RecoveryBinding:
    if not isinstance(value, Mapping) or set(value) != _BINDING_KEYS:
        raise ValueError("base binding must be a logical recovery binding")
    try:
        return RecoveryBinding(
            rank=value["rank"], world_size=value["world_size"],
            data_parallel_rank=value["data_parallel_rank"],
            pipeline_parallel_rank=value["pipeline_parallel_rank"],
            tensor_parallel_rank=value["tensor_parallel_rank"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("base binding must be a logical recovery binding") from error


def _topology(value: object) -> tuple[tuple[str, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("base parameter topology is invalid")
    result = tuple(tuple(group) for group in value)
    if any(
        isinstance(group, (str, bytes))
        or any(not isinstance(name, str) or not name for name in group)
        for group in result
    ) or len({name for group in result for name in group}) != sum(map(len, result)):
        raise ValueError("base parameter topology is invalid")
    return result


def _base_metadata(
    state: Mapping[str, Any], *, epoch: int, rank: int, world_size: int
) -> tuple[RecoveryBinding, tuple[tuple[str, ...], ...]]:
    required = {
        "schema_version", "kind", "iteration", "model", "optimizer", "lr_scheduler",
        "parameter_ids", "binding", "runtime_state",
    }
    if (
        not required.issubset(state)
        or state["schema_version"] != 1
        or state["kind"] != "merak_incremental_base"
        or int(state["iteration"]) != epoch
        or not all(isinstance(state[name], Mapping) for name in ("model", "optimizer", "lr_scheduler"))
        or not isinstance(state["runtime_state"], Mapping)
        or not _RUNTIME_KEYS.issubset(state["runtime_state"])
        or int(state["runtime_state"]["iteration"]) != epoch
    ):
        raise ValueError("full training base is incomplete or incompatible")
    binding, topology = _binding(state["binding"]), _topology(state["parameter_ids"])
    if binding.rank != rank or binding.world_size != world_size:
        raise ValueError("full training base belongs to a different logical rank")
    return binding, topology


def _layout(bindings: Sequence[RankBinding], world_size: int) -> tuple[RankBinding, ...]:
    value = tuple(bindings)
    if (
        len(value) != world_size
        or tuple(item.rank for item in value) != tuple(range(world_size))
        or {item.host for item in value} != {0, 1}
    ):
        raise ValueError("base commit requires a complete two-host logical layout")
    return value


def _layout_json(bindings: Sequence[RankBinding]) -> list[dict[str, int]]:
    return [
        {"rank": item.rank, "pipeline_rank": item.pipeline_rank,
         "tensor_rank": item.tensor_rank, "data_rank": item.data_rank, "host": item.host}
        for item in bindings
    ]


def _layout_from_json(value: object, world_size: int) -> tuple[RankBinding, ...]:
    if not isinstance(value, list):
        raise ValueError("base control layout schema changed")
    try:
        return _layout(tuple(RankBinding(**dict(item)) for item in value), world_size)
    except (TypeError, ValueError) as error:
        raise ValueError("base control layout schema changed") from error


def _require_group(group: object | None, bindings: Sequence[RankBinding]) -> tuple[int, int]:
    if group is None or not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("base commit requires an explicit initialized gloo group")
    if dist.get_backend(group) != "gloo":
        raise RuntimeError("base commit requires a gloo group")
    rank, world_size = dist.get_rank(group), dist.get_world_size(group)
    if rank != dist.get_rank() or world_size != dist.get_world_size():
        raise ValueError("base commit requires a world-size gloo group")
    _layout(bindings, world_size)
    return rank, world_size


@dataclass(frozen=True)
class ObjectSemanticBaseTransferMetrics:
    remote_data_payload_bytes: int
    remote_control_payload_bytes: int
    remote_wire_bytes: int
    maximum_remote_put_ms: float

    @property
    def total_remote_payload_bytes(self) -> int:
        return self.remote_data_payload_bytes + self.remote_control_payload_bytes


@dataclass(frozen=True)
class ObjectSemanticBaseLocator:
    run_id: str
    epoch: int
    world_size: int
    layout: tuple[RankBinding, ...]
    control_payload_checksum: str
    control_manifests_by_host: tuple[tuple[int, dict[str, Any]], ...]
    locator_hash: str

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "kind": "object_semantic_base_locator",
            "run_id": self.run_id, "epoch": self.epoch, "world_size": self.world_size,
            "rank_bindings": _layout_json(self.layout),
            "control_payload_checksum": self.control_payload_checksum,
            "control_manifests_by_host": [
                [host, evidence] for host, evidence in self.control_manifests_by_host
            ],
        }

    def validate(self) -> None:
        if (
            not self.run_id or self.epoch < 0 or self.world_size < 1
            or _layout(self.layout, self.world_size) != self.layout
            or tuple(host for host, _ in self.control_manifests_by_host) != (0, 1)
            or self.locator_hash != _digest(self._body())
        ):
            raise ValueError("base locator identity or checksum changed")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {**self._body(), "locator_hash": self.locator_hash}

    @classmethod
    def create(
        cls, *, run_id: str, epoch: int, layout: Sequence[RankBinding],
        control_payload_checksum: str,
        control_manifests_by_host: Mapping[int, Any],
    ) -> "ObjectSemanticBaseLocator":
        checked = _layout(layout, len(layout))
        value = cls(
            run_id, epoch, len(checked), checked, control_payload_checksum,
            tuple((host, manifest_evidence(control_manifests_by_host[host])) for host in (0, 1)), "",
        )
        return replace(value, locator_hash=_digest(value._body()))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObjectSemanticBaseLocator":
        expected = {
            "schema_version", "kind", "run_id", "epoch", "world_size", "rank_bindings",
            "control_payload_checksum", "control_manifests_by_host", "locator_hash",
        }
        if set(value) != expected or value["schema_version"] != 1 or value["kind"] != "object_semantic_base_locator":
            raise ValueError("base locator schema changed")
        world_size = int(value["world_size"])
        try:
            locator = cls(
                str(value["run_id"]), int(value["epoch"]), world_size,
                _layout_from_json(value["rank_bindings"], world_size),
                str(value["control_payload_checksum"]),
                tuple((int(host), dict(evidence)) for host, evidence in value["control_manifests_by_host"]),
                str(value["locator_hash"]),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("base locator schema changed") from error
        locator.validate()
        return locator


@dataclass(frozen=True)
class PublishedObjectSemanticBase:
    locator: ObjectSemanticBaseLocator
    metrics: ObjectSemanticBaseTransferMetrics


@dataclass(frozen=True)
class RestoredObjectSemanticBase:
    state: dict[str, Any]
    base_id: str
    payload_bytes: int
    wire_bytes: int
    get_ms: float


def write_object_semantic_base_locator(path: str | PathLike[str], locator: ObjectSemanticBaseLocator) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(locator.to_dict(), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def load_object_semantic_base_locator(path: str | PathLike[str]) -> ObjectSemanticBaseLocator:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, Mapping):
        raise ValueError("base locator must be a JSON object")
    return ObjectSemanticBaseLocator.from_dict(value)


def _local_evidence(
    *, rank: int, epoch: int, run_id: str, state: Mapping[str, Any],
    layout: Sequence[RankBinding], clients: Mapping[int, ReplicaServiceClient],
) -> dict[str, Any]:
    world_size, binding = len(layout), layout[rank]
    snapshot = dict(state)
    if not _cpu(snapshot):
        raise ValueError("full training base snapshot must be CPU-only")
    recovery_binding, topology = _base_metadata(snapshot, epoch=epoch, rank=rank, world_size=world_size)
    if (
        recovery_binding.data_parallel_rank != binding.data_rank
        or recovery_binding.pipeline_parallel_rank != binding.pipeline_rank
        or recovery_binding.tensor_parallel_rank != binding.tensor_rank
    ):
        raise ValueError("full training base binding differs from the logical layout")
    base_id = f"sha256:{checkpoint_state_checksum(snapshot)}"
    target_ranks = {host: min(item.rank for item in layout if item.host == host) for host in (0, 1)}
    manifests: dict[int, dict[str, Any]] = {}
    remote_payload = remote_wire = 0
    maximum_put_ms = 0.0
    for host in (0, 1):
        bundle = {
            "schema_version": 1, "kind": "object_semantic_base_snapshot", "run_id": run_id,
            "epoch": epoch, "source_rank": rank, "source_host": binding.host,
            "binding": dict(snapshot["binding"]), "parameter_ids": [list(item) for item in topology],
            "base_id": base_id, "snapshot": snapshot,
        }
        manifest, wire_bytes, put_ms = commit_object_update_state(
            clients[host], bundle, source_rank=rank, source_host=binding.host,
            target_host=host, target_rank=target_ranks[host], group_ranks=tuple(range(world_size)),
        )
        manifests[host] = manifest_evidence(manifest)
        if host != binding.host:
            remote_payload += manifest.payload_bytes
            remote_wire += wire_bytes
            maximum_put_ms = max(maximum_put_ms, put_ms)
    return {
        "rank": rank, "base_id": base_id, "binding": dict(snapshot["binding"]),
        "parameter_ids": [list(item) for item in topology],
        "manifests_by_host": [[host, manifests[host]] for host in (0, 1)],
        "metrics": {"remote_data_payload_bytes": remote_payload, "remote_wire_bytes": remote_wire,
                    "maximum_remote_put_ms": maximum_put_ms},
    }


def _control(
    *, run_id: str, epoch: int, layout: Sequence[RankBinding], gathered: Sequence[object]
) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]], dict[str, Any]]:
    world_size = len(layout)
    by_host: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    totals = {"remote_data_payload_bytes": 0, "remote_wire_bytes": 0, "maximum_remote_put_ms": 0.0}
    seen = set()
    for value in gathered:
        if not isinstance(value, Mapping) or set(value) != {"rank", "base_id", "binding", "parameter_ids", "manifests_by_host", "metrics"}:
            raise ValueError("gathered base evidence schema changed")
        rank = value["rank"]
        if type(rank) is not int or rank in seen or not 0 <= rank < world_size:
            raise ValueError("gathered base evidence rank is invalid")
        seen.add(rank)
        binding, topology = _binding(value["binding"]), _topology(value["parameter_ids"])
        item = layout[rank]
        if (
            binding.rank != rank or binding.world_size != world_size
            or (binding.data_parallel_rank, binding.pipeline_parallel_rank, binding.tensor_parallel_rank)
            != (item.data_rank, item.pipeline_rank, item.tensor_rank)
            or not isinstance(value["base_id"], str)
            or len(value["base_id"]) != 71
            or not value["base_id"].startswith("sha256:")
            or any(character not in "0123456789abcdef" for character in value["base_id"][7:])
        ):
            raise ValueError("gathered base evidence binding changed")
        manifests = dict(value["manifests_by_host"])
        if set(manifests) != {0, 1} or any(not isinstance(item, Mapping) for item in manifests.values()):
            raise ValueError("gathered base evidence lacks a host manifest")
        for host in (0, 1):
            evidence = dict(manifests[host])
            if (
                int(evidence.get("source_rank", -1)) != rank
                or int(evidence.get("epoch", -1)) != epoch
                or int(evidence.get("target_rank", -1))
                != min(item.rank for item in layout if item.host == host)
                or tuple(evidence.get("group_ranks", ())) != tuple(range(world_size))
            ):
                raise ValueError("gathered base manifest changed")
            by_host[host].append(evidence)
        metrics = value["metrics"]
        if not isinstance(metrics, Mapping) or set(metrics) != set(totals):
            raise ValueError("gathered base metrics schema changed")
        for field in ("remote_data_payload_bytes", "remote_wire_bytes"):
            if type(metrics[field]) is not int or metrics[field] < 0:
                raise ValueError("gathered base metrics are invalid")
            totals[field] += metrics[field]
        if not isinstance(metrics["maximum_remote_put_ms"], float):
            raise ValueError("gathered base metrics are invalid")
        totals["maximum_remote_put_ms"] = max(totals["maximum_remote_put_ms"], metrics["maximum_remote_put_ms"])
    if seen != set(range(world_size)):
        raise ValueError("gathered base evidence lacks a rank")
    normalized = {host: sorted(items, key=lambda item: int(item["source_rank"])) for host, items in by_host.items()}
    records = {}
    for value in sorted(gathered, key=lambda item: int(item["rank"])):
        records[value["rank"]] = {
            name: copy.deepcopy(value[name])
            for name in ("rank", "base_id", "binding", "parameter_ids")
        }
    control = {
        "schema_version": 1, "kind": "object_semantic_base_control", "run_id": run_id,
        "epoch": epoch, "world_size": world_size, "rank_bindings": _layout_json(layout),
        "bases_by_host": [[host, [
            {**evidence, **records[evidence["source_rank"]]}
            for evidence in normalized[host]
        ]] for host in (0, 1)],
    }
    return control, normalized, totals


def _publish(
    *, clients: Mapping[int, ReplicaServiceClient], run_id: str, epoch: int,
    layout: Sequence[RankBinding], locator_path: str | PathLike[str], gathered: Sequence[object],
) -> dict[str, Any]:
    control, evidence, totals = _control(run_id=run_id, epoch=epoch, layout=layout, gathered=gathered)
    target_ranks = {host: min(item.rank for item in layout if item.host == host) for host in (0, 1)}
    controls = {}
    for host in (0, 1):
        manifest, wire_bytes, put_ms = commit_object_update_state(
            clients[host], control, source_rank=CONTROL_SOURCE_RANK, source_host=0,
            target_host=host, target_rank=target_ranks[host], group_ranks=tuple(range(len(layout))),
        )
        controls[host] = manifest
        if host != 0:
            totals["remote_wire_bytes"] += wire_bytes
            totals["maximum_remote_put_ms"] = max(totals["maximum_remote_put_ms"], put_ms)
            totals["remote_control_payload_bytes"] = manifest.payload_bytes
    checksum = controls[0].payload_checksum
    if any(manifest.payload_checksum != checksum for manifest in controls.values()):
        raise ValueError("two hosts committed different base controls")
    for host in (0, 1):
        seal_object_update_host(
            clients[host], epoch=epoch, control_checksum=checksum,
            control_evidence=manifest_evidence(controls[host]), bundle_evidence=evidence[host],
        )
    locator = ObjectSemanticBaseLocator.create(
        run_id=run_id, epoch=epoch, layout=layout, control_payload_checksum=checksum,
        control_manifests_by_host=controls,
    )
    write_object_semantic_base_locator(locator_path, locator)
    return {"locator": locator.to_dict(), "metrics": {
        "remote_data_payload_bytes": totals["remote_data_payload_bytes"],
        "remote_control_payload_bytes": totals.get("remote_control_payload_bytes", 0),
        "remote_wire_bytes": totals["remote_wire_bytes"],
        "maximum_remote_put_ms": totals["maximum_remote_put_ms"],
    }}


def commit_distributed_object_semantic_base(
    clients: Mapping[int, ReplicaServiceClient], *, run_id: str, epoch: int,
    local_base: Mapping[str, Any], rank_bindings: Sequence[RankBinding],
    locator_path: str | PathLike[str], group: object | None,
    max_parallel_uploads: int = 4,
) -> PublishedObjectSemanticBase:
    """Commit one CPU base per logical rank and publish only after both seals."""
    rank, world_size = _require_group(group, rank_bindings)
    layout = _layout(rank_bindings, world_size)
    if (
        set(clients) != {0, 1}
        or not run_id
        or type(epoch) is not int
        or epoch < 0
        or type(max_parallel_uploads) is not int
        or max_parallel_uploads < 1
    ):
        raise ValueError("base commit requires two host clients, a run id, and a non-negative epoch")
    gathered: list[object] = []
    for start in range(0, world_size, max_parallel_uploads):
        stop = min(start + max_parallel_uploads, world_size)
        local: object = None
        if start <= rank < stop:
            try:
                local = {"ok": True, "evidence": _local_evidence(
                    rank=rank, epoch=epoch, run_id=run_id, state=local_base,
                    layout=layout, clients=clients,
                )}
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
            gathered.append(value)
    response: object = None
    if rank == 0:
        try:
            failures = [item for item in gathered if not isinstance(item, Mapping) or not item.get("ok")]
            if failures:
                raise RuntimeError(str(failures[0].get("error", "rank base commit failed")))
            response = {"ok": True, "published": _publish(
                clients=clients, run_id=run_id, epoch=epoch, layout=layout,
                locator_path=locator_path, gathered=[item["evidence"] for item in gathered],
            )}
        except Exception as error:
            response = {"ok": False, "error": f"rank 0 {type(error).__name__}: {error}"}
    broadcast = [response]
    dist.broadcast_object_list(broadcast, src=0, group=group)
    value = broadcast[0]
    if not isinstance(value, Mapping) or set(value) != ({"ok", "published"} if value.get("ok") else {"ok", "error"}):
        raise RuntimeError("base commit broadcast schema changed")
    if not value["ok"]:
        raise RuntimeError(str(value["error"]))
    published = value["published"]
    if not isinstance(published, Mapping) or set(published) != {"locator", "metrics"}:
        raise RuntimeError("base commit publication schema changed")
    locator = ObjectSemanticBaseLocator.from_dict(published["locator"])
    metrics = published["metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != {
        "remote_data_payload_bytes", "remote_control_payload_bytes", "remote_wire_bytes", "maximum_remote_put_ms",
    }:
        raise RuntimeError("base commit metrics schema changed")
    if locator.run_id != run_id or locator.epoch != epoch or locator.layout != layout:
        raise RuntimeError("broadcast base locator disagrees with this rank")
    return PublishedObjectSemanticBase(locator, ObjectSemanticBaseTransferMetrics(
        int(metrics["remote_data_payload_bytes"]), int(metrics["remote_control_payload_bytes"]),
        int(metrics["remote_wire_bytes"]), float(metrics["maximum_remote_put_ms"]),
    ))


def restore_object_semantic_base(
    client: ReplicaServiceClient, *, surviving_host: int, locator: ObjectSemanticBaseLocator,
    logical_rank: int, expected_rank_bindings: Sequence[RankBinding],
    expected_parameter_ids: Sequence[Sequence[str]], expected_base_id: str | None = None,
) -> RestoredObjectSemanticBase:
    """Read only one logical-rank base from either surviving sealed host."""
    locator.validate()
    expected_layout = _layout(expected_rank_bindings, locator.world_size)
    topology = _topology(expected_parameter_ids)
    if surviving_host not in (0, 1) or not 0 <= logical_rank < locator.world_size or locator.layout != expected_layout:
        raise ValueError("base recovery refuses a different logical layout")
    control_evidence = dict(locator.control_manifests_by_host)[surviving_host]
    started = time.perf_counter()
    envelope, metrics = client.get(
        CONTROL_SOURCE_RANK, locator.epoch, control_checksum=locator.control_payload_checksum,
        expected_manifest=control_evidence,
    )
    control_ms = (time.perf_counter() - started) * 1000.0
    control = deserialize_checkpoint(envelope.payload, locator.control_payload_checksum)
    validate_checkpoint_objects(control, envelope.manifest.objects)
    if not isinstance(control, Mapping) or set(control) != _CONTROL_KEYS:
        raise ValueError("base control schema changed")
    if (
        control.get("schema_version") != 1 or control.get("kind") != "object_semantic_base_control"
        or control.get("run_id") != locator.run_id or control.get("epoch") != locator.epoch
        or control.get("world_size") != locator.world_size
        or _layout_from_json(control.get("rank_bindings"), locator.world_size) != expected_layout
    ):
        raise ValueError("base control identity or layout changed")
    by_host = dict(control["bases_by_host"])
    if (
        set(by_host) != {0, 1}
        or any(not isinstance(by_host[host], list) for host in (0, 1))
        or any(
            {item.get("rank") for item in by_host[host] if isinstance(item, Mapping)}
            != set(range(locator.world_size))
            or len(by_host[host]) != locator.world_size
            for host in (0, 1)
        )
    ):
        raise ValueError("base control lacks two-host evidence")
    selected = [item for item in by_host[surviving_host] if isinstance(item, Mapping) and item.get("rank") == logical_rank]
    if len(selected) != 1:
        raise ValueError("base control lacks the target logical rank")
    evidence = dict(selected[0])
    required = {"manifest_id", "epoch", "source_rank", "target_rank", "group_ranks", "payload_checksum", "rank", "base_id", "binding", "parameter_ids"}
    if set(evidence) != required or _binding(evidence["binding"]) != RecoveryBinding(
        logical_rank, locator.world_size, expected_layout[logical_rank].data_rank,
        expected_layout[logical_rank].pipeline_rank, expected_layout[logical_rank].tensor_rank,
    ) or _topology(evidence["parameter_ids"]) != topology:
        raise ValueError("base control binding or parameter topology changed")
    manifest_evidence_only = {name: evidence[name] for name in manifest_evidence(envelope.manifest)}
    started = time.perf_counter()
    bundle_envelope, bundle_metrics = client.get(
        logical_rank, locator.epoch, control_checksum=locator.control_payload_checksum,
        expected_manifest=manifest_evidence_only,
    )
    bundle_ms = (time.perf_counter() - started) * 1000.0
    bundle = deserialize_checkpoint(bundle_envelope.payload, bundle_envelope.manifest.payload_checksum)
    validate_checkpoint_objects(bundle, bundle_envelope.manifest.objects)
    if not isinstance(bundle, Mapping) or set(bundle) != _SNAPSHOT_KEYS:
        raise ValueError("base snapshot schema changed")
    state = bundle.get("snapshot")
    if (
        bundle.get("schema_version") != 1 or bundle.get("kind") != "object_semantic_base_snapshot"
        or bundle.get("run_id") != locator.run_id or bundle.get("epoch") != locator.epoch
        or bundle.get("source_rank") != logical_rank
        or bundle.get("source_host") != expected_layout[logical_rank].host
        or not isinstance(state, Mapping) or not _cpu(state)
    ):
        raise ValueError("base snapshot identity changed")
    binding, restored_topology = _base_metadata(
        state, epoch=locator.epoch, rank=logical_rank, world_size=locator.world_size,
    )
    base_id = f"sha256:{checkpoint_state_checksum(state)}"
    if (
        binding != _binding(bundle["binding"]) or binding != _binding(evidence["binding"])
        or restored_topology != _topology(bundle["parameter_ids"]) or restored_topology != topology
        or bundle.get("base_id") != base_id or evidence.get("base_id") != base_id
        or (expected_base_id is not None and expected_base_id != base_id)
    ):
        raise ValueError("base id, binding, or parameter topology changed")
    return RestoredObjectSemanticBase(
        copy.deepcopy(dict(state)), base_id, len(envelope.payload) + len(bundle_envelope.payload),
        metrics.wire_bytes + bundle_metrics.wire_bytes, max(control_ms, bundle_ms),
    )
