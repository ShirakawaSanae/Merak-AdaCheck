"""Sealed two-host transport for semantic optimizer-update objects."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .catalog import (
    deserialize_checkpoint,
    serialize_checkpoint,
    tensor_checksum,
    validate_checkpoint_objects,
)
from .global_commit import manifest_evidence
from .manifest import CheckpointManifest, create_prepared_manifest
from .matched import CONTROL_SOURCE_RANK
from .object_optimizer_update import (
    GradientPayload,
    ObjectOptimizerUpdate,
    materialize_optimizer_update,
)
from .object_semantic_plan import ActiveGradientPlan, ConfirmedGradientEncoding
from .object_semantic_schema import export_gradient_encoding_metadata
from .replica_service import ReplicaServiceClient
from .transport import ReplicaEnvelope


_BUNDLE_KEYS = frozenset(
    (
        "schema_version",
        "kind",
        "epoch",
        "arm",
        "source_rank",
        "source_host",
        "target_host",
        "detector_catalog_hash",
        "semantic_plan_hash",
        "plan_hash",
        "canonical_hash",
        "encoding_hash",
        "record",
        "occurrences",
    )
)
_OCCURRENCE_KEYS = frozenset(
    (
        "action_index",
        "version_id",
        "object_id",
        "epoch",
        "fqn",
        "parameter_id",
        "shape",
        "dtype",
        "checksum",
        "logical_bytes",
        "tensor",
    )
)
_CONTROL_KEYS = frozenset(
    (
        "schema_version",
        "kind",
        "run_id",
        "epoch",
        "world_size",
        "arm",
        "detector_catalog_hash",
        "semantic_plan_hash",
        "plan_hash",
        "canonical_hash",
        "encoding_hash",
        "gradient_encoding",
        "bundle_manifests_by_host",
    )
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class ObjectUpdateTransferMetrics:
    remote_gradient_logical_bytes: int
    remote_data_payload_bytes: int
    remote_control_payload_bytes: int
    remote_wire_bytes: int
    maximum_remote_put_ms: float

    @property
    def total_remote_payload_bytes(self) -> int:
        return self.remote_data_payload_bytes + self.remote_control_payload_bytes


@dataclass(frozen=True)
class ObjectUpdateLocator:
    run_id: str
    epoch: int
    world_size: int
    arm: str
    detector_catalog_hash: str
    semantic_plan_hash: str
    plan_hash: str
    canonical_hash: str
    encoding_hash: str
    control_payload_checksum: str
    control_manifests_by_host: tuple[tuple[int, dict[str, Any]], ...]
    locator_hash: str

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "object_optimizer_update_locator",
            "run_id": self.run_id,
            "epoch": self.epoch,
            "world_size": self.world_size,
            "arm": self.arm,
            "detector_catalog_hash": self.detector_catalog_hash,
            "semantic_plan_hash": self.semantic_plan_hash,
            "plan_hash": self.plan_hash,
            "canonical_hash": self.canonical_hash,
            "encoding_hash": self.encoding_hash,
            "control_payload_checksum": self.control_payload_checksum,
            "control_manifests_by_host": [
                [host, evidence]
                for host, evidence in self.control_manifests_by_host
            ],
        }

    def validate(self) -> None:
        body = self._body()
        if (
            not self.run_id
            or self.epoch < 0
            or self.world_size < 1
            or self.locator_hash != _digest(body)
            or tuple(host for host, _ in self.control_manifests_by_host) != (0, 1)
        ):
            raise ValueError("object update locator identity or checksum changed")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {**self._body(), "locator_hash": self.locator_hash}

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        plan: ActiveGradientPlan,
        encoding: ConfirmedGradientEncoding,
        control_payload_checksum: str,
        control_manifests_by_host: Mapping[int, CheckpointManifest],
    ) -> "ObjectUpdateLocator":
        evidence = tuple(
            (host, manifest_evidence(control_manifests_by_host[host]))
            for host in (0, 1)
        )
        value = cls(
            run_id,
            encoding.epoch,
            len(plan.rank_bindings),
            plan.arm,
            plan.detector_catalog_hash,
            plan.semantic_plan_hash,
            plan.plan_hash,
            plan.canonical_hash,
            encoding.encoding_hash,
            control_payload_checksum,
            evidence,
            "",
        )
        return replace(value, locator_hash=_digest(value._body()))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ObjectUpdateLocator":
        expected = {
            "schema_version",
            "kind",
            "run_id",
            "epoch",
            "world_size",
            "arm",
            "detector_catalog_hash",
            "semantic_plan_hash",
            "plan_hash",
            "canonical_hash",
            "encoding_hash",
            "control_payload_checksum",
            "control_manifests_by_host",
            "locator_hash",
        }
        if (
            set(value) != expected
            or value["schema_version"] != 1
            or value["kind"] != "object_optimizer_update_locator"
        ):
            raise ValueError("object update locator schema changed")
        locator = cls(
            str(value["run_id"]),
            int(value["epoch"]),
            int(value["world_size"]),
            str(value["arm"]),
            str(value["detector_catalog_hash"]),
            str(value["semantic_plan_hash"]),
            str(value["plan_hash"]),
            str(value["canonical_hash"]),
            str(value["encoding_hash"]),
            str(value["control_payload_checksum"]),
            tuple(
                (int(host), dict(evidence))
                for host, evidence in value["control_manifests_by_host"]
            ),
            str(value["locator_hash"]),
        )
        locator.validate()
        return locator


@dataclass(frozen=True)
class PublishedObjectUpdate:
    locator: ObjectUpdateLocator
    metrics: ObjectUpdateTransferMetrics
    timing: Mapping[str, float | None] | None = None


@dataclass(frozen=True)
class RestoredObjectUpdate:
    records: tuple[tuple[int, dict[str, Any]], ...]
    payload_bytes: int
    wire_bytes: int
    maximum_get_ms: float


@dataclass(frozen=True)
class FetchedObjectUpdateControl:
    value: dict[str, Any]
    payload_bytes: int
    wire_bytes: int
    get_ms: float


def write_object_update_locator(
    path: str | os.PathLike[str], locator: ObjectUpdateLocator
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(locator.to_dict(), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def load_object_update_locator(path: str | os.PathLike[str]) -> ObjectUpdateLocator:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, Mapping):
        raise ValueError("object update locator must be a JSON object")
    return ObjectUpdateLocator.from_dict(value)


def build_local_object_update_bundles(
    plan: ActiveGradientPlan,
    encoding: ConfirmedGradientEncoding,
    update: ObjectOptimizerUpdate,
    *,
    owned_payloads: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Create one bundle per target host without moving another rank's tensor."""
    if (
        update.epoch != encoding.epoch
        or update.detector_catalog_hash != plan.detector_catalog_hash
        or update.semantic_plan_hash != plan.semantic_plan_hash
        or update.plan_hash != plan.plan_hash
        or update.canonical_hash != plan.canonical_hash
        or update.encoding_hash != encoding.encoding_hash
        or encoding.canonical_hash != plan.canonical_hash
    ):
        raise ValueError("local object update has different semantic provenance")
    if len(update.records) != 1:
        raise ValueError("local object update must contain exactly one rank record")
    source_rank, record = update.records[0]
    bindings = {binding.rank: binding for binding in plan.rank_bindings}
    if source_rank not in bindings:
        raise ValueError("local object update rank is outside the plan")
    payloads = {payload.version_id: payload for payload in update.payloads}
    if len(payloads) != len(update.payloads):
        raise ValueError("local object update has duplicate payload versions")
    required = {
        action.version_id
        for action in encoding.actions
        if action.source_rank == source_rank
    }
    if set(payloads) != required:
        raise ValueError("local object update does not own its planned publications")

    bundles = []
    for target_host in (0, 1):
        occurrences = []
        for action_index, action in enumerate(encoding.actions):
            if action.source_rank != source_rank or action.target_host != target_host:
                continue
            payload = payloads[action.version_id]
            if (
                payload.object_id != action.object_id
                or payload.epoch != encoding.epoch
                or payload.checksum != action.checksum
                or payload.tensor.numel() * payload.tensor.element_size()
                != action.logical_bytes
            ):
                raise ValueError("published gradient disagrees with its placement")
            if owned_payloads and payload.tensor.device.type != "cpu":
                raise ValueError("owned payloads require CPU tensors")
            occurrences.append(
                {
                    "action_index": action_index,
                    "version_id": payload.version_id,
                    "object_id": payload.object_id,
                    "epoch": payload.epoch,
                    "fqn": payload.fqn,
                    "parameter_id": payload.parameter_id,
                    "shape": payload.shape,
                    "dtype": payload.dtype,
                    "checksum": payload.checksum,
                    "logical_bytes": action.logical_bytes,
                    # The default owns separate storage. The runtime-only path keeps
                    # its private snapshot alive until both serializers finish.
                    "tensor": (
                        payload.tensor.detach()
                        if owned_payloads
                        else payload.tensor.detach().cpu().clone()
                    ),
                }
            )
        bundles.append(
            {
                "schema_version": 1,
                "kind": "object_optimizer_update_bundle",
                "epoch": encoding.epoch,
                "arm": plan.arm,
                "source_rank": source_rank,
                "source_host": bindings[source_rank].host,
                "target_host": target_host,
                "detector_catalog_hash": plan.detector_catalog_hash,
                "semantic_plan_hash": plan.semantic_plan_hash,
                "plan_hash": plan.plan_hash,
                "canonical_hash": plan.canonical_hash,
                "encoding_hash": encoding.encoding_hash,
                "record": copy.deepcopy(record),
                "occurrences": occurrences,
            }
        )
    return tuple(bundles)


def commit_object_update_state(
    client: ReplicaServiceClient,
    state: Mapping[str, Any],
    *,
    source_rank: int,
    source_host: int,
    target_host: int,
    target_rank: int,
    group_ranks: tuple[int, ...],
    object_checksums: bool = True,
) -> tuple[CheckpointManifest, int, float]:
    serialized = serialize_checkpoint(
        state,
        state_is_snapshot=True,
        object_checksums=object_checksums,
    )
    prepared = create_prepared_manifest(
        epoch=int(state["epoch"]),
        source_rank=source_rank,
        source_host=f"host-{source_host}",
        target_rank=target_rank,
        target_host=client.endpoint,
        group_ranks=group_ranks,
        serialized=serialized,
    )
    started = time.perf_counter()
    put_metrics = client.put_prepared(ReplicaEnvelope(prepared, serialized.payload))
    put_ms = (time.perf_counter() - started) * 1000.0
    committed = replace(prepared, state="committed", committed_time_ns=time.time_ns())
    commit_metrics = client.commit(committed)
    return committed, put_metrics.wire_bytes + commit_metrics.wire_bytes, put_ms


def create_object_update_control(
    *,
    run_id: str,
    plan: ActiveGradientPlan,
    encoding: ConfirmedGradientEncoding,
    bundle_evidence_by_host: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if set(bundle_evidence_by_host) != {0, 1}:
        raise ValueError("object update control requires both target hosts")
    expected_ranks = {binding.rank for binding in plan.rank_bindings}
    normalized = {}
    for host in (0, 1):
        evidence = [dict(value) for value in bundle_evidence_by_host[host]]
        if (
            len(evidence) != len(expected_ranks)
            or {int(value.get("source_rank", -1)) for value in evidence}
            != expected_ranks
            or any(int(value.get("epoch", -1)) != encoding.epoch for value in evidence)
        ):
            raise ValueError("object update control has incomplete bundle evidence")
        normalized[host] = sorted(evidence, key=lambda value: int(value["source_rank"]))
    return {
        "schema_version": 1,
        "kind": "object_optimizer_update_control",
        "run_id": run_id,
        "epoch": encoding.epoch,
        "world_size": len(plan.rank_bindings),
        "arm": plan.arm,
        "detector_catalog_hash": plan.detector_catalog_hash,
        "semantic_plan_hash": plan.semantic_plan_hash,
        "plan_hash": plan.plan_hash,
        "canonical_hash": plan.canonical_hash,
        "encoding_hash": encoding.encoding_hash,
        "gradient_encoding": export_gradient_encoding_metadata(plan, encoding),
        "bundle_manifests_by_host": [
            [host, normalized[host]] for host in (0, 1)
        ],
    }


def seal_object_update_host(
    client: ReplicaServiceClient,
    *,
    epoch: int,
    control_checksum: str,
    control_evidence: Mapping[str, Any],
    bundle_evidence: Sequence[Mapping[str, Any]],
) -> None:
    evidence = [*bundle_evidence, control_evidence]
    for value in evidence:
        client.pin(
            kind="record",
            source_rank=int(value["source_rank"]),
            epoch=epoch,
            control_checksum=control_checksum,
            expected_manifest=value,
        )
    client.seal(
        kind="record",
        source_rank=CONTROL_SOURCE_RANK,
        epoch=epoch,
        control_checksum=control_checksum,
    )
    for value in bundle_evidence:
        client.seal(
            kind="record",
            source_rank=int(value["source_rank"]),
            epoch=epoch,
            control_checksum=control_checksum,
        )


def publish_object_update_epoch(
    clients: Mapping[int, ReplicaServiceClient],
    *,
    run_id: str,
    plan: ActiveGradientPlan,
    encoding: ConfirmedGradientEncoding,
    local_updates: Mapping[int, ObjectOptimizerUpdate],
    owned_payloads: bool = False,
) -> PublishedObjectUpdate:
    """CPU oracle/coordinator; the distributed runtime performs the same phases."""
    bindings = {binding.rank: binding for binding in plan.rank_bindings}
    if set(clients) != {0, 1} or set(local_updates) != set(bindings):
        raise ValueError("object update publication requires both hosts and every rank")
    target_ranks = {
        host: min(rank for rank, binding in bindings.items() if binding.host == host)
        for host in (0, 1)
    }
    bundle_manifests: dict[int, list[CheckpointManifest]] = {0: [], 1: []}
    remote_data_payload = 0
    remote_wire = 0
    remote_put_times = []
    all_ranks = tuple(sorted(bindings))
    for rank in all_ranks:
        for bundle in build_local_object_update_bundles(
            plan, encoding, local_updates[rank], owned_payloads=owned_payloads
        ):
            target_host = int(bundle["target_host"])
            manifest, wire_bytes, put_ms = commit_object_update_state(
                clients[target_host],
                bundle,
                source_rank=rank,
                source_host=bindings[rank].host,
                target_host=target_host,
                target_rank=target_ranks[target_host],
                group_ranks=all_ranks,
                object_checksums=False,
            )
            bundle_manifests[target_host].append(manifest)
            if bindings[rank].host != target_host:
                remote_data_payload += manifest.payload_bytes
                remote_wire += wire_bytes
                remote_put_times.append(put_ms)

    bundle_evidence = {
        host: [manifest_evidence(value) for value in bundle_manifests[host]]
        for host in (0, 1)
    }
    control = create_object_update_control(
        run_id=run_id,
        plan=plan,
        encoding=encoding,
        bundle_evidence_by_host=bundle_evidence,
    )
    control_manifests = {}
    control_payload_bytes = 0
    for host in (0, 1):
        manifest, wire_bytes, put_ms = commit_object_update_state(
            clients[host],
            control,
            source_rank=CONTROL_SOURCE_RANK,
            source_host=0,
            target_host=host,
            target_rank=target_ranks[host],
            group_ranks=all_ranks,
            object_checksums=False,
        )
        control_manifests[host] = manifest
        if host != 0:
            control_payload_bytes += manifest.payload_bytes
            remote_wire += wire_bytes
            remote_put_times.append(put_ms)
    control_checksum = control_manifests[0].payload_checksum
    if any(
        manifest.payload_checksum != control_checksum
        for manifest in control_manifests.values()
    ):
        raise ValueError("two hosts committed different object controls")

    for host, client in clients.items():
        seal_object_update_host(
            client,
            epoch=encoding.epoch,
            control_checksum=control_checksum,
            control_evidence=manifest_evidence(control_manifests[host]),
            bundle_evidence=bundle_evidence[host],
        )

    locator = ObjectUpdateLocator.create(
        run_id=run_id,
        plan=plan,
        encoding=encoding,
        control_payload_checksum=control_checksum,
        control_manifests_by_host=control_manifests,
    )
    return PublishedObjectUpdate(
        locator,
        ObjectUpdateTransferMetrics(
            encoding.total_remote_logical_bytes,
            remote_data_payload,
            control_payload_bytes,
            remote_wire,
            max(remote_put_times, default=0.0),
        ),
    )


def _validate_identity(
    value: Mapping[str, Any],
    *,
    locator: ObjectUpdateLocator,
    plan: ActiveGradientPlan,
    encoding: ConfirmedGradientEncoding,
) -> None:
    expected = {
        "epoch": locator.epoch,
        "arm": plan.arm,
        "detector_catalog_hash": plan.detector_catalog_hash,
        "semantic_plan_hash": plan.semantic_plan_hash,
        "plan_hash": plan.plan_hash,
        "canonical_hash": plan.canonical_hash,
        "encoding_hash": encoding.encoding_hash,
    }
    if any(value.get(name) != expected_value for name, expected_value in expected.items()):
        raise ValueError("restored object update provenance changed")


def fetch_object_update_control(
    client: ReplicaServiceClient,
    *,
    surviving_host: int,
    locator: ObjectUpdateLocator,
) -> FetchedObjectUpdateControl:
    """Fetch one sealed tensor-free authority using only its external locator."""
    locator.validate()
    if surviving_host not in (0, 1):
        raise ValueError("surviving object service host must be 0 or 1")
    control_evidence = dict(locator.control_manifests_by_host)[surviving_host]
    started = time.perf_counter()
    envelope, metrics = client.get(
        CONTROL_SOURCE_RANK,
        locator.epoch,
        control_checksum=locator.control_payload_checksum,
        expected_manifest=control_evidence,
    )
    get_ms = (time.perf_counter() - started) * 1000.0
    control = deserialize_checkpoint(
        envelope.payload, locator.control_payload_checksum
    )
    validate_checkpoint_objects(control, envelope.manifest.objects)
    if (
        not isinstance(control, dict)
        or set(control) != _CONTROL_KEYS
        or control["schema_version"] != 1
        or control["kind"] != "object_optimizer_update_control"
        or control["run_id"] != locator.run_id
        or control["epoch"] != locator.epoch
        or control["world_size"] != locator.world_size
    ):
        raise ValueError("object update control schema or identity changed")
    return FetchedObjectUpdateControl(
        control, len(envelope.payload), metrics.wire_bytes, get_ms
    )


def restore_object_update_epoch(
    client: ReplicaServiceClient,
    *,
    surviving_host: int,
    locator: ObjectUpdateLocator,
    plan: ActiveGradientPlan,
    encoding: ConfirmedGradientEncoding,
    logical_rank: int | None = None,
    fetched_control: FetchedObjectUpdateControl | None = None,
) -> RestoredObjectUpdate:
    """Restore every update, or only one logical rank and its payload sources."""
    locator.validate()
    if (
        surviving_host not in (0, 1)
        or locator.epoch != encoding.epoch
        or locator.world_size != len(plan.rank_bindings)
        or locator.arm != plan.arm
        or locator.detector_catalog_hash != plan.detector_catalog_hash
        or locator.semantic_plan_hash != plan.semantic_plan_hash
        or locator.plan_hash != plan.plan_hash
        or locator.canonical_hash != plan.canonical_hash
        or locator.encoding_hash != encoding.encoding_hash
    ):
        raise ValueError("object update locator differs from the expected plan")
    if fetched_control is not None and not isinstance(
        fetched_control, FetchedObjectUpdateControl
    ):
        raise ValueError("object update fetched control is invalid")
    fetched = fetched_control or fetch_object_update_control(
        client, surviving_host=surviving_host, locator=locator
    )
    control = fetched.value
    get_times = [fetched.get_ms]
    payload_bytes = fetched.payload_bytes
    wire_bytes = fetched.wire_bytes
    _validate_identity(control, locator=locator, plan=plan, encoding=encoding)
    manifests_by_host = dict(control.get("bundle_manifests_by_host", ()))
    if set(manifests_by_host) != {0, 1}:
        raise ValueError("object update control does not cover both hosts")
    expected_evidence = manifests_by_host[surviving_host]
    if (
        not isinstance(expected_evidence, list)
        or len(expected_evidence) != locator.world_size
    ):
        raise ValueError("object update control has incomplete rank bundles")

    bindings = {binding.rank: binding for binding in plan.rank_bindings}
    evidence_by_rank: dict[int, Mapping[str, Any]] = {}
    for evidence in expected_evidence:
        if not isinstance(evidence, Mapping):
            raise ValueError("object update control has invalid rank bundles")
        source_rank = evidence.get("source_rank")
        if (
            type(source_rank) is not int
            or source_rank not in bindings
            or source_rank in evidence_by_rank
        ):
            raise ValueError("object update control has invalid rank bundles")
        evidence_by_rank[source_rank] = evidence
    if set(evidence_by_rank) != set(bindings):
        raise ValueError("object update control has incomplete rank bundles")

    required_versions = set(dict(encoding.required_versions_by_host)[surviving_host])
    selected_ranks = set(bindings)
    if logical_rank is not None:
        if type(logical_rank) is not int or logical_rank not in bindings:
            raise ValueError("object update target logical rank is invalid")
        actions_by_reference = {}
        for action in encoding.actions:
            key = (
                action.object_id,
                action.source_rank if action.action == "direct_fallback" else None,
            )
            actions_by_reference.setdefault(key, action)
        target_actions = []
        for item in plan.objects:
            if logical_rank not in item.holders:
                continue
            referenced = actions_by_reference.get(
                (
                    item.object_id,
                    logical_rank if item.object_id in encoding.mismatched_object_ids else None,
                )
            )
            if referenced is None:
                raise ValueError("object update target lacks a payload reference")
            candidates = [
                action
                for action in encoding.actions
                if action.object_id == item.object_id
                and action.target_host == surviving_host
                and action.version_id == referenced.version_id
                and action.action == referenced.action
                and (
                    referenced.action != "direct_fallback"
                    or action.source_rank == referenced.source_rank
                )
            ]
            if not candidates:
                raise ValueError("object update target publisher is unavailable")
            target_actions.append(candidates[0])
        required_versions = {action.version_id for action in target_actions}
        publishers = {action.source_rank for action in target_actions}
        selected_ranks = publishers | {logical_rank}

    records: dict[int, dict[str, Any]] = {}
    payloads: dict[str, GradientPayload] = {}
    observed_actions = set()
    for source_rank in sorted(selected_ranks):
        evidence = evidence_by_rank[source_rank]
        started = time.perf_counter()
        envelope, metrics = client.get(
            source_rank,
            locator.epoch,
            control_checksum=locator.control_payload_checksum,
            expected_manifest=evidence,
        )
        get_times.append((time.perf_counter() - started) * 1000.0)
        payload_bytes += len(envelope.payload)
        wire_bytes += metrics.wire_bytes
        bundle = deserialize_checkpoint(
            envelope.payload, envelope.manifest.payload_checksum
        )
        validate_checkpoint_objects(bundle, envelope.manifest.objects)
        if not isinstance(bundle, Mapping) or set(bundle) != _BUNDLE_KEYS:
            raise ValueError("object update bundle schema changed")
        _validate_identity(bundle, locator=locator, plan=plan, encoding=encoding)
        if (
            bundle["schema_version"] != 1
            or bundle["kind"] != "object_optimizer_update_bundle"
            or bundle["source_rank"] != source_rank
            or source_rank not in bindings
            or bundle["source_host"] != bindings[source_rank].host
            or bundle["target_host"] != surviving_host
        ):
            raise ValueError("object update bundle rank or target changed")
        if logical_rank is None or source_rank == logical_rank:
            records[source_rank] = dict(bundle["record"])
        for occurrence in bundle["occurrences"]:
            if not isinstance(occurrence, Mapping) or set(occurrence) != _OCCURRENCE_KEYS:
                raise ValueError("object update occurrence schema changed")
            action_index = int(occurrence["action_index"])
            if not 0 <= action_index < len(encoding.actions):
                raise ValueError("object update occurrence has an invalid action")
            action = encoding.actions[action_index]
            tensor = occurrence["tensor"]
            expected_action = (
                action.source_rank == source_rank
                and action.target_host == surviving_host
                and action.version_id == occurrence["version_id"]
                and action.object_id == occurrence["object_id"]
                and action.checksum == occurrence["checksum"]
                and action.logical_bytes == occurrence["logical_bytes"]
            )
            if (
                not expected_action
                or action_index in observed_actions
                or not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != tuple(occurrence["shape"])
                or str(tensor.dtype) != occurrence["dtype"]
                or tensor.numel() * tensor.element_size() != action.logical_bytes
                or tensor_checksum(tensor) != action.checksum
            ):
                raise ValueError("object update occurrence failed validation")
            observed_actions.add(action_index)
            if occurrence["version_id"] not in required_versions:
                continue
            payload = GradientPayload(
                str(occurrence["version_id"]),
                str(occurrence["object_id"]),
                int(occurrence["epoch"]),
                str(occurrence["fqn"]),
                str(occurrence["parameter_id"]),
                tuple(int(size) for size in occurrence["shape"]),
                str(occurrence["dtype"]),
                str(occurrence["checksum"]),
                tensor.detach().cpu().clone(),
            )
            previous = payloads.setdefault(payload.version_id, payload)
            if (
                previous.object_id != payload.object_id
                or previous.parameter_id != payload.parameter_id
                or previous.checksum != payload.checksum
            ):
                raise ValueError("duplicate object versions disagree")

    expected_actions = {
        index
        for index, action in enumerate(encoding.actions)
        if action.target_host == surviving_host and action.source_rank in selected_ranks
    }
    expected_records = set(bindings) if logical_rank is None else {logical_rank}
    if (
        set(records) != expected_records
        or observed_actions != expected_actions
        or set(payloads) != required_versions
    ):
        raise ValueError("surviving host lacks a complete object update")
    update = ObjectOptimizerUpdate(
        locator.epoch,
        locator.detector_catalog_hash,
        locator.semantic_plan_hash,
        locator.plan_hash,
        locator.canonical_hash,
        locator.encoding_hash,
        tuple(sorted(records.items())),
        tuple(payloads[key] for key in sorted(payloads)),
    )
    restored = tuple(
        (
            rank,
            materialize_optimizer_update(
                update,
                rank,
                payloads=payloads if logical_rank is not None else None,
                expected_epoch=locator.epoch,
                detector_catalog_hash=locator.detector_catalog_hash,
                semantic_plan_hash=locator.semantic_plan_hash,
                plan_hash=locator.plan_hash,
                canonical_hash=locator.canonical_hash,
                encoding_hash=locator.encoding_hash,
            ),
        )
        for rank in sorted(records)
    )
    return RestoredObjectUpdate(
        restored,
        payload_bytes,
        wire_bytes,
        max(get_times, default=0.0),
    )
