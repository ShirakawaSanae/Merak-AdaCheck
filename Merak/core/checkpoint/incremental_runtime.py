"""Base-plus-update checkpoint session built on the replica service."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .async_checkpoint import AsyncCommitResult, AsyncServiceRemoteCheckpointSession
from .incremental_checkpoint import (
    RecoveryBinding,
    capture_optimizer_update,
)
from .catalog import (
    deserialize_checkpoint,
    iter_checkpoint_objects,
    validate_checkpoint_objects,
)
from .global_commit import manifest_evidence, rank_evidence
from .recovery import CommitResult, ServiceRemoteCheckpointSession
from .replica_service import ReplicaServiceClient


@dataclass(frozen=True)
class IncrementalChain:
    base_epoch: int
    final_epoch: int
    base_id: str
    base_state: Any
    updates: tuple[Mapping[str, Any], ...]
    restore_wire_bytes: int


@dataclass(frozen=True)
class PeerIncrementalCommit:
    epoch: int
    local_full: AsyncCommitResult
    remote_delta: AsyncCommitResult


@dataclass(frozen=True)
class PeerIncrementalChain:
    full: IncrementalChain
    runtime_states: tuple[Mapping[str, Any], ...]
    delta_restore_wire_bytes: int


def _validate_service_manifest(manifest, *, source_rank: int, epoch: int) -> None:
    if (
        manifest.source_rank != source_rank
        or manifest.epoch != epoch
        or not manifest.committed
    ):
        raise ValueError("replica service returned an invalid incremental record")


def write_optimizer_update_catalog(
    root: Path, record: Mapping[str, Any], *, run_id: str
) -> Path:
    """Write an opt-in tensor catalog for one pre-optimizer update."""
    objects = tuple(iter_checkpoint_objects(record, include_checksums=True))
    gradients = tuple(item for item in objects if item.role == "gradients")
    missing_gradients = sum(
        gradient is None
        for group in record["gradients"]
        for gradient in group
    )
    binding = record["binding"]
    semantic_encoder = record.get("metadata", {}).get("adacheck_semantic_encoder")
    snapshot_point = (
        semantic_encoder.get("gradient_snapshot_point")
        if isinstance(semantic_encoder, Mapping)
        else "post_unscale_pre_optimizer_step"
    )
    body = {
        "schema_version": 1,
        "run_kind": "merak_pre_optimizer_update_catalog",
        "snapshot_point": snapshot_point,
        "gradient_multiplier": float(record.get("gradient_multiplier", 1.0)),
        "run_id": run_id,
        "rank": int(binding["rank"]),
        "world_size": int(binding["world_size"]),
        "epoch": int(record["epoch"]),
        "base_epoch": int(record["base_epoch"]),
        "base_id": str(record["base_id"]),
        "binding": dict(binding),
        "topology_hash": str(record["topology_hash"]),
        "parameter_ids": record["parameter_ids"],
        "gradient_tensor_count": len(gradients),
        "missing_gradient_count": missing_gradients,
        "gradient_logical_bytes": sum(item.logical_bytes for item in gradients),
        "tensor_logical_bytes": sum(item.logical_bytes for item in objects),
        "objects": [item.to_dict() for item in objects],
        "adacheck_detector_filtered": False,
        "adacheck_payload_faithful": False,
        "eligible_for_runtime_gate": False,
        "limitations": [
            "dense gradients have not been filtered by AdaCheck redundancy evidence",
            "non-tensor metadata and wire framing bytes are not counted",
            "matching on the evaluated epoch would be a hindsight oracle",
            "synchronous device-to-host hashing perturbs checkpoint timing",
        ],
    }
    body["catalog_hash"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    destination = (
        root
        / f"epoch_{body['epoch']:08d}"
        / f"rank_{body['rank']:05d}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return destination


class IncrementalCheckpointSession:
    """Publish one full base followed by a contiguous asynchronous update log."""

    def __init__(
        self,
        session: ServiceRemoteCheckpointSession,
        optimizer: torch.optim.Optimizer,
        *,
        parameter_ids: Sequence[Sequence[str]],
        binding: RecoveryBinding,
        max_inflight: int = 2,
        update_catalog_dir: Path | None = None,
        run_id: str = "test",
        gradient_dtype: torch.dtype | None = None,
    ) -> None:
        self.session = session
        self.optimizer = optimizer
        self.parameter_ids = tuple(tuple(group) for group in parameter_ids)
        self.binding = binding
        self.max_inflight = max_inflight
        self.update_catalog_dir = update_catalog_dir
        self.run_id = run_id
        self.gradient_dtype = gradient_dtype
        self.base_epoch: int | None = None
        self.base_id: str | None = None
        self.publisher: AsyncServiceRemoteCheckpointSession | None = None

    def save_base(self, epoch: int, state: Any) -> CommitResult:
        if self.base_epoch is not None:
            raise RuntimeError("incremental base is already committed")
        result = self.session.save(epoch, state)
        self.base_epoch = epoch
        self.base_id = f"sha256:{result.manifest.payload_checksum}"
        self.publisher = AsyncServiceRemoteCheckpointSession(
            self.session,
            max_inflight=self.max_inflight,
            object_checksums=False,
        )
        return result

    def enqueue_update(
        self,
        epoch: int,
        *,
        metadata: Mapping[str, Any] | None = None,
        runtime_state: Mapping[str, Any] | None = None,
        gradient_multiplier: float = 1.0,
    ) -> float:
        if self.publisher is None or self.base_epoch is None or self.base_id is None:
            raise RuntimeError("incremental base must be committed before updates")
        record = capture_optimizer_update(
            self.optimizer,
            parameter_ids=self.parameter_ids,
            binding=self.binding,
            base_id=self.base_id,
            base_epoch=self.base_epoch,
            epoch=epoch,
            metadata=metadata,
            runtime_state=runtime_state,
            snapshot_gradients=False,
            gradient_dtype=self.gradient_dtype,
            gradient_multiplier=gradient_multiplier,
        )
        if self.update_catalog_dir is not None:
            write_optimizer_update_catalog(
                self.update_catalog_dir, record, run_id=self.run_id
            )
        return self.publisher.enqueue(epoch, record)

    def poll(self) -> tuple[AsyncCommitResult, ...]:
        return () if self.publisher is None else self.publisher.poll()

    def close(self) -> tuple[AsyncCommitResult, ...]:
        return () if self.publisher is None else self.publisher.close()

    def restore_chain(self, final_epoch: int) -> IncrementalChain:
        if self.base_epoch is None or self.base_id is None:
            raise RuntimeError("incremental base is unknown")
        if final_epoch < self.base_epoch:
            raise ValueError("final epoch predates incremental base")
        return restore_incremental_chain(
            self.session,
            base_epoch=self.base_epoch,
            final_epoch=final_epoch,
        )

    def pin_epoch(self, control: Mapping[str, Any]) -> None:
        rank = rank_evidence(control, self.binding.rank)
        self.session.client.pin(
            kind="materialized",
            source_rank=self.binding.rank,
            epoch=int(control["epoch"]),
            control_checksum=str(control["control_checksum"]),
            expected_manifest=rank["paths"]["remote_full"],
        )

    def seal_epoch(self, control: Mapping[str, Any]) -> None:
        self.session.client.seal(
            kind="materialized",
            source_rank=self.binding.rank,
            epoch=int(control["epoch"]),
            control_checksum=str(control["control_checksum"]),
        )

    def wait_sealed(self, epoch: int) -> None:
        self.session.client.wait_materialized(self.binding.rank, epoch)


class PeerIncrementalCheckpointSession:
    """Keep a full local peer log and send only rank-local state remotely."""

    def __init__(
        self,
        local_full_session: ServiceRemoteCheckpointSession,
        remote_delta_session: ServiceRemoteCheckpointSession,
        optimizer: torch.optim.Optimizer,
        *,
        parameter_ids: Sequence[Sequence[str]],
        binding: RecoveryBinding,
        peer_rank: int,
        max_inflight: int = 2,
        run_id: str = "test",
        local_update_session: Any | None = None,
        update_catalog_dir: Path | None = None,
        gradient_dtype: torch.dtype | None = None,
    ) -> None:
        self.local_full_session = local_full_session
        self.remote_delta_session = remote_delta_session
        self.optimizer = optimizer
        self.parameter_ids = tuple(tuple(group) for group in parameter_ids)
        self.binding = binding
        self.peer_rank = peer_rank
        self.max_inflight = max_inflight
        self.run_id = run_id
        self.local_update_session = local_update_session or local_full_session
        self.update_catalog_dir = update_catalog_dir
        self.gradient_dtype = gradient_dtype
        self.base_epoch: int | None = None
        self.base_id: str | None = None
        self._full_publisher: AsyncServiceRemoteCheckpointSession | None = None
        self._delta_publisher: AsyncServiceRemoteCheckpointSession | None = None
        self._full_completed: dict[int, AsyncCommitResult] = {}
        self._delta_completed: dict[int, AsyncCommitResult] = {}

    def save_base(
        self,
        epoch: int,
        full_state: Any,
        *,
        runtime_state: Mapping[str, Any],
    ) -> tuple[CommitResult, CommitResult]:
        full = self.local_full_session.save(epoch, full_state)
        delta = self.remote_delta_session.save(
            epoch,
            {
                "schema_version": 1,
                "kind": "peer_runtime_delta",
                "source_rank": self.binding.rank,
                "peer_rank": self.peer_rank,
                "epoch": epoch,
                "run_id": self.run_id,
                "runtime_state": runtime_state,
                "adacheck_semantic": full_state.get("adacheck_semantic"),
                "adacheck_semantic_encoder": full_state.get(
                    "adacheck_semantic_encoder"
                ),
            },
        )
        self.base_epoch = epoch
        self.base_id = f"sha256:{full.manifest.payload_checksum}"
        self._full_publisher = AsyncServiceRemoteCheckpointSession(
            self.local_update_session,
            max_inflight=self.max_inflight,
            object_checksums=False,
        )
        self._delta_publisher = AsyncServiceRemoteCheckpointSession(
            self.remote_delta_session,
            max_inflight=self.max_inflight,
            object_checksums=False,
        )
        return full, delta

    def enqueue_update(
        self,
        epoch: int,
        *,
        metadata: Mapping[str, Any] | None = None,
        runtime_state: Mapping[str, Any],
        gradient_multiplier: float = 1.0,
    ) -> tuple[float, float]:
        if (
            self.base_epoch is None
            or self.base_id is None
            or self._full_publisher is None
            or self._delta_publisher is None
        ):
            raise RuntimeError("peer incremental base must be committed before updates")
        full_record = capture_optimizer_update(
            self.optimizer,
            parameter_ids=self.parameter_ids,
            binding=self.binding,
            base_id=self.base_id,
            base_epoch=self.base_epoch,
            epoch=epoch,
            metadata=metadata,
            runtime_state=runtime_state,
            snapshot_gradients=False,
            gradient_dtype=self.gradient_dtype,
            gradient_multiplier=gradient_multiplier,
        )
        if self.update_catalog_dir is not None:
            write_optimizer_update_catalog(
                self.update_catalog_dir, full_record, run_id=self.run_id
            )
        delta_record = {
            "schema_version": 1,
            "kind": "peer_runtime_delta",
            "source_rank": self.binding.rank,
            "peer_rank": self.peer_rank,
            "epoch": epoch,
            "run_id": self.run_id,
            "runtime_state": runtime_state,
        }
        if metadata is not None:
            for name in ("adacheck_semantic", "adacheck_semantic_encoder"):
                if name in metadata:
                    delta_record[name] = metadata[name]
        full_ms = self._full_publisher.enqueue(epoch, full_record)
        delta_ms = self._delta_publisher.enqueue_snapshot(epoch, delta_record)
        return full_ms, delta_ms

    def _paired(self) -> tuple[PeerIncrementalCommit, ...]:
        epochs = sorted(set(self._full_completed) & set(self._delta_completed))
        return tuple(
            PeerIncrementalCommit(
                epoch,
                self._full_completed.pop(epoch),
                self._delta_completed.pop(epoch),
            )
            for epoch in epochs
        )

    def poll(self) -> tuple[PeerIncrementalCommit, ...]:
        if self._full_publisher is None or self._delta_publisher is None:
            return ()
        self._full_completed.update(
            (item.epoch, item) for item in self._full_publisher.poll()
        )
        self._delta_completed.update(
            (item.epoch, item) for item in self._delta_publisher.poll()
        )
        return self._paired()

    def close(self) -> tuple[PeerIncrementalCommit, ...]:
        if self._full_publisher is None or self._delta_publisher is None:
            return ()
        self._full_completed.update(
            (item.epoch, item) for item in self._full_publisher.close()
        )
        self._delta_completed.update(
            (item.epoch, item) for item in self._delta_publisher.close()
        )
        results = self._paired()
        if self._full_completed or self._delta_completed:
            raise RuntimeError("peer incremental log committed only one side of an epoch")
        return results

    def pin_epoch(self, control: Mapping[str, Any]) -> None:
        rank = rank_evidence(control, self.binding.rank)
        checksum = str(control["control_checksum"])
        epoch = int(control["epoch"])
        self.local_full_session.client.pin(
            kind="materialized",
            source_rank=self.binding.rank,
            epoch=epoch,
            control_checksum=checksum,
            expected_manifest=rank["paths"]["local_full"],
        )
        self.remote_delta_session.client.pin(
            kind="record",
            source_rank=self.binding.rank,
            epoch=epoch,
            control_checksum=checksum,
            expected_manifest=rank["paths"]["remote_delta"],
        )

    def seal_epoch(self, control: Mapping[str, Any]) -> None:
        checksum = str(control["control_checksum"])
        epoch = int(control["epoch"])
        self.local_full_session.client.seal(
            kind="materialized",
            source_rank=self.binding.rank,
            epoch=epoch,
            control_checksum=checksum,
        )
        self.remote_delta_session.client.seal(
            kind="record",
            source_rank=self.binding.rank,
            epoch=epoch,
            control_checksum=checksum,
        )

    def wait_sealed(self, epoch: int) -> None:
        self.local_full_session.client.wait_materialized(self.binding.rank, epoch)
        wait_applied = getattr(self.local_update_session, "wait_applied", None)
        if wait_applied is not None:
            wait_applied(epoch)


def restore_incremental_chain(
    session: ServiceRemoteCheckpointSession,
    *,
    base_epoch: int,
    final_epoch: int,
) -> IncrementalChain:
    """Fetch and validate a committed base and every update through final_epoch."""
    if final_epoch < base_epoch:
        raise ValueError("final epoch predates incremental base")
    states = []
    wire_bytes = 0
    base_id = ""
    for epoch in range(base_epoch, final_epoch + 1):
        envelope, metrics = session.client.get(session.rank, epoch)
        manifest = envelope.manifest
        _validate_service_manifest(
            manifest, source_rank=session.rank, epoch=epoch
        )
        if (
            manifest.group_ranks != session.group_ranks
            or manifest.target_rank != session.target_rank
        ):
            raise ValueError("incremental record has incompatible placement metadata")
        state = deserialize_checkpoint(envelope.payload, manifest.payload_checksum)
        validate_checkpoint_objects(state, manifest.objects)
        if epoch == base_epoch:
            base_id = f"sha256:{manifest.payload_checksum}"
        states.append(state)
        wire_bytes += metrics.wire_bytes
    return IncrementalChain(
        base_epoch=base_epoch,
        final_epoch=final_epoch,
        base_id=base_id,
        base_state=states[0],
        updates=tuple(states[1:]),
        restore_wire_bytes=wire_bytes,
    )


def restore_peer_incremental_chain(
    client: ReplicaServiceClient,
    *,
    failed_rank: int,
    peer_rank: int,
    base_epoch: int,
    final_epoch: int,
) -> PeerIncrementalChain:
    """Recover a failed rank from its surviving peer's full log and deltas."""
    if final_epoch < base_epoch:
        raise ValueError("final epoch predates incremental base")
    full_states = []
    runtime_states = []
    full_wire_bytes = 0
    delta_wire_bytes = 0
    base_id = ""
    for epoch in range(base_epoch, final_epoch + 1):
        full_envelope, full_metrics = client.get(peer_rank, epoch)
        full_manifest = full_envelope.manifest
        _validate_service_manifest(
            full_manifest, source_rank=peer_rank, epoch=epoch
        )
        full_state = deserialize_checkpoint(
            full_envelope.payload, full_manifest.payload_checksum
        )
        validate_checkpoint_objects(full_state, full_manifest.objects)
        if epoch == base_epoch:
            base_id = f"sha256:{full_manifest.payload_checksum}"
        full_states.append(full_state)
        full_wire_bytes += full_metrics.wire_bytes

        delta_envelope, delta_metrics = client.get(failed_rank, epoch)
        _validate_service_manifest(
            delta_envelope.manifest, source_rank=failed_rank, epoch=epoch
        )
        delta = deserialize_checkpoint(
            delta_envelope.payload, delta_envelope.manifest.payload_checksum
        )
        validate_checkpoint_objects(delta, delta_envelope.manifest.objects)
        if (
            delta.get("kind") != "peer_runtime_delta"
            or int(delta.get("source_rank", -1)) != failed_rank
            or int(delta.get("peer_rank", -1)) != peer_rank
            or int(delta.get("epoch", -1)) != epoch
        ):
            raise ValueError("peer runtime delta does not match the recovery chain")
        metadata = full_state.get("metadata")
        semantic = full_state.get("adacheck_semantic")
        if semantic is None and isinstance(metadata, Mapping):
            semantic = metadata.get("adacheck_semantic")
        if semantic is not None and delta.get("adacheck_semantic") != semantic:
            raise ValueError("peer runtime delta has incompatible AdaCheck semantic binding")
        semantic_encoder = full_state.get("adacheck_semantic_encoder")
        if semantic_encoder is None and isinstance(metadata, Mapping):
            semantic_encoder = metadata.get("adacheck_semantic_encoder")
        if (
            semantic_encoder is not None
            and delta.get("adacheck_semantic_encoder") != semantic_encoder
        ):
            raise ValueError("peer runtime delta has incompatible SemanticFixed plan")
        runtime_states.append(delta["runtime_state"])
        delta_wire_bytes += delta_metrics.wire_bytes
    return PeerIncrementalChain(
        full=IncrementalChain(
            base_epoch=base_epoch,
            final_epoch=final_epoch,
            base_id=base_id,
            base_state=full_states[0],
            updates=tuple(full_states[1:]),
            restore_wire_bytes=full_wire_bytes,
        ),
        runtime_states=tuple(runtime_states),
        delta_restore_wire_bytes=delta_wire_bytes,
    )


def restore_peer_materialized(
    client: ReplicaServiceClient,
    *,
    failed_rank: int,
    peer_rank: int,
    epoch: int,
    group_ranks: tuple[int, ...],
    run_id: str,
    expected_full_evidence: Mapping[str, Any],
    expected_delta_evidence: Mapping[str, Any],
    control_checksum: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Fetch one materialized peer state and the failed rank's runtime state."""
    full_envelope, _ = client.get_materialized(
        peer_rank,
        epoch,
        control_checksum=control_checksum,
        expected_manifest=expected_full_evidence,
    )
    _validate_service_manifest(
        full_envelope.manifest, source_rank=peer_rank, epoch=epoch
    )
    if (
        full_envelope.manifest.group_ranks != group_ranks
        or full_envelope.manifest.target_rank != peer_rank
    ):
        raise ValueError("materialized checkpoint has an invalid peer placement")
    state = deserialize_checkpoint(
        full_envelope.payload, full_envelope.manifest.payload_checksum
    )
    validate_checkpoint_objects(state, full_envelope.manifest.objects)
    if (
        not isinstance(state, Mapping)
        or state.get("kind") != "merak_materialized_checkpoint"
        or int(state.get("iteration", -1)) != epoch
        or state.get("run_id") != run_id
        or state.get("materialization", {}).get("source_manifest")
        != dict(expected_full_evidence)
    ):
        raise ValueError("peer service returned an invalid materialized checkpoint")

    delta_envelope, _ = client.get(
        failed_rank,
        epoch,
        control_checksum=control_checksum,
        expected_manifest=expected_delta_evidence,
    )
    _validate_service_manifest(
        delta_envelope.manifest, source_rank=failed_rank, epoch=epoch
    )
    if (
        delta_envelope.manifest.group_ranks != group_ranks
        or delta_envelope.manifest.target_rank != peer_rank
    ):
        raise ValueError("peer runtime delta has an invalid placement")
    delta = deserialize_checkpoint(
        delta_envelope.payload, delta_envelope.manifest.payload_checksum
    )
    validate_checkpoint_objects(delta, delta_envelope.manifest.objects)
    if (
        not isinstance(delta, Mapping)
        or delta.get("kind") != "peer_runtime_delta"
        or int(delta.get("source_rank", -1)) != failed_rank
        or int(delta.get("peer_rank", -1)) != peer_rank
        or int(delta.get("epoch", -1)) != epoch
        or not isinstance(delta.get("runtime_state"), Mapping)
        or delta.get("run_id") != run_id
        or (
            state.get("adacheck_semantic") is not None
            and delta.get("adacheck_semantic") != state.get("adacheck_semantic")
        )
        or (
            state.get("adacheck_semantic_encoder") is not None
            and delta.get("adacheck_semantic_encoder")
            != state.get("adacheck_semantic_encoder")
        )
        or manifest_evidence(delta_envelope.manifest)
        != dict(expected_delta_evidence)
    ):
        raise ValueError("peer runtime delta does not match materialized recovery")
    return state, delta["runtime_state"]


def restore_own_materialized(
    client: ReplicaServiceClient,
    *,
    rank: int,
    epoch: int,
    group_ranks: tuple[int, ...],
    run_id: str,
    expected_full_evidence: Mapping[str, Any],
    control_checksum: str,
    expected_target_rank: int | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Recover a rank from one sealed, fully materialized stream."""
    envelope, _ = client.get_materialized(
        rank,
        epoch,
        control_checksum=control_checksum,
        expected_manifest=expected_full_evidence,
    )
    _validate_service_manifest(envelope.manifest, source_rank=rank, epoch=epoch)
    target_rank = rank if expected_target_rank is None else expected_target_rank
    if (
        envelope.manifest.group_ranks != group_ranks
        or envelope.manifest.target_rank != target_rank
    ):
        raise ValueError("own materialized checkpoint has an invalid placement")
    state = deserialize_checkpoint(envelope.payload, envelope.manifest.payload_checksum)
    validate_checkpoint_objects(state, envelope.manifest.objects)
    runtime = state.get("runtime_state") if isinstance(state, Mapping) else None
    if (
        not isinstance(state, Mapping)
        or state.get("kind") != "merak_materialized_checkpoint"
        or int(state.get("iteration", -1)) != epoch
        or state.get("run_id") != run_id
        or not isinstance(runtime, Mapping)
        or int(runtime.get("iteration", -1)) != epoch
        or state.get("materialization", {}).get("source_manifest")
        != dict(expected_full_evidence)
    ):
        raise ValueError("service returned an invalid own materialized checkpoint")
    return state, runtime
