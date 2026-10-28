"""Bounded out-of-process checkpoint replica service and client."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import socketserver
import struct
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import torch

from .manifest import CheckpointManifest, create_prepared_manifest
from .transport import ReplicaEnvelope
from .catalog import (
    deserialize_checkpoint,
    serialize_checkpoint,
    validate_checkpoint_objects,
)
from .cpu_materializer import AdamWCheckpointMaterializer
from .global_commit import manifest_evidence
from .incremental_checkpoint import RecoveryBinding

_FRAME = struct.Struct("!Q")
_MAX_HEADER_BYTES = 64 * 1024 * 1024
_MATERIALIZER_EVENT_LIMIT = 1024


def _same_checkpoint(left: CheckpointManifest, right: CheckpointManifest) -> bool:
    """Compare checkpoint identity while ignoring phase state and timestamps."""
    return replace(
        left,
        state=right.state,
        prepared_time_ns=right.prepared_time_ns,
        committed_time_ns=right.committed_time_ns,
    ) == right


def _receive_exact(stream: socket.socket, size: int) -> bytes:
    buffer = bytearray(size)
    view = memoryview(buffer)
    received = 0
    while received < size:
        count = stream.recv_into(view[received:])
        if count == 0:
            raise ConnectionError("replica service connection closed mid-frame")
        received += count
    return bytes(buffer)


def _encode_header(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _send_header(stream: socket.socket, value: dict[str, Any]) -> int:
    encoded = _encode_header(value)
    stream.sendall(_FRAME.pack(len(encoded)))
    stream.sendall(encoded)
    return _FRAME.size + len(encoded)


def _receive_header(stream: socket.socket) -> tuple[dict[str, Any], int]:
    size = _FRAME.unpack(_receive_exact(stream, _FRAME.size))[0]
    if size > _MAX_HEADER_BYTES:
        raise ValueError(f"replica service header is too large: {size}")
    encoded = _receive_exact(stream, size)
    return json.loads(encoded.decode("utf-8")), _FRAME.size + size


class ServiceReplicaStore:
    """Thread-safe store that never evicts the latest committed source epoch."""

    def __init__(
        self,
        max_bytes: int,
        retained_epochs_per_source: int = 2,
        *,
        materialize_incremental: bool = False,
        sealed_incremental: bool = False,
    ):
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        if retained_epochs_per_source < 1:
            raise ValueError("retained_epochs_per_source must be positive")
        if sealed_incremental and not materialize_incremental:
            raise ValueError("sealed incremental storage requires materialization")
        self.max_bytes = max_bytes
        self.retained_epochs_per_source = retained_epochs_per_source
        self.materialize_incremental = materialize_incremental
        self.sealed_incremental = sealed_incremental
        self.bytes_used = 0
        self.materialized_bytes = 0
        self._pending_materialized_bytes = 0
        self._staged_shared_bytes = 0
        self._records: OrderedDict[tuple[int, int], ReplicaEnvelope] = OrderedDict()
        self._materializers: dict[int, AdamWCheckpointMaterializer] = {}
        self._materialized_sizes: dict[int, int] = {}
        self._materialized_manifests: dict[int, CheckpointManifest] = {}
        self._materialized_bases: dict[int, CheckpointManifest] = {}
        self._applied_manifests: dict[tuple[int, int], CheckpointManifest] = {}
        self._staged_shared: dict[
            tuple[int, int], tuple[CheckpointManifest, Mapping[str, Any]]
        ] = {}
        self._pins: dict[tuple[str, int, int], str] = {}
        self._sealed: dict[tuple[str, int], tuple[int, str]] = {}
        self._promotion_targets: dict[int, int] = {}
        self._promotion_running: set[int] = set()
        self._materializer_apply_events: dict[int, deque[dict[str, int]]] = {}
        self._materializer_apply_summary: dict[int, dict[str, int]] = {}
        self._committing_keys: set[tuple[int, int]] = set()
        self._poisoned_sources: dict[int, str] = {}
        self._source_locks: dict[int, threading.Lock] = {}
        self._ingest_locks: dict[int, threading.Lock] = {}
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)

    def _total_bytes(self) -> int:
        return (
            self.bytes_used
            + self.materialized_bytes
            + self._pending_materialized_bytes
            + self._staged_shared_bytes
        )

    def _source_lock(self, source_rank: int) -> threading.Lock:
        with self._lock:
            return self._source_locks.setdefault(source_rank, threading.Lock())

    def _ingest_lock(self, source_rank: int) -> threading.Lock:
        with self._lock:
            return self._ingest_locks.setdefault(source_rank, threading.Lock())

    def _record_write_lock(self, source_rank: int) -> threading.Lock:
        if self.materialize_incremental and self.sealed_incremental:
            return self._ingest_lock(source_rank)
        return self._source_lock(source_rank)

    def _expected_stage_epoch(
        self, source_rank: int, current_key: tuple[int, int] | None = None
    ) -> int:
        materializer = self._materializers[source_rank]
        staged = {
            epoch
            for rank, epoch in self._staged_shared
            if rank == source_rank and epoch > materializer.current_epoch
        }
        staged.update(
            epoch
            for (rank, epoch), envelope in self._records.items()
            if rank == source_rank
            and (rank, epoch) != current_key
            and epoch > materializer.current_epoch
            and envelope.manifest.committed
        )
        if staged and sorted(staged) != list(
            range(materializer.current_epoch + 1, max(staged) + 1)
        ):
            raise RuntimeError("materialized staged epochs are not contiguous")
        return max(staged, default=materializer.current_epoch) + 1

    def _latest_committed_keys(self) -> set[tuple[int, int]]:
        latest: dict[int, tuple[int, int]] = {}
        for key, envelope in self._records.items():
            if not envelope.manifest.committed:
                continue
            previous = latest.get(key[0])
            if previous is None or key[1] > previous[1]:
                latest[key[0]] = key
        return set(latest.values())

    def _protected_record_keys(self) -> set[tuple[int, int]]:
        protected = self._latest_committed_keys() | self._committing_keys
        protected.update(
            (source_rank, epoch)
            for kind, source_rank, epoch in self._pins
            if kind == "record"
        )
        for (kind, source_rank), (sealed_epoch, _checksum) in self._sealed.items():
            if kind == "record":
                protected.update(
                    key
                    for key in self._records
                    if key[0] == source_rank and key[1] >= sealed_epoch
                )
        return protected

    def _make_room(self, required: int) -> None:
        protected = self._protected_record_keys()
        for key in list(self._records):
            if self._total_bytes() + required <= self.max_bytes:
                return
            envelope = self._records[key]
            if key in protected:
                continue
            self.bytes_used -= len(envelope.payload)
            del self._records[key]
        if self._total_bytes() + required > self.max_bytes:
            raise MemoryError(
                f"replica service capacity {self.max_bytes} cannot admit {required} bytes "
                "without evicting a latest committed epoch"
            )

    def put_prepared(self, envelope: ReplicaEnvelope) -> None:
        manifest = envelope.manifest
        if manifest.state != "prepared":
            raise ValueError("PUT_PREPARED requires a prepared manifest")
        if len(envelope.payload) != manifest.payload_bytes:
            raise ValueError("payload length does not match manifest")
        checksum = hashlib.sha256(envelope.payload).hexdigest()
        if checksum != manifest.payload_checksum:
            raise ValueError("checkpoint payload checksum mismatch")
        key = (manifest.source_rank, manifest.epoch)
        with self._record_write_lock(manifest.source_rank):
            with self._lock:
                applied = self._applied_manifests.get(key)
                if applied is not None:
                    if not _same_checkpoint(applied, manifest):
                        raise ValueError(f"refusing to replace applied replica {key}")
                    return
                previous = self._records.get(key)
                if previous is not None:
                    if _same_checkpoint(previous.manifest, manifest) and (
                        previous.payload == envelope.payload
                    ):
                        self._records.move_to_end(key)
                        return
                    if previous.manifest.committed:
                        raise ValueError(f"refusing to replace committed replica {key}")
                previous = self._records.pop(key, None)
                if previous is not None:
                    self.bytes_used -= len(previous.payload)
                try:
                    self._make_room(len(envelope.payload))
                except Exception:
                    if previous is not None:
                        self._records[key] = previous
                        self.bytes_used += len(previous.payload)
                    raise
                self._records[key] = envelope
                self.bytes_used += len(envelope.payload)

    def commit(self, manifest: CheckpointManifest) -> None:
        if not manifest.committed:
            raise ValueError("COMMIT requires a committed manifest")
        with self._record_write_lock(manifest.source_rank):
            self._commit_serialized(manifest)

    def _commit_serialized(self, manifest: CheckpointManifest) -> None:
        key = (manifest.source_rank, manifest.epoch)
        with self._lock:
            applied = self._applied_manifests.get(key)
            if applied is not None:
                if not _same_checkpoint(applied, manifest):
                    raise ValueError("commit conflicts with an applied manifest")
                return
            try:
                prepared = self._records[key]
            except KeyError as error:
                raise KeyError(f"prepared replica not found for {key}") from error
            if not _same_checkpoint(prepared.manifest, manifest):
                raise ValueError("commit decision does not match prepared manifest")
            self._committing_keys.add(key)
        try:
            if self.materialize_incremental:
                with self._lock:
                    has_materialized_base = manifest.source_rank in self._materializers
                if self.sealed_incremental and has_materialized_base:
                    self._stage_serialized_materialized_update(manifest, prepared)
                    return
                state = deserialize_checkpoint(
                    prepared.payload, prepared.manifest.payload_checksum
                )
                validate_checkpoint_objects(state, prepared.manifest.objects)
                kind = state.get("kind") if isinstance(state, dict) else None
                if kind in {"merak_incremental_base", "optimizer_update"}:
                    self._commit_materialized(manifest, state, prepared)
                    return
            with self._lock:
                self._commit_record(manifest)
        finally:
            with self._lock:
                self._committing_keys.discard(key)

    def _stage_serialized_materialized_update(
        self, manifest: CheckpointManifest, prepared: ReplicaEnvelope
    ) -> None:
        source_rank = manifest.source_rank
        key = (source_rank, manifest.epoch)
        with self._lock:
            base = self._materialized_bases[source_rank]
            expected_target = next(
                (rank for rank in manifest.group_ranks if rank != source_rank),
                source_rank,
            )
            if (
                manifest.target_rank != expected_target
                or manifest.group_ranks != base.group_ranks
                or manifest.target_rank != base.target_rank
                or manifest.source_host != base.source_host
            ):
                raise ValueError("materialized update changed stream identity")
            if manifest.epoch != self._expected_stage_epoch(source_rank, key):
                raise ValueError("materialized staged update is not contiguous")
            self._records[key] = ReplicaEnvelope(manifest, prepared.payload)
            self._records.move_to_end(key)

    def _commit_record(self, manifest: CheckpointManifest) -> None:
        key = (manifest.source_rank, manifest.epoch)
        prepared = self._records[key]
        self._records[key] = ReplicaEnvelope(manifest, prepared.payload)
        self._records.move_to_end(key)
        self._trim_records(manifest.source_rank)

    def _trim_records(self, source_rank: int) -> None:
        protected = self._protected_record_keys()
        committed = sorted(
            (
                record_key
                for record_key, envelope in self._records.items()
                if record_key[0] == source_rank and envelope.manifest.committed
            ),
            key=lambda item: item[1],
        )
        while len(committed) > self.retained_epochs_per_source:
            evicted_key = next(
                (
                    key
                    for key in committed
                    if key not in protected
                ),
                None,
            )
            if evicted_key is None:
                return
            committed.remove(evicted_key)
            evicted = self._records.pop(evicted_key)
            self.bytes_used -= len(evicted.payload)

    def _commit_materialized(
        self,
        manifest: CheckpointManifest,
        state: dict[str, Any],
        prepared: ReplicaEnvelope | None,
    ) -> None:
        source_rank = manifest.source_rank
        key = (source_rank, manifest.epoch)
        try:
            if source_rank in self._poisoned_sources:
                raise RuntimeError(
                    f"materialized source is poisoned: "
                    f"{self._poisoned_sources[source_rank]}"
                )
            metadata = state.get("metadata", {})
            stream_role = (
                state.get("stream_role")
                if state["kind"] == "merak_incremental_base"
                else metadata.get("stream_role")
            )
            expected_target = (
                source_rank
                if stream_role == "peer_local_full"
                else next(
                    (rank for rank in manifest.group_ranks if rank != source_rank),
                    source_rank,
                )
            )
            if (
                stream_role not in {"peer_local_full", "remote_full"}
                or manifest.target_rank != expected_target
                or source_rank not in manifest.group_ranks
            ):
                raise ValueError("materialized stream must target its source rank")
            if state["kind"] == "merak_incremental_base":
                if prepared is None:
                    raise ValueError("materializer base requires a serialized record")
                if state.get("stream_role") not in {
                    "peer_local_full",
                    "remote_full",
                }:
                    raise ValueError("materializer base has an invalid stream role")
                if not state.get("run_id"):
                    raise ValueError("materializer base has no run identity")
                binding = RecoveryBinding(**state["binding"])
                if binding.rank != source_rank:
                    raise ValueError("materializer base binding does not match source")
                with self._lock:
                    if source_rank in self._materializers:
                        raise ValueError("incremental materializer base already exists")
                    envelope = self._records[key]
                    reservation = 2 * manifest.logical_bytes
                    projected = self._total_bytes() - len(envelope.payload) + reservation
                    if projected > self.max_bytes:
                        raise MemoryError("materialized base exceeds service capacity")
                    self._pending_materialized_bytes += reservation
                try:
                    materializer = AdamWCheckpointMaterializer(
                        state,
                        base_id=f"sha256:{manifest.payload_checksum}",
                        binding=binding,
                        parameter_ids=state["parameter_ids"],
                    )
                except Exception:
                    with self._lock:
                        self._pending_materialized_bytes -= reservation
                    raise
                with self._lock:
                    self._pending_materialized_bytes -= reservation
                    self._materializers[source_rank] = materializer
                    self._materialized_sizes[source_rank] = manifest.logical_bytes
                    self._materialized_bases[source_rank] = manifest
                    self.materialized_bytes += reservation
            else:
                if metadata.get("stream_role") not in {
                    "peer_local_full",
                    "remote_full",
                }:
                    raise ValueError("materializer update has an invalid stream role")
                with self._lock:
                    materializer = self._materializers[source_rank]
                    base = self._materialized_bases[source_rank]
                if metadata.get("stream_role") != materializer.state.get("stream_role"):
                    raise ValueError("materializer update changed stream role")
                if metadata.get("run_id") != materializer.state.get("run_id"):
                    raise ValueError("materializer update changed run identity")
                if (
                    manifest.group_ranks != base.group_ranks
                    or manifest.target_rank != base.target_rank
                    or manifest.source_host != base.source_host
                ):
                    raise ValueError("materializer update changed stream identity")
                if self.sealed_incremental:
                    if manifest.epoch != self._expected_stage_epoch(source_rank, key):
                        raise ValueError("materialized staged update is not contiguous")
                    materializer.validate(state, epoch=manifest.epoch)
                    if prepared is None:
                        raise ValueError("serialized staged update has no payload")
                    with self._lock:
                        self._records[key] = ReplicaEnvelope(manifest, prepared.payload)
                        self._records.move_to_end(key)
                    return
                materializer.apply(state)
            with self._lock:
                self._materialized_manifests[source_rank] = manifest
                self._applied_manifests[key] = manifest
                if prepared is not None:
                    if self._records.get(key) is not prepared:
                        raise RuntimeError("prepared replica changed during commit")
                    removed = self._records.pop(key)
                    self.bytes_used -= len(removed.payload)
        except Exception as error:
            with self._lock:
                self._poisoned_sources[source_rank] = f"{type(error).__name__}: {error}"
            raise

    def commit_shared_update(
        self,
        *,
        epoch: int,
        source_rank: int,
        source_host: str,
        target_rank: int,
        target_host: str,
        group_ranks: tuple[int, ...],
        state: Mapping[str, Any],
    ) -> tuple[CheckpointManifest, float, float, str]:
        if not self.materialize_incremental:
            raise ValueError("shared updates require an incremental materializer")
        from .shared_tensor import describe_shared_update

        describe_started = time.perf_counter()
        described = describe_shared_update(state)
        describe_ms = (time.perf_counter() - describe_started) * 1000
        prepared = create_prepared_manifest(
            epoch=epoch,
            source_rank=source_rank,
            source_host=source_host,
            target_rank=target_rank,
            target_host=target_host,
            group_ranks=group_ranks,
            serialized=described,
        )
        committed = replace(
            prepared, state="committed", committed_time_ns=time.time_ns()
        )
        key = (source_rank, epoch)
        with self._source_lock(source_rank):
            with self._lock:
                applied = self._applied_manifests.get(key)
                if applied is not None:
                    if not _same_checkpoint(applied, committed):
                        raise ValueError("shared update conflicts with applied manifest")
                    return applied, describe_ms, 0.0, str(
                        described.equivalence_checksum
                    )
                if key in self._records:
                    raise ValueError("shared update conflicts with a byte-stream record")
                staged = self._staged_shared.get(key)
                if staged is not None:
                    if not _same_checkpoint(staged[0], committed):
                        raise ValueError("shared update conflicts with staged manifest")
                    return staged[0], describe_ms, 0.0, str(
                        described.equivalence_checksum
                    )
            if self.sealed_incremental:
                stage_started = time.perf_counter()
                try:
                    with self._lock:
                        materializer = self._materializers[source_rank]
                        base = self._materialized_bases[source_rank]
                    metadata = state.get("metadata", {})
                    if (
                        metadata.get("stream_role") != "peer_local_full"
                        or metadata.get("run_id") != materializer.state.get("run_id")
                        or committed.group_ranks != base.group_ranks
                        or committed.target_rank != base.target_rank
                        or committed.source_host != base.source_host
                    ):
                        raise ValueError("shared update changed stream identity")
                    if epoch != self._expected_stage_epoch(source_rank):
                        raise ValueError("shared materialized stage is not contiguous")
                    materializer.validate(state, epoch=epoch)
                    with self._lock:
                        self._make_room(committed.logical_bytes)
                        self._staged_shared[key] = (committed, dict(state))
                        self._staged_shared_bytes += committed.logical_bytes
                except Exception as error:
                    with self._lock:
                        self._poisoned_sources[source_rank] = (
                            f"{type(error).__name__}: {error}"
                        )
                    raise
                return (
                    committed,
                    describe_ms,
                    (time.perf_counter() - stage_started) * 1000,
                    str(described.equivalence_checksum),
                )
            apply_started = time.perf_counter()
            self._commit_materialized(committed, dict(state), None)
            apply_ms = (time.perf_counter() - apply_started) * 1000
        return committed, describe_ms, apply_ms, str(
            described.equivalence_checksum
        )

    def pin(
        self,
        *,
        kind: str,
        source_rank: int,
        epoch: int,
        control_checksum: str,
        expected_manifest: Mapping[str, Any],
    ) -> None:
        if kind not in {"materialized", "record"}:
            raise ValueError("pin kind must be materialized or record")
        if len(control_checksum) != 64:
            raise ValueError("control checksum must be a SHA-256 digest")
        if kind == "record":
            with self._source_lock(source_rank):
                with self._lock:
                    key = (source_rank, epoch)
                    try:
                        envelope = self._records[key]
                    except KeyError as error:
                        raise KeyError(
                            f"record not found for rank={source_rank}, epoch={epoch}"
                        ) from error
                    if not envelope.manifest.committed:
                        raise ValueError("pin requires a committed record")
                    manifest = envelope.manifest
                    if manifest_evidence(manifest) != dict(expected_manifest):
                        raise ValueError("pin manifest evidence mismatch")
                    self._pins[(kind, source_rank, epoch)] = control_checksum
            return

        # A staged manifest is already durable enough to pin. Avoid waiting for
        # the CPU materializer, which holds the per-source lock while applying.
        with self._lock:
            key = (source_rank, epoch)
            staged = self._staged_shared.get(key)
            if staged is not None:
                manifest = staged[0]
            elif self._materializers[source_rank].current_epoch == epoch:
                manifest = self._materialized_manifests[source_rank]
            else:
                try:
                    manifest = self._records[key].manifest
                except KeyError as error:
                    raise KeyError(
                        f"materialized stage not found for rank={source_rank}, "
                        f"epoch={epoch}"
                    ) from error
            if manifest_evidence(manifest) != dict(expected_manifest):
                raise ValueError("pin manifest evidence mismatch")
            self._pins[(kind, source_rank, epoch)] = control_checksum

    def seal(
        self, *, kind: str, source_rank: int, epoch: int, control_checksum: str
    ) -> None:
        pin = (kind, source_rank, epoch)
        with self._lock:
            if self._pins.get(pin) != control_checksum:
                raise ValueError("seal has no matching pin")
            self._sealed[(kind, source_rank)] = (epoch, control_checksum)
            for key in list(self._pins):
                if key[0] == kind and key[1] == source_rank and key[2] < epoch:
                    del self._pins[key]
            if kind == "record":
                self._trim_records(source_rank)
                return
            if kind != "materialized":
                raise ValueError("seal kind must be materialized or record")
            self._promotion_targets[source_rank] = max(
                epoch, self._promotion_targets.get(source_rank, epoch)
            )
            if source_rank in self._promotion_running:
                return
            self._promotion_running.add(source_rank)
        threading.Thread(
            target=self._promotion_loop,
            args=(source_rank,),
            name=f"checkpoint-materializer-rank-{source_rank}",
            daemon=True,
        ).start()

    def _promotion_loop(self, source_rank: int) -> None:
        try:
            while True:
                with self._source_lock(source_rank):
                    with self._lock:
                        materializer = self._materializers[source_rank]
                        target = self._promotion_targets[source_rank]
                        if materializer.current_epoch >= target:
                            self._promotion_running.discard(source_rank)
                            self._condition.notify_all()
                            return
                        epoch = materializer.current_epoch + 1
                        key = (source_rank, epoch)
                        shared = self._staged_shared.get(key)
                        serialized = self._records.get(key)
                    if shared is not None:
                        manifest, state = shared
                    elif serialized is not None:
                        manifest = serialized.manifest
                        state = deserialize_checkpoint(
                            serialized.payload, serialized.manifest.payload_checksum
                        )
                        validate_checkpoint_objects(state, serialized.manifest.objects)
                    else:
                        raise KeyError(
                            f"sealed materialized stage is missing for rank={source_rank}, "
                            f"epoch={epoch}"
                        )
                    apply_started_ns = time.perf_counter_ns()
                    materializer.apply(state)
                    apply_finished_ns = time.perf_counter_ns()
                    with self._lock:
                        duration_ns = apply_finished_ns - apply_started_ns
                        events = self._materializer_apply_events.setdefault(
                            source_rank, deque(maxlen=_MATERIALIZER_EVENT_LIMIT)
                        )
                        events.append(
                            {
                                "epoch": epoch,
                                "started_monotonic_ns": apply_started_ns,
                                "finished_monotonic_ns": apply_finished_ns,
                                "duration_ns": duration_ns,
                            }
                        )
                        summary = self._materializer_apply_summary.setdefault(
                            source_rank,
                            {
                                "count": 0,
                                "total_duration_ns": 0,
                                "max_duration_ns": 0,
                                "first_started_monotonic_ns": apply_started_ns,
                                "last_finished_monotonic_ns": apply_finished_ns,
                            },
                        )
                        summary["count"] += 1
                        summary["total_duration_ns"] += duration_ns
                        summary["max_duration_ns"] = max(
                            summary["max_duration_ns"], duration_ns
                        )
                        summary["last_finished_monotonic_ns"] = apply_finished_ns
                        self._materialized_manifests[source_rank] = manifest
                        self._applied_manifests[key] = manifest
                        staged = self._staged_shared.pop(key, None)
                        if staged is not None:
                            self._staged_shared_bytes -= manifest.logical_bytes
                        record = self._records.pop(key, None)
                        if record is not None:
                            self.bytes_used -= len(record.payload)
                        self._condition.notify_all()
        except Exception as error:
            with self._lock:
                self._poisoned_sources[source_rank] = f"{type(error).__name__}: {error}"
                self._promotion_running.discard(source_rank)
                self._condition.notify_all()

    def wait_materialized(
        self, source_rank: int, epoch: int, timeout_seconds: float
    ) -> int:
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            while self._materializers[source_rank].current_epoch < epoch:
                if source_rank in self._poisoned_sources:
                    raise RuntimeError(
                        f"materialized source is poisoned: "
                        f"{self._poisoned_sources[source_rank]}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"materialized rank={source_rank} did not reach epoch {epoch}"
                    )
                self._condition.wait(remaining)
            return self._materializers[source_rank].current_epoch

    def materialized_epoch(self, source_rank: int) -> int:
        with self._lock:
            try:
                return self._materializers[source_rank].current_epoch
            except KeyError as error:
                raise KeyError(f"materialized rank not found: {source_rank}") from error

    def _authorize(
        self, kind: str, source_rank: int, epoch: int, control_checksum: str | None
    ) -> None:
        if not self.sealed_incremental:
            return
        if control_checksum is None or (
            self._pins.get((kind, source_rank, epoch)) != control_checksum
            and self._sealed.get((kind, source_rank)) != (epoch, control_checksum)
        ):
            raise ValueError("checkpoint epoch is not sealed by this control")

    def get_materialized(
        self,
        source_rank: int,
        epoch: int,
        *,
        control_checksum: str | None = None,
        expected_manifest: Mapping[str, Any] | None = None,
    ) -> ReplicaEnvelope:
        with self._lock:
            self._authorize("materialized", source_rank, epoch, control_checksum)
            if source_rank in self._poisoned_sources:
                raise RuntimeError(
                    f"materialized source is poisoned: "
                    f"{self._poisoned_sources[source_rank]}"
                )
            if self.sealed_incremental and expected_manifest is not None:
                key = (source_rank, epoch)
                staged = self._staged_shared.get(key)
                if staged is not None:
                    manifest = staged[0]
                elif self._materializers[source_rank].current_epoch == epoch:
                    manifest = self._materialized_manifests[source_rank]
                else:
                    manifest = self._records[key].manifest
                if manifest_evidence(manifest) != dict(expected_manifest):
                    raise ValueError("materialized recovery evidence mismatch")
            if self._materializers[source_rank].current_epoch < epoch:
                self._promotion_targets[source_rank] = max(
                    epoch, self._promotion_targets.get(source_rank, epoch)
                )
                if source_rank not in self._promotion_running:
                    self._promotion_running.add(source_rank)
                    threading.Thread(
                        target=self._promotion_loop,
                        args=(source_rank,),
                        daemon=True,
                    ).start()
        if self.sealed_incremental:
            self.wait_materialized(source_rank, epoch, 600)
        with self._lock:
            try:
                source_lock = self._source_locks[source_rank]
            except KeyError as error:
                raise KeyError(f"materialized rank not found: {source_rank}") from error
        with source_lock:
            with self._lock:
                if source_rank in self._poisoned_sources:
                    raise RuntimeError(
                        f"materialized source is poisoned: "
                        f"{self._poisoned_sources[source_rank]}"
                    )
                materializer = self._materializers[source_rank]
                latest = self._materialized_manifests[source_rank]
            if materializer.current_epoch != epoch:
                raise KeyError(
                    f"materialized rank={source_rank} is at epoch "
                    f"{materializer.current_epoch}, not {epoch}"
                )
            serialized = serialize_checkpoint(
                materializer.snapshot(source_manifest=manifest_evidence(latest))
            )
            prepared = create_prepared_manifest(
                epoch=epoch,
                source_rank=source_rank,
                source_host=latest.source_host,
                target_rank=latest.target_rank,
                target_host=latest.target_host,
                group_ranks=latest.group_ranks,
                serialized=serialized,
                replica_count=latest.replica_count,
                tolerated_failures=latest.tolerated_failures,
            )
            committed = replace(
                prepared, state="committed", committed_time_ns=time.time_ns()
            )
            return ReplicaEnvelope(committed, serialized.payload)

    def get_committed(
        self,
        source_rank: int,
        epoch: int,
        *,
        control_checksum: str | None = None,
        expected_manifest: Mapping[str, Any] | None = None,
    ) -> ReplicaEnvelope:
        key = (source_rank, epoch)
        with self._lock:
            self._authorize("record", source_rank, epoch, control_checksum)
            try:
                envelope = self._records[key]
            except KeyError as error:
                raise KeyError(
                    f"replica not found for rank={source_rank}, epoch={epoch}"
                ) from error
            if not envelope.manifest.committed:
                raise ValueError(f"replica for rank={source_rank}, epoch={epoch} is not committed")
            if self.sealed_incremental and expected_manifest is not None and manifest_evidence(
                envelope.manifest
            ) != dict(expected_manifest):
                raise ValueError("record recovery evidence mismatch")
            self._records.move_to_end(key)
            return envelope

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "capacity_bytes": self.max_bytes,
                "bytes_used": self._total_bytes(),
                "record_bytes": self.bytes_used,
                "materialized_bytes": self.materialized_bytes,
                "materialized_logical_bytes": sum(self._materialized_sizes.values()),
                "pending_materialized_bytes": self._pending_materialized_bytes,
                "staged_shared_bytes": self._staged_shared_bytes,
                "materialized_epochs": {
                    str(rank): materializer.current_epoch
                    for rank, materializer in self._materializers.items()
                },
                "materializer_apply": {
                    str(rank): {
                        **self._materializer_apply_summary.get(rank, {}),
                        "events": list(
                            self._materializer_apply_events.get(rank, ())
                        ),
                    }
                    for rank in self._materializers
                },
                "record_count": len(self._records),
                "staged_materialized_epochs": sorted(
                    [list(key) for key in self._staged_shared]
                ),
                "sealed_epochs": {
                    f"{kind}:{rank}": epoch
                    for (kind, rank), (epoch, _checksum) in self._sealed.items()
                },
                "poisoned_sources": dict(self._poisoned_sources),
                "records": [
                    {
                        "source_rank": key[0],
                        "epoch": key[1],
                        "state": envelope.manifest.state,
                        "payload_bytes": len(envelope.payload),
                    }
                    for key, envelope in self._records.items()
                ],
            }


class ReplicaTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 128

    def __init__(
        self,
        address: tuple[str, int],
        store: ServiceReplicaStore,
        *,
        materialized_response_workers: int = 1,
    ):
        if materialized_response_workers < 1:
            raise ValueError("materialized response workers must be positive")
        super().__init__(address, ReplicaRequestHandler)
        self.store = store
        self.materialized_response_lock = threading.BoundedSemaphore(
            materialized_response_workers
        )
        self._counter_lock = threading.Lock()
        self.rx_bytes = 0
        self.tx_bytes = 0

    def add_rx(self, value: int) -> None:
        with self._counter_lock:
            self.rx_bytes += value

    def add_tx(self, value: int) -> None:
        with self._counter_lock:
            self.tx_bytes += value

    def service_stats(self) -> dict[str, Any]:
        result = self.store.stats()
        with self._counter_lock:
            result.update(
                {
                    "protocol_rx_bytes": self.rx_bytes,
                    "protocol_tx_bytes": self.tx_bytes,
                }
            )
        return result


class ReplicaRequestHandler(socketserver.BaseRequestHandler):
    server: ReplicaTCPServer

    def handle(self) -> None:
        operation_start = time.perf_counter()
        try:
            request, received = _receive_header(self.request)
            self.server.add_rx(received)
            operation = request.get("operation")
            response_payload = b""
            if operation == "HEALTH":
                response = {"status": "ok"}
            elif operation == "PUT_PREPARED":
                payload_bytes = int(request["payload_bytes"])
                if payload_bytes > self.server.store.max_bytes:
                    raise MemoryError(
                        f"payload {payload_bytes} exceeds service capacity "
                        f"{self.server.store.max_bytes}"
                    )
                payload = _receive_exact(self.request, payload_bytes)
                self.server.add_rx(payload_bytes)
                manifest = CheckpointManifest.from_dict(request["manifest"])
                self.server.store.put_prepared(ReplicaEnvelope(manifest, payload))
                response = {"status": "ok"}
            elif operation == "COMMIT":
                manifest = CheckpointManifest.from_dict(request["manifest"])
                self.server.store.commit(manifest)
                response = {"status": "ok"}
            elif operation == "PIN":
                self.server.store.pin(
                    kind=str(request["kind"]),
                    source_rank=int(request["source_rank"]),
                    epoch=int(request["epoch"]),
                    control_checksum=str(request["control_checksum"]),
                    expected_manifest=request["expected_manifest"],
                )
                response = {"status": "ok"}
            elif operation == "SEAL":
                self.server.store.seal(
                    kind=str(request["kind"]),
                    source_rank=int(request["source_rank"]),
                    epoch=int(request["epoch"]),
                    control_checksum=str(request["control_checksum"]),
                )
                response = {"status": "ok"}
            elif operation == "WAIT_MATERIALIZED":
                current = self.server.store.wait_materialized(
                    int(request["source_rank"]),
                    int(request["epoch"]),
                    float(request["timeout_seconds"]),
                )
                response = {"status": "ok", "current_epoch": current}
            elif operation == "GET":
                envelope = self.server.store.get_committed(
                    int(request["source_rank"]),
                    int(request["epoch"]),
                    control_checksum=request.get("control_checksum"),
                    expected_manifest=request.get("expected_manifest"),
                )
                response_payload = envelope.payload
                response = {
                    "status": "ok",
                    "manifest": envelope.manifest.to_dict(),
                    "payload_bytes": len(response_payload),
                }
            elif operation == "GET_MATERIALIZED":
                with self.server.materialized_response_lock:
                    envelope = self.server.store.get_materialized(
                        int(request["source_rank"]),
                        int(request["epoch"]),
                        control_checksum=request.get("control_checksum"),
                        expected_manifest=request.get("expected_manifest"),
                    )
                    response_payload = envelope.payload
                    response = {
                        "status": "ok",
                        "manifest": envelope.manifest.to_dict(),
                        "payload_bytes": len(response_payload),
                    }
                    response["server_ms"] = (
                        time.perf_counter() - operation_start
                    ) * 1000
                    sent = _send_header(self.request, response)
                    self.request.sendall(response_payload)
                    self.server.add_tx(sent + len(response_payload))
                    return
            elif operation == "STATS":
                response = {"status": "ok", "stats": self.server.service_stats()}
            else:
                raise ValueError(f"unsupported replica service operation: {operation}")
        except Exception as error:
            response = {"status": "error", "error": f"{type(error).__name__}: {error}"}
            response_payload = b""

        response["server_ms"] = (time.perf_counter() - operation_start) * 1000
        sent = _send_header(self.request, response)
        if response_payload:
            self.request.sendall(response_payload)
            sent += len(response_payload)
        self.server.add_tx(sent)


@dataclass(frozen=True)
class ServiceTransferMetrics:
    request_bytes: int
    response_bytes: int
    server_ms: float = 0.0

    @property
    def wire_bytes(self) -> int:
        return self.request_bytes + self.response_bytes


class ReplicaServiceClient:
    def __init__(self, host: str, port: int, timeout_seconds: float = 60):
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_endpoint(cls, endpoint: str, timeout_seconds: float = 60):
        host, port = endpoint.rsplit(":", 1)
        return cls(host, int(port), timeout_seconds)

    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"

    def _request(
        self, request: dict[str, Any], payload: bytes = b""
    ) -> tuple[dict[str, Any], bytes, ServiceTransferMetrics]:
        with socket.create_connection(
            (self.host, self.port), timeout=self.timeout_seconds
        ) as stream:
            stream.settimeout(self.timeout_seconds)
            request_bytes = _send_header(stream, request)
            if payload:
                stream.sendall(payload)
                request_bytes += len(payload)
            response, response_bytes = _receive_header(stream)
            response_payload = b""
            payload_bytes = int(response.get("payload_bytes", 0))
            if payload_bytes:
                response_payload = _receive_exact(stream, payload_bytes)
                response_bytes += payload_bytes
        if response.get("status") != "ok":
            raise RuntimeError(response.get("error", "replica service request failed"))
        return (
            response,
            response_payload,
            ServiceTransferMetrics(
                request_bytes, response_bytes, float(response.get("server_ms", 0.0))
            ),
        )

    def health(self) -> None:
        self._request({"operation": "HEALTH"})

    def put_prepared(self, envelope: ReplicaEnvelope) -> ServiceTransferMetrics:
        _, _, metrics = self._request(
            {
                "operation": "PUT_PREPARED",
                "manifest": envelope.manifest.to_dict(),
                "payload_bytes": len(envelope.payload),
            },
            envelope.payload,
        )
        return metrics

    def commit(self, manifest: CheckpointManifest) -> ServiceTransferMetrics:
        _, _, metrics = self._request(
            {"operation": "COMMIT", "manifest": manifest.to_dict()}
        )
        return metrics

    def pin(
        self,
        *,
        kind: str,
        source_rank: int,
        epoch: int,
        control_checksum: str,
        expected_manifest: Mapping[str, Any],
    ) -> ServiceTransferMetrics:
        _, _, metrics = self._request(
            {
                "operation": "PIN",
                "kind": kind,
                "source_rank": source_rank,
                "epoch": epoch,
                "control_checksum": control_checksum,
                "expected_manifest": dict(expected_manifest),
            }
        )
        return metrics

    def seal(
        self,
        *,
        kind: str,
        source_rank: int,
        epoch: int,
        control_checksum: str,
    ) -> ServiceTransferMetrics:
        _, _, metrics = self._request(
            {
                "operation": "SEAL",
                "kind": kind,
                "source_rank": source_rank,
                "epoch": epoch,
                "control_checksum": control_checksum,
            }
        )
        return metrics

    def wait_materialized(self, source_rank: int, epoch: int) -> int:
        response, _, _ = self._request(
            {
                "operation": "WAIT_MATERIALIZED",
                "source_rank": source_rank,
                "epoch": epoch,
                "timeout_seconds": self.timeout_seconds,
            }
        )
        return int(response["current_epoch"])

    def get(
        self,
        source_rank: int,
        epoch: int,
        *,
        control_checksum: str | None = None,
        expected_manifest: Mapping[str, Any] | None = None,
    ) -> tuple[ReplicaEnvelope, ServiceTransferMetrics]:
        response, payload, metrics = self._request(
            {
                "operation": "GET",
                "source_rank": source_rank,
                "epoch": epoch,
                "control_checksum": control_checksum,
                "expected_manifest": (
                    None if expected_manifest is None else dict(expected_manifest)
                ),
            }
        )
        manifest = CheckpointManifest.from_dict(response["manifest"])
        return ReplicaEnvelope(manifest, payload), metrics

    def get_materialized(
        self,
        source_rank: int,
        epoch: int,
        *,
        control_checksum: str | None = None,
        expected_manifest: Mapping[str, Any] | None = None,
    ) -> tuple[ReplicaEnvelope, ServiceTransferMetrics]:
        response, payload, metrics = self._request(
            {
                "operation": "GET_MATERIALIZED",
                "source_rank": source_rank,
                "epoch": epoch,
                "control_checksum": control_checksum,
                "expected_manifest": (
                    None if expected_manifest is None else dict(expected_manifest)
                ),
            }
        )
        manifest = CheckpointManifest.from_dict(response["manifest"])
        return ReplicaEnvelope(manifest, payload), metrics

    def stats(self) -> dict[str, Any]:
        stats, _ = self.stats_with_metrics()
        return stats

    def stats_with_metrics(
        self,
    ) -> tuple[dict[str, Any], ServiceTransferMetrics]:
        response, _, metrics = self._request({"operation": "STATS"})
        return response["stats"], metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29590)
    parser.add_argument("--max-store-mib", type=int, default=4096)
    parser.add_argument("--retained-epochs", type=int, default=2)
    parser.add_argument("--materialize-incremental", action="store_true")
    parser.add_argument("--sealed-incremental", action="store_true")
    parser.add_argument("--shared-tensor-socket", type=Path)
    parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument("--materialized-response-workers", type=int, default=1)
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args()
    if args.torch_num_threads < 0:
        raise ValueError("torch-num-threads must be non-negative")
    if args.torch_num_threads:
        torch.set_num_threads(args.torch_num_threads)
    if args.shared_tensor_socket is not None and not args.materialize_incremental:
        raise ValueError("shared-tensor service requires materialize-incremental")

    store = ServiceReplicaStore(
        max_bytes=args.max_store_mib * 1024 * 1024,
        retained_epochs_per_source=args.retained_epochs,
        materialize_incremental=args.materialize_incremental,
        sealed_incremental=args.sealed_incremental,
    )
    shared_server = None
    if args.shared_tensor_socket is not None:
        from .shared_tensor import SharedTensorServer

        shared_server = SharedTensorServer(args.shared_tensor_socket, store)
        shared_server.start()
    try:
        with ReplicaTCPServer(
            (args.bind, args.port),
            store,
            materialized_response_workers=args.materialized_response_workers,
        ) as server:
            endpoint = f"{server.server_address[0]}:{server.server_address[1]}"
            if args.ready_file is not None:
                args.ready_file.parent.mkdir(parents=True, exist_ok=True)
                args.ready_file.write_text(endpoint + "\n")
            print(json.dumps({"status": "ready", "endpoint": endpoint}), flush=True)
            try:
                server.serve_forever(poll_interval=0.2)
            except KeyboardInterrupt:
                pass
    finally:
        if shared_server is not None:
            shared_server.stop()


if __name__ == "__main__":
    main()
