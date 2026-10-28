"""Bounded remote replica storage, commit, and recovery orchestration."""

from __future__ import annotations

import socket
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch.distributed as dist

from .catalog import (
    deserialize_checkpoint,
    serialize_checkpoint,
    validate_checkpoint_objects,
)
from .manifest import (
    CheckpointManifest,
    ManifestRepository,
    create_prepared_manifest,
)
from .transport import GlooPairTransport, ReplicaEnvelope, TransportMetrics
from .replica_service import ReplicaServiceClient


@dataclass(frozen=True)
class CommitResult:
    manifest: CheckpointManifest
    serialize_ms: float
    transport: TransportMetrics
    commit_ms: float
    store_bytes: int
    put_service_ms: float = 0.0
    commit_service_ms: float = 0.0
    transport_kind: str = "byte_stream"
    shared_memory_bytes: int = 0
    equivalence_checksum: str | None = None
    shared_slot_wait_ms: float = 0.0
    shared_copy_ms: float = 0.0


class ReplicaStore:
    """An insertion-ordered CPU byte store with a strict capacity bound."""

    def __init__(self, max_bytes: int):
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes
        self._records: OrderedDict[tuple[int, int], ReplicaEnvelope] = OrderedDict()
        self.bytes_used = 0

    def put(self, envelope: ReplicaEnvelope) -> None:
        size = len(envelope.payload)
        if size > self.max_bytes:
            raise MemoryError(
                f"replica payload {size} exceeds store capacity {self.max_bytes}"
            )
        key = (envelope.manifest.source_rank, envelope.manifest.epoch)
        previous = self._records.pop(key, None)
        if previous is not None:
            self.bytes_used -= len(previous.payload)
        while self._records and self.bytes_used + size > self.max_bytes:
            _, evicted = self._records.popitem(last=False)
            self.bytes_used -= len(evicted.payload)
        self._records[key] = envelope
        self.bytes_used += size

    def get(self, source_rank: int, epoch: int) -> ReplicaEnvelope:
        key = (source_rank, epoch)
        try:
            envelope = self._records[key]
        except KeyError as error:
            raise KeyError(f"replica not found for rank={source_rank}, epoch={epoch}") from error
        self._records.move_to_end(key)
        return envelope

    def drop(self, source_rank: int, epoch: int) -> None:
        envelope = self._records.pop((source_rank, epoch), None)
        if envelope is not None:
            self.bytes_used -= len(envelope.payload)


class RemoteCheckpointSession:
    """Collective fixed-pair checkpoint save and restore session."""

    def __init__(
        self,
        *,
        group: object,
        group_ranks: tuple[int, ...],
        manifest_root: Path | str,
        max_store_bytes: int,
    ):
        self.transport = GlooPairTransport(group, group_ranks)
        self.rank = self.transport.rank
        self.peer_rank = self.transport.peer_rank
        self.group_ranks = group_ranks
        self.repository = ManifestRepository(manifest_root)
        self.store = ReplicaStore(max_store_bytes)
        self.host = socket.gethostname()
        hosts: list[str | None] = [None] * len(group_ranks)
        dist.all_gather_object(hosts, self.host, group=group)
        peer_host = hosts[group_ranks.index(self.peer_rank)]
        if peer_host is None:
            raise RuntimeError(f"missing hostname for checkpoint peer {self.peer_rank}")
        self.peer_host = peer_host

    def save(
        self,
        epoch: int,
        state: Any,
        *,
        state_is_snapshot: bool = False,
        object_checksums: bool = True,
    ) -> CommitResult:
        operation_start = time.perf_counter()
        serialize_start = time.perf_counter()
        serialized = serialize_checkpoint(
            state,
            state_is_snapshot=state_is_snapshot,
            object_checksums=object_checksums,
        )
        serialize_ms = (time.perf_counter() - serialize_start) * 1000
        manifest = create_prepared_manifest(
            epoch=epoch,
            source_rank=self.rank,
            source_host=self.host,
            target_rank=self.peer_rank,
            target_host=self.peer_host,
            group_ranks=self.group_ranks,
            serialized=serialized,
        )
        self.repository.prepare(manifest)
        local = ReplicaEnvelope(manifest=manifest, payload=serialized.payload)
        received, transport_metrics = self.transport.exchange(local)

        valid = False
        try:
            recovered_peer = deserialize_checkpoint(
                received.payload, received.manifest.payload_checksum
            )
            validate_checkpoint_objects(recovered_peer, received.manifest.objects)
            valid = True
        finally:
            peer_accepted = self.transport.acknowledge(valid)
        if not valid or not peer_accepted:
            raise RuntimeError(
                f"checkpoint epoch {epoch} failed replica validation in group {self.group_ranks}"
            )

        committed = self.repository.commit(manifest)
        peer_committed = self.transport.exchange_manifest(committed)
        if not peer_committed.committed:
            raise RuntimeError("peer did not publish a committed checkpoint decision")
        if peer_committed.manifest_id != received.manifest.manifest_id:
            raise RuntimeError("peer commit decision does not match received replica")

        committed_local = ReplicaEnvelope(committed, local.payload)
        self.store.put(committed_local)
        committed_remote = ReplicaEnvelope(peer_committed, received.payload)
        self.store.put(committed_remote)
        self.repository.record_replica(self.rank, peer_committed)
        return CommitResult(
            manifest=committed,
            serialize_ms=serialize_ms,
            transport=transport_metrics,
            commit_ms=(time.perf_counter() - operation_start) * 1000,
            store_bytes=self.store.bytes_used,
        )

    def drop_local(self, epoch: int) -> None:
        self.store.drop(self.rank, epoch)

    def restore_from_peer(self, epoch: int) -> Any:
        peer_replica = self.store.get(self.peer_rank, epoch)
        if not peer_replica.manifest.committed:
            raise ValueError("refusing to restore from a non-committed replica")
        received, _ = self.transport.exchange(peer_replica)
        if received.manifest.source_rank != self.rank or received.manifest.epoch != epoch:
            raise ValueError(
                "peer returned a replica for the wrong source rank or checkpoint epoch"
            )
        if not received.manifest.committed:
            raise ValueError("peer returned a non-committed replica")
        state = deserialize_checkpoint(
            received.payload, received.manifest.payload_checksum
        )
        validate_checkpoint_objects(state, received.manifest.objects)
        self.store.put(received)
        return state


class ServiceRemoteCheckpointSession:
    """Checkpoint session backed by a worker-independent replica service."""

    def __init__(
        self,
        *,
        rank: int,
        group_ranks: tuple[int, ...],
        service_endpoint: str,
        mirror_service_endpoint: str | None = None,
        manifest_root: Path | str,
        max_local_store_bytes: int,
        record_manifests: bool = True,
        collect_store_stats: bool = True,
        target_rank: int | None = None,
        timeout_seconds: float = 60,
    ):
        if len(group_ranks) != 2 or rank not in group_ranks:
            raise ValueError(f"service session requires a rank pair, got {group_ranks}")
        self.rank = rank
        self.group_ranks = group_ranks
        self.peer_rank = group_ranks[1] if group_ranks[0] == rank else group_ranks[0]
        self.target_rank = self.peer_rank if target_rank is None else target_rank
        if self.target_rank not in group_ranks:
            raise ValueError("target rank must belong to the checkpoint group")
        self.host = socket.gethostname()
        self.client = ReplicaServiceClient.from_endpoint(
            service_endpoint, timeout_seconds=timeout_seconds
        )
        self.client.health()
        self.mirror_client = None
        if (
            mirror_service_endpoint is not None
            and mirror_service_endpoint != self.client.endpoint
        ):
            self.mirror_client = ReplicaServiceClient.from_endpoint(
                mirror_service_endpoint, timeout_seconds=timeout_seconds
            )
            self.mirror_client.health()
        self.repository = ManifestRepository(manifest_root)
        self.store = ReplicaStore(max_local_store_bytes)
        self.last_restore_wire_bytes = 0
        self.record_manifests = record_manifests
        self.collect_store_stats = collect_store_stats

    def save(
        self,
        epoch: int,
        state: Any,
        *,
        state_is_snapshot: bool = False,
        object_checksums: bool = True,
    ) -> CommitResult:
        operation_start = time.perf_counter()
        serialize_start = time.perf_counter()
        serialized = serialize_checkpoint(
            state,
            state_is_snapshot=state_is_snapshot,
            object_checksums=object_checksums,
        )
        serialize_ms = (time.perf_counter() - serialize_start) * 1000
        manifest = create_prepared_manifest(
            epoch=epoch,
            source_rank=self.rank,
            source_host=self.host,
            target_rank=self.target_rank,
            target_host=self.client.endpoint,
            group_ranks=self.group_ranks,
            serialized=serialized,
            replica_count=2 if self.mirror_client is not None else 1,
            tolerated_failures=1 if self.mirror_client is not None else 0,
        )
        if self.record_manifests:
            self.repository.prepare(manifest)
        prepared = ReplicaEnvelope(manifest, serialized.payload)

        transport_start = time.perf_counter()
        clients = [self.client]
        if self.mirror_client is not None:
            clients.append(self.mirror_client)
        put_metrics = [client.put_prepared(prepared) for client in clients]
        committed = replace(
            manifest, state="committed", committed_time_ns=time.time_ns()
        )
        commit_metrics = [client.commit(committed) for client in clients]
        transport_ms = (time.perf_counter() - transport_start) * 1000

        if self.record_manifests:
            self.repository.record_replica(self.target_rank, committed)
            self.repository.publish_committed(committed)
        if len(serialized.payload) <= self.store.max_bytes:
            self.store.put(ReplicaEnvelope(committed, serialized.payload))
        store_bytes = -1
        if self.collect_store_stats:
            try:
                store_bytes = int(self.client.stats()["bytes_used"])
            except (ConnectionError, OSError, RuntimeError, TimeoutError):
                pass
        wire_bytes = sum(
            metrics.wire_bytes for metrics in put_metrics + commit_metrics
        )
        transport = TransportMetrics(
            metadata_bytes=max(0, wire_bytes - len(serialized.payload)),
            payload_bytes=len(serialized.payload),
            wire_bytes=wire_bytes,
            duration_ms=transport_ms,
        )
        return CommitResult(
            manifest=committed,
            serialize_ms=serialize_ms,
            transport=transport,
            commit_ms=(time.perf_counter() - operation_start) * 1000,
            store_bytes=store_bytes,
            put_service_ms=max(metrics.server_ms for metrics in put_metrics),
            commit_service_ms=max(metrics.server_ms for metrics in commit_metrics),
        )

    def drop_local(self, epoch: int) -> None:
        self.store.drop(self.rank, epoch)

    def restore_from_peer(self, epoch: int) -> Any:
        envelope, metrics = self.client.get(self.rank, epoch)
        self.last_restore_wire_bytes = metrics.wire_bytes
        if envelope.manifest.source_rank != self.rank:
            raise ValueError("replica service returned another source rank")
        if envelope.manifest.epoch != epoch or not envelope.manifest.committed:
            raise ValueError("replica service returned an invalid checkpoint epoch")
        state = deserialize_checkpoint(
            envelope.payload, envelope.manifest.payload_checksum
        )
        validate_checkpoint_objects(state, envelope.manifest.objects)
        if len(envelope.payload) <= self.store.max_bytes:
            self.store.put(envelope)
        return state
