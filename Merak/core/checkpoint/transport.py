"""Pairwise CPU/Gloo transport for checkpoint replica envelopes."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import P2POp

from .manifest import CheckpointManifest


@dataclass(frozen=True)
class ReplicaEnvelope:
    manifest: CheckpointManifest
    payload: bytes


@dataclass(frozen=True)
class TransportMetrics:
    metadata_bytes: int
    payload_bytes: int
    wire_bytes: int
    duration_ms: float


def _wait_all(works: list[Any]) -> None:
    for work in works:
        work.wait()


class GlooPairTransport:
    """Symmetric exchange for one fixed two-rank checkpoint group."""

    def __init__(self, group: object, group_ranks: tuple[int, ...]):
        if len(group_ranks) != 2:
            raise ValueError(f"GlooPairTransport requires a rank pair, got {group_ranks}")
        rank = dist.get_rank()
        if rank not in group_ranks:
            raise ValueError(f"rank {rank} is absent from checkpoint group {group_ranks}")
        self.group = group
        self.group_ranks = group_ranks
        self.rank = rank
        self.peer_rank = group_ranks[1] if group_ranks[0] == rank else group_ranks[0]

    def _exchange_bytes(self, payload: bytes) -> bytes:
        send_size = torch.tensor([len(payload)], dtype=torch.int64)
        recv_size = torch.empty(1, dtype=torch.int64)
        _wait_all(
            dist.batch_isend_irecv(
                [
                    P2POp(dist.isend, send_size, self.peer_rank, self.group),
                    P2POp(dist.irecv, recv_size, self.peer_rank, self.group),
                ]
            )
        )
        send_buffer = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        recv_buffer = torch.empty(int(recv_size.item()), dtype=torch.uint8)
        _wait_all(
            dist.batch_isend_irecv(
                [
                    P2POp(dist.isend, send_buffer, self.peer_rank, self.group),
                    P2POp(dist.irecv, recv_buffer, self.peer_rank, self.group),
                ]
            )
        )
        return recv_buffer.numpy().tobytes()

    def exchange(self, envelope: ReplicaEnvelope) -> tuple[ReplicaEnvelope, TransportMetrics]:
        metadata = json.dumps(
            envelope.manifest.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        start = time.perf_counter()
        received_metadata = self._exchange_bytes(metadata)
        received_payload = self._exchange_bytes(envelope.payload)
        duration_ms = (time.perf_counter() - start) * 1000
        received_manifest = CheckpointManifest.from_dict(
            json.loads(received_metadata.decode("utf-8"))
        )
        return (
            ReplicaEnvelope(received_manifest, received_payload),
            TransportMetrics(
                metadata_bytes=len(metadata),
                payload_bytes=len(envelope.payload),
                wire_bytes=16 + len(metadata) + len(envelope.payload),
                duration_ms=duration_ms,
            ),
        )

    def exchange_manifest(self, manifest: CheckpointManifest) -> CheckpointManifest:
        metadata = json.dumps(
            manifest.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        received = self._exchange_bytes(metadata)
        return CheckpointManifest.from_dict(json.loads(received.decode("utf-8")))

    def acknowledge(self, valid: bool) -> bool:
        send_ack = torch.tensor([1 if valid else 0], dtype=torch.int64)
        recv_ack = torch.empty(1, dtype=torch.int64)
        _wait_all(
            dist.batch_isend_irecv(
                [
                    P2POp(dist.isend, send_ack, self.peer_rank, self.group),
                    P2POp(dist.irecv, recv_ack, self.peer_rank, self.group),
                ]
            )
        )
        return bool(recv_ack.item())
