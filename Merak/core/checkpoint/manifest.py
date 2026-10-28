"""Two-phase checkpoint manifests with atomic committed visibility."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .catalog import CheckpointObject, SerializedCheckpoint


@dataclass(frozen=True)
class CheckpointManifest:
    schema_version: int
    manifest_id: str
    epoch: int
    source_rank: int
    source_host: str
    target_rank: int
    target_host: str
    group_ranks: tuple[int, ...]
    replica_count: int
    tolerated_failures: int
    payload_bytes: int
    logical_bytes: int
    payload_checksum: str
    objects: tuple[CheckpointObject, ...]
    state: str
    prepared_time_ns: int
    committed_time_ns: int | None

    @property
    def committed(self) -> bool:
        return self.state == "committed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "epoch": self.epoch,
            "source_rank": self.source_rank,
            "source_host": self.source_host,
            "target_rank": self.target_rank,
            "target_host": self.target_host,
            "group_ranks": list(self.group_ranks),
            "replica_count": self.replica_count,
            "tolerated_failures": self.tolerated_failures,
            "payload_bytes": self.payload_bytes,
            "logical_bytes": self.logical_bytes,
            "payload_checksum": self.payload_checksum,
            "objects": [item.to_dict() for item in self.objects],
            "state": self.state,
            "prepared_time_ns": self.prepared_time_ns,
            "committed_time_ns": self.committed_time_ns,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointManifest":
        return cls(
            schema_version=int(value["schema_version"]),
            manifest_id=str(value["manifest_id"]),
            epoch=int(value["epoch"]),
            source_rank=int(value["source_rank"]),
            source_host=str(value["source_host"]),
            target_rank=int(value["target_rank"]),
            target_host=str(value["target_host"]),
            group_ranks=tuple(int(rank) for rank in value["group_ranks"]),
            replica_count=int(value["replica_count"]),
            tolerated_failures=int(value["tolerated_failures"]),
            payload_bytes=int(value["payload_bytes"]),
            logical_bytes=int(value["logical_bytes"]),
            payload_checksum=str(value["payload_checksum"]),
            objects=tuple(
                CheckpointObject.from_dict(item) for item in value["objects"]
            ),
            state=str(value["state"]),
            prepared_time_ns=int(value["prepared_time_ns"]),
            committed_time_ns=(
                int(value["committed_time_ns"])
                if value.get("committed_time_ns") is not None
                else None
            ),
        )


def create_prepared_manifest(
    *,
    epoch: int,
    source_rank: int,
    source_host: str,
    target_rank: int,
    target_host: str,
    group_ranks: tuple[int, ...],
    serialized: SerializedCheckpoint,
    replica_count: int = 2,
    tolerated_failures: int = 1,
) -> CheckpointManifest:
    immutable = {
        "schema_version": 1,
        "epoch": epoch,
        "source_rank": source_rank,
        "source_host": source_host,
        "target_rank": target_rank,
        "target_host": target_host,
        "group_ranks": list(group_ranks),
        "replica_count": replica_count,
        "tolerated_failures": tolerated_failures,
        "payload_bytes": len(serialized.payload),
        "logical_bytes": serialized.logical_bytes,
        "payload_checksum": serialized.payload_checksum,
        "objects": [item.to_dict() for item in serialized.objects],
    }
    encoded = json.dumps(immutable, sort_keys=True, separators=(",", ":")).encode()
    return CheckpointManifest(
        schema_version=immutable["schema_version"],
        manifest_id=hashlib.sha256(encoded).hexdigest(),
        epoch=epoch,
        source_rank=source_rank,
        source_host=source_host,
        target_rank=target_rank,
        target_host=target_host,
        group_ranks=tuple(group_ranks),
        replica_count=replica_count,
        tolerated_failures=tolerated_failures,
        payload_bytes=len(serialized.payload),
        logical_bytes=serialized.logical_bytes,
        payload_checksum=serialized.payload_checksum,
        objects=serialized.objects,
        state="prepared",
        prepared_time_ns=time.time_ns(),
        committed_time_ns=None,
    )


class ManifestRepository:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def _rank_dir(self, rank: int) -> Path:
        return self.root / f"rank_{rank:05d}"

    def prepared_path(self, manifest: CheckpointManifest) -> Path:
        return self._rank_dir(manifest.source_rank) / f"epoch_{manifest.epoch:08d}.prepared.json"

    def committed_path(self, manifest: CheckpointManifest) -> Path:
        return self._rank_dir(manifest.source_rank) / f"epoch_{manifest.epoch:08d}.manifest.json"

    def replica_path(self, holder_rank: int, manifest: CheckpointManifest) -> Path:
        return (
            self.root
            / "replicas"
            / f"holder_{holder_rank:05d}"
            / f"source_{manifest.source_rank:05d}"
            / f"epoch_{manifest.epoch:08d}.manifest.json"
        )

    @staticmethod
    def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)

    def prepare(self, manifest: CheckpointManifest) -> Path:
        if manifest.state != "prepared":
            raise ValueError("only prepared manifests can enter the prepare phase")
        path = self.prepared_path(manifest)
        self._atomic_write(path, manifest.to_dict())
        return path

    def commit(self, manifest: CheckpointManifest) -> CheckpointManifest:
        if manifest.state != "prepared":
            raise ValueError("manifest must be prepared before commit")
        committed = replace(
            manifest, state="committed", committed_time_ns=time.time_ns()
        )
        return self.publish_committed(committed)

    def publish_committed(
        self, manifest: CheckpointManifest
    ) -> CheckpointManifest:
        if not manifest.committed:
            raise ValueError("only a committed manifest may be published")
        self._atomic_write(self.committed_path(manifest), manifest.to_dict())
        return manifest

    def record_replica(
        self, holder_rank: int, manifest: CheckpointManifest
    ) -> Path:
        if not manifest.committed:
            raise ValueError("a holder may publish only a committed replica manifest")
        path = self.replica_path(holder_rank, manifest)
        self._atomic_write(path, manifest.to_dict())
        return path

    def read(self, path: Path | str) -> CheckpointManifest:
        return CheckpointManifest.from_dict(json.loads(Path(path).read_text()))

    def latest_committed(self, source_rank: int) -> CheckpointManifest | None:
        rank_dir = self._rank_dir(source_rank)
        candidates = sorted(rank_dir.glob("epoch_*.manifest.json"))
        if not candidates:
            return None
        manifest = self.read(candidates[-1])
        if not manifest.committed:
            raise ValueError(f"non-committed manifest published at {candidates[-1]}")
        return manifest
