"""Unix-domain shared-tensor handoff for a host-local materializer."""

from __future__ import annotations

import copy
import hashlib
import os
import socket
import stat
import threading
import time
from collections.abc import Mapping
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Any

import torch

from .catalog import (
    SerializedCheckpoint,
    checkpoint_state_checksum,
    iter_checkpoint_objects,
)
from .manifest import CheckpointManifest, ManifestRepository
from .recovery import CommitResult
from .transport import TransportMetrics


def _authkey(path: Path | str) -> bytes:
    return hashlib.sha256(str(path).encode()).digest()


def _configure_tensor_sharing() -> None:
    # Named shared-memory handles avoid multiprocessing's per-process file
    # descriptor authentication, which is unsuitable for independent workers.
    if torch.multiprocessing.get_sharing_strategy() != "file_system":
        torch.multiprocessing.set_sharing_strategy("file_system")


def _shared_snapshot(value: Any, reusable: Any = None) -> Any:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("shared-tensor transport requires a CPU snapshot")
        source = value.detach().contiguous()
        if (
            isinstance(reusable, torch.Tensor)
            and reusable.is_shared()
            and reusable.shape == source.shape
            and reusable.dtype == source.dtype
        ):
            result = reusable
        else:
            result = torch.empty_like(source).share_memory_()
        result.copy_(source)
        return result
    if isinstance(value, Mapping):
        previous = reusable if isinstance(reusable, Mapping) else {}
        return {
            key: _shared_snapshot(item, previous.get(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        previous = reusable if isinstance(reusable, list) else []
        return [
            _shared_snapshot(item, previous[index] if index < len(previous) else None)
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        previous = reusable if isinstance(reusable, tuple) else ()
        return tuple(
            _shared_snapshot(item, previous[index] if index < len(previous) else None)
            for index, item in enumerate(value)
        )
    return copy.deepcopy(value)


def describe_shared_update(state: Mapping[str, Any]) -> SerializedCheckpoint:
    if state.get("kind") != "optimizer_update":
        raise ValueError("shared-tensor service accepts optimizer updates only")
    objects = tuple(iter_checkpoint_objects(state, include_checksums=False))
    rank_specific = {
        key: state[key]
        for key in ("base_id", "binding", "runtime_state")
        if key in state
    }
    common = {key: value for key, value in state.items() if key not in rank_specific}
    equivalence_checksum = checkpoint_state_checksum(common)
    payload_checksum = hashlib.sha256(
        f"{equivalence_checksum}:{checkpoint_state_checksum(rank_specific)}".encode()
    ).hexdigest()
    return SerializedCheckpoint(
        payload=b"",
        payload_checksum=payload_checksum,
        objects=objects,
        logical_bytes=sum(item.logical_bytes for item in objects),
        equivalence_checksum=equivalence_checksum,
    )


class SharedTensorServer:
    def __init__(self, path: Path | str, store: Any):
        self.path = Path(path)
        self.store = store
        self.host = socket.gethostname()
        self._listener: Listener | None = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            if not stat.S_ISSOCK(self.path.stat().st_mode):
                raise ValueError(f"refusing to replace non-socket path: {self.path}")
            self.path.unlink()
        self._listener = Listener(
            str(self.path), family="AF_UNIX", authkey=_authkey(self.path)
        )
        os.chmod(self.path, 0o600)
        self._thread = threading.Thread(
            target=self._serve, name="shared-tensor-listener", daemon=True
        )
        self._thread.start()

    def _serve(self) -> None:
        assert self._listener is not None
        while not self._stopping.is_set():
            connection = self._listener.accept()
            threading.Thread(
                target=self._handle, args=(connection,), daemon=True
            ).start()

    def _handle(self, connection: Any) -> None:
        try:
            request = connection.recv()
            operation = request.get("operation")
            if operation == "HEALTH":
                response = {"status": "ok"}
            elif operation == "SAVE_UPDATE":
                manifest, describe_ms, apply_ms, equivalence_checksum = (
                    self.store.commit_shared_update(
                        epoch=int(request["epoch"]),
                        source_rank=int(request["source_rank"]),
                        source_host=str(request["source_host"]),
                        target_rank=int(request["target_rank"]),
                        target_host=self.host,
                        group_ranks=tuple(
                            int(rank) for rank in request["group_ranks"]
                        ),
                        state=request["state"],
                    )
                )
                response = {
                    "status": "ok",
                    "manifest": manifest.to_dict(),
                    "describe_ms": describe_ms,
                    "apply_ms": apply_ms,
                    "equivalence_checksum": equivalence_checksum,
                }
            elif operation == "APPLIED_THROUGH":
                response = {
                    "status": "ok",
                    "epoch": self.store.materialized_epoch(
                        int(request["source_rank"])
                    ),
                }
            elif operation == "WAIT_APPLIED":
                response = {
                    "status": "ok",
                    "epoch": self.store.wait_materialized(
                        int(request["source_rank"]),
                        int(request["epoch"]),
                        float(request["timeout_seconds"]),
                    ),
                }
            elif operation == "STOP" and self._stopping.is_set():
                response = {"status": "ok"}
            else:
                raise ValueError(f"unsupported shared-tensor operation: {operation}")
        except Exception as error:
            response = {
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
        try:
            connection.send(response)
        finally:
            connection.close()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stopping.set()
        connection = Client(
            str(self.path), family="AF_UNIX", authkey=_authkey(self.path)
        )
        connection.send({"operation": "STOP"})
        connection.recv()
        connection.close()
        self._thread.join(timeout=5)
        if self._listener is not None:
            self._listener.close()
        if self.path.exists() and stat.S_ISSOCK(self.path.stat().st_mode):
            self.path.unlink()


class SharedTensorCheckpointSession:
    """Service-session interface backed by shared CPU tensor storage."""

    def __init__(
        self,
        *,
        rank: int,
        group_ranks: tuple[int, int],
        target_rank: int,
        socket_path: Path | str,
        manifest_root: Path | str,
        max_shared_slots: int = 2,
        timeout_seconds: float = 60,
    ) -> None:
        if rank not in group_ranks or target_rank != rank:
            raise ValueError("shared updates must target their source rank")
        self.rank = rank
        self.group_ranks = group_ranks
        self.target_rank = target_rank
        self.socket_path = Path(socket_path)
        _configure_tensor_sharing()
        self.host = socket.gethostname()
        self.repository = ManifestRepository(manifest_root)
        if max_shared_slots < 2:
            raise ValueError("sealed shared transport requires at least two slots")
        self.max_shared_slots = max_shared_slots
        self.timeout_seconds = timeout_seconds
        self._free_shared_states: list[Any] = []
        self._leased_shared_states: dict[int, Any] = {}
        self._request({"operation": "HEALTH"})

    def _reclaim(self) -> None:
        response = self._request(
            {"operation": "APPLIED_THROUGH", "source_rank": self.rank}
        )
        applied = int(response["epoch"])
        for epoch in sorted(self._leased_shared_states):
            if epoch > applied:
                break
            self._free_shared_states.append(self._leased_shared_states.pop(epoch))

    def wait_applied(self, epoch: int) -> None:
        self._request(
            {
                "operation": "WAIT_APPLIED",
                "source_rank": self.rank,
                "epoch": epoch,
                "timeout_seconds": self.timeout_seconds,
            }
        )
        self._reclaim()

    def _request(self, request: Mapping[str, Any]) -> dict[str, Any]:
        connection = Client(
            str(self.socket_path),
            family="AF_UNIX",
            authkey=_authkey(self.socket_path),
        )
        try:
            connection.send(dict(request))
            response = connection.recv()
        finally:
            connection.close()
        if response.get("status") != "ok":
            raise RuntimeError(response.get("error", "shared-tensor service failed"))
        return response

    def save(
        self,
        epoch: int,
        state: Any,
        *,
        state_is_snapshot: bool = False,
        object_checksums: bool = True,
    ) -> CommitResult:
        if not state_is_snapshot or object_checksums:
            raise ValueError("shared-tensor updates require an unchecked CPU snapshot")
        started = time.perf_counter()
        slot_wait_started = time.perf_counter()
        deadline = time.monotonic() + self.timeout_seconds
        while (
            not self._free_shared_states
            and len(self._leased_shared_states) >= self.max_shared_slots
        ):
            self._reclaim()
            if not self._free_shared_states:
                if time.monotonic() >= deadline:
                    raise TimeoutError("no sealed shared-tensor slot became reusable")
                time.sleep(0.01)
        reusable = self._free_shared_states.pop() if self._free_shared_states else None
        slot_wait_ms = (time.perf_counter() - slot_wait_started) * 1000
        copy_started = time.perf_counter()
        shared_state = _shared_snapshot(state, reusable)
        copy_ms = (time.perf_counter() - copy_started) * 1000
        transport_started = time.perf_counter()
        try:
            response = self._request(
                {
                    "operation": "SAVE_UPDATE",
                    "epoch": epoch,
                    "source_rank": self.rank,
                    "source_host": self.host,
                    "target_rank": self.target_rank,
                    "group_ranks": self.group_ranks,
                    "state": shared_state,
                }
            )
        except Exception:
            self._free_shared_states.append(shared_state)
            raise
        self._leased_shared_states[epoch] = shared_state
        transport_ms = (time.perf_counter() - transport_started) * 1000
        manifest = CheckpointManifest.from_dict(response["manifest"])
        if (
            not manifest.committed
            or manifest.epoch != epoch
            or manifest.source_rank != self.rank
            or manifest.target_rank != self.target_rank
            or manifest.group_ranks != self.group_ranks
        ):
            raise ValueError("shared-tensor service returned an invalid manifest")
        self.repository.publish_committed(manifest)
        self.repository.record_replica(self.target_rank, manifest)
        return CommitResult(
            manifest=manifest,
            serialize_ms=slot_wait_ms + copy_ms,
            transport=TransportMetrics(
                metadata_bytes=0,
                payload_bytes=0,
                wire_bytes=0,
                duration_ms=transport_ms,
            ),
            commit_ms=(time.perf_counter() - started) * 1000,
            store_bytes=-1,
            put_service_ms=float(response["describe_ms"]),
            commit_service_ms=float(response["apply_ms"]),
            transport_kind="shared_tensor",
            shared_memory_bytes=manifest.logical_bytes,
            equivalence_checksum=str(response["equivalence_checksum"]),
            shared_slot_wait_ms=slot_wait_ms,
            shared_copy_ms=copy_ms,
        )
