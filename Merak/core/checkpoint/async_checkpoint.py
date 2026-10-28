"""Bounded asynchronous checkpoint snapshots for replica-service sessions."""

from __future__ import annotations

import copy
import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from .recovery import CommitResult, ServiceRemoteCheckpointSession


class CheckpointBackpressureError(RuntimeError):
    """The checkpoint producer outran the bounded publisher."""


@dataclass(frozen=True)
class AsyncCommitResult:
    epoch: int
    capture_ms: float
    queue_ms: float
    committed_at_ns: int
    result: CommitResult
    publisher_queue_ms: float = 0.0
    snapshot_ready_wait_ms: float = 0.0
    enqueue_to_commit_ms: float = 0.0


@dataclass
class _SnapshotSlot:
    state: Any = None


@dataclass(frozen=True)
class _PendingSnapshot:
    epoch: int
    slot: _SnapshotSlot
    event: Any
    capture_ms: float
    enqueued_at: float


def _copy_snapshot(value: Any, previous: Any = None) -> Any:
    if isinstance(value, torch.Tensor):
        reusable = (
            isinstance(previous, torch.Tensor)
            and previous.device.type == "cpu"
            and previous.dtype == value.dtype
            and tuple(previous.shape) == tuple(value.shape)
        )
        if not reusable:
            previous = torch.empty_like(
                value,
                device="cpu",
                pin_memory=value.device.type == "cuda",
            )
        previous.copy_(value.detach(), non_blocking=value.device.type == "cuda")
        return previous
    if isinstance(value, defaultdict):
        target = defaultdict(value.default_factory)
        old = previous if isinstance(previous, Mapping) else {}
        target.update((key, _copy_snapshot(item, old.get(key))) for key, item in value.items())
        return target
    if isinstance(value, Mapping):
        old = previous if isinstance(previous, Mapping) else {}
        items = [(key, _copy_snapshot(item, old.get(key))) for key, item in value.items()]
        try:
            return value.__class__(items)
        except TypeError:
            return dict(items)
    if isinstance(value, list):
        old = previous if isinstance(previous, list) else []
        return [
            _copy_snapshot(item, old[index] if index < len(old) else None)
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        old = previous if isinstance(previous, tuple) else ()
        items = [
            _copy_snapshot(item, old[index] if index < len(old) else None)
            for index, item in enumerate(value)
        ]
        return value.__class__(*items) if hasattr(value, "_fields") else tuple(items)
    try:
        return copy.deepcopy(value)
    except (TypeError, RuntimeError):
        return value


def _first_cuda_device(value: Any) -> torch.device | None:
    if isinstance(value, torch.Tensor):
        return value.device if value.device.type == "cuda" else None
    if isinstance(value, Mapping):
        values = value.values()
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        return None
    for item in values:
        device = _first_cuda_device(item)
        if device is not None:
            return device
    return None


def _record_stream(value: Any, stream: torch.cuda.Stream) -> None:
    if isinstance(value, torch.Tensor):
        if value.device.type == "cuda":
            value.record_stream(stream)
        return
    if isinstance(value, Mapping):
        values = value.values()
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        return
    for item in values:
        _record_stream(item, stream)


class AsyncServiceRemoteCheckpointSession:
    """Capture without remote waits and publish through one FIFO worker."""

    def __init__(
        self,
        session: ServiceRemoteCheckpointSession,
        *,
        max_inflight: int = 2,
        object_checksums: bool = True,
        initial_slot_state_factory: Callable[[], Any] | None = None,
    ) -> None:
        if max_inflight < 1:
            raise ValueError("max_inflight must be positive")
        self.session = session
        self.object_checksums = object_checksums
        self.repository = session.repository
        self._available: queue.SimpleQueue[_SnapshotSlot] = queue.SimpleQueue()
        for _ in range(max_inflight):
            state = (
                None
                if initial_slot_state_factory is None
                else initial_slot_state_factory()
            )
            self._available.put(_SnapshotSlot(state))
        self._pending: queue.Queue[_PendingSnapshot | None] = queue.Queue()
        self._completed: queue.SimpleQueue[AsyncCommitResult] = queue.SimpleQueue()
        self._failure: BaseException | None = None
        self._last_epoch: int | None = None
        self._closed = False
        self._streams: dict[torch.device, torch.cuda.Stream] = {}
        self._worker = threading.Thread(
            target=self._publish_loop,
            name=f"checkpoint-publisher-rank-{session.rank}",
            daemon=True,
        )
        self._worker.start()

    def _raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError("asynchronous checkpoint publisher failed") from self._failure

    def enqueue(self, epoch: int, state: Any) -> float:
        return self._enqueue(epoch, state, copy_state=True)

    def enqueue_snapshot(self, epoch: int, state: Any) -> float:
        """Publish an immutable CPU snapshot owned by the caller."""
        if _first_cuda_device(state) is not None:
            raise ValueError("enqueue_snapshot requires CPU-only state")
        return self._enqueue(epoch, state, copy_state=False)

    def _enqueue(self, epoch: int, state: Any, *, copy_state: bool) -> float:
        if self._closed:
            raise RuntimeError("asynchronous checkpoint session is closed")
        self._raise_if_failed()
        if self._last_epoch is not None and epoch != self._last_epoch + 1:
            raise ValueError(
                f"checkpoint epochs must be contiguous: {epoch} after {self._last_epoch}"
            )
        try:
            slot = self._available.get_nowait()
        except queue.Empty as error:
            raise CheckpointBackpressureError(
                f"checkpoint epoch {epoch} exceeded the bounded inflight capacity"
            ) from error

        started = time.perf_counter()
        try:
            device = _first_cuda_device(state) if copy_state else None
            if not copy_state:
                slot.state = state
                event = None
            elif device is None:
                slot.state = _copy_snapshot(state, slot.state)
                event = None
            else:
                stream = self._streams.setdefault(device, torch.cuda.Stream(device=device))
                current = torch.cuda.current_stream(device)
                stream.wait_stream(current)
                with torch.cuda.stream(stream):
                    _record_stream(state, stream)
                    slot.state = _copy_snapshot(state, slot.state)
                    event = torch.cuda.Event()
                    event.record(stream)
        except BaseException:
            self._available.put(slot)
            raise
        capture_ms = (time.perf_counter() - started) * 1000
        self._last_epoch = epoch
        self._pending.put_nowait(
            _PendingSnapshot(epoch, slot, event, capture_ms, time.perf_counter())
        )
        return capture_ms

    def _publish_loop(self) -> None:
        while True:
            pending = self._pending.get()
            if pending is None:
                self._pending.task_done()
                return
            try:
                if self._failure is None:
                    dequeued_at = time.perf_counter()
                    if pending.event is not None:
                        pending.event.synchronize()
                    started = time.perf_counter()
                    publisher_queue_ms = (
                        dequeued_at - pending.enqueued_at
                    ) * 1000
                    snapshot_ready_wait_ms = (started - dequeued_at) * 1000
                    result = self.session.save(
                        pending.epoch,
                        pending.slot.state,
                        state_is_snapshot=True,
                        object_checksums=self.object_checksums,
                    )
                    enqueue_to_commit_ms = (
                        time.perf_counter() - pending.enqueued_at
                    ) * 1000
                    self._completed.put(
                        AsyncCommitResult(
                            epoch=pending.epoch,
                            capture_ms=pending.capture_ms,
                            queue_ms=publisher_queue_ms + snapshot_ready_wait_ms,
                            committed_at_ns=time.time_ns(),
                            result=result,
                            publisher_queue_ms=publisher_queue_ms,
                            snapshot_ready_wait_ms=snapshot_ready_wait_ms,
                            enqueue_to_commit_ms=enqueue_to_commit_ms,
                        )
                    )
            except BaseException as error:
                self._failure = error
            finally:
                self._available.put(pending.slot)
                self._pending.task_done()

    def poll(self) -> tuple[AsyncCommitResult, ...]:
        self._raise_if_failed()
        completed = []
        while True:
            try:
                completed.append(self._completed.get_nowait())
            except queue.Empty:
                break
        return tuple(completed)

    def close(self) -> tuple[AsyncCommitResult, ...]:
        if not self._closed:
            self._pending.join()
            self._pending.put(None)
            self._worker.join()
            self._closed = True
        self._raise_if_failed()
        return self.poll()

    def restore_from_peer(self, epoch: int) -> Any:
        if not self._closed:
            self.close()
        return self.session.restore_from_peer(epoch)
