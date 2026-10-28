"""Low-overhead communication interval profiling primitives.

Callers supply timestamps from one monotonic clock domain, normally
``time.monotonic_ns()``.  This module does not read clocks, synchronize CUDA,
or interact with distributed process groups.
"""

from __future__ import annotations

import json
import os
from statistics import fmean
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple


Interval = Tuple[int, int]
_SCHEMA_VERSION = 1


def _timestamp(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer monotonic timestamp")
    return value


def _interval(start_ns: int, end_ns: int) -> Interval:
    start = _timestamp(start_ns, "start_ns")
    end = _timestamp(end_ns, "end_ns")
    if end < start:
        raise ValueError("end_ns must not precede start_ns")
    return start, end


def clip_sort_merge(
    intervals: Iterable[Interval], start_ns: int, end_ns: int
) -> list[Interval]:
    """Clip, sort, and merge overlapping or adjacent intervals."""

    horizon_start, horizon_end = _interval(start_ns, end_ns)
    clipped = []
    for interval_start, interval_end in intervals:
        current_start, current_end = _interval(interval_start, interval_end)
        current_start = max(horizon_start, current_start)
        current_end = min(horizon_end, current_end)
        if current_end > current_start:
            clipped.append((current_start, current_end))
    if not clipped:
        return []

    clipped.sort()
    merged = [clipped[0]]
    for current_start, current_end in clipped[1:]:
        previous_start, previous_end = merged[-1]
        if current_start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, current_end))
        else:
            merged.append((current_start, current_end))
    return merged


def idle_slack_ns(
    start_ns: int,
    end_ns: int,
    busy_intervals: Iterable[Interval],
    *,
    min_gap_ns: int = 0,
) -> int:
    """Return horizon time outside busy intervals, ignoring gaps below a threshold."""

    horizon_start, horizon_end = _interval(start_ns, end_ns)
    minimum_gap = _timestamp(min_gap_ns, "min_gap_ns")
    idle = 0
    cursor = horizon_start
    for busy_start, busy_end in clip_sort_merge(
        busy_intervals, horizon_start, horizon_end
    ):
        gap = busy_start - cursor
        if gap >= minimum_gap:
            idle += gap
        cursor = busy_end
    tail = horizon_end - cursor
    if tail >= minimum_gap:
        idle += tail
    return idle


def directional_idle_slack_ns(
    start_ns: int,
    end_ns: int,
    source_send_busy: Iterable[Interval],
    destination_recv_busy: Iterable[Interval],
    *,
    min_gap_ns: int = 0,
) -> int:
    """Return slack outside the union of source-send and destination-recv work."""

    return idle_slack_ns(
        start_ns,
        end_ns,
        tuple(source_send_busy) + tuple(destination_recv_busy),
        min_gap_ns=min_gap_ns,
    )


def quantile(values: Sequence[int], probability: float) -> Optional[float]:
    """Return a linear-interpolated quantile, or ``None`` for no samples."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0.0, 1.0]")
    if not values:
        return None
    ordered = sorted(_timestamp(value, "sample_ns") for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_tail_ns(values: Sequence[int]) -> dict[str, Optional[float] | int]:
    """Summarize duration samples without imposing a transport interpretation."""

    if not values:
        return {
            "count": 0,
            "mean_ns": None,
            "min_ns": None,
            "p50_ns": None,
            "p95_ns": None,
            "p99_ns": None,
            "max_ns": None,
        }
    samples = [_timestamp(value, "sample_ns") for value in values]
    return {
        "count": len(samples),
        "mean_ns": fmean(samples),
        "min_ns": min(samples),
        "p50_ns": quantile(samples, 0.50),
        "p95_ns": quantile(samples, 0.95),
        "p99_ns": quantile(samples, 0.99),
        "max_ns": max(samples),
    }


class CommunicationProfile:
    """Optional JSONL interval recorder with no timing or synchronization side effects."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        jsonl_path: Optional[os.PathLike[str] | str] = None,
        retain_events: bool = False,
    ) -> None:
        self.enabled = bool(enabled)
        self._retain_events = bool(retain_events)
        self._events: list[dict[str, Any]] = []
        self._descriptor: Optional[int] = None
        if self.enabled and jsonl_path is not None:
            path = os.fspath(jsonl_path)
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._descriptor = os.open(
                path,
                os.O_APPEND | os.O_CREAT | os.O_WRONLY,
                0o644,
            )

    @property
    def events(self) -> tuple[Mapping[str, Any], ...]:
        """Retained records; empty unless ``retain_events=True`` was requested."""

        return tuple(self._events)

    def record_event(self, name: str, at_ns: int, **fields: Any) -> bool:
        """Record a point event supplied by the caller's monotonic clock."""

        if not self.enabled:
            return False
        record = {
            "schema_version": _SCHEMA_VERSION,
            "kind": "event",
            "name": str(name),
            "at_ns": _timestamp(at_ns, "at_ns"),
            "fields": fields,
        }
        self._write(record)
        return True

    def record_interval(
        self, name: str, start_ns: int, end_ns: int, **fields: Any
    ) -> bool:
        """Record a completed interval supplied by the caller's monotonic clock."""

        if not self.enabled:
            return False
        start, end = _interval(start_ns, end_ns)
        record = {
            "schema_version": _SCHEMA_VERSION,
            "kind": "interval",
            "name": str(name),
            "start_ns": start,
            "end_ns": end,
            "duration_ns": end - start,
            "fields": fields,
        }
        self._write(record)
        return True

    def close(self) -> None:
        """Close the optional JSONL descriptor.  The operation is idempotent."""

        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None

    def __enter__(self) -> "CommunicationProfile":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _write(self, record: dict[str, Any]) -> None:
        if self._retain_events:
            self._events.append(record)
        if self._descriptor is not None:
            payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
            os.write(self._descriptor, (payload + "\n").encode("utf-8"))
