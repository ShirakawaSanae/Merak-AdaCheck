"""No-op compatibility object for the source snapshot's profiler calls."""

from __future__ import annotations

from typing import Any


class _DisabledCommunicationProfiler:
    """Preserve nnScaler execution while making profiling explicitly inert."""

    enabled = False

    def init_profiler_manager(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def record_start_time(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def record_end_time(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def compute_pp_idle_time(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def reset_comm_event(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def record_pp_send_event(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def record_pp_recv_event(self, *_args: Any, **_kwargs: Any) -> None:
        return None


comm_profiler = _DisabledCommunicationProfiler()
