"""Compatibility namespace for a removed external checkpoint-profiler hook.

This vendored nnScaler snapshot does not include MemCkpt.  The only supported
symbol is the disabled profiler used by two dangling imports in the source
snapshot; no checkpoint behavior is implemented here.
"""

from .memckpt_config import comm_profiler

__all__ = ["comm_profiler"]
