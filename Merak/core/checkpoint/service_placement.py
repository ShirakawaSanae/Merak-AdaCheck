"""Failure-domain placement for worker-independent checkpoint services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ReplicaServicePlacement:
    primary_endpoint: str
    mirror_endpoint: str | None


def plan_replica_service_placement(
    *,
    rank: int,
    group_ranks: tuple[int, ...],
    ranks_per_node: int,
    service_endpoints: Sequence[str],
    mirror_local: bool,
    restore_host_index: int | None = None,
) -> ReplicaServicePlacement:
    """Map a checkpoint pair to peer/local services or a surviving service."""

    if len(group_ranks) != 2 or rank not in group_ranks:
        raise ValueError(f"service placement requires a rank pair, got {group_ranks}")
    if ranks_per_node < 1:
        raise ValueError("ranks_per_node must be positive")
    endpoints = tuple(str(endpoint) for endpoint in service_endpoints)
    if not endpoints or any(not endpoint for endpoint in endpoints):
        raise ValueError("service_endpoints cannot be empty")

    if restore_host_index is not None:
        try:
            primary = endpoints[restore_host_index]
        except IndexError as error:
            raise ValueError(
                f"no replica service for restore host index {restore_host_index}"
            ) from error
        return ReplicaServicePlacement(primary, None)

    peer_rank = group_ranks[1] if group_ranks[0] == rank else group_ranks[0]
    peer_host_index = peer_rank // ranks_per_node
    local_host_index = rank // ranks_per_node
    try:
        primary = endpoints[peer_host_index]
        local = endpoints[local_host_index]
    except IndexError as error:
        raise ValueError("service_endpoints do not cover all rank hosts") from error
    mirror = local if mirror_local and local != primary else None
    return ReplicaServicePlacement(primary, mirror)
