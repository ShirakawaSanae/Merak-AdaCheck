"""Topology-aware local-rank pairing for remote checkpoints.

Host placement and local GPU/NIC placement are separate decisions.  This
module handles the latter as a deterministic bipartite assignment between two
hosts.  It deliberately consumes measured costs instead of embedding a
machine-specific GPU numbering assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Sequence

RankPairingPolicy = Literal["identity", "cross_numa", "profile_aware"]


@dataclass(frozen=True)
class LocalRankPairing:
    """Map each local rank on the first host to one on the peer host."""

    policy: RankPairingPolicy
    permutation: tuple[int, ...]
    estimated_cost: float

    def peer_local_rank(self, local_rank: int, *, first_host: bool) -> int:
        if local_rank < 0 or local_rank >= len(self.permutation):
            raise ValueError(f"local rank {local_rank} is outside the pairing")
        if first_host:
            return self.permutation[local_rank]
        return self.permutation.index(local_rank)


def _as_nonnegative_tuple(
    values: Sequence[int | float] | None,
    size: int,
    name: str,
    default: float,
) -> tuple[float, ...]:
    if values is None:
        return (default,) * size
    converted = tuple(float(value) for value in values)
    if len(converted) != size:
        raise ValueError(f"{name} must contain {size} entries, got {len(converted)}")
    if any(value < 0 for value in converted):
        raise ValueError(f"{name} cannot contain negative values")
    return converted


def bidirectional_pairing_cost(
    permutation: Sequence[int],
    first_payload_bytes: Sequence[int | float],
    second_payload_bytes: Sequence[int | float],
    first_nic_costs: Sequence[int | float],
    second_nic_costs: Sequence[int | float],
) -> float:
    """Return a measured-profile proxy for bidirectional transfer cost.

    For pair ``(i, j)``, the first host sends its shard through local egress
    ``i`` and remote ingress ``j``; the reverse transfer follows ``j -> i``.
    The cost is a planner score, not elapsed time or predicted wire bytes.
    """

    size = len(permutation)
    sequences = (
        first_payload_bytes,
        second_payload_bytes,
        first_nic_costs,
        second_nic_costs,
    )
    if any(len(sequence) != size for sequence in sequences):
        raise ValueError("all pairing cost inputs must have the same length")
    if sorted(permutation) != list(range(size)):
        raise ValueError(f"not a local-rank permutation: {tuple(permutation)}")

    total = 0.0
    for first_rank, second_rank in enumerate(permutation):
        first_bytes = float(first_payload_bytes[first_rank])
        second_bytes = float(second_payload_bytes[second_rank])
        if first_bytes < 0 or second_bytes < 0:
            raise ValueError("payload bytes cannot be negative")
        path_cost = float(first_nic_costs[first_rank]) + float(
            second_nic_costs[second_rank]
        )
        total += (first_bytes + second_bytes) * path_cost
    return total


def _minimum_cost_permutation(cost_matrix: Sequence[Sequence[float]]) -> tuple[int, ...]:
    """Solve the small host-local assignment exactly with bitmask DP."""

    size = len(cost_matrix)
    if size < 1 or any(len(row) != size for row in cost_matrix):
        raise ValueError("cost matrix must be non-empty and square")
    if size > 16:
        raise ValueError("exact local-rank assignment is limited to 16 ranks per host")

    @lru_cache(maxsize=None)
    def solve(first_rank: int, used_mask: int) -> tuple[float, tuple[int, ...]]:
        if first_rank == size:
            return 0.0, ()
        best: tuple[float, tuple[int, ...]] | None = None
        for second_rank in range(size):
            bit = 1 << second_rank
            if used_mask & bit:
                continue
            tail_cost, tail = solve(first_rank + 1, used_mask | bit)
            candidate = (
                float(cost_matrix[first_rank][second_rank]) + tail_cost,
                (second_rank,) + tail,
            )
            if best is None or candidate < best:
                best = candidate
        if best is None:
            raise RuntimeError("assignment solver reached an invalid state")
        return best

    return solve(0, 0)[1]


def plan_local_rank_pairing(
    ranks_per_node: int,
    policy: RankPairingPolicy = "identity",
    *,
    first_payload_bytes: Sequence[int | float] | None = None,
    second_payload_bytes: Sequence[int | float] | None = None,
    first_nic_costs: Sequence[int | float] | None = None,
    second_nic_costs: Sequence[int | float] | None = None,
) -> LocalRankPairing:
    """Plan a deterministic, bijective rank pairing between two hosts."""

    if ranks_per_node < 1:
        raise ValueError("ranks_per_node must be at least one")
    first_payload = _as_nonnegative_tuple(
        first_payload_bytes, ranks_per_node, "first_payload_bytes", 1.0
    )
    second_payload = _as_nonnegative_tuple(
        second_payload_bytes, ranks_per_node, "second_payload_bytes", 1.0
    )
    first_cost = _as_nonnegative_tuple(
        first_nic_costs, ranks_per_node, "first_nic_costs", 1.0
    )
    second_cost = _as_nonnegative_tuple(
        second_nic_costs, ranks_per_node, "second_nic_costs", 1.0
    )

    if policy == "identity":
        permutation = tuple(range(ranks_per_node))
    elif policy == "cross_numa":
        if ranks_per_node % 2:
            raise ValueError("cross_numa requires an even ranks_per_node")
        half = ranks_per_node // 2
        permutation = tuple((rank + half) % ranks_per_node for rank in range(ranks_per_node))
    elif policy == "profile_aware":
        cost_matrix = tuple(
            tuple(
                (first_payload[first_rank] + second_payload[second_rank])
                * (first_cost[first_rank] + second_cost[second_rank])
                for second_rank in range(ranks_per_node)
            )
            for first_rank in range(ranks_per_node)
        )
        permutation = _minimum_cost_permutation(cost_matrix)
    else:
        raise ValueError(f"unsupported rank pairing policy: {policy}")

    estimated_cost = bidirectional_pairing_cost(
        permutation,
        first_payload,
        second_payload,
        first_cost,
        second_cost,
    )
    return LocalRankPairing(
        policy=policy,
        permutation=permutation,
        estimated_cost=estimated_cost,
    )
