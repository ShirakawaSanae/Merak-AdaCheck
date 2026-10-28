"""Pair-local checkpoint delta opportunity model for two hosts."""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from math import inf, isfinite, isnan
from typing import Any, Mapping, Sequence

from .opportunity import RankGrouping
from .reduced import ReducedCheckpoint


# The only selectable two-host peer grouping policies. Availability controls and
# exact diagnostics deliberately remain outside this registry.
FORMAL_PEER_GROUPING_POLICIES = (
    "fixed",
    "random",
    "redundancy_max",
    "profile_exact",
)


@dataclass(frozen=True)
class PeerResourceProfile:
    """Measured capacities used only as a deterministic makespan proxy."""

    first_to_second_bytes_per_second: float
    second_to_first_bytes_per_second: float
    first_service_bytes_per_second: float = inf
    second_service_bytes_per_second: float = inf

    def __post_init__(self) -> None:
        network_values = (
            self.first_to_second_bytes_per_second,
            self.second_to_first_bytes_per_second,
        )
        service_values = (
            self.first_service_bytes_per_second,
            self.second_service_bytes_per_second,
        )
        if any(not isfinite(value) or value <= 0 for value in network_values):
            raise ValueError("network capacities must be finite and positive")
        if any(isnan(value) or value <= 0 for value in service_values):
            raise ValueError("service capacities must be positive")

    def to_dict(self) -> dict[str, float | str]:
        def encode(value: float) -> float | str:
            return "infinity" if value == inf else value

        return {
            "first_to_second_bytes_per_second": encode(
                self.first_to_second_bytes_per_second
            ),
            "second_to_first_bytes_per_second": encode(
                self.second_to_first_bytes_per_second
            ),
            "first_service_bytes_per_second": encode(
                self.first_service_bytes_per_second
            ),
            "second_service_bytes_per_second": encode(
                self.second_service_bytes_per_second
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PeerResourceProfile":
        def decode(item: Any) -> float:
            return inf if item == "infinity" else float(item)

        return cls(
            first_to_second_bytes_per_second=decode(
                value["first_to_second_bytes_per_second"]
            ),
            second_to_first_bytes_per_second=decode(
                value["second_to_first_bytes_per_second"]
            ),
            first_service_bytes_per_second=decode(
                value.get("first_service_bytes_per_second", "infinity")
            ),
            second_service_bytes_per_second=decode(
                value.get("second_service_bytes_per_second", "infinity")
            ),
        )


@dataclass(frozen=True)
class PeerDeltaEdge:
    first_rank: int
    second_rank: int
    first_required_bytes: int
    second_required_bytes: int
    overlap_bytes: int
    first_to_second_value_ids: tuple[str, ...]
    second_to_first_value_ids: tuple[str, ...]
    first_to_second_bytes: int
    second_to_first_bytes: int

    @property
    def total_transfer_bytes(self) -> int:
        return self.first_to_second_bytes + self.second_to_first_bytes

    @property
    def full_duplex_peak_bytes(self) -> int:
        return max(self.first_to_second_bytes, self.second_to_first_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_rank": self.first_rank,
            "second_rank": self.second_rank,
            "first_required_bytes": self.first_required_bytes,
            "second_required_bytes": self.second_required_bytes,
            "overlap_bytes": self.overlap_bytes,
            "first_to_second_value_ids": list(self.first_to_second_value_ids),
            "second_to_first_value_ids": list(self.second_to_first_value_ids),
            "first_to_second_bytes": self.first_to_second_bytes,
            "second_to_first_bytes": self.second_to_first_bytes,
            "total_transfer_bytes": self.total_transfer_bytes,
            "full_duplex_peak_bytes": self.full_duplex_peak_bytes,
        }


@dataclass(frozen=True)
class PeerDeltaMatrix:
    catalog_hash: str
    world_size: int
    ranks_per_host: int
    rank_value_ids: Mapping[int, tuple[str, ...]]
    object_bytes: Mapping[str, int]
    first_to_second_host_global_value_ids: tuple[str, ...]
    second_to_first_host_global_value_ids: tuple[str, ...]
    edges: tuple[PeerDeltaEdge, ...]
    profile: PeerResourceProfile | None = None

    @property
    def first_ranks(self) -> tuple[int, ...]:
        return tuple(range(self.ranks_per_host))

    @property
    def second_ranks(self) -> tuple[int, ...]:
        return tuple(range(self.ranks_per_host, self.world_size))

    def edge(self, first_rank: int, second_rank: int) -> PeerDeltaEdge:
        for edge in self.edges:
            if edge.first_rank == first_rank and edge.second_rank == second_rank:
                return edge
        raise KeyError(f"pair ({first_rank}, {second_rank}) is absent")

    def bytes_for(self, value_ids: Sequence[str]) -> int:
        return sum(self.object_bytes[value_id] for value_id in value_ids)

    @property
    def host_global_transfer_bytes(self) -> int:
        return self.bytes_for(self.first_to_second_host_global_value_ids) + self.bytes_for(
            self.second_to_first_host_global_value_ids
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "catalog_hash": self.catalog_hash,
            "world_size": self.world_size,
            "ranks_per_host": self.ranks_per_host,
            "availability": "same-epoch rank requirements assumed committed",
            "rank_value_ids": {
                str(rank): list(value_ids)
                for rank, value_ids in sorted(self.rank_value_ids.items())
            },
            "object_bytes": dict(sorted(self.object_bytes.items())),
            "host_global_control": {
                "first_to_second_value_ids": list(
                    self.first_to_second_host_global_value_ids
                ),
                "second_to_first_value_ids": list(
                    self.second_to_first_host_global_value_ids
                ),
                "transfer_bytes": self.host_global_transfer_bytes,
            },
            "profile": self.profile.to_dict() if self.profile is not None else None,
            "edges": [edge.to_dict() for edge in self.edges],
        }


@dataclass(frozen=True)
class PeerGroupingEvaluation:
    grouping: RankGrouping
    pair_local_transfer_bytes: int
    first_to_second_bytes: int
    second_to_first_bytes: int
    aggregate_pair_peak_bytes: int
    maximum_pair_peak_bytes: int
    overlap_bytes: int
    host_global_transfer_bytes: int
    makespan_proxy: float
    makespan_proxy_unit: str
    worst_rank_state_bytes: int
    planned_tolerated_host_failures: int
    static_failure_sets_covered: tuple[tuple[int, ...], ...]
    runtime_failure_sets_validated: tuple[tuple[int, ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "grouping": self.grouping.to_dict(),
            "pair_local_transfer_bytes": self.pair_local_transfer_bytes,
            "first_to_second_bytes": self.first_to_second_bytes,
            "second_to_first_bytes": self.second_to_first_bytes,
            "aggregate_pair_peak_bytes": self.aggregate_pair_peak_bytes,
            "maximum_pair_peak_bytes": self.maximum_pair_peak_bytes,
            "overlap_bytes": self.overlap_bytes,
            "host_global_transfer_bytes": self.host_global_transfer_bytes,
            "makespan_proxy": self.makespan_proxy,
            "makespan_proxy_unit": self.makespan_proxy_unit,
            "worst_rank_state_bytes": self.worst_rank_state_bytes,
            "planned_tolerated_host_failures": self.planned_tolerated_host_failures,
            "static_failure_sets_covered": [
                list(failed) for failed in self.static_failure_sets_covered
            ],
            "runtime_failure_sets_validated": [
                list(failed) for failed in self.runtime_failure_sets_validated
            ],
        }


def build_peer_delta_matrix(
    reduced: ReducedCheckpoint,
    *,
    ranks_per_host: int,
    catalog_hash: str,
    profile: PeerResourceProfile | None = None,
) -> PeerDeltaMatrix:
    """Build all cross-host rank edges from one frozen reduced catalog."""
    reduced.validate()
    ranks = tuple(sorted(int(rank) for rank in reduced.templates))
    rank_values: dict[int, set[str]] = {rank: set() for rank in ranks}
    object_bytes = {item.value_id: item.logical_bytes for item in reduced.objects}
    for item in reduced.objects:
        requirement_ranks = {requirement.rank for requirement in item.requirements}
        if requirement_ranks != set(item.source_ranks):
            raise ValueError(f"object {item.value_id} has inconsistent holders")
        for rank in requirement_ranks:
            rank_values[rank].add(item.value_id)
    return _build_peer_delta_matrix(
        rank_values,
        object_bytes,
        ranks_per_host=ranks_per_host,
        catalog_hash=catalog_hash,
        profile=profile,
    )


def build_peer_delta_matrix_from_catalog(
    objects: Sequence[Mapping[str, Any]],
    *,
    world_size: int,
    ranks_per_host: int,
    catalog_hash: str,
    profile: PeerResourceProfile | None = None,
) -> PeerDeltaMatrix:
    """Build a matrix from the immutable JSONL catalog without loading tensors."""
    rank_values: dict[int, set[str]] = {rank: set() for rank in range(world_size)}
    object_bytes: dict[str, int] = {}
    for item in objects:
        value_id = str(item["value_id"])
        logical_bytes = int(item["logical_bytes"])
        if not value_id or value_id in object_bytes or logical_bytes < 0:
            raise ValueError(f"invalid catalog object {value_id!r}")
        requirement_ranks = {
            int(requirement["rank"]) for requirement in item["requirements"]
        }
        source_ranks = {int(rank) for rank in item["source_ranks"]}
        if requirement_ranks != source_ranks or not requirement_ranks:
            raise ValueError(f"object {value_id} has inconsistent holders")
        if any(rank not in rank_values for rank in requirement_ranks):
            raise ValueError(f"object {value_id} has a rank outside the world")
        object_bytes[value_id] = logical_bytes
        for rank in requirement_ranks:
            rank_values[rank].add(value_id)
    if not object_bytes:
        raise ValueError("peer delta catalog cannot be empty")
    return _build_peer_delta_matrix(
        rank_values,
        object_bytes,
        ranks_per_host=ranks_per_host,
        catalog_hash=catalog_hash,
        profile=profile,
    )


def build_peer_delta_matrix_for_hosts_from_catalog(
    objects: Sequence[Mapping[str, Any]],
    *,
    world_size: int,
    ranks_per_host: int,
    first_host: int,
    second_host: int,
    catalog_hash: str,
) -> PeerDeltaMatrix:
    """Project a multi-host catalog onto one ordered host pair."""
    if world_size % ranks_per_host:
        raise ValueError("world_size must be divisible by ranks_per_host")
    host_count = world_size // ranks_per_host
    if not 0 <= first_host < second_host < host_count:
        raise ValueError("host pair must be distinct, ordered, and inside the world")
    selected = {
        first_host * ranks_per_host + local_rank: local_rank
        for local_rank in range(ranks_per_host)
    }
    selected.update(
        {
            second_host * ranks_per_host + local_rank: ranks_per_host + local_rank
            for local_rank in range(ranks_per_host)
        }
    )
    projected = []
    for item in objects:
        requirements = [
            {"rank": selected[int(requirement["rank"])]}
            for requirement in item["requirements"]
            if int(requirement["rank"]) in selected
        ]
        if not requirements:
            continue
        value = dict(item)
        value["requirements"] = requirements
        value["source_ranks"] = sorted(item["rank"] for item in requirements)
        projected.append(value)
    return build_peer_delta_matrix_from_catalog(
        projected,
        world_size=2 * ranks_per_host,
        ranks_per_host=ranks_per_host,
        catalog_hash=catalog_hash,
    )


def _build_peer_delta_matrix(
    rank_values: Mapping[int, set[str]],
    object_bytes: Mapping[str, int],
    *,
    ranks_per_host: int,
    catalog_hash: str,
    profile: PeerResourceProfile | None,
) -> PeerDeltaMatrix:
    ranks = tuple(sorted(rank_values))
    if ranks != tuple(range(len(ranks))):
        raise ValueError("peer delta ranks must be contiguous from zero")
    if len(ranks) != 2 * ranks_per_host:
        raise ValueError("peer delta matrix requires exactly two equal-sized hosts")
    if not catalog_hash:
        raise ValueError("catalog_hash is required")

    first_host_values = set().union(
        *(rank_values[rank] for rank in range(ranks_per_host))
    )
    second_host_values = set().union(
        *(rank_values[rank] for rank in range(ranks_per_host, len(ranks)))
    )

    def byte_sum(value_ids: set[str]) -> int:
        return sum(object_bytes[value_id] for value_id in value_ids)

    edges = []
    for first_rank in range(ranks_per_host):
        for second_rank in range(ranks_per_host, len(ranks)):
            first_values = rank_values[first_rank]
            second_values = rank_values[second_rank]
            first_delta = first_values - second_values
            second_delta = second_values - first_values
            edges.append(
                PeerDeltaEdge(
                    first_rank=first_rank,
                    second_rank=second_rank,
                    first_required_bytes=byte_sum(first_values),
                    second_required_bytes=byte_sum(second_values),
                    overlap_bytes=byte_sum(first_values & second_values),
                    first_to_second_value_ids=tuple(sorted(first_delta)),
                    second_to_first_value_ids=tuple(sorted(second_delta)),
                    first_to_second_bytes=byte_sum(first_delta),
                    second_to_first_bytes=byte_sum(second_delta),
                )
            )
    return PeerDeltaMatrix(
        catalog_hash=catalog_hash,
        world_size=len(ranks),
        ranks_per_host=ranks_per_host,
        rank_value_ids={
            rank: tuple(sorted(value_ids)) for rank, value_ids in rank_values.items()
        },
        object_bytes=object_bytes,
        first_to_second_host_global_value_ids=tuple(
            sorted(first_host_values - second_host_values)
        ),
        second_to_first_host_global_value_ids=tuple(
            sorted(second_host_values - first_host_values)
        ),
        edges=tuple(edges),
        profile=profile,
    )


def _validate_grouping(matrix: PeerDeltaMatrix, grouping: RankGrouping) -> None:
    flattened = [rank for group in grouping.groups for rank in group]
    if sorted(flattened) != list(range(matrix.world_size)):
        raise ValueError("grouping must contain every rank exactly once")
    for first, second in grouping.groups:
        if first not in matrix.first_ranks or second not in matrix.second_ranks:
            raise ValueError("peer groups must be ordered first-host/second-host pairs")


def _makespan_proxy(
    matrix: PeerDeltaMatrix,
    edges: Sequence[PeerDeltaEdge],
) -> tuple[float, str]:
    first_to_second = sum(edge.first_to_second_bytes for edge in edges)
    second_to_first = sum(edge.second_to_first_bytes for edge in edges)
    if matrix.profile is None:
        return (
            float(
                max(
                    first_to_second,
                    second_to_first,
                    max((edge.full_duplex_peak_bytes for edge in edges), default=0),
                )
            ),
            "unit_bandwidth_bytes",
        )
    profile = matrix.profile
    pair_seconds = max(
        (
            max(
                edge.first_to_second_bytes
                / profile.first_to_second_bytes_per_second,
                edge.second_to_first_bytes
                / profile.second_to_first_bytes_per_second,
            )
            for edge in edges
        ),
        default=0.0,
    )
    return (
        max(
            pair_seconds,
            first_to_second / profile.first_to_second_bytes_per_second,
            second_to_first / profile.second_to_first_bytes_per_second,
            first_to_second / profile.second_service_bytes_per_second,
            second_to_first / profile.first_service_bytes_per_second,
        ),
        "seconds",
    )


def _edge_proxy(matrix: PeerDeltaMatrix, edge: PeerDeltaEdge) -> float:
    if matrix.profile is None:
        return float(edge.full_duplex_peak_bytes)
    return max(
        edge.first_to_second_bytes
        / matrix.profile.first_to_second_bytes_per_second,
        edge.second_to_first_bytes
        / matrix.profile.second_to_first_bytes_per_second,
    )


def evaluate_peer_grouping(
    matrix: PeerDeltaMatrix, grouping: RankGrouping
) -> PeerGroupingEvaluation:
    """Evaluate delta cost and enumerate both single-host failure sets."""
    _validate_grouping(matrix, grouping)
    edges = tuple(matrix.edge(first, second) for first, second in grouping.groups)
    value_sets = {
        rank: set(value_ids) for rank, value_ids in matrix.rank_value_ids.items()
    }
    peer = {
        rank: other
        for first, second in grouping.groups
        for rank, other in ((first, second), (second, first))
    }
    for failed_host in (0, 1):
        failed_ranks = (
            matrix.first_ranks if failed_host == 0 else matrix.second_ranks
        )
        for failed_rank in failed_ranks:
            survivor = peer[failed_rank]
            edge = matrix.edge(*sorted((failed_rank, survivor)))
            delta = (
                set(edge.first_to_second_value_ids)
                if failed_rank == edge.first_rank
                else set(edge.second_to_first_value_ids)
            )
            available = value_sets[survivor] | delta
            if not value_sets[failed_rank] <= available:
                raise ValueError(
                    f"rank {failed_rank} is not recoverable after host {failed_host} fails"
                )

    first_to_second = sum(edge.first_to_second_bytes for edge in edges)
    second_to_first = sum(edge.second_to_first_bytes for edge in edges)
    makespan, unit = _makespan_proxy(matrix, edges)
    return PeerGroupingEvaluation(
        grouping=grouping,
        pair_local_transfer_bytes=first_to_second + second_to_first,
        first_to_second_bytes=first_to_second,
        second_to_first_bytes=second_to_first,
        aggregate_pair_peak_bytes=sum(
            edge.full_duplex_peak_bytes for edge in edges
        ),
        maximum_pair_peak_bytes=max(
            (edge.full_duplex_peak_bytes for edge in edges), default=0
        ),
        overlap_bytes=sum(edge.overlap_bytes for edge in edges),
        host_global_transfer_bytes=matrix.host_global_transfer_bytes,
        makespan_proxy=makespan,
        makespan_proxy_unit=unit,
        worst_rank_state_bytes=max(
            (
                max(edge.first_required_bytes, edge.second_required_bytes)
                for edge in edges
            ),
            default=0,
        ),
        planned_tolerated_host_failures=1,
        static_failure_sets_covered=((0,), (1,)),
    )


def fixed_peer_grouping(matrix: PeerDeltaMatrix) -> RankGrouping:
    return RankGrouping(
        "fixed",
        tuple(zip(matrix.first_ranks, matrix.second_ranks)),
    )


def random_peer_grouping(
    matrix: PeerDeltaMatrix, seed: int = 20260901
) -> RankGrouping:
    second = list(matrix.second_ranks)
    random.Random(seed).shuffle(second)
    return RankGrouping(
        "random",
        tuple(zip(matrix.first_ranks, second)),
        seed=seed,
    )


@dataclass(frozen=True)
class _PartialMatching:
    second_ranks: tuple[int, ...]
    first_to_second_bytes: int
    second_to_first_bytes: int
    maximum_pair_peak_bytes: int
    maximum_pair_proxy: float


def _cost_vector(candidate: _PartialMatching) -> tuple[int, int, int, float]:
    return (
        candidate.first_to_second_bytes,
        candidate.second_to_first_bytes,
        candidate.maximum_pair_peak_bytes,
        candidate.maximum_pair_proxy,
    )


def _dominates(left: _PartialMatching, right: _PartialMatching) -> bool:
    left_values = _cost_vector(left)
    right_values = _cost_vector(right)
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def _exact_candidates(matrix: PeerDeltaMatrix) -> tuple[_PartialMatching, ...]:
    """Return the monotone Pareto frontier from an exact bitmask DP."""
    first = matrix.first_ranks
    second = matrix.second_ranks
    if len(first) > 12:
        raise ValueError("exact peer matching is limited to 12 ranks per host")

    @lru_cache(maxsize=None)
    def solve(index: int, used_mask: int) -> tuple[_PartialMatching, ...]:
        if index == len(first):
            return (_PartialMatching((), 0, 0, 0, 0.0),)
        candidates = []
        for second_index, second_rank in enumerate(second):
            bit = 1 << second_index
            if used_mask & bit:
                continue
            edge = matrix.edge(first[index], second_rank)
            for tail in solve(index + 1, used_mask | bit):
                candidates.append(
                    _PartialMatching(
                        second_ranks=(second_rank,) + tail.second_ranks,
                        first_to_second_bytes=(
                            edge.first_to_second_bytes
                            + tail.first_to_second_bytes
                        ),
                        second_to_first_bytes=(
                            edge.second_to_first_bytes
                            + tail.second_to_first_bytes
                        ),
                        maximum_pair_peak_bytes=max(
                            edge.full_duplex_peak_bytes,
                            tail.maximum_pair_peak_bytes,
                        ),
                        maximum_pair_proxy=max(
                            _edge_proxy(matrix, edge),
                            tail.maximum_pair_proxy,
                        ),
                    )
                )
        frontier = []
        for candidate in sorted(candidates, key=lambda item: item.second_ranks):
            if any(
                _dominates(existing, candidate)
                or _cost_vector(existing) == _cost_vector(candidate)
                for existing in frontier
            ):
                continue
            frontier = [
                existing
                for existing in frontier
                if not _dominates(candidate, existing)
            ]
            if not any(
                existing.second_ranks == candidate.second_ranks
                for existing in frontier
            ):
                frontier.append(candidate)
        return tuple(frontier)

    return solve(0, 0)


def exact_peer_grouping(
    matrix: PeerDeltaMatrix, *, policy: str = "profile_exact"
) -> RankGrouping:
    if policy not in {"redundancy_max", "profile_exact"}:
        raise ValueError("exact policy must be redundancy_max or profile_exact")
    candidates = []
    for candidate in _exact_candidates(matrix):
        grouping = RankGrouping(
            policy,
            tuple(zip(matrix.first_ranks, candidate.second_ranks)),
        )
        evaluation = evaluate_peer_grouping(matrix, grouping)
        if policy == "redundancy_max":
            score = (
                evaluation.pair_local_transfer_bytes,
                evaluation.maximum_pair_peak_bytes,
                grouping.groups,
            )
        else:
            score = (
                evaluation.makespan_proxy,
                evaluation.pair_local_transfer_bytes,
                evaluation.maximum_pair_peak_bytes,
                grouping.groups,
            )
        candidates.append((score, grouping))
    if not candidates:
        raise RuntimeError("exact peer solver produced no feasible matching")
    return min(candidates, key=lambda item: item[0])[1]


def peer_grouping_for_policy(
    matrix: PeerDeltaMatrix,
    policy: str,
    *,
    seed: int = 20260901,
) -> RankGrouping:
    if policy == "fixed":
        return fixed_peer_grouping(matrix)
    if policy == "random":
        return random_peer_grouping(matrix, seed)
    if policy in {"redundancy_max", "profile_exact"}:
        return exact_peer_grouping(matrix, policy=policy)
    raise ValueError(
        "peer policy must be fixed, random, redundancy_max, or profile_exact"
    )
