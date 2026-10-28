"""Offline checkpoint-grouping opportunity model for asymmetric plans."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence

from .reduced import ReducedCheckpoint


IndexRanges = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class PlanRequirement:
    name: str
    rank: int
    index_ranges: IndexRanges

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rank": self.rank,
            "index_ranges": [list(item) for item in self.index_ranges],
        }


@dataclass(frozen=True)
class PlannedCheckpointObject:
    object_id: str
    role: str
    dtype: str
    shape: tuple[int, ...]
    logical_bytes: int
    checksum: str
    holders: tuple[int, ...]
    requirements: tuple[PlanRequirement, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "role": self.role,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "logical_bytes": self.logical_bytes,
            "checksum": self.checksum,
            "holders": list(self.holders),
            "requirements": [item.to_dict() for item in self.requirements],
        }


@dataclass(frozen=True)
class AsymmetricCheckpointPlan:
    name: str
    producer: str
    world_size: int
    ranks_per_host: int
    objects: tuple[PlannedCheckpointObject, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.world_size < 2 or self.ranks_per_host < 1:
            raise ValueError("a plan requires at least two ranks and one rank per host")
        if self.world_size % self.ranks_per_host:
            raise ValueError("world_size must be divisible by ranks_per_host")
        if not self.objects:
            raise ValueError("a checkpoint plan requires at least one object")
        object_ids = [item.object_id for item in self.objects]
        if len(set(object_ids)) != len(object_ids):
            raise ValueError("checkpoint object IDs must be unique")
        for item in self.objects:
            if item.logical_bytes < 0:
                raise ValueError(f"negative object size for {item.object_id}")
            holders = tuple(sorted(set(item.holders)))
            if holders != item.holders or not holders:
                raise ValueError(f"invalid holders for {item.object_id}")
            if any(rank < 0 or rank >= self.world_size for rank in holders):
                raise ValueError(f"holder outside plan world for {item.object_id}")
            requirement_ranks = tuple(
                sorted(set(requirement.rank for requirement in item.requirements))
            )
            if requirement_ranks != holders:
                raise ValueError(
                    f"requirements and holders disagree for {item.object_id}"
                )

    @property
    def host_count(self) -> int:
        return self.world_size // self.ranks_per_host

    def host_for_rank(self, rank: int) -> int:
        if rank < 0 or rank >= self.world_size:
            raise ValueError(f"rank {rank} is outside the plan")
        return rank // self.ranks_per_host

    def holder_hosts(self, item: PlannedCheckpointObject) -> tuple[int, ...]:
        return tuple(sorted({self.host_for_rank(rank) for rank in item.holders}))

    def underprotected_objects(
        self, replica_count: int = 2
    ) -> tuple[PlannedCheckpointObject, ...]:
        if replica_count < 1 or replica_count > self.host_count:
            raise ValueError("replica_count is incompatible with the host topology")
        return tuple(
            item
            for item in self.objects
            if len(self.holder_hosts(item)) < replica_count
        )

    def canonical_source_payloads(self, replica_count: int = 2) -> tuple[int, ...]:
        payloads = [0] * self.world_size
        for item in self.underprotected_objects(replica_count):
            if len(self.holder_hosts(item)) != replica_count - 1:
                raise ValueError(
                    "the pairwise evaluator only supports one missing failure-domain replica"
                )
            payloads[min(item.holders)] += item.logical_bytes
        return tuple(payloads)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": self.name,
            "producer": self.producer,
            "world_size": self.world_size,
            "ranks_per_host": self.ranks_per_host,
            "host_count": self.host_count,
            "metadata": dict(self.metadata),
            "objects": [item.to_dict() for item in self.objects],
        }

    @property
    def plan_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def asymmetric_plan_from_reduced(
    reduced: ReducedCheckpoint,
    *,
    name: str,
    ranks_per_host: int,
    metadata: Mapping[str, Any] | None = None,
) -> AsymmetricCheckpointPlan:
    """Adapt one validated detector-built bundle to the pairing evaluator."""
    reduced.validate()
    world_size = len(reduced.templates)
    if set(reduced.templates) != set(range(world_size)):
        raise ValueError("reduced checkpoint ranks must be contiguous from zero")
    return AsymmetricCheckpointPlan(
        name=name,
        producer="Merak RedundancyChecker",
        world_size=world_size,
        ranks_per_host=ranks_per_host,
        objects=tuple(
            PlannedCheckpointObject(
                object_id=item.value_id,
                role=item.role,
                dtype=item.dtype,
                shape=item.shape,
                logical_bytes=item.logical_bytes,
                checksum=item.checksum,
                holders=item.source_ranks,
                requirements=tuple(
                    PlanRequirement(
                        name=requirement.path,
                        rank=requirement.rank,
                        index_ranges=((0, item.logical_bytes),),
                    )
                    for requirement in item.requirements
                ),
            )
            for item in reduced.objects
        ),
        metadata=dict(metadata or {}),
    )


@dataclass(frozen=True)
class RankGrouping:
    policy: str
    groups: tuple[tuple[int, int], ...]
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "policy": self.policy,
            "groups": [list(group) for group in self.groups],
        }
        if self.seed is not None:
            value["seed"] = self.seed
        value["plan_hash"] = self.plan_hash
        return value

    @property
    def plan_hash(self) -> str:
        value = {
            "policy": self.policy,
            "groups": [list(group) for group in self.groups],
            "seed": self.seed,
        }
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ProtectionTransfer:
    object_id: str
    source_rank: int
    source_host: int
    target_rank: int
    target_host: int
    logical_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class GroupingEvaluation:
    grouping: RankGrouping
    replica_count: int
    tolerated_host_failures: int
    reduced_logical_bytes: int
    initially_underprotected_bytes: int
    cross_domain_transfer_bytes: int
    global_full_duplex_lower_bound_bytes: int
    aggregate_pair_peak_bytes: int
    underprotected_bytes_after: int
    source_payload_bytes: tuple[int, ...]
    transfers: tuple[ProtectionTransfer, ...]

    def score(self) -> tuple[int, int, tuple[tuple[int, int], ...]]:
        return (
            self.global_full_duplex_lower_bound_bytes,
            self.aggregate_pair_peak_bytes,
            self.grouping.groups,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "grouping": self.grouping.to_dict(),
            "replica_count": self.replica_count,
            "tolerated_host_failures": self.tolerated_host_failures,
            "reduced_logical_bytes": self.reduced_logical_bytes,
            "initially_underprotected_bytes": self.initially_underprotected_bytes,
            "cross_domain_transfer_bytes": self.cross_domain_transfer_bytes,
            "global_full_duplex_lower_bound_bytes": (
                self.global_full_duplex_lower_bound_bytes
            ),
            "aggregate_pair_peak_bytes": self.aggregate_pair_peak_bytes,
            "underprotected_bytes_after": self.underprotected_bytes_after,
            "source_payload_bytes": list(self.source_payload_bytes),
            "transfers": [item.to_dict() for item in self.transfers],
        }


def _canonical_groups(groups: Iterable[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    pairs = []
    for group in groups:
        if len(group) != 2:
            raise ValueError("the opportunity evaluator requires rank pairs")
        first, second = sorted((int(group[0]), int(group[1])))
        pairs.append((first, second))
    return tuple(sorted(pairs))


def _validate_grouping(plan: AsymmetricCheckpointPlan, grouping: RankGrouping) -> None:
    flattened = [rank for group in grouping.groups for rank in group]
    if sorted(flattened) != list(range(plan.world_size)):
        raise ValueError("grouping must contain every rank exactly once")
    for first, second in grouping.groups:
        if plan.host_for_rank(first) == plan.host_for_rank(second):
            raise ValueError("checkpoint peers must belong to different hosts")


def evaluate_grouping(
    plan: AsymmetricCheckpointPlan,
    grouping: RankGrouping,
    *,
    replica_count: int = 2,
    tolerated_host_failures: int = 1,
) -> GroupingEvaluation:
    if replica_count != tolerated_host_failures + 1:
        raise ValueError("replica_count must equal tolerated_host_failures + 1")
    if replica_count != 2 or tolerated_host_failures != 1:
        raise ValueError("the pairwise opportunity evaluator supports only F_host=1")
    _validate_grouping(plan, grouping)

    peer = {}
    for first, second in grouping.groups:
        peer[first] = second
        peer[second] = first

    transfers = []
    protected_hosts: dict[str, set[int]] = {}
    for item in plan.objects:
        hosts = set(plan.holder_hosts(item))
        protected_hosts[item.object_id] = hosts
        if len(hosts) >= replica_count:
            continue
        if len(hosts) != replica_count - 1:
            raise ValueError(
                f"object {item.object_id} needs more than one pairwise transfer"
            )
        source_rank = min(item.holders)
        target_rank = peer[source_rank]
        source_host = plan.host_for_rank(source_rank)
        target_host = plan.host_for_rank(target_rank)
        if target_host in hosts:
            raise ValueError(
                f"grouping leaves object {item.object_id} in one failure domain"
            )
        transfers.append(
            ProtectionTransfer(
                object_id=item.object_id,
                source_rank=source_rank,
                source_host=source_host,
                target_rank=target_rank,
                target_host=target_host,
                logical_bytes=item.logical_bytes,
            )
        )
        hosts.add(target_host)

    for failed_count in range(1, tolerated_host_failures + 1):
        for failed in combinations(range(plan.host_count), failed_count):
            failed_hosts = set(failed)
            missing = [
                object_id
                for object_id, hosts in protected_hosts.items()
                if not hosts - failed_hosts
            ]
            if missing:
                raise ValueError(
                    f"host failure set {failed} loses checkpoint objects {missing}"
                )

    source_payloads = [0] * plan.world_size
    directional = {}
    for item in transfers:
        source_payloads[item.source_rank] += item.logical_bytes
        key = (item.source_rank, item.target_rank)
        directional[key] = directional.get(key, 0) + item.logical_bytes

    pair_peaks = []
    for first, second in grouping.groups:
        pair_peaks.append(
            max(
                directional.get((first, second), 0),
                directional.get((second, first), 0),
            )
        )
    underprotected = plan.underprotected_objects(replica_count)
    return GroupingEvaluation(
        grouping=grouping,
        replica_count=replica_count,
        tolerated_host_failures=tolerated_host_failures,
        reduced_logical_bytes=sum(item.logical_bytes for item in plan.objects),
        initially_underprotected_bytes=sum(
            item.logical_bytes for item in underprotected
        ),
        cross_domain_transfer_bytes=sum(item.logical_bytes for item in transfers),
        global_full_duplex_lower_bound_bytes=max(pair_peaks, default=0),
        aggregate_pair_peak_bytes=sum(pair_peaks),
        underprotected_bytes_after=0,
        source_payload_bytes=tuple(source_payloads),
        transfers=tuple(transfers),
    )


def _host_matchings(hosts: tuple[int, ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
    if not hosts:
        return ((),)
    if len(hosts) % 2:
        raise ValueError("host count must be even for pairwise grouping")
    first = hosts[0]
    results = []
    for index in range(1, len(hosts)):
        second = hosts[index]
        remaining = hosts[1:index] + hosts[index + 1 :]
        for tail in _host_matchings(remaining):
            results.append(((first, second),) + tail)
    return tuple(results)


def _host_ranks(plan: AsymmetricCheckpointPlan, host: int) -> tuple[int, ...]:
    start = host * plan.ranks_per_host
    return tuple(range(start, start + plan.ranks_per_host))


def fixed_grouping(plan: AsymmetricCheckpointPlan) -> RankGrouping:
    if plan.host_count % 2:
        raise ValueError("fixed pairwise grouping requires an even host count")
    groups = []
    for first_host in range(0, plan.host_count, 2):
        second_host = first_host + 1
        groups.extend(
            zip(_host_ranks(plan, first_host), _host_ranks(plan, second_host))
        )
    return RankGrouping("fixed", _canonical_groups(groups))


def random_grouping(plan: AsymmetricCheckpointPlan, seed: int) -> RankGrouping:
    if plan.host_count % 2:
        raise ValueError("random pairwise grouping requires an even host count")
    generator = random.Random(seed)
    hosts = list(range(plan.host_count))
    generator.shuffle(hosts)
    groups = []
    for index in range(0, len(hosts), 2):
        first = list(_host_ranks(plan, hosts[index]))
        second = list(_host_ranks(plan, hosts[index + 1]))
        generator.shuffle(second)
        groups.extend(zip(first, second))
    return RankGrouping("random", _canonical_groups(groups), seed=seed)


def _pair_cost(payloads: Sequence[int], first: int, second: int) -> int:
    return max(int(payloads[first]), int(payloads[second]))


def _exact_host_pairing(
    payloads: Sequence[int], first: Sequence[int], second: Sequence[int]
) -> tuple[tuple[int, int], ...]:
    if len(first) != len(second) or not first:
        raise ValueError("exact host pairing requires equal non-empty rank sets")
    if len(first) > 16:
        raise ValueError("exact host pairing is limited to 16 ranks per host")

    @lru_cache(maxsize=None)
    def solve(index: int, used_mask: int) -> tuple[int, int, tuple[int, ...]]:
        if index == len(first):
            return 0, 0, ()
        best = None
        for right_index, right in enumerate(second):
            bit = 1 << right_index
            if used_mask & bit:
                continue
            tail_peak, tail_sum, tail_permutation = solve(index + 1, used_mask | bit)
            pair_cost = _pair_cost(payloads, first[index], right)
            candidate = (
                max(pair_cost, tail_peak),
                pair_cost + tail_sum,
                (right_index,) + tail_permutation,
            )
            if best is None or candidate < best:
                best = candidate
        if best is None:
            raise RuntimeError("exact pairing reached an invalid state")
        return best

    _, _, permutation = solve(0, 0)
    return tuple(
        (left, second[right_index])
        for left, right_index in zip(first, permutation)
    )


def _optimized_grouping(
    plan: AsymmetricCheckpointPlan, policy: str
) -> RankGrouping:
    payloads = plan.canonical_source_payloads()
    candidates = []
    for host_matching in _host_matchings(tuple(range(plan.host_count))):
        groups = []
        for first_host, second_host in host_matching:
            first = _host_ranks(plan, first_host)
            second = _host_ranks(plan, second_host)
            if policy == "exact_oracle":
                groups.extend(_exact_host_pairing(payloads, first, second))
            else:
                raise ValueError(f"unsupported optimized grouping policy: {policy}")
        grouping = RankGrouping(policy, _canonical_groups(groups))
        evaluation = evaluate_grouping(plan, grouping)
        candidates.append((evaluation.score(), grouping))
    return min(candidates, key=lambda item: item[0])[1]


def exact_oracle_grouping(plan: AsymmetricCheckpointPlan) -> RankGrouping:
    """Solve the host-pair and local-rank assignment space exactly.

    The primary objective is the global unit-bandwidth full-duplex lower bound.
    The sum of pair peaks is a deterministic secondary objective, not a latency
    prediction for a shared physical fabric.
    """

    return _optimized_grouping(plan, "exact_oracle")


def grouping_for_policy(
    plan: AsymmetricCheckpointPlan, policy: str, *, seed: int = 20260831
) -> RankGrouping:
    if policy == "fixed":
        return fixed_grouping(plan)
    if policy == "random":
        return random_grouping(plan, seed)
    if policy == "exact":
        return exact_oracle_grouping(plan)
    raise ValueError("policy must be fixed, random, or exact")
