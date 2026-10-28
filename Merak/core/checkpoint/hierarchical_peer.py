"""Offline host/rank grouping for AdaCheck-derived checkpoint catalogs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .matched import reduced_catalog_hash
from .opportunity import RankGrouping
from .peer_delta import (
    build_peer_delta_matrix_for_hosts_from_catalog,
    evaluate_peer_grouping,
    exact_peer_grouping,
    fixed_peer_grouping,
)
from .reduced import ReducedCheckpoint


def catalog_rows(reduced: ReducedCheckpoint) -> tuple[dict[str, Any], ...]:
    reduced.validate()
    return tuple(
        {
            "value_id": item.value_id,
            "logical_bytes": item.logical_bytes,
            "source_ranks": list(item.source_ranks),
            "requirements": [
                {"rank": requirement.rank} for requirement in item.requirements
            ],
        }
        for item in reduced.objects
    )


def host_matchings(host_count: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Enumerate perfect host matchings in deterministic order."""
    if host_count < 2 or host_count % 2:
        raise ValueError("host_count must be positive and even")

    def visit(hosts: tuple[int, ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
        if not hosts:
            return ((),)
        first = hosts[0]
        result = []
        for index in range(1, len(hosts)):
            second = hosts[index]
            remaining = hosts[1:index] + hosts[index + 1 :]
            for tail in visit(remaining):
                result.append(((first, second),) + tail)
        return tuple(result)

    return visit(tuple(range(host_count)))


@dataclass(frozen=True)
class HostPairEvaluation:
    first_host: int
    second_host: int
    grouping: RankGrouping
    peer_local_transfer_bytes: int
    first_to_second_bytes: int
    second_to_first_bytes: int
    host_pooled_transfer_bytes: int
    host_pooled_max_publisher_bytes: int

    @property
    def maximum_directional_bytes(self) -> int:
        return max(self.first_to_second_bytes, self.second_to_first_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hosts": [self.first_host, self.second_host],
            "grouping": self.grouping.to_dict(),
            "peer_local_transfer_bytes": self.peer_local_transfer_bytes,
            "first_to_second_bytes": self.first_to_second_bytes,
            "second_to_first_bytes": self.second_to_first_bytes,
            "maximum_directional_bytes": self.maximum_directional_bytes,
            "host_pooled_transfer_bytes": self.host_pooled_transfer_bytes,
            "host_pooled_max_publisher_bytes": (
                self.host_pooled_max_publisher_bytes
            ),
        }


@dataclass(frozen=True)
class HierarchicalEvaluation:
    policy: str
    catalog_hash: str
    world_size: int
    ranks_per_host: int
    grouping: RankGrouping
    host_pairs: tuple[tuple[int, int], ...]
    pair_evaluations: tuple[HostPairEvaluation, ...]
    peer_local_transfer_bytes: int
    host_pooled_transfer_bytes: int
    host_pooled_max_publisher_bytes: int
    maximum_host_pair_bytes: int
    maximum_directional_bytes: int

    @property
    def grouping_hash(self) -> str:
        payload = json.dumps(
            {
                "catalog_hash": self.catalog_hash,
                "world_size": self.world_size,
                "ranks_per_host": self.ranks_per_host,
                "host_pairs": self.host_pairs,
                "groups": self.grouping.groups,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "policy": self.policy,
            "catalog_hash": self.catalog_hash,
            "world_size": self.world_size,
            "ranks_per_host": self.ranks_per_host,
            "host_pairs": [list(item) for item in self.host_pairs],
            "grouping": self.grouping.to_dict(),
            "peer_local_transfer_bytes": self.peer_local_transfer_bytes,
            "host_pooled_transfer_bytes": self.host_pooled_transfer_bytes,
            "host_pooled_max_publisher_bytes": (
                self.host_pooled_max_publisher_bytes
            ),
            "maximum_host_pair_bytes": self.maximum_host_pair_bytes,
            "maximum_directional_bytes": self.maximum_directional_bytes,
            "pair_evaluations": [item.to_dict() for item in self.pair_evaluations],
            "grouping_hash": self.grouping_hash,
        }


def _host_pooled_metrics(
    objects: Sequence[Mapping[str, Any]],
    *,
    first_host: int,
    second_host: int,
    ranks_per_host: int,
    balance_publishers: bool,
) -> tuple[int, int]:
    loads: dict[int, int] = {
        rank: 0
        for host in (first_host, second_host)
        for rank in range(host * ranks_per_host, (host + 1) * ranks_per_host)
    }
    transfer_bytes = 0
    for item in sorted(
        objects,
        key=lambda value: (-int(value["logical_bytes"]), str(value["value_id"])),
    ):
        holders = {int(value["rank"]) for value in item["requirements"]}
        holder_hosts = {rank // ranks_per_host for rank in holders}
        for source_host, target_host in (
            (first_host, second_host),
            (second_host, first_host),
        ):
            if source_host not in holder_hosts or target_host in holder_hosts:
                continue
            eligible = sorted(
                rank for rank in holders if rank // ranks_per_host == source_host
            )
            publisher = (
                min(eligible, key=lambda rank: (loads[rank], rank))
                if balance_publishers
                else eligible[0]
            )
            logical_bytes = int(item["logical_bytes"])
            loads[publisher] += logical_bytes
            transfer_bytes += logical_bytes
    return transfer_bytes, max(loads.values(), default=0)


def _pair_evaluation(
    objects: Sequence[Mapping[str, Any]],
    *,
    world_size: int,
    ranks_per_host: int,
    catalog_hash: str,
    first_host: int,
    second_host: int,
    optimize_ranks: bool,
    balance_publishers: bool,
) -> HostPairEvaluation:
    matrix = build_peer_delta_matrix_for_hosts_from_catalog(
        objects,
        world_size=world_size,
        ranks_per_host=ranks_per_host,
        first_host=first_host,
        second_host=second_host,
        catalog_hash=catalog_hash,
    )
    local = (
        exact_peer_grouping(matrix, policy="redundancy_max")
        if optimize_ranks
        else fixed_peer_grouping(matrix)
    )
    measured = evaluate_peer_grouping(matrix, local)
    groups = tuple(
        (
            first_host * ranks_per_host + first,
            second_host * ranks_per_host + second - ranks_per_host,
        )
        for first, second in local.groups
    )
    rank_grouping = RankGrouping(
        "rank_opt" if optimize_ranks else "local_rank", groups
    )
    host_pooled, max_publisher = _host_pooled_metrics(
        objects,
        first_host=first_host,
        second_host=second_host,
        ranks_per_host=ranks_per_host,
        balance_publishers=balance_publishers,
    )
    return HostPairEvaluation(
        first_host=first_host,
        second_host=second_host,
        grouping=rank_grouping,
        peer_local_transfer_bytes=measured.pair_local_transfer_bytes,
        first_to_second_bytes=measured.first_to_second_bytes,
        second_to_first_bytes=measured.second_to_first_bytes,
        host_pooled_transfer_bytes=host_pooled,
        host_pooled_max_publisher_bytes=max_publisher,
    )


def evaluate_host_matching(
    objects: Sequence[Mapping[str, Any]],
    *,
    world_size: int,
    ranks_per_host: int,
    catalog_hash: str,
    host_pairs: Sequence[tuple[int, int]],
    optimize_ranks: bool,
    policy: str,
    balance_publishers: bool = False,
) -> HierarchicalEvaluation:
    host_count = world_size // ranks_per_host
    flattened = sorted(host for pair in host_pairs for host in pair)
    if world_size % ranks_per_host or flattened != list(range(host_count)):
        raise ValueError("host pairs must cover every host exactly once")
    pairs = tuple(tuple(sorted(pair)) for pair in host_pairs)
    pair_evaluations = tuple(
        _pair_evaluation(
            objects,
            world_size=world_size,
            ranks_per_host=ranks_per_host,
            catalog_hash=catalog_hash,
            first_host=first,
            second_host=second,
            optimize_ranks=optimize_ranks,
            balance_publishers=balance_publishers,
        )
        for first, second in pairs
    )
    groups = tuple(
        group
        for pair_evaluation in pair_evaluations
        for group in pair_evaluation.grouping.groups
    )
    grouping = RankGrouping(policy, groups)
    peer_bytes = sum(item.peer_local_transfer_bytes for item in pair_evaluations)
    pooled_bytes = sum(item.host_pooled_transfer_bytes for item in pair_evaluations)
    return HierarchicalEvaluation(
        policy=policy,
        catalog_hash=catalog_hash,
        world_size=world_size,
        ranks_per_host=ranks_per_host,
        grouping=grouping,
        host_pairs=pairs,
        pair_evaluations=pair_evaluations,
        peer_local_transfer_bytes=peer_bytes,
        host_pooled_transfer_bytes=pooled_bytes,
        host_pooled_max_publisher_bytes=max(
            (
                item.host_pooled_max_publisher_bytes
                for item in pair_evaluations
            ),
            default=0,
        ),
        maximum_host_pair_bytes=max(
            (item.peer_local_transfer_bytes for item in pair_evaluations), default=0
        ),
        maximum_directional_bytes=max(
            (item.maximum_directional_bytes for item in pair_evaluations), default=0
        ),
    )


def _best(
    candidates: Sequence[HierarchicalEvaluation], *, metric: str
) -> HierarchicalEvaluation:
    if metric not in {"peer", "host"}:
        raise ValueError("metric must be peer or host")
    return min(
        candidates,
        key=lambda item: (
            item.peer_local_transfer_bytes
            if metric == "peer"
            else item.host_pooled_transfer_bytes,
            (
                item.maximum_host_pair_bytes
                if metric == "peer"
                else item.host_pooled_max_publisher_bytes
            ),
            item.maximum_directional_bytes,
            item.grouping.groups,
        ),
    )


def build_hierarchical_study_from_catalog(
    objects: Sequence[Mapping[str, Any]],
    *,
    world_size: int,
    ranks_per_host: int,
    catalog_hash: str,
) -> dict[str, HierarchicalEvaluation]:
    if world_size % ranks_per_host:
        raise ValueError("world_size must be divisible by ranks_per_host")
    matchings = host_matchings(world_size // ranks_per_host)
    fixed_hosts = matchings[0]
    fixed = evaluate_host_matching(
        objects,
        world_size=world_size,
        ranks_per_host=ranks_per_host,
        catalog_hash=catalog_hash,
        host_pairs=fixed_hosts,
        optimize_ranks=False,
        balance_publishers=False,
        policy="sequential_fixed",
    )
    rank_only = evaluate_host_matching(
        objects,
        world_size=world_size,
        ranks_per_host=ranks_per_host,
        catalog_hash=catalog_hash,
        host_pairs=fixed_hosts,
        optimize_ranks=True,
        balance_publishers=False,
        policy="rank_only",
    )
    host_candidates = [
        evaluate_host_matching(
            objects,
            world_size=world_size,
            ranks_per_host=ranks_per_host,
            catalog_hash=catalog_hash,
            host_pairs=matching,
            optimize_ranks=False,
            balance_publishers=False,
            policy="host_only",
        )
        for matching in matchings
    ]
    joint_candidates = [
        evaluate_host_matching(
            objects,
            world_size=world_size,
            ranks_per_host=ranks_per_host,
            catalog_hash=catalog_hash,
            host_pairs=matching,
            optimize_ranks=True,
            balance_publishers=False,
            policy="hierarchical_joint",
        )
        for matching in matchings
    ]
    host_only = _best(host_candidates, metric="peer")
    joint = _best(joint_candidates, metric="peer")
    host_pooled = replace(
        _best(host_candidates, metric="host"), policy="host_pooled_opt"
    )
    host_pooled_fixed = replace(fixed, policy="host_pooled_fixed")
    publisher_balanced = evaluate_host_matching(
        objects,
        world_size=world_size,
        ranks_per_host=ranks_per_host,
        catalog_hash=catalog_hash,
        host_pairs=fixed_hosts,
        optimize_ranks=False,
        balance_publishers=True,
        policy="host_pooled_publisher",
    )
    host_pooled_joint = _best(
        [
            evaluate_host_matching(
                objects,
                world_size=world_size,
                ranks_per_host=ranks_per_host,
                catalog_hash=catalog_hash,
                host_pairs=matching,
                optimize_ranks=False,
                balance_publishers=True,
                policy="host_pooled_joint",
            )
            for matching in matchings
        ],
        metric="host",
    )
    result = {
        "sequential_fixed": fixed,
        "host_only": host_only,
        "rank_only": rank_only,
        "hierarchical_joint": joint,
        "host_pooled_fixed": host_pooled_fixed,
        "host_pooled_publisher": publisher_balanced,
        "host_pooled_opt": host_pooled,
        "host_pooled_joint": host_pooled_joint,
    }
    if not (
        joint.peer_local_transfer_bytes
        <= min(
            host_only.peer_local_transfer_bytes,
            rank_only.peer_local_transfer_bytes,
        )
        <= fixed.peer_local_transfer_bytes
    ):
        raise AssertionError("hierarchical peer-local opportunity is not monotone")
    if host_pooled.host_pooled_transfer_bytes > fixed.host_pooled_transfer_bytes:
        raise AssertionError("host-pooled opportunity is not monotone")
    if host_pooled_joint.host_pooled_transfer_bytes > host_pooled.host_pooled_transfer_bytes:
        raise AssertionError("host-pooled joint opportunity is not monotone")
    return result


def build_hierarchical_study(
    reduced: ReducedCheckpoint, *, ranks_per_host: int
) -> dict[str, HierarchicalEvaluation]:
    return build_hierarchical_study_from_catalog(
        catalog_rows(reduced),
        world_size=len(reduced.templates),
        ranks_per_host=ranks_per_host,
        catalog_hash=reduced_catalog_hash(reduced),
    )
