"""Plan host protection for checksum-confirmed semantic gradient objects."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import torch

from .catalog import tensor_checksum


Arm = Literal["adacheck_fixed", "host_global_fixed", "m1"]


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _volume(shape: Sequence[int]) -> int:
    result = 1
    for size in shape:
        result *= int(size)
    return result


def _dtype(name: str) -> torch.dtype:
    value = getattr(torch, name.removeprefix("torch."), None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported gradient dtype: {name}")
    return value


@dataclass(frozen=True)
class RankBinding:
    rank: int
    pipeline_rank: int
    tensor_rank: int
    data_rank: int
    host: int

    def __post_init__(self) -> None:
        if min(
            self.rank,
            self.pipeline_rank,
            self.tensor_rank,
            self.data_rank,
            self.host,
        ) < 0:
            raise ValueError("rank binding values must be non-negative")


@dataclass(frozen=True)
class GradientShard:
    """One rank's encoded-gradient slice metadata; dtype is the encoded dtype."""

    rank: int
    fqn: str
    detector_class_id: str
    global_shape: tuple[int, ...]
    offsets: tuple[int, ...]
    lengths: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if (
            self.rank < 0
            or not self.fqn
            or not self.detector_class_id
            or not self.global_shape
        ):
            raise ValueError("gradient shard identity is incomplete")
        if not (
            len(self.global_shape) == len(self.offsets) == len(self.lengths)
        ):
            raise ValueError("gradient shard dimensions disagree")
        if any(size <= 0 for size in self.global_shape + self.lengths):
            raise ValueError("gradient shard shapes must be positive")
        if any(offset < 0 for offset in self.offsets) or any(
            offset + length > size
            for offset, length, size in zip(
                self.offsets, self.lengths, self.global_shape
            )
        ):
            raise ValueError("gradient shard lies outside its global shape")
        _dtype(self.dtype)


@dataclass(frozen=True)
class ActiveGradientObject:
    object_id: str
    fqn: str
    detector_class_id: str
    global_shape: tuple[int, ...]
    offsets: tuple[int, ...]
    lengths: tuple[int, ...]
    dtype: str
    pipeline_rank: int
    tensor_ranks: tuple[int, ...]
    holders: tuple[int, ...]
    logical_bytes: int
    direct_fallback_ranks: tuple[int, ...]
    fallback_action: str = "direct_fallback"


@dataclass(frozen=True)
class GradientPlacement:
    object_id: str
    publisher_rank: int
    source_host: int
    target_host: int
    group: str
    logical_bytes: int

    @property
    def remote(self) -> bool:
        return self.source_host != self.target_host


@dataclass(frozen=True)
class ActiveGradientPlan:
    arm: Arm
    detector_catalog_hash: str
    semantic_plan_hash: str
    canonical_hash: str
    objects: tuple[ActiveGradientObject, ...]
    rank_bindings: tuple[RankBinding, ...]
    placements: tuple[GradientPlacement, ...]
    required_objects_by_host: tuple[tuple[int, tuple[str, ...]], ...]
    failure_sets: tuple[tuple[int, ...], ...]
    total_remote_logical_bytes: int
    maximum_remote_publisher_bytes: int
    all_publication_publisher_bytes: tuple[tuple[int, int], ...]
    total_all_publication_logical_bytes: int
    maximum_all_publication_publisher_bytes: int
    publisher_policy: str
    plan_hash: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "arm": self.arm,
            "detector_catalog_hash": self.detector_catalog_hash,
            "semantic_plan_hash": self.semantic_plan_hash,
            "canonical_hash": self.canonical_hash,
            "publisher_policy": self.publisher_policy,
            "total_remote_logical_bytes": self.total_remote_logical_bytes,
            "maximum_remote_publisher_bytes": self.maximum_remote_publisher_bytes,
            "all_publication_publisher_bytes": [
                [rank, logical_bytes]
                for rank, logical_bytes in self.all_publication_publisher_bytes
            ],
            "total_all_publication_logical_bytes": self.total_all_publication_logical_bytes,
            "maximum_all_publication_publisher_bytes": self.maximum_all_publication_publisher_bytes,
            "required_objects_by_host": [
                [host, list(value_ids)]
                for host, value_ids in self.required_objects_by_host
            ],
            "failure_sets": [list(value) for value in self.failure_sets],
            "plan_hash": self.plan_hash,
        }


@dataclass(frozen=True)
class GradientEncodingAction:
    object_id: str
    version_id: str
    source_rank: int
    source_host: int
    target_host: int
    action: Literal["confirmed_object", "direct_fallback"]
    checksum: str
    logical_bytes: int

    @property
    def remote(self) -> bool:
        return self.source_host != self.target_host


@dataclass(frozen=True)
class ConfirmedGradientEncoding:
    canonical_hash: str
    epoch: int
    actions: tuple[GradientEncodingAction, ...]
    mismatched_object_ids: tuple[str, ...]
    required_versions_by_host: tuple[tuple[int, tuple[str, ...]], ...]
    failure_sets: tuple[tuple[int, ...], ...]
    total_remote_logical_bytes: int
    maximum_remote_publisher_bytes: int
    all_publication_publisher_bytes: tuple[tuple[int, int], ...]
    total_all_publication_logical_bytes: int
    maximum_all_publication_publisher_bytes: int
    encoding_hash: str

    @property
    def all_equivalences_confirmed(self) -> bool:
        return not self.mismatched_object_ids


def _canonical_objects(
    shards: Sequence[GradientShard], bindings: Mapping[int, RankBinding]
) -> tuple[ActiveGradientObject, ...]:
    by_fqn: dict[str, tuple[tuple[int, ...], str]] = {}
    by_coordinate: dict[
        tuple[str, int, int],
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], str],
    ] = {}
    candidates: dict[
        tuple[
            str,
            str,
            tuple[int, ...],
            tuple[int, ...],
            tuple[int, ...],
            str,
            int,
        ],
        list[GradientShard],
    ] = {}
    seen = set()
    for shard in shards:
        binding = bindings.get(shard.rank)
        if binding is None:
            raise ValueError(f"gradient shard rank {shard.rank} has no binding")
        identity = (shard.rank, shard.fqn, shard.offsets, shard.lengths)
        if identity in seen:
            raise ValueError("duplicate rank gradient shard")
        seen.add(identity)
        fqn_schema = (shard.global_shape, shard.dtype)
        previous_schema = by_fqn.setdefault(shard.fqn, fqn_schema)
        if previous_schema != fqn_schema:
            raise ValueError("one FQN has conflicting global shape or dtype")
        coordinate = (shard.fqn, binding.pipeline_rank, binding.tensor_rank)
        coordinate_schema = (
            shard.global_shape,
            shard.offsets,
            shard.lengths,
            shard.dtype,
        )
        previous_coordinate = by_coordinate.setdefault(coordinate, coordinate_schema)
        if previous_coordinate != coordinate_schema:
            raise ValueError("one FQN/PP/TP coordinate has conflicting slice schema")
        key = (
            shard.detector_class_id,
            shard.fqn,
            shard.global_shape,
            shard.offsets,
            shard.lengths,
            shard.dtype,
            binding.tensor_rank,
        )
        candidates.setdefault(key, []).append(shard)

    objects = []
    for key, values in sorted(candidates.items()):
        pipeline_ranks = {bindings[item.rank].pipeline_rank for item in values}
        if len(pipeline_ranks) != 1:
            raise ValueError(
                "equal FQN/global slice appears at conflicting pipeline coordinates"
            )
        logical_coordinates = [
            (bindings[item.rank].tensor_rank, bindings[item.rank].data_rank)
            for item in values
        ]
        if len(logical_coordinates) != len(set(logical_coordinates)):
            raise ValueError("gradient equality candidates repeat a TP/DP coordinate")
        (
            detector_class_id,
            fqn,
            global_shape,
            offsets,
            lengths,
            dtype,
            tensor_rank,
        ) = key
        pipeline_rank = pipeline_ranks.pop()
        tensor_ranks = (tensor_rank,)
        holders = tuple(sorted(item.rank for item in values))
        object_id = _digest(
            {
                "fqn": fqn,
                "detector_class_id": detector_class_id,
                "global_shape": global_shape,
                "offsets": offsets,
                "lengths": lengths,
                "dtype": dtype,
                "pipeline_rank": pipeline_rank,
                "tensor_ranks": tensor_ranks,
            }
        )
        logical_bytes = _volume(lengths) * torch.empty(
            (), dtype=_dtype(dtype)
        ).element_size()
        objects.append(
            ActiveGradientObject(
                object_id,
                fqn,
                detector_class_id,
                global_shape,
                offsets,
                lengths,
                dtype,
                pipeline_rank,
                tensor_ranks,
                holders,
                logical_bytes,
                holders,
            )
        )
    return tuple(objects)


def _fixed_pairs(
    pairs: Sequence[Sequence[int]], bindings: Mapping[int, RankBinding]
) -> tuple[tuple[int, int], ...]:
    normalized = tuple(tuple(sorted(int(rank) for rank in pair)) for pair in pairs)
    if any(len(pair) != 2 for pair in normalized):
        raise ValueError("fixed pairs must contain exactly two ranks")
    flattened = [rank for pair in normalized for rank in pair]
    if sorted(flattened) != sorted(bindings) or len(flattened) != len(set(flattened)):
        raise ValueError("fixed pairs must cover every bound rank exactly once")
    if any(bindings[left].host == bindings[right].host for left, right in normalized):
        raise ValueError("fixed pairs must cross hosts")
    return tuple(sorted(normalized))


def build_active_gradient_plan(
    shards: Sequence[GradientShard],
    rank_bindings: Sequence[RankBinding],
    *,
    arm: Arm,
    detector_catalog_hash: str,
    semantic_plan_hash: str,
    fixed_pairs: Sequence[Sequence[int]] = (),
    failure_sets: Sequence[Sequence[int]] | None = None,
) -> ActiveGradientPlan:
    """Build a placement plan; equality remains conditional on runtime checksums."""

    if arm not in {"adacheck_fixed", "host_global_fixed", "m1"}:
        raise ValueError(f"unsupported semantic object arm: {arm}")
    detector_catalog_hash = _sha256(
        detector_catalog_hash, "detector_catalog_hash"
    )
    semantic_plan_hash = _sha256(semantic_plan_hash, "semantic_plan_hash")
    bindings = {item.rank: item for item in rank_bindings}
    if len(bindings) != len(rank_bindings) or not bindings:
        raise ValueError("rank bindings must be non-empty and unique")
    hosts = tuple(sorted({item.host for item in rank_bindings}))
    if hosts != (0, 1):
        raise ValueError("semantic object pre-gate requires hosts 0 and 1")
    failures = tuple(
        tuple(int(host) for host in value)
        for value in (
            failure_sets
            if failure_sets is not None
            else ((), (0,), (1,))
        )
    )
    if failures != ((), (0,), (1,)):
        raise ValueError("semantic object pre-gate requires no loss and either host loss")

    objects = _canonical_objects(shards, bindings)
    if not objects:
        raise ValueError("active gradient plan requires at least one object")
    pairs = _fixed_pairs(fixed_pairs, bindings) if arm == "adacheck_fixed" else ()
    all_publication_loads = {rank: 0 for rank in bindings}
    placements = []
    ordered_objects = sorted(
        objects,
        key=(
            (lambda item: (-item.logical_bytes, item.object_id))
            if arm == "m1"
            else (lambda item: item.object_id)
        ),
    )
    for item in ordered_objects:
        if arm == "adacheck_fixed":
            groups = [pair for pair in pairs if set(pair) & set(item.holders)]
        else:
            groups = [tuple(bindings)]
        for group in groups:
            group_holders = tuple(rank for rank in item.holders if rank in group)
            m1_publisher = (
                min(
                    group_holders,
                    key=lambda rank: (all_publication_loads[rank], rank),
                )
                if arm == "m1"
                else None
            )
            for target_host in hosts:
                local = tuple(
                    rank
                    for rank in group_holders
                    if bindings[rank].host == target_host
                )
                eligible = local or group_holders
                if not eligible:
                    continue
                publisher = min(eligible)
                remote = not local
                if m1_publisher is not None:
                    publisher = m1_publisher
                placements.append(
                    GradientPlacement(
                        item.object_id,
                        publisher,
                        bindings[publisher].host,
                        target_host,
                        (
                            f"pair-{group[0]}-{group[1]}"
                            if arm == "adacheck_fixed"
                            else "host-global"
                        ),
                        item.logical_bytes,
                    )
                )
                if arm == "m1":
                    all_publication_loads[publisher] += item.logical_bytes

    object_ids = tuple(sorted(item.object_id for item in objects))
    required = tuple((host, object_ids) for host in hosts)
    for host, required_ids in required:
        placed = {
            item.object_id for item in placements if item.target_host == host
        }
        if placed != set(required_ids):
            raise ValueError(f"placement leaves host {host} without required objects")
    for failed in failures:
        surviving = {
            object_id
            for host, required_ids in required
            if host not in failed
            for object_id in required_ids
        }
        if surviving != set(object_ids):
            raise ValueError(f"failure set {failed} leaves no complete object catalog")

    canonical_hash = _digest(
        {
            "objects": [
                {
                    "object_id": item.object_id,
                    "fqn": item.fqn,
                    "detector_class_id": item.detector_class_id,
                    "global_shape": item.global_shape,
                    "offsets": item.offsets,
                    "lengths": item.lengths,
                    "dtype": item.dtype,
                    "pipeline_rank": item.pipeline_rank,
                    "tensor_ranks": item.tensor_ranks,
                    "holders": item.holders,
                    "direct_fallback_ranks": item.direct_fallback_ranks,
                }
                for item in objects
            ],
            "detector_catalog_hash": detector_catalog_hash,
            "semantic_plan_hash": semantic_plan_hash,
            "rank_bindings": [
                (
                    item.rank,
                    item.pipeline_rank,
                    item.tensor_rank,
                    item.data_rank,
                    item.host,
                )
                for item in sorted(rank_bindings, key=lambda value: value.rank)
            ],
        }
    )
    remote_total = sum(item.logical_bytes for item in placements if item.remote)
    remote_loads = {
        rank: sum(
            item.logical_bytes
            for item in placements
            if item.remote and item.publisher_rank == rank
        )
        for rank in bindings
    }
    all_publication_loads = {
        rank: sum(
            item.logical_bytes
            for item in placements
            if item.publisher_rank == rank
        )
        for rank in bindings
    }
    policy = {
        "adacheck_fixed": "fixed_pair_first_eligible",
        "host_global_fixed": "host_global_first_eligible",
        "m1": "host_global_single_publisher_all_destinations_least_load",
    }[arm]
    plan_body = {
        "arm": arm,
        "detector_catalog_hash": detector_catalog_hash,
        "semantic_plan_hash": semantic_plan_hash,
        "canonical_hash": canonical_hash,
        "placements": [
            (
                item.object_id,
                item.publisher_rank,
                item.source_host,
                item.target_host,
                item.group,
                item.logical_bytes,
            )
            for item in placements
        ],
        "required": required,
        "failures": failures,
        "publisher_policy": policy,
    }
    return ActiveGradientPlan(
        arm,
        detector_catalog_hash,
        semantic_plan_hash,
        canonical_hash,
        objects,
        tuple(sorted(rank_bindings, key=lambda value: value.rank)),
        tuple(placements),
        required,
        failures,
        remote_total,
        max(remote_loads.values(), default=0),
        tuple(sorted(all_publication_loads.items())),
        sum(all_publication_loads.values()),
        max(all_publication_loads.values(), default=0),
        policy,
        _digest(plan_body),
    )


def assert_fair_host_global_ab(
    fixed: ActiveGradientPlan, m1: ActiveGradientPlan
) -> None:
    if fixed.arm != "host_global_fixed" or m1.arm != "m1":
        raise ValueError("fair host-global A/B requires Fixed and M1 arms")
    if (
        fixed.detector_catalog_hash != m1.detector_catalog_hash
        or fixed.semantic_plan_hash != m1.semantic_plan_hash
        or fixed.canonical_hash != m1.canonical_hash
        or fixed.required_objects_by_host != m1.required_objects_by_host
        or fixed.failure_sets != m1.failure_sets
        or fixed.total_remote_logical_bytes != m1.total_remote_logical_bytes
        or fixed.total_all_publication_logical_bytes
        != m1.total_all_publication_logical_bytes
    ):
        raise ValueError("host-global A/B changes the object or failure contract")


def confirm_gradient_equivalence(
    plan: ActiveGradientPlan,
    tensors: Mapping[tuple[int, str], torch.Tensor],
    *,
    epoch: int,
) -> ConfirmedGradientEncoding:
    """Validate test/offline tensors, then confirm from their compact checksums."""

    expected = {
        (rank, item.object_id)
        for item in plan.objects
        for rank in item.holders
    }
    if set(tensors) != expected:
        raise ValueError("runtime gradient tensors do not cover the planned candidates")
    checksums = {}
    for item in plan.objects:
        for rank in item.holders:
            tensor = tensors[(rank, item.object_id)]
            if tuple(tensor.shape) != item.lengths or str(tensor.dtype) != item.dtype:
                raise ValueError("runtime gradient tensor differs from its planned schema")
            checksums[(rank, item.object_id)] = tensor_checksum(tensor)
    return confirm_gradient_checksums(plan, checksums, epoch=epoch)


def confirm_gradient_checksums(
    plan: ActiveGradientPlan,
    checksums: Mapping[tuple[int, str], str],
    *,
    epoch: int,
) -> ConfirmedGradientEncoding:
    """Confirm candidates from gathered hashes without gathering tensor payloads."""

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("gradient encoding epoch must be non-negative")
    expected = {
        (rank, item.object_id)
        for item in plan.objects
        for rank in item.holders
    }
    if set(checksums) != expected:
        raise ValueError("runtime gradient checksums do not cover the planned candidates")
    if any(
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(character not in "0123456789abcdef" for character in checksum)
        for checksum in checksums.values()
    ):
        raise ValueError("runtime gradient checksum must be a lowercase SHA-256 digest")

    bindings = {item.rank: item for item in plan.rank_bindings}
    actions = []
    mismatches = []
    for item in plan.objects:
        item_checksums = {
            rank: checksums[(rank, item.object_id)] for rank in item.holders
        }
        if len(set(item_checksums.values())) == 1:
            checksum = next(iter(item_checksums.values()))
            version_id = _digest((item.object_id, epoch, checksum))
            for placement in plan.placements:
                if placement.object_id == item.object_id:
                    actions.append(
                        GradientEncodingAction(
                            item.object_id,
                            version_id,
                            placement.publisher_rank,
                            placement.source_host,
                            placement.target_host,
                            "confirmed_object",
                            checksum,
                            item.logical_bytes,
                        )
                    )
            continue

        mismatches.append(item.object_id)
        if tuple(sorted(item_checksums)) != item.direct_fallback_ranks:
            raise ValueError("runtime fallback ranks differ from the planned candidates")
        hosts = sorted({value.host for value in plan.rank_bindings})
        for rank in item.holders:
            for target_host in hosts:
                actions.append(
                    GradientEncodingAction(
                        item.object_id,
                        _digest(
                            (
                                item.object_id,
                                epoch,
                                item_checksums[rank],
                                rank,
                                "direct_fallback",
                            )
                        ),
                        rank,
                        bindings[rank].host,
                        target_host,
                        "direct_fallback",
                        item_checksums[rank],
                        item.logical_bytes,
                    )
                )
    hosts = tuple(sorted({item.host for item in plan.rank_bindings}))
    required_versions = tuple(
        (
            host,
            tuple(
                sorted(
                    {
                        action.version_id
                        for action in actions
                        if action.target_host == host
                    }
                )
            ),
        )
        for host in hosts
    )
    if len({value_ids for _, value_ids in required_versions}) != 1:
        raise ValueError("confirmed encoding does not replicate every version per host")
    required_set = set(required_versions[0][1])
    for failed in plan.failure_sets:
        surviving = {
            version_id
            for host, version_ids in required_versions
            if host not in failed
            for version_id in version_ids
        }
        if surviving != required_set:
            raise ValueError("confirmed encoding is incomplete after a planned host loss")
    loads: dict[int, int] = {}
    all_publication_loads = {
        rank: 0 for rank in (item.rank for item in plan.rank_bindings)
    }
    for action in actions:
        all_publication_loads[action.source_rank] += action.logical_bytes
        if action.remote:
            loads[action.source_rank] = (
                loads.get(action.source_rank, 0) + action.logical_bytes
            )
    encoding_body = {
        "canonical_hash": plan.canonical_hash,
        "epoch": epoch,
        "actions": [
            (
                action.object_id,
                action.version_id,
                action.source_rank,
                action.source_host,
                action.target_host,
                action.action,
                action.checksum,
                action.logical_bytes,
            )
            for action in actions
        ],
        "mismatched_object_ids": sorted(mismatches),
        "required_versions_by_host": required_versions,
        "failure_sets": plan.failure_sets,
    }
    return ConfirmedGradientEncoding(
        plan.canonical_hash,
        epoch,
        tuple(actions),
        tuple(sorted(mismatches)),
        required_versions,
        plan.failure_sets,
        sum(item.logical_bytes for item in actions if item.remote),
        max(loads.values(), default=0),
        tuple(sorted(all_publication_loads.items())),
        sum(all_publication_loads.values()),
        max(all_publication_loads.values(), default=0),
        _digest(encoding_body),
    )
