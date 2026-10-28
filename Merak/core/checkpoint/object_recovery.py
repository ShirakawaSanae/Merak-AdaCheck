"""Rank-independent, object-addressed CPU recovery reference.

This module intentionally sits beside the existing AdaCheck and peer-delta
paths.  It consumes immutable object fragments and never changes their
detector, encoder, transport, or legacy recovery semantics.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

import torch

from .catalog import checkpoint_state_checksum, snapshot_to_cpu, tensor_checksum


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _slice_key(address: "ObjectAddress", offsets: Sequence[int], lengths: Sequence[int]) -> tuple[Any, ...]:
    return (address.key, tuple(offsets), tuple(lengths))


def _volume(lengths: Sequence[int]) -> int:
    result = 1
    for length in lengths:
        result *= int(length)
    return result


def _rectangles_overlap(left: ObjectFragment, right: ObjectFragment) -> bool:
    return _ranges_overlap(left.offsets, left.lengths, right.offsets, right.lengths)


def _ranges_overlap(
    left_offsets: Sequence[int],
    left_lengths: Sequence[int],
    right_offsets: Sequence[int],
    right_lengths: Sequence[int],
) -> bool:
    return all(
        max(left_offset, right_offset) < min(left_offset + left_length, right_offset + right_length)
        for left_offset, left_length, right_offset, right_length in zip(
            left_offsets, left_lengths, right_offsets, right_lengths
        )
    )


@dataclass(frozen=True)
class ObjectAddress:
    """Stable identity of one global tensor or exact opaque state value."""

    semantic_id: str
    role: str
    global_shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.semantic_id or not self.role or not self.dtype:
            raise ValueError("object address fields cannot be empty")
        if not self.global_shape or any(int(size) <= 0 for size in self.global_shape):
            raise ValueError("object address needs a positive global shape")

    @property
    def key(self) -> tuple[Any, ...]:
        return (self.semantic_id, self.role, self.global_shape, self.dtype)

    @property
    def address_id(self) -> str:
        return _digest(self.key)


@dataclass(frozen=True)
class ObjectFragment:
    """One versioned rectangular global slice stored in a failure domain."""

    address: ObjectAddress
    epoch: int
    offsets: tuple[int, ...]
    lengths: tuple[int, ...]
    tensor: torch.Tensor | None
    source_rank: int
    storage_host: int
    checksum: str
    shard_id: str
    version_id: str
    base_epoch: int | None
    dependency_epoch: int | None

    @classmethod
    def from_tensor(
        cls,
        address: ObjectAddress,
        *,
        epoch: int,
        offsets: Sequence[int],
        tensor: torch.Tensor,
        source_rank: int,
        storage_host: int,
        base_epoch: int | None = None,
        dependency_epoch: int | None = None,
        clone: bool = True,
    ) -> "ObjectFragment":
        copied = tensor.detach().cpu().contiguous()
        if clone:
            copied = copied.clone()
        normalized_offsets = tuple(int(value) for value in offsets)
        normalized_base_epoch = int(epoch) if base_epoch is None else int(base_epoch)
        normalized_dependency_epoch = (
            None if dependency_epoch is None else int(dependency_epoch)
        )
        checksum = tensor_checksum(copied)
        shard_id = _digest((address.key, normalized_offsets, tuple(copied.shape)))
        version_id = _digest(
            (
                shard_id,
                int(epoch),
                checksum,
                normalized_base_epoch,
                normalized_dependency_epoch,
            )
        )
        return cls(
            address=address,
            epoch=int(epoch),
            offsets=normalized_offsets,
            lengths=tuple(int(value) for value in copied.shape),
            tensor=copied,
            source_rank=int(source_rank),
            storage_host=int(storage_host),
            checksum=checksum,
            shard_id=shard_id,
            version_id=version_id,
            base_epoch=normalized_base_epoch,
            dependency_epoch=normalized_dependency_epoch,
        )

    def validate(self) -> None:
        if self.epoch < 0 or self.source_rank < 0 or self.storage_host < 0:
            raise ValueError("object fragment has an invalid epoch or location")
        if not self.shard_id or not self.version_id or self.base_epoch is None:
            raise ValueError("object fragment lacks shard or version identity")
        if self.base_epoch != self.epoch or self.dependency_epoch is not None:
            raise ValueError(
                "object recovery accepts only a materialized final snapshot"
            )
        if len(self.offsets) != len(self.address.global_shape) or len(self.lengths) != len(self.offsets):
            raise ValueError("object fragment rank differs from its global shape")
        if self.tensor is not None:
            if tuple(self.tensor.shape) != self.lengths:
                raise ValueError("object fragment tensor shape differs from its slice")
            if str(self.tensor.dtype) != self.address.dtype:
                raise ValueError("object fragment tensor dtype differs from its address")
        if any(offset < 0 or length <= 0 for offset, length in zip(self.offsets, self.lengths)):
            raise ValueError("object fragment has an invalid slice")
        if any(
            offset + length > size
            for offset, length, size in zip(self.offsets, self.lengths, self.address.global_shape)
        ):
            raise ValueError("object fragment slice exceeds the global shape")
        if self.tensor is not None and tensor_checksum(self.tensor) != self.checksum:
            raise ValueError("object fragment checksum mismatch")
        expected_shard = _digest((self.address.key, self.offsets, self.lengths))
        expected_version = _digest(
            (
                self.shard_id,
                self.epoch,
                self.checksum,
                self.base_epoch,
                self.dependency_epoch,
            )
        )
        if self.shard_id != expected_shard or self.version_id != expected_version:
            raise ValueError("object fragment shard or version identity mismatch")


@dataclass(frozen=True)
class OpaqueRecord:
    """Exact rank-local state that is deliberately not remapped or reshaped."""

    owner_rank: int
    key: str
    epoch: int
    value: Any
    storage_hosts: tuple[int, ...]
    checksum: str

    @classmethod
    def from_value(
        cls,
        *,
        owner_rank: int,
        key: str,
        epoch: int,
        value: Any,
        storage_hosts: Sequence[int],
    ) -> "OpaqueRecord":
        copied = snapshot_to_cpu(value)
        return cls(
            owner_rank=int(owner_rank),
            key=str(key),
            epoch=int(epoch),
            value=copied,
            storage_hosts=tuple(sorted({int(host) for host in storage_hosts})),
            checksum=checkpoint_state_checksum(copied),
        )

    def validate(self) -> None:
        if self.owner_rank < 0 or self.epoch < 0 or not self.key or not self.storage_hosts:
            raise ValueError("opaque record identity is invalid")
        if any(host < 0 for host in self.storage_hosts):
            raise ValueError("opaque record has an invalid storage host")
        if checkpoint_state_checksum(self.value) != self.checksum:
            raise ValueError("opaque record checksum mismatch")


@dataclass(frozen=True)
class ObjectCatalog:
    epoch: int
    fragments: tuple[ObjectFragment, ...]
    opaque_records: tuple[OpaqueRecord, ...] = ()

    def validate(self) -> None:
        if self.epoch < 0 or not self.fragments:
            raise ValueError("object catalog needs an epoch and at least one fragment")
        opaque_keys = set()
        logical_fragments: dict[tuple[Any, ...], ObjectFragment] = {}
        for item in self.fragments:
            item.validate()
            if item.epoch != self.epoch:
                raise ValueError("object catalog contains a stale fragment")
            logical_key = (item.address.key, item.offsets, item.lengths)
            previous = logical_fragments.get(logical_key)
            if previous is not None and (
                previous.checksum != item.checksum
                or previous.version_id != item.version_id
                or previous.base_epoch != item.base_epoch
                or previous.dependency_epoch != item.dependency_epoch
            ):
                raise ValueError("replicas of one logical fragment conflict")
            logical_fragments.setdefault(logical_key, item)
        by_address: dict[tuple[Any, ...], list[ObjectFragment]] = {}
        for item in logical_fragments.values():
            by_address.setdefault(item.address.key, []).append(item)
        for fragments in by_address.values():
            address = fragments[0].address
            for index, left in enumerate(fragments):
                if any(_rectangles_overlap(left, right) for right in fragments[index + 1 :]):
                    raise ValueError("logical source fragments overlap")
            if sum(_volume(item.lengths) for item in fragments) != _volume(address.global_shape):
                ranges = [
                    (item.offsets, item.lengths, item.storage_host, item.source_rank)
                    for item in fragments[:8]
                ]
                raise ValueError(
                    "logical source fragments have a coverage gap: "
                    f"semantic_id={address.semantic_id!r}, role={address.role!r}, "
                    f"global_shape={address.global_shape}, dtype={address.dtype!r}, "
                    f"fragments={ranges}, fragment_count={len(fragments)}"
                )
        for item in self.opaque_records:
            item.validate()
            if item.epoch != self.epoch:
                raise ValueError("object catalog contains stale opaque state")
            key = (item.owner_rank, item.key)
            if key in opaque_keys:
                raise ValueError("duplicate opaque record identity")
            opaque_keys.add(key)

    @property
    def content_hash(self) -> str:
        """Placement-independent semantic/content identity for fair A/B runs."""
        self.validate()
        fragments = {
            (
                item.address.key,
                item.epoch,
                item.offsets,
                item.lengths,
                item.shard_id,
                item.version_id,
                item.checksum,
                item.base_epoch,
                item.dependency_epoch,
            )
            for item in self.fragments
        }
        return _digest(
            {
                "epoch": self.epoch,
                "fragments": sorted(fragments),
                "opaque": [
                    (item.owner_rank, item.key, item.epoch, item.checksum)
                    for item in sorted(self.opaque_records, key=lambda item: (item.owner_rank, item.key))
                ],
            }
        )

    @property
    def availability_hash(self) -> str:
        """Physical publishers and failure-domain availability, kept separate."""
        self.validate()
        return _digest(
            {
                "fragments": sorted(
                    (
                        item.shard_id,
                        item.version_id,
                        item.source_rank,
                        item.storage_host,
                    )
                    for item in self.fragments
                ),
                "opaque": [
                    (item.owner_rank, item.key, item.epoch, item.storage_hosts)
                    for item in sorted(self.opaque_records, key=lambda item: (item.owner_rank, item.key))
                ],
            }
        )

    @property
    def catalog_hash(self) -> str:
        """Backward-compatible name for the placement-independent content hash."""
        return self.content_hash


@dataclass(frozen=True)
class SliceRequest:
    target_rank: int
    address: ObjectAddress
    offsets: tuple[int, ...]
    lengths: tuple[int, ...]
    source_address: ObjectAddress | None = None

    def __post_init__(self) -> None:
        if self.target_rank < 0:
            raise ValueError("target rank cannot be negative")
        if len(self.offsets) != len(self.address.global_shape) or len(self.lengths) != len(self.offsets):
            raise ValueError("slice request rank differs from its global shape")
        if any(offset < 0 or length <= 0 for offset, length in zip(self.offsets, self.lengths)):
            raise ValueError("slice request has an invalid range")
        if any(
            offset + length > size
            for offset, length, size in zip(self.offsets, self.lengths, self.address.global_shape)
        ):
            raise ValueError("slice request exceeds the global shape")
        if self.source_address is not None and (
            self.source_address.role != self.address.role
            or self.source_address.global_shape != self.address.global_shape
            or self.source_address.dtype != self.address.dtype
        ):
            raise ValueError("source alias is incompatible with the target tensor")

    @property
    def source(self) -> ObjectAddress:
        return self.address if self.source_address is None else self.source_address

    @property
    def key(self) -> tuple[Any, ...]:
        return (
            self.target_rank,
            *_slice_key(self.address, self.offsets, self.lengths),
            self.source.key,
        )


@dataclass(frozen=True)
class OpaqueRequest:
    target_rank: int
    source_rank: int
    key: str

    def __post_init__(self) -> None:
        if self.target_rank < 0 or self.source_rank < 0 or not self.key:
            raise ValueError("opaque request identity is invalid")


@dataclass(frozen=True)
class TargetLayout:
    slices: tuple[SliceRequest, ...]
    opaque: tuple[OpaqueRequest, ...] = ()

    def __post_init__(self) -> None:
        keys = [item.key for item in self.slices]
        if len(keys) != len(set(keys)):
            raise ValueError("target layout has duplicate tensor slice requests")
        tensor_keys = [(item.target_rank, item.address.key) for item in self.slices]
        if len(tensor_keys) != len(set(tensor_keys)):
            raise ValueError("target layout maps one tensor identity to multiple slices")
        output_keys = [(item.target_rank, item.address.semantic_id) for item in self.slices]
        if len(output_keys) != len(set(output_keys)):
            raise ValueError("target layout would overwrite one tensor semantic ID")
        opaque_keys = [(item.target_rank, item.key) for item in self.opaque]
        if len(opaque_keys) != len(set(opaque_keys)):
            raise ValueError("target layout has duplicate opaque requests")

    @property
    def layout_hash(self) -> str:
        return _digest(
            {
                "slices": [item.key for item in sorted(self.slices, key=lambda item: item.key)],
                "opaque": [
                    (item.target_rank, item.source_rank, item.key)
                    for item in sorted(self.opaque, key=lambda item: (item.target_rank, item.key))
                ],
            }
        )


@dataclass(frozen=True)
class ObjectRuntimeIdentity:
    """Fields which must be identical for a fair Fixed versus M1 comparison."""

    detector_catalog_hash: str
    semantic_plan_hash: str
    encoder_name: str
    source_layout_hash: str
    target_layout_hash: str
    epoch: int
    failure_sets: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if (
            not self.detector_catalog_hash
            or not self.semantic_plan_hash
            or not self.encoder_name
            or not self.source_layout_hash
            or not self.target_layout_hash
        ):
            raise ValueError("object runtime identity is incomplete")
        if self.epoch < 0:
            raise ValueError("object runtime identity epoch is invalid")
        canonical = tuple(sorted(tuple(sorted(set(item))) for item in self.failure_sets))
        if canonical != self.failure_sets or len(set(canonical)) != len(canonical):
            raise ValueError("failure sets must be unique canonical host sets")
        if any(host < 0 for failure_set in canonical for host in failure_set):
            raise ValueError("failure sets cannot contain negative hosts")


@dataclass(frozen=True)
class SelectedFragment:
    request: SliceRequest
    source_rank: int
    storage_host: int
    offsets: tuple[int, ...]
    lengths: tuple[int, ...]
    shard_id: str
    version_id: str
    checksum: str

    @property
    def logical_bytes(self) -> int:
        dtype = getattr(torch, self.request.address.dtype.split(".")[-1])
        return _volume(self.lengths) * dtype.itemsize


@dataclass(frozen=True)
class FailureLoadPlan:
    failed_hosts: tuple[int, ...]
    selections: tuple[SelectedFragment, ...]
    publisher_bytes: tuple[tuple[int, int], ...]
    opaque_sources: tuple[tuple[int, str, int], ...] = ()

    @property
    def total_selected_bytes(self) -> int:
        return sum(item.logical_bytes for item in self.selections)

    @property
    def maximum_publisher_bytes(self) -> int:
        return max((value for _, value in self.publisher_bytes), default=0)


@dataclass(frozen=True)
class ObjectRecoveryPlan:
    arm: Literal["adacheck_object_fixed", "m1_object"]
    runtime: ObjectRuntimeIdentity
    catalog_hash: str
    availability_hash: str
    failure_plans: tuple[FailureLoadPlan, ...]
    sealed_plan_hash: str

    def __post_init__(self) -> None:
        if self.arm not in {"adacheck_object_fixed", "m1_object"}:
            raise ValueError("unsupported object recovery arm")
        if not self.catalog_hash or not self.availability_hash:
            raise ValueError("sealed object plan lacks catalog identity")
        if tuple(item.failed_hosts for item in self.failure_plans) != self.runtime.failure_sets:
            raise ValueError("sealed object plan does not cover its failure contract")
        if self.sealed_plan_hash != self.plan_hash:
            raise ValueError("sealed object plan hash mismatch")

    def failure_plan(self, failed_hosts: Sequence[int]) -> FailureLoadPlan:
        key = tuple(sorted(set(int(host) for host in failed_hosts)))
        for item in self.failure_plans:
            if item.failed_hosts == key:
                return item
        raise ValueError("sealed object recovery plan has no requested failure set")

    @property
    def total_selected_bytes(self) -> int:
        return self.failure_plan(()).total_selected_bytes

    @property
    def publisher_bytes(self) -> tuple[tuple[int, int], ...]:
        return self.failure_plan(()).publisher_bytes

    @property
    def maximum_publisher_bytes(self) -> int:
        return self.failure_plan(()).maximum_publisher_bytes

    @property
    def plan_hash(self) -> str:
        return self._digest_value(
            self.arm,
            self.runtime,
            self.catalog_hash,
            self.availability_hash,
            self.failure_plans,
        )

    @staticmethod
    def _digest_value(
        arm: str,
        runtime: ObjectRuntimeIdentity,
        catalog_hash: str,
        availability_hash: str,
        failure_plans: Sequence[FailureLoadPlan],
    ) -> str:
        return _digest(
            {
                "arm": arm,
                "runtime": {
                    "detector_catalog_hash": runtime.detector_catalog_hash,
                    "semantic_plan_hash": runtime.semantic_plan_hash,
                    "encoder_name": runtime.encoder_name,
                    "source_layout_hash": runtime.source_layout_hash,
                    "target_layout_hash": runtime.target_layout_hash,
                    "epoch": runtime.epoch,
                    "failure_sets": runtime.failure_sets,
                },
                "catalog_hash": catalog_hash,
                "availability_hash": availability_hash,
                "failure_plans": [
                    {
                        "failed_hosts": item.failed_hosts,
                        "publisher_bytes": item.publisher_bytes,
                        "opaque_sources": item.opaque_sources,
                        "selections": [
                            (
                                selected.request.key,
                                selected.source_rank,
                                selected.storage_host,
                                selected.offsets,
                                selected.lengths,
                                selected.shard_id,
                                selected.version_id,
                                selected.checksum,
                            )
                            for selected in item.selections
                        ],
                    }
                    for item in failure_plans
                ],
            }
        )


@dataclass(frozen=True)
class RestoredRankState:
    tensors: Mapping[str, torch.Tensor]
    opaque: Mapping[str, Any]


def _intersection(
    request: SliceRequest, fragment: ObjectFragment
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    if fragment.address.key != request.source.key:
        return None
    starts = tuple(max(left, right) for left, right in zip(request.offsets, fragment.offsets))
    ends = tuple(
        min(left + left_size, right + right_size)
        for left, left_size, right, right_size in zip(
            request.offsets, request.lengths, fragment.offsets, fragment.lengths
        )
    )
    if any(end <= start for start, end in zip(starts, ends)):
        return None
    return starts, tuple(end - start for start, end in zip(starts, ends))


def _eligible_fragments(
    catalog: ObjectCatalog, request: SliceRequest, *, failed_hosts: set[int]
) -> list[ObjectFragment]:
    candidates = []
    for item in catalog.fragments:
        if item.address.key != request.source.key or item.storage_host in failed_hosts:
            continue
        if _intersection(request, item) is None:
            continue
        if item.epoch != catalog.epoch:
            raise ValueError("stale object fragment intersects the requested epoch")
        candidates.append(item)
    return candidates


def _plan_fragments(
    catalog: ObjectCatalog,
    request: SliceRequest,
    *,
    failed_hosts: set[int],
    policy: Literal["fixed", "m1"],
    publisher_loads: dict[int, int],
) -> list[SelectedFragment]:
    candidates = _eligible_fragments(catalog, request, failed_hosts=failed_hosts)
    if not candidates:
        same_semantic = sorted(
            (
                item.address.role,
                item.address.global_shape,
                item.address.dtype,
                item.offsets,
                item.lengths,
                item.storage_host,
                item.source_rank,
            )
            for item in catalog.fragments
            if item.address.semantic_id == request.source.semantic_id
        )
        raise ValueError(
            "requested object slice has no surviving source: "
            f"target_rank={request.target_rank}, "
            f"semantic_id={request.address.semantic_id!r}, "
            f"role={request.address.role!r}, "
            f"global_shape={request.address.global_shape}, dtype={request.address.dtype!r}, "
            f"offsets={request.offsets}, lengths={request.lengths}, "
            f"failed_hosts={sorted(failed_hosts)}, "
            f"same_semantic_candidates={same_semantic[:4]}, "
            f"candidate_count={len(same_semantic)}"
        )
    replicas: dict[tuple[str, str], list[ObjectFragment]] = {}
    for item in candidates:
        replicas.setdefault((item.shard_id, item.version_id), []).append(item)
    selections = []
    total_volume = 0
    ordered_chunks = sorted(replicas)
    if policy == "m1":
        ordered_chunks.sort(
            key=lambda key: (
                -_volume(_intersection(request, replicas[key][0])[1]),
                key,
            )
        )
    for chunk_key in ordered_chunks:
        replicas_for_chunk = replicas[chunk_key]
        if policy == "fixed":
            item = min(
                replicas_for_chunk,
                key=lambda value: (value.source_rank, value.storage_host),
            )
        else:
            item = min(
                replicas_for_chunk,
                key=lambda value: (
                    publisher_loads.get(value.source_rank, 0),
                    value.source_rank,
                    value.storage_host,
                ),
            )
        intersection = _intersection(request, item)
        assert intersection is not None
        starts, lengths = intersection
        dtype = getattr(torch, item.address.dtype.split(".")[-1])
        transferred = _volume(lengths) * dtype.itemsize
        publisher_loads[item.source_rank] = publisher_loads.get(item.source_rank, 0) + transferred
        total_volume += _volume(lengths)
        selections.append(
            SelectedFragment(
                request,
                item.source_rank,
                item.storage_host,
                starts,
                lengths,
                item.shard_id,
                item.version_id,
                item.checksum,
            )
        )
    if total_volume != _volume(request.lengths):
        raise ValueError("requested object slice has a coverage gap")
    return selections


def validate_failure_closure(
    catalog: ObjectCatalog, layout: TargetLayout, failure_sets: Iterable[Sequence[int]]
) -> None:
    """Require every target tensor and opaque value after each declared failure."""
    catalog.validate()
    for failed in failure_sets:
        failed_hosts = {int(host) for host in failed}
        if any(host < 0 for host in failed_hosts):
            raise ValueError("failure sets cannot contain negative hosts")
        loads: dict[int, int] = {}
        for request in layout.slices:
            _plan_fragments(
                catalog, request, failed_hosts=failed_hosts, policy="fixed", publisher_loads=loads
            )
        opaque_by_key = {(item.owner_rank, item.key): item for item in catalog.opaque_records}
        for request in layout.opaque:
            if request.target_rank != request.source_rank:
                raise ValueError("opaque state cannot be remapped across ranks")
            item = opaque_by_key.get((request.source_rank, request.key))
            if item is None or not set(item.storage_hosts) - failed_hosts:
                raise ValueError("opaque state lacks a surviving exact replica")


def build_object_recovery_plan(
    catalog: ObjectCatalog,
    layout: TargetLayout,
    *,
    arm: Literal["adacheck_object_fixed", "m1_object"],
    runtime: ObjectRuntimeIdentity,
) -> ObjectRecoveryPlan:
    """Build a sealed CPU load plan; only source selection differs by arm."""
    if arm not in {"adacheck_object_fixed", "m1_object"}:
        raise ValueError("unsupported object recovery arm")
    if catalog.epoch != runtime.epoch:
        raise ValueError("object catalog epoch differs from the runtime contract")
    if layout.layout_hash != runtime.target_layout_hash:
        raise ValueError("target layout differs from the runtime contract")
    validate_failure_closure(catalog, layout, runtime.failure_sets)
    policy: Literal["fixed", "m1"] = "fixed" if arm == "adacheck_object_fixed" else "m1"
    opaque_by_key = {(item.owner_rank, item.key): item for item in catalog.opaque_records}
    failure_plans = []
    for failure_set in runtime.failure_sets:
        failed_hosts = set(failure_set)
        loads: dict[int, int] = {}
        selections = []
        for request in sorted(layout.slices, key=lambda item: item.key):
            selected = _plan_fragments(
                catalog,
                request,
                failed_hosts=failed_hosts,
                policy=policy,
                publisher_loads=loads,
            )
            selections.extend(selected)
        opaque_sources = []
        for request in sorted(layout.opaque, key=lambda item: (item.target_rank, item.key)):
            item = opaque_by_key[(request.source_rank, request.key)]
            host = min(set(item.storage_hosts) - failed_hosts)
            opaque_sources.append((request.target_rank, request.key, host))
        failure_plans.append(
            FailureLoadPlan(
                failed_hosts=failure_set,
                selections=tuple(selections),
                publisher_bytes=tuple(sorted(loads.items())),
                opaque_sources=tuple(opaque_sources),
            )
        )
    frozen_plans = tuple(failure_plans)
    content_hash = catalog.content_hash
    availability_hash = catalog.availability_hash
    return ObjectRecoveryPlan(
        arm,
        runtime,
        content_hash,
        availability_hash,
        frozen_plans,
        ObjectRecoveryPlan._digest_value(
            arm, runtime, content_hash, availability_hash, frozen_plans
        ),
    )


def assert_fair_object_ab(
    fixed: ObjectRecoveryPlan, m1: ObjectRecoveryPlan
) -> None:
    """Reject comparisons that silently change detector, encoder, layout, or faults."""
    if fixed.arm != "adacheck_object_fixed" or m1.arm != "m1_object":
        raise ValueError("object A/B needs fixed and M1 arms")
    if (
        fixed.runtime != m1.runtime
        or fixed.catalog_hash != m1.catalog_hash
    ):
        raise ValueError("object A/B changes a common-runtime input")


def _sealed_tensor(
    catalog: ObjectCatalog,
    request: SliceRequest,
    selections: Sequence[SelectedFragment],
    *,
    failed_hosts: set[int],
) -> torch.Tensor:
    resolved = []
    for selected in selections:
        if selected.request != request or selected.storage_host in failed_hosts:
            raise ValueError("sealed selection is invalid for the restore failure set")
        matches = [
            item
            for item in catalog.fragments
            if item.address.key == request.source.key
            and item.epoch == catalog.epoch
            and item.source_rank == selected.source_rank
            and item.storage_host == selected.storage_host
            and item.shard_id == selected.shard_id
            and item.version_id == selected.version_id
            and item.checksum == selected.checksum
        ]
        if len(matches) != 1:
            raise ValueError("sealed selection fragment is missing or ambiguous")
        item = matches[0]
        if item.tensor is None:
            raise ValueError("sealed selection payload is unavailable")
        intersection = _intersection(request, item)
        if intersection != (selected.offsets, selected.lengths):
            raise ValueError("sealed selection slice differs from its fragment")
        resolved.append((selected, item))
    if any(
        _ranges_overlap(left.offsets, left.lengths, right.offsets, right.lengths)
        for index, (left, _) in enumerate(resolved)
        for right, _ in resolved[index + 1 :]
    ):
        raise ValueError("sealed selections overlap")
    if sum(_volume(selected.lengths) for selected, _ in resolved) != _volume(request.lengths):
        raise ValueError("sealed selections leave a coverage gap")
    result = torch.empty(
        request.lengths,
        dtype=getattr(torch, request.address.dtype.split(".")[-1]),
    )
    for selected, item in resolved:
        destination = tuple(
            slice(start - base, start - base + length)
            for start, base, length in zip(selected.offsets, request.offsets, selected.lengths)
        )
        source = tuple(
            slice(start - base, start - base + length)
            for start, base, length in zip(selected.offsets, item.offsets, selected.lengths)
        )
        result[destination] = item.tensor[source]
    return result


def restore_object_layout(
    catalog: ObjectCatalog,
    plan: ObjectRecoveryPlan,
    layout: TargetLayout,
    *,
    failed_hosts: Sequence[int] = (),
) -> dict[int, RestoredRankState]:
    """Assemble target slices from surviving services and exact opaque state."""
    if plan.arm not in {"adacheck_object_fixed", "m1_object"}:
        raise ValueError("unsupported object recovery arm")
    if catalog.content_hash != plan.catalog_hash or catalog.epoch != plan.runtime.epoch:
        raise ValueError("catalog differs from the sealed object recovery plan")
    if catalog.availability_hash != plan.availability_hash:
        raise ValueError("catalog availability differs from the sealed object recovery plan")
    if layout.layout_hash != plan.runtime.target_layout_hash:
        raise ValueError("target layout differs from the sealed object recovery plan")
    failed = {int(host) for host in failed_hosts}
    if tuple(sorted(failed)) not in plan.runtime.failure_sets:
        raise ValueError("restore failure set is outside the sealed recovery contract")
    sealed = plan.failure_plan(tuple(sorted(failed)))
    tensor_states: dict[int, dict[str, torch.Tensor]] = {}
    for request in layout.slices:
        selected = tuple(
            item for item in sealed.selections if item.request == request
        )
        tensor = _sealed_tensor(catalog, request, selected, failed_hosts=failed)
        tensor_states.setdefault(request.target_rank, {})[request.address.semantic_id] = tensor
    opaque_by_key = {(item.owner_rank, item.key): item for item in catalog.opaque_records}
    opaque_states: dict[int, dict[str, Any]] = {}
    sealed_opaque = {(rank, key): host for rank, key, host in sealed.opaque_sources}
    for request in layout.opaque:
        if request.target_rank != request.source_rank:
            raise ValueError("opaque state cannot be remapped across ranks")
        item = opaque_by_key.get((request.source_rank, request.key))
        host = sealed_opaque.get((request.target_rank, request.key))
        if item is None or host is None or host in failed or host not in item.storage_hosts:
            raise ValueError("opaque state is unavailable after the declared failure")
        opaque_states.setdefault(request.target_rank, {})[request.key] = copy.deepcopy(item.value)
    ranks = set(tensor_states) | set(opaque_states)
    return {
        rank: RestoredRankState(tensor_states.get(rank, {}), opaque_states.get(rank, {}))
        for rank in sorted(ranks)
    }
