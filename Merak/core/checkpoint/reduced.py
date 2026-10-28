"""Recoverable tensor deduplication and its replication cost model."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import torch

from .catalog import (
    CheckpointObject,
    TensorRedundancyEvidence,
    checkpoint_object_id,
    checkpoint_path,
    iter_checkpoint_objects,
    snapshot_to_cpu,
    tensor_checksum,
)


@dataclass(frozen=True)
class TensorReference:
    value_id: str
    alias_id: str


@dataclass(frozen=True)
class TensorRequirement:
    rank: int
    path: str
    object_id: str
    alias_id: str


@dataclass(frozen=True)
class ReducedObject:
    value_id: str
    role: str
    dtype: str
    shape: tuple[int, ...]
    logical_bytes: int
    checksum: str
    source_ranks: tuple[int, ...]
    requirements: tuple[TensorRequirement, ...]


@dataclass(frozen=True)
class ReducedCheckpoint:
    templates: Mapping[int, Any]
    tensors: Mapping[str, torch.Tensor]
    objects: tuple[ReducedObject, ...]
    full_logical_bytes: int

    @property
    def reduced_logical_bytes(self) -> int:
        return sum(item.logical_bytes for item in self.objects)

    def to_wire_state(self) -> dict[str, Any]:
        """Expose canonical tensors to the generic checkpoint catalog."""
        self.validate()
        return {
            "schema_version": 1,
            "templates": dict(self.templates),
            "tensors": dict(self.tensors),
            "objects": self.objects,
            "full_logical_bytes": self.full_logical_bytes,
        }

    @classmethod
    def from_wire_state(cls, value: Mapping[str, Any]) -> "ReducedCheckpoint":
        if int(value.get("schema_version", -1)) != 1:
            raise ValueError("unsupported reduced checkpoint wire schema")
        templates = value.get("templates")
        tensors = value.get("tensors")
        objects = value.get("objects")
        if not isinstance(templates, Mapping) or not isinstance(tensors, Mapping):
            raise ValueError("reduced checkpoint wire mappings are missing")
        if not isinstance(objects, (list, tuple)) or not all(
            isinstance(item, ReducedObject) for item in objects
        ):
            raise ValueError("reduced checkpoint wire objects are invalid")
        reduced = cls(
            templates=templates,
            tensors=tensors,
            objects=tuple(objects),
            full_logical_bytes=int(value["full_logical_bytes"]),
        )
        reduced.validate()
        return reduced

    def validate(self) -> None:
        object_ids = {item.value_id for item in self.objects}
        if len(object_ids) != len(self.objects):
            raise ValueError("duplicate reduced object identity")
        if set(self.tensors) != object_ids:
            raise ValueError("reduced tensor/object identities do not match")

        expected_requirements = {}
        for item in self.objects:
            tensor = self.tensors.get(item.value_id)
            if tensor is None:
                raise ValueError(f"missing reduced tensor {item.value_id}")
            if str(tensor.dtype) != item.dtype or tuple(tensor.shape) != item.shape:
                raise ValueError(f"metadata mismatch for reduced tensor {item.value_id}")
            if tensor_checksum(tensor) != item.checksum:
                raise ValueError(f"checksum mismatch for reduced tensor {item.value_id}")
            requirement_ranks = set()
            for requirement in item.requirements:
                key = (requirement.rank, requirement.path, requirement.alias_id)
                if key in expected_requirements:
                    raise ValueError(f"duplicate reduced requirement {key}")
                expected_requirements[key] = item.value_id
                requirement_ranks.add(requirement.rank)
            if tuple(sorted(requirement_ranks)) != item.source_ranks:
                raise ValueError(
                    f"source ranks do not match requirements for {item.value_id}"
                )

        actual_requirements = {}
        aliases = {}

        def visit(value: Any, rank: int, components: tuple[Any, ...]) -> None:
            if isinstance(value, TensorReference):
                path = checkpoint_path(components)
                key = (rank, path, value.alias_id)
                if key in actual_requirements:
                    raise ValueError(f"duplicate template tensor reference {key}")
                actual_requirements[key] = value.value_id
                alias_key = (rank, value.alias_id)
                previous = aliases.setdefault(alias_key, value.value_id)
                if previous != value.value_id:
                    raise ValueError(f"alias {value.alias_id} has conflicting values")
                return
            if isinstance(value, Mapping):
                for key, item in value.items():
                    visit(item, rank, components + (key,))
                return
            if isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    visit(item, rank, components + (index,))

        for rank, template in self.templates.items():
            visit(template, int(rank), ())
        if actual_requirements != expected_requirements:
            raise ValueError("reduced templates and object requirements do not match")

    def reconstruct_rank(self, rank: int) -> Any:
        if rank not in self.templates:
            raise KeyError(f"rank {rank} is absent from the reduced checkpoint")
        self.validate()

        aliases: dict[str, torch.Tensor] = {}

        def expand(value: Any) -> Any:
            if isinstance(value, TensorReference):
                tensor = aliases.get(value.alias_id)
                if tensor is None:
                    tensor = self.tensors[value.value_id].clone()
                    aliases[value.alias_id] = tensor
                return tensor
            if isinstance(value, defaultdict):
                result = defaultdict(value.default_factory)
                result.update((key, expand(item)) for key, item in value.items())
                return result
            if isinstance(value, Mapping):
                items = [(key, expand(item)) for key, item in value.items()]
                try:
                    return value.__class__(items)
                except TypeError:
                    return dict(items)
            if isinstance(value, list):
                return [expand(item) for item in value]
            if isinstance(value, tuple):
                items = [expand(item) for item in value]
                if hasattr(value, "_fields"):
                    return value.__class__(*items)
                return tuple(items)
            return snapshot_to_cpu(value)

        return expand(self.templates[rank])


@dataclass(frozen=True)
class ProtectionTransfer:
    value_id: str
    source_rank: int
    source_domain: str
    target_domain: str
    logical_bytes: int
    wire_bytes: int
    weighted_cost: float


@dataclass(frozen=True)
class ProtectionPlacement:
    value_id: str
    target_domains: tuple[str, ...]
    transfers: tuple[ProtectionTransfer, ...]


@dataclass(frozen=True)
class ReplicationCost:
    replica_count: int
    tolerated_failures: int
    full_replication_bytes: int
    reduced_replication_bytes: int
    cross_domain_bytes: int
    weighted_cost: float
    placements: tuple[ProtectionPlacement, ...]

    @property
    def logical_reduction_ratio(self) -> float:
        if self.full_replication_bytes == 0:
            return 0.0
        return 1.0 - self.reduced_replication_bytes / self.full_replication_bytes

    def unrecoverable_values(self, failed_domains: Sequence[str]) -> tuple[str, ...]:
        failed = set(failed_domains)
        return tuple(
            item.value_id
            for item in self.placements
            if not set(item.target_domains) - failed
        )

    def validate_failure_budget(self) -> None:
        domains = sorted(
            {domain for item in self.placements for domain in item.target_domains}
        )
        for failure_count in range(1, self.tolerated_failures + 1):
            for failed in combinations(domains, failure_count):
                missing = self.unrecoverable_values(failed)
                if missing:
                    raise ValueError(
                        f"failure domains {failed} lose reduced values {missing}"
                    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "replica_count": self.replica_count,
            "tolerated_failures": self.tolerated_failures,
            "full_replication_bytes": self.full_replication_bytes,
            "reduced_replication_bytes": self.reduced_replication_bytes,
            "cross_domain_bytes": self.cross_domain_bytes,
            "weighted_cost": self.weighted_cost,
            "logical_reduction_ratio": self.logical_reduction_ratio,
            "placements": [
                {
                    "value_id": item.value_id,
                    "target_domains": list(item.target_domains),
                    "transfers": [transfer.__dict__ for transfer in item.transfers],
                }
                for item in self.placements
            ],
        }


@dataclass(frozen=True)
class DetectorEvidenceConfirmation:
    input_records: int
    candidate_components: int
    confirmed_components: int
    candidate_pairs: int
    confirmed_pairs: int
    rejected_pairs: int
    unmatched_locations: int


def confirm_detector_object_evidence(
    objects: Mapping[tuple[int, str], CheckpointObject],
    detector_evidence: Sequence[TensorRedundancyEvidence],
) -> tuple[tuple[TensorRedundancyEvidence, ...], DetectorEvidenceConfirmation]:
    """Split detector hash buckets using strong object-catalog identities."""
    evidence_by_path = {}
    evidence_by_location: dict[
        tuple[int, int], list[TensorRedundancyEvidence]
    ] = defaultdict(list)
    graph: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    for item in detector_evidence:
        key = (item.rank, item.path)
        if key in evidence_by_path:
            raise ValueError(f"duplicate detector evidence for rank/path {key}")
        checkpoint_object = objects.get(key)
        if checkpoint_object is None:
            raise ValueError(f"detector evidence does not map to checkpoint tensor {key}")
        if checkpoint_object.role != item.role:
            raise ValueError(f"detector role mismatch for rank {item.rank} {item.path}")
        local = (item.rank, item.local_index)
        if local not in item.equivalent_locations:
            raise ValueError(f"detector evidence omits its local location {local}")
        evidence_by_path[key] = item
        evidence_by_location[local].append(item)
        graph[local]
        for remote in item.equivalent_locations:
            graph[local].add(remote)
            graph[remote].add(local)

    components = []
    unseen = set(graph)
    while unseen:
        stack = [min(unseen)]
        component = set()
        while stack:
            location = stack.pop()
            if location in component:
                continue
            component.add(location)
            stack.extend(graph[location] - component)
        components.append(component)
        unseen -= component

    confirmed = []
    candidate_pairs = 0
    confirmed_pairs = 0
    confirmed_components = 0
    unmatched_locations = 0
    next_index = 0
    for component in components:
        records = [
            item
            for location in component
            for item in evidence_by_location.get(location, ())
        ]
        unmatched_locations += sum(
            location not in evidence_by_location for location in component
        )
        candidate_pairs += len(records) * (len(records) - 1) // 2
        by_signature = defaultdict(list)
        for item in records:
            checkpoint_object = objects[(item.rank, item.path)]
            signature = (
                item.role,
                checkpoint_object.dtype,
                checkpoint_object.shape,
                checkpoint_object.checksum,
            )
            by_signature[signature].append(item)
        for records_for_signature in by_signature.values():
            confirmed_components += 1
            confirmed_pairs += (
                len(records_for_signature) * (len(records_for_signature) - 1) // 2
            )
            locations = tuple(
                sorted({(item.rank, next_index) for item in records_for_signature})
            )
            for item in records_for_signature:
                confirmed.append(
                    TensorRedundancyEvidence(
                        rank=item.rank,
                        role=item.role,
                        path=item.path,
                        local_index=next_index,
                        equivalent_locations=locations,
                    )
                )
            next_index += 1

    if len(confirmed) != len(detector_evidence):
        raise ValueError("detector evidence confirmation lost tensor records")
    stats = DetectorEvidenceConfirmation(
        input_records=len(detector_evidence),
        candidate_components=len(components),
        confirmed_components=confirmed_components,
        candidate_pairs=candidate_pairs,
        confirmed_pairs=confirmed_pairs,
        rejected_pairs=candidate_pairs - confirmed_pairs,
        unmatched_locations=unmatched_locations,
    )
    return tuple(sorted(confirmed, key=lambda item: (item.rank, item.path))), stats


def confirm_detector_evidence(
    states: Mapping[int, Any],
    detector_evidence: Sequence[TensorRedundancyEvidence],
) -> tuple[tuple[TensorRedundancyEvidence, ...], DetectorEvidenceConfirmation]:
    """Split detector hash buckets using exact checkpoint tensor identity."""
    objects = {
        (int(rank), item.path): item
        for rank, state in states.items()
        for item in iter_checkpoint_objects(state)
    }
    return confirm_detector_object_evidence(objects, detector_evidence)


def _value_id(
    role: str,
    tensor: torch.Tensor,
    checksum: str,
    detector_scope: tuple[int, int] | None = None,
) -> str:
    identity = json.dumps(
        {
            "role": role,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "checksum": checksum,
            "detector_scope": detector_scope,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(identity).hexdigest()


def build_reduced_checkpoint(
    states: Mapping[int, Any],
    detector_evidence: Sequence[TensorRedundancyEvidence] | None = None,
) -> ReducedCheckpoint:
    """Build a recoverable checkpoint from content or detector-confirmed equality."""
    if not states:
        raise ValueError("at least one rank state is required")

    evidence_by_path = {}
    component_by_location = {}
    evidence_roles = set()
    if detector_evidence is not None:
        graph = defaultdict(set)
        for item in detector_evidence:
            key = (item.rank, item.path)
            if key in evidence_by_path:
                raise ValueError(f"duplicate detector evidence for rank/path {key}")
            if item.rank not in states or item.local_index < 0:
                raise ValueError(f"invalid detector evidence source {key}")
            local = (item.rank, item.local_index)
            if local not in item.equivalent_locations:
                raise ValueError(f"detector evidence omits its local location {local}")
            evidence_by_path[key] = item
            evidence_roles.add(item.role)
            graph[local]
            for remote in item.equivalent_locations:
                if remote[0] not in states or remote[1] < 0:
                    raise ValueError(f"invalid detector evidence location {remote}")
                graph[local].add(remote)
                graph[remote].add(local)
        unseen = set(graph)
        while unseen:
            root = min(unseen)
            stack = [root]
            component = set()
            while stack:
                location = stack.pop()
                if location in component:
                    continue
                component.add(location)
                stack.extend(graph[location] - component)
            scope = min(component)
            for location in component:
                component_by_location[location] = scope
            unseen -= component

    tensors: dict[str, torch.Tensor] = {}
    records: dict[str, dict[str, Any]] = {}
    alias_ids: dict[tuple[Any, ...], str] = {}
    alias_values: dict[str, str] = {}
    storage_views: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    component_signatures: dict[tuple[int, int], tuple[Any, ...]] = {}
    used_evidence = set()
    full_logical_bytes = 0

    def reduce_value(current: Any, rank: int, components: tuple[Any, ...]) -> Any:
        nonlocal full_logical_bytes
        if isinstance(current, torch.Tensor):
            checksum = tensor_checksum(current)
            role = str(components[0]) if components else "root"
            path = checkpoint_path(components)
            detector_scope = None
            if detector_evidence is not None:
                evidence = evidence_by_path.get((rank, path))
                if evidence is None:
                    if role in evidence_roles:
                        raise ValueError(
                            f"detector evidence is missing rank {rank} tensor {path}"
                        )
                    detector_scope = (rank, -1)
                else:
                    if evidence.role != role:
                        raise ValueError(f"detector role mismatch for rank {rank} {path}")
                    used_evidence.add((rank, path))
                    location = (rank, evidence.local_index)
                    detector_scope = component_by_location[location]
                    signature = (
                        role,
                        str(current.dtype),
                        tuple(current.shape),
                        checksum,
                    )
                    previous_signature = component_signatures.setdefault(
                        detector_scope, signature
                    )
                    if previous_signature != signature:
                        raise ValueError(
                            f"detector component {detector_scope} contains unequal tensors"
                        )
            value_id = _value_id(role, current, checksum, detector_scope)
            logical_bytes = current.numel() * current.element_size()
            storage = current.untyped_storage()
            storage_key = (
                rank,
                str(current.device),
                storage.data_ptr() if storage.nbytes() else id(current),
                storage.nbytes(),
            )
            view = (
                str(current.dtype),
                current.storage_offset(),
                tuple(current.shape),
                tuple(current.stride()),
            )
            previous_view = storage_views.setdefault(storage_key, view)
            if previous_view != view:
                raise ValueError("shared tensor storage with distinct views is unsupported")
            alias_id = alias_ids.setdefault(
                storage_key + view, f"rank:{rank}:alias:{len(alias_ids)}"
            )
            previous_value = alias_values.setdefault(alias_id, value_id)
            if previous_value != value_id:
                raise ValueError(f"alias {alias_id} refers to unequal tensor values")
            full_logical_bytes += logical_bytes
            tensors.setdefault(value_id, current.detach().cpu().contiguous().clone())
            record = records.setdefault(
                value_id,
                {
                    "role": role,
                    "dtype": str(current.dtype),
                    "shape": tuple(current.shape),
                    "logical_bytes": logical_bytes,
                    "checksum": checksum,
                    "source_ranks": set(),
                    "requirements": [],
                },
            )
            record["source_ranks"].add(rank)
            record["requirements"].append(
                TensorRequirement(
                    rank=rank,
                    path=path,
                    object_id=checkpoint_object_id(path),
                    alias_id=alias_id,
                )
            )
            return TensorReference(value_id, alias_id)
        if isinstance(current, defaultdict):
            result = defaultdict(current.default_factory)
            result.update(
                (key, reduce_value(item, rank, components + (key,)))
                for key, item in current.items()
            )
            return result
        if isinstance(current, Mapping):
            items = [
                (key, reduce_value(item, rank, components + (key,)))
                for key, item in current.items()
            ]
            try:
                return current.__class__(items)
            except TypeError:
                return dict(items)
        if isinstance(current, list):
            return [
                reduce_value(item, rank, components + (index,))
                for index, item in enumerate(current)
            ]
        if isinstance(current, tuple):
            items = [
                reduce_value(item, rank, components + (index,))
                for index, item in enumerate(current)
            ]
            if hasattr(current, "_fields"):
                return current.__class__(*items)
            return tuple(items)
        return current

    templates = {
        int(rank): reduce_value(state, int(rank), ())
        for rank, state in sorted(states.items())
    }
    if detector_evidence is not None and used_evidence != set(evidence_by_path):
        unused = sorted(set(evidence_by_path) - used_evidence)
        raise ValueError(f"detector evidence does not map to checkpoint tensors: {unused}")
    objects = tuple(
        ReducedObject(
            value_id=value_id,
            role=record["role"],
            dtype=record["dtype"],
            shape=record["shape"],
            logical_bytes=record["logical_bytes"],
            checksum=record["checksum"],
            source_ranks=tuple(sorted(record["source_ranks"])),
            requirements=tuple(
                sorted(record["requirements"], key=lambda item: (item.rank, item.path))
            ),
        )
        for value_id, record in sorted(records.items())
    )
    reduced = ReducedCheckpoint(
        templates=templates,
        tensors=tensors,
        objects=objects,
        full_logical_bytes=full_logical_bytes,
    )
    reduced.validate()
    return reduced


def plan_replication_cost(
    reduced: ReducedCheckpoint,
    rank_to_domain: Mapping[int, str],
    target_domains: Sequence[str],
    *,
    replica_count: int,
    tolerated_failures: int,
    path_cost: Mapping[tuple[str, str], float] | None = None,
) -> ReplicationCost:
    """Choose deterministic per-object service domains for an independent cost model."""
    domains = tuple(sorted(set(target_domains)))
    if replica_count <= 0:
        raise ValueError("replica_count must be positive")
    if tolerated_failures < 0:
        raise ValueError("tolerated_failures must be non-negative")
    if replica_count <= tolerated_failures:
        raise ValueError("replica_count must exceed tolerated_failures")
    if len(domains) < replica_count:
        raise ValueError(
            f"need {replica_count} target domains, only {len(domains)} are available"
        )

    costs = path_cost or {}
    invalid_costs = [
        edge
        for edge, value in costs.items()
        if not math.isfinite(float(value)) or float(value) < 0
    ]
    if invalid_costs:
        raise ValueError(f"path costs must be finite and non-negative: {invalid_costs}")
    placements = []
    all_transfers = []
    for item in reduced.objects:
        missing_ranks = [rank for rank in item.source_ranks if rank not in rank_to_domain]
        if missing_ranks:
            raise ValueError(f"missing failure domain for source ranks {missing_ranks}")

        candidates = []
        for target in domains:
            source_rank, source_domain, unit_cost = min(
                (
                    (
                        rank,
                        rank_to_domain[rank],
                        float(
                            costs.get(
                                (rank_to_domain[rank], target),
                                0.0 if rank_to_domain[rank] == target else 1.0,
                            )
                        ),
                    )
                    for rank in item.source_ranks
                ),
                key=lambda choice: (choice[2], choice[0], choice[1]),
            )
            candidates.append((unit_cost, target, source_rank, source_domain))

        selected = sorted(candidates)[:replica_count]
        transfers = tuple(
            ProtectionTransfer(
                value_id=item.value_id,
                source_rank=source_rank,
                source_domain=source_domain,
                target_domain=target,
                logical_bytes=item.logical_bytes,
                wire_bytes=(
                    0 if source_domain == target else item.logical_bytes
                ),
                weighted_cost=unit_cost * item.logical_bytes,
            )
            for unit_cost, target, source_rank, source_domain in selected
        )
        placement = ProtectionPlacement(
            value_id=item.value_id,
            target_domains=tuple(sorted(transfer.target_domain for transfer in transfers)),
            transfers=transfers,
        )
        placements.append(placement)
        all_transfers.extend(transfers)

    result = ReplicationCost(
        replica_count=replica_count,
        tolerated_failures=tolerated_failures,
        full_replication_bytes=reduced.full_logical_bytes * replica_count,
        reduced_replication_bytes=sum(
            transfer.logical_bytes for transfer in all_transfers
        ),
        cross_domain_bytes=sum(transfer.wire_bytes for transfer in all_transfers),
        weighted_cost=sum(transfer.weighted_cost for transfer in all_transfers),
        placements=tuple(placements),
    )
    result.validate_failure_budget()
    return result
