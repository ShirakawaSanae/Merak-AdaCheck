"""Pure metadata adapter for MoE detector candidates and replica classes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

import torch

from Merak.core.checkpoint.catalog import (
    CheckpointObject,
    TensorRedundancyEvidence,
    iter_checkpoint_objects,
)
from Merak.core.checkpoint.reduced import DetectorEvidenceConfirmation


ReplicaClass = Literal["shared", "expert"]


@dataclass(frozen=True)
class MoERank:
    rank: int
    pipeline: int
    expert: int
    replica_dp: int
    tensor: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 0
            for value in (self.rank, self.pipeline, self.expert, self.replica_dp, self.tensor)
        ):
            raise ValueError("rank coordinates must be non-negative integers")


@dataclass(frozen=True)
class MoECandidate:
    rank: int
    detector_class: str
    replica_class: ReplicaClass
    role: str
    name: str
    shape: tuple[int, ...]
    offsets: tuple[int, ...]
    lengths: tuple[int, ...]
    dtype: str
    pipeline: int
    expert: int | None
    replica_dp: int
    tensor: int
    local_name: str = ""

    def __post_init__(self) -> None:
        if self.rank < 0 or self.pipeline < 0 or self.replica_dp < 0 or self.tensor < 0:
            raise ValueError("rank coordinates must be non-negative")
        if self.replica_class not in ("shared", "expert"):
            raise ValueError("replica_class must be 'shared' or 'expert'")
        if self.replica_class == "expert" and (type(self.expert) is not int or self.expert < 0):
            raise ValueError("expert candidates require a non-negative expert ID")
        if self.replica_class == "shared" and self.expert is not None:
            raise ValueError("shared candidates cannot have an expert ID")
        if not self.detector_class or not self.role or not self.name or not self.dtype:
            raise ValueError("candidate identity fields must be non-empty")
        if not (len(self.shape) == len(self.offsets) == len(self.lengths)):
            raise ValueError("candidate slice dimensions disagree")
        if any(type(size) is not int or size <= 0 for size in self.shape + self.lengths):
            raise ValueError("candidate shapes and lengths must be positive integers")
        if any(type(offset) is not int or offset < 0 for offset in self.offsets):
            raise ValueError("candidate offsets must be non-negative integers")
        if any(offset + length > size for offset, length, size in zip(self.offsets, self.lengths, self.shape)):
            raise ValueError("candidate slice lies outside its shape")

    @property
    def semantic_key(self) -> tuple[object, ...]:
        """Identity shared by ranks allowed to hold replicas of this value."""
        return (
            self.replica_class,
            self.role,
            self.shape,
            self.offsets,
            self.lengths,
            self.dtype,
            self.pipeline,
            self.expert,
            self.tensor,
        )


@dataclass(frozen=True)
class MoEDetectorGroups:
    """Topology-safe detector classes; these are metadata, not placements."""

    groups: tuple[tuple[MoECandidate, ...], ...]
    evidence: tuple[TensorRedundancyEvidence, ...]
    detector_confirmation: DetectorEvidenceConfirmation
    ambiguous_proposals_dropped: int = 0

    def model_aliases(self) -> dict[int, dict[str, str]]:
        """Map complete confirmed replica parameters to canonical model IDs."""
        aliases: dict[int, dict[str, str]] = {}
        for group in self.groups:
            if len(group) < 2:
                continue
            first = group[0]
            identity = (
                first.detector_class, first.replica_class, first.role,
                first.shape, first.dtype, first.pipeline, first.expert, first.tensor,
            )
            canonical = "moe/model/" + hashlib.sha256(
                json.dumps(identity, separators=(",", ":")).encode()
            ).hexdigest()
            for item in group:
                aliases.setdefault(item.rank, {})[item.local_name] = canonical
        return aliases


@dataclass(frozen=True)
class MoEOptimizerShard:
    rank: int
    parameter_id: str
    expert_id: int | None
    process_group: tuple[int, ...]
    group_index: int
    chunk_index: int
    parameter_offset: int
    chunk_offset: int
    length: int
    roles: tuple[str, ...]


def build_moe_optimizer_manifest(
    module: Any,
    optimizer: Any,
    *,
    rank: int,
    shared_group: Sequence[int],
    expert_group: Sequence[int],
) -> tuple[MoEOptimizerShard, ...]:
    """Bind role-aware ZeRO-1 ranges to parameters and their real holder groups."""
    from Merak.core.checkpoint.zero1_adapter import iter_zero1_shards

    shared = tuple(sorted(set(shared_group)))
    expert = tuple(sorted(set(expert_group)))
    if rank not in shared or rank not in expert:
        raise ValueError("rank must belong to both shared and expert optimizer groups")
    if len(shared) == 0 or len(expert) == 0:
        raise ValueError("optimizer process groups cannot be empty")

    parameters = dict(module.named_parameters())
    group_roles = tuple(getattr(optimizer, "group_roles", ()))
    if len(group_roles) != len(optimizer.fp16_groups):
        raise ValueError("role-aware ZeRO-1 group roles are incomplete")
    result = []
    raw_shards = iter_zero1_shards(module, optimizer)
    expected = {
        (
            item.parameter_id,
            item.group_index,
            item.local_chunk,
            item.parameter_offset,
            item.chunk_offset,
            item.length,
        )
        for item in raw_shards
    }
    manifest_rows = []
    for shard in raw_shards:
        parameter = parameters.get(shard.parameter_id)
        if parameter is None:
            raise ValueError(f"ZeRO-1 shard references unknown parameter {shard.parameter_id}")
        is_expert = bool(getattr(parameter, "_merak_expert_parallel", False))
        role = group_roles[shard.group_index]
        expected_role = "expert" if is_expert else "shared"
        if role != expected_role:
            raise ValueError("ZeRO-1 group role disagrees with parameter ownership")
        expert_id = getattr(parameter, "_merak_expert_id", None) if is_expert else None
        if is_expert and (type(expert_id) is not int or expert_id < 0):
            raise ValueError("expert parameter is missing a valid expert ID")
        manifest_rows.append(
            MoEOptimizerShard(
                rank=rank,
                parameter_id=shard.parameter_id,
                expert_id=expert_id,
                process_group=expert if is_expert else shared,
                group_index=shard.group_index,
                chunk_index=shard.local_chunk,
                parameter_offset=shard.parameter_offset,
                chunk_offset=shard.chunk_offset,
                length=shard.length,
                roles=shard.roles,
            )
        )
    actual = {
        (
            item.parameter_id,
            item.group_index,
            item.chunk_index,
            item.parameter_offset,
            item.chunk_offset,
            item.length,
        )
        for item in manifest_rows
    }
    if actual != expected:
        raise ValueError("MoE optimizer manifest does not exactly cover local ZeRO-1 ranges")
    result.extend(manifest_rows)
    if not result:
        raise ValueError("ZeRO-1 manifest contains no parameter fragments")
    return tuple(result)


def moe_detector_audit_report(groups: MoEDetectorGroups) -> dict[str, object]:
    """Return a stable audit summary without implying a placement decision."""
    shared_groups = [group for group in groups.groups if group[0].replica_class == "shared"]
    expert_groups = [group for group in groups.groups if group[0].replica_class == "expert"]
    payload = {
        "schema_version": 1,
        "detector": "Merak.RedundancyChecker.blake2s-56",
        "scope": "world-proposals+moe-semantic-filter+sha256-confirmation",
        "coverage": "positive detector proposals only; singleton negatives are not exported",
        "input_records": groups.detector_confirmation.input_records,
        "candidate_pairs": groups.detector_confirmation.candidate_pairs,
        "confirmed_pairs": groups.detector_confirmation.confirmed_pairs,
        "rejected_pairs": groups.detector_confirmation.rejected_pairs,
        "unmatched_locations": groups.detector_confirmation.unmatched_locations,
        "shared_groups": len(shared_groups),
        "shared_records": sum(map(len, shared_groups)),
        "expert_groups": len(expert_groups),
        "expert_records": sum(map(len, expert_groups)),
        "semantic_group_sizes": sorted(len(group) for group in groups.groups),
        "ambiguous_proposals_dropped": groups.ambiguous_proposals_dropped,
        "placement_applied": False,
    }
    payload["evidence_sha256"] = hashlib.sha256(
        json.dumps(
            [item.to_dict() for item in groups.evidence],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return payload


def capture_moe_adacheck_groups(module: Any, *, host_by_rank: Mapping[int, int]):
    """Capture real AdaCheck proposals and verify their MoE semantics."""
    import torch.distributed as dist

    from Merak.core.checkpoint.catalog import checkpoint_path
    from Merak.core.checkpoint.redundancy_checker import RedundancyChecker
    from Merak.core.moe.expert_parallel import ExpertParallelLayout

    if not dist.is_initialized():
        raise RuntimeError("MoE AdaCheck audit requires initialized distributed state")
    layout = getattr(module, "_merak_ep_layout", None)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if not isinstance(layout, ExpertParallelLayout) or layout.world_size != world_size:
        raise ValueError("MoE AdaCheck audit requires a matching expert-parallel layout")
    if set(host_by_rank) != set(range(world_size)):
        raise ValueError("MoE AdaCheck audit host map must cover the world")

    parameters = dict(module.named_parameters())
    model_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in parameters.items()
    }
    checker = RedundancyChecker(zero_stage=0)
    checker.comm_groups_for_redundancy_checking = False
    checker.add_model_state(model_state)
    checker.get_hash_values()
    checker.pack_to_tensor()
    checker.packed_tensor = checker.packed_tensor.to(torch.cuda.current_device())
    checker.do_redundancy_checking()
    local_evidence = checker.tensor_redundancy_evidence(rank)

    path_by_name = {
        name: checkpoint_path(("model", name)) for name in parameters
    }
    name_by_path = {path: name for name, path in path_by_name.items()}
    replica_class = {}
    expert_ids = {}
    for item in local_evidence:
        name = name_by_path.get(item.path)
        if name is None:
            raise ValueError(f"AdaCheck parameter path is unmapped: {item.path}")
        parameter = parameters[name]
        key = (rank, item.path)
        if bool(getattr(parameter, "_merak_expert_parallel", False)):
            expert_id = getattr(parameter, "_merak_expert_id", None)
            if type(expert_id) is not int or expert_id < 0:
                raise ValueError(f"expert parameter lacks global expert ID: {name}")
            replica_class[key] = "expert"
            expert_ids[key] = expert_id
        else:
            replica_class[key] = "shared"

    state_rows = [None] * world_size
    evidence_rows = [None] * world_size
    class_rows = [None] * world_size
    expert_rows = [None] * world_size
    dist.all_gather_object(state_rows, tuple(iter_checkpoint_objects({"model": model_state})))
    dist.all_gather_object(evidence_rows, local_evidence)
    dist.all_gather_object(class_rows, replica_class)
    dist.all_gather_object(expert_rows, expert_ids)
    evidence = tuple(item for row in evidence_rows for item in row)
    replica_classes = {key: value for row in class_rows for key, value in row.items()}
    global_expert_ids = {key: value for row in expert_rows for key, value in row.items()}
    ranks = tuple(
        MoERank(
            item,
            pipeline=layout.coordinates(item)["pp"],
            expert=layout.coordinates(item)["ep"],
            replica_dp=layout.coordinates(item)["dp"],
            tensor=layout.coordinates(item)["tp"],
        )
        for item in range(world_size)
    )
    path_map = {(item.rank, item.path): item.path for item in evidence}
    filtered = filter_moe_detector_evidence(
        ranks,
        evidence,
        objects_by_rank={item: value for item, value in enumerate(state_rows)},
        replica_class_by_path=replica_classes,
        expert_id_by_path=global_expert_ids,
        host_by_rank=dict(host_by_rank),
        state_path_by_detector_path=path_map,
    )
    return filtered


def capture_moe_adacheck_audit(module: Any, *, host_by_rank: Mapping[int, int]):
    """Capture an AdaCheck audit without applying a grouping policy."""
    return moe_detector_audit_report(
        capture_moe_adacheck_groups(module, host_by_rank=host_by_rank)
    )


def filter_moe_detector_evidence(
    ranks: Iterable[MoERank],
    evidence: Iterable[TensorRedundancyEvidence],
    *,
    values_by_rank: dict[int, dict[str, object]] | None = None,
    objects_by_rank: Mapping[int, Sequence[CheckpointObject]] | None = None,
    replica_class_by_path: dict[tuple[int, str], ReplicaClass],
    expert_id_by_path: dict[tuple[int, str], int],
    host_by_rank: dict[int, int],
    state_path_by_detector_path: dict[tuple[int, str], str] | None = None,
) -> MoEDetectorGroups:
    """Filter real AdaCheck proposals by MoE identity and exact tensor value.

    The rank-to-host map is validated independently and intentionally does not
    enter candidate identity. It is consumed later by failure-domain planning.
    """
    from Merak.core.checkpoint.reduced import confirm_detector_object_evidence

    rank_values = tuple(ranks)
    rank_by_id = {item.rank: item for item in rank_values}
    if not rank_values or len(rank_by_id) != len(rank_values):
        raise ValueError("ranks must contain non-negative unique rank IDs")
    if set(host_by_rank) != set(rank_by_id) or any(
        type(host) is not int or host < 0 for host in host_by_rank.values()
    ):
        raise ValueError("host map must cover every rank with non-negative host IDs")
    if objects_by_rank is None:
        objects_by_rank = {
            rank: tuple(iter_checkpoint_objects(value))
            for rank, value in (values_by_rank or {}).items()
        }
    if set(objects_by_rank) != set(rank_by_id):
        raise ValueError("checkpoint values must cover every topology rank")
    objects = {(rank, item.path): item
               for rank, rows in objects_by_rank.items() for item in rows}

    detector_proposals = tuple(evidence)
    proposal_paths = {(item.rank, item.path) for item in detector_proposals}
    if len(proposal_paths) != len(detector_proposals):
        raise ValueError("duplicate detector evidence for rank/path")
    if set(replica_class_by_path) != proposal_paths:
        raise ValueError("replica-class mapping must cover detector rank/path records exactly")
    if set(expert_id_by_path) - proposal_paths:
        raise ValueError("expert mapping contains an unknown detector rank/path")
    state_paths = (
        {key: key[1] for key in proposal_paths}
        if state_path_by_detector_path is None
        else state_path_by_detector_path
    )
    if set(state_paths) != proposal_paths or any(
        not isinstance(path, str) or not path for path in state_paths.values()
    ):
        raise ValueError("state-path mapping must cover detector records exactly")
    index_by_path = {
        (item.rank, state_paths[(item.rank, item.path)]): index
        for index, item in enumerate(sorted(detector_proposals, key=lambda row: (row.rank, row.path)))
    }
    proposal_component_by_record = {}
    proposal_records_by_location: dict[tuple[int, int], list[TensorRedundancyEvidence]] = {}
    ambiguous_proposal_components = set()
    for item in detector_proposals:
        local = (item.rank, item.local_index)
        proposal_records_by_location.setdefault(local, []).append(item)
    ambiguous_locations = {
        location
        for location, records in proposal_records_by_location.items()
        if len({state_paths[(record.rank, record.path)] for record in records}) > 1
    }
    ambiguous_proposal_components = {
        (item.rank, item.path)
        for item in detector_proposals
        if any(location in ambiguous_locations for location in item.equivalent_locations)
    }
    for item in detector_proposals:
        local_path = state_paths[(item.rank, item.path)]
        local_index = index_by_path[(item.rank, local_path)]
        locations = {(item.rank, local_index)}
        ambiguous = local in ambiguous_locations
        for remote_rank, remote_index in item.equivalent_locations:
            if (remote_rank, remote_index) in ambiguous_locations:
                ambiguous = True
                continue
            remote_records = proposal_records_by_location.get((remote_rank, remote_index), ())
            remote_paths = {
                state_paths[(record.rank, record.path)] for record in remote_records
            }
            if len(remote_paths) != 1:
                ambiguous = True
                continue
            remote_path = next(iter(remote_paths))
            remote_location = (remote_rank, index_by_path[(remote_rank, remote_path)])
            if remote_location[0] == item.rank and remote_location[1] != local_index:
                ambiguous = True
                continue
            locations.add(remote_location)
        proposal_component_by_record[(item.rank, item.path)] = tuple(sorted(locations))
        if ambiguous:
            ambiguous_proposal_components.add((item.rank, item.path))
    detector_by_local_path = {}
    local_proposals = []
    for item in detector_proposals:
        if (item.rank, item.path) in ambiguous_proposal_components:
            continue
        local_path = state_paths[(item.rank, item.path)]
        local_key = (item.rank, local_path)
        if local_key in detector_by_local_path:
            raise ValueError("detector paths alias the same local checkpoint tensor")
        detector_by_local_path[local_key] = item
        local_proposals.append(
            TensorRedundancyEvidence(
                rank=item.rank,
                role=item.role,
                path=local_path,
                local_index=index_by_path[(item.rank, local_path)],
                equivalent_locations=proposal_component_by_record[(item.rank, item.path)],
            )
        )
    confirmed_local, confirmation = confirm_detector_object_evidence(
        objects, tuple(local_proposals)
    )
    confirmed = tuple(
        TensorRedundancyEvidence(
            rank=item.rank,
            role=item.role,
            path=detector_by_local_path[(item.rank, item.path)].path,
            local_index=item.local_index,
            equivalent_locations=item.equivalent_locations,
        )
        for item in confirmed_local
    )

    grouped: dict[tuple[object, ...], list[MoECandidate]] = {}
    candidate_evidence: dict[tuple[int, str], TensorRedundancyEvidence] = {}
    for item in confirmed:
        key = (item.rank, item.path)
        replica_class = replica_class_by_path.get(key)
        if replica_class not in ("shared", "expert"):
            raise ValueError(f"missing MoE replica class for rank/path {key}")
        state_path = state_paths[key]
        if (item.rank, state_path) not in objects:
            raise ValueError(f"detector path does not identify a checkpoint tensor: {key}")
        checkpoint_object = objects[item.rank, state_path]
        if checkpoint_object.role != item.role:
            raise ValueError(f"detector role differs from state path: {key}")
        expert_id = None
        if replica_class == "expert":
            expert_id = expert_id_by_path.get(key)
            if type(expert_id) is not int or expert_id < 0:
                raise ValueError(f"missing expert ID for rank/path {key}")
        elif key in expert_id_by_path:
            raise ValueError(f"shared value cannot carry an expert ID: {key}")
        coordinate = rank_by_id[item.rank]
        candidate = MoECandidate(
            rank=item.rank,
            detector_class=repr(item.equivalent_locations),
            replica_class=replica_class,
            role=item.role,
            name=item.path,
            shape=tuple(checkpoint_object.shape),
            offsets=(0,) * len(checkpoint_object.shape),
            lengths=tuple(checkpoint_object.shape),
            dtype=checkpoint_object.dtype,
            pipeline=coordinate.pipeline,
            expert=expert_id,
            replica_dp=coordinate.replica_dp,
            tensor=coordinate.tensor,
            local_name=_path_parameter_name(state_path),
        )
        component = item.equivalent_locations
        grouped.setdefault((component, candidate.semantic_key), []).append(candidate)
        candidate_evidence[key] = TensorRedundancyEvidence(
            rank=item.rank,
            role=item.role,
            path=state_path,
            local_index=item.local_index,
            equivalent_locations=item.equivalent_locations,
        )

    semantic_groups = []
    ambiguous_components = set()
    for (component, _), candidates in sorted(grouped.items(), key=lambda item: repr(item[0])):
        proposal_is_ambiguous = any(
            (candidate.rank, candidate.name) in ambiguous_proposal_components
            for candidate in candidates
        )
        if not proposal_is_ambiguous and len({item.rank for item in candidates}) == len(candidates):
            semantic_groups.append(tuple(sorted(candidates, key=lambda item: item.rank)))
            continue

        ambiguous_components.add(component)
        semantic_groups.extend((candidate,) for candidate in candidates)

    groups = tuple(semantic_groups)
    filtered_evidence = []
    for group_index, group in enumerate(groups):
        locations = tuple((candidate.rank, group_index) for candidate in group)
        for candidate in group:
            original = candidate_evidence[(candidate.rank, candidate.name)]
            filtered_evidence.append(
                TensorRedundancyEvidence(
                    rank=original.rank,
                    role=original.role,
                    path=original.path,
                    local_index=group_index,
                    equivalent_locations=locations,
                )
            )

    return MoEDetectorGroups(
        groups,
        tuple(filtered_evidence),
        confirmation,
        ambiguous_proposals_dropped=len(ambiguous_proposal_components),
    )


def _lookup_checkpoint_tensor(state: object, path: str) -> torch.Tensor:
    value: object = state
    for component in path.strip("/").split("/"):
        type_name, raw = component.split(":", 1)
        key_text = raw.replace("~1", "/").replace("~0", "~")
        key: object = int(key_text) if type_name == "int" else key_text
        if isinstance(value, Mapping) and key in value:
            value = value[key]
        elif isinstance(value, (list, tuple)) and type(key) is int and 0 <= key < len(value):
            value = value[key]
        else:
            raise ValueError(f"checkpoint path is absent: {path}")
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"checkpoint path is not a tensor: {path}")
    return value


def _path_parameter_name(path: str) -> str:
    parts = []
    for component in path.strip("/").split("/"):
        type_name, raw = component.split(":", 1)
        value = raw.replace("~1", "/").replace("~0", "~")
        parts.append(value if type_name == "str" else f"{int(value)}")
    if len(parts) < 2 or parts[0] != "model":
        raise ValueError("detector model path has no local parameter name")
    return ".".join(parts[1:])
