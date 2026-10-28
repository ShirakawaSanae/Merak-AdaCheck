"""Post-allreduce DP evidence for the SemanticFixed gradient encoder."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

import torch
import torch.distributed as dist

from .object_semantic_plan import (
    ActiveGradientPlan,
    ConfirmedGradientEncoding,
    confirm_gradient_checksums,
)


def verifier_object_ids_by_rank(
    plan: ActiveGradientPlan,
) -> dict[int, tuple[str, ...]]:
    """Choose one verifier per complete PP/TP-local DP set."""
    bindings = {item.rank: item for item in plan.rank_bindings}
    publishers = {
        item.object_id: {
            placement.publisher_rank
            for placement in plan.placements
            if placement.object_id == item.object_id
        }
        for item in plan.objects
    }
    loads = {rank: 0 for rank in bindings}
    selected = {rank: [] for rank in bindings}
    for item in sorted(plan.objects, key=lambda value: (-value.logical_bytes, value.object_id)):
        by_coordinate: dict[tuple[int, int], list[int]] = {}
        for rank in item.holders:
            binding = bindings[rank]
            if binding.pipeline_rank != item.pipeline_rank:
                raise ValueError("semantic object crosses pipeline coordinates")
            by_coordinate.setdefault(
                (binding.pipeline_rank, binding.tensor_rank), []
            ).append(rank)
        if tuple(sorted(tensor for _, tensor in by_coordinate)) != item.tensor_ranks:
            raise ValueError("semantic object tensor coordinates changed")

        for coordinate, holders in sorted(by_coordinate.items()):
            complete = {
                rank
                for rank, binding in bindings.items()
                if (binding.pipeline_rank, binding.tensor_rank) == coordinate
            }
            candidates = set(holders)
            # Only a complete post-allreduce DP set may omit duplicate hashes.
            if candidates != complete:
                verifiers = sorted(candidates)
            else:
                preferred = candidates & publishers[item.object_id]
                verifier = min(
                    preferred or candidates,
                    key=lambda rank: (loads[rank], rank),
                )
                verifiers = [verifier]
            for rank in verifiers:
                selected[rank].append(item.object_id)
                loads[rank] += item.logical_bytes
    return {rank: tuple(sorted(object_ids)) for rank, object_ids in selected.items()}


def staged_object_ids_by_rank(
    plan: ActiveGradientPlan,
) -> dict[int, tuple[str, ...]]:
    """Return the union of objects a rank must publish or verify."""
    verifiers = verifier_object_ids_by_rank(plan)
    published = {rank: set() for rank in verifiers}
    for placement in plan.placements:
        published[placement.publisher_rank].add(placement.object_id)
    return {
        rank: tuple(sorted(set(verifiers[rank]) | published[rank]))
        for rank in verifiers
    }


def _tensor_checksum(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(value)).hexdigest()


def gather_dp_invariant_gradient_encoding(
    plan: ActiveGradientPlan,
    local_tensors: Mapping[str, torch.Tensor],
    *,
    epoch: int,
    group: object | None = None,
) -> ConfirmedGradientEncoding:
    """Collect one integrity checksum per post-allreduce DP object."""
    if group is None or not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("DP-invariant confirmation requires an initialized group")
    if dist.get_backend(group) != "gloo":
        raise ValueError("DP-invariant confirmation requires a Gloo group")
    rank, world_size = dist.get_rank(group), dist.get_world_size(group)
    if rank != dist.get_rank() or world_size != len(plan.rank_bindings):
        raise ValueError("DP-invariant confirmation requires the world-size group")

    assignments = verifier_object_ids_by_rank(plan)
    expected = set(assignments[rank])
    if set(local_tensors) != expected:
        raise ValueError("local tensors do not match the DP verifier assignment")
    objects = {item.object_id: item for item in plan.objects}
    local = []
    for object_id in sorted(expected):
        item, tensor = objects[object_id], local_tensors[object_id]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tuple(tensor.shape) != item.lengths
            or str(tensor.dtype) != item.dtype
        ):
            raise ValueError("DP verifier tensor differs from its semantic object")
        local.append(
            {"rank": rank, "object_id": object_id, "checksum": _tensor_checksum(tensor)}
        )
    json.dumps(local, sort_keys=True)
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, local, group=group)

    observed: dict[str, list[str]] = {item.object_id: [] for item in plan.objects}
    for source_rank, values in enumerate(gathered):
        if not isinstance(values, list):
            raise ValueError("gathered DP verifier evidence must be a JSON array")
        expected_ids = set(assignments[source_rank])
        if {
            value.get("object_id")
            for value in values
            if isinstance(value, dict)
        } != expected_ids:
            raise ValueError("gathered DP verifier assignment changed")
        for value in values:
            if (
                not isinstance(value, dict)
                or set(value) != {"rank", "object_id", "checksum"}
                or value["rank"] != source_rank
                or not isinstance(value["object_id"], str)
                or not isinstance(value["checksum"], str)
            ):
                raise ValueError("gathered DP verifier evidence is invalid")
            observed[value["object_id"]].append(value["checksum"])

    expanded = {}
    for item in plan.objects:
        checksums = observed[item.object_id]
        if len(checksums) != 1:
            raise ValueError("DP object checksum evidence is incomplete")
        for holder in item.holders:
            expanded[(holder, item.object_id)] = checksums[0]
    return confirm_gradient_checksums(plan, expanded, epoch=epoch)
