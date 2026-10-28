"""CPU/Gloo metadata gate for semantic gradient objects."""

from __future__ import annotations

from dataclasses import asdict
from typing import Mapping, Sequence

import torch
import torch.distributed as dist

from .catalog import tensor_checksum
from .object_semantic_plan import (
    ActiveGradientPlan,
    Arm,
    ConfirmedGradientEncoding,
    GradientShard,
    RankBinding,
    build_active_gradient_plan,
    confirm_gradient_checksums,
)


_BINDING_FIELDS = frozenset(
    ("rank", "pipeline_rank", "tensor_rank", "data_rank", "host")
)
_SHARD_FIELDS = frozenset(
    (
        "rank",
        "fqn",
        "detector_class_id",
        "global_shape",
        "offsets",
        "lengths",
        "dtype",
    )
)
_CHECKSUM_FIELDS = frozenset(("rank", "object_id", "checksum"))


def _require_gloo(group: object | None) -> tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("semantic object gate requires an initialized process group")
    if dist.get_backend(group) != "gloo":
        raise RuntimeError("semantic object gate requires the gloo backend")
    rank, world_size = dist.get_rank(group), dist.get_world_size(group)
    if rank != dist.get_rank() or world_size != dist.get_world_size():
        raise RuntimeError("semantic object gate requires a world-size gloo group")
    return rank, world_size


def _json_like(value: object) -> bool:
    if value is None or type(value) in (bool, int, float, str):
        return True
    if isinstance(value, list):
        return all(_json_like(item) for item in value)
    if isinstance(value, dict):
        return all(type(key) is str and _json_like(item) for key, item in value.items())
    return False


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _binding(value: object) -> RankBinding:
    if not isinstance(value, dict) or set(value) != _BINDING_FIELDS:
        raise ValueError("rank binding metadata schema changed")
    return RankBinding(**{name: _integer(value[name], name) for name in _BINDING_FIELDS})


def _shape(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"gradient shard {name} must be a JSON array")
    return tuple(_integer(item, name) for item in value)


def _shard(value: object) -> GradientShard:
    if not isinstance(value, dict) or set(value) != _SHARD_FIELDS:
        raise ValueError("gradient shard metadata schema changed")
    for name in ("fqn", "detector_class_id", "dtype"):
        if not isinstance(value[name], str):
            raise ValueError(f"gradient shard {name} must be a string")
    return GradientShard(
        _integer(value["rank"], "rank"),
        value["fqn"],
        value["detector_class_id"],
        _shape(value["global_shape"], "global_shape"),
        _shape(value["offsets"], "offsets"),
        _shape(value["lengths"], "lengths"),
        value["dtype"],
    )


def gather_active_gradient_plan(
    binding: RankBinding,
    shards: Sequence[GradientShard],
    *,
    arm: Arm,
    detector_catalog_hash: str,
    semantic_plan_hash: str,
    fixed_pairs: Sequence[Sequence[int]] = (),
    group: object | None = None,
) -> ActiveGradientPlan:
    """Gather only static rank/shard metadata and build the canonical plan."""
    rank, world_size = _require_gloo(group)
    if binding.rank != rank:
        raise ValueError("local rank binding disagrees with the process group")
    if any(shard.rank != rank for shard in shards):
        raise ValueError("a rank may submit only its own gradient shard metadata")
    local = {
        "binding": asdict(binding),
        "shards": [
            {
                **asdict(shard),
                "global_shape": list(shard.global_shape),
                "offsets": list(shard.offsets),
                "lengths": list(shard.lengths),
            }
            for shard in shards
        ],
    }
    if not _json_like(local):
        raise ValueError("static semantic metadata must be JSON-like")
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, local, group=group)

    bindings = []
    all_shards = []
    for source_rank, value in enumerate(gathered):
        if not _json_like(value) or not isinstance(value, dict) or set(value) != {
            "binding",
            "shards",
        }:
            raise ValueError("gathered static semantic metadata schema changed")
        remote_binding = _binding(value["binding"])
        if remote_binding.rank != source_rank:
            raise ValueError("gathered rank binding disagrees with its sender")
        if not isinstance(value["shards"], list):
            raise ValueError("gathered gradient shards must be a JSON array")
        remote_shards = tuple(_shard(item) for item in value["shards"])
        if any(shard.rank != source_rank for shard in remote_shards):
            raise ValueError("gathered gradient shard disagrees with its sender")
        bindings.append(remote_binding)
        all_shards.extend(remote_shards)
    return build_active_gradient_plan(
        tuple(all_shards),
        tuple(bindings),
        arm=arm,
        detector_catalog_hash=detector_catalog_hash,
        semantic_plan_hash=semantic_plan_hash,
        fixed_pairs=fixed_pairs,
    )


def gather_confirmed_gradient_encoding(
    plan: ActiveGradientPlan,
    local_tensors: Mapping[str, torch.Tensor],
    *,
    epoch: int,
    group: object | None = None,
) -> ConfirmedGradientEncoding:
    """Gather compact local checksums, never gradient tensor payloads."""
    rank, world_size = _require_gloo(group)
    if len(plan.rank_bindings) != world_size:
        raise ValueError("semantic plan world size differs from the process group")
    expected = {item.object_id: item for item in plan.objects if rank in item.holders}
    if set(local_tensors) != set(expected):
        raise ValueError("local gradient tensors do not cover the planned candidates")
    local = []
    for object_id, item in sorted(expected.items()):
        tensor = local_tensors[object_id]
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != item.lengths
            or str(tensor.dtype) != item.dtype
        ):
            raise ValueError("local gradient tensor differs from its planned CPU schema")
        local.append(
            {"rank": rank, "object_id": object_id, "checksum": tensor_checksum(tensor)}
        )
    if not _json_like(local):
        raise ValueError("gradient checksum metadata must be JSON-like")
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, local, group=group)

    checksums = {}
    for source_rank, values in enumerate(gathered):
        if not _json_like(values) or not isinstance(values, list):
            raise ValueError("gathered gradient checksums must be a JSON array")
        for value in values:
            if not isinstance(value, dict) or set(value) != _CHECKSUM_FIELDS:
                raise ValueError("gathered gradient checksum metadata schema changed")
            if _integer(value["rank"], "rank") != source_rank:
                raise ValueError("gathered gradient checksum disagrees with its sender")
            object_id, checksum = value["object_id"], value["checksum"]
            if not isinstance(object_id, str) or not isinstance(checksum, str):
                raise ValueError("gathered gradient checksum metadata is invalid")
            key = (source_rank, object_id)
            if key in checksums:
                raise ValueError("duplicate gathered gradient checksum")
            checksums[key] = checksum
    return confirm_gradient_checksums(plan, checksums, epoch=epoch)
