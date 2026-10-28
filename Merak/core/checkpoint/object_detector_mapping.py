"""Map a frozen AdaCheck detector catalog to live gradient shards."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from .object_semantic_plan import GradientShard, RankBinding


_FP16_PATH = re.compile(
    r"^/str:optimizer/str:fp16_groups/int:(0|[1-9]\d*)/int:(0|[1-9]\d*)$"
)
_OBJECT_FIELDS = frozenset(
    {
        "value_id",
        "role",
        "dtype",
        "shape",
        "logical_bytes",
        "checksum",
        "source_ranks",
        "source_hosts",
        "requirements",
    }
)
_REQUIREMENT_FIELDS = frozenset({"rank", "path", "object_id", "alias_id"})
_CANONICAL_FIELDS = (
    "value_id",
    "role",
    "dtype",
    "shape",
    "logical_bytes",
    "checksum",
    "source_ranks",
    "requirements",
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class DetectorMappingRecord:
    group_index: int
    parameter_index: int
    fqn: str
    detector_class_id: str
    source: str


@dataclass(frozen=True)
class DetectorGradientMapping:
    catalog_hash: str
    mapping_hash: str
    detector_mapped: int
    rank_local_fallback: int
    records: tuple[DetectorMappingRecord, ...]
    shards: tuple[GradientShard, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "catalog_hash": self.catalog_hash,
            "mapping_hash": self.mapping_hash,
            "detector_mapped": self.detector_mapped,
            "rank_local_fallback": self.rank_local_fallback,
            "records": [record.__dict__ for record in self.records],
            "shards": [shard.__dict__ for shard in self.shards],
        }


def _load_catalog(
    path: Path, *, world_size: int, expected_catalog_hash: str
) -> tuple[list[dict[str, object]], dict[tuple[int, int, int], str], str]:
    if world_size < 1:
        raise ValueError("world_size must be positive")
    _sha256(expected_catalog_hash, "expected_catalog_hash")
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise ValueError(f"cannot read detector catalog: {path}") from error
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("detector catalog must be non-empty JSONL")

    objects: list[dict[str, object]] = []
    slots: dict[tuple[int, int, int], str] = {}
    requirement_keys: set[tuple[int, str]] = set()
    value_ids: set[str] = set()
    all_ranks: set[int] = set()
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("detector catalog contains invalid JSON") from error
        if not isinstance(item, dict) or set(item) != _OBJECT_FIELDS:
            raise ValueError("detector catalog object fields do not match schema")
        value_id = _sha256(item["value_id"], "value_id")
        _sha256(item["checksum"], "checksum")
        if (
            not isinstance(item["role"], str)
            or not item["role"]
            or not isinstance(item["dtype"], str)
            or not item["dtype"]
            or not isinstance(item["shape"], list)
            or any(type(size) is not int or size <= 0 for size in item["shape"])
            or type(item["logical_bytes"]) is not int
            or item["logical_bytes"] <= 0
            or not isinstance(item["source_hosts"], list)
            or not item["source_hosts"]
            or any(type(host) is not str or not host for host in item["source_hosts"])
            or item["source_hosts"] != sorted(set(item["source_hosts"]))
            or not isinstance(item["requirements"], list)
            or not item["requirements"]
        ):
            raise ValueError("detector catalog object has invalid metadata")
        requirements = item["requirements"]
        ranks = []
        previous_requirement = None
        for requirement in requirements:
            if not isinstance(requirement, dict) or set(requirement) != _REQUIREMENT_FIELDS:
                raise ValueError("detector requirement fields do not match schema")
            rank = requirement["rank"]
            path_value = requirement["path"]
            if (
                type(rank) is not int
                or not 0 <= rank < world_size
                or type(path_value) is not str
                or not path_value
                or type(requirement["object_id"]) is not str
                or not requirement["object_id"]
                or type(requirement["alias_id"]) is not str
                or not requirement["alias_id"]
            ):
                raise ValueError("detector requirement is invalid")
            ordering = (rank, path_value)
            if previous_requirement is not None and ordering < previous_requirement:
                raise ValueError("detector requirements are not canonical")
            previous_requirement = ordering
            if ordering in requirement_keys:
                raise ValueError("duplicate detector requirement")
            requirement_keys.add(ordering)
            ranks.append(rank)
            all_ranks.add(rank)
            match = _FP16_PATH.fullmatch(path_value)
            if item["role"] == "optimizer" and match:
                slot = (rank, int(match.group(1)), int(match.group(2)))
                if slot in slots:
                    raise ValueError("duplicate optimizer fp16_groups requirement")
                slots[slot] = value_id
        if item["source_ranks"] != sorted(set(ranks)):
            raise ValueError("source_ranks do not match requirements")
        if value_id in value_ids:
            raise ValueError("duplicate detector value_id")
        value_ids.add(value_id)
        objects.append(item)

    if [item["value_id"] for item in objects] != sorted(
        item["value_id"] for item in objects
    ):
        raise ValueError("detector catalog objects are not canonical")
    if all_ranks != set(range(world_size)):
        raise ValueError("detector catalog does not cover the declared world")
    full_logical_bytes = sum(
        int(item["logical_bytes"]) * len(item["requirements"]) for item in objects
    )
    canonical = {
        "full_logical_bytes": full_logical_bytes,
        "template_ranks": list(range(world_size)),
        "objects": [
            {key: item[key] for key in _CANONICAL_FIELDS} for item in objects
        ],
    }
    catalog_hash = _digest(canonical)
    if catalog_hash != expected_catalog_hash:
        raise ValueError("detector catalog hash mismatch")
    return objects, slots, catalog_hash


def _global_slice(
    parameter: torch.Tensor, *, tensor_rank: int, tensor_world_size: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shape = tuple(int(size) for size in parameter.shape)
    if not shape:
        raise ValueError("gradient parameter must have at least one dimension")
    if tensor_world_size < 1 or not 0 <= tensor_rank < tensor_world_size:
        raise ValueError("tensor rank is outside the declared world")
    if tensor_world_size == 1:
        return shape, (0,) * len(shape)
    if not bool(getattr(parameter, "model_parallel", False)):
        return shape, (0,) * len(shape)
    has_dimension = hasattr(parameter, "partition_dim")
    partition_stride = getattr(parameter, "partition_stride", None)
    legacy_stride = getattr(parameter, "__dict__", {}).get("stride")
    if not has_dimension:
        if partition_stride is not None or isinstance(legacy_stride, int):
            raise ValueError("partitioned parameter lacks TP metadata")
        # PipelineModule marks every stage-local parameter as model_parallel,
        # including replicated layer norms and row-parallel biases.
        return shape, (0,) * len(shape)
    stride = partition_stride if partition_stride is not None else legacy_stride
    if not isinstance(stride, int):
        raise ValueError("partitioned parameter lacks TP metadata")
    dimension = int(parameter.partition_dim)
    if stride != 1:
        raise ValueError("only contiguous tensor-parallel slices are supported")
    if not 0 <= dimension < len(shape):
        raise ValueError("partitioned parameter has an invalid dimension")
    global_shape = list(shape)
    global_shape[dimension] *= tensor_world_size
    offsets = [0] * len(shape)
    offsets[dimension] = tensor_rank * shape[dimension]
    return tuple(global_shape), tuple(offsets)


def map_detector_catalog_to_gradient_shards(
    catalog_path: str | Path,
    *,
    expected_catalog_hash: str,
    world_size: int,
    binding: RankBinding,
    parameter_ids: Sequence[Sequence[str]],
    parameter_groups: Sequence[Sequence[torch.Tensor]],
    tensor_parallel_rank: int,
    tensor_parallel_world_size: int,
    encoded_dtype: torch.dtype,
) -> DetectorGradientMapping:
    """Return fail-closed detector classes for one rank's live gradients."""
    _, detector_slots, catalog_hash = _load_catalog(
        Path(catalog_path),
        world_size=world_size,
        expected_catalog_hash=expected_catalog_hash,
    )
    if not 0 <= binding.rank < world_size:
        raise ValueError("rank binding is outside the detector world")
    if binding.tensor_rank != tensor_parallel_rank:
        raise ValueError("rank binding and tensor-parallel rank disagree")
    if not isinstance(encoded_dtype, torch.dtype):
        raise ValueError("encoded_dtype must be a torch dtype")
    if len(parameter_ids) != len(parameter_groups):
        raise ValueError("parameter id and model groups disagree")
    for group_index, (ids, parameters) in enumerate(
        zip(parameter_ids, parameter_groups)
    ):
        if len(ids) != len(parameters):
            raise ValueError("parameter id and model group lengths disagree")
    for rank, group_index, parameter_index in sorted(detector_slots):
        if rank != binding.rank:
            continue
        if (
            group_index >= len(parameter_groups)
            or parameter_index >= len(parameter_groups[group_index])
        ):
            raise ValueError("detector optimizer slot is outside live parameter groups")

    records = []
    shards = []
    seen_fqns = set()
    mapped = 0
    fallback = 0
    for group_index, (ids, parameters) in enumerate(zip(parameter_ids, parameter_groups)):
        for parameter_index, (fqn, parameter) in enumerate(zip(ids, parameters)):
            if not isinstance(fqn, str) or not fqn or fqn in seen_fqns:
                raise ValueError("parameter FQNs must be non-empty and unique")
            if not isinstance(parameter, torch.Tensor):
                raise ValueError("model parameter groups must contain tensors")
            seen_fqns.add(fqn)
            global_shape, offsets = _global_slice(
                parameter,
                tensor_rank=tensor_parallel_rank,
                tensor_world_size=tensor_parallel_world_size,
            )
            slot = (binding.rank, group_index, parameter_index)
            detector_class = detector_slots.get(slot)
            source = "detector"
            if detector_class is None:
                source = "rank_local_fallback"
                detector_class = _digest(
                    {
                        "kind": source,
                        "catalog_hash": catalog_hash,
                        "rank": binding.rank,
                        "group_index": group_index,
                        "parameter_index": parameter_index,
                        "fqn": fqn,
                        "global_shape": global_shape,
                        "offsets": offsets,
                        "lengths": tuple(parameter.shape),
                        "dtype": str(encoded_dtype),
                    }
                )
                fallback += 1
            else:
                mapped += 1
            records.append(
                DetectorMappingRecord(
                    group_index,
                    parameter_index,
                    fqn,
                    detector_class,
                    source,
                )
            )
            shards.append(
                GradientShard(
                    binding.rank,
                    fqn,
                    detector_class,
                    global_shape,
                    offsets,
                    tuple(int(size) for size in parameter.shape),
                    str(encoded_dtype),
                )
            )
    evidence = {
        "catalog_hash": catalog_hash,
        "rank": binding.rank,
        "tensor_parallel_rank": tensor_parallel_rank,
        "tensor_parallel_world_size": tensor_parallel_world_size,
        "encoded_dtype": str(encoded_dtype),
        "records": [record.__dict__ for record in records],
        "shards": [shard.__dict__ for shard in shards],
    }
    return DetectorGradientMapping(
        catalog_hash,
        _digest(evidence),
        mapped,
        fallback,
        tuple(records),
        tuple(shards),
    )
