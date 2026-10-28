"""Canonical metadata for Merak's ZeRO-1 optimizer shards.

The existing SemanticFixed object path consumes one full FP32 master tensor per
model parameter.  Merak's ZeRO-1 wrapper instead owns flattened local ranges.
This adapter exposes those ranges without pretending they are already
SemanticFixed-compatible; a later runtime can use the manifest to publish and
restore exact shards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .object_recovery import ObjectAddress, ObjectFragment, SliceRequest
from .object_recovery_adapter import _address, _named_parameters


@dataclass(frozen=True)
class Zero1Shard:
    parameter_id: str
    group_index: int
    parameter_offset: int
    length: int
    local_chunk: int
    roles: tuple[str, ...]
    chunk_offset: int = 0

    @property
    def semantic_id(self) -> str:
        return f"{self.parameter_id}::zero1[{self.parameter_offset}:{self.parameter_offset + self.length}]"


def _zero1_steps(optimizer: Any) -> dict[torch.Tensor, torch.Tensor]:
    result = {}
    for chunks in optimizer.local_sub_partitions_of_fp32_groups:
        for chunk_index, chunk in enumerate(chunks):
            state = optimizer.optimizer.state.get(chunk, {})
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("ZeRO-1 AdamW moments are incomplete")
            step = state["step"]
            if not isinstance(step, torch.Tensor) or step.dtype != torch.float32 or step.numel() != 1:
                raise ValueError("ZeRO-1 AdamW step must be one float32 scalar")
            result[chunk] = step
    return result


def _uniform_parameter_steps(
    shards: tuple[Zero1Shard, ...],
    steps: Mapping[torch.Tensor, torch.Tensor],
    optimizer: Any,
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for shard in shards:
        chunk = optimizer.local_sub_partitions_of_fp32_groups[shard.group_index][shard.local_chunk]
        value = steps[chunk].detach().cpu().reshape(1)
        previous = result.setdefault(shard.parameter_id, value)
        if not torch.equal(previous, value):
            raise ValueError("ZeRO-1 parameter spans chunks with unequal AdamW steps")
    return result


def capture_zero1_step_fragments(
    module: Any,
    optimizer: Any,
    *,
    rank: int,
    storage_host: int,
    epoch: int,
    manifest: tuple[Any, ...] | None = None,
) -> tuple[ObjectFragment, ...]:
    """Capture AdamW's rank-local scalar clock for every local chunk."""
    if bool(getattr(module, "_merak_semantic_zero1", False)):
        if manifest is None or any(item.rank != rank for item in manifest):
            raise ValueError("semantic ZeRO-1 requires its validated rank manifest")
        values = _role_aware_chunk_steps(optimizer, rank)
    elif bool(getattr(module, "_merak_role_aware_zero1", False)):
        values = _role_aware_chunk_steps(optimizer, rank)
    else:
        values = _uniform_parameter_steps(
            iter_zero1_shards(module, optimizer), _zero1_steps(optimizer), optimizer
        )
    return tuple(
        ObjectFragment.from_tensor(
            ObjectAddress(
                f"{_step_fragment_id(name, module, rank)}::optimizer.step",
                "optimizer.step", (1,), "torch.float32",
            ),
            epoch=epoch, offsets=(0,), tensor=value,
            source_rank=rank, storage_host=storage_host,
        )
        for name, value in sorted(values.items())
    )


def _role_aware_chunk_steps(optimizer: Any, rank: int) -> dict[str, torch.Tensor]:
    chunk_steps = _zero1_steps(optimizer)
    return {
        f"zero1@rank={rank}/group={group}/chunk={index}": chunk_steps[chunk].detach().cpu().reshape(1)
        for group, chunks in enumerate(optimizer.local_sub_partitions_of_fp32_groups)
        if optimizer.fp16_groups[group]
        for index, chunk in enumerate(chunks)
    }


def _step_fragment_id(name: str, module: Any, rank: int) -> str:
    if name.startswith("zero1@rank="):
        return name
    return _optimizer_parameter_id(name, module, rank)


def _parameter_ids(module: Any, optimizer: Any) -> tuple[tuple[str, ...], ...]:
    aliases: dict[int, list[str]] = {}
    for name, parameter in module.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    result = []
    seen: set[str] = set()
    for group_index, group in enumerate(optimizer.fp16_groups):
        ids = []
        for parameter in group:
            names = aliases.get(id(parameter), ())
            if len(names) != 1:
                raise ValueError("ZeRO-1 parameter lacks one canonical model name")
            name = names[0]
            if name in seen:
                raise ValueError(f"duplicate ZeRO-1 parameter identity: {name}")
            seen.add(name)
            ids.append(name)
        result.append(tuple(ids))
    if hasattr(optimizer, "group_roles") and len(result) != len(optimizer.group_roles):
        raise ValueError("ZeRO-1 optimizer role metadata is incomplete")
    return tuple(result)


def _model_parameter_id(name: str, parameter: torch.Tensor) -> str:
    if bool(getattr(parameter, "_merak_expert_parallel", False)):
        return f"{name}@expert={int(parameter._merak_expert_id)}"
    return name


def _model_object_id(
    name: str,
    module: Any,
    rank: int,
    parameter: torch.Tensor | None = None,
    *,
    role: str = "model",
) -> str:
    if bool(getattr(module, "_merak_semantic_zero1", False)):
        return name
    if parameter is not None:
        name = _model_parameter_id(name, parameter)
    if bool(getattr(module, "_merak_role_aware_zero1", False)):
        return f"{name}@rank={rank}"
    return name


def _buffer_object_id(name: str, module: Any, rank: int) -> str:
    if bool(getattr(module, "_merak_semantic_zero1", False)):
        return f"{name}@rank={rank}"
    return _model_object_id(name, module, rank)


def persistent_model_buffers(module: Any):
    non_persistent = {
        f"{prefix}.{name}" if prefix else name
        for prefix, child in module.named_modules()
        for name in child._non_persistent_buffers_set
    }
    return (
        (name, tensor)
        for name, tensor in module.named_buffers()
        if name not in non_persistent
    )


def _optimizer_parameter_id(name: str, module: Any, rank: int) -> str:
    if bool(getattr(module, "_merak_role_aware_zero1", False)) and not bool(
        getattr(module, "_merak_semantic_zero1", False)
    ):
        return f"{name}@rank={rank}"
    return name


def _optimizer_model_id(name: str, module: Any, rank: int, parameter: torch.Tensor) -> str:
    return _optimizer_parameter_id(_model_parameter_id(name, parameter), module, rank)


def iter_zero1_shards(module: Any, optimizer: Any) -> tuple[Zero1Shard, ...]:
    """Enumerate every local ZeRO-1 master/moment range.

    This function reads only the public bookkeeping retained by
    ``ZeroOptimizer_Stage1``.  It does not allocate or mutate optimizer state.
    """
    required = (
        "fp16_groups",
        "params_in_rank_sub_partitions",
        "params_in_rank_sub_partitions_offsets",
        "local_sub_partitions_of_fp32_groups",
    )
    missing = [name for name in required if not hasattr(optimizer, name)]
    if missing:
        raise TypeError(f"not a Merak ZeRO-1 optimizer; missing {', '.join(missing)}")
    parameter_ids = _parameter_ids(module, optimizer)
    result: list[Zero1Shard] = []
    if bool(getattr(module, "_merak_role_aware_zero1", False)):
        if not hasattr(optimizer, "zero1_shard_ranges"):
            raise TypeError("role-aware ZeRO-1 lacks explicit local shard ranges")
        for group_index, (group, identifiers) in enumerate(zip(optimizer.fp16_groups, parameter_ids)):
            if not group:
                continue
            ranges = optimizer.zero1_shard_ranges[group_index]
            chunks = optimizer.local_sub_partitions_of_fp32_groups[group_index]
            if len(ranges) != len(chunks):
                raise ValueError("role-aware ZeRO-1 shard range count differs from local chunks")
            for chunk_index, chunk_ranges in enumerate(ranges):
                cursor = 0
                for parameter, parameter_offset, length, chunk_offset in chunk_ranges:
                    parameter_index = next(
                        (index for index, value in enumerate(group) if value is parameter),
                        None,
                    )
                    if parameter_index is None:
                        raise ValueError("role-aware ZeRO-1 range references an unknown parameter")
                    if chunk_offset != cursor or length <= 0 or chunk_offset + length > chunks[chunk_index].numel():
                        raise ValueError("role-aware ZeRO-1 chunk ranges are invalid")
                    result.append(Zero1Shard(
                        identifiers[parameter_index], group_index, parameter_offset, length,
                        chunk_index,
                        ("optimizer.master", "optimizer.exp_avg", "optimizer.exp_avg_sq"),
                        chunk_offset,
                    ))
                    cursor += length
        return tuple(result)
    if (
        len(optimizer.fp16_groups) != len(optimizer.params_in_rank_sub_partitions)
        or len(optimizer.fp16_groups) != len(optimizer.params_in_rank_sub_partitions_offsets)
        or len(optimizer.fp16_groups) != len(optimizer.local_sub_partitions_of_fp32_groups)
    ):
        raise ValueError("ZeRO-1 optimizer group metadata is incomplete")
    for group_index, (group, identifiers) in enumerate(zip(optimizer.fp16_groups, parameter_ids)):
        params_by_rank = optimizer.params_in_rank_sub_partitions[group_index]
        offsets_by_rank = optimizer.params_in_rank_sub_partitions_offsets[group_index]
        local_rank = getattr(optimizer, "local_rank", None)
        if local_rank is None and hasattr(optimizer, "local_ranks"):
            local_rank = optimizer.local_ranks[group_index]
        if local_rank is None:
            process_groups = getattr(optimizer, "dp_process_groups", None)
            process_group = (
                process_groups[group_index] if process_groups is not None
                else getattr(optimizer, "dp_process_group", None)
            )
            if process_group is None or not torch.distributed.is_initialized():
                raise ValueError("ZeRO-1 requires an explicit DP rank or process group")
            local_rank = torch.distributed.get_rank(group=process_group)
        if type(local_rank) is not int or not 0 <= local_rank < len(params_by_rank):
            raise ValueError("ZeRO-1 local rank is outside partition metadata")
        chunks = optimizer.local_sub_partitions_of_fp32_groups[group_index]
        parameters = params_by_rank[local_rank]
        offsets = offsets_by_rank[local_rank]
        if len(parameters) != len(offsets) or len(parameters) != len(chunks):
            raise ValueError("ZeRO-1 local chunk metadata is incomplete")
        for chunk_index, (chunk_parameters, chunk_offsets) in enumerate(zip(parameters, offsets)):
            if len(chunk_parameters) != len(chunk_offsets):
                raise ValueError("ZeRO-1 parameter/offset metadata is incomplete")
            cursor = 0
            capacity = chunks[chunk_index].numel()
            for parameter, offset in zip(chunk_parameters, chunk_offsets):
                if parameter is None:
                    continue
                if offset is None:
                    raise ValueError("ZeRO-1 parameter offset is missing")
                index = next(
                    (index for index, value in enumerate(group) if value is parameter),
                    None,
                )
                if index is None:
                    raise ValueError("ZeRO-1 partition references an unknown parameter")
                start = int(offset)
                length = min(parameter.numel() - start, capacity - cursor)
                if start < 0 or length <= 0:
                    raise ValueError("ZeRO-1 partition metadata contains an empty range")
                result.append(Zero1Shard(
                    identifiers[index], group_index, start, length, chunk_index,
                    ("optimizer.master", "optimizer.exp_avg", "optimizer.exp_avg_sq"),
                    cursor,
                ))
                cursor += length
            if cursor > capacity:
                raise ValueError("ZeRO-1 partition exceeds its master chunk")
    if not result and not bool(getattr(module, "_merak_role_aware_zero1", False)):
        raise ValueError("ZeRO-1 optimizer has no local shard ranges")
    for group_index, chunks in enumerate(optimizer.local_sub_partitions_of_fp32_groups):
        for chunk_index, chunk in enumerate(chunks):
            ranges = sorted(
                (shard.chunk_offset, shard.chunk_offset + shard.length)
                for shard in result
                if shard.group_index == group_index and shard.local_chunk == chunk_index
            )
            if any(right[0] < left[1] for left, right in zip(ranges, ranges[1:])) or (
                ranges and ranges[-1][1] > chunk.numel()
            ):
                raise ValueError("ZeRO-1 shard ranges overlap or exceed a master chunk")
    by_parameter: dict[str, list[tuple[int, int]]] = {}
    for shard in result:
        by_parameter.setdefault(shard.parameter_id, []).append(
            (shard.parameter_offset, shard.parameter_offset + shard.length)
        )
    for parameter_id, ranges in by_parameter.items():
        ranges.sort()
        if any(right[0] < left[1] for left, right in zip(ranges, ranges[1:])):
            raise ValueError(f"ZeRO-1 parameter ranges overlap: {parameter_id}")
    return tuple(result)


def capture_zero1_fragments(
    module: Any,
    optimizer: Any,
    *,
    rank: int,
    storage_host: int,
    epoch: int,
    manifest: tuple[Any, ...] | None = None,
) -> tuple[ObjectFragment, ...]:
    """Materialize per-parameter FP32 shard and Adam moments with global IDs.

    This does not publish a checkpoint: callers must still protect complete
    model, optimizer metadata, scheduler and exact rank-local runtime state.
    """
    if type(optimizer.optimizer) is not torch.optim.AdamW:
        raise TypeError("ZeRO-1 object shards require AdamW")
    shards = iter_zero1_shards(module, optimizer)
    fragments: list[ObjectFragment] = []
    names = dict(module.named_parameters())
    _zero1_steps(optimizer)
    semantic_ranges: dict[tuple[int, int], list[Any]] = {}
    semantic_fallback_chunks: set[tuple[int, int]] = set()
    if bool(getattr(module, "_merak_semantic_zero1", False)):
        if manifest is None or any(item.rank != rank for item in manifest):
            raise ValueError("semantic ZeRO-1 requires its validated rank manifest")
        for item in manifest:
            semantic_ranges.setdefault((item.group_index, item.chunk_index), []).append(item)
        expected_chunks = {(item.group_index, item.local_chunk) for item in shards}
        if set(semantic_ranges) != expected_chunks:
            raise ValueError("semantic ZeRO-1 manifest does not cover optimizer chunks")
        for key, items in semantic_ranges.items():
            chunk = optimizer.local_sub_partitions_of_fp32_groups[key[0]][key[1]]
            ranges = sorted((item.chunk_offset, item.chunk_offset + item.length) for item in items)
            if ranges and ranges[0][0] != 0 or any(
                left[1] != right[0] for left, right in zip(ranges, ranges[1:])
            ):
                raise ValueError("semantic ZeRO-1 manifest has a chunk gap or overlap")
            if ranges and ranges[-1][1] > chunk.numel():
                raise ValueError("semantic ZeRO-1 manifest exceeds its chunk")
            if ranges and ranges[-1][1] < chunk.numel():
                semantic_fallback_chunks.add(key)
    for chunk_index, chunks in enumerate(optimizer.local_sub_partitions_of_fp32_groups):
        for local_chunk, chunk in enumerate(chunks):
            chunk_shards = [
                shard for shard in shards
                if shard.group_index == chunk_index and shard.local_chunk == local_chunk
            ]
            if not chunk_shards:
                if bool(getattr(module, "_merak_role_aware_zero1", False)):
                    object_id = f"zero1@rank={rank}/group={chunk_index}/chunk={local_chunk}"
                    state = optimizer.optimizer.state[chunk]
                    if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                        raise ValueError("ZeRO-1 AdamW moments are incomplete")
                    for role, value in (
                        ("optimizer.master", chunk),
                        ("optimizer.exp_avg", state["exp_avg"]),
                        ("optimizer.exp_avg_sq", state["exp_avg_sq"]),
                    ):
                        if value.dtype != torch.float32 or value.shape != chunk.shape:
                            raise ValueError("ZeRO-1 AdamW chunk has incompatible shape or dtype")
                        fragments.append(ObjectFragment.from_tensor(
                            ObjectAddress(object_id + f"::{role}", role, tuple(chunk.shape), str(value.dtype)),
                            epoch=epoch, offsets=(0,), tensor=value,
                            source_rank=rank, storage_host=storage_host,
                        ))
                continue
            if not isinstance(chunk, torch.Tensor) or chunk.ndim != 1:
                raise ValueError("ZeRO-1 master chunk must be flat")
            state: Mapping[str, Any] = optimizer.optimizer.state.get(chunk, {})
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("ZeRO-1 AdamW moments are incomplete")
            object_id = (
                f"zero1@rank={rank}/group={chunk_index}/chunk={local_chunk}"
                if bool(getattr(module, "_merak_role_aware_zero1", False))
                else f"{chunk_shards[0].parameter_id}"
            )
            roles = (
                ("optimizer.master", chunk),
                ("optimizer.exp_avg", state["exp_avg"]),
                ("optimizer.exp_avg_sq", state["exp_avg_sq"]),
            )
            for role, value in roles:
                if not isinstance(value, torch.Tensor) or value.shape != chunk.shape or value.dtype != torch.float32:
                    raise ValueError("ZeRO-1 AdamW chunk has incompatible shape or dtype")
                semantic_key = (chunk_index, local_chunk)
                if bool(getattr(module, "_merak_semantic_zero1", False)) and semantic_key not in semantic_fallback_chunks:
                    key = semantic_key
                    for item in semantic_ranges.get(key, ()):
                        parameter = names[item.parameter_id]
                        parameter_id = _optimizer_model_id(
                            item.parameter_id, module, rank, parameter
                        )
                        address = ObjectAddress(
                            f"{parameter_id}::{role}",
                            role,
                            (parameter.numel(),),
                            "torch.float32",
                        )
                        fragments.append(ObjectFragment.from_tensor(
                            address,
                            epoch=epoch,
                            offsets=(item.parameter_offset,),
                            tensor=value.narrow(0, item.chunk_offset, item.length),
                            source_rank=rank,
                            storage_host=storage_host,
                        ))
                    continue
                if bool(getattr(module, "_merak_role_aware_zero1", False)):
                    address = ObjectAddress(
                        f"{object_id}::{role}", role, tuple(chunk.shape), str(value.dtype)
                    )
                    fragments.append(ObjectFragment.from_tensor(
                        address, epoch=epoch, offsets=(0,), tensor=value,
                        source_rank=rank, storage_host=storage_host,
                    ))
                    continue
                for shard in chunk_shards:
                    model_parameter = names[shard.parameter_id]
                    object_id = _optimizer_model_id(
                        shard.parameter_id, module, rank, model_parameter
                    )
                    address = ObjectAddress(
                        f"{object_id}::{role}", role,
                        (model_parameter.numel(),), str(value.dtype),
                    )
                    fragments.append(ObjectFragment.from_tensor(
                        address, epoch=epoch, offsets=(shard.parameter_offset,),
                        tensor=value.narrow(0, shard.chunk_offset, shard.length),
                        source_rank=rank, storage_host=storage_host,
                    ))
    return tuple(fragments)


def capture_zero1_model_fragments(
    module: Any,
    optimizer: Any,
    *,
    rank: int,
    storage_host: int,
    epoch: int,
    tensor_parallel_rank: int = 0,
    tensor_parallel_world_size: int = 1,
    detector_evidence: Mapping[str, torch.Tensor] | None = None,
    model_aliases: Mapping[str, str] | None = None,
) -> tuple[ObjectFragment, ...]:
    """Capture every model parameter using the common UCP-style global slice identity."""
    bound = {name for group in _parameter_ids(module, optimizer) for name in group}
    names = _named_parameters(module)
    trainable = {name for name, parameter in names.values() if parameter.requires_grad}
    if bound != trainable:
        raise ValueError("ZeRO-1 capture lacks exact trainable parameter bindings")
    named_parameters = {name: parameter for name, parameter in names.values()}
    semantic_names = {
        name: _model_object_id(name, module, rank, parameter)
        for name, parameter in named_parameters.items()
    }
    if set(model_aliases or {}) - set(semantic_names):
        raise ValueError("MoE model alias does not match a live parameter")
    model_ids = {
        name: _model_object_id(name, module, rank, parameter)
        for name, parameter in named_parameters.items()
    }
    fragments = []
    for name, parameter in sorted(names.values()):
        model_id = model_ids[name]
        if bool(getattr(module, "_merak_semantic_zero1", False)):
            value = None if detector_evidence is None else detector_evidence.get(name)
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or not torch.equal(
                value.detach().cpu().to(dtype=parameter.dtype), parameter.detach().cpu()
            ):
                raise ValueError(f"semantic ZeRO-1 lacks matching detector value for {name}")
        address, offsets, tensor = _address(
            model_id, "model", parameter, parameter,
            tensor_parallel_rank=tensor_parallel_rank,
            tensor_parallel_world_size=tensor_parallel_world_size,
        )
        canonical = (model_aliases or {}).get(name)
        if canonical is not None:
            address = ObjectAddress(
                f"{canonical}::model", address.role, address.global_shape, address.dtype
            )
        fragments.append(ObjectFragment.from_tensor(
            address, epoch=epoch, offsets=offsets, tensor=tensor,
            source_rank=rank, storage_host=storage_host,
        ))
    for name, tensor in sorted(persistent_model_buffers(module)):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("ZeRO-1 model buffer must be tensor-valued")
        address, offsets, value = _address(
            _buffer_object_id(name, module, rank), "model.buffer", tensor, tensor,
            tensor_parallel_rank=0,
            tensor_parallel_world_size=1,
        )
        fragments.append(ObjectFragment.from_tensor(
            address, epoch=epoch, offsets=offsets, tensor=value,
            source_rank=rank, storage_host=storage_host,
        ))
    return tuple(fragments)


def zero1_model_requests(
    module: Any, *, rank: int, tensor_parallel_rank: int = 0,
    tensor_parallel_world_size: int = 1,
    model_aliases: Mapping[str, str] | None = None,
) -> tuple[SliceRequest, ...]:
    """Ask the object core for a fresh rank's model tensors, including buffers."""
    aliases = dict(model_aliases or {})
    named_parameters = dict(module.named_parameters())
    model_ids = {
        name: _model_object_id(name, module, rank, parameter)
        for name, parameter in named_parameters.items()
    }
    if set(aliases) - set(named_parameters):
        raise ValueError("MoE model alias does not match a live parameter")
    requests = []
    entries = [
        (model_ids[name], name, parameter, "model")
        for name, parameter in named_parameters.items()
    ] + [
        (_buffer_object_id(name, module, rank), name, buffer, "model.buffer")
        for name, buffer in persistent_model_buffers(module)
    ]
    for semantic_id, name, tensor, role in entries:
        address, offsets, value = _address(
            semantic_id, role, tensor, tensor,
            tensor_parallel_rank=(0 if role == "model.buffer" else tensor_parallel_rank),
            tensor_parallel_world_size=(1 if role == "model.buffer" else tensor_parallel_world_size),
        )
        canonical = aliases.get(name) if role == "model" else None
        source = None if canonical is None else ObjectAddress(
            f"{canonical}::model",
            address.role,
            address.global_shape,
            address.dtype,
        )
        requests.append(SliceRequest(
            rank, address, offsets, tuple(value.shape), source_address=source
        ))
    return tuple(requests)


def zero1_optimizer_requests(
    module: Any,
    optimizer: Any,
    *,
    rank: int,
    manifest: tuple[Any, ...] | None = None,
) -> tuple[SliceRequest, ...]:
    """Request complete named optimizer objects for the target rank's local shards."""
    requests = []
    if bool(getattr(module, "_merak_semantic_zero1", False)):
        if manifest is None or any(item.rank != rank for item in manifest):
            raise ValueError("semantic ZeRO-1 requires its validated rank manifest")
        names = dict(module.named_parameters())
        shards = iter_zero1_shards(module, optimizer)
        by_chunk: dict[tuple[int, int], list[Any]] = {}
        for item in manifest:
            by_chunk.setdefault((item.group_index, item.chunk_index), []).append(item)
        if set(by_chunk) != {(item.group_index, item.local_chunk) for item in shards}:
            raise ValueError("semantic ZeRO-1 manifest does not cover optimizer chunks")
        fallback_chunks = set()
        for (group, index), items in by_chunk.items():
            chunk = optimizer.local_sub_partitions_of_fp32_groups[group][index]
            ranges = sorted((item.chunk_offset, item.chunk_offset + item.length) for item in items)
            if not ranges or ranges[0][0] != 0 or any(
                left[1] != right[0] for left, right in zip(ranges, ranges[1:])
            ) or ranges[-1][1] > chunk.numel():
                raise ValueError("semantic ZeRO-1 requests have an invalid chunk range")
            if ranges[-1][1] < chunk.numel():
                fallback_chunks.add((group, index))
        for item in manifest:
            if item.parameter_id not in names:
                raise ValueError("semantic ZeRO-1 manifest references an unknown parameter")
            if (item.group_index, item.chunk_index) in fallback_chunks:
                continue
            parameter = names[item.parameter_id]
            object_id = _optimizer_model_id(item.parameter_id, module, rank, parameter)
            for role in ("optimizer.master", "optimizer.exp_avg", "optimizer.exp_avg_sq"):
                address = ObjectAddress(
                    f"{object_id}::{role}", role, (parameter.numel(),), "torch.float32"
                )
                requests.append(SliceRequest(
                    rank, address, (item.parameter_offset,), (item.length,)
                ))
        for (group, index), items in by_chunk.items():
            chunk = optimizer.local_sub_partitions_of_fp32_groups[group][index]
            if (group, index) in fallback_chunks:
                padding_id = f"zero1@rank={rank}/group={group}/chunk={index}"
                for role in ("optimizer.master", "optimizer.exp_avg", "optimizer.exp_avg_sq"):
                    address = ObjectAddress(
                        f"{padding_id}::{role}", role, tuple(chunk.shape), "torch.float32",
                    )
                    requests.append(SliceRequest(rank, address, (0,), address.global_shape))
            step_id = f"zero1@rank={rank}/group={group}/chunk={index}"
            step = ObjectAddress(
                f"{step_id}::optimizer.step", "optimizer.step", (1,), "torch.float32"
            )
            requests.append(SliceRequest(rank, step, (0,), (1,)))
        return tuple(requests)
    if bool(getattr(module, "_merak_role_aware_zero1", False)):
        for group, chunks in enumerate(optimizer.local_sub_partitions_of_fp32_groups):
            if not optimizer.fp16_groups[group]:
                continue
            for index, chunk in enumerate(chunks):
                object_id = f"zero1@rank={rank}/group={group}/chunk={index}"
                for role in ("optimizer.master", "optimizer.exp_avg", "optimizer.exp_avg_sq"):
                    address = ObjectAddress(
                        f"{object_id}::{role}", role, tuple(chunk.shape), "torch.float32"
                    )
                    requests.append(SliceRequest(rank, address, (0,), address.global_shape))
                step = ObjectAddress(
                    f"{object_id}::optimizer.step", "optimizer.step", (1,), "torch.float32"
                )
                requests.append(SliceRequest(rank, step, (0,), (1,)))
        return tuple(requests)

    names = dict(module.named_parameters())
    for parameter_id in sorted({shard.parameter_id for shard in iter_zero1_shards(module, optimizer)}):
        object_id = _optimizer_model_id(parameter_id, module, rank, names[parameter_id])
        for role in ("optimizer.master", "optimizer.exp_avg", "optimizer.exp_avg_sq"):
            address = ObjectAddress(
                f"{object_id}::{role}", role,
                (names[parameter_id].numel(),), "torch.float32",
            )
            requests.append(SliceRequest(rank, address, (0,), address.global_shape))
        step = ObjectAddress(
            f"{object_id}::optimizer.step", "optimizer.step", (1,), "torch.float32"
        )
        requests.append(SliceRequest(rank, step, (0,), (1,)))
    return tuple(requests)


def apply_zero1_model_tensors(
    module: Any,
    tensors: Mapping[str, torch.Tensor],
    *,
    rank: int = 0,
    model_aliases: Mapping[str, str] | None = None,
) -> None:
    """Validate every live model tensor before changing weights or buffers."""
    aliases = dict(model_aliases or {})
    named_parameters = dict(module.named_parameters())
    model_ids = {
        name: _model_object_id(name, module, rank, parameter)
        for name, parameter in named_parameters.items()
    }
    if set(aliases) - set(named_parameters):
        raise ValueError("MoE model alias does not match a live parameter")
    live = {
        **{
            f"{model_ids[name]}::model": tensor
            for name, tensor in named_parameters.items()
        },
        **{
            f"{_buffer_object_id(name, module, rank)}::model.buffer": tensor
            for name, tensor in persistent_model_buffers(module)
        },
    }
    if set(live) != set(tensors):
        raise ValueError("ZeRO-1 restore lacks complete model or buffer objects")
    for name, destination in live.items():
        source = tensors[name]
        if not isinstance(source, torch.Tensor) or destination.shape != source.shape or destination.dtype != source.dtype:
            raise ValueError(f"ZeRO-1 model object differs from live binding: {name}")
    with torch.no_grad():
        for name, destination in live.items():
            destination.copy_(tensors[name].to(destination.device))


def apply_zero1_fragments(
    module: Any,
    optimizer: Any,
    tensors: Mapping[str, torch.Tensor],
    *,
    steps: Mapping[str, torch.Tensor],
    rank: int = 0,
    manifest: tuple[Any, ...] | None = None,
) -> None:
    """Apply complete, named FP32 ZeRO-1 shards and their AdamW slot values."""
    if type(optimizer.optimizer) is not torch.optim.AdamW:
        raise ValueError("ZeRO-1 restore requires AdamW")
    shards = iter_zero1_shards(module, optimizer)
    if bool(getattr(module, "_merak_role_aware_zero1", False)) and not bool(
        getattr(module, "_merak_semantic_zero1", False)
    ):
        expected = {}
        pending = {}
        chunk_steps = {}
        expected_steps = set()
        for group, chunks in enumerate(optimizer.local_sub_partitions_of_fp32_groups):
            if not optimizer.fp16_groups[group]:
                continue
            for index, chunk in enumerate(chunks):
                object_id = f"zero1@rank={rank}/group={group}/chunk={index}"
                step_key = f"{object_id}::optimizer.step"
                expected_steps.add(step_key)
                for role in ("optimizer.master", "optimizer.exp_avg", "optimizer.exp_avg_sq"):
                    expected[f"{object_id}::{role}"] = chunk
                state = optimizer.optimizer.state.get(chunk)
                if state and set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                    raise ValueError("ZeRO-1 AdamW slots have an unexpected schema")
                if not state:
                    pending[chunk] = {
                        "step": torch.empty((), dtype=torch.float32, device="cpu"),
                        "exp_avg": torch.empty_like(chunk),
                        "exp_avg_sq": torch.empty_like(chunk),
                    }
                chunk_steps[chunk] = steps.get(step_key)
        if set(tensors) != set(expected) or set(steps) != expected_steps:
            raise ValueError("ZeRO-1 restore is missing or has unexpected optimizer shards")
        for key, destination in expected.items():
            source = tensors[key]
            if (not isinstance(source, torch.Tensor) or source.shape != destination.shape
                or source.dtype != torch.float32):
                raise ValueError(f"ZeRO-1 restored chunk differs from destination: {key}")
        for key in expected_steps:
            value = steps[key]
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.shape != (1,):
                raise ValueError("ZeRO-1 restore needs every exact named AdamW step scalar")
        if any(
            not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.shape != (1,)
            for value in chunk_steps.values()
        ):
            raise ValueError("ZeRO-1 restore needs every exact named AdamW step scalar")
        with torch.no_grad():
            for group, chunks in enumerate(optimizer.local_sub_partitions_of_fp32_groups):
                if not optimizer.fp16_groups[group]:
                    continue
                for index, chunk in enumerate(chunks):
                    object_id = f"zero1@rank={rank}/group={group}/chunk={index}"
                    slots = optimizer.optimizer.state.get(chunk) or pending[chunk]
                    for role, destination in (
                        ("optimizer.master", chunk),
                        ("optimizer.exp_avg", slots["exp_avg"]),
                        ("optimizer.exp_avg_sq", slots["exp_avg_sq"]),
                    ):
                        destination.copy_(tensors[f"{object_id}::{role}"].to(destination.device))
                    if chunk in pending:
                        optimizer.optimizer.state[chunk] = pending[chunk]
                    optimizer.optimizer.state[chunk]["step"].copy_(
                        steps[f"{object_id}::optimizer.step"].reshape_as(
                            optimizer.optimizer.state[chunk]["step"]
                        )
                    )
        return

    names = dict(module.named_parameters())
    if bool(getattr(module, "_merak_semantic_zero1", False)):
        if manifest is None or any(item.rank != rank for item in manifest):
            raise ValueError("semantic ZeRO-1 restore requires its validated rank manifest")
        expected = {
            (
                item.parameter_id,
                item.group_index,
                item.chunk_index,
                item.parameter_offset,
                item.chunk_offset,
                item.length,
            )
            for item in iter_zero1_shards(module, optimizer)
        }
        actual = {
            (
                item.parameter_id,
                item.group_index,
                item.chunk_index,
                item.parameter_offset,
                item.chunk_offset,
                item.length,
            )
            for item in manifest
        }
        if actual != expected:
            raise ValueError("semantic ZeRO-1 restore manifest differs from live chunk ranges")
    semantic_fallback_chunks: set[tuple[int, int]] = set()
    semantic_by_chunk: dict[tuple[int, int], list[Any]] = {}
    if bool(getattr(module, "_merak_semantic_zero1", False)):
        for item in manifest:
            semantic_by_chunk.setdefault((item.group_index, item.chunk_index), []).append(item)
        expected_chunks = {(shard.group_index, shard.local_chunk) for shard in shards}
        if set(semantic_by_chunk) != expected_chunks:
            raise ValueError("semantic ZeRO-1 manifest does not cover optimizer chunks")
        for key, items in semantic_by_chunk.items():
            chunk = optimizer.local_sub_partitions_of_fp32_groups[key[0]][key[1]]
            ranges = sorted((item.chunk_offset, item.chunk_offset + item.length) for item in items)
            if not ranges or ranges[0][0] != 0 or any(
                left[1] != right[0] for left, right in zip(ranges, ranges[1:])
            ) or ranges[-1][1] > chunk.numel():
                raise ValueError("semantic ZeRO-1 manifest has an invalid chunk range")
            if ranges[-1][1] < chunk.numel():
                semantic_fallback_chunks.add(key)
    object_ids = {
        shard.parameter_id: _optimizer_model_id(
            shard.parameter_id, module, rank, names[shard.parameter_id]
        )
        for shard in shards
    }
    if bool(getattr(module, "_merak_semantic_zero1", False)):
        step_names = {
            f"zero1@rank={rank}/group={group}/chunk={index}::optimizer.step"
            for group, chunks in enumerate(optimizer.local_sub_partitions_of_fp32_groups)
            if optimizer.fp16_groups[group]
            for index, _ in enumerate(chunks)
        }
    else:
        step_names = {f"{object_id}::optimizer.step" for object_id in object_ids.values()}
    if set(steps) != set(step_names) or any(
        not isinstance(steps[key], torch.Tensor) or steps[key].dtype != torch.float32
        or steps[key].shape != (1,)
        for key in step_names
    ):
        raise ValueError("ZeRO-1 restore needs every exact named AdamW step scalar")
    expected: dict[str, list[tuple[int, int, torch.Tensor, int, torch.Tensor]]] = {}
    pending: dict[torch.Tensor, dict[str, torch.Tensor]] = {}
    chunk_steps: dict[torch.Tensor, torch.Tensor] = {}
    fallback_values: dict[str, torch.Tensor | None] = {}
    fallback_destinations: dict[str, torch.Tensor] = {}
    for shard in shards:
        chunk = optimizer.local_sub_partitions_of_fp32_groups[shard.group_index][shard.local_chunk]
        object_id = object_ids[shard.parameter_id]
        chunk_key = (shard.group_index, shard.local_chunk)
        step_key = (
            f"zero1@rank={rank}/group={shard.group_index}/chunk={shard.local_chunk}::optimizer.step"
            if bool(getattr(module, "_merak_semantic_zero1", False))
            else f"{object_id}::optimizer.step"
        )
        step_slice = steps[step_key]
        if chunk in chunk_steps and not torch.equal(chunk_steps[chunk], step_slice):
            raise ValueError("ZeRO-1 target chunk mixes unequal AdamW step values")
        chunk_steps[chunk] = step_slice
        state = optimizer.optimizer.state.get(chunk)
        if state is not None and state and set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("ZeRO-1 AdamW slots have an unexpected schema")
        if not state:
            pending.setdefault(chunk, {
                "step": torch.empty((), dtype=torch.float32, device="cpu"),
                "exp_avg": torch.empty_like(chunk),
                "exp_avg_sq": torch.empty_like(chunk),
            })
        slots = state if state else pending[chunk]
        if chunk_key in semantic_fallback_chunks:
            continue
        for role, destination in (
            ("optimizer.master", chunk),
            ("optimizer.exp_avg", slots["exp_avg"]),
            ("optimizer.exp_avg_sq", slots["exp_avg_sq"]),
        ):
            key = f"{object_id}::{role}"
            expected.setdefault(key, []).append(
                (shard.parameter_offset, shard.length, destination, shard.chunk_offset, names[shard.parameter_id])
            )
    if semantic_fallback_chunks:
        for group, index in semantic_fallback_chunks:
            chunk = optimizer.local_sub_partitions_of_fp32_groups[group][index]
            state = optimizer.optimizer.state.get(chunk)
            if state is not None and state and set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("ZeRO-1 AdamW slots have an unexpected schema")
            if not state:
                pending.setdefault(chunk, {
                    "step": torch.empty((), dtype=torch.float32, device="cpu"),
                    "exp_avg": torch.empty_like(chunk),
                    "exp_avg_sq": torch.empty_like(chunk),
                })
            slots = state if state else pending[chunk]
            object_id = f"zero1@rank={rank}/group={group}/chunk={index}"
            step_key = f"{object_id}::optimizer.step"
            value = steps.get(step_key)
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or value.shape != (1,):
                raise ValueError("ZeRO-1 restore needs every exact named AdamW step scalar")
            chunk_steps[chunk] = value
            for role, destination in (
                ("optimizer.master", chunk),
                ("optimizer.exp_avg", slots["exp_avg"]),
                ("optimizer.exp_avg_sq", slots["exp_avg_sq"]),
            ):
                key = f"{object_id}::{role}"
                fallback_values[key] = tensors.get(key)
                fallback_destinations[key] = destination
    if bool(getattr(module, "_merak_semantic_zero1", False)):
        if set(steps) != {
            f"zero1@rank={rank}/group={group}/chunk={index}::optimizer.step"
            for group, chunks in enumerate(optimizer.local_sub_partitions_of_fp32_groups)
            if optimizer.fp16_groups[group]
            for index, _ in enumerate(chunks)
        }:
            raise ValueError("ZeRO-1 restore needs every exact named AdamW step scalar")
    expected_keys = set(expected) | set(fallback_values)
    if set(tensors) != expected_keys:
        raise ValueError("ZeRO-1 restore is missing or has unexpected optimizer shards")
    for key, destination in fallback_destinations.items():
        value = fallback_values[key]
        if not isinstance(value, torch.Tensor) or value.shape != destination.shape or value.dtype != torch.float32:
            raise ValueError(f"ZeRO-1 restored chunk differs from destination: {key}")
    for key, ranges in expected.items():
        source = tensors[key]
        if not isinstance(source, torch.Tensor) or source.ndim != 1 or source.dtype != torch.float32:
            raise ValueError(f"ZeRO-1 restored tensor has wrong dtype or dimensions: {key}")
        if source.numel() != ranges[0][4].numel():
            raise ValueError(f"ZeRO-1 restored parameter shape disagrees: {key}")
        for start, length, destination, offset, _ in ranges:
            if destination.dtype != source.dtype or offset + length > destination.numel() or start + length > source.numel():
                raise ValueError(f"ZeRO-1 shard range differs from destination: {key}")
    if any(
        state and ("step" not in state or state["step"].dtype != torch.float32 or state["step"].numel() != 1)
        for state in optimizer.optimizer.state.values()
    ):
        raise ValueError("ZeRO-1 AdamW step schema differs from the checkpoint")
    for group, chunks in enumerate(optimizer.local_sub_partitions_of_fp32_groups):
        for chunk_index, chunk in enumerate(chunks):
            ranges = sorted(
                (offset, offset + length)
                for key, items in expected.items()
                if key.endswith("::optimizer.master")
                for _, length, target, offset, _ in items
                if target is chunk
            )
            if not ranges and bool(getattr(module, "_merak_role_aware_zero1", False)) and not bool(
                getattr(module, "_merak_semantic_zero1", False)
            ):
                object_id = f"zero1@rank={rank}/group={group}/chunk={chunk_index}"
                for role in ("optimizer.master", "optimizer.exp_avg", "optimizer.exp_avg_sq"):
                    key = f"{object_id}::{role}"
                    if key not in tensors:
                        raise ValueError(f"ZeRO-1 restore is missing padding-only chunk: {key}")
                continue
            if (group, chunk_index) in semantic_fallback_chunks:
                continue
            if not ranges:
                raise ValueError("ZeRO-1 master chunk has no parameter coverage")
            if (not bool(getattr(module, "_merak_semantic_zero1", False)) and ranges[0][0] != 0) or any(
                left[1] != right[0] for left, right in zip(ranges, ranges[1:])
            ):
                raise ValueError("ZeRO-1 master chunk has an uncovered internal range")
            # Final alignment padding is not trainable; preserve its live value.
            if ranges[-1][1] > chunk.numel():
                raise ValueError("ZeRO-1 master ranges exceed their chunk")
    with torch.no_grad():
        for key, ranges in expected.items():
            source = tensors[key]
            for start, length, destination, offset, _ in ranges:
                destination.narrow(0, offset, length).copy_(source.narrow(0, start, length))
        for key, destination in fallback_destinations.items():
            destination.copy_(fallback_values[key].to(destination.device))
        for chunk, value in chunk_steps.items():
            if chunk in pending:
                optimizer.optimizer.state[chunk] = pending[chunk]
            destination = optimizer.optimizer.state[chunk]["step"]
            destination.copy_(value.reshape_as(destination))


def zero1_semantic_compatibility(optimizer: Any) -> dict[str, Any]:
    """Report the ZeRO-1 snapshot adapter without implying SemanticFixed support."""
    is_zero1 = all(
        hasattr(optimizer, name)
        for name in ("local_sub_partitions_of_fp32_groups", "params_in_rank_sub_partitions")
    )
    if not is_zero1:
        return {"zero1": False, "semantic_fixed_compatible": False, "blockers": ["optimizer is not Merak ZeRO-1"]}
    return {
        "zero1": True,
        "semantic_fixed_compatible": False,
        "ucp_snapshot_adapter_available": all(
            hasattr(optimizer, name)
            for name in ("group_roles", "local_sub_partitions_of_fp32_groups")
        ),
        "blockers": [
            "the Trainer lifecycle does not yet publish ZeRO-1 objects through UCP semantic planning",
        ],
        "adapter": "zero1_shard_metadata_only",
    }
