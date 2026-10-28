"""Live tensor bindings for the independent object recovery core."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from .object_recovery import ObjectAddress, ObjectCatalog, ObjectFragment, OpaqueRecord


_OPAQUE_KEYS = frozenset(("scheduler", "rng", "sampler", "progress"))


def _named_parameters(module: torch.nn.Module) -> dict[int, tuple[str, torch.nn.Parameter]]:
    try:
        named = module.named_parameters(remove_duplicate=False)
    except TypeError:
        named = module.named_parameters()
    result: dict[int, tuple[str, torch.nn.Parameter]] = {}
    for name, parameter in named:
        if id(parameter) in result:
            raise ValueError("object recovery adapter does not support tied parameters")
        result[id(parameter)] = (name, parameter)
    return result


def _optimizer_masters(
    module: torch.nn.Module, optimizer: Any
) -> tuple[dict[int, tuple[str, torch.nn.Parameter]], torch.optim.Optimizer]:
    names = _named_parameters(module)
    try:
        fp16_groups: Sequence[Sequence[torch.nn.Parameter]] = optimizer.fp16_groups
        fp32_groups: Sequence[Sequence[torch.nn.Parameter]] = optimizer.fp32_groups
        inner: torch.optim.Optimizer = optimizer.optimizer
    except AttributeError as error:
        raise ValueError("adapter requires a HalfPrecisionOptimizer-shaped wrapper") from error
    if len(fp16_groups) != len(fp32_groups) or len(fp16_groups) != len(inner.param_groups):
        raise ValueError("optimizer parameter groups are inconsistent")
    masters: dict[int, tuple[str, torch.nn.Parameter]] = {}
    for model_group, master_group, inner_group in zip(
        fp16_groups, fp32_groups, inner.param_groups
    ):
        if len(model_group) != len(master_group) or any(
            current is not master
            for current, master in zip(inner_group["params"], master_group)
        ):
            raise ValueError("optimizer master parameters are inconsistent")
        for model_parameter, master_parameter in zip(model_group, master_group):
            binding = names.get(id(model_parameter))
            if binding is None or id(model_parameter) in masters:
                raise ValueError("model parameter lacks one unambiguous optimizer binding")
            masters[id(model_parameter)] = (binding[0], master_parameter)
    unbound = [
        name
        for name, parameter in names.values()
        if parameter.requires_grad and id(parameter) not in masters
    ]
    if unbound:
        raise ValueError(f"trainable parameters lack optimizer bindings: {unbound}")
    return masters, inner


def _slice_for(
    parameter: torch.Tensor,
    *,
    tensor_parallel_rank: int,
    tensor_parallel_world_size: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shape = tuple(int(value) for value in parameter.shape)
    if not shape:
        raise ValueError("model parameters must be tensors with at least one dimension")
    if tensor_parallel_world_size < 1 or not 0 <= tensor_parallel_rank < tensor_parallel_world_size:
        raise ValueError("tensor-parallel rank is outside the declared world")
    if tensor_parallel_world_size == 1 or not bool(getattr(parameter, "model_parallel", False)):
        return shape, (0,) * len(shape)
    has_dimension = hasattr(parameter, "partition_dim")
    partition_stride = getattr(parameter, "partition_stride", None)
    legacy_stride = getattr(parameter, "__dict__", {}).get("stride")
    if not has_dimension:
        if partition_stride is not None or isinstance(legacy_stride, int):
            raise ValueError("partitioned parameter lacks partition metadata")
        return shape, (0,) * len(shape)
    stride = partition_stride if partition_stride is not None else legacy_stride
    if not isinstance(stride, int):
        raise ValueError("partitioned parameter lacks partition metadata")
    dimension = int(parameter.partition_dim)
    if stride != 1:
        raise ValueError("object recovery adapter supports only contiguous TP slices")
    if not 0 <= dimension < len(shape):
        raise ValueError("partitioned parameter has an invalid partition dimension")
    global_shape = list(shape)
    global_shape[dimension] *= tensor_parallel_world_size
    offsets = [0] * len(shape)
    offsets[dimension] = tensor_parallel_rank * shape[dimension]
    return tuple(global_shape), tuple(offsets)


def _address(
    fqn: str,
    role: str,
    tensor: torch.Tensor,
    partition_source: torch.Tensor,
    *,
    tensor_parallel_rank: int,
    tensor_parallel_world_size: int,
) -> tuple[ObjectAddress, tuple[int, ...], torch.Tensor]:
    if tensor.ndim == 0:
        return (
            ObjectAddress(f"{fqn}::{role}", role, (1,), str(tensor.dtype)),
            (0,),
            tensor.reshape(1),
        )
    global_shape, offsets = _slice_for(
        partition_source,
        tensor_parallel_rank=tensor_parallel_rank,
        tensor_parallel_world_size=tensor_parallel_world_size,
    )
    if tuple(tensor.shape) != tuple(partition_source.shape):
        raise ValueError("bound tensor shape conflicts with its model parameter slice")
    return (
        ObjectAddress(f"{fqn}::{role}", role, global_shape, str(tensor.dtype)),
        offsets,
        tensor,
    )


def _live_entries(
    module: torch.nn.Module,
    optimizer: Any,
    *,
    tensor_parallel_rank: int,
    tensor_parallel_world_size: int,
) -> tuple[tuple[ObjectAddress, tuple[int, ...], torch.Tensor, bool], ...]:
    masters, inner = _optimizer_masters(module, optimizer)
    names = _named_parameters(module)
    entries = []
    for _, (fqn, model_parameter) in sorted(names.items(), key=lambda item: item[1][0]):
        address, offsets, _ = _address(
            fqn,
            "model",
            model_parameter,
            model_parameter,
            tensor_parallel_rank=tensor_parallel_rank,
            tensor_parallel_world_size=tensor_parallel_world_size,
        )
        entries.append((address, offsets, model_parameter, False))
        bound = masters.get(id(model_parameter))
        if bound is None:
            continue
        _, master = bound
        address, offsets, _ = _address(
            fqn,
            "optimizer.master",
            master,
            model_parameter,
            tensor_parallel_rank=tensor_parallel_rank,
            tensor_parallel_world_size=tensor_parallel_world_size,
        )
        entries.append((address, offsets, master, False))
        for slot, state_value in sorted(inner.state.get(master, {}).items()):
            if not isinstance(state_value, torch.Tensor):
                raise ValueError("non-tensor optimizer state is unsupported")
            address, offsets, _ = _address(
                fqn,
                f"optimizer.{slot}",
                state_value,
                model_parameter,
                tensor_parallel_rank=tensor_parallel_rank,
                tensor_parallel_world_size=tensor_parallel_world_size,
            )
            entries.append((address, offsets, state_value, state_value.ndim == 0))
    ids = [item[0].semantic_id for item in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("adapter produced duplicate semantic object IDs")
    return tuple(entries)


def _fresh_adam_slots(
    module: torch.nn.Module,
    optimizer: Any,
    tensors: Mapping[str, torch.Tensor],
) -> dict[torch.nn.Parameter, dict[str, torch.Tensor]]:
    """Build, but do not install, missing Adam slots for a fresh optimizer."""
    masters, inner = _optimizer_masters(module, optimizer)
    pending = {}
    for _, (fqn, model_parameter) in sorted(
        _named_parameters(module).items(), key=lambda item: item[1][0]
    ):
        bound = masters.get(id(model_parameter))
        if bound is None:
            continue
        _, master = bound
        if inner.state.get(master):
            continue
        restored = {
            slot: tensors[f"{fqn}::optimizer.{slot}"]
            for slot in ("step", "exp_avg", "exp_avg_sq")
            if f"{fqn}::optimizer.{slot}" in tensors
        }
        if not restored:
            continue
        if not isinstance(inner, (torch.optim.Adam, torch.optim.AdamW)):
            raise ValueError("fresh optimizer slot bootstrap requires Adam or AdamW")
        if set(restored) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("fresh Adam state requires step, exp_avg, and exp_avg_sq")
        if tuple(restored["step"].shape) != (1,) or restored["step"].dtype != torch.float32:
            raise ValueError("restored Adam step must be one float32 scalar")
        for slot in ("exp_avg", "exp_avg_sq"):
            if (
                tuple(restored[slot].shape) != tuple(master.shape)
                or restored[slot].dtype != master.dtype
            ):
                raise ValueError("restored Adam moment shape or dtype is incompatible")
        group = next(
            group
            for group in inner.param_groups
            if any(current is master for current in group["params"])
        )
        if group.get("amsgrad", False):
            raise ValueError("fresh Adam slot bootstrap does not support amsgrad")
        step_device = (
            master.device
            if group.get("capturable", False) or group.get("fused", False)
            else torch.device("cpu")
        )
        pending[master] = {
            "step": torch.empty((), dtype=torch.float32, device=step_device),
            "exp_avg": torch.empty_like(master, memory_format=torch.preserve_format),
            "exp_avg_sq": torch.empty_like(master, memory_format=torch.preserve_format),
        }
    return pending


def capture_live_object_fragments(
    module: torch.nn.Module,
    optimizer: Any,
    *,
    rank: int,
    storage_host: int,
    epoch: int,
    tensor_parallel_rank: int,
    tensor_parallel_world_size: int,
    base_epoch: int | None = None,
    dependency_epoch: int | None = None,
) -> tuple[ObjectFragment, ...]:
    """Capture the core's versioned fragments from live model and AdamW state."""
    if rank < 0 or storage_host < 0 or epoch < 0:
        raise ValueError("rank, storage host, and epoch must be non-negative")
    return tuple(
        ObjectFragment.from_tensor(
            address,
            epoch=epoch,
            offsets=offsets,
            tensor=value.reshape(1) if scalar else value,
            source_rank=rank,
            storage_host=storage_host,
            base_epoch=base_epoch,
            dependency_epoch=dependency_epoch,
        )
        for address, offsets, value, scalar in _live_entries(
            module,
            optimizer,
            tensor_parallel_rank=tensor_parallel_rank,
            tensor_parallel_world_size=tensor_parallel_world_size,
        )
    )


def build_live_object_catalog(
    epoch: int,
    fragments: Sequence[ObjectFragment],
    opaque_records: Sequence[OpaqueRecord] = (),
) -> ObjectCatalog:
    catalog = ObjectCatalog(epoch, tuple(fragments), tuple(opaque_records))
    catalog.validate()
    return catalog


def capture_opaque_records(
    epoch: int,
    values: Mapping[str, Any],
    *,
    owner_rank: int,
    storage_hosts: Sequence[int],
) -> tuple[OpaqueRecord, ...]:
    if epoch < 0 or owner_rank < 0 or set(values) != _OPAQUE_KEYS:
        raise ValueError("opaque capture requires scheduler, rng, sampler, and progress")
    return tuple(
        OpaqueRecord.from_value(
            owner_rank=owner_rank,
            key=key,
            epoch=epoch,
            value=values[key],
            storage_hosts=storage_hosts,
        )
        for key in sorted(_OPAQUE_KEYS)
    )


def apply_live_object_tensors(
    module: torch.nn.Module,
    optimizer: Any,
    tensors: Mapping[str, torch.Tensor],
    *,
    tensor_parallel_rank: int,
    tensor_parallel_world_size: int,
) -> None:
    """Copy a sealed core restore into compatible live model and AdamW tensors."""
    pending = _fresh_adam_slots(module, optimizer, tensors)
    entries = _live_entries(
        module,
        optimizer,
        tensor_parallel_rank=tensor_parallel_rank,
        tensor_parallel_world_size=tensor_parallel_world_size,
    )
    expected = {
        address.semantic_id: (value, scalar)
        for address, _, value, scalar in entries
    }
    names = _named_parameters(module)
    masters, inner = _optimizer_masters(module, optimizer)
    for model_id, (_, master) in masters.items():
        fqn = names[model_id][0]
        for slot, value in pending.get(master, {}).items():
            expected[f"{fqn}::optimizer.{slot}"] = (value, slot == "step")
    if set(tensors) != set(expected):
        raise ValueError("restored tensors do not match the live object bindings")
    for semantic_id, (destination, scalar) in expected.items():
        value = tensors[semantic_id]
        expected_shape = (1,) if scalar else tuple(destination.shape)
        if str(value.dtype) != str(destination.dtype) or tuple(value.shape) != expected_shape:
            raise ValueError("restored tensor shape or dtype is incompatible with live state")
    for master, state in pending.items():
        inner.state[master] = state
    with torch.no_grad():
        for semantic_id, (destination, scalar) in expected.items():
            value = tensors[semantic_id]
            destination.copy_(value.reshape_as(destination) if scalar else value)
