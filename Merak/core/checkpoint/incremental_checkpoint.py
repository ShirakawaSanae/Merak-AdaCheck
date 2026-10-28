"""Validated AdamW update records for incremental checkpoint recovery."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch


SCHEMA_VERSION = 2
_BASE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ADAMW_OPTION_KEYS = {
    "amsgrad",
    "betas",
    "capturable",
    "differentiable",
    "eps",
    "foreach",
    "fused",
    "initial_lr",
    "lr",
    "maximize",
    "weight_decay",
}
_GRADIENT_DTYPES = {
    "torch.bfloat16": torch.bfloat16,
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
}


@dataclass(frozen=True)
class RecoveryBinding:
    """Logical training shard to which a base and its update suffix belong."""

    rank: int
    world_size: int
    data_parallel_rank: int
    pipeline_parallel_rank: int
    tensor_parallel_rank: int

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be within world_size")
        for name in (
            "data_parallel_rank",
            "pipeline_parallel_rank",
            "tensor_parallel_rank",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


def _optimizer_name(optimizer: torch.optim.Optimizer) -> str:
    cls = type(optimizer)
    return f"{cls.__module__}.{cls.__qualname__}"


def _snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    return copy.deepcopy(value)


def _json_value(value: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{field} must be a string-keyed JSON object")
    try:
        payload = json.dumps(
            value, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must contain only finite JSON values") from error
    restored = json.loads(payload)
    if not isinstance(restored, dict):
        raise ValueError(f"{field} must be a JSON object")
    return restored


def _validate_base_id(base_id: str) -> None:
    if not isinstance(base_id, str) or _BASE_ID.fullmatch(base_id) is None:
        raise ValueError("base_id must be a sha256:<64 lowercase hex> digest")


def _validate_parameter_ids(
    optimizer: torch.optim.Optimizer,
    parameter_ids: Sequence[Sequence[str]],
) -> list[list[str]]:
    if isinstance(parameter_ids, (str, bytes)) or len(parameter_ids) != len(
        optimizer.param_groups
    ):
        raise ValueError("parameter_ids must match optimizer parameter groups")
    normalized: list[list[str]] = []
    seen: set[str] = set()
    for group, identifiers in zip(optimizer.param_groups, parameter_ids):
        if isinstance(identifiers, (str, bytes)) or len(identifiers) != len(
            group["params"]
        ):
            raise ValueError("parameter_ids must match optimizer parameter order")
        current = list(identifiers)
        if any(not isinstance(identifier, str) or not identifier for identifier in current):
            raise ValueError("parameter_ids must be non-empty strings")
        if any(identifier in seen for identifier in current):
            raise ValueError("parameter_ids must be unique")
        seen.update(current)
        normalized.append(current)
    return normalized


def _validate_adamw(optimizer: torch.optim.Optimizer) -> None:
    if type(optimizer) is not torch.optim.AdamW:
        raise TypeError("incremental replay currently supports torch.optim.AdamW only")
    for group in optimizer.param_groups:
        unknown = set(group) - _ADAMW_OPTION_KEYS - {"params"}
        if unknown:
            raise ValueError(f"unsupported AdamW options: {sorted(unknown)}")
        if group.get("foreach") is not False:
            raise ValueError("incremental replay requires AdamW foreach=False")
        if group.get("fused") not in (None, False):
            raise ValueError("incremental replay requires non-fused AdamW")
        if group.get("capturable", False) or group.get("differentiable", False):
            raise ValueError(
                "incremental replay requires non-capturable, non-differentiable AdamW"
            )


def optimizer_topology_hash(
    optimizer: torch.optim.Optimizer,
    parameter_ids: Sequence[Sequence[str]],
) -> str:
    """Identify optimizer type and the ordered identity of every parameter."""
    identifiers = _validate_parameter_ids(optimizer, parameter_ids)
    topology = {
        "optimizer": _optimizer_name(optimizer),
        "groups": [
            [
                {
                    "id": identifier,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                }
                for identifier, parameter in zip(ids, group["params"])
            ]
            for ids, group in zip(identifiers, optimizer.param_groups)
        ],
    }
    payload = json.dumps(topology, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def capture_optimizer_update(
    optimizer: torch.optim.Optimizer,
    *,
    parameter_ids: Sequence[Sequence[str]],
    binding: RecoveryBinding,
    base_id: str,
    base_epoch: int,
    epoch: int,
    metadata: Mapping[str, Any] | None = None,
    runtime_state: Mapping[str, Any] | None = None,
    snapshot_gradients: bool = True,
    gradient_dtype: torch.dtype | None = None,
    gradient_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Snapshot the exact gradients and options consumed by one AdamW step."""
    _validate_adamw(optimizer)
    _validate_base_id(base_id)
    identifiers = _validate_parameter_ids(optimizer, parameter_ids)
    if epoch <= base_epoch:
        raise ValueError("update epoch must be newer than its base epoch")
    if gradient_dtype is not None and gradient_dtype not in _GRADIENT_DTYPES.values():
        raise ValueError("unsupported optimizer-update gradient dtype")
    gradient_multiplier = float(gradient_multiplier)
    if not math.isfinite(gradient_multiplier) or gradient_multiplier <= 0.0:
        raise ValueError("gradient multiplier must be finite and positive")
    gradients = []
    options = []
    for group in optimizer.param_groups:
        gradients.append(
            [
                None
                if parameter.grad is None
                else (
                    _snapshot(parameter.grad.to(dtype=gradient_dtype))
                    if snapshot_gradients and gradient_dtype is not None
                    else _snapshot(parameter.grad)
                    if snapshot_gradients
                    else parameter.grad.detach().to(dtype=gradient_dtype)
                    if gradient_dtype is not None
                    else parameter.grad.detach()
                )
                for parameter in group["params"]
            ]
        )
        options.append(
            {
                key: _snapshot(value)
                for key, value in group.items()
                if key != "params"
            }
        )
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": "optimizer_update",
        "base_id": base_id,
        "base_epoch": int(base_epoch),
        "epoch": int(epoch),
        "optimizer": _optimizer_name(optimizer),
        "binding": asdict(binding),
        "parameter_ids": identifiers,
        "topology_hash": optimizer_topology_hash(optimizer, identifiers),
        "param_group_options": options,
        "gradients": gradients,
        "gradient_multiplier": gradient_multiplier,
        "metadata": _json_value(metadata or {}, field="metadata"),
        "runtime_state": _snapshot(dict(runtime_state or {})),
    }
    if gradient_dtype is not None:
        record["gradient_dtype"] = str(gradient_dtype)
    return record


def _validated_update(
    optimizer: torch.optim.Optimizer,
    record: Mapping[str, Any],
    *,
    expected_parameter_ids: Sequence[Sequence[str]],
    expected_binding: RecoveryBinding,
    expected_base_id: str,
    expected_base_epoch: int,
    expected_epoch: int,
) -> tuple[list[list[torch.Tensor | None]], list[dict[str, Any]], float]:
    _validate_adamw(optimizer)
    _validate_base_id(expected_base_id)
    identifiers = _validate_parameter_ids(optimizer, expected_parameter_ids)
    if not isinstance(record, Mapping):
        raise ValueError("optimizer update must be a mapping")
    if int(record.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported optimizer-update schema")
    if record.get("kind") != "optimizer_update":
        raise ValueError("record is not an optimizer update")
    if record.get("base_id") != expected_base_id:
        raise ValueError("optimizer update refers to a different base")
    if int(record.get("base_epoch", -1)) != expected_base_epoch:
        raise ValueError("optimizer update has an unexpected base epoch")
    if int(record.get("epoch", -1)) != expected_epoch:
        raise ValueError("optimizer update epochs are not contiguous")
    if record.get("optimizer") != _optimizer_name(optimizer):
        raise ValueError("optimizer update uses a different optimizer type")
    if record.get("binding") != asdict(expected_binding):
        raise ValueError("optimizer update belongs to a different logical rank")
    if record.get("parameter_ids") != identifiers:
        raise ValueError("optimizer parameter identity or order changed")
    if record.get("topology_hash") != optimizer_topology_hash(optimizer, identifiers):
        raise ValueError("optimizer parameter topology changed")
    _json_value(record.get("metadata", {}), field="metadata")
    try:
        gradient_multiplier = float(record.get("gradient_multiplier", 1.0))
    except (TypeError, ValueError) as error:
        raise ValueError("optimizer update has an invalid gradient multiplier") from error
    if not math.isfinite(gradient_multiplier) or gradient_multiplier <= 0.0:
        raise ValueError("optimizer update has an invalid gradient multiplier")

    encoded_dtype_name = record.get("gradient_dtype")
    if encoded_dtype_name is not None and encoded_dtype_name not in _GRADIENT_DTYPES:
        raise ValueError("optimizer update has an unsupported gradient dtype")
    encoded_dtype = (
        _GRADIENT_DTYPES[encoded_dtype_name]
        if encoded_dtype_name is not None
        else None
    )
    gradients = record.get("gradients")
    options = record.get("param_group_options")
    if (
        isinstance(gradients, (str, bytes))
        or not isinstance(gradients, Sequence)
        or len(gradients) != len(optimizer.param_groups)
    ):
        raise ValueError("optimizer update has invalid gradient groups")
    if (
        isinstance(options, (str, bytes))
        or not isinstance(options, Sequence)
        or len(options) != len(optimizer.param_groups)
    ):
        raise ValueError("optimizer update has invalid parameter-group options")

    checked_gradients: list[list[torch.Tensor | None]] = []
    checked_options: list[dict[str, Any]] = []
    for group, group_gradients, group_options in zip(
        optimizer.param_groups, gradients, options
    ):
        if (
            isinstance(group_gradients, (str, bytes))
            or not isinstance(group_gradients, Sequence)
            or len(group_gradients) != len(group["params"])
        ):
            raise ValueError("optimizer update gradient count changed")
        if not isinstance(group_options, Mapping):
            raise ValueError("optimizer update options must be a mapping")
        if set(group_options) != set(group) - {"params"}:
            raise ValueError("optimizer update option keys changed")

        current_gradients: list[torch.Tensor | None] = []
        for parameter, gradient in zip(group["params"], group_gradients):
            if gradient is None:
                current_gradients.append(None)
                continue
            if not isinstance(gradient, torch.Tensor):
                raise ValueError("optimizer update gradients must be tensors or null")
            if gradient.is_sparse:
                raise ValueError("sparse gradients are not supported")
            expected_dtype = parameter.dtype if encoded_dtype is None else encoded_dtype
            if gradient.shape != parameter.shape or gradient.dtype != expected_dtype:
                raise ValueError("optimizer update gradient shape or dtype changed")
            current_gradients.append(gradient.detach())
        checked_gradients.append(current_gradients)
        checked_options.append(_snapshot(dict(group_options)))
    return checked_gradients, checked_options, gradient_multiplier


def apply_optimizer_update(
    optimizer: torch.optim.Optimizer,
    record: Mapping[str, Any],
    *,
    expected_parameter_ids: Sequence[Sequence[str]],
    expected_binding: RecoveryBinding,
    expected_base_id: str,
    expected_base_epoch: int,
    expected_epoch: int,
) -> None:
    """Validate, then apply one update to an optimizer restored from its base."""
    gradients, options, gradient_multiplier = _validated_update(
        optimizer,
        record,
        expected_parameter_ids=expected_parameter_ids,
        expected_binding=expected_binding,
        expected_base_id=expected_base_id,
        expected_base_epoch=expected_base_epoch,
        expected_epoch=expected_epoch,
    )
    for group, group_gradients, group_options in zip(
        optimizer.param_groups, gradients, options
    ):
        for key in tuple(group):
            if key != "params":
                del group[key]
        group.update(group_options)
        for parameter, gradient in zip(group["params"], group_gradients):
            parameter.grad = (
                None
                if gradient is None
                else gradient.to(device=parameter.device, dtype=parameter.dtype).mul_(
                    gradient_multiplier
                )
            )

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def validate_optimizer_update(
    optimizer: torch.optim.Optimizer,
    record: Mapping[str, Any],
    *,
    expected_parameter_ids: Sequence[Sequence[str]],
    expected_binding: RecoveryBinding,
    expected_base_id: str,
    expected_base_epoch: int,
    expected_epoch: int,
) -> None:
    """Validate an update without mutating the materialized optimizer."""
    _validated_update(
        optimizer,
        record,
        expected_parameter_ids=expected_parameter_ids,
        expected_binding=expected_binding,
        expected_base_id=expected_base_id,
        expected_base_epoch=expected_base_epoch,
        expected_epoch=expected_epoch,
    )


def replay_optimizer_updates(
    optimizer: torch.optim.Optimizer,
    records: Sequence[Mapping[str, Any]],
    *,
    parameter_ids: Sequence[Sequence[str]],
    binding: RecoveryBinding,
    base_id: str,
    base_epoch: int,
) -> int:
    """Apply an ordered, gap-free update suffix and return its final epoch."""
    epoch = int(base_epoch)
    for record in records:
        apply_optimizer_update(
            optimizer,
            record,
            expected_parameter_ids=parameter_ids,
            expected_binding=binding,
            expected_base_id=base_id,
            expected_base_epoch=base_epoch,
            expected_epoch=epoch + 1,
        )
        epoch += 1
    return epoch
