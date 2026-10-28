"""Fresh-process replay of a sealed SemanticFixed object-update suffix."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from .catalog import checkpoint_state_checksum
from .incremental_checkpoint import (
    RecoveryBinding,
    apply_optimizer_update,
    validate_optimizer_update,
)
from .object_semantic_plan import ActiveGradientPlan
from .object_semantic_schema import recover_gradient_encoding
from .object_update_transport import (
    ObjectUpdateLocator,
    fetch_object_update_control,
    restore_object_update_epoch,
)
from .replica_service import ReplicaServiceClient


_BASE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_RUNTIME_KEYS = frozenset(
    (
        "iteration",
        "random_rng_state",
        "np_rng_state",
        "torch_rng_state",
        "cuda_rng_state",
        "rng_tracker_states",
        "sampler",
    )
)


def _binding(value: object, *, field: str) -> RecoveryBinding:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a logical recovery binding")
    try:
        return RecoveryBinding(
            rank=int(value["rank"]),
            world_size=int(value["world_size"]),
            data_parallel_rank=int(value["data_parallel_rank"]),
            pipeline_parallel_rank=int(value["pipeline_parallel_rank"]),
            tensor_parallel_rank=int(value["tensor_parallel_rank"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a logical recovery binding") from error


def _parameter_ids(value: object) -> tuple[tuple[str, ...], ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("base parameter ids are invalid")
    result = tuple(tuple(group) for group in value)
    if any(
        isinstance(group, (str, bytes))
        or any(not isinstance(name, str) or not name for name in group)
        for group in result
    ):
        raise ValueError("base parameter ids are invalid")
    return result


def _runtime(value: object, *, epoch: int) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not _RUNTIME_KEYS.issubset(value):
        raise ValueError("SemanticFixed recovery requires complete runtime state")
    if int(value["iteration"]) != epoch:
        raise ValueError("SemanticFixed runtime iteration disagrees with its epoch")
    return value


def _validate_base(
    base_state: Mapping[str, Any],
    *,
    base_id: str,
    base_epoch: int,
    logical_rank: int,
) -> tuple[RecoveryBinding, tuple[tuple[str, ...], ...]]:
    required = {
        "schema_version",
        "kind",
        "iteration",
        "model",
        "optimizer",
        "lr_scheduler",
        "parameter_ids",
        "binding",
        "runtime_state",
    }
    if (
        not isinstance(base_state, Mapping)
        or not required.issubset(base_state)
        or base_state["schema_version"] != 1
        or base_state["kind"] != "merak_incremental_base"
        or int(base_state["iteration"]) != base_epoch
        or not isinstance(base_state["model"], Mapping)
        or not isinstance(base_state["optimizer"], Mapping)
        or not isinstance(base_state["lr_scheduler"], Mapping)
    ):
        raise ValueError("SemanticFixed base is incomplete or incompatible")
    binding = _binding(base_state["binding"], field="base binding")
    ids = _parameter_ids(base_state["parameter_ids"])
    if binding.rank != logical_rank:
        raise ValueError("SemanticFixed base cannot be remapped to another logical rank")
    if base_id != f"sha256:{checkpoint_state_checksum(base_state)}":
        raise ValueError("SemanticFixed base content does not match its sealed id")
    _runtime(base_state["runtime_state"], epoch=base_epoch)
    return binding, ids


def _replay_optimizer(optimizer: Any) -> tuple[torch.optim.Optimizer, bool]:
    inner = getattr(optimizer, "optimizer", optimizer)
    wrapped = inner is not optimizer
    if not isinstance(inner, torch.optim.Optimizer) or not callable(
        getattr(optimizer, "load_state_dict", None)
    ):
        raise ValueError("SemanticFixed recovery requires an AdamW optimizer")
    if wrapped and not all(
        callable(getattr(optimizer, name, None))
        for name in ("refresh_model_params",)
    ):
        raise ValueError("SemanticFixed optimizer wrapper is incomplete")
    return inner, wrapped


def restore_semantic_fixed_object_suffix(
    client: ReplicaServiceClient,
    *,
    surviving_host: int,
    base_state: Mapping[str, Any],
    base_id: str,
    base_epoch: int,
    locators: Sequence[ObjectUpdateLocator],
    plans_by_epoch: Mapping[int, ActiveGradientPlan],
    logical_rank: int,
    module: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    runtime_restore: Callable[[Mapping[str, Any]], int],
) -> int:
    """Load one sealed base, then replay its ordered object-update suffix.

    ``logical_rank`` selects the persisted logical record; it is not the
    recovering process rank.  A changed logical layout is rejected because the
    current base and opaque runtime records have no transformation contract.
    The base id is the deterministic ``checkpoint_state_checksum`` of the
    complete base state and is checked before any live state changes.
    """
    if (
        not isinstance(base_epoch, int)
        or base_epoch < 0
        or not isinstance(logical_rank, int)
        or logical_rank < 0
        or _BASE_ID.fullmatch(base_id) is None
        or not callable(runtime_restore)
        or not all(
            callable(getattr(scheduler, name, None))
            for name in ("load_state_dict", "step")
        )
    ):
        raise ValueError("SemanticFixed recovery inputs are invalid")
    replay_optimizer, wrapped_optimizer = _replay_optimizer(optimizer)
    binding, parameter_ids = _validate_base(
        base_state,
        base_id=base_id,
        base_epoch=base_epoch,
        logical_rank=logical_rank,
    )
    if wrapped_optimizer and (
        int(base_state["optimizer"].get("global_step", -1)) != base_epoch
        or not isinstance(base_state["optimizer"].get("cur_iter"), int)
    ):
        raise ValueError("SemanticFixed wrapper base optimizer epoch is inconsistent")
    ordered = tuple(locators)
    epochs = tuple(locator.epoch for locator in ordered)
    if epochs != tuple(range(base_epoch + 1, base_epoch + len(ordered) + 1)):
        raise ValueError("SemanticFixed locators must form an ordered contiguous suffix")
    if set(plans_by_epoch) != set(epochs):
        raise ValueError("SemanticFixed suffix plans must match locator epochs")

    records: list[Mapping[str, Any]] = []
    runtimes: list[Mapping[str, Any]] = []
    for locator in ordered:
        plan = plans_by_epoch[locator.epoch]
        planned = {item.rank: item for item in plan.rank_bindings}.get(logical_rank)
        if (
            planned is None
            or planned.data_rank != binding.data_parallel_rank
            or planned.pipeline_rank != binding.pipeline_parallel_rank
            or planned.tensor_rank != binding.tensor_parallel_rank
        ):
            raise ValueError("SemanticFixed object plan changes the logical layout")
        control = fetch_object_update_control(
            client, surviving_host=surviving_host, locator=locator
        )
        encoding = recover_gradient_encoding(
            plan, control.value["gradient_encoding"], epoch=locator.epoch
        )
        restored = restore_object_update_epoch(
            client,
            surviving_host=surviving_host,
            locator=locator,
            plan=plan,
            encoding=encoding,
            logical_rank=logical_rank,
            fetched_control=control,
        )
        record = dict(restored.records).get(logical_rank)
        if record is None:
            raise ValueError("SemanticFixed suffix lacks the target logical rank")
        record_binding = _binding(record.get("binding"), field="update binding")
        if record_binding != binding:
            raise ValueError("SemanticFixed update cannot be remapped to another logical rank")
        if _parameter_ids(record.get("parameter_ids")) != parameter_ids:
            raise ValueError("SemanticFixed update parameter FQNs changed")
        validate_optimizer_update(
            replay_optimizer,
            record,
            expected_parameter_ids=parameter_ids,
            expected_binding=binding,
            expected_base_id=base_id,
            expected_base_epoch=base_epoch,
            expected_epoch=locator.epoch,
        )
        records.append(record)
        runtimes.append(_runtime(record.get("runtime_state"), epoch=locator.epoch))

    module.load_state_dict(copy.deepcopy(dict(base_state["model"])), strict=True)
    optimizer.load_state_dict(copy.deepcopy(dict(base_state["optimizer"])))
    scheduler.load_state_dict(copy.deepcopy(dict(base_state["lr_scheduler"])))
    for epoch, record in enumerate(records, start=base_epoch + 1):
        apply_optimizer_update(
            replay_optimizer,
            record,
            expected_parameter_ids=parameter_ids,
            expected_binding=binding,
            expected_base_id=base_id,
            expected_base_epoch=base_epoch,
            expected_epoch=epoch,
        )
        scheduler.step()
    final_epoch = epochs[-1] if epochs else base_epoch
    if wrapped_optimizer:
        optimizer.global_step = final_epoch
        optimizer.cur_iter += len(records)
        optimizer.refresh_model_params()
    runtime = (
        runtimes[-1]
        if runtimes
        else _runtime(base_state["runtime_state"], epoch=base_epoch)
    )
    iteration = int(runtime_restore(copy.deepcopy(dict(runtime))))
    if iteration != final_epoch:
        raise ValueError("SemanticFixed runtime restore returned a different iteration")
    return iteration
