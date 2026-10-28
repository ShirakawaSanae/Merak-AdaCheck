"""Restore-only live controller for the independent ObjectRecovery path."""

from __future__ import annotations

import copy
import importlib
from collections.abc import Callable, Mapping
from typing import Any

from .object_recovery import RestoredRankState
from .object_recovery_adapter import apply_live_object_tensors


_OPAQUE_KEYS = frozenset(("scheduler", "rng", "sampler", "progress"))
_RNG_KEYS = frozenset(
    (
        "random_rng_state",
        "np_rng_state",
        "torch_rng_state",
        "cuda_rng_state",
        "rng_tracker_states",
    )
)
_OPTIMIZER_KEYS = frozenset(
    ("dynamic_loss_scale", "cur_scale", "cur_iter", "global_step")
)
_DYNAMIC_OPTIMIZER_KEYS = frozenset(
    ("last_overflow_iter", "scale_factor", "scale_window")
)


def _validated_opaque(
    opaque: Mapping[str, Any],
    *,
    optimizer: Any,
    scheduler: Any,
    sampler: Any,
    expected_iteration: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], int]:
    if set(opaque) != _OPAQUE_KEYS:
        raise ValueError("object restore requires scheduler, rng, sampler, and progress")
    scheduler_state, rng, sampler_state, progress = (
        opaque["scheduler"],
        opaque["rng"],
        opaque["sampler"],
        opaque["progress"],
    )
    if not all(
        isinstance(value, Mapping)
        for value in (scheduler_state, rng, sampler_state, progress)
    ):
        raise ValueError("object restore opaque roots must be mappings")
    if set(rng) != _RNG_KEYS:
        raise ValueError("object restore RNG state is incomplete")
    if set(progress) != {"iteration", "optimizer"} or not isinstance(
        progress["optimizer"], Mapping
    ):
        raise ValueError("object restore progress state is incomplete")
    iteration = int(progress["iteration"])
    if iteration != expected_iteration:
        raise ValueError("object restore progress differs from the requested epoch")
    optimizer_state = progress["optimizer"]
    expected_optimizer_keys = set(_OPTIMIZER_KEYS)
    if bool(optimizer_state.get("dynamic_loss_scale", False)):
        expected_optimizer_keys.update(_DYNAMIC_OPTIMIZER_KEYS)
    if set(optimizer_state) != expected_optimizer_keys:
        raise ValueError("HalfPrecisionOptimizer progress state is incompatible")
    for name in _OPTIMIZER_KEYS:
        if not hasattr(optimizer, name):
            raise ValueError("object restore requires a HalfPrecisionOptimizer")
    if sampler is None and sampler_state:
        raise ValueError("object restore has sampler state but no live sampler")
    for name in sampler_state:
        if name not in {"consumed_samples", "epoch"} or not hasattr(sampler, name):
            raise ValueError(f"unsupported sampler state field: {name}")
    if set(scheduler_state) != set(scheduler.state_dict()):
        raise ValueError("object restore scheduler state is incompatible")
    return scheduler_state, rng, sampler_state, iteration


def apply_live_object_state(
    state: RestoredRankState,
    module: Any,
    optimizer: Any,
    scheduler: Any,
    *,
    sampler: Any,
    expected_iteration: int,
    tensor_parallel_rank: int,
    tensor_parallel_world_size: int,
    runtime_restore: Callable[[Mapping[str, Any]], int],
) -> int:
    """Validate a sealed rank state, then apply every live training root."""
    if not isinstance(state, RestoredRankState):
        raise TypeError("object state loader must return RestoredRankState")
    scheduler_state, rng, sampler_state, iteration = _validated_opaque(
        state.opaque,
        optimizer=optimizer,
        scheduler=scheduler,
        sampler=sampler,
        expected_iteration=expected_iteration,
    )
    apply_live_object_tensors(
        module,
        optimizer,
        state.tensors,
        tensor_parallel_rank=tensor_parallel_rank,
        tensor_parallel_world_size=tensor_parallel_world_size,
    )
    scheduler.load_state_dict(copy.deepcopy(dict(scheduler_state)))
    optimizer_state = state.opaque["progress"]["optimizer"]
    for name, value in optimizer_state.items():
        setattr(optimizer, name, value)
    runtime = dict(rng)
    runtime["iteration"] = iteration
    runtime["sampler"] = dict(sampler_state)
    restored_iteration = int(runtime_restore(runtime))
    if restored_iteration != iteration:
        raise ValueError("runtime restore returned a different iteration")
    return restored_iteration


class LiveObjectRecoveryController:
    """Bind a sealed-state loader to one freshly initialized Merak rank."""

    def __init__(
        self,
        module: Any,
        optimizer: Any,
        scheduler: Any,
        *,
        sampler: Any,
        tensor_parallel_rank: int,
        tensor_parallel_world_size: int,
        state_loader: Callable[[int], RestoredRankState],
        runtime_restore: Callable[[Mapping[str, Any]], int] | None = None,
    ) -> None:
        if runtime_restore is None:
            from .merak_incremental import restore_runtime_state

            runtime_restore = lambda state: restore_runtime_state(state, sampler)
        self.module = module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sampler = sampler
        self.tensor_parallel_rank = tensor_parallel_rank
        self.tensor_parallel_world_size = tensor_parallel_world_size
        self.state_loader = state_loader
        self.runtime_restore = runtime_restore

    def restore(self, epoch: int) -> int:
        if epoch < 0:
            raise ValueError("object restore epoch must be non-negative")
        return apply_live_object_state(
            self.state_loader(epoch),
            self.module,
            self.optimizer,
            self.scheduler,
            sampler=self.sampler,
            expected_iteration=epoch,
            tensor_parallel_rank=self.tensor_parallel_rank,
            tensor_parallel_world_size=self.tensor_parallel_world_size,
            runtime_restore=self.runtime_restore,
        )


def load_object_recovery_controller(factory_spec: str, **kwargs: Any) -> Any:
    """Load the opt-in experiment factory without embedding storage policy."""
    module_name, separator, factory_name = factory_spec.partition(":")
    if not separator or not module_name or not factory_name or ":" in factory_name:
        raise ValueError("object recovery factory must use module:function")
    factory = getattr(importlib.import_module(module_name), factory_name)
    controller = factory(**kwargs)
    if not callable(getattr(controller, "restore", None)):
        raise TypeError("object recovery factory must return a restore controller")
    return controller
