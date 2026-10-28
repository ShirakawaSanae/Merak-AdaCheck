"""Fresh-process ObjectRecovery factory for sealed SemanticFixed runs."""

from __future__ import annotations

import os
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ..amp.amp_optimizer import HalfPrecisionOptimizer
from .catalog import checkpoint_state_checksum
from .merak_incremental import restore_runtime_state
from .object_semantic_base import (
    load_object_semantic_base_locator,
    restore_object_semantic_base,
)
from .object_semantic_deduplicated_base import (
    restore_deduplicated_object_semantic_base,
)
from .object_semantic_recovery import restore_semantic_fixed_object_suffix
from .object_semantic_training import (
    SemanticFixedStaticContext,
    build_semantic_fixed_static_context,
)
from .object_update_transport import load_object_update_locator


def _integer(name: str, *, minimum: int = 0) -> int:
    try:
        value = int(os.environ[name])
    except (KeyError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _validate_live_inputs(optimizer: Any, args: Any) -> None:
    if not isinstance(optimizer, HalfPrecisionOptimizer) or not bool(
        getattr(args, "bf16", False)
    ):
        raise ValueError("SemanticFixed recovery requires the BF16 HalfPrecisionOptimizer")
    if getattr(args, "zero_stage", None) is not None:
        raise ValueError("SemanticFixed recovery does not support ZeRO")
    if getattr(args, "dataloader_num_workers", None) != 0:
        raise ValueError("SemanticFixed recovery requires dataloader_num_workers=0")
    if type(optimizer.optimizer) is not torch.optim.AdamW:
        raise ValueError("SemanticFixed recovery requires torch.optim.AdamW")


class SemanticFixedObjectRestoreController:
    """Restore one unchanged logical rank from a surviving object service."""

    def __init__(
        self,
        module: torch.nn.Module,
        optimizer: Any,
        scheduler: Any,
        *,
        sampler: Any,
        context: SemanticFixedStaticContext,
        base_epoch: int,
        surviving_host: int,
        base_mode: str = "rank_local_v1",
    ) -> None:
        if surviving_host not in (0, 1):
            raise ValueError("CKPT_OBJECT_SURVIVING_HOST must be 0 or 1")
        self.module = module
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sampler = sampler
        self.context = context
        self.base_epoch = base_epoch
        self.surviving_host = surviving_host
        if base_mode not in {"rank_local_v1", "dp_deduplicated_v1"}:
            raise ValueError("SemanticFixed base mode is unsupported")
        self.base_mode = base_mode

    def _base_locator_path(self) -> Path:
        return self.context.root / "base" / f"epoch-{self.base_epoch:08d}.json"

    def _update_locator_path(self, epoch: int) -> Path:
        return self.context.root / "updates" / f"epoch-{epoch:08d}.json"

    def _write_metrics(self, *, requested_epoch: int, final_epoch: int, base: Any, wall_ms: float) -> None:
        path = self.context.root / "recovery_metrics" / (
            f"rank_{self.context.binding.rank:05d}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        components = {
            "model": self.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict(),
        }
        component_checksums = {
            name: checkpoint_state_checksum(state) for name, state in components.items()
        }
        value = {
            "rank": self.context.binding.rank,
            "requested_epoch": requested_epoch,
            "base_epoch": self.base_epoch,
            "base_mode": self.base_mode,
            "final_epoch": final_epoch,
            "arm": self.context.arm,
            "plan_hash": self.context.plan.plan_hash,
            "mapping_hash": self.context.mapping_hash,
            "surviving_host": self.surviving_host,
            "base_payload_bytes": base.payload_bytes,
            "base_wire_bytes": base.wire_bytes,
            "base_get_ms": base.get_ms,
            "suffix_get_metrics": "unavailable",
            "total_restore_wall_ms": wall_ms,
            "component_checksums": component_checksums,
            "boundary_checksum": checkpoint_state_checksum(components),
            "checksum_diagnostic_only": True,
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)

    def _validate_base_provenance(self, state: Mapping[str, Any]) -> None:
        expected_encoder = {
            "schema_version": 1,
            "kind": "adacheck_semantic_fixed",
            "plan_hash": self.context.semantic_plan.plan_hash,
            "trainable_action": "gradient_bf16",
            "gradient_dtype": "torch.bfloat16",
            "gradient_snapshot_point": "raw_bf16_pre_unscale",
            "gradient_transform": "upcast_fp32_times_exact_step_scale",
            "metadata_action": "metadata_exact",
            "protection_action": "object_semantic_runtime",
            "remote_trainable_payload": "semantic_gradient_bf16",
            "peer_equivalence": "post_allreduce_dp_invariant",
            "object_partition": "fqn_global_slice_pp_tp",
            "dp_equivalence_basis": "post_allreduce_training_invariant",
            "integrity": "semantic_tensor_sha256_plus_serialized_payload_sha256",
            "rank_action_count": len(
                self.context.semantic_plan.actions_for_rank(self.context.binding.rank)
            ),
            "global_action_count": len(self.context.semantic_plan.actions),
        }
        if (
            state.get("run_id") != self.context.run_id
            or state.get("adacheck_semantic") != self.context.semantic_binding.to_dict()
            or state.get("adacheck_semantic_encoder") != expected_encoder
            or state.get("adacheck_detector_evidence") != self.context.detector_evidence
        ):
            raise ValueError("SemanticFixed base arm or detector provenance changed")

    def restore(self, epoch: int) -> int:
        if type(epoch) is not int or epoch < self.base_epoch:
            raise ValueError("SemanticFixed restore epoch precedes its base")
        started = time.perf_counter()
        locator = load_object_semantic_base_locator(self._base_locator_path())
        if locator.run_id != self.context.run_id or locator.epoch != self.base_epoch:
            raise ValueError("SemanticFixed base locator continuity changed")
        restore_base = (
            restore_deduplicated_object_semantic_base
            if self.base_mode == "dp_deduplicated_v1"
            else restore_object_semantic_base
        )
        base = restore_base(
            self.context.clients[self.surviving_host],
            surviving_host=self.surviving_host,
            locator=locator,
            logical_rank=self.context.binding.rank,
            expected_rank_bindings=self.context.rank_bindings,
            expected_parameter_ids=self.context.parameter_ids,
        )
        self._validate_base_provenance(base.state)
        locators = []
        for update_epoch in range(self.base_epoch + 1, epoch + 1):
            try:
                update = load_object_update_locator(self._update_locator_path(update_epoch))
            except OSError as error:
                raise ValueError("SemanticFixed update locator suffix is not contiguous") from error
            if (
                update.run_id != self.context.run_id
                or update.epoch != update_epoch
                or update.world_size != len(self.context.rank_bindings)
                or update.arm != self.context.arm
                or update.detector_catalog_hash != self.context.plan.detector_catalog_hash
                or update.semantic_plan_hash != self.context.plan.semantic_plan_hash
                or update.canonical_hash != self.context.plan.canonical_hash
                or update.plan_hash != self.context.plan.plan_hash
            ):
                raise ValueError("SemanticFixed update arm, config, or plan changed")
            locators.append(update)
        final_epoch = restore_semantic_fixed_object_suffix(
            self.context.clients[self.surviving_host],
            surviving_host=self.surviving_host,
            base_state=base.state,
            base_id=base.base_id,
            base_epoch=self.base_epoch,
            locators=locators,
            plans_by_epoch={item.epoch: self.context.plan for item in locators},
            logical_rank=self.context.binding.rank,
            module=self.module,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            runtime_restore=lambda state: restore_runtime_state(state, self.sampler),
        )
        self._write_metrics(
            requested_epoch=epoch,
            final_epoch=final_epoch,
            base=base,
            wall_ms=(time.perf_counter() - started) * 1000.0,
        )
        return final_epoch


def build_object_semantic_restore_controller(
    module: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    *,
    sampler: Any,
    args: Any,
) -> SemanticFixedObjectRestoreController:
    """CKPT_OBJECT_RECOVERY_FACTORY entry point; intentionally not a source hook."""
    if os.environ.get("CKPT_OBJECT_SEMANTIC_ENABLE", "0") != "0":
        raise ValueError("SemanticFixed source and object recovery modes are mutually exclusive")
    _validate_live_inputs(optimizer, args)
    context = build_semantic_fixed_static_context(module, optimizer, args)
    base_mode = os.environ.get("CKPT_OBJECT_BASE_MODE", "rank_local_v1")
    if base_mode not in {"rank_local_v1", "dp_deduplicated_v1"}:
        raise ValueError("CKPT_OBJECT_BASE_MODE is unsupported")
    return SemanticFixedObjectRestoreController(
        module,
        optimizer,
        scheduler,
        sampler=sampler,
        context=context,
        base_epoch=_integer("CKPT_OBJECT_BASE_EPOCH"),
        surviving_host=_integer("CKPT_OBJECT_SURVIVING_HOST", minimum=0),
        base_mode=base_mode,
    )
