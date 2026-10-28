"""Bounded CPU materialization for Merak AdamW update logs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from .catalog import snapshot_to_cpu
from .incremental_checkpoint import (
    RecoveryBinding,
    apply_optimizer_update,
    validate_optimizer_update,
)
from .adacheck_semantic import semantic_binding_from_state


class AdamWCheckpointMaterializer:
    """Consume a base and contiguous updates while retaining only current state."""

    def __init__(
        self,
        base_state: Mapping[str, Any],
        *,
        base_id: str,
        binding: RecoveryBinding,
        parameter_ids: Sequence[Sequence[str]],
    ) -> None:
        state = snapshot_to_cpu(base_state)
        base_epoch = int(state.get("iteration", -1))
        if (
            state.get("schema_version") != 1
            or state.get("kind") != "merak_incremental_base"
            or base_epoch < 0
        ):
            raise ValueError("materializer requires a Merak incremental base")
        identifiers = tuple(tuple(group) for group in parameter_ids)
        if state.get("parameter_ids") != [list(group) for group in identifiers]:
            raise ValueError("materializer parameter identity does not match its base")
        wrapper = state.get("optimizer")
        if not isinstance(wrapper, Mapping):
            raise ValueError("materializer base has no optimizer state")
        if int(wrapper.get("global_step", -1)) != base_epoch:
            raise ValueError("materializer base optimizer epoch is inconsistent")
        if wrapper.get("dynamic_loss_scale"):
            raise ValueError("materializer requires static BF16 loss scaling")

        saved_groups = wrapper.get("fp32_groups")
        inner_state = wrapper.get("optimizer_state_dict")
        if not isinstance(saved_groups, Sequence) or not isinstance(
            inner_state, Mapping
        ):
            raise ValueError("materializer base has incomplete AdamW state")
        saved_options = inner_state.get("param_groups")
        if not isinstance(saved_options, Sequence) or len(saved_options) != len(
            saved_groups
        ):
            raise ValueError("materializer AdamW parameter groups are inconsistent")

        self.parameters = [
            [torch.nn.Parameter(value.detach(), requires_grad=True) for value in group]
            for group in saved_groups
        ]
        groups = []
        for parameters, options in zip(self.parameters, saved_options):
            group = {key: value for key, value in options.items() if key != "params"}
            group["params"] = parameters
            groups.append(group)
        self.optimizer = torch.optim.AdamW(groups)
        self.optimizer.load_state_dict(inner_state)
        # The live optimizer owns the masters and Adam state; retain no stale copy.
        del wrapper["optimizer_state_dict"]
        del wrapper["fp32_groups"]
        self.state = state
        self.semantic_binding = semantic_binding_from_state(state)
        self.semantic_encoder = state.get("adacheck_semantic_encoder")
        if self.semantic_encoder is not None and not isinstance(
            self.semantic_encoder, Mapping
        ):
            raise ValueError("materializer base has invalid SemanticFixed evidence")
        self.base_id = base_id
        self.base_epoch = base_epoch
        self.current_epoch = base_epoch
        self.binding = binding
        self.parameter_ids = identifiers
        self.base_cur_iter = int(wrapper["cur_iter"])
        self.scheduler_before_step: Mapping[str, Any] | None = None
        self.runtime_state = snapshot_to_cpu(state.get("runtime_state", {}))

    def validate(self, record: Mapping[str, Any], *, epoch: int) -> None:
        if self.semantic_binding is not None:
            self.semantic_binding.validate_update(record)
        metadata = record.get("metadata")
        if self.semantic_encoder is not None and (
            not isinstance(metadata, Mapping)
            or metadata.get("adacheck_semantic_encoder") != self.semantic_encoder
        ):
            raise ValueError("materialized update has a different SemanticFixed plan")
        if not isinstance(metadata, Mapping) or not isinstance(
            metadata.get("scheduler_state_before_step"), Mapping
        ):
            raise ValueError("materialized update lacks scheduler state")
        runtime = record.get("runtime_state")
        if not isinstance(runtime, Mapping) or (
            "iteration" in self.runtime_state
            and int(runtime.get("iteration", -1)) != epoch
        ):
            raise ValueError("materialized update lacks matching runtime state")
        validate_optimizer_update(
            self.optimizer,
            record,
            expected_parameter_ids=self.parameter_ids,
            expected_binding=self.binding,
            expected_base_id=self.base_id,
            expected_base_epoch=self.base_epoch,
            expected_epoch=epoch,
        )

    def apply(self, record: Mapping[str, Any]) -> None:
        next_epoch = self.current_epoch + 1
        self.validate(record, epoch=next_epoch)
        metadata = record["metadata"]
        apply_optimizer_update(
            self.optimizer,
            record,
            expected_parameter_ids=self.parameter_ids,
            expected_binding=self.binding,
            expected_base_id=self.base_id,
            expected_base_epoch=self.base_epoch,
            expected_epoch=next_epoch,
        )
        self.scheduler_before_step = snapshot_to_cpu(
            metadata["scheduler_state_before_step"]
        )
        if record["runtime_state"]:
            self.runtime_state = snapshot_to_cpu(record["runtime_state"])
        self.current_epoch = next_epoch

    def snapshot(
        self, *, source_manifest: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        state = snapshot_to_cpu(self.state)
        model = state["model"]
        for identifiers, parameters in zip(self.parameter_ids, self.parameters):
            for identifier, parameter in zip(identifiers, parameters):
                if identifier not in model or not isinstance(
                    model[identifier], torch.Tensor
                ):
                    raise ValueError(
                        f"materialized parameter is absent from model state: {identifier}"
                    )
                model[identifier] = parameter.detach().to(model[identifier].dtype)

        wrapper = state["optimizer"]
        wrapper["optimizer_state_dict"] = snapshot_to_cpu(
            self.optimizer.state_dict()
        )
        wrapper["fp32_groups"] = [
            [parameter.detach().clone() for parameter in group]
            for group in self.parameters
        ]
        applied = self.current_epoch - self.base_epoch
        wrapper["global_step"] = self.current_epoch
        wrapper["cur_iter"] = self.base_cur_iter + applied
        state["iteration"] = self.current_epoch
        state["kind"] = "merak_materialized_checkpoint"
        state["runtime_state"] = snapshot_to_cpu(self.runtime_state)
        state["materialization"] = {
            "origin_base_epoch": self.base_epoch,
            "origin_base_id": self.base_id,
            "applied_updates": applied,
            "scheduler_steps_after_snapshot": int(applied > 0),
            "optimizer_replay": "cpu_numerical",
        }
        if source_manifest is not None:
            state["materialization"]["source_manifest"] = dict(source_manifest)
        if self.scheduler_before_step is not None:
            state["lr_scheduler"] = snapshot_to_cpu(self.scheduler_before_step)
        return state
