"""Bridge optimizer-update records to confirmed semantic gradient objects."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .catalog import tensor_checksum
from .incremental_checkpoint import SCHEMA_VERSION
from .object_semantic_plan import (
    ActiveGradientPlan,
    ConfirmedGradientEncoding,
    confirm_gradient_equivalence,
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
class GradientPayload:
    version_id: str
    object_id: str
    epoch: int
    fqn: str
    parameter_id: str
    shape: tuple[int, ...]
    dtype: str
    checksum: str
    tensor: torch.Tensor


@dataclass(frozen=True)
class ObjectOptimizerUpdate:
    epoch: int
    detector_catalog_hash: str
    semantic_plan_hash: str
    plan_hash: str
    canonical_hash: str
    encoding_hash: str
    records: tuple[tuple[int, dict[str, Any]], ...]
    payloads: tuple[GradientPayload, ...]


def _encoding_digest(
    plan: ActiveGradientPlan, encoding: ConfirmedGradientEncoding
) -> str:
    return _digest(
        {
            "canonical_hash": plan.canonical_hash,
            "epoch": encoding.epoch,
            "actions": [
                (
                    action.object_id,
                    action.version_id,
                    action.source_rank,
                    action.source_host,
                    action.target_host,
                    action.action,
                    action.checksum,
                    action.logical_bytes,
                )
                for action in encoding.actions
            ],
            "mismatched_object_ids": sorted(encoding.mismatched_object_ids),
            "required_versions_by_host": encoding.required_versions_by_host,
            "failure_sets": plan.failure_sets,
        }
    )


def _validate_provenance(
    plan: ActiveGradientPlan, encoding: ConfirmedGradientEncoding
) -> None:
    for name in (
        "detector_catalog_hash",
        "semantic_plan_hash",
        "plan_hash",
        "canonical_hash",
    ):
        _sha256(getattr(plan, name), name)
    _sha256(encoding.encoding_hash, "encoding_hash")
    if encoding.canonical_hash != plan.canonical_hash:
        raise ValueError("gradient encoding has a different canonical hash")
    if encoding.failure_sets != plan.failure_sets:
        raise ValueError("gradient encoding has different failure provenance")
    if encoding.encoding_hash != _encoding_digest(plan, encoding):
        raise ValueError("gradient encoding hash does not match its contents")


def _record_slots(record: Mapping[str, Any]) -> list[tuple[int, int, str, Any]]:
    identifiers = record.get("parameter_ids")
    gradients = record.get("gradients")
    if (
        not isinstance(identifiers, Sequence)
        or isinstance(identifiers, (str, bytes))
        or not isinstance(gradients, Sequence)
        or isinstance(gradients, (str, bytes))
        or len(identifiers) != len(gradients)
    ):
        raise ValueError("optimizer update has invalid parameter or gradient groups")
    slots = []
    for group_index, (group_ids, group_gradients) in enumerate(
        zip(identifiers, gradients)
    ):
        if (
            not isinstance(group_ids, Sequence)
            or isinstance(group_ids, (str, bytes))
            or not isinstance(group_gradients, Sequence)
            or isinstance(group_gradients, (str, bytes))
            or len(group_ids) != len(group_gradients)
        ):
            raise ValueError("optimizer update parameter and gradient counts differ")
        for parameter_index, (parameter_id, gradient) in enumerate(
            zip(group_ids, group_gradients)
        ):
            if not isinstance(parameter_id, str) or not parameter_id:
                raise ValueError("optimizer parameter ids must be non-empty strings")
            slots.append((group_index, parameter_index, parameter_id, gradient))
    return slots


def _actions_by_reference(
    plan: ActiveGradientPlan, encoding: ConfirmedGradientEncoding
) -> dict[tuple[str, int | None], Any]:
    objects = {item.object_id: item for item in plan.objects}
    actions: dict[tuple[str, int | None], Any] = {}
    for action in encoding.actions:
        if action.object_id not in objects:
            raise ValueError("gradient encoding refers to an unknown object")
        key = (
            action.object_id,
            action.source_rank if action.action == "direct_fallback" else None,
        )
        previous = actions.setdefault(key, action)
        if (
            previous.version_id != action.version_id
            or previous.checksum != action.checksum
            or previous.action != action.action
        ):
            raise ValueError("gradient encoding has conflicting object actions")
    return actions


def _externalize_rank(
    plan: ActiveGradientPlan,
    encoding: ConfirmedGradientEncoding,
    rank: int,
    source: Mapping[str, Any],
    actions: Mapping[tuple[str, int | None], Any],
    *,
    owned_snapshot: bool = False,
    allow_sparse: bool = False,
    trust_encoding_checksum: bool = False,
) -> tuple[
    dict[str, Any],
    dict[str, GradientPayload],
    dict[tuple[int, str], torch.Tensor],
]:
    bindings = {binding.rank: binding for binding in plan.rank_bindings}
    if rank not in bindings:
        raise ValueError("optimizer update rank is not present in the semantic plan")
    if not isinstance(source, Mapping):
        raise ValueError("optimizer update must be a mapping")
    if source.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported optimizer-update schema")
    if source.get("kind") != "optimizer_update":
        raise ValueError("record is not an optimizer update")
    if source.get("epoch") != encoding.epoch:
        raise ValueError("optimizer update and encoding epochs differ")
    binding = source.get("binding")
    expected_binding = bindings[rank]
    if not isinstance(binding, Mapping) or (
        binding.get("rank") != rank
        or binding.get("data_parallel_rank") != expected_binding.data_rank
        or binding.get("pipeline_parallel_rank") != expected_binding.pipeline_rank
        or binding.get("tensor_parallel_rank") != expected_binding.tensor_rank
    ):
        raise ValueError("optimizer update belongs to a different planned rank")

    rank_objects = {item.fqn: item for item in plan.objects if rank in item.holders}
    slots = _record_slots(source)
    slot_ids = {parameter_id for _, _, parameter_id, _ in slots}
    non_null_ids = {
        parameter_id for _, _, parameter_id, value in slots if value is not None
    }
    if not set(rank_objects) <= slot_ids or not non_null_ids <= set(rank_objects) or (
        non_null_ids != set(rank_objects)
        if not allow_sparse
        else False
    ):
        raise ValueError("gradient FQN/parameter ids differ from the semantic plan")
    published_versions = {
        action.version_id for action in encoding.actions if action.source_rank == rank
    }
    # The owned runtime holds this CPU record until synchronous serialization ends.
    # Its gradient entries become references below, so never deepcopy discarded tensors.
    template_source = dict(source)
    if owned_snapshot:
        template_source["gradients"] = [
            [None] * len(group) for group in source["gradients"]
        ]
    template = copy.deepcopy(template_source)
    payloads: dict[str, GradientPayload] = {}
    observed = {}
    for group_index, parameter_index, parameter_id, gradient in slots:
        if parameter_id not in rank_objects:
            continue
        item = rank_objects[parameter_id]
        action = actions.get(
            (
                item.object_id,
                rank if item.object_id in encoding.mismatched_object_ids else None,
            )
        )
        if action is None:
            raise ValueError("gradient encoding is missing a rank-specific action")
        publishes = action.version_id in published_versions
        if gradient is None:
            if publishes:
                raise ValueError("sparse snapshot omits a planned publication")
            checksum = action.checksum
        else:
            if not isinstance(gradient, torch.Tensor):
                raise ValueError("optimizer update gradients must be tensors or null")
            if tuple(gradient.shape) != item.lengths or str(gradient.dtype) != item.dtype:
                raise ValueError("gradient shape or dtype differs from semantic object")
            checksum = (
                action.checksum if trust_encoding_checksum else tensor_checksum(gradient)
            )
            if checksum != action.checksum:
                raise ValueError("gradient checksum differs from confirmed encoding")
            observed[(rank, item.object_id)] = gradient
        if publishes:
            assert isinstance(gradient, torch.Tensor)
            if owned_snapshot and gradient.device.type != "cpu":
                raise ValueError("owned snapshots require CPU gradient tensors")
            payload = GradientPayload(
                action.version_id,
                item.object_id,
                encoding.epoch,
                item.fqn,
                parameter_id,
                tuple(gradient.shape),
                str(gradient.dtype),
                checksum,
                gradient.detach() if owned_snapshot else gradient.detach().cpu().clone(),
            )
            payloads.setdefault(payload.version_id, payload)
        template["gradients"][group_index][parameter_index] = {
            "kind": "semantic_gradient_ref",
            "version_id": action.version_id,
            "object_id": item.object_id,
            "epoch": encoding.epoch,
            "fqn": item.fqn,
            "parameter_id": parameter_id,
            "shape": item.lengths,
            "dtype": item.dtype,
            "checksum": checksum,
        }
    if set(payloads) != published_versions:
        raise ValueError("rank cannot resolve every version it must publish")
    return template, payloads, observed


def _update(
    plan: ActiveGradientPlan,
    encoding: ConfirmedGradientEncoding,
    records: tuple[tuple[int, dict[str, Any]], ...],
    payloads: Mapping[str, GradientPayload],
) -> ObjectOptimizerUpdate:
    return ObjectOptimizerUpdate(
        encoding.epoch,
        plan.detector_catalog_hash,
        plan.semantic_plan_hash,
        plan.plan_hash,
        plan.canonical_hash,
        encoding.encoding_hash,
        records,
        tuple(payloads[key] for key in sorted(payloads)),
    )


def externalize_local_optimizer_update(
    plan: ActiveGradientPlan,
    encoding: ConfirmedGradientEncoding,
    rank: int,
    record: Mapping[str, Any],
) -> ObjectOptimizerUpdate:
    """Externalize one rank, retaining payloads only when it is the publisher."""
    _validate_provenance(plan, encoding)
    if encoding.epoch < 0:
        raise ValueError("gradient encoding epoch must be non-negative")
    template, payloads, _ = _externalize_rank(
        plan, encoding, rank, record, _actions_by_reference(plan, encoding)
    )
    return _update(plan, encoding, ((rank, template),), payloads)


def externalize_owned_snapshot_optimizer_update(
    plan: ActiveGradientPlan,
    encoding: ConfirmedGradientEncoding,
    rank: int,
    record: Mapping[str, Any],
) -> ObjectOptimizerUpdate:
    """Externalize a private CPU snapshot retained through synchronous publication."""
    _validate_provenance(plan, encoding)
    if encoding.epoch < 0:
        raise ValueError("gradient encoding epoch must be non-negative")
    template, payloads, _ = _externalize_rank(
        plan,
        encoding,
        rank,
        record,
        _actions_by_reference(plan, encoding),
        owned_snapshot=True,
    )
    return _update(plan, encoding, ((rank, template),), payloads)


def externalize_sparse_owned_snapshot_optimizer_update(
    plan: ActiveGradientPlan,
    encoding: ConfirmedGradientEncoding,
    rank: int,
    record: Mapping[str, Any],
) -> ObjectOptimizerUpdate:
    """Externalize a DP-confirmed snapshot containing only verifiers/publishers."""
    _validate_provenance(plan, encoding)
    if encoding.mismatched_object_ids:
        raise ValueError("sparse snapshots do not support direct fallback")
    template, payloads, _ = _externalize_rank(
        plan,
        encoding,
        rank,
        record,
        _actions_by_reference(plan, encoding),
        owned_snapshot=True,
        allow_sparse=True,
        trust_encoding_checksum=True,
    )
    return _update(plan, encoding, ((rank, template),), payloads)


def externalize_optimizer_updates(
    plan: ActiveGradientPlan,
    encoding: ConfirmedGradientEncoding,
    records: Mapping[int, Mapping[str, Any]],
) -> ObjectOptimizerUpdate:
    """CPU oracle: externalize and cross-confirm every planned rank."""
    _validate_provenance(plan, encoding)
    if encoding.epoch < 0:
        raise ValueError("gradient encoding epoch must be non-negative")
    bindings = {binding.rank: binding for binding in plan.rank_bindings}
    if set(records) != set(bindings):
        raise ValueError("optimizer records must cover every planned rank")
    actions = _actions_by_reference(plan, encoding)

    payloads: dict[str, GradientPayload] = {}
    observed_tensors = {}
    templates = []
    for rank, source in sorted(records.items()):
        template, local_payloads, observed = _externalize_rank(
            plan, encoding, rank, source, actions
        )
        for version_id, payload in local_payloads.items():
            previous = payloads.setdefault(version_id, payload)
            if (
                previous.object_id != payload.object_id
                or previous.parameter_id != payload.parameter_id
                or previous.checksum != payload.checksum
            ):
                raise ValueError(
                    "one semantic version resolves to conflicting payloads"
                )
        observed_tensors.update(observed)
        templates.append((rank, template))
    if confirm_gradient_equivalence(
        plan, observed_tensors, epoch=encoding.epoch
    ) != encoding:
        raise ValueError("gradient encoding differs from the optimizer records")
    return _update(plan, encoding, tuple(templates), payloads)


def materialize_optimizer_update(
    update: ObjectOptimizerUpdate,
    rank: int,
    *,
    payloads: Mapping[str, GradientPayload] | None = None,
    expected_epoch: int,
    detector_catalog_hash: str,
    semantic_plan_hash: str,
    plan_hash: str,
    canonical_hash: str,
    encoding_hash: str,
) -> dict[str, Any]:
    """Rebuild one standard record after validating all payload references."""
    expected = {
        "detector_catalog_hash": detector_catalog_hash,
        "semantic_plan_hash": semantic_plan_hash,
        "plan_hash": plan_hash,
        "canonical_hash": canonical_hash,
        "encoding_hash": encoding_hash,
    }
    for name, value in expected.items():
        _sha256(value, name)
        if getattr(update, name) != value:
            raise ValueError(f"object optimizer update has a different {name}")
    if update.epoch != expected_epoch:
        raise ValueError("object optimizer update has an unexpected epoch")
    records = dict(update.records)
    if len(records) != len(update.records) or rank not in records:
        raise ValueError("object optimizer update has missing or duplicate ranks")
    record = copy.deepcopy(records[rank])
    references = []
    reference_fields = {
        "kind",
        "version_id",
        "object_id",
        "epoch",
        "fqn",
        "parameter_id",
        "shape",
        "dtype",
        "checksum",
    }
    for group in record.get("gradients", ()):
        for value in group:
            if value is None:
                continue
            if (
                not isinstance(value, Mapping)
                or value.get("kind") != "semantic_gradient_ref"
                or set(value) != reference_fields
            ):
                raise ValueError(
                    "object optimizer update contains a non-reference gradient"
                )
            references.append(value)
    required = {value["version_id"] for value in references}
    if payloads is not None and any(
        not isinstance(payload, GradientPayload) or key != payload.version_id
        for key, payload in payloads.items()
    ):
        raise ValueError("semantic gradient payload mapping key disagrees with version")
    source_payloads = (
        tuple(payloads.values()) if payloads is not None else update.payloads
    )
    available = {payload.version_id: payload for payload in source_payloads}
    if len(available) != len(source_payloads):
        raise ValueError("semantic gradient payload ids are duplicated")
    if payloads is None:
        available = {key: available[key] for key in required if key in available}
    if set(available) != required:
        raise ValueError(
            "semantic gradient payload set has missing or unknown versions"
        )
    for group_index, group in enumerate(record["gradients"]):
        for parameter_index, reference in enumerate(group):
            if reference is None:
                continue
            payload = available[reference["version_id"]]
            fields = (
                "version_id",
                "object_id",
                "epoch",
                "fqn",
                "parameter_id",
                "shape",
                "dtype",
                "checksum",
            )
            if any(reference.get(name) != getattr(payload, name) for name in fields):
                raise ValueError("semantic gradient reference and payload disagree")
            if (
                payload.parameter_id
                != record["parameter_ids"][group_index][parameter_index]
                or payload.fqn != payload.parameter_id
            ):
                raise ValueError("semantic gradient FQN/parameter id changed")
            if payload.epoch != expected_epoch:
                raise ValueError("semantic gradient payload has an unexpected epoch")
            tensor = payload.tensor
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != payload.shape
                or str(tensor.dtype) != payload.dtype
                or tensor_checksum(tensor) != payload.checksum
            ):
                raise ValueError("semantic gradient payload tensor failed validation")
            record["gradients"][group_index][parameter_index] = tensor.detach().clone()
    return record
