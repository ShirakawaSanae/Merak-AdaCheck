"""Sealed, tensor-free control metadata for confirmed gradient encodings."""

from __future__ import annotations

from typing import Mapping

from .object_semantic_plan import (
    ActiveGradientPlan,
    ConfirmedGradientEncoding,
    confirm_gradient_checksums,
)


_CONTROL_FIELDS = frozenset({"schema_version", "provenance", "evidence"})
_PROVENANCE_FIELDS = frozenset(
    {"plan_hash", "canonical_hash", "encoding_hash", "epoch", "failure_sets"}
)
_EVIDENCE_FIELDS = frozenset({"rank", "object_id", "checksum"})


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _evidence_from_encoding(
    plan: ActiveGradientPlan, encoding: ConfirmedGradientEncoding
) -> list[dict[str, object]]:
    result = []
    for item in plan.objects:
        actions = [action for action in encoding.actions if action.object_id == item.object_id]
        kinds = {action.action for action in actions}
        if kinds == {"confirmed_object"}:
            checksums = {action.checksum for action in actions}
            if len(checksums) != 1:
                raise ValueError("confirmed object actions disagree on checksum")
            checksum = _sha256(next(iter(checksums)), "encoding checksum")
            result.extend(
                {"rank": rank, "object_id": item.object_id, "checksum": checksum}
                for rank in item.holders
            )
        elif kinds == {"direct_fallback"}:
            by_rank: dict[int, str] = {}
            for action in actions:
                if action.source_rank not in item.holders:
                    raise ValueError("fallback action has an unknown holder")
                checksum = _sha256(action.checksum, "encoding checksum")
                previous = by_rank.setdefault(action.source_rank, checksum)
                if previous != checksum:
                    raise ValueError("fallback actions disagree on holder checksum")
            if set(by_rank) != set(item.holders):
                raise ValueError("fallback actions do not cover the planned holders")
            result.extend(
                {"rank": rank, "object_id": item.object_id, "checksum": by_rank[rank]}
                for rank in item.holders
            )
        else:
            raise ValueError("encoding actions do not describe a planned object")
    return result


def _checksums_from_control(
    plan: ActiveGradientPlan, control: Mapping[str, object], epoch: int
) -> tuple[dict[tuple[int, str], str], str]:
    if not isinstance(control, dict) or set(control) != _CONTROL_FIELDS:
        raise ValueError("gradient encoding control schema changed")
    if control["schema_version"] != 1:
        raise ValueError("unsupported gradient encoding control schema")
    provenance = control["provenance"]
    evidence = control["evidence"]
    if not isinstance(provenance, dict) or set(provenance) != _PROVENANCE_FIELDS:
        raise ValueError("gradient encoding provenance schema changed")
    if (
        provenance["plan_hash"] != plan.plan_hash
        or provenance["canonical_hash"] != plan.canonical_hash
        or provenance["epoch"] != epoch
    ):
        raise ValueError("gradient encoding control has different plan or epoch")
    encoding_hash = _sha256(provenance["encoding_hash"], "encoding_hash")
    failure_sets = provenance["failure_sets"]
    expected_failures = [list(value) for value in plan.failure_sets]
    if failure_sets != expected_failures:
        raise ValueError("gradient encoding control has different failure sets")
    if not isinstance(evidence, list):
        raise ValueError("gradient encoding evidence must be a list")

    checksums: dict[tuple[int, str], str] = {}
    for value in evidence:
        if not isinstance(value, dict) or set(value) != _EVIDENCE_FIELDS:
            raise ValueError("gradient encoding evidence schema changed")
        rank, object_id = value["rank"], value["object_id"]
        if type(rank) is not int or rank < 0:
            raise ValueError("gradient encoding evidence rank is invalid")
        _sha256(object_id, "object_id")
        key = (rank, object_id)
        if key in checksums:
            raise ValueError("gradient encoding evidence is duplicated")
        checksums[key] = _sha256(value["checksum"], "checksum")
    expected = {
        (rank, item.object_id) for item in plan.objects for rank in item.holders
    }
    if set(checksums) != expected:
        raise ValueError("gradient encoding evidence does not cover planned holders")
    return checksums, encoding_hash


def recover_gradient_encoding(
    plan: ActiveGradientPlan, control: Mapping[str, object], *, epoch: int
) -> ConfirmedGradientEncoding:
    """Reconfirm a sealed epoch control without deserializing tensor payloads."""

    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("gradient encoding epoch must be non-negative")
    checksums, expected_hash = _checksums_from_control(plan, control, epoch)
    encoding = confirm_gradient_checksums(plan, checksums, epoch=epoch)
    if (
        encoding.canonical_hash != plan.canonical_hash
        or encoding.failure_sets != plan.failure_sets
        or encoding.encoding_hash != expected_hash
    ):
        raise ValueError("gradient encoding control was tampered")
    return encoding


def export_gradient_encoding_metadata(
    plan: ActiveGradientPlan, encoding: ConfirmedGradientEncoding
) -> dict[str, object]:
    """Export a sealed, JSON-like per-epoch control for a confirmed encoding."""

    control: dict[str, object] = {
        "schema_version": 1,
        "provenance": {
            "plan_hash": plan.plan_hash,
            "canonical_hash": encoding.canonical_hash,
            "encoding_hash": encoding.encoding_hash,
            "epoch": encoding.epoch,
            "failure_sets": [list(value) for value in encoding.failure_sets],
        },
        "evidence": _evidence_from_encoding(plan, encoding),
    }
    if recover_gradient_encoding(plan, control, epoch=encoding.epoch) != encoding:
        raise ValueError("encoding does not match its sealed control metadata")
    return control
