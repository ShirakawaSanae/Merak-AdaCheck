"""Identity binding for the recoverable AdaCheck semantic baseline."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .catalog import checkpoint_state_checksum, snapshot_to_cpu


_POLICIES = {"rank-k2", "host-f1"}
_ACTIONS = {
    "gradient_bf16",
    "direct_poststep",
    "peer_derived",
    "metadata_exact",
}


def _hash(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_value(value: Any, *, action: str) -> Any:
    copied = snapshot_to_cpu(value)
    if action == "gradient_bf16":
        if not isinstance(copied, torch.Tensor) or not copied.is_floating_point():
            raise ValueError("gradient_bf16 actions require a floating-point tensor")
        return copied.to(dtype=torch.bfloat16).contiguous()
    return copied


@dataclass(frozen=True)
class SemanticAction:
    """One frozen recovery action for one rank-local semantic value."""

    rank: int
    value_id: str
    action: str
    canonical_owner: int
    fallback_action: str | None = None

    def __post_init__(self) -> None:
        if self.rank < 0 or self.canonical_owner < 0 or not self.value_id:
            raise ValueError("semantic action has an invalid rank, owner, or value ID")
        if self.action not in _ACTIONS:
            raise ValueError(f"unsupported semantic action: {self.action}")
        if self.fallback_action is not None and self.fallback_action not in {
            "direct_poststep",
            "metadata_exact",
        }:
            raise ValueError("semantic fallback must be an exact direct action")
        if self.action != "peer_derived" and self.canonical_owner != self.rank:
            raise ValueError("non-peer semantic actions must be canonical locally")
        if self.action != "peer_derived" and self.fallback_action is not None:
            raise ValueError("only peer-derived semantic actions may declare a fallback")

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "rank": self.rank,
            "value_id": self.value_id,
            "action": self.action,
            "canonical_owner": self.canonical_owner,
        }
        if self.fallback_action is not None:
            value["fallback_action"] = self.fallback_action
        return value


@dataclass(frozen=True)
class SemanticEncoderPlan:
    """Frozen fixed-pair semantic encoding plan shared by baseline and M1."""

    policy: str
    catalog_hash: str
    parallel_plan_hash: str
    protection_plan_hash: str
    world_size: int
    pairs: tuple[tuple[int, int], ...]
    actions: tuple[SemanticAction, ...]

    def __post_init__(self) -> None:
        if self.policy not in _POLICIES:
            raise ValueError("AdaCheck policy must be rank-k2 or host-f1")
        _hash(self.catalog_hash, name="catalog_hash")
        _hash(self.parallel_plan_hash, name="parallel_plan_hash")
        _hash(self.protection_plan_hash, name="protection_plan_hash")
        if self.world_size < 2:
            raise ValueError("semantic plan requires at least two ranks")
        pairs = tuple(tuple(sorted(pair)) for pair in self.pairs)
        if pairs != self.pairs or any(len(pair) != 2 for pair in pairs):
            raise ValueError("semantic plan pairs must be sorted rank pairs")
        flattened = [rank for pair in pairs for rank in pair]
        if sorted(flattened) != list(range(self.world_size)):
            raise ValueError("semantic plan pairs must cover every rank exactly once")
        if tuple(sorted(self.actions, key=lambda item: (item.rank, item.value_id))) != self.actions:
            raise ValueError("semantic plan actions must use canonical rank/value order")
        keys = [(item.rank, item.value_id) for item in self.actions]
        if len(set(keys)) != len(keys):
            raise ValueError("semantic plan contains duplicate rank/value actions")
        peer = {
            rank: other
            for first, second in pairs
            for rank, other in ((first, second), (second, first))
        }
        by_key = {(item.rank, item.value_id): item for item in self.actions}
        for item in self.actions:
            if item.rank >= self.world_size or item.canonical_owner >= self.world_size:
                raise ValueError("semantic action rank is outside the plan world")
            if item.action != "peer_derived":
                continue
            if peer[item.rank] != item.canonical_owner:
                raise ValueError("peer-derived action must use the fixed peer owner")
            owner = by_key.get((item.canonical_owner, item.value_id))
            if owner is None or owner.action == "peer_derived":
                raise ValueError("peer-derived action lacks a terminal canonical owner")

    def actions_for_rank(self, rank: int) -> tuple[SemanticAction, ...]:
        if not 0 <= rank < self.world_size:
            raise ValueError("rank is outside the semantic plan world")
        return tuple(item for item in self.actions if item.rank == rank)

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": 1,
            "policy": self.policy,
            "catalog_hash": self.catalog_hash,
            "parallel_plan_hash": self.parallel_plan_hash,
            "protection_plan_hash": self.protection_plan_hash,
            "world_size": self.world_size,
            "pairs": [list(pair) for pair in self.pairs],
            "actions": [item.to_dict() for item in self.actions],
        }
        return {
            **body,
            "plan_hash": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
        }

    @property
    def plan_hash(self) -> str:
        return str(self.to_dict()["plan_hash"])


def build_gradient_semantic_plan(
    binding: "AdaCheckSemanticBinding",
    *,
    world_size: int,
    pairs: Sequence[Sequence[int]],
    parameter_ids_by_rank: Sequence[Sequence[Sequence[str]]],
) -> SemanticEncoderPlan:
    """Build the paper's half-precision-gradient action for every trainable value."""
    if len(parameter_ids_by_rank) != world_size:
        raise ValueError("semantic parameter topology must cover every rank")
    actions = []
    for rank, groups in enumerate(parameter_ids_by_rank):
        seen = set()
        for group in groups:
            for value_id in group:
                if not isinstance(value_id, str) or not value_id or value_id in seen:
                    raise ValueError("semantic parameter IDs must be unique per rank")
                seen.add(value_id)
                actions.append(
                    SemanticAction(rank, value_id, "gradient_bf16", rank)
                )
    return SemanticEncoderPlan(
        policy=binding.policy,
        catalog_hash=binding.catalog_hash,
        parallel_plan_hash=binding.parallel_plan_hash,
        protection_plan_hash=binding.protection_plan_hash,
        world_size=world_size,
        pairs=tuple(tuple(int(rank) for rank in pair) for pair in pairs),
        actions=tuple(sorted(actions, key=lambda item: (item.rank, item.value_id))),
    )


def encode_mixed_action_record(
    plan: SemanticEncoderPlan,
    *,
    rank: int,
    epoch: int,
    values: Mapping[str, Any],
    metadata: Mapping[str, Any],
    fallback_value_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Encode one immutable CPU semantic update record.

    A peer-derived value normally has no payload.  A fallback is permitted only
    when the frozen action explicitly declares an exact fallback action.
    """
    if epoch < 0:
        raise ValueError("semantic record epoch must be non-negative")
    actions = plan.actions_for_rank(rank)
    expected_ids = {item.value_id for item in actions}
    if set(values) != expected_ids:
        raise ValueError("semantic record values do not match the frozen plan")
    fallback_ids = set(fallback_value_ids)
    if not fallback_ids <= expected_ids:
        raise ValueError("semantic fallback value is absent from the frozen plan")
    by_key = {(item.rank, item.value_id): item for item in plan.actions}
    records = []
    for item in actions:
        fallback = item.value_id in fallback_ids
        if fallback and (item.action != "peer_derived" or item.fallback_action is None):
            raise ValueError("semantic fallback is not declared by the frozen plan")
        action = item.fallback_action if fallback else item.action
        assert action is not None
        if item.action == "peer_derived" and not fallback:
            owner = by_key[(item.canonical_owner, item.value_id)]
            action = owner.action
        value = _cpu_value(values[item.value_id], action=action)
        encoded: dict[str, Any] = {
            "value_id": item.value_id,
            "action": item.action,
            "canonical_owner": item.canonical_owner,
            "fallback": fallback,
            "value_checksum": checkpoint_state_checksum(value),
        }
        if item.action != "peer_derived" or fallback:
            encoded["payload"] = value
        records.append(encoded)
    return {
        "schema_version": 1,
        "kind": "adacheck_semantic_mixed_update",
        "plan_hash": plan.plan_hash,
        "rank": rank,
        "epoch": epoch,
        "actions": records,
        "metadata": snapshot_to_cpu(dict(metadata)),
        "metadata_checksum": checkpoint_state_checksum(metadata),
    }


def restore_mixed_action_record(
    plan: SemanticEncoderPlan,
    record: Mapping[str, Any],
    *,
    rank: int,
    epoch: int,
    canonical_values: Mapping[tuple[int, str], Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fail closed while resolving direct, gradient, peer, and metadata actions."""
    if (
        record.get("schema_version") != 1
        or record.get("kind") != "adacheck_semantic_mixed_update"
        or record.get("plan_hash") != plan.plan_hash
        or int(record.get("rank", -1)) != rank
        or int(record.get("epoch", -1)) != epoch
    ):
        raise ValueError("semantic mixed record identity mismatch")
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping) or (
        record.get("metadata_checksum") != checkpoint_state_checksum(metadata)
    ):
        raise ValueError("semantic mixed record metadata is missing or corrupt")
    encoded_actions = record.get("actions")
    if not isinstance(encoded_actions, Sequence) or isinstance(encoded_actions, (str, bytes)):
        raise ValueError("semantic mixed record actions are invalid")
    indexed = {
        str(item.get("value_id")): item
        for item in encoded_actions
        if isinstance(item, Mapping) and isinstance(item.get("value_id"), str)
    }
    expected = plan.actions_for_rank(rank)
    if len(indexed) != len(encoded_actions) or set(indexed) != {
        item.value_id for item in expected
    }:
        raise ValueError("semantic mixed record actions do not match the frozen plan")
    restored: dict[str, Any] = {}
    for action in expected:
        item = indexed[action.value_id]
        if (
            item.get("action") != action.action
            or int(item.get("canonical_owner", -1)) != action.canonical_owner
            or not isinstance(item.get("fallback"), bool)
        ):
            raise ValueError("semantic mixed record action identity mismatch")
        fallback = bool(item["fallback"])
        if fallback and (action.action != "peer_derived" or action.fallback_action is None):
            raise ValueError("semantic mixed record has an undeclared fallback")
        if action.action == "peer_derived" and not fallback:
            key = (action.canonical_owner, action.value_id)
            if key not in canonical_values:
                raise ValueError("semantic peer-derived value is unavailable")
            value = snapshot_to_cpu(canonical_values[key])
        else:
            if "payload" not in item:
                raise ValueError("semantic direct action lacks a payload")
            effective_action = action.fallback_action if fallback else action.action
            assert effective_action is not None
            value = _cpu_value(item["payload"], action=effective_action)
        if item.get("value_checksum") != checkpoint_state_checksum(value):
            raise ValueError("semantic action payload is corrupt")
        restored[action.value_id] = value
    return restored, snapshot_to_cpu(dict(metadata))


@dataclass(frozen=True)
class AdaCheckSemanticBinding:
    """The immutable detector, placement, and protection inputs of one run."""

    policy: str
    catalog_hash: str
    parallel_plan_hash: str
    protection_plan_hash: str

    def __post_init__(self) -> None:
        if self.policy not in _POLICIES:
            raise ValueError("AdaCheck policy must be rank-k2 or host-f1")
        _hash(self.catalog_hash, name="catalog_hash")
        _hash(self.parallel_plan_hash, name="parallel_plan_hash")
        _hash(self.protection_plan_hash, name="protection_plan_hash")

    def to_dict(self) -> dict[str, str | int]:
        body: dict[str, str | int] = {
            "schema_version": 1,
            "policy": self.policy,
            "catalog_hash": self.catalog_hash,
            "parallel_plan_hash": self.parallel_plan_hash,
            "protection_plan_hash": self.protection_plan_hash,
        }
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        return {**body, "binding_hash": hashlib.sha256(encoded).hexdigest()}

    def validate(self, value: Mapping[str, Any]) -> None:
        expected = self.to_dict()
        if dict(value) != expected:
            raise ValueError("AdaCheck semantic binding mismatch")

    def validate_base(self, state: Mapping[str, Any]) -> None:
        value = state.get("adacheck_semantic")
        if not isinstance(value, Mapping):
            raise ValueError("AdaCheck base lacks semantic binding")
        self.validate(value)

    def validate_update(self, record: Mapping[str, Any]) -> None:
        metadata = record.get("metadata")
        value = metadata.get("adacheck_semantic") if isinstance(metadata, Mapping) else None
        if not isinstance(value, Mapping):
            raise ValueError("AdaCheck update lacks semantic binding")
        self.validate(value)


def semantic_binding_from_detector_config(
    config_path: Path,
    *,
    expected_model: str,
    world_size: int,
    ranks_per_host: int,
    rank_bindings: Sequence[Mapping[str, Any]],
    pairs: Sequence[Sequence[int]],
) -> tuple[AdaCheckSemanticBinding, dict[str, Any]]:
    """Validate a two-pass detector artifact and bind it to the live topology."""
    config_path = config_path.resolve()
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("AdaCheck detector config is unreadable") from error
    if (
        not isinstance(config, Mapping)
        or config.get("run_kind") != "matched_detector_capture"
        or config.get("status_at_creation") != "capture_and_freeze_complete"
        or config.get("model_argument") != expected_model
        or config.get("detector_comparison_scope") != "world"
        or config.get("precision") != "bf16"
    ):
        raise ValueError("AdaCheck detector config does not match this experiment")

    parallelism = config.get("parallelism")
    frozen = config.get("frozen_catalog")
    if not isinstance(parallelism, Mapping) or not isinstance(frozen, Mapping):
        raise ValueError("AdaCheck detector config is incomplete")
    observed_degrees = {
        "world_size": world_size,
        "hosts": world_size // ranks_per_host,
        "ranks_per_host": ranks_per_host,
        "pipeline": 1
        + max(int(item["pipeline_parallel_rank"]) for item in rank_bindings),
        "tensor": 1
        + max(int(item["tensor_parallel_rank"]) for item in rank_bindings),
        "data": 1
        + max(int(item["data_parallel_rank"]) for item in rank_bindings),
    }
    if any(int(parallelism.get(name, -1)) != value for name, value in observed_degrees.items()):
        raise ValueError("AdaCheck detector parallel layout differs from the live layout")
    if len(rank_bindings) != world_size or sorted(
        int(item["rank"]) for item in rank_bindings
    ) != list(range(world_size)):
        raise ValueError("live rank bindings are incomplete")
    captured_workload = config.get("workload")
    captured_topology = (
        captured_workload.get("rank_topology")
        if isinstance(captured_workload, Mapping)
        else None
    )
    if not isinstance(captured_topology, Sequence) or len(captured_topology) != world_size:
        raise ValueError("AdaCheck detector config lacks rank topology evidence")
    live_by_rank = {int(item["rank"]): item for item in rank_bindings}
    for captured in captured_topology:
        if not isinstance(captured, Mapping):
            raise ValueError("AdaCheck detector rank topology is invalid")
        rank = int(captured.get("rank", -1))
        live = live_by_rank.get(rank)
        if live is None or any(
            int(captured.get(captured_name, -1)) != int(live[live_name])
            for captured_name, live_name in (
                ("data", "data_parallel_rank"),
                ("pipeline", "pipeline_parallel_rank"),
                ("tensor", "tensor_parallel_rank"),
            )
        ):
            raise ValueError("AdaCheck detector rank topology differs from the live layout")
    if int(frozen.get("detector_stability_steps", 0)) < 2:
        raise ValueError("SemanticFixed requires two stable detector observations")

    summary_path = Path(str(frozen.get("summary_path", "")))
    if not summary_path.is_absolute():
        summary_path = config_path.parent / summary_path
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("AdaCheck frozen detector summary is unreadable") from error
    if (
        _sha256_file(summary_path) != frozen.get("summary_sha256")
        or summary.get("catalog_hash") != frozen.get("catalog_hash")
        or int(summary.get("detector_stability_steps", 0)) < 2
        or int(summary.get("world_size", -1)) != world_size
        or int(summary.get("ranks_per_host", -1)) != ranks_per_host
    ):
        raise ValueError("AdaCheck frozen detector evidence is stale or inconsistent")

    normalized_bindings = [
        {
            name: int(item[name])
            for name in (
                "rank",
                "world_size",
                "data_parallel_rank",
                "pipeline_parallel_rank",
                "tensor_parallel_rank",
            )
        }
        for item in sorted(rank_bindings, key=lambda item: int(item["rank"]))
    ]
    normalized_pairs = [sorted(int(rank) for rank in pair) for pair in pairs]
    parallel_body = {
        "schema_version": 1,
        "rank_bindings": normalized_bindings,
        "ranks_per_host": ranks_per_host,
    }
    protection_body = {
        "schema_version": 1,
        "failure_model": "host-f1",
        "checkpoint_group_size": 2,
        "rank_to_host": [rank // ranks_per_host for rank in range(world_size)],
        "pairs": sorted(normalized_pairs),
    }
    binding = AdaCheckSemanticBinding(
        policy="host-f1",
        catalog_hash=str(frozen["catalog_hash"]),
        parallel_plan_hash=hashlib.sha256(
            _canonical_json(parallel_body).encode()
        ).hexdigest(),
        protection_plan_hash=hashlib.sha256(
            _canonical_json(protection_body).encode()
        ).hexdigest(),
    )
    evidence = {
        "schema_version": 1,
        "detector_config_path": str(config_path),
        "detector_config_sha256": _sha256_file(config_path),
        "detector_summary_path": str(summary_path.resolve()),
        "detector_summary_sha256": str(frozen["summary_sha256"]),
        "detector_stability_steps": int(summary["detector_stability_steps"]),
        "detector_comparison_scope": str(config["detector_comparison_scope"]),
        "catalog_hash": binding.catalog_hash,
    }
    return binding, evidence


def semantic_binding_from_state(
    state: Mapping[str, Any],
) -> AdaCheckSemanticBinding | None:
    value = state.get("adacheck_semantic")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("AdaCheck base semantic binding is invalid")
    try:
        binding = AdaCheckSemanticBinding(
            policy=str(value["policy"]),
            catalog_hash=str(value["catalog_hash"]),
            parallel_plan_hash=str(value["parallel_plan_hash"]),
            protection_plan_hash=str(value["protection_plan_hash"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("AdaCheck base semantic binding is invalid") from error
    binding.validate(value)
    return binding


def semantic_binding_from_environment() -> AdaCheckSemanticBinding | None:
    policy = os.environ.get("CKPT_ADACHECK_SEMANTIC_POLICY")
    if policy is None:
        return None
    names = (
        "CKPT_ADACHECK_CATALOG_HASH",
        "CKPT_ADACHECK_PARALLEL_PLAN_HASH",
        "CKPT_ADACHECK_PROTECTION_PLAN_HASH",
    )
    values = [os.environ.get(name) for name in names]
    if any(value is None for value in values):
        raise ValueError("AdaCheck semantic policy requires catalog, parallel, and protection hashes")
    return AdaCheckSemanticBinding(policy, *values)  # type: ignore[arg-type]


def semantic_binding_from_rank_evidence(
    ranks: Sequence[Mapping[str, Any]],
) -> dict[str, str | int] | None:
    values = [rank.get("adacheck_semantic") for rank in ranks]
    if not any(value is not None for value in values):
        return None
    if any(not isinstance(value, Mapping) for value in values):
        raise ValueError("global epoch has partial AdaCheck semantic evidence")
    first = dict(values[0])
    binding = semantic_binding_from_state({"adacheck_semantic": first})
    assert binding is not None
    if any(dict(value) != first for value in values[1:]):
        raise ValueError("global epoch has inconsistent AdaCheck semantic evidence")
    return binding.to_dict()
