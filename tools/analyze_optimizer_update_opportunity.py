#!/usr/bin/env python3
"""Evaluate a previous-step grouping on the next optimizer-update catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from Merak.core.checkpoint.peer_delta import (
    build_peer_delta_matrix_from_catalog,
    evaluate_peer_grouping,
    exact_peer_grouping,
    fixed_peer_grouping,
)


def _hash_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_epoch_catalogs(root: Path) -> tuple[dict[str, Any], ...]:
    records = tuple(
        json.loads(path.read_text()) for path in sorted(root.glob("rank_*.json"))
    )
    if not records:
        raise ValueError(f"optimizer-update catalog is empty: {root}")
    world_size = int(records[0]["world_size"])
    ranks = tuple(int(record["rank"]) for record in records)
    if ranks != tuple(range(world_size)):
        raise ValueError("optimizer-update catalogs must contain every rank in order")
    common = (records[0]["run_id"], int(records[0]["epoch"]), world_size)
    for record in records:
        if record.get("run_kind") != "merak_pre_optimizer_update_catalog":
            raise ValueError("unsupported optimizer-update catalog producer")
        if record.get("schema_version") != 1 or record.get("snapshot_point") not in {
            "post_unscale_pre_optimizer_step",
            "raw_bf16_pre_unscale",
        }:
            raise ValueError("unsupported optimizer-update catalog schema or boundary")
        if (record["run_id"], int(record["epoch"]), int(record["world_size"])) != common:
            raise ValueError("optimizer-update catalogs mix runs, epochs, or worlds")
        expected_hash = str(record["catalog_hash"])
        unhashed = dict(record)
        del unhashed["catalog_hash"]
        if _hash_json(unhashed) != expected_hash:
            raise ValueError(f"optimizer-update catalog hash mismatch for rank {record['rank']}")
        if record.get("adacheck_payload_faithful") is not False:
            raise ValueError("dense-gradient catalog must not claim AdaCheck payload fidelity")
    return records


def _value_catalog(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict], str]:
    buckets: dict[str, dict[str, Any]] = {}
    rank_hashes = []
    for record in records:
        rank = int(record["rank"])
        gradients = [
            item for item in record["objects"] if item["role"] == "gradients"
        ]
        if len(gradients) != int(record["gradient_tensor_count"]):
            raise ValueError(f"gradient tensor count mismatch for rank {rank}")
        if sum(int(item["logical_bytes"]) for item in gradients) != int(
            record["gradient_logical_bytes"]
        ):
            raise ValueError(f"gradient byte count mismatch for rank {rank}")
        rank_hashes.append(str(record["catalog_hash"]))
        seen_paths = set()
        for item in gradients:
            path = str(item["path"])
            if path in seen_paths:
                raise ValueError(f"duplicate gradient path for rank {rank}: {path}")
            seen_paths.add(path)
            identity = {
                "dtype": str(item["dtype"]),
                "shape": [int(value) for value in item["shape"]],
                "logical_bytes": int(item["logical_bytes"]),
                "checksum": str(item["checksum"]),
            }
            value_id = _hash_json(identity)
            bucket = buckets.setdefault(
                value_id,
                {
                    "value_id": value_id,
                    "role": "gradients",
                    **identity,
                    "source_ranks": set(),
                    "requirements": [],
                },
            )
            bucket["source_ranks"].add(rank)
            bucket["requirements"].append({"rank": rank, "path": path})
    if not buckets:
        raise ValueError("optimizer-update catalog contains no gradient tensors")
    objects = []
    for value_id in sorted(buckets):
        bucket = buckets[value_id]
        bucket["source_ranks"] = sorted(bucket["source_ranks"])
        bucket["requirements"] = sorted(
            bucket["requirements"], key=lambda item: (item["rank"], item["path"])
        )
        objects.append(bucket)
    return objects, _hash_json({"rank_catalog_hashes": rank_hashes})


def analyze_transition(
    plan_records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
    *,
    ranks_per_host: int,
) -> dict[str, Any]:
    plan_by_rank = {int(record["rank"]): record for record in plan_records}
    evaluation_by_rank = {
        int(record["rank"]): record for record in evaluation_records
    }
    if plan_by_rank.keys() != evaluation_by_rank.keys():
        raise ValueError("plan and evaluation catalogs have different ranks")
    plan_epoch = int(plan_records[0]["epoch"])
    evaluation_epoch = int(evaluation_records[0]["epoch"])
    if evaluation_epoch != plan_epoch + 1:
        raise ValueError("evaluation catalog must immediately follow the plan catalog")
    if plan_records[0]["run_id"] != evaluation_records[0]["run_id"]:
        raise ValueError("plan and evaluation catalogs must come from one training run")
    for rank in plan_by_rank:
        before, after = plan_by_rank[rank], evaluation_by_rank[rank]
        for field in ("binding", "topology_hash", "parameter_ids", "base_id", "base_epoch"):
            if before[field] != after[field]:
                raise ValueError(f"optimizer-update topology changed for rank {rank}: {field}")

    plan_objects, plan_hash = _value_catalog(plan_records)
    evaluation_objects, evaluation_hash = _value_catalog(evaluation_records)
    world_size = len(plan_records)
    plan_matrix = build_peer_delta_matrix_from_catalog(
        plan_objects,
        world_size=world_size,
        ranks_per_host=ranks_per_host,
        catalog_hash=plan_hash,
    )
    evaluation_matrix = build_peer_delta_matrix_from_catalog(
        evaluation_objects,
        world_size=world_size,
        ranks_per_host=ranks_per_host,
        catalog_hash=evaluation_hash,
    )
    fixed = evaluate_peer_grouping(
        evaluation_matrix, fixed_peer_grouping(evaluation_matrix)
    )
    frozen_grouping = exact_peer_grouping(plan_matrix)
    frozen = evaluate_peer_grouping(evaluation_matrix, frozen_grouping)
    hindsight = evaluate_peer_grouping(
        evaluation_matrix, exact_peer_grouping(evaluation_matrix)
    )

    def peer_delta_compatible(groups: Sequence[Sequence[int]]) -> bool:
        for first, second in groups:
            left = evaluation_by_rank[int(first)]["binding"]
            right = evaluation_by_rank[int(second)]["binding"]
            if (
                left["pipeline_parallel_rank"] != right["pipeline_parallel_rank"]
                or left["tensor_parallel_rank"] != right["tensor_parallel_rank"]
                or left["data_parallel_rank"] == right["data_parallel_rank"]
            ):
                return False
        return True

    def reduction(candidate: int) -> float:
        baseline = fixed.pair_local_transfer_bytes
        return 0.0 if baseline == 0 else 100.0 * (baseline - candidate) / baseline

    return {
        "schema_version": 1,
        "run_kind": "optimizer_update_offline_previous_epoch_backtest",
        "run_id": str(plan_records[0]["run_id"]),
        "cost_input": "dense_post_unscale_pre_optimizer_gradients",
        "plan_epoch": plan_epoch,
        "evaluation_epoch": evaluation_epoch,
        "previous_epoch_data_split": True,
        "runtime_causal_plan_executed": False,
        "grouping_constraint": "cross_host_only_unrestricted_parallel_role",
        "plan_catalog_hash": plan_hash,
        "evaluation_catalog_hash": evaluation_hash,
        "world_size": world_size,
        "ranks_per_host": ranks_per_host,
        "policies": {
            "fixed": fixed.to_dict(),
            "frozen_previous_epoch_exact": frozen.to_dict(),
            "current_epoch_hindsight_exact": hindsight.to_dict(),
        },
        "frozen_plan_delta_reduction_percent": reduction(
            frozen.pair_local_transfer_bytes
        ),
        "hindsight_delta_reduction_percent": reduction(
            hindsight.pair_local_transfer_bytes
        ),
        "frozen_plan_current_peer_delta_compatible": peer_delta_compatible(
            frozen_grouping.groups
        ),
        "adacheck_detector_filtered": False,
        "adacheck_payload_faithful": False,
        "eligible_for_runtime_gate": False,
        "runtime_recovery_validated": False,
        "limitations": [
            "dense gradients are not filtered by AdaCheck detector evidence",
            "direct-saved state, metadata, serialization, and NIC framing are absent",
            "group migration and complete recovery-object coverage are not modeled",
            "synchronous catalog hashing perturbs timing",
            "the unrestricted content-oracle grouping may violate current peer-delta topology",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-catalog-dir", type=Path, required=True)
    parser.add_argument("--evaluation-catalog-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ranks-per-host", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_records = load_epoch_catalogs(args.plan_catalog_dir)
    evaluation_records = load_epoch_catalogs(args.evaluation_catalog_dir)
    result = analyze_transition(
        plan_records, evaluation_records, ranks_per_host=args.ranks_per_host
    )
    plan = {
        "schema_version": 1,
        "policy": "offline_previous_epoch_unrestricted_content_oracle",
        "input_catalog_hash": result["plan_catalog_hash"],
        "input_epoch": result["plan_epoch"],
        "effective_epoch": result["evaluation_epoch"],
        "grouping": result["policies"]["frozen_previous_epoch_exact"]["grouping"],
        "runtime_recovery_validated": False,
    }
    _atomic_json(args.output_dir / "offline-previous-epoch-plan.json", plan)
    _atomic_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
