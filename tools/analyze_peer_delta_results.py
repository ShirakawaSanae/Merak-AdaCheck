#!/usr/bin/env python3
"""Validate peer-delta artifacts and generate research-facing figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable



POLICIES = ("fixed", "redundancy_max")


def parse_named_path(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected NAME=PATH") from error
    if not name or not path:
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opportunity", action="append", type=parse_named_path, default=[])
    parser.add_argument("--pilot", action="append", type=parse_named_path, default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs/figures"))
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Iterable[float], percentage: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or not 0 <= percentage <= 100:
        raise ValueError("percentile requires samples and a percentage in [0, 100]")
    return ordered[max(0, math.ceil(len(ordered) * percentage / 100) - 1)]


def load_opportunity(name: str, root: Path) -> dict[str, Any]:
    summary_path = root / "summary.json"
    matrix_path = root / "pair-matrix.json"
    summary = json.loads(summary_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    if (
        summary.get("run_kind") != "peer_delta_offline_opportunity"
        or summary.get("detector_evidence") is not True
        or summary.get("detector_comparison_scope") != "world"
        or summary.get("catalog_hash") != matrix.get("catalog_hash")
    ):
        raise ValueError(f"invalid world-scope detector opportunity: {root}")
    workload = summary.get("workload", {}).get("workload", {})
    display_name = str(workload.get("display_name", name))
    ranks_per_host = int(summary["ranks_per_host"])
    heatmap = [[0.0] * ranks_per_host for _ in range(ranks_per_host)]
    for edge in matrix["edges"]:
        denominator = min(
            int(edge["first_required_bytes"]), int(edge["second_required_bytes"])
        )
        heatmap[int(edge["first_rank"])][
            int(edge["second_rank"]) - ranks_per_host
        ] = 100.0 * int(edge["overlap_bytes"]) / denominator if denominator else 0.0
    return {
        "name": name,
        "display_name": display_name,
        "root": str(root.resolve()),
        "summary_sha256": file_sha256(summary_path),
        "matrix_sha256": file_sha256(matrix_path),
        "catalog_hash": summary["catalog_hash"],
        "parallelism": summary["workload"]["parallelism"],
        "detector_comparison_scope": "world",
        "fixed_logical_bytes": int(
            summary["policies"]["fixed"]["pair_local_transfer_bytes"]
        ),
        "optimized_logical_bytes": int(
            summary["policies"]["redundancy_max"]["pair_local_transfer_bytes"]
        ),
        "host_global_logical_bytes": int(summary["host_global_control_bytes"]),
        "delta_reduction_percent": float(
            summary["profile_exact_delta_reduction_percent"]
        ),
        "overlap_percent_matrix": heatmap,
    }


def load_condition_dir(
    condition: Path, policy: str, *, warmup_epochs: int = 0
) -> dict[str, Any]:
    summary_path = condition / "summary.json"
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("run_kind") != "peer_delta_service_save"
        or summary.get("policy") != policy
        or not summary.get("last_epoch_committed")
        or int(summary.get("control_commit_acknowledged_services", 0)) != 2
    ):
        raise ValueError(f"invalid committed peer-delta condition: {condition}")
    rank_events = [
        json.loads(path.read_text())["events"]
        for path in sorted(condition.glob("rank-*.json"))
    ]
    world_size = int(summary["world_size"])
    epochs = int(summary["epochs"])
    if warmup_epochs < 0 or warmup_epochs >= epochs:
        raise ValueError("warmup epochs must leave at least one measured epoch")
    if len(rank_events) != world_size or any(len(items) != epochs for items in rank_events):
        raise ValueError(f"incomplete rank events: {condition}")
    logical_per_epoch = []
    protocol_per_epoch = []
    rank_commit_samples = []
    checkpoint_completion_samples = []
    for epoch_index in range(epochs):
        events = [items[epoch_index] for items in rank_events]
        expected_epoch = int(summary["first_epoch"]) + epoch_index
        if any(int(event["epoch"]) != expected_epoch for event in events):
            raise ValueError(f"rank event epoch changed: {condition}")
        if not all(event["committed"] for event in events):
            raise ValueError(f"uncommitted rank event: {condition}")
        logical_per_epoch.append(
            sum(int(event["remote_delta"]["logical_bytes"]) for event in events)
        )
        protocol_per_epoch.append(
            sum(
                int(event["remote_delta"]["protocol_wire_bytes"])
                + int(event["remote_control_protocol_request_bytes"])
                + int(event["remote_control_protocol_response_bytes"])
                for event in events
            )
        )
        if epoch_index >= warmup_epochs:
            epoch_commits = [float(event["commit_ms"]) for event in events]
            rank_commit_samples.extend(epoch_commits)
            checkpoint_completion_samples.append(max(epoch_commits))
    planned = int(summary["planned_pair_local_transfer_bytes"])
    if any(value != planned for value in logical_per_epoch):
        raise ValueError(f"runtime delta does not match plan: {condition}")
    recorded_protocol = int(summary["remote_cross_host_protocol"]["wire_bytes"])
    if sum(protocol_per_epoch) != recorded_protocol:
        raise ValueError(f"protocol byte accounting changed: {condition}")
    expected_records = world_size + 1
    for stats in summary["service_stats_after"]:
        current = [
            item
            for item in stats["records"]
            if int(item["epoch"]) == int(summary["last_epoch"])
            and item["state"] == "committed"
        ]
        if len(current) != expected_records:
            raise ValueError(f"service record count changed: {condition}")
    accounting = summary.get("service_protocol_accounting")
    if (
        not isinstance(accounting, list)
        or len(accounting) != 2
        or not all(item.get("validated") for item in accounting)
    ):
        raise ValueError(f"service protocol accounting failed: {condition}")
    nic_records = [item for item in summary["nic_records"] if item.get("status") == "available"]
    nic_contaminated = any(item.get("contaminated") is True for item in nic_records)
    nic_xmit = None
    if len(nic_records) == 2 and not nic_contaminated:
        nic_xmit = sum(
            int(item["workload_delta"]["total_xmit_bytes"]) for item in nic_records
        ) / epochs
    return {
        "policy": policy,
        "summary": str(summary_path.resolve()),
        "summary_sha256": file_sha256(summary_path),
        "catalog_hash": summary["catalog_hash"],
        "plan_hash": summary["plan_hash"],
        "epoch_count": epochs,
        "measured_epoch_count": epochs - warmup_epochs,
        "rank_commit_sample_count": len(rank_commit_samples),
        "checkpoint_completion_sample_count": len(checkpoint_completion_samples),
        "logical_delta_bytes_per_epoch": planned,
        "protocol_wire_bytes_per_epoch": sum(
            protocol_per_epoch[warmup_epochs:]
        )
        / (epochs - warmup_epochs),
        "nic_xmit_bytes_per_epoch": nic_xmit,
        "nic_contaminated": nic_contaminated,
        "rank_commit_p50_ms": percentile(rank_commit_samples, 50),
        "rank_commit_p95_ms": percentile(rank_commit_samples, 95),
        "rank_commit_p99_ms": percentile(rank_commit_samples, 99),
        "checkpoint_completion_p50_ms": percentile(
            checkpoint_completion_samples, 50
        ),
        "checkpoint_completion_p95_ms": percentile(
            checkpoint_completion_samples, 95
        ),
        "checkpoint_completion_p99_ms": percentile(
            checkpoint_completion_samples, 99
        ),
    }


def load_condition(root: Path, policy: str) -> dict[str, Any]:
    return load_condition_dir(root / policy, policy)


def load_pilot(name: str, root: Path) -> dict[str, Any]:
    if not (root / "COMPLETE").exists():
        raise ValueError(f"pilot has no COMPLETE marker: {root}")
    config_path = root / "experiment-config.json"
    config = json.loads(config_path.read_text())
    if config.get("run_kind") != "peer_delta_two_node_pilot":
        raise ValueError(f"invalid peer-delta experiment config: {root}")
    conditions = [load_condition(root, policy) for policy in POLICIES]
    catalog_hashes = {item["catalog_hash"] for item in conditions}
    expected_catalog = config["source_capture"]["catalog_hash"]
    if catalog_hashes != {expected_catalog}:
        raise ValueError(f"pilot catalog provenance changed: {root}")
    fault_summaries = []
    for path in sorted(root.glob("service-failure-*/restore-summary.json")):
        value = json.loads(path.read_text())
        if (
            not value.get("catalog_restore_smoke")
            or not value.get("all_rank_templates_reconstructed")
            or len(value.get("service_failure_sets_validated", [])) != 1
            or value.get("runtime_failure_sets_validated") != []
        ):
            raise ValueError(f"invalid service-failure restore: {path}")
        fault_summaries.append(
            {
                "path": str(path.resolve()),
                "summary_sha256": file_sha256(path),
                "restore_ms": float(value["restore_ms"]),
                "restore_protocol_wire_bytes": int(
                    value["restore_protocol_wire_bytes"]
                ),
                "service_failure_sets_validated": value[
                    "service_failure_sets_validated"
                ],
            }
        )
    if len(fault_summaries) != 1:
        raise ValueError(f"pilot must contain one survivor service restore: {root}")
    return {
        "name": name,
        "display_name": config["workload"]["workload"]["display_name"],
        "root": str(root.resolve()),
        "config": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "catalog_hash": expected_catalog,
        "conditions": conditions,
        "service_failure_restores": fault_summaries,
        "research_claim_scope": config["research_claim_scope"],
    }














def main() -> None:
    args = parse_args()
    if not args.opportunity or not args.pilot:
        raise ValueError("at least one opportunity and one pilot are required")
    opportunities = [load_opportunity(name, path) for name, path in args.opportunity]
    pilots = [load_pilot(name, path) for name, path in args.pilot]
    opportunity_by_name = {item["name"]: item for item in opportunities}
    for pilot in pilots:
        opportunity = opportunity_by_name.get(pilot["name"])
        if opportunity is None or opportunity["catalog_hash"] != pilot["catalog_hash"]:
            raise ValueError(f"offline/runtime catalog mismatch for {pilot['name']}")
    result = {
        "schema_version": 1,
        "run_kind": "peer_delta_research_analysis",
        "opportunities": opportunities,
        "pilots": pilots,
        "limitations": [
            "pilot latency samples are descriptive and are not independent trials",
            "service process failure is not a physical host failure",
            "catalog reconstruction is not continued-training recovery",
            "two-service control visibility is sequential, not distributed atomic",
            "protocol bytes and PMA NIC counters are distinct metrics",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"analysis": str(args.output_json), "pilots": len(pilots)}))


if __name__ == "__main__":
    main()
