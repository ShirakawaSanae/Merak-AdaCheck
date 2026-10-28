#!/usr/bin/env python3
"""Analyze matched grouping trials and generate paper-ready archival figures."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable



POLICY_ORDER = ("fixed", "random", "exact")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--capture-summary", action="append", type=Path, default=[])
    return parser.parse_args()


def percentile(values: Iterable[float], percentage: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of no values")
    return ordered[max(0, math.ceil(len(ordered) * percentage / 100) - 1)]


def mean_ci95(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    # Formal matrices use n=3; keep a conservative finite-sample t interval.
    t_critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(
        len(values), 1.96
    )
    return mean, t_critical * statistics.stdev(values) / math.sqrt(len(values))


def load_trial(condition_path: Path, warmup_epochs: int) -> dict[str, object]:
    root = condition_path.parent
    condition = json.loads(condition_path.read_text())
    summary = json.loads((root / "summary.json").read_text())
    if summary.get("run_kind") != "matched_detector_grouping_save":
        raise ValueError(f"not a matched grouping summary: {root}")
    if summary.get("fault") != "none" or not summary.get("last_epoch_committed"):
        raise ValueError(f"formal performance trial is not a clean commit: {root}")
    rank_events = []
    for path in sorted(root.glob("rank-*.json")):
        rank_events.append(json.loads(path.read_text())["events"])
    if len(rank_events) != int(summary["world_size"]):
        raise ValueError(f"rank event count disagrees with world size: {root}")
    epoch_count = int(summary["epochs"])
    if warmup_epochs >= epoch_count:
        raise ValueError("warmup epochs must leave measured checkpoint epochs")

    commit_samples = []
    exchange_samples = []
    epoch_wire = []
    epoch_aggregate_pair_peak = []
    epoch_max_service_wire = []
    groups = [tuple(int(rank) for rank in pair) for pair in summary["grouping"]["groups"]]
    for epoch_index in range(warmup_epochs, epoch_count):
        events = [items[epoch_index] for items in rank_events]
        if not all(item["committed"] for item in events):
            raise ValueError(f"uncommitted measured epoch in {root}")
        commit_samples.extend(float(item["commit_ms"]) for item in events)
        exchange_samples.extend(float(item["exchange_ms"]) for item in events)
        epoch_wire.append(
            sum(
                int(item["exchange_wire_bytes"]) + int(item["service_wire_bytes"])
                for item in events
            )
            + int(events[0]["control_wire_bytes"])
        )
        epoch_aggregate_pair_peak.append(
            sum(
                max(
                    int(events[first]["exchange_payload_bytes"]),
                    int(events[second]["exchange_payload_bytes"]),
                )
                for first, second in groups
            )
        )
        epoch_max_service_wire.append(
            max(int(item["service_wire_bytes"]) for item in events)
        )
    nic_records = [
        item for item in summary.get("nic_records", []) if item.get("status") == "available"
    ]
    nic_xmit_per_epoch = None
    nic_rcv_per_epoch = None
    idle_xmit_bytes = None
    idle_rcv_bytes = None
    if nic_records:
        nic_xmit_per_epoch = sum(
            int(item["workload_delta"]["total_xmit_bytes"]) for item in nic_records
        ) / epoch_count
        nic_rcv_per_epoch = sum(
            int(item["workload_delta"]["total_rcv_bytes"]) for item in nic_records
        ) / epoch_count
        idle_xmit_bytes = sum(
            int(item["idle_delta"]["total_xmit_bytes"]) for item in nic_records
        )
        idle_rcv_bytes = sum(
            int(item["idle_delta"]["total_rcv_bytes"]) for item in nic_records
        )
    return {
        **condition,
        "root": str(root),
        "catalog_hash": summary["catalog_hash"],
        "plan_hash": summary["plan_hash"],
        "cross_host_logical_bytes": int(summary["cross_host_logical_bytes"]),
        "commit_p99_ms": percentile(commit_samples, 99),
        "commit_mean_ms": statistics.fmean(commit_samples),
        "exchange_p99_ms": percentile(exchange_samples, 99),
        "application_wire_bytes_per_epoch": statistics.fmean(epoch_wire),
        "aggregate_pair_peak_bytes_per_epoch": statistics.fmean(
            epoch_aggregate_pair_peak
        ),
        "max_service_wire_bytes_per_epoch": statistics.fmean(
            epoch_max_service_wire
        ),
        "nic_xmit_bytes_per_epoch": nic_xmit_per_epoch,
        "nic_rcv_bytes_per_epoch": nic_rcv_per_epoch,
        "idle_xmit_bytes": idle_xmit_bytes,
        "idle_rcv_bytes": idle_rcv_bytes,
        "measured_epoch_count": epoch_count - warmup_epochs,
        "rank_commit_sample_count": len(commit_samples),
    }


def aggregate(trials: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for trial in trials:
        grouped[(str(trial["model"]), str(trial["policy"]))].append(trial)
    results = []
    for (model, policy), items in sorted(grouped.items()):
        result: dict[str, object] = {
            "model": model,
            "policy": policy,
            "trial_count": len(items),
        }
        for metric in (
            "commit_p99_ms",
            "commit_mean_ms",
            "exchange_p99_ms",
            "application_wire_bytes_per_epoch",
            "aggregate_pair_peak_bytes_per_epoch",
            "max_service_wire_bytes_per_epoch",
        ):
            mean, ci = mean_ci95([float(item[metric]) for item in items])
            result[f"{metric}_mean"] = mean
            result[f"{metric}_ci95"] = ci
        if all(item["nic_xmit_bytes_per_epoch"] is not None for item in items):
            for metric in ("nic_xmit_bytes_per_epoch", "nic_rcv_bytes_per_epoch"):
                mean, ci = mean_ci95([float(item[metric]) for item in items])
                result[f"{metric}_mean"] = mean
                result[f"{metric}_ci95"] = ci
        results.append(result)
    return results


def load_capture_summary(path: Path) -> dict[str, object]:
    summary = json.loads(path.read_text())
    workload_path = path.parent.parent / "workload.json"
    workload = json.loads(workload_path.read_text())["workload"]
    if summary.get("run_kind") != "matched_detector_capture_build":
        raise ValueError(f"not a matched detector capture summary: {path}")
    if not summary.get("all_rank_catalogs_reconstructed"):
        raise ValueError(f"capture catalogs do not reconstruct: {path}")
    return {
        "model": workload["name"],
        "display_name": workload["display_name"],
        "catalog_hash": summary["catalog_hash"],
        "full_logical_bytes": int(summary["full_logical_bytes"]),
        "reduced_logical_bytes": int(summary["reduced_logical_bytes"]),
        "object_count": int(summary["object_count"]),
        "confirmed_pairs": int(summary["detector_confirmation"]["confirmed_pairs"]),
        "rejected_pairs": int(summary["detector_confirmation"]["rejected_pairs"]),
        "summary": str(path),
    }






def main() -> None:
    args = parse_args()
    condition_paths = sorted(
        {path for root in args.roots for path in root.rglob("condition.json")}
    )
    if not condition_paths:
        raise ValueError("no matched trial condition.json files found")
    trials = [load_trial(path, args.warmup_epochs) for path in condition_paths]
    aggregated = aggregate(trials)
    catalogs = [load_capture_summary(path) for path in args.capture_summary]
    result = {
        "schema_version": 1,
        "warmup_epochs": args.warmup_epochs,
        "trial_count": len(trials),
        "trials": trials,
        "aggregated": aggregated,
        "catalogs": catalogs,
    }
    output_json = args.output_json or args.roots[0] / "analysis.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"analysis": str(output_json), "trials": len(trials)}, sort_keys=True))


if __name__ == "__main__":
    main()
