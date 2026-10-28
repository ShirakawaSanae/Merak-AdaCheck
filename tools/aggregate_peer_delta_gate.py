#!/usr/bin/env python3
"""Aggregate independent peer-delta opportunity results into one runtime gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-passing-workloads", type=int, default=2)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def workload_key(summary: dict) -> str:
    workload = summary.get("workload")
    if not isinstance(workload, dict):
        return f"unidentified:{summary['catalog_hash']}"
    model = workload.get("workload", {}).get("name", "unknown")
    parallelism = workload.get("parallelism", {})
    return json.dumps(
        {"model": model, "parallelism": parallelism},
        sort_keys=True,
        separators=(",", ":"),
    )


def main() -> None:
    args = parse_args()
    if args.minimum_passing_workloads < 1:
        raise ValueError("minimum passing workloads must be positive")
    summaries = [json.loads(path.read_text()) for path in args.summaries]
    for summary in summaries:
        if summary.get("run_kind") != "peer_delta_offline_opportunity":
            raise ValueError("all inputs must be peer-delta opportunity summaries")
    detector_results = [
        summary
        for summary in summaries
        if summary.get("detector_evidence") is True
        and summary.get("eligible_for_runtime_gate") is True
    ]
    passing = {
        workload_key(summary): summary
        for summary in detector_results
        if summary.get("passes_single_catalog_gate") is True
    }
    result = {
        "schema_version": 1,
        "run_kind": "peer_delta_multi_catalog_gate",
        "minimum_passing_workloads": args.minimum_passing_workloads,
        "input_count": len(summaries),
        "eligible_detector_result_count": len(detector_results),
        "passing_workload_count": len(passing),
        "passing_workloads": [
            {
                "workload_key": key,
                "catalog_hash": value["catalog_hash"],
                "detector_comparison_scope": value.get(
                    "detector_comparison_scope"
                ),
                "delta_reduction_percent": value[
                    "profile_exact_delta_reduction_percent"
                ],
                "makespan_reduction_percent": value[
                    "profile_exact_makespan_reduction_percent"
                ],
            }
            for key, value in sorted(passing.items())
        ],
        "passes_multi_catalog_gate": len(passing)
        >= args.minimum_passing_workloads,
        "excluded_upper_bound_count": sum(
            summary.get("detector_evidence") is False for summary in summaries
        ),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
