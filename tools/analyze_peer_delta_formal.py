#!/usr/bin/env python3
"""Aggregate independent peer-delta trials with finite-sample intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


from tools.analyze_peer_delta_results import load_condition_dir


POLICIES = ("fixed", "random", "redundancy_max", "profile_exact")
FORMAL_REPETITIONS = 5
FORMAL_EPOCHS_PER_CONDITION = 8


def parse_named_path(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected NAME=PATH") from error
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", action="append", type=parse_named_path, required=True)
    parser.add_argument("--opportunity", action="append", type=parse_named_path)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs/figures"))
    return parser.parse_args()


def mean_ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("cannot aggregate an empty sample")
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, 0.0
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(
        len(values), 1.96
    )
    return mean, critical * statistics.stdev(values) / math.sqrt(len(values))


def load_formal(name: str, root: Path, warmup_epochs: int) -> dict[str, Any]:
    if not (root / "COMPLETE").exists():
        raise ValueError(f"formal root is incomplete: {root}")
    config = json.loads((root / "experiment-config.json").read_text())
    expected_policies = tuple(config["policies"])
    if set(expected_policies) != set(POLICIES):
        raise ValueError(f"unexpected formal policies: {expected_policies}")
    trials = []
    for condition in sorted(root.glob("trial-[0-9][0-9]-*")):
        prefix, trial_value, policy = condition.name.split("-", 2)
        if prefix != "trial" or policy not in expected_policies:
            continue
        value = load_condition_dir(
            condition, policy, warmup_epochs=warmup_epochs
        )
        value["trial"] = int(trial_value)
        trials.append(value)
    repetitions = int(config["repetitions"])
    epochs_per_condition = int(config["epochs_per_policy"])
    if (
        repetitions != FORMAL_REPETITIONS
        or epochs_per_condition != FORMAL_EPOCHS_PER_CONDITION
    ):
        raise ValueError(
            "formal configuration must use exactly "
            f"{FORMAL_REPETITIONS} repetitions x "
            f"{FORMAL_EPOCHS_PER_CONDITION} epochs"
        )
    if len(trials) != repetitions * len(expected_policies):
        raise ValueError(f"formal condition count is incomplete: {root}")
    if {item["catalog_hash"] for item in trials} != {
        config["source_capture"]["catalog_hash"]
    }:
        raise ValueError(f"formal catalog changed: {root}")
    if {item["epoch_count"] for item in trials} != {epochs_per_condition}:
        raise ValueError(f"formal epoch count changed across conditions: {root}")
    counts = defaultdict(int)
    for trial in trials:
        counts[(trial["trial"], trial["policy"])] += 1
    if any(
        counts[(trial, policy)] != 1
        for trial in range(1, repetitions + 1)
        for policy in expected_policies
    ):
        raise ValueError(f"formal trial/policy cells are not unique: {root}")
    return {
        "name": name,
        "display_name": config["workload"]["workload"]["display_name"],
        "root": str(root.resolve()),
        "catalog_hash": config["source_capture"]["catalog_hash"],
        "repetitions": repetitions,
        "epochs_per_condition": epochs_per_condition,
        "warmup_epochs": warmup_epochs,
        "policies": list(expected_policies),
        "trials": trials,
    }


def aggregate(formal: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for policy in formal["policies"]:
        trials = [item for item in formal["trials"] if item["policy"] == policy]
        row: dict[str, Any] = {
            "policy": policy,
            "trial_count": len(trials),
            "rank_commit_samples_per_trial": trials[0]["rank_commit_sample_count"],
            "checkpoint_completion_samples_per_trial": trials[0][
                "checkpoint_completion_sample_count"
            ],
        }
        for metric in (
            "logical_delta_bytes_per_epoch",
            "protocol_wire_bytes_per_epoch",
            "nic_xmit_bytes_per_epoch",
            "rank_commit_p50_ms",
            "rank_commit_p95_ms",
            "rank_commit_p99_ms",
            "checkpoint_completion_p50_ms",
            "checkpoint_completion_p95_ms",
            "checkpoint_completion_p99_ms",
        ):
            values = [
                float(item[metric])
                for item in trials
                if item[metric] is not None
            ]
            row[f"{metric}_valid_trial_count"] = len(values)
            if values:
                mean, ci = mean_ci95(values)
                row[f"{metric}_mean"] = mean
                row[f"{metric}_ci95"] = ci
            else:
                row[f"{metric}_mean"] = None
                row[f"{metric}_ci95"] = None
        result.append(row)
    fixed = {item["trial"]: item for item in formal["trials"] if item["policy"] == "fixed"}
    for row in result:
        if row["policy"] == "fixed":
            row["paired_reductions_vs_fixed"] = {}
            continue
        candidate = {
            item["trial"]: item
            for item in formal["trials"]
            if item["policy"] == row["policy"]
        }
        paired = {}
        for metric in (
            "logical_delta_bytes_per_epoch",
            "protocol_wire_bytes_per_epoch",
            "nic_xmit_bytes_per_epoch",
            "checkpoint_completion_p99_ms",
        ):
            reductions = [
                100.0 * (fixed[index][metric] - candidate[index][metric])
                / fixed[index][metric]
                for index in sorted(fixed)
                if fixed[index][metric] is not None
                and candidate[index][metric] is not None
            ]
            if reductions:
                mean, ci = mean_ci95(reductions)
                paired[metric] = {
                    "mean_percent": mean,
                    "ci95_percent": ci,
                    "valid_pair_count": len(reductions),
                }
            else:
                paired[metric] = {
                    "mean_percent": None,
                    "ci95_percent": None,
                    "valid_pair_count": 0,
                }
        row["paired_reductions_vs_fixed"] = paired
    return result






def load_opportunity(name: str, path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if (
        value.get("run_kind") != "peer_delta_offline_opportunity"
        or value.get("detector_evidence") is not True
        or value.get("detector_comparison_scope") != "world"
    ):
        raise ValueError(f"invalid world-scope detector opportunity: {path}")
    workload = value.get("workload", {}).get("workload", {})
    return {
        "name": name,
        "display_name": workload["display_name"],
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "catalog_hash": value["catalog_hash"],
        "fixed_bytes": value["policies"]["fixed"]["pair_local_transfer_bytes"],
        "redundancy_max_bytes": value["policies"]["redundancy_max"][
            "pair_local_transfer_bytes"
        ],
        "host_global_bytes": value["host_global_control_bytes"],
    }






def main() -> None:
    args = parse_args()
    formals = [
        load_formal(name, path, args.warmup_epochs)
        for name, path in args.formal
    ]
    for formal in formals:
        formal["aggregated"] = aggregate(formal)
    opportunities = [
        load_opportunity(name, path) for name, path in (args.opportunity or [])
    ]
    result = {
        "schema_version": 1,
        "run_kind": "peer_delta_formal_analysis",
        "formals": formals,
        "opportunities": opportunities,
        "limitations": [
            "PMA counters cover every epoch because counters are sampled per condition",
            "PMA trials exceeding the pre-registered idle threshold are excluded",
            "rank commit samples within a trial are not treated as independent trials",
            "catalog restore and service failure do not establish continued-training recovery",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"analysis": str(args.output_json), "models": len(formals)}))


if __name__ == "__main__":
    main()
