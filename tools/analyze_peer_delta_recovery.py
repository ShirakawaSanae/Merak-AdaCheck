#!/usr/bin/env python3
"""Validate paired recovery trials and test a pre-registered slowdown margin."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


from tools.analyze_peer_delta_formal import mean_ci95
from tools.analyze_peer_delta_results import load_condition_dir


POLICIES = ("fixed", "redundancy_max")
REPETITIONS = 5
EPOCHS_PER_CONDITION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs/figures"))
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_restore(path: Path, *, catalog_hash: str, epoch: int, host: int) -> dict[str, Any]:
    summary_path = path / "restore-summary.json"
    value = json.loads(summary_path.read_text())
    if (
        value.get("run_kind") != "peer_delta_service_restore"
        or value.get("catalog_hash") != catalog_hash
        or int(value.get("restore_epoch", -1)) != epoch
        or int(value.get("restore_host_index", -1)) != host
        or value.get("catalog_valid") is not True
        or value.get("all_rank_templates_reconstructed") is not True
        or value.get("service_failure_sets_validated") != []
        or value.get("runtime_failure_sets_validated") != []
    ):
        raise ValueError(f"invalid ordinary service restore: {path}")
    restore_ms = float(value["restore_ms"])
    wire_bytes = int(value["restore_protocol_wire_bytes"])
    if restore_ms <= 0 or wire_bytes <= 0:
        raise ValueError(f"invalid recovery measurements: {path}")
    return {
        "host_index": host,
        "restore_ms": restore_ms,
        "protocol_wire_bytes": wire_bytes,
        "summary": str(summary_path.resolve()),
        "summary_sha256": file_sha256(summary_path),
    }


def classify_noninferiority(mean: float, ci95: float, margin: float) -> str:
    if mean + ci95 <= margin:
        return "pass"
    if mean - ci95 > margin:
        return "fail"
    return "inconclusive"


def load_recovery(root: Path) -> dict[str, Any]:
    if not (root / "COMPLETE").exists():
        raise ValueError(f"recovery root is incomplete: {root}")
    config_path = root / "experiment-config.json"
    config = json.loads(config_path.read_text())
    if (
        config.get("run_kind") != "peer_delta_two_node_recovery_matrix"
        or config.get("restore_each_condition") is not True
        or set(config.get("policies", [])) != set(POLICIES)
        or int(config.get("repetitions", -1)) != REPETITIONS
        or int(config.get("epochs_per_policy", -1)) != EPOCHS_PER_CONDITION
    ):
        raise ValueError(
            "recovery matrix must be exactly 5 paired trials x 1 epoch for "
            "fixed and redundancy_max"
        )

    catalog_hash = str(config["source_capture"]["catalog_hash"])
    trials = []
    for trial in range(1, REPETITIONS + 1):
        for policy in POLICIES:
            condition = root / f"trial-{trial:02d}-{policy}"
            save = load_condition_dir(condition, policy, warmup_epochs=0)
            save_summary = json.loads((condition / "summary.json").read_text())
            if save["catalog_hash"] != catalog_hash:
                raise ValueError(f"recovery catalog changed: {condition}")
            restores = [
                load_restore(
                    condition / f"restore-host-{host}",
                    catalog_hash=catalog_hash,
                    epoch=int(save_summary["last_epoch"]),
                    host=host,
                )
                for host in (0, 1)
            ]
            trials.append(
                {
                    "trial": trial,
                    "policy": policy,
                    "condition": str(condition.resolve()),
                    "condition_summary_sha256": file_sha256(condition / "summary.json"),
                    "restores": restores,
                    "worst_host_restore_ms": max(
                        item["restore_ms"] for item in restores
                    ),
                    "worst_host_protocol_wire_bytes": max(
                        item["protocol_wire_bytes"] for item in restores
                    ),
                }
            )

    aggregated = []
    for policy in POLICIES:
        rows = [item for item in trials if item["policy"] == policy]
        row: dict[str, Any] = {"policy": policy, "trial_count": len(rows)}
        for metric in ("worst_host_restore_ms", "worst_host_protocol_wire_bytes"):
            mean, ci95 = mean_ci95([float(item[metric]) for item in rows])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95"] = ci95
        aggregated.append(row)

    by_trial = {
        (item["trial"], item["policy"]): item for item in trials
    }
    slowdowns = [
        100.0
        * (
            by_trial[(trial, "redundancy_max")]["worst_host_restore_ms"]
            / by_trial[(trial, "fixed")]["worst_host_restore_ms"]
            - 1.0
        )
        for trial in range(1, REPETITIONS + 1)
    ]
    mean_slowdown, ci95_slowdown = mean_ci95(slowdowns)
    margin = float(config["recovery_noninferiority_margin_percent"])
    paired = {
        "metric": "worst_host_restore_ms",
        "candidate": "redundancy_max",
        "baseline": "fixed",
        "trial_count": len(slowdowns),
        "slowdown_percent_by_trial": slowdowns,
        "mean_slowdown_percent": mean_slowdown,
        "ci95_percent": ci95_slowdown,
        "ci95_lower_percent": mean_slowdown - ci95_slowdown,
        "ci95_upper_percent": mean_slowdown + ci95_slowdown,
        "noninferiority_margin_percent": margin,
        "noninferiority": classify_noninferiority(
            mean_slowdown, ci95_slowdown, margin
        ),
    }
    return {
        "schema_version": 1,
        "run_kind": "peer_delta_recovery_analysis",
        "root": str(root.resolve()),
        "experiment_config": str(config_path.resolve()),
        "experiment_config_sha256": file_sha256(config_path),
        "display_name": config["workload"]["workload"]["display_name"],
        "catalog_hash": catalog_hash,
        "repetitions": REPETITIONS,
        "epochs_per_condition": EPOCHS_PER_CONDITION,
        "policies": list(POLICIES),
        "trial_metric": "maximum restore_ms across the two independent services",
        "trials": trials,
        "aggregated": aggregated,
        "paired_noninferiority": paired,
        "limitations": [
            "restore is a single-coordinator catalog fetch, validation, and 16-template reconstruction",
            "the experiment does not resume training or measure full recovery time objective",
            "service-process failure smoke does not establish physical-host failure",
        ],
    }




def main() -> None:
    args = parse_args()
    result = load_recovery(args.recovery_root)
    analyzer = Path(__file__).resolve()
    result["analysis_software"] = {
        "path": str(analyzer),
        "sha256": file_sha256(analyzer),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "analysis": str(args.output_json),
                "noninferiority": result["paired_noninferiority"]["noninferiority"],
            }
        )
    )


if __name__ == "__main__":
    main()
