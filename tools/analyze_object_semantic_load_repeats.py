#!/usr/bin/env python3
"""Fail closed on three HostGlobalFixed/M1 SemanticFixed repetitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile


GATE = Path(__file__).with_name("analyze_object_semantic_gate.py")
COMMON = {
    "nodes": [17, 8],
    "world_size": 16,
    "ranks_per_host": 8,
    "model": "gpt2",
    "parallelism": {"pp": 4, "tp": 1, "dp": 4, "mesh_order": "pp-dp"},
    "workload": {
        "sequence_length": 128,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "seed": 20260913,
        "precision": "bf16",
    },
    "scheduler": "constant",
    "optimizer": {
        "name": "scalar_adamw",
        "learning_rate": 5e-5,
        "weight_decay": 0.0,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
    },
}
SOURCE_SCOPE = "synchronized base, ten training steps, and final drain"
TIMINGS = (
    "publisher_queue_ms",
    "enqueue_to_commit_ms",
    "distributed_commit_ms",
    "local_bundle_commit_ms",
    "manifest_gather_ms",
)


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a finite non-negative number")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("expected a finite non-negative number")
    return value


def summary(values: list[float]) -> dict:
    if not values:
        raise ValueError("summary has no samples")
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
    }


def p95(values: list[float]) -> float:
    if not values:
        raise ValueError("p95 has no samples")
    return sorted(values)[math.ceil(len(values) * 0.95) - 1]


def source_digests(root: Path) -> list[str]:
    paths = sorted((root / "source" / "digests").glob("rank_*.json"))
    expected = [f"rank_{rank:05d}.json" for rank in range(16)]
    if [path.name for path in paths] != expected:
        raise ValueError(f"{root}: source digests are incomplete")
    values = [load(path) for path in paths]
    if any(value.get("rank") != rank or value.get("iteration") != 10 for rank, value in enumerate(values)):
        raise ValueError(f"{root}: source digest identity changed")
    digests = [value.get("state_checksum") for value in values]
    if any(not isinstance(value, str) for value in digests):
        raise ValueError(f"{root}: source digest is missing")
    return digests


def plan_load(value: object, *, required: bool) -> dict | None:
    if value is None:
        if required:
            raise ValueError("run is missing required plan_load_metrics")
        return None
    expected = {
        "publisher_policy",
        "all_publication_publisher_bytes",
        "total_all_publication_logical_bytes",
        "maximum_all_publication_publisher_bytes",
        "total_remote_logical_bytes",
        "maximum_remote_publisher_bytes",
    }
    if not isinstance(value, dict) or set(value) != expected or not isinstance(value["publisher_policy"], str):
        raise ValueError("plan_load_metrics schema changed")
    loads = value["all_publication_publisher_bytes"]
    if (
        not isinstance(loads, list)
        or len(loads) != 16
        or any(not isinstance(item, list) or len(item) != 2 or type(item[0]) is not int or type(item[1]) is not int or item[1] < 0 for item in loads)
        or [item[0] for item in loads] != list(range(16))
        or any(type(value[key]) is not int or value[key] < 0 for key in expected - {"publisher_policy", "all_publication_publisher_bytes"})
        or value["total_all_publication_logical_bytes"] != sum(item[1] for item in loads)
        or value["maximum_all_publication_publisher_bytes"] != max(item[1] for item in loads)
        or value["maximum_remote_publisher_bytes"] > value["total_remote_logical_bytes"]
    ):
        raise ValueError("plan_load_metrics are invalid")
    return value


def timing_summary(value: object) -> dict:
    if not isinstance(value, list) or [item.get("rank") for item in value] != list(range(16)):
        raise ValueError("per-rank timing evidence is incomplete")
    result = {}
    for name in TIMINGS:
        samples = []
        for rank in value:
            epochs = rank.get("epochs")
            if not isinstance(epochs, list) or [item.get("epoch") for item in epochs] != list(range(1, 11)):
                raise ValueError("per-rank timing epochs are incomplete")
            samples.extend(number(item.get(name)) for item in epochs)
        result[name] = {**summary(samples), "p95": p95(samples)}
    return result


def rerun(root: Path, temporary: Path) -> dict:
    output = temporary / f"{hashlib.sha256(str(root).encode()).hexdigest()}.json"
    subprocess.run([sys.executable, str(GATE), str(root), "--output", str(output)], check=True)
    return load(output)


def run_metrics(root: Path, analysis: dict, *, require_plan: bool) -> dict:
    contract = load(root / "run-contract.json")
    if any(contract.get(key) != value for key, value in COMMON.items()):
        raise ValueError(f"{root}: fixed GPT-2 contract changed")
    if contract.get("source_steps") != 10 or contract.get("checkpoint_enabled") is not True:
        raise ValueError(f"{root}: source contract changed")
    source = analysis.get("source")
    phase = analysis.get("phase_times", {}).get("source")
    if (
        not isinstance(source, dict)
        or source.get("rank_metrics") != 16
        or source.get("rank_digests") != 16
        or not isinstance(phase, dict)
        or phase.get("scope") != SOURCE_SCOPE
        or not isinstance(phase.get("duration_ns"), int)
        or phase["duration_ns"] < 0
    ):
        raise ValueError(f"{root}: source evidence is incomplete")
    suffix = source.get("suffix_transfer")
    if not isinstance(suffix, list) or [item.get("epoch") for item in suffix] != list(range(1, 11)):
        raise ValueError(f"{root}: suffix epochs are incomplete")
    remote = [number(item.get("remote_gradient_logical_bytes")) for item in suffix]
    base_metrics = source.get("base_transfer", {}).get("metrics")
    if not isinstance(base_metrics, dict):
        raise ValueError(f"{root}: base transfer metrics are incomplete")
    base_maximum_put = number(base_metrics.get("maximum_remote_put_ms"))
    suffix_maximum_put = max(number(item.get("maximum_remote_put_ms")) for item in suffix)
    drains = source.get("per_rank_final_drain_ms")
    if not isinstance(drains, list) or len(drains) != 16:
        raise ValueError(f"{root}: final-drain evidence is incomplete")
    return {
        "contract": contract,
        "source_phase": {"duration_ns": phase["duration_ns"], "scope": phase["scope"]},
        "source_final_digests": source_digests(root),
        "source_loss": analysis.get("loss_trace", {}).get("source"),
        "remote_gradient_logical_bytes": sum(remote),
        "base_maximum_remote_put_ms": base_maximum_put,
        "suffix_maximum_remote_put_ms": suffix_maximum_put,
        "overall_maximum_remote_put_ms": max(base_maximum_put, suffix_maximum_put),
        "final_drain_ms": {**summary([number(value) for value in drains]), "p95": p95([number(value) for value in drains])},
        "timing_ms": timing_summary(source.get("per_rank_local_commit_timing")),
        "plan_load_metrics": plan_load(source.get("plan_load_metrics"), required=require_plan),
    }


def arm_statistics(runs: list[dict]) -> dict:
    result = {
        "source_phase_duration_ns": summary([run["source_phase"]["duration_ns"] for run in runs]),
        "final_drain_ms": {key: summary([run["final_drain_ms"][key] for run in runs]) for key in ("mean", "p95", "max")},
        "timing_ms": {},
    }
    for name in ("base_maximum_remote_put_ms", "suffix_maximum_remote_put_ms", "overall_maximum_remote_put_ms"):
        result[name] = summary([run[name] for run in runs])
    for name in TIMINGS:
        result["timing_ms"][name] = {key: summary([run["timing_ms"][name][key] for run in runs]) for key in ("mean", "p95", "max")}
    return result


def reduction(host: dict, m1: dict) -> dict:
    value = {"host_mean": host["mean"], "m1_mean": m1["mean"]}
    value["reduction_percent"] = None if host["mean"] == 0 else 100.0 * (host["mean"] - m1["mean"]) / host["mean"]
    return value


def reductions(host: dict, m1: dict) -> dict:
    result = {
        "source_phase_duration_ns": reduction(host["source_phase_duration_ns"], m1["source_phase_duration_ns"]),
        "final_drain_ms": {key: reduction(host["final_drain_ms"][key], m1["final_drain_ms"][key]) for key in ("mean", "p95", "max")},
        "timing_ms": {},
    }
    for name in ("base_maximum_remote_put_ms", "suffix_maximum_remote_put_ms", "overall_maximum_remote_put_ms"):
        result[name] = reduction(host[name], m1[name])
    for name in TIMINGS:
        result["timing_ms"][name] = {key: reduction(host["timing_ms"][name][key], m1["timing_ms"][name][key]) for key in ("mean", "p95", "max")}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("host1", type=Path)
    parser.add_argument("m1_1", type=Path)
    parser.add_argument("host2", type=Path)
    parser.add_argument("m1_2", type=Path)
    parser.add_argument("host3", type=Path)
    parser.add_argument("m1_3", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roles = (("host1", "host_global_fixed", False), ("m1_1", "m1", True), ("host2", "host_global_fixed", True), ("m1_2", "m1", True), ("host3", "host_global_fixed", True), ("m1_3", "m1", True))
    roots = {name: getattr(args, name).resolve() for name, _, _ in roles}
    if len(set(roots.values())) != len(roots):
        raise ValueError("repeat roots must be distinct")
    runs = {}
    with tempfile.TemporaryDirectory(prefix="object-semantic-load-repeats-") as directory:
        temporary = Path(directory)
        for name, arm, require_plan in roles:
            root = roots[name]
            if (
                not (root / "COMPLETE").is_file()
                or any((root / marker).exists() for marker in ("FAILED", "INVALIDATED", "EXCLUDED"))
            ):
                raise ValueError(f"{name}: run is not complete")
            analysis = rerun(root, temporary)
            if analysis.get("arm") != arm:
                raise ValueError(f"{name}: arm changed")
            runs[name] = {"root": str(root), "arm": arm, **run_metrics(root, analysis, require_plan=require_plan)}
    detector_hashes = {name: hashlib.sha256((Path(run["root"]) / "detector-experiment-config.json").read_bytes()).hexdigest() for name, run in runs.items()}
    if len(set(detector_hashes.values())) != 1:
        raise ValueError("detector config differs across repeats")
    first = next(iter(runs.values()))
    if any(run["source_loss"] != first["source_loss"] or run["source_final_digests"] != first["source_final_digests"] for run in runs.values()):
        raise ValueError("source loss or final digests differ across repeats")
    if any(run["remote_gradient_logical_bytes"] != first["remote_gradient_logical_bytes"] for run in runs.values()):
        raise ValueError("source remote logical total differs across repeats")
    by_arm = {arm: [runs[name] for name, candidate, _ in roles if candidate == arm] for arm in ("host_global_fixed", "m1")}
    for arm, values in by_arm.items():
        present = [value["plan_load_metrics"] for value in values if value["plan_load_metrics"] is not None]
        if not present or any(value != present[0] for value in present):
            raise ValueError(f"{arm}: plan load differs across repeats")
    host_plan = by_arm["host_global_fixed"][1]["plan_load_metrics"]
    m1_plan = by_arm["m1"][0]["plan_load_metrics"]
    if any(host_plan[key] != m1_plan[key] for key in ("total_all_publication_logical_bytes", "total_remote_logical_bytes")):
        raise ValueError("Host/M1 logical totals differ")
    statistics_by_arm = {arm: arm_statistics(values) for arm, values in by_arm.items()}
    result = {
        "schema_version": 1,
        "detector_config_sha256": next(iter(detector_hashes.values())),
        "source_loss": first["source_loss"],
        "source_final_digests": first["source_final_digests"],
        "remote_gradient_logical_bytes": first["remote_gradient_logical_bytes"],
        "runs": runs,
        "three_sample_statistics": statistics_by_arm,
        "m1_vs_host_reduction_percent": reductions(statistics_by_arm["host_global_fixed"], statistics_by_arm["m1"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
