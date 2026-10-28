#!/usr/bin/env python3
"""Fail closed on three recovery-valid SemanticFixed treatment repeats."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile


ARMS = ("adacheck_fixed", "host_global_fixed", "m1")
SUITE_ANALYZER = Path(__file__).with_name("analyze_object_semantic_suite.py")
TIMINGS = (
    "publisher_queue_ms",
    "enqueue_to_commit_ms",
    "distributed_commit_ms",
)


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("expected a finite non-negative number")
    return result


def sample_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def p95(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute p95 of an empty sample")
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def run_suite(reference: Path, roots: dict[str, Path], output: Path) -> dict:
    subprocess.run(
        [
            sys.executable,
            str(SUITE_ANALYZER),
            str(reference),
            *(str(roots[arm]) for arm in ARMS),
            "--output",
            str(output),
        ],
        check=True,
    )
    result = load(output)
    if result.get("valid") is not True:
        raise ValueError("recomputed four-arm suite is invalid")
    expected = {"nockpt": str(reference.resolve())}
    expected.update({arm: str(root.resolve()) for arm, root in roots.items()})
    if result.get("arms") != expected:
        raise ValueError("recomputed suite roots changed")
    return result


def raw_metrics(root: Path, arm: str, source_steps: int, world_size: int) -> dict:
    raw = load(root / "analysis.json")
    contract = load(root / "run-contract.json")
    if raw.get("arm") != arm or contract.get("arm") != arm:
        raise ValueError(f"{root}: arm identity changed")
    if raw.get("run_id") != contract.get("run_id") or raw.get("run_id") != root.name:
        raise ValueError(f"{root}: run identity changed")
    source = raw.get("source")
    if not isinstance(source, dict):
        raise ValueError(f"{root}: source analysis is missing")
    drains = source.get("per_rank_final_drain_ms")
    if not isinstance(drains, list) or len(drains) != world_size:
        raise ValueError(f"{root}: final-drain evidence is incomplete")
    drain_values = [number(value) for value in drains]
    timing = source.get("per_rank_local_commit_timing")
    if not isinstance(timing, list) or [item.get("rank") for item in timing] != list(range(world_size)):
        raise ValueError(f"{root}: commit timing ranks are incomplete")
    flattened = {name: [] for name in TIMINGS}
    for rank in timing:
        epochs = rank.get("epochs")
        if not isinstance(epochs, list) or [item.get("epoch") for item in epochs] != list(range(1, source_steps + 1)):
            raise ValueError(f"{root}: commit timing epochs are incomplete")
        for item in epochs:
            for name in TIMINGS:
                flattened[name].append(number(item.get(name)))
    plan = source.get("plan_load_metrics")
    if not isinstance(plan, dict):
        raise ValueError(f"{root}: plan-load metrics are missing")
    loads = plan.get("all_publication_publisher_bytes")
    if (
        not isinstance(loads, list)
        or any(not isinstance(item, list) or len(item) != 2 for item in loads)
        or [item[0] for item in loads] != list(range(world_size))
        or any(type(item[1]) is not int or item[1] < 0 for item in loads)
        or plan.get("maximum_all_publication_publisher_bytes") != max(item[1] for item in loads)
    ):
        raise ValueError(f"{root}: plan-load metrics are invalid")
    return {
        "final_drain_ms": {
            **sample_summary(drain_values),
            "p95": p95(drain_values),
        },
        "commit_timing_ms": {
            name: {**sample_summary(values), "p95": p95(values)}
            for name, values in flattened.items()
        },
        "plan_load_metrics": plan,
    }


def reduction(baseline: dict, m1: dict) -> dict:
    baseline_mean = number(baseline["mean"])
    m1_mean = number(m1["mean"])
    return {
        "baseline_mean": baseline_mean,
        "m1_mean": m1_mean,
        "reduction_percent": None if baseline_mean == 0 else 100.0 * (baseline_mean - m1_mean) / baseline_mean,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("nockpt", type=Path)
    for trial in range(1, 4):
        for arm in ARMS:
            parser.add_argument(f"{arm}_{trial}", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = args.nockpt.resolve()
    trial_roots = [
        {arm: getattr(args, f"{arm}_{trial}").resolve() for arm in ARMS}
        for trial in range(1, 4)
    ]
    checkpoint_roots = [root for roots in trial_roots for root in roots.values()]
    if len(set(checkpoint_roots)) != 9 or reference in checkpoint_roots:
        raise ValueError("each treatment repeat must use a distinct root")
    for root in [reference, *checkpoint_roots]:
        if (
            not (root / "COMPLETE").is_file()
            or any((root / marker).exists() for marker in ("FAILED", "INVALIDATED", "EXCLUDED"))
        ):
            raise ValueError(f"{root}: run is not an admitted COMPLETE root")

    trials = []
    with tempfile.TemporaryDirectory(prefix="object-semantic-recovery-repeats-") as directory:
        temporary = Path(directory)
        for index, roots in enumerate(trial_roots, start=1):
            suite = run_suite(reference, roots, temporary / f"trial-{index}.json")
            source_steps = int(suite["checkpoint_fault_contract"]["source_steps"])
            world_size = int(suite["common_training_contract"]["world_size"])
            trials.append(
                {
                    "trial": index,
                    "suite": suite,
                    "raw": {
                        arm: raw_metrics(roots[arm], arm, source_steps, world_size)
                        for arm in ARMS
                    },
                }
            )

    first = trials[0]["suite"]
    first_detector = set(first["detector_config_sha256"].values())
    for trial in trials[1:]:
        suite = trial["suite"]
        if suite["common_training_contract"] != first["common_training_contract"]:
            raise ValueError("training contract differs across repeats")
        if suite["checkpoint_fault_contract"] != first["checkpoint_fault_contract"]:
            raise ValueError("fault contract differs across repeats")
        if set(suite["detector_config_sha256"].values()) != first_detector:
            raise ValueError("detector configuration differs across repeats")
    if len(first_detector) != 1:
        raise ValueError("suite arms do not share one detector configuration")

    fields = {
        "suffix_gradient_logical_bytes": lambda item: item["transfer"]["suffix_remote_gradient_logical_bytes"],
        "suffix_application_bytes": lambda item: item["transfer"]["suffix_remote_payload_bytes"],
        "total_application_bytes": lambda item: item["transfer"]["total_remote_payload_bytes"],
        "total_wire_bytes": lambda item: item["transfer"]["total_remote_wire_bytes"],
        "suffix_maximum_put_ms": lambda item: item["maximum_remote_put_ms"]["suffix"],
        "source_phase_seconds": lambda item: item["phase"]["duration_ns"]["source"] / 1e9,
        "restore_phase_seconds": lambda item: item["phase"]["duration_ns"]["restore"] / 1e9,
        "maximum_rto_ms": lambda item: item["rto_ms"]["max"],
        "source_step_wall_seconds": lambda item: item["step_wall"]["source"]["total_seconds"],
        "maximum_boundary_lag": lambda item: item["cadence"]["maximum_boundary_lag"],
    }
    statistics_by_arm = {}
    reference_summary = first["summary"]["nockpt"]
    reference_step_wall = number(
        reference_summary["step_wall"]["reference"]["mean_seconds"]
    )
    for arm in ARMS:
        statistics_by_arm[arm] = {
            name: sample_summary([number(extract(trial["suite"]["summary"][arm])) for trial in trials])
            for name, extract in fields.items()
        }
        statistics_by_arm[arm]["maximum_final_drain_ms"] = sample_summary(
            [number(trial["raw"][arm]["final_drain_ms"]["max"]) for trial in trials]
        )
        statistics_by_arm[arm]["commit_timing_ms"] = {
            name: {
                key: sample_summary([number(trial["raw"][arm]["commit_timing_ms"][name][key]) for trial in trials])
                for key in ("mean", "p95", "max")
            }
            for name in TIMINGS
        }
        plans = [trial["raw"][arm]["plan_load_metrics"] for trial in trials]
        if any(plan != plans[0] for plan in plans[1:]):
            raise ValueError(f"{arm}: planned load differs across repeats")
        statistics_by_arm[arm]["plan_load_metrics"] = plans[0]
        arm_step_wall = number(
            statistics_by_arm[arm]["source_step_wall_seconds"]["mean"]
        ) / int(first["checkpoint_fault_contract"]["source_steps"])
        statistics_by_arm[arm]["training_step_wall_vs_nockpt"] = {
            "single_reference_mean_seconds": reference_step_wall,
            "treatment_three_run_mean_seconds": arm_step_wall,
            "overhead_percent": 100.0 * (arm_step_wall - reference_step_wall)
            / reference_step_wall,
        }

    reductions = {}
    for baseline in ("adacheck_fixed", "host_global_fixed"):
        reductions[f"m1_vs_{baseline}"] = {
            name: reduction(statistics_by_arm[baseline][name], statistics_by_arm["m1"][name])
            for name in (
                "suffix_gradient_logical_bytes",
                "suffix_application_bytes",
                "total_application_bytes",
                "total_wire_bytes",
                "suffix_maximum_put_ms",
                "source_phase_seconds",
                "restore_phase_seconds",
                "maximum_rto_ms",
                "source_step_wall_seconds",
                "maximum_final_drain_ms",
            )
        }

    result = {
        "schema_version": 1,
        "valid": True,
        "repeat_count": 3,
        "reference_root": str(reference),
        "common_training_contract": first["common_training_contract"],
        "checkpoint_fault_contract": first["checkpoint_fault_contract"],
        "detector_config_sha256": next(iter(first_detector)),
        "single_nockpt_reference": reference_summary,
        "trial_roots": [{arm: str(root) for arm, root in roots.items()} for roots in trial_roots],
        "three_sample_statistics": statistics_by_arm,
        "reductions": reductions,
        "trials": trials,
        "claim_boundary": {
            "recovery": "three fresh-process node8 checkpoint-service-loss trials per arm; exact suite oracle required",
            "cadence": "final drain allowed; strict 1C1S is reported but is not an admission requirement",
            "timing": "n=3 run-level descriptive statistics; no inferential significance claim",
            "fault": "checkpoint-service loss, not physical compute-host loss",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
