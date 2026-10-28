#!/usr/bin/env python3
"""Fail closed on the four-arm two-node SemanticFixed comparison suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


ARMS = ("nockpt", "adacheck_fixed", "host_global_fixed", "m1")
COMMON_KEYS = (
    "nodes",
    "world_size",
    "ranks_per_host",
    "model",
    "parallelism",
    "workload",
    "scheduler",
    "optimizer",
)
FAULT_KEYS = (
    "restore_enabled",
    "source_steps",
    "reference_steps",
    "restore_epoch",
    "restore_continue_steps",
    "failed_service_node",
    "surviving_host",
    "max_inflight",
    "base_upload_parallelism",
    "service_capacity",
)
GATE_ANALYZER = Path(__file__).with_name("analyze_object_semantic_gate.py")


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_digest_hashes(root: Path, phase: str, world_size: int) -> list[str]:
    paths = sorted((root / phase / "digests").glob("rank_*.json"))
    expected = [f"rank_{rank:05d}.json" for rank in range(world_size)]
    if [path.name for path in paths] != expected:
        raise ValueError(f"{root}: final rank digests are incomplete")
    return [digest(path) for path in paths]


def same_contract(runs: dict[str, dict], keys: tuple[str, ...], name: str) -> dict:
    reference = next(iter(runs.values()))["contract"]
    result = {key: reference.get(key) for key in keys}
    if any(run["contract"].get(key) != value for run in runs.values() for key, value in result.items()):
        raise ValueError(f"{name} contract differs across arms")
    return result


def samples(value: object, expected_steps: range, name: str) -> list[dict]:
    if not isinstance(value, list) or [item.get("step") for item in value] != list(expected_steps):
        raise ValueError(f"{name}: step range is incomplete")
    if any(not isinstance(item.get("loss"), (int, float)) or not math.isfinite(item["loss"]) for item in value):
        raise ValueError(f"{name}: loss is not finite")
    return value


def loss_comparison(actual: list[dict], reference: list[dict], *, rtol: float, atol: float) -> dict:
    errors = [abs(float(item["loss"]) - float(expected["loss"])) for item, expected in zip(actual, reference)]
    close = all(
        error <= atol + rtol * abs(float(expected["loss"]))
        for error, expected in zip(errors, reference)
    )
    return {"numerically_close": close, "max_absolute_error": max(errors, default=0.0)}


def wall_summary(values: object, expected_steps: range, name: str) -> dict:
    if not isinstance(values, list) or [item.get("step") for item in values] != list(expected_steps):
        raise ValueError(f"{name}: step-wall evidence is incomplete")
    seconds = [float(item["wall_time_seconds"]) for item in values]
    if any(not math.isfinite(value) or value < 0 for value in seconds):
        raise ValueError(f"{name}: step-wall evidence is invalid")
    return {"count": len(seconds), "total_seconds": sum(seconds), "mean_seconds": sum(seconds) / len(seconds), "max_seconds": max(seconds)}


def phase_scopes(source_steps: int, continue_steps: int, reference_steps: int) -> dict[str, str]:
    return {
        "source": (
            "synchronized base, ten training steps, and final drain"
            if source_steps == 10
            else f"synchronized base, {source_steps} training steps, and final drain"
        ),
        "restore": (
            "fresh construction, restore to step 10, and five continued steps"
            if (source_steps, continue_steps) == (10, 5)
            else f"fresh construction, restore to step {source_steps}, and {continue_steps} continued steps"
        ),
        "reference": (
            "uninterrupted fifteen-step no-checkpoint reference"
            if reference_steps == 15
            else f"uninterrupted {reference_steps}-step no-checkpoint reference"
        ),
    }


def phase_summary(phases: object, expected: set[str], scopes: dict[str, str], name: str) -> dict:
    if not isinstance(phases, dict) or set(phases) != expected:
        raise ValueError(f"{name}: phase evidence is incomplete")
    values = {}
    for phase, item in phases.items():
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("started_monotonic_ns"), int)
            or not isinstance(item.get("finished_monotonic_ns"), int)
            or not isinstance(item.get("duration_ns"), int)
            or item["duration_ns"] != item["finished_monotonic_ns"] - item["started_monotonic_ns"]
            or item["duration_ns"] < 0
            or item.get("scope") != scopes[phase]
        ):
            raise ValueError(f"{name}: phase duration is invalid")
        values[phase] = item["duration_ns"]
    return {"duration_ns": values, "total_duration_ns": sum(values.values())}


def checkpoint_metrics(
    root: Path, analysis: dict, arm: str, *, world_size: int,
    source_steps: int, continue_steps: int, reference_steps: int, scopes: dict[str, str],
) -> dict:
    source, restore = analysis.get("source"), analysis.get("restore")
    if not isinstance(source, dict) or not isinstance(restore, dict):
        raise ValueError(f"{arm}: source/restore analysis is missing")
    if source.get("rank_metrics") != world_size or source.get("rank_digests") != world_size or restore.get("rank_metrics") != world_size or restore.get("rank_digests") != world_size:
        raise ValueError(f"{arm}: rank evidence is incomplete")
    suffix = source.get("suffix_transfer")
    if not isinstance(suffix, list) or [item.get("epoch") for item in suffix] != list(range(1, source_steps + 1)):
        raise ValueError(f"{arm}: suffix transfer epochs are incomplete")
    fields = (
        "remote_gradient_logical_bytes",
        "remote_data_payload_bytes",
        "remote_control_payload_bytes",
        "total_remote_payload_bytes",
        "remote_wire_bytes",
        "maximum_remote_put_ms",
    )
    if any(
        any(not isinstance(item.get(field), (int, float)) or not math.isfinite(item[field]) or item[field] < 0 for field in fields)
        for item in suffix
    ):
        raise ValueError(f"{arm}: suffix transfer metrics are incomplete")
    base = source.get("base_transfer", {}).get("metrics")
    if not isinstance(base, dict) or any(
        not isinstance(base.get(field), (int, float)) or not math.isfinite(base[field]) or base[field] < 0
        for field in ("remote_data_payload_bytes", "remote_control_payload_bytes", "remote_wire_bytes", "maximum_remote_put_ms")
    ):
        raise ValueError(f"{arm}: base transfer metrics are incomplete")
    recovery_paths = sorted((root / "source" / "recovery_metrics").glob("rank_*.json"))
    expected = [f"rank_{rank:05d}.json" for rank in range(world_size)]
    if [path.name for path in recovery_paths] != expected:
        raise ValueError(f"{arm}: per-rank RTO evidence is incomplete")
    rto = [load(path).get("total_restore_wall_ms") for path in recovery_paths]
    if any(not isinstance(value, (int, float)) or value < 0 for value in rto):
        raise ValueError(f"{arm}: RTO evidence is invalid")
    if (
        not isinstance(source.get("maximum_boundary_lag"), int)
        or source["maximum_boundary_lag"] < 0
        or not isinstance(source.get("strict_1c1s_valid"), bool)
        or not isinstance(source.get("cadence_claim"), str)
    ):
        raise ValueError(f"{arm}: cadence evidence is invalid")
    return {
        "transfer": {
            "base_remote_payload_bytes": base["remote_data_payload_bytes"] + base["remote_control_payload_bytes"],
            "base_remote_wire_bytes": base["remote_wire_bytes"],
            "suffix_remote_gradient_logical_bytes": sum(item["remote_gradient_logical_bytes"] for item in suffix),
            "suffix_remote_data_payload_bytes": sum(item["remote_data_payload_bytes"] for item in suffix),
            "suffix_remote_control_payload_bytes": sum(item["remote_control_payload_bytes"] for item in suffix),
            "suffix_remote_payload_bytes": sum(item["total_remote_payload_bytes"] for item in suffix),
            "suffix_remote_wire_bytes": sum(item["remote_wire_bytes"] for item in suffix),
            "total_remote_payload_bytes": base["remote_data_payload_bytes"] + base["remote_control_payload_bytes"] + sum(item["total_remote_payload_bytes"] for item in suffix),
            "total_remote_wire_bytes": base["remote_wire_bytes"] + sum(item["remote_wire_bytes"] for item in suffix),
        },
        "maximum_remote_put_ms": {
            "base": base["maximum_remote_put_ms"],
            "suffix": max(item["maximum_remote_put_ms"] for item in suffix),
            "overall": max([base["maximum_remote_put_ms"], *[item["maximum_remote_put_ms"] for item in suffix]]),
        },
        "rto_ms": {"count": len(rto), "mean": sum(rto) / len(rto), "max": max(rto)},
        "cadence": {
            "maximum_boundary_lag": source.get("maximum_boundary_lag"),
            "strict_1c1s_valid": source.get("strict_1c1s_valid"),
            "claim": source.get("cadence_claim"),
        },
        "step_wall": {
            "source": wall_summary(analysis.get("step_wall", {}).get("source"), range(1, source_steps + 1), f"{arm} source"),
            "restore": wall_summary(restore.get("step_wall"), range(source_steps + 1, reference_steps + 1), f"{arm} restore"),
        },
        "phase": phase_summary(analysis.get("phase_times"), {"source", "restore"}, scopes, arm),
    }


def contract_steps(contract: dict, *, checkpoint_enabled: bool) -> tuple[int, int, int]:
    source = contract.get("source_steps")
    continued = contract.get("restore_continue_steps")
    reference = contract.get("reference_steps")
    if type(source) is not int or source < 1 or type(continued) is not int or continued < 0:
        raise ValueError("step contract is invalid")
    if checkpoint_enabled:
        reference = source + continued if reference is None else reference
        if continued < 1 or reference != source + continued or contract.get("restore_epoch") != source:
            raise ValueError("checkpoint step contract is invalid")
    else:
        reference = source if reference is None else reference
        if continued != 0 or reference != source:
            raise ValueError("reference step contract is invalid")
    return source, continued, reference


def validate_common_contract(common: dict) -> int:
    topology = common.get("parallelism")
    workload = common.get("workload")
    optimizer = {
        "name": "scalar_adamw", "learning_rate": 5e-5, "weight_decay": 0.0,
        "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8,
    }
    if (
        common.get("nodes") != [17, 8] or common.get("world_size") != 16
        or common.get("ranks_per_host") != 8
        or common.get("model") not in {"gpt2", "gpt_6b", "llama_3_8b"}
        or not isinstance(topology, dict) or set(topology) != {"pp", "tp", "dp", "mesh_order"}
        or topology.get("pp") not in (2, 4) or topology.get("tp") not in (1, 2)
        or type(topology.get("dp")) is not int or topology["pp"] * topology["tp"] * topology["dp"] != 16
        or topology.get("mesh_order") != "pp-dp"
        or not isinstance(workload, dict)
        or set(workload) != {"sequence_length", "per_device_train_batch_size", "gradient_accumulation_steps", "seed", "precision"}
        or type(workload["sequence_length"]) is not int or workload["sequence_length"] < 1
        or workload["per_device_train_batch_size"] != 1
        or type(workload["gradient_accumulation_steps"]) is not int or workload["gradient_accumulation_steps"] < 1
        or workload["seed"] != 20260913 or workload["precision"] != "bf16"
        or common.get("scheduler") != "constant" or common.get("optimizer") != optimizer
    ):
        raise ValueError("training contract is not a supported GPT-2 gate")
    return int(common["world_size"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("nockpt", type=Path)
    parser.add_argument("adacheck_fixed", type=Path)
    parser.add_argument("host_global_fixed", type=Path)
    parser.add_argument("m1", type=Path)
    parser.add_argument("--loss-rtol", type=float, default=1e-4)
    parser.add_argument("--loss-atol", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.loss_rtol < 0 or args.loss_atol < 0:
        parser.error("loss tolerances must be non-negative")
    roots = dict(zip(ARMS, (args.nockpt, args.adacheck_fixed, args.host_global_fixed, args.m1)))
    runs = {arm: {"root": root.resolve(), "contract": load(root / "run-contract.json")} for arm, root in roots.items()}
    for arm, run in runs.items():
        if (
            not (run["root"] / "COMPLETE").is_file()
            or any((run["root"] / marker).exists() for marker in ("FAILED", "INVALIDATED", "EXCLUDED"))
        ):
            raise ValueError(f"{arm}: run is not complete")
        if run["contract"].get("arm") != arm:
            raise ValueError(f"{arm}: contract arm mismatch")
    with tempfile.TemporaryDirectory(prefix="object-semantic-suite-") as temporary:
        for arm, run in runs.items():
            analysis_path = Path(temporary) / f"{arm}.json"
            subprocess.run(
                [sys.executable, str(GATE_ANALYZER), str(run["root"]), "--output", str(analysis_path)],
                check=True,
            )
            run["analysis"] = load(analysis_path)
            if run["analysis"].get("arm") != arm:
                raise ValueError(f"{arm}: recomputed analysis arm mismatch")
    common = same_contract(runs, COMMON_KEYS, "training")
    world_size = validate_common_contract(common)
    detector_hashes = {arm: digest(run["root"] / "detector-experiment-config.json") for arm, run in runs.items()}
    if len(set(detector_hashes.values())) != 1:
        raise ValueError("detector config differs across arms")
    reference = runs["nockpt"]
    if reference["contract"].get("checkpoint_enabled") is not False or reference["contract"].get("restore_enabled"):
        raise ValueError("NoCheckpoint reference has a checkpoint fault contract")
    _, _, reference_steps = contract_steps(reference["contract"], checkpoint_enabled=False)
    reference_analysis = reference["analysis"].get("reference", {})
    if reference_analysis.get("rank_digests") != world_size:
        raise ValueError("NoCheckpoint reference lacks rank digests")
    reference_loss = samples(reference_analysis.get("loss_trace"), range(1, reference_steps + 1), "NoCheckpoint")
    reference_digests = rank_digest_hashes(reference["root"], "reference", world_size)
    reference_metrics = {
        "transfer": None,
        "maximum_remote_put_ms": None,
        "rto_ms": None,
        "cadence": None,
        "step_wall": {"reference": wall_summary(reference_analysis.get("step_wall"), range(1, reference_steps + 1), "NoCheckpoint")},
        "phase": phase_summary(reference["analysis"].get("phase_times"), {"reference"}, phase_scopes(reference_steps, 0, reference_steps), "NoCheckpoint"),
    }
    checkpoint_runs = {arm: runs[arm] for arm in ARMS[1:]}
    # Older admitted runs predate the knob and used the launcher's hard-coded 4.
    for run in checkpoint_runs.values():
        run["contract"].setdefault("base_upload_parallelism", 4)
    fault = same_contract(checkpoint_runs, FAULT_KEYS, "checkpoint fault")
    capacity = fault.get("service_capacity")
    source_steps, continue_steps, checkpoint_reference_steps = contract_steps(
        next(iter(checkpoint_runs.values()))["contract"], checkpoint_enabled=True
    )
    if (
        any(run["contract"].get("checkpoint_enabled") is not True for run in checkpoint_runs.values())
        or fault["restore_enabled"] is not True
        or fault["source_steps"] != source_steps
        or fault["restore_epoch"] != source_steps
        or fault["restore_continue_steps"] != continue_steps
        or fault["failed_service_node"] not in (8, 17)
        or fault["surviving_host"] != (0 if fault["failed_service_node"] == 8 else 1)
        or type(fault["max_inflight"]) is not int or fault["max_inflight"] < source_steps
        or type(fault["base_upload_parallelism"]) is not int
        or fault["base_upload_parallelism"] < 1
        or not isinstance(capacity, dict)
        or not isinstance(capacity.get("configured_mib"), int)
        or capacity["configured_mib"] < 1
        or type(capacity.get("minimum_mib", 1)) is not int
        or capacity["configured_mib"] < capacity.get("minimum_mib", 1)
    ):
        raise ValueError("checkpoint fault contract is incomplete")
    summaries = {"nockpt": reference_metrics}
    continuity = {}
    final_digest_match = {}
    for arm, run in checkpoint_runs.items():
        analysis = run["analysis"]
        source_loss = samples(analysis.get("loss_trace", {}).get("source"), range(1, source_steps + 1), f"{arm} source")
        restore_loss = samples(analysis.get("loss_trace", {}).get("restore"), range(source_steps + 1, checkpoint_reference_steps + 1), f"{arm} restore")
        source_result = loss_comparison(source_loss, reference_loss[:source_steps], rtol=args.loss_rtol, atol=args.loss_atol)
        restore_result = loss_comparison(restore_loss, reference_loss[source_steps:], rtol=args.loss_rtol, atol=args.loss_atol)
        continuity[arm] = {"source": source_result, "restore": restore_result, "valid": source_result["numerically_close"] and restore_result["numerically_close"]}
        if not continuity[arm]["valid"]:
            raise ValueError(f"{arm}: loss continuation differs from NoCheckpoint")
        final_digest_match[arm] = rank_digest_hashes(run["root"], "restore", world_size) == reference_digests
        if not final_digest_match[arm]:
            raise ValueError(f"{arm}: final state differs from NoCheckpoint")
        summaries[arm] = checkpoint_metrics(
            run["root"], analysis, arm, world_size=world_size,
            source_steps=source_steps, continue_steps=continue_steps,
            reference_steps=checkpoint_reference_steps,
            scopes=phase_scopes(source_steps, continue_steps, checkpoint_reference_steps),
        )
    result = {
        "schema_version": 1,
        "arms": {arm: str(run["root"]) for arm, run in runs.items()},
        "common_training_contract": common,
        "detector_config_sha256": detector_hashes,
        "checkpoint_fault_contract": fault,
        "loss_continuity": continuity,
        "final_digest_match": final_digest_match,
        "loss_tolerances": {"rtol": args.loss_rtol, "atol": args.loss_atol},
        "summary": summaries,
        "valid": all(value["valid"] for value in continuity.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
