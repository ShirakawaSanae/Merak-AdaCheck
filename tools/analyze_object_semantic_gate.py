#!/usr/bin/env python3
"""Fail closed on the small evidence set produced by the SemanticFixed gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def exact_rank_files(root: Path, world_size: int, *, require_rank_field: bool = True) -> list[dict]:
    paths = sorted(root.glob("rank_*.json"))
    expected = [f"rank_{rank:05d}.json" for rank in range(world_size)]
    if [path.name for path in paths] != expected:
        raise ValueError(f"{root}: expected exactly 16 rank digests")
    values = [load(path) for path in paths]
    if require_rank_field and any(value.get("rank") != rank for rank, value in enumerate(values)):
        raise ValueError(f"{root}: rank digest identity mismatch")
    return values


def rank_digests(root: Path, world_size: int, expected_iteration: int) -> list[dict]:
    values = exact_rank_files(root, world_size)
    if any(value.get("iteration") != expected_iteration for value in values):
        raise ValueError(f"{root}: rank digest iteration mismatch")
    if any(not isinstance(value.get("state_checksum"), str) for value in values):
        raise ValueError(f"{root}: rank digest checksum missing")
    return values


def identical_metrics(values: list[dict], key: str) -> object:
    records = [value.get(key) for value in values]
    if any(record is None for record in records):
        raise ValueError(f"source: {key} is missing")
    encoded = {json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records}
    if len(encoded) != 1:
        raise ValueError(f"source: {key} differs across ranks and cannot be counted once")
    return records[0]


def shared_base_result(values: list[dict]) -> dict:
    modes = {value.get("base_mode", "rank_local_v1") for value in values}
    if modes != {"dp_deduplicated_v1"}:
        record = identical_metrics(values, "base_result")
        if not isinstance(record, dict):
            raise ValueError("source: base_result schema changed")
        return record
    records = [value.get("base_result") for value in values]
    if any(
        not isinstance(record, dict)
        or set(record) != {"locator", "metrics", "local_base_id"}
        for record in records
    ):
        raise ValueError("source: deduplicated base_result schema changed")
    for key in ("locator", "metrics"):
        if len({json.dumps(record[key], sort_keys=True, separators=(",", ":")) for record in records}) != 1:
            raise ValueError(f"source: deduplicated base {key} differs across ranks")
    if any(record["local_base_id"] != value.get("base_id") for value, record in zip(values, records)):
        raise ValueError("source: deduplicated local base id differs from its rank metric")
    return {
        "base_mode": "dp_deduplicated_v1",
        "locator": records[0]["locator"],
        "metrics": records[0]["metrics"],
        "rank_base_ids": [record["local_base_id"] for record in records],
    }


def optional_plan_load_metrics(values: list[dict], world_size: int) -> dict | None:
    records = [value.get("plan_load_metrics") for value in values]
    if not any(record is not None for record in records):
        return None
    if any(record is None for record in records):
        raise ValueError("source: plan load metrics are missing on some ranks")
    expected = {
        "publisher_policy",
        "all_publication_publisher_bytes",
        "total_all_publication_logical_bytes",
        "maximum_all_publication_publisher_bytes",
        "total_remote_logical_bytes",
        "maximum_remote_publisher_bytes",
    }
    extended = expected | {
        "confirmation_mode",
        "local_staged_logical_bytes",
        "local_verifier_logical_bytes",
    }
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("source: plan load metric schema changed")
    schemas = {frozenset(record) for record in records}
    if schemas == {frozenset(expected)}:
        record = identical_metrics(values, "plan_load_metrics")
    elif schemas == {frozenset(extended)}:
        local_fields = {
            "local_staged_logical_bytes",
            "local_verifier_logical_bytes",
        }
        common = [
            {key: value for key, value in record.items() if key not in local_fields}
            for record in records
        ]
        if len(
            {
                json.dumps(value, sort_keys=True, separators=(",", ":"))
                for value in common
            }
        ) != 1 or any(
            type(record[name]) is not int or record[name] < 0
            for record in records
            for name in local_fields
        ):
            raise ValueError("source: extended plan load metrics are inconsistent")
        record = {
            **common[0],
            "staged_logical_bytes_by_rank": [
                [rank, value["local_staged_logical_bytes"]]
                for rank, value in enumerate(records)
            ],
            "verifier_logical_bytes_by_rank": [
                [rank, value["local_verifier_logical_bytes"]]
                for rank, value in enumerate(records)
            ],
        }
    else:
        raise ValueError("source: plan load metric schema changed")
    loads = record["all_publication_publisher_bytes"]
    if (
        not isinstance(record["publisher_policy"], str)
        or (
            "confirmation_mode" in record
            and record["confirmation_mode"] != "post_allreduce_dp_invariant"
        )
        or not isinstance(loads, list)
        or len(loads) != world_size
        or any(
            not isinstance(item, list)
            or len(item) != 2
            or type(item[0]) is not int
            or type(item[1]) is not int
            or item[1] < 0
            for item in loads
        )
        or [item[0] for item in loads] != list(range(world_size))
        or any(
            type(record[name]) is not int or record[name] < 0
            for name in expected - {"publisher_policy", "all_publication_publisher_bytes"}
        )
        or record["total_all_publication_logical_bytes"]
        != sum(item[1] for item in loads)
        or record["maximum_all_publication_publisher_bytes"]
        != max(item[1] for item in loads)
        or record["maximum_remote_publisher_bytes"]
        > record["total_remote_logical_bytes"]
    ):
        raise ValueError("source: plan load metrics are invalid")
    return record


TRANSFER_FIELDS = (
    "remote_gradient_logical_bytes",
    "remote_data_payload_bytes",
    "remote_control_payload_bytes",
    "remote_wire_bytes",
    "maximum_remote_put_ms",
    "total_remote_payload_bytes",
)


def shared_suffix_transfer(values: list[dict], source_steps: int) -> tuple[list[dict], list[dict]]:
    records = [value.get("commit_metrics") for value in values]
    if any(not isinstance(items, list) or len(items) != source_steps for items in records):
        raise ValueError("source: commit_metrics must cover every source epoch on every rank")
    shared = []
    for index, epoch in enumerate(range(1, source_steps + 1)):
        candidates = []
        for items in records:
            item = items[index]
            if item.get("epoch") != epoch or any(
                not isinstance(item.get(name), (int, float)) for name in TRANSFER_FIELDS
            ):
                raise ValueError("source: commit transfer metrics are incomplete")
            candidates.append({"epoch": epoch, **{name: item[name] for name in TRANSFER_FIELDS}})
        if len({json.dumps(item, sort_keys=True) for item in candidates}) != 1:
            raise ValueError("source: global commit transfer differs across ranks")
        shared.append(candidates[0])
    local = [
        {
            "rank": rank,
            "epochs": [
                {
                    name: item[name]
                    for name in (
                        "epoch",
                        "capture_ms",
                        "publisher_queue_ms",
                        "snapshot_ready_wait_ms",
                        "enqueue_to_commit_ms",
                        "checksum_confirmation_ms",
                        "externalize_ms",
                        "distributed_commit_ms",
                        "local_bundle_commit_ms",
                        "manifest_gather_ms",
                        "coordinator_evidence_ms",
                        "coordinator_control_commit_ms",
                        "coordinator_seal_locator_ms",
                        "result_broadcast_ms",
                    )
                    if name in item
                }
                for item in items
            ],
        }
        for rank, items in enumerate(records)
    ]
    return shared, local


def restore_base_transfer(value: dict) -> dict:
    base = value.get("base")
    if not isinstance(base, dict):
        base = {
            "payload_bytes": value.get("base_payload_bytes"),
            "wire_bytes": value.get("base_wire_bytes"),
            "get_ms": value.get("base_get_ms"),
        }
    if any(not isinstance(base.get(name), (int, float)) for name in ("payload_bytes", "wire_bytes", "get_ms")):
        raise ValueError("restore: base transfer metrics are incomplete")
    return base


def step_wall(root: Path, phase: str, expected_start: int, expected_count: int) -> list[dict]:
    path = root / "logs" / phase / "node-0.log"
    samples = [
        {"step": int(step), "wall_time_seconds": float(duration)}
        for step, duration in re.findall(
            r"step wall: (\d+) train-plus-checkpoint-poll time \(s\): ([0-9.]+)",
            path.read_text(),
        )
    ]
    steps = [sample["step"] for sample in samples]
    if len(samples) != expected_count or steps != list(
        range(expected_start, expected_start + expected_count)
    ):
        raise ValueError(f"{phase}: rank-zero step-wall samples are incomplete")
    return samples


def loss_trace(root: Path, phase: str, expected_start: int, expected_count: int) -> list[dict]:
    samples = [
        {"step": int(step), "loss": float(loss)}
        for step, loss in re.findall(
            r"steps: (\d+) loss: ([^ ]+)",
            (root / "logs" / phase / "node-0.log").read_text(),
        )
    ]
    expected = list(range(expected_start, expected_start + expected_count))
    if [sample["step"] for sample in samples] != expected or any(
        not math.isfinite(sample["loss"]) for sample in samples
    ):
        raise ValueError(f"{phase}: rank-zero loss trace is incomplete or non-finite")
    return samples


def locator(path: Path, *, kind: str, epoch: int, arm: str, run_id: str, world_size: int) -> dict:
    value = load(path)
    expected = "object_semantic_base_locator" if kind == "base" else "object_optimizer_update_locator"
    if value.get("kind") != expected or value.get("epoch") != epoch:
        raise ValueError(f"{path}: locator kind or epoch mismatch")
    if value.get("world_size") != world_size or value.get("run_id") != run_id:
        raise ValueError(f"{path}: locator run identity mismatch")
    if kind == "suffix" and value.get("arm") != arm:
        raise ValueError(f"{path}: locator arm mismatch")
    stored = value.pop("locator_hash", None)
    if stored != digest(value):
        raise ValueError(f"{path}: locator hash mismatch")
    value["locator_hash"] = stored
    return value


def contract_steps(contract: dict, *, checkpoint_enabled: bool) -> tuple[int, int, int]:
    source = contract.get("source_steps")
    continued = contract.get("restore_continue_steps")
    reference = contract.get("reference_steps")
    if type(source) is not int or source < 1 or type(continued) is not int or continued < 0:
        raise ValueError("run step contract is invalid")
    if not checkpoint_enabled:
        if continued != 0:
            raise ValueError("reference continuation contract is invalid")
        reference = source if reference is None else reference
    else:
        reference = source + continued if reference is None else reference
    if checkpoint_enabled and (type(reference) is not int or reference != source + continued or continued < 1):
        raise ValueError("checkpoint step contract is invalid")
    if type(reference) is not int or reference < 1:
        raise ValueError("reference step contract is invalid")
    return source, continued, reference


def validate_topology(contract: dict) -> int:
    parallelism = contract.get("parallelism")
    if (
        contract.get("nodes") != [17, 8]
        or contract.get("world_size") != 16
        or contract.get("ranks_per_host") != 8
        or contract.get("model") not in {"gpt2", "gpt_6b", "llama_3_8b"}
        or not isinstance(parallelism, dict)
        or set(parallelism) != {"pp", "tp", "dp", "mesh_order"}
        or any(type(parallelism[name]) is not int or parallelism[name] < 1 for name in ("pp", "tp", "dp"))
        or parallelism["pp"] not in (2, 4)
        or parallelism["tp"] not in (1, 2)
        or parallelism["pp"] * parallelism["tp"] * parallelism["dp"] != 16
        or parallelism["mesh_order"] != "pp-dp"
    ):
        raise ValueError("run workload/parallelism contract is not frozen")
    return int(contract["world_size"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    contract = load(root / "run-contract.json")
    arm, run_id = contract["arm"], contract["run_id"]
    restore_enabled = bool(contract["restore_enabled"])
    if contract.get("scheduler") != "constant" or contract.get("optimizer") != {
        "name": "scalar_adamw",
        "learning_rate": 5e-5,
        "weight_decay": 0.0,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
    }:
        raise ValueError("run optimizer/scheduler contract is not frozen")
    workload = contract.get("workload")
    if (
        not isinstance(workload, dict)
        or set(workload) != {
            "sequence_length",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "seed",
            "precision",
        }
        or type(workload["sequence_length"]) is not int
        or workload["sequence_length"] < 1
        or workload["per_device_train_batch_size"] != 1
        or type(workload["gradient_accumulation_steps"]) is not int
        or workload["gradient_accumulation_steps"] < 1
        or workload["seed"] != 20260913
        or workload["precision"] != "bf16"
    ):
        raise ValueError("run workload contract is not frozen")
    world_size = validate_topology(contract)
    if arm == "nockpt":
        if contract.get("checkpoint_enabled") is not False or restore_enabled:
            raise ValueError("reference: checkpoint contract is not disabled")
        _, _, reference_steps = contract_steps(contract, checkpoint_enabled=False)
        statuses = load(root / "launcher-status.json")
        if statuses != {"reference": {"local": 0, "remote": 0}}:
            raise ValueError("reference: distributed launcher status is not clean")
        phase_times = load(root / "phase-times.json")
        if set(phase_times) != {"reference"} or phase_times["reference"].get("duration_ns", -1) < 0:
            raise ValueError("reference: phase timing evidence is incomplete")
        result = {
            "schema_version": 1,
            "arm": arm,
            "run_id": run_id,
            "reference": {
                "rank_digests": len(rank_digests(root / "reference" / "digests", world_size, reference_steps)),
                "step_wall": step_wall(root, "reference", 1, reference_steps),
                "loss_trace": loss_trace(root, "reference", 1, reference_steps),
            },
            "phase_times": phase_times,
        }
        destination = args.output or root / "analysis.json"
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return 0
    if restore_enabled and (
        contract.get("failed_service_node") not in (17, 8)
        or contract.get("surviving_host") not in (0, 1)
    ):
        raise ValueError("restore: failure-domain contract is incomplete")
    source_steps, continue_steps, reference_steps = contract_steps(
        contract, checkpoint_enabled=True
    )
    if contract.get("restore_epoch") != source_steps:
        raise ValueError("restore: epoch differs from the source contract")
    statuses = load(root / "launcher-status.json")
    expected_phases = ["source"] + (["restore"] if restore_enabled else [])
    if set(statuses) != set(expected_phases):
        raise ValueError("launcher-status phases mismatch")
    for phase in expected_phases:
        if statuses[phase] != {"local": 0, "remote": 0}:
            raise ValueError(f"{phase}: distributed launcher status is not clean")
    phase_times = load(root / "phase-times.json")
    if set(phase_times) != set(expected_phases) or any(
        not isinstance(phase_times[phase].get("started_monotonic_ns"), int)
        or not isinstance(phase_times[phase].get("finished_monotonic_ns"), int)
        or not isinstance(phase_times[phase].get("duration_ns"), int)
        or phase_times[phase]["duration_ns"] != (
            phase_times[phase]["finished_monotonic_ns"] - phase_times[phase]["started_monotonic_ns"]
        )
        or phase_times[phase]["duration_ns"] < 0
        or not isinstance(phase_times[phase].get("scope"), str)
        for phase in expected_phases
    ):
        raise ValueError("phase timing evidence is incomplete")

    source = root / "source"
    source_step_wall = step_wall(root, "source", 1, source_steps)
    source_losses = loss_trace(root, "source", 1, source_steps)
    metrics = exact_rank_files(source / "metrics", world_size)
    if any(
        value.get("arm") != arm
        or value.get("base_epoch") != 0
        or value.get("committed_epochs") != list(range(1, source_steps + 1))
        or not value.get("valid")
        or value.get("all_epochs_committed") is not True
        or not isinstance(value.get("maximum_boundary_lag"), int)
        or not isinstance(value.get("strict_1c1s_valid"), bool)
        for value in metrics
    ):
        raise ValueError("source: SemanticFixed metric contract mismatch")
    source_digests = rank_digests(source / "digests", world_size, source_steps)
    base = locator(source / "base" / "epoch-00000000.json", kind="base", epoch=0, arm=arm, run_id=run_id, world_size=world_size)
    suffixes = [
        locator(source / "updates" / f"epoch-{epoch:08d}.json", kind="suffix", epoch=epoch, arm=arm, run_id=run_id, world_size=world_size)
        for epoch in range(1, source_steps + 1)
    ]
    if len({value["plan_hash"] for value in suffixes}) != 1:
        raise ValueError("source: suffix plan hash changed")
    if len({value["detector_catalog_hash"] for value in suffixes}) != 1:
        raise ValueError("source: suffix detector hash changed")
    base_transfer = shared_base_result(metrics)
    plan_load_metrics = optional_plan_load_metrics(metrics, world_size)
    suffix_transfer, local_commit_timing = shared_suffix_transfer(metrics, source_steps)
    final_drain = [value.get("final_drain_ms") for value in metrics]
    if any(value is not None for value in final_drain) and any(
        not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
        for value in final_drain
    ):
        raise ValueError("source: final-drain timing is incomplete")

    result = {
        "schema_version": 1,
        "arm": arm,
        "run_id": run_id,
        "source": {
            "rank_metrics": len(metrics),
            "rank_digests": len(source_digests),
            "base_locator_hash": base["locator_hash"],
            "suffix_locator_hashes": [value["locator_hash"] for value in suffixes],
            "base_transfer": base_transfer,
            "plan_load_metrics": plan_load_metrics,
            "suffix_transfer": suffix_transfer,
            "per_rank_local_commit_timing": local_commit_timing,
            "per_rank_final_drain_ms": (
                [float(value) for value in final_drain]
                if all(value is not None for value in final_drain)
                else None
            ),
            "maximum_boundary_lag": max(value["maximum_boundary_lag"] for value in metrics),
            "strict_1c1s_valid": all(value["strict_1c1s_valid"] for value in metrics),
            "cadence_claim": contract.get("cadence_claim"),
            "shared_result_counting": f"base and suffix results match on all {world_size} ranks and are counted once",
        },
        "phase_times": phase_times,
        "step_wall": {"source": source_step_wall},
        "loss_trace": {"source": source_losses},
        "restore": None,
    }
    if restore_enabled:
        restore_digests = rank_digests(root / "restore" / "digests", world_size, reference_steps)
        restore_step_wall = step_wall(root, "restore", source_steps + 1, continue_steps)
        restore_losses = loss_trace(root, "restore", source_steps + 1, continue_steps)
        restore_metrics = exact_rank_files(source / "recovery_metrics", world_size)
        expected_plan_hash = suffixes[0]["plan_hash"]
        if any(
            value.get("requested_epoch") != source_steps
            or value.get("final_epoch") != source_steps
            or value.get("base_epoch") != 0
            or value.get("arm") != arm
            or value.get("plan_hash") != expected_plan_hash
            or not isinstance(value.get("mapping_hash"), str)
            or value.get("surviving_host") != contract["surviving_host"]
            or not isinstance(value.get("total_restore_wall_ms", value.get("restore_wall_ms")), (int, float))
            or value.get("total_restore_wall_ms", value.get("restore_wall_ms")) < 0
            or not isinstance(value.get("boundary_checksum", value.get("boundary_state_checksum")), str)
            or not isinstance(value.get("component_checksums", value.get("boundary_component_checksums")), dict)
            or value.get("suffix_get_metrics") != "unavailable"
            or value.get("checksum_diagnostic_only") is not True
            for value in restore_metrics
        ):
            raise ValueError("restore: boundary/RTO metrics are incomplete")
        restore_transfer = [
            {"rank": rank, "base": restore_base_transfer(value)}
            for rank, value in enumerate(restore_metrics)
        ]
        result["restore"] = {
            "rank_digests": len(restore_digests),
            "rank_metrics": len(restore_metrics),
            "per_rank_base_transfer": restore_transfer,
            "step_wall": restore_step_wall,
            "restored_from_epoch": source_steps,
            "continued_to_epoch": reference_steps,
            "service_stats_before": "restore-service-before.json",
            "service_stats_after": "restore-service-after.json",
            "transfer_accounting": (
                "per-rank base GET payload, wire, and bottleneck time come from "
                "the recovery controller; service snapshots remain the aggregate audit"
            ),
        }
        result["loss_trace"]["restore"] = restore_losses
        for name in ("restore-service-before.json", "restore-service-after.json"):
            value = load(root / name)
            if sorted(value) != ["node17", "node8"]:
                raise ValueError(f"{name}: service endpoints missing")
            failed = f"node{contract['failed_service_node']}"
            survivor = "node17" if failed == "node8" else "node8"
            if (
                value[failed].get("status") != "unavailable"
                or value[survivor].get("status") != "ok"
            ):
                raise ValueError(f"{name}: service-loss evidence disagrees with the contract")
    destination = args.output or root / "analysis.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
