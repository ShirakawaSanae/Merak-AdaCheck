#!/usr/bin/env python3
"""Combine protocol tests and two-host restore artifacts into a fault matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


PROTOCOL_CASES = {
    "missing_delta_before_control": "test_peer_control_rejects_missing_delta_record",
    "corrupt_source_before_send": "test_peer_shard_rejects_corrupt_source_tensor",
    "prepared_delta_without_control": (
        "test_prepared_peer_delta_without_control_is_not_recoverable"
    ),
    "stale_epoch_restore": "test_peer_delta_restore_rejects_stale_epoch",
    "corrupt_transport_payload": "test_service_rejects_corrupt_payload",
    "prepared_record_visibility": (
        "test_service_requires_commit_before_get_and_reports_protocol_bytes"
    ),
    "restore_from_either_service": (
        "test_pair_delta_services_restore_from_either_host"
    ),
}


def parse_named_path(value: str) -> tuple[str, Path]:
    try:
        name, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected NAME=PATH") from error
    return name, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--pilot", action="append", type=parse_named_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_protocol_results(path: Path) -> list[dict[str, object]]:
    testcases = list(ET.parse(path).iter("testcase"))
    failed = {
        item.attrib.get("name", "")
        for item in testcases
        if item.find("failure") is not None or item.find("error") is not None
    }
    names = {item.attrib.get("name", "") for item in testcases}
    results = []
    for scenario, expected_prefix in PROTOCOL_CASES.items():
        matching = sorted(name for name in names if name.startswith(expected_prefix))
        if not matching:
            raise ValueError(f"JUnit report lacks {expected_prefix}")
        passed = all(name not in failed for name in matching)
        if not passed:
            raise ValueError(f"protocol fault scenario failed: {scenario}")
        results.append(
            {
                "scenario": scenario,
                "scope": "single-process TCP protocol integration",
                "test_cases": matching,
                "passed": True,
            }
        )
    return results


def load_pilot(name: str, root: Path) -> dict[str, object]:
    if not (root / "COMPLETE").exists():
        raise ValueError(f"pilot is incomplete: {root}")
    config_path = root / "experiment-config.json"
    config = json.loads(config_path.read_text())
    restore_paths = [root / f"restore-host-{host}" / "restore-summary.json" for host in (0, 1)]
    restores = [json.loads(path.read_text()) for path in restore_paths]
    for host, restore in enumerate(restores):
        if (
            int(restore["restore_host_index"]) != host
            or not restore.get("catalog_restore_smoke")
            or not restore.get("all_rank_templates_reconstructed")
            or restore.get("service_failure_sets_validated") != []
        ):
            raise ValueError(f"invalid ordinary restore: {restore_paths[host]}")
    failure_paths = sorted(root.glob("service-failure-*/restore-summary.json"))
    if len(failure_paths) != 1:
        raise ValueError(f"expected one service-failure restore: {root}")
    failure = json.loads(failure_paths[0].read_text())
    if (
        not failure.get("catalog_restore_smoke")
        or not failure.get("all_rank_templates_reconstructed")
        or len(failure.get("service_failure_sets_validated", [])) != 1
        or failure.get("runtime_failure_sets_validated") != []
    ):
        raise ValueError(f"invalid survivor restore: {failure_paths[0]}")
    conditions = []
    for summary_path in sorted(root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        if summary.get("run_kind") != "peer_delta_service_save":
            continue
        if (
            not summary.get("last_epoch_committed")
            or int(summary.get("control_commit_acknowledged_services", 0)) != 2
            or int(summary.get("last_epoch_committed_record_count_per_service", 0))
            != int(summary["world_size"]) + 1
        ):
            raise ValueError(f"invalid committed condition: {summary_path}")
        conditions.append(
            {
                "policy": summary["policy"],
                "summary": str(summary_path.resolve()),
                "summary_sha256": sha256(summary_path),
                "availability_scope": summary.get(
                    "availability_scope",
                    "host" if summary["policy"] == "host_global" else "peer",
                ),
                "availability_scope_provenance": (
                    "recorded"
                    if "availability_scope" in summary
                    else "inferred_from_policy"
                ),
            }
        )
    if not conditions:
        raise ValueError(f"pilot has no committed conditions: {root}")
    return {
        "name": name,
        "root": str(root.resolve()),
        "catalog_hash": config["source_capture"]["catalog_hash"],
        "conditions": conditions,
        "ordinary_restores": [
            {
                "service_host_index": host,
                "restore_ms": float(restore["restore_ms"]),
                "summary": str(path.resolve()),
                "summary_sha256": sha256(path),
            }
            for host, (path, restore) in enumerate(zip(restore_paths, restores))
        ],
        "service_process_failure": {
            "failed_service_sets": failure["service_failure_sets_validated"],
            "survivor_service_host_index": failure["restore_host_index"],
            "restore_ms": float(failure["restore_ms"]),
            "summary": str(failure_paths[0].resolve()),
            "summary_sha256": sha256(failure_paths[0]),
            "physical_host_failure": False,
            "continued_training": False,
        },
    }


def main() -> None:
    args = parse_args()
    command_path = args.junit.with_name("protocol-fault-command.txt")
    if not command_path.is_file():
        raise ValueError(f"missing protocol test command: {command_path}")
    protocol_results = load_protocol_results(args.junit)
    pilots = [load_pilot(name, path) for name, path in args.pilot]
    result = {
        "schema_version": 1,
        "run_kind": "peer_delta_fault_matrix_audit",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "tracked_worktree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                text=True,
            ).strip()
        ),
        "source_sha256": {
            str(path): sha256(path)
            for path in (
                Path("Merak/core/checkpoint/peer_delta_runtime.py"),
                Path("Merak/core/checkpoint/replica_service.py"),
                Path("test/core/checkpoint/test_peer_delta_runtime.py"),
                Path("test/core/checkpoint/test_replica_service.py"),
                Path("tools/audit_peer_delta_fault_matrix.py"),
            )
        },
        "protocol_test_report": str(args.junit.resolve()),
        "protocol_test_report_sha256": sha256(args.junit),
        "protocol_test_command": command_path.read_text().strip(),
        "protocol_test_command_sha256": sha256(command_path),
        "protocol_scenarios": protocol_results,
        "two_host_artifacts": pilots,
        "validated_claims": [
            "missing or corrupt input cannot produce a valid peer control path",
            "prepared data without committed control is not restorable",
            "stale epoch restore is rejected",
            "either committed service can reconstruct every rank catalog",
            "one stopped checkpoint-service process leaves a survivor catalog restore",
        ],
        "not_validated": [
            "physical compute-host failure",
            "continued-training equivalence after peer-delta restore",
            "worker death during an in-flight peer-delta save",
            "dual-service or correlated failure",
            "a second failure during restore",
            "chunk interruption, retry, or backpressure",
        ],
    }
    atomic_json(args.output, result)
    print(json.dumps({"output": str(args.output), "pilots": len(pilots)}))


if __name__ == "__main__":
    main()
