"""Summarize completed MoE snapshots without contacting replica services."""

import argparse
import csv
from collections import defaultdict
import json
import re
from pathlib import Path


def sampled_resources(root):
    gpu = {}
    for node in (17, 8):
        with (root / f"node{node}-gpu.csv").open() as stream:
            rows = list(csv.reader(stream))
        gpu[str(node)] = max(float(row[2].strip().split()[0]) for row in rows)
    with (root / "host-available-KiB.csv").open() as stream:
        hosts = list(csv.reader(stream))
    return {
        "maximum_sampled_single_gpu_used_mib_by_node": gpu,
        "minimum_sampled_host_available_kib_by_node": {
            str(node): min(int(row[index]) for row in hosts)
            for index, node in enumerate((17, 8), start=1)
        },
        "scope": "sampled whole-device usage and whole-host availability; not per-job peak RSS",
    }


def restore_reads(root):
    rows = []
    pattern = r"ZERO1_TARGET_RESTORE rank=(\d+) epoch=(\d+) bundles=(\d+) data_payload_bytes=(\d+)"
    for node in (17, 8):
        for match in re.finditer(pattern, (root / f"node{node}.log").read_text()):
            rows.append(dict(zip(("rank", "epoch", "bundles", "data_payload_bytes"),
                                 map(int, match.groups()))))
    if len(rows) != 16 or {row["rank"] for row in rows} != set(range(16)):
        raise ValueError(f"incomplete or repeated target-read records: {root}")
    return sorted(rows, key=lambda row: row["rank"])


def analyze_full(root, restore_override=None):
    restore = restore_override or root.parent.parent / "moe-restore" / f"{root.name}-r"
    if restore_override is not None:
        if not (root / "PUBLICATION_COMPLETE").exists():
            raise ValueError(f"retry source was not published: {root}")
        recorded_source = Path((restore / "source-run.txt").read_text().strip())
        if (recorded_source.resolve() != root.resolve()
                and recorded_source.parts[-3:] != root.parts[-3:]):
            raise ValueError("retry restore is bound to a different source run")
        for node in (17, 8):
            for directory in (root, restore):
                if "exit_code=0" not in (directory / f"node{node}.log.status").read_text().splitlines():
                    raise ValueError(f"worker did not exit successfully: {directory}, node{node}")
    for directory in ((root, restore) if restore_override is None else (restore,)):
        if not (directory / "COMPLETE").exists() or (directory / "FAILED").exists():
            raise ValueError(f"not a successful complete chain: {directory}")
    metadata = json.loads((root / "metadata.json").read_text())
    oracle = (json.loads((root / "recovery-oracle.json").read_text())
              if restore_override is None else {
                  "ranks": 16, "step": 2, "bitwise_equal": True,
                  "comparison": "source continuation versus explicitly selected fresh restore; raw digests checked",
              })
    if oracle["ranks"] != 16 or oracle["step"] != 2 or not oracle["bitwise_equal"]:
        raise ValueError(f"full-state oracle did not pass: {root}")
    for rank in range(16):
        name = f"rank_{rank:05d}.json"
        source = json.loads((root / "digests" / name).read_text())
        target = json.loads((restore / "digests" / name).read_text())
        if (source["iteration"] != 2 or target["iteration"] != 2
                or source["state_checksum"] != target["state_checksum"]
                or source["component_checksums"] != target["component_checksums"]):
            raise ValueError(f"digest evidence differs: {name}")
    metric = json.loads((root / "trainer-output/zero1-object-metrics.jsonl").read_text())
    recovery = json.loads((restore / "trainer-output/zero1-object-metrics.jsonl").read_text())
    if (metric["epoch"] != 1 or not metric["model_aliases_applied"]
            or recovery["epoch"] != 1 or recovery["event"] != "fresh_restore"):
        raise ValueError(f"unexpected checkpoint contract: {root}")
    step_summary = {}
    loop_wall = {}
    for phase, directory, expected in (("source", root, {1, 2}), ("restore", restore, {2})):
        ranks = []
        for rank in range(16):
            path = directory / f"trainer-output/zero1-step-metrics/rank_{rank:05d}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if len(rows) != len(expected) or {row["step"] for row in rows} != expected:
                raise ValueError(f"unexpected step coverage: {path}")
            ranks.append({row["step"]: row for row in rows})
        loop_wall[phase] = max(
            sum(row["step_wall_seconds"] for row in rank.values()) for rank in ranks
        )
        step_summary[phase] = {
            str(step): {
                "step_wall_seconds_max": max(rank[step]["step_wall_seconds"] for rank in ranks),
                "checkpoint_exposed_stall_seconds_max": max(
                    rank[step].get("checkpoint_exposed_stall_seconds", 0) for rank in ranks),
            } for step in sorted(expected)
        }
    reads = restore_reads(restore)
    if any(row["epoch"] != metric["epoch"] for row in reads):
        raise ValueError(f"restore reads do not match sealed epoch: {restore}")
    return {
        "run": str(root), "restore_run": str(restore), "model": metadata["workload"],
        "explicit_restore_retry": restore_override is not None,
        "original_source_chain_failed": (root / "FAILED").exists(),
        "parallelism": metadata["parallelism"], "seed": metadata["seed"],
        "arm": metric["arm"], "bitwise_equal": True,
        "remote_role_logical_bytes": None,
        "role_breakdown_scope": "not exported in full-model timing runs",
        "remote_data_logical_bytes": metric["remote_data_logical_bytes"],
        "remote_data_payload_bytes": metric["remote_edge_payload_bytes"],
        "remote_total_payload_bytes": metric["remote_payload_bytes"],
        "remote_wire_bytes": metric["remote_wire_bytes"],
        "capture_seconds_rank0": metric["capture_seconds"],
        "publish_seconds_rank0": metric["publish_seconds"],
        "maximum_remote_put_ms": metric["maximum_remote_put_ms"],
        "maximum_remote_data_put_to_commit_ack_ms": metric["maximum_remote_data_put_to_commit_ack_ms"],
        "steps": step_summary, "recovery": recovery,
        "maximum_rank_sum_step_wall_seconds": loop_wall,
        "source_optimizer_steps": 2,
        "source_sealed_epochs": [metric["epoch"]],
        "observed_commits_per_source_step": 0.5,
        "cadence_scope": "one sealed epoch in two source steps; not sustained checkpoint frequency",
        "sampled_resources": {"source": sampled_resources(root), "restore": sampled_resources(restore)},
        "per_rank_restore_reads": reads,
        "restore_data_payload_bytes": sum(row["data_payload_bytes"] for row in reads),
        "restore_data_bundle_reads": sum(row["bundles"] for row in reads),
        "timing_scope": "single two-step capacity/admission run; one epoch1 synchronous snapshot",
        "recovery_scope": "controller timing excludes startup; not end-to-end RTO or recovery wasted time",
        "oracle_scope": oracle["comparison"],
    }


def analyze(root):
    if not (root / "source/metadata.json").exists():
        return analyze_full(root)
    if not (root / "COMPLETE").exists():
        raise ValueError(f"incomplete run: {root}")
    metadata = json.loads((root / "source/metadata.json").read_text())
    locator = json.loads((root / "locator.json").read_text())
    oracle = json.loads((root / "recovery-oracle.json").read_text())
    metrics = [json.loads(line) for line in
               (root / "source/trainer-output/zero1-object-metrics.jsonl").read_text().splitlines()]
    if len(metrics) != 1 or not oracle["bitwise_equal"]:
        raise ValueError(f"expected one recoverable smoke checkpoint: {root}")
    metric = metrics[0]
    inventory = metric["committed_inventory"]
    topology = sorted(metadata["rank_topology"], key=lambda row: row["rank"])
    hosts = list(dict.fromkeys(row["host"] for row in topology))
    rank_hosts = {row["rank"]: hosts.index(row["host"]) for row in topology}
    roles = {key: 0 for key in ("model", "optimizer", "runtime")}
    publishers = defaultdict(int)
    payload_publishers = defaultdict(int)
    for host in inventory["hosts"]:
        target = host["host"]
        for bundle in host["bundles"]:
            rank = bundle["source_rank"]
            evidence = locator["data_evidence"][target][rank]
            for key in ("manifest_id", "payload_checksum"):
                if bundle[key] != evidence[key]:
                    raise ValueError(f"inventory/locator mismatch: {root}, {key}")
            if rank_hosts[rank] == target:
                continue
            for item in bundle["fragments"]:
                role = item["role"].split(".")[0]
                roles[role] += item["logical_bytes"]
                publishers[rank] += item["logical_bytes"]
            runtime = sum(item["logical_bytes"] for item in bundle["opaque"])
            roles["runtime"] += runtime
            publishers[rank] += runtime
            payload_publishers[rank] += bundle["payload_bytes"]
    if (sum(roles.values()) != metric["remote_data_logical_bytes"]
            or sum(payload_publishers.values()) != metric["remote_edge_payload_bytes"]):
        raise ValueError(f"inventory/transfer accounting mismatch: {root}")
    return {
        "run": str(root), "model": metadata["workload"],
        "parallelism": metadata["parallelism"], "arm": metric["arm"],
        "seed": metadata["seed"], "bitwise_equal": oracle["bitwise_equal"],
        "remote_role_logical_bytes": roles,
        "remote_data_logical_bytes": sum(roles.values()),
        "remote_data_payload_bytes": sum(payload_publishers.values()),
        "remote_total_payload_bytes": metric["remote_payload_bytes"],
        "remote_wire_bytes": metric["remote_wire_bytes"],
        "maximum_remote_publisher_logical_bytes": max(publishers.values()),
        "maximum_remote_publisher_payload_bytes": max(payload_publishers.values()),
        "inventory_export_enabled": True,
        "timing_scope": "diagnostic only: publication includes committed-bundle readback",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--restore-run", type=Path, help="Explicit full-model restore retry; validate raw source/restore digests")
    args = parser.parse_args()
    if args.restore_run is not None and len(args.runs) != 1:
        parser.error("--restore-run requires exactly one source run")
    rows = ([analyze_full(args.runs[0], args.restore_run)] if args.restore_run is not None
            else [analyze(root) for root in args.runs])
    result = {"schema_version": 1, "runs": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for row in result["runs"]:
        print(Path(row["run"]).name, row["parallelism"]["mesh_order"],
              row["arm"], row["remote_role_logical_bytes"], "restore=pass")
