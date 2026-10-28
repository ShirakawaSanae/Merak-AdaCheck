#!/usr/bin/env python3
"""Aggregate repeated two-host rail-pairing trials with matched controls."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

T95_BY_DF = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
CONDITIONS = (
    "off_identity",
    "sync_identity",
    "sync_profile_aware",
    "async_identity",
    "async_profile_aware",
)


def mean_ci95(values: Iterable[float]) -> dict[str, float | int | None]:
    data = list(values)
    result: dict[str, float | int | None] = {
        "count": len(data),
        "mean": statistics.mean(data) if data else None,
        "ci95": None,
        "min": min(data) if data else None,
        "max": max(data) if data else None,
    }
    if len(data) > 1:
        critical = T95_BY_DF.get(len(data) - 1, 1.96)
        result["ci95"] = critical * statistics.stdev(data) / math.sqrt(len(data))
    return result


def load_run(summary_path: Path, idle_limit_bytes: int) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text())
    if summary.get("run_kind") != "two_host_adacheck_rank_pairing_data_plane":
        raise ValueError(f"unexpected run kind: {summary_path}")
    mode = summary["condition"]["checkpoint_mode"]
    pairing = summary["condition"]["pairing"]
    key = f"{mode}_{pairing}"
    if key not in CONDITIONS:
        raise ValueError(f"unexpected condition {key}: {summary_path}")
    counter = summary["physical_nic_counter"]
    if counter["node_records_available"] != 2:
        raise ValueError(f"incomplete NIC counters: {summary_path}")

    idle_xmit_bytes = 0
    idle_rcv_bytes = 0
    hca_xmit_bytes: dict[str, int] = {}
    for node in range(2):
        record = json.loads((summary_path.parent / f"nic-node-{node}.json").read_text())
        if record.get("status") != "available":
            raise ValueError(f"unavailable NIC record: {summary_path.parent}")
        idle_xmit_bytes += int(record["idle_delta"]["total_xmit_bytes"])
        idle_rcv_bytes += int(record["idle_delta"]["total_rcv_bytes"])
        for hca, values in record["measured_delta"]["per_hca"].items():
            hca_xmit_bytes[hca] = hca_xmit_bytes.get(hca, 0) + int(
                values["xmit_bytes"]
            )

    local_rank_commits: dict[int, list[float]] = {}
    for event_path in summary_path.parent.glob("events-rank-*.jsonl"):
        for line in event_path.read_text().splitlines():
            if not line:
                continue
            event = json.loads(line)
            local_rank_commits.setdefault(int(event["local_rank"]), []).append(
                float(event["commit_ms"])
            )
    local_rank_commit_mean_ms = {
        str(rank): statistics.mean(values)
        for rank, values in sorted(local_rank_commits.items())
    }

    commit = summary["checkpoint_commit_ms"]
    exposed = summary["checkpoint_exposed_stall_ms"]
    return {
        "path": str(summary_path.parent),
        "seed": int(summary["seed"]),
        "condition": key,
        "runtime_ms": max(float(value) for value in summary["rank_measured_wall_ms"]),
        "nic_xmit_bytes": int(counter["aggregate_xmit_bytes"]),
        "nic_rcv_bytes": int(counter["aggregate_rcv_bytes"]),
        "mlx5_0_xmit_bytes": hca_xmit_bytes.get("mlx5_0", 0),
        "mlx5_1_xmit_bytes": hca_xmit_bytes.get("mlx5_1", 0),
        "rail_imbalance_percent": (
            100.0 * abs(hca_xmit_bytes.get("mlx5_0", 0) - hca_xmit_bytes.get("mlx5_1", 0))
            / sum(hca_xmit_bytes.values())
            if sum(hca_xmit_bytes.values())
            else 0.0
        ),
        "idle_xmit_bytes": idle_xmit_bytes,
        "idle_rcv_bytes": idle_rcv_bytes,
        "idle_contaminated": (
            idle_xmit_bytes > idle_limit_bytes or idle_rcv_bytes > idle_limit_bytes
        ),
        "commit_p99_ms": None if commit["p99"] is None else float(commit["p99"]),
        "commit_mean_ms": None if commit["mean"] is None else float(commit["mean"]),
        "exposed_p99_ms": None if exposed["p99"] is None else float(exposed["p99"]),
        "local_rank_commit_mean_ms": local_rank_commit_mean_ms,
        "checkpoint_events": int(summary["checkpoint_event_count"]),
        "application_send_bytes": int(summary["application_checkpoint_send_bytes"]),
    }


def percent_reduction(baseline: float, candidate: float) -> float:
    if baseline == 0:
        raise ValueError("cannot calculate percent reduction from zero baseline")
    return 100.0 * (baseline - candidate) / baseline


def format_ci(metric: dict[str, float | int | None], scale: float = 1.0) -> str:
    mean = metric["mean"]
    if mean is None:
        return "n/a"
    rendered = f"{float(mean) / scale:.3f}"
    if metric["ci95"] is not None:
        rendered += f" +/- {float(metric['ci95']) / scale:.3f}"
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--idle-limit-bytes", type=int, default=1_048_576)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    runs = [
        load_run(path, args.idle_limit_bytes)
        for path in sorted(args.root.rglob("summary.json"))
    ]
    if not runs:
        raise SystemExit(f"no summary.json files under {args.root}")
    contaminated = [run["path"] for run in runs if run["idle_contaminated"]]
    if contaminated:
        raise SystemExit(f"background RDMA contamination exceeds threshold: {contaminated}")
    application_sizes = {
        run["application_send_bytes"]
        for run in runs
        if run["condition"] != "off_identity"
    }
    if len(application_sizes) > 1:
        raise SystemExit(f"checkpoint conditions used different payload totals: {application_sizes}")

    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for run in runs:
        condition_runs = by_seed.setdefault(run["seed"], {})
        if run["condition"] in condition_runs:
            raise ValueError(f"duplicate seed/condition: {run['seed']} {run['condition']}")
        condition_runs[run["condition"]] = run
    missing = {
        seed: sorted(set(CONDITIONS).difference(condition_runs))
        for seed, condition_runs in by_seed.items()
        if set(condition_runs) != set(CONDITIONS)
    }
    if missing and not args.allow_incomplete:
        raise SystemExit(f"incomplete experiment matrix: {missing}")

    metrics = (
        "runtime_ms",
        "nic_xmit_bytes",
        "nic_rcv_bytes",
        "mlx5_0_xmit_bytes",
        "mlx5_1_xmit_bytes",
        "rail_imbalance_percent",
        "idle_xmit_bytes",
        "idle_rcv_bytes",
        "commit_p99_ms",
        "commit_mean_ms",
        "exposed_p99_ms",
    )
    aggregate: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for condition in CONDITIONS:
        selected = [run for run in runs if run["condition"] == condition]
        aggregate[condition] = {
            metric: mean_ci95(
                float(run[metric]) for run in selected if run[metric] is not None
            )
            for metric in metrics
        }

    local_rank_commit_aggregate = {
        condition: {
            str(local_rank): mean_ci95(
                run["local_rank_commit_mean_ms"][str(local_rank)]
                for run in runs
                if run["condition"] == condition
                and str(local_rank) in run["local_rank_commit_mean_ms"]
            )
            for local_rank in range(8)
        }
        for condition in CONDITIONS
    }

    paired: list[dict[str, Any]] = []
    for seed, condition_runs in sorted(by_seed.items()):
        if set(condition_runs) != set(CONDITIONS):
            continue
        off = condition_runs["off_identity"]
        derived: dict[str, Any] = {"seed": seed}
        for condition in CONDITIONS[1:]:
            run = condition_runs[condition]
            derived[f"{condition}_runtime_overhead_ms"] = (
                run["runtime_ms"] - off["runtime_ms"]
            )
            derived[f"{condition}_nic_xmit_excess_bytes"] = (
                run["nic_xmit_bytes"] - off["nic_xmit_bytes"]
            )
            for hca in ("mlx5_0", "mlx5_1"):
                metric = f"{hca}_xmit_bytes"
                derived[f"{condition}_{hca}_xmit_excess_bytes"] = (
                    run[metric] - off[metric]
                )
        for pairing in ("identity", "profile_aware"):
            sync_overhead = derived[f"sync_{pairing}_runtime_overhead_ms"]
            async_overhead = derived[f"async_{pairing}_runtime_overhead_ms"]
            derived[f"{pairing}_overlap_efficiency"] = (
                None if sync_overhead <= 0 else 1.0 - async_overhead / sync_overhead
            )
        for mode in ("sync", "async"):
            identity = condition_runs[f"{mode}_identity"]
            profile = condition_runs[f"{mode}_profile_aware"]
            derived[f"{mode}_commit_p99_reduction_percent"] = percent_reduction(
                identity["commit_p99_ms"], profile["commit_p99_ms"]
            )
        paired.append(derived)

    derived_metrics = sorted(
        {key for record in paired for key in record if key != "seed"}
    )
    paired_aggregate = {
        metric: mean_ci95(
            float(record[metric])
            for record in paired
            if record.get(metric) is not None
        )
        for metric in derived_metrics
    }
    report = {
        "schema_version": 1,
        "root": str(args.root.resolve()),
        "run_count": len(runs),
        "seeds": sorted(by_seed),
        "idle_limit_bytes": args.idle_limit_bytes,
        "runs": runs,
        "aggregate": aggregate,
        "local_rank_commit_aggregate": local_rank_commit_aggregate,
        "paired_trials": paired,
        "paired_aggregate": paired_aggregate,
        "interpretation_limits": [
            "PMA counters are node-global and are interpreted only after matched off-control subtraction.",
            "Pairing does not change required application checkpoint bytes; it may change path balance and tails.",
            "Overlap efficiency uses max-rank wall time: 1 - (async-off)/(sync-off).",
            "The supplied 1/4 NIC cost profile encodes PIX/PXB versus SYS topology classes; it is not a bandwidth measurement.",
        ],
    }
    (args.root / "analysis.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Two-node rail-pairing analysis",
        "",
        "| condition | runtime ms | HCA0 TX MiB | HCA1 TX MiB | commit P99 ms | exposed P99 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition in CONDITIONS:
        values = aggregate[condition]
        lines.append(
            f"| {condition} | {format_ci(values['runtime_ms'])} | "
            f"{format_ci(values['mlx5_0_xmit_bytes'], 1024 * 1024)} | "
            f"{format_ci(values['mlx5_1_xmit_bytes'], 1024 * 1024)} | "
            f"{format_ci(values['commit_p99_ms'])} | {format_ci(values['exposed_p99_ms'])} |"
        )
    lines.extend(["", "Derived paired metrics use the same seed's off control.", ""])
    for metric, values in paired_aggregate.items():
        lines.append(f"- `{metric}`: {format_ci(values)}")
    lines.extend(["", *[f"- Limit: {value}" for value in report["interpretation_limits"]], ""])
    (args.root / "analysis.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
