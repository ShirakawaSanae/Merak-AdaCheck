#!/usr/bin/env python3
"""Aggregate one Merak incremental run without changing its raw evidence."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from Merak.core.checkpoint.communication_profile import summarize_tail_ns


STEP_WALL = re.compile(
    r"step wall: (\d+) train-plus-checkpoint-poll time \(s\): ([0-9.]+)"
)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_training_steps(path: Path) -> dict:
    samples = [
        {"step": int(step), "wall_time_seconds": float(duration)}
        for step, duration in STEP_WALL.findall(path.read_text())
    ]
    if not samples:
        raise ValueError(f"training log has no step-wall samples: {path}")
    expected = list(range(samples[0]["step"], samples[-1]["step"] + 1))
    if [item["step"] for item in samples] != expected:
        raise ValueError("training step-wall samples are not contiguous")
    steady = [item["wall_time_seconds"] for item in samples[1:]]
    return {
        "definition": (
            "rank-zero train_batch plus checkpoint-poll wall time; the steady "
            "distribution excludes the first warmup step"
        ),
        "samples": samples,
        "sample_count": len(samples),
        "steady_sample_count": len(steady),
        "steady_mean_seconds": statistics.fmean(steady) if steady else None,
        "steady_p50_seconds": _percentile(steady, 0.50),
        "steady_p95_seconds": _percentile(steady, 0.95),
        "steady_p99_seconds": _percentile(steady, 0.99),
        "steady_max_seconds": max(steady) if steady else None,
        "training_window_wall_seconds": sum(
            item["wall_time_seconds"] for item in samples
        ),
    }


def phase_wall_time(root: Path) -> dict:
    phases = {}
    for phase in ("train", "restore"):
        path = root / f"{phase}-phase-timing.txt"
        if not path.exists():
            continue
        values = dict(
            line.split("=", 1)
            for line in path.read_text().splitlines()
            if "=" in line
        )
        phases[phase] = {
            "duration_seconds": int(values["duration_ns"]) / 1_000_000_000,
            "started_utc": values["started_utc"],
            "finished_utc": values["finished_utc"],
        }
    return {
        "definition": (
            "launcher phase wall time includes distributed process startup, "
            "training or restore work, checkpoint finalization, and teardown"
        ),
        "phases": phases,
        "total_recorded_phase_wall_seconds": sum(
            item["duration_seconds"] for item in phases.values()
        ),
    }


def summarize_remote_data_plane(events: list[dict]) -> dict:
    """Summarize completed cross-host transfer edges without summing rank time."""

    by_epoch = defaultdict(list)
    for event in events:
        if not str(event["path"]).startswith("remote_"):
            continue
        if "transport_ms" not in event:
            raise ValueError("remote data-plane event is missing transport_ms")
        by_epoch[int(event["epoch"])].append(event)

    epoch_maxima = []
    for epoch, edges in sorted(by_epoch.items()):
        logical_bytes = (
            sum(int(edge["logical_bytes"]) for edge in edges)
            if all("logical_bytes" in edge for edge in edges)
            else None
        )
        epoch_maxima.append(
            {
                "epoch": epoch,
                "edge_count": len(edges),
                "max_time_ms": max(float(edge["transport_ms"]) for edge in edges),
                "application_payload_bytes": sum(
                    int(edge["payload_bytes"]) for edge in edges
                ),
                "application_wire_bytes": sum(
                    int(edge["wire_bytes"]) for edge in edges
                ),
                "logical_bytes": logical_bytes,
            }
        )

    has_logical_bytes = all(
        item["logical_bytes"] is not None for item in epoch_maxima
    )
    result = {
        "definition": (
            "max_time_ms is the maximum completed cross-host data-plane edge "
            "transport_ms in an epoch, measured after serialization/publication "
            "readiness through the remote commit ACK; rank times are not summed"
        ),
        "epoch_maxima": epoch_maxima,
        "max_time_ms": max(
            (item["max_time_ms"] for item in epoch_maxima), default=None
        ),
        "mean_epoch_max_time_ms": (
            statistics.fmean(item["max_time_ms"] for item in epoch_maxima)
            if epoch_maxima
            else None
        ),
        "application_payload_bytes": sum(
            item["application_payload_bytes"] for item in epoch_maxima
        ),
        "application_wire_bytes": sum(
            item["application_wire_bytes"] for item in epoch_maxima
        ),
        "logical_bytes": (
            sum(int(item["logical_bytes"]) for item in epoch_maxima)
            if has_logical_bytes
            else None
        ),
        "nic_bytes": None,
        "nic_bytes_reason": "no per-epoch NIC counter samples are recorded",
        "training_blocking_ms": None,
        "training_blocking_reason": (
            "transport_ms is background publisher work; foreground exposed stall "
            "requires paired training-step instrumentation"
        ),
    }
    if not has_logical_bytes:
        result["logical_bytes_reason"] = (
            "one or more legacy per-edge records omit logical object bytes"
        )
    return result


def _milliseconds(summary: dict) -> dict:
    return {
        key.removesuffix("_ns") + "_ms" if key.endswith("_ns") else key: (
            value / 1_000_000 if key.endswith("_ns") and value is not None else value
        )
        for key, value in summary.items()
    }


def summarize_communication_profiles(root: Path) -> dict:
    """Aggregate standalone, caller-timestamped communication profile records."""
    by_name = defaultdict(list)
    per_epoch = defaultdict(lambda: defaultdict(list))
    paths = sorted((root / "checkpoint" / "communication-profile").glob("rank_*.jsonl"))
    for path in paths:
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record.get("schema_version") != 1 or record.get("kind") != "interval":
                raise ValueError(f"invalid communication profile record: {path}")
            duration = int(record["duration_ns"])
            name = str(record["name"])
            by_name[name].append(duration)
            fields = record.get("fields", {})
            if isinstance(fields, dict) and "epoch" in fields:
                per_epoch[name][int(fields["epoch"])].append(duration)
    return {
        "definition": (
            "standalone monotonic-clock intervals; no CUDA synchronization or "
            "distributed barrier is inserted. Per-epoch tails are maxima across ranks"
        ),
        "rank_file_count": len(paths),
        "intervals": {
            name: {
                "all_rank_samples": _milliseconds(summarize_tail_ns(values)),
                "per_epoch_rank_maxima": _milliseconds(
                    summarize_tail_ns(
                        [max(samples) for samples in per_epoch[name].values()]
                    )
                ),
            }
            for name, values in sorted(by_name.items())
        },
    }


def summarize_training_checkpoint_rate(
    lag_by_step: list[dict], *, base_epoch: int = 0
) -> dict:
    """Measure globally visible checkpoint progress before the final drain."""

    previous_epoch = int(base_epoch)
    stalled_steps = 0
    longest_stall = 0
    for item in lag_by_step:
        current_epoch = int(base_epoch) + int(item["step"])
        visible_epoch = max(
            int(base_epoch), current_epoch - int(item["max_global_lag_steps"])
        )
        if visible_epoch < previous_epoch:
            raise ValueError("globally visible checkpoint epoch moved backwards")
        newly_visible = visible_epoch - previous_epoch
        item["globally_visible_epoch"] = visible_epoch
        item["newly_visible_checkpoint_count"] = newly_visible
        if newly_visible:
            stalled_steps = 0
        else:
            stalled_steps += 1
            longest_stall = max(longest_stall, stalled_steps)
        previous_epoch = visible_epoch

    observed_steps = len(lag_by_step)
    return {
        "definition": (
            "training-window optimizer steps divided by checkpoints that became "
            "globally visible at a polled step boundary; excludes final drain and "
            "is an observed completion rate, not a configured save interval"
        ),
        "optimizer_steps_observed": observed_steps,
        "base_epoch": int(base_epoch),
        "globally_visible_checkpoints_during_training": previous_epoch - base_epoch,
        "optimizer_steps_per_visible_checkpoint": (
            observed_steps / (previous_epoch - base_epoch)
            if previous_epoch > base_epoch
            else None
        ),
        "longest_steps_without_global_progress": longest_stall,
    }


def summarize_events(events: list[dict]) -> dict:
    fields = (
        "capture_ms",
        "queue_ms",
        "publisher_queue_ms",
        "snapshot_ready_wait_ms",
        "serialize_ms",
        "shared_slot_wait_ms",
        "shared_copy_ms",
        "transport_ms",
        "commit_ms",
        "enqueue_to_commit_ms",
        "put_service_ms",
        "commit_service_ms",
    )
    result = {
        "count": len(events),
        "payload_bytes": sum(int(event["payload_bytes"]) for event in events),
        "wire_bytes": sum(int(event["wire_bytes"]) for event in events),
        "cumulative_shared_handoff_bytes": sum(
            int(event.get("shared_memory_bytes", 0)) for event in events
        ),
        "transport_kinds": sorted(
            {event.get("transport_kind", "byte_stream") for event in events}
        ),
    }
    if all("logical_bytes" in event for event in events):
        result["logical_bytes"] = sum(
            int(event["logical_bytes"]) for event in events
        )
    for field in fields:
        if field == "enqueue_to_commit_ms" and all(
            field not in event and "queue_ms" in event and "commit_ms" in event
            for event in events
        ):
            values = [
                float(event["queue_ms"]) + float(event["commit_ms"])
                for event in events
            ]
        else:
            values = [float(event[field]) for event in events if field in event]
        if len(values) != len(events):
            continue
        result[f"mean_{field}"] = statistics.fmean(values)
        result[f"max_{field}"] = max(values)
    return result


def manifest_logical_bytes(root: Path, subdir: str, epoch: int) -> dict[str, int]:
    per_host = defaultdict(int)
    pattern = f"rank_*/epoch_{epoch:08d}.manifest.json"
    for path in (root / "checkpoint" / "manifests" / subdir).glob(pattern):
        manifest = json.loads(path.read_text())
        host = manifest["source_host"]
        per_host[host] += int(manifest["logical_bytes"])
    return dict(sorted(per_host.items()))


def manifest_update_role_bytes(
    root: Path, subdir: str, *, base_epoch: int, final_epoch: int
) -> dict:
    by_role = defaultdict(int)
    by_epoch = defaultdict(lambda: defaultdict(int))
    pattern = "rank_*/epoch_*.manifest.json"
    manifest_count = 0
    for path in (root / "checkpoint" / "manifests" / subdir).glob(pattern):
        manifest = json.loads(path.read_text())
        epoch = int(manifest["epoch"])
        if not base_epoch < epoch <= final_epoch:
            continue
        manifest_count += 1
        for item in manifest.get("objects", ()):
            role = str(item["role"])
            logical_bytes = int(item["logical_bytes"])
            by_role[role] += logical_bytes
            by_epoch[epoch][role] += logical_bytes
    return {
        "definition": (
            "logical tensor bytes in committed update manifests, summed across all "
            "ranks; gradients are recovery-operation records for paper p/o cases, "
            "not parameter or optimizer snapshot tensors"
        ),
        "manifest_count": manifest_count,
        "total_by_role": dict(sorted(by_role.items())),
        "per_epoch": {
            str(epoch): dict(sorted(values.items()))
            for epoch, values in sorted(by_epoch.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = [
        json.loads(path.read_text())
        for path in sorted((args.run / "checkpoint" / "metrics").glob("rank_*.json"))
    ]
    training_steps = summarize_training_steps(
        args.run / "logs" / "train" / "node-0.log"
    )
    if not metrics:
        result = {
            "schema_version": 3,
            "run": str(args.run),
            "checkpoint_mode": "off",
            "rank_count": 0,
            "training_step_time": training_steps,
            "phase_wall_time": phase_wall_time(args.run),
            "driver_failed": (args.run / "FAILED").exists(),
        }
        destination = args.output or args.run / "analysis.json"
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(destination)
        return
    final_epoch = max(int(item["final_epoch"]) for item in metrics)
    events = defaultdict(list)
    for item in metrics:
        for event in item["commits"]:
            events[event["path"]].append(event)
    commit_scopes = {item.get("commit_scope", "rank_ack_only") for item in metrics}
    if len(commit_scopes) != 1:
        raise ValueError("rank metrics disagree on commit scope")
    lag_limits = {int(item.get("configured_max_lag_steps", 1)) for item in metrics}
    if len(lag_limits) != 1:
        raise ValueError("rank metrics disagree on configured maximum lag")
    base_epochs = {int(item.get("base_epoch", 0)) for item in metrics}
    if len(base_epochs) != 1:
        raise ValueError("rank metrics disagree on base epoch")
    base_epoch = base_epochs.pop()
    modes = {str(item["mode"]) for item in metrics}
    if len(modes) != 1:
        raise ValueError("rank metrics disagree on checkpoint mode")
    checkpoint_mode = modes.pop()
    full_manifest_subdir = (
        "full-remote" if checkpoint_mode == "full_remote" else "local-full"
    )
    global_epochs = sorted(
        int(path.stem.removeprefix("epoch_"))
        for path in (args.run / "checkpoint" / "global").glob("epoch_*.json")
    )
    observation_count = max(
        (len(item.get("boundary_lag_steps", ())) for item in metrics), default=0
    )
    lag_by_step = []
    for index in range(observation_count):
        local = [
            int(item["boundary_lag_steps"][index])
            for item in metrics
            if index < len(item.get("boundary_lag_steps", ()))
        ]
        global_lag = [
            int(item["global_boundary_lag_steps"][index])
            for item in metrics
            if index < len(item.get("global_boundary_lag_steps", ()))
        ]
        lag_by_step.append(
            {
                "step": index + 1,
                "max_rank_lag_steps": max(local),
                "max_global_lag_steps": max(global_lag),
            }
        )
    checkpoint_rate = summarize_training_checkpoint_rate(
        lag_by_step, base_epoch=base_epoch
    )
    result = {
        "schema_version": 3,
        "run": str(args.run),
        "checkpoint_mode": checkpoint_mode,
        "rank_count": len(metrics),
        "final_epoch": final_epoch,
        "all_rank_epochs_contiguous": all(item["all_epochs_contiguous"] for item in metrics),
        "all_ranks_valid": all(item["valid"] for item in metrics),
        "max_boundary_lag_steps": max(int(item["max_boundary_lag_steps"]) for item in metrics),
        "configured_max_lag_steps": lag_limits.pop(),
        "commit_scope": commit_scopes.pop(),
        "global_control_epochs": global_epochs,
        "global_committed_epoch": min(
            (int(item.get("global_committed_epoch", final_epoch)) for item in metrics),
            default=None,
        ),
        "max_global_boundary_lag_steps": max(
            (int(item.get("max_global_boundary_lag_steps", 0)) for item in metrics),
            default=None,
        ),
        "boundary_lag_by_step": lag_by_step,
        "training_checkpoint_rate": checkpoint_rate,
        "training_step_time": training_steps,
        "phase_wall_time": phase_wall_time(args.run),
        "path_summary": {
            path: summarize_events(items) for path, items in sorted(events.items())
        },
        "remote_data_plane": summarize_remote_data_plane(
            [event for items in events.values() for event in items]
        ),
        "communication_profile": summarize_communication_profiles(args.run),
        "base_logical_bytes_per_host": manifest_logical_bytes(
            args.run, full_manifest_subdir, 0
        ),
        "final_update_logical_bytes_per_host": manifest_logical_bytes(
            args.run, full_manifest_subdir, final_epoch
        ),
        "update_logical_bytes_by_role": manifest_update_role_bytes(
            args.run,
            full_manifest_subdir,
            base_epoch=base_epoch,
            final_epoch=final_epoch,
        ),
        "full_state_manifest_path": full_manifest_subdir,
        "driver_failed": (args.run / "FAILED").exists(),
    }
    destination = args.output or args.run / "analysis.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
