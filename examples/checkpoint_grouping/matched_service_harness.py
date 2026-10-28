#!/usr/bin/env python3
"""Replay one frozen detector catalog through matched two-host placement."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist

from Merak.core.checkpoint.catalog import (
    deserialize_checkpoint,
    serialize_checkpoint,
    validate_checkpoint_objects,
)
from Merak.core.checkpoint.manifest import create_prepared_manifest
from Merak.core.checkpoint.matched import (
    DataShardRecord,
    build_control_state,
    build_data_shard,
    build_two_host_protection_plan,
    commit_control,
    commit_data_shard,
    restore_reduced_from_service,
)
from Merak.core.checkpoint.opportunity import (
    asymmetric_plan_from_reduced,
    grouping_for_policy,
)
from Merak.core.checkpoint.reduced import ReducedCheckpoint
from Merak.core.checkpoint.replica_service import ReplicaServiceClient
from Merak.core.checkpoint.transport import GlooPairTransport, ReplicaEnvelope
from rail_pairing_harness import PerfqueryReader, snapshot_delta


FAULTS = ("none", "pre_commit", "post_receive", "missing", "corrupt", "stale")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("save", "restore"), required=True)
    parser.add_argument("--bundle-path", type=Path, required=True)
    parser.add_argument("--service-endpoints", required=True)
    parser.add_argument(
        "--policy", choices=("fixed", "random", "exact"), default="fixed"
    )
    parser.add_argument("--grouping-seed", type=int, default=20260831)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--fault", choices=FAULTS, default="none")
    parser.add_argument("--fault-rank", type=int, default=0)
    parser.add_argument("--restore-host-index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--ranks-per-host", type=int)
    parser.add_argument("--hcas", default="mlx5_0,mlx5_1")
    parser.add_argument("--nic-idle-probe-seconds", type=float, default=1.0)
    parser.add_argument("--require-nic-counters", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def provenance() -> dict[str, Any]:
    return {
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "tracked_worktree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                text=True,
            ).strip()
        ),
        "hostname": socket.gethostname(),
    }


def load_bundle(path: Path) -> ReducedCheckpoint:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, ReducedCheckpoint):
        raise ValueError(f"not a ReducedCheckpoint: {path}")
    value.validate()
    return value


def create_pair_group(grouping, rank: int):
    own = None
    for pair in grouping.groups:
        group = dist.new_group(ranks=list(pair), backend="gloo")
        if rank in pair:
            own = (group, pair)
    if own is None:
        raise RuntimeError(f"rank {rank} is absent from grouping")
    return own


def outbound_state(reduced, plan, rank: int, peer_rank: int) -> dict[str, Any]:
    value_ids = tuple(
        sorted(
            item.value_id
            for item in plan.transfers
            if item.source_rank == rank and item.target_rank == peer_rank
        )
    )
    return {
        "schema_version": 1,
        "catalog_hash": plan.catalog_hash,
        "plan_hash": plan.plan_hash,
        "source_rank": rank,
        "target_rank": peer_rank,
        "tensors": {value_id: reduced.tensors[value_id] for value_id in value_ids},
    }


def exchange_outbound(reduced, plan, rank: int, group, pair):
    peer_rank = pair[1] if pair[0] == rank else pair[0]
    state = outbound_state(reduced, plan, rank, peer_rank)
    serialized = serialize_checkpoint(state)
    prepared = create_prepared_manifest(
        epoch=0,
        source_rank=rank,
        source_host=f"host-{rank // plan.ranks_per_host}",
        target_rank=peer_rank,
        target_host=f"host-{peer_rank // plan.ranks_per_host}",
        group_ranks=pair,
        serialized=serialized,
        replica_count=2,
        tolerated_failures=1,
    )
    received, metrics = GlooPairTransport(group, pair).exchange(
        ReplicaEnvelope(prepared, serialized.payload)
    )
    state = deserialize_checkpoint(received.payload, received.manifest.payload_checksum)
    validate_checkpoint_objects(state, received.manifest.objects)
    if state.get("target_rank") != rank or state.get("plan_hash") != plan.plan_hash:
        raise ValueError("received transfer targets another rank or placement plan")
    return state, metrics


def corrupt_first_tensor(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {key: value for key, value in values.items()}
    if not result:
        raise ValueError("fault-rank has no received tensor to corrupt")
    key = min(result)
    tensor = result[key].clone().contiguous()
    raw = tensor.view(torch.uint8).reshape(-1)
    raw[0] ^= 1
    result[key] = tensor
    return result


def save_epoch(args, reduced, plan, endpoints, rank, group, pair, epoch):
    started = time.perf_counter()
    received, exchange = exchange_outbound(reduced, plan, rank, group, pair)
    assignment = plan.assignment_for_rank(rank)
    expected_received = assignment.received_value_ids
    received_values = received.get("tensors")
    if not isinstance(received_values, Mapping):
        raise ValueError("received transfer has no tensor mapping")
    if tuple(sorted(received_values)) != expected_received:
        raise ValueError("received transfer object set disagrees with placement")

    local_error = None
    record = None
    service_wire_bytes = 0
    should_prepare = args.fault not in ("pre_commit",)
    should_commit = args.fault != "post_receive" and not (
        args.fault == "missing" and rank == args.fault_rank
    )
    combined = {
        value_id: reduced.tensors[value_id]
        for value_id in assignment.local_value_ids
    }
    combined.update(received_values)
    if args.fault == "corrupt" and rank == args.fault_rank:
        try:
            combined = corrupt_first_tensor(received_values)
            combined.update(
                {
                    value_id: reduced.tensors[value_id]
                    for value_id in assignment.local_value_ids
                }
            )
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"
            should_prepare = False
    if should_prepare:
        try:
            shard = build_data_shard(reduced, plan, rank, combined)
            record, service_metrics = commit_data_shard(
                ReplicaServiceClient.from_endpoint(endpoints[rank // plan.ranks_per_host]),
                epoch=epoch,
                shard=shard,
                group_ranks=pair,
                commit=should_commit,
            )
            service_wire_bytes = sum(item.wire_bytes for item in service_metrics)
        except Exception as error:
            local_error = f"{type(error).__name__}: {error}"

    local = {
        "rank": rank,
        "epoch": epoch,
        "record": record.to_dict() if record is not None else None,
        "error": local_error,
        "exchange_payload_bytes": exchange.payload_bytes,
        "exchange_wire_bytes": exchange.wire_bytes,
        "exchange_ms": exchange.duration_ms,
        "service_wire_bytes": service_wire_bytes,
    }
    gathered: list[dict[str, Any] | None] = [None] * plan.world_size
    dist.all_gather_object(gathered, local)
    committed = False
    control_error = None
    control_wire_bytes = 0
    if rank == 0:
        try:
            records = [
                DataShardRecord.from_dict(item["record"])
                for item in gathered
                if item is not None and item["record"] is not None
            ]
            control = build_control_state(reduced, plan, records)
            _, metrics = commit_control(
                tuple(ReplicaServiceClient.from_endpoint(item) for item in endpoints),
                epoch=epoch,
                control_state=control,
            )
            control_wire_bytes = sum(item.wire_bytes for item in metrics)
            committed = True
        except Exception as error:
            control_error = f"{type(error).__name__}: {error}"
    decision = [committed, control_error, control_wire_bytes]
    dist.broadcast_object_list(decision, src=0)
    committed, control_error, control_wire_bytes = decision
    expected_commit = args.fault == "none"
    if bool(committed) != expected_commit:
        raise RuntimeError(
            f"fault={args.fault} committed={committed}, expected={expected_commit}: "
            f"{control_error}"
        )
    local["committed"] = bool(committed)
    local["control_error"] = control_error
    local["control_wire_bytes"] = int(control_wire_bytes)
    local["commit_ms"] = (time.perf_counter() - started) * 1000
    return local, gathered


def save_phase(args, reduced, plan, endpoints, rank, group, pair):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank % plan.ranks_per_host)))
    node_index = rank // plan.ranks_per_host
    nic_record = None
    reader = None
    if local_rank == 0:
        try:
            hcas = tuple(item.strip() for item in args.hcas.split(",") if item.strip())
            reader = PerfqueryReader(Path.cwd(), hcas)
            idle_before = reader.read()
            time.sleep(args.nic_idle_probe_seconds)
            idle_after = reader.read()
            nic_record = {
                "status": "available",
                "node_index": node_index,
                "hcas": list(hcas),
                "idle_before": idle_before,
                "idle_after": idle_after,
                "idle_delta": snapshot_delta(idle_before, idle_after, hcas),
            }
        except Exception as error:
            if args.require_nic_counters:
                raise
            nic_record = {
                "status": "unavailable",
                "node_index": node_index,
                "error": f"{type(error).__name__}: {error}",
            }
    dist.barrier()
    if reader is not None:
        nic_record["workload_before"] = reader.read()
    events = []
    final_gathered = None
    for offset in range(args.epochs):
        dist.barrier()
        event, final_gathered = save_epoch(
            args,
            reduced,
            plan,
            endpoints,
            rank,
            group,
            pair,
            args.epoch + offset,
        )
        events.append(event)
    dist.barrier()
    if reader is not None:
        workload_after = reader.read()
        nic_record["workload_after"] = workload_after
        nic_record["workload_delta"] = snapshot_delta(
            nic_record["workload_before"], workload_after, tuple(nic_record["hcas"])
        )
    nic_records: list[dict[str, Any] | None] = [None] * plan.world_size
    dist.all_gather_object(nic_records, nic_record)
    atomic_json(args.output_dir / f"rank-{rank:05d}.json", {"events": events})
    dist.barrier()
    if rank == 0:
        summary = {
            **provenance(),
            "schema_version": 1,
            "run_kind": "matched_detector_grouping_save",
            "policy": args.policy,
            "grouping_seed": args.grouping_seed,
            "fault": args.fault,
            "world_size": plan.world_size,
            "ranks_per_host": plan.ranks_per_host,
            "catalog_hash": plan.catalog_hash,
            "plan_hash": plan.plan_hash,
            "grouping": plan.grouping.to_dict(),
            "transfer_count": len(plan.transfers),
            "cross_host_logical_bytes": sum(
                item.logical_bytes for item in plan.transfers
            ),
            "epochs": args.epochs,
            "first_epoch": args.epoch,
            "last_epoch_committed": events[-1]["committed"],
            "rank_results": final_gathered,
            "nic_records": [item for item in nic_records if item is not None],
        }
        atomic_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def restore_phase(args, endpoints, rank):
    outcome = None
    if rank == 0:
        client = ReplicaServiceClient.from_endpoint(endpoints[args.restore_host_index])
        restore_epoch = args.epoch - 1 if args.fault == "stale" else args.epoch
        try:
            reduced, wire_bytes = restore_reduced_from_service(
                client, epoch=restore_epoch, host_index=args.restore_host_index
            )
            for template_rank in reduced.templates:
                reduced.reconstruct_rank(int(template_rank))
            outcome = {
                "restored": True,
                "restore_epoch": restore_epoch,
                "catalog_valid": True,
                "all_rank_templates_reconstructed": True,
                "restore_wire_bytes": wire_bytes,
                "catalog_hash": plan_catalog_hash(reduced),
                "error": None,
            }
        except Exception as error:
            outcome = {
                "restored": False,
                "restore_epoch": restore_epoch,
                "catalog_valid": False,
                "all_rank_templates_reconstructed": False,
                "restore_wire_bytes": 0,
                "error": f"{type(error).__name__}: {error}",
            }
        expected_restore = args.fault != "stale"
        if bool(outcome["restored"]) != expected_restore:
            raise RuntimeError(
                f"fault={args.fault} restored={outcome['restored']}, "
                f"expected={expected_restore}: {outcome['error']}"
            )
        atomic_json(
            args.output_dir / "restore-summary.json",
            {
                **provenance(),
                "schema_version": 1,
                "run_kind": "matched_detector_grouping_restore",
                "service_endpoint": endpoints[args.restore_host_index],
                "restore_host_index": args.restore_host_index,
                "fault": args.fault,
                **outcome,
            },
        )
    dist.barrier()


def plan_catalog_hash(reduced: ReducedCheckpoint) -> str:
    from Merak.core.checkpoint.matched import reduced_catalog_hash

    return reduced_catalog_hash(reduced)


def main() -> None:
    args = parse_args()
    endpoints = json.loads(args.service_endpoints)
    if not isinstance(endpoints, list) or len(endpoints) != 2:
        raise ValueError("service-endpoints must be a JSON list of two endpoints")
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.nic_idle_probe_seconds < 0:
        raise ValueError("nic-idle-probe-seconds cannot be negative")
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    ranks_per_host = args.ranks_per_host or int(os.environ["LOCAL_WORLD_SIZE"])
    if world_size != 2 * ranks_per_host:
        raise ValueError("matched service harness requires exactly two hosts")
    reduced = load_bundle(args.bundle_path)
    opportunity = asymmetric_plan_from_reduced(
        reduced,
        name=args.bundle_path.parent.name,
        ranks_per_host=ranks_per_host,
    )
    grouping = grouping_for_policy(
        opportunity, args.policy, seed=args.grouping_seed
    )
    plan = build_two_host_protection_plan(
        reduced, grouping, ranks_per_host=ranks_per_host
    )
    group, pair = create_pair_group(grouping, rank)
    if args.phase == "save":
        if args.fault == "stale":
            raise ValueError("stale is a restore-only fault")
        save_phase(args, reduced, plan, endpoints, rank, group, pair)
    else:
        restore_phase(args, endpoints, rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
