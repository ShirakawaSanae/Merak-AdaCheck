#!/usr/bin/env python3
"""Replay pair-local base/delta checkpoints through two replica services."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from Merak.core.checkpoint.matched import (
    DataShardRecord,
    commit_control,
    commit_data_shard,
    reduced_catalog_hash,
)
from Merak.core.checkpoint.hierarchical_peer import build_hierarchical_study
from Merak.core.checkpoint.peer_delta import (
    FORMAL_PEER_GROUPING_POLICIES,
    PeerResourceProfile,
    build_peer_delta_matrix,
    peer_grouping_for_policy,
)
from Merak.core.checkpoint.peer_delta_runtime import (
    build_peer_delta_control_state,
    build_peer_delta_protection_plan,
    build_peer_delta_shard,
    restore_peer_delta_from_service,
    restore_peer_delta_from_services,
)
from Merak.core.checkpoint.reduced import ReducedCheckpoint
from Merak.core.checkpoint.replica_service import ReplicaServiceClient
from rail_pairing_harness import PerfqueryReader, snapshot_delta


POLICIES = (
    *FORMAL_PEER_GROUPING_POLICIES,
    "sequential_fixed",
    "host_only",
    "rank_only",
    "hierarchical_joint",
    "host_pooled_fixed",
    "host_pooled_host",
    "host_pooled_publisher",
    "host_pooled_joint",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("save", "restore"), required=True)
    parser.add_argument("--bundle-path", type=Path, required=True)
    parser.add_argument("--service-endpoints", required=True)
    parser.add_argument("--policy", choices=POLICIES, default="fixed")
    parser.add_argument("--grouping-seed", type=int, default=20260901)
    parser.add_argument("--profile-json", type=Path)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--restore-host-index", type=int, default=0)
    parser.add_argument("--failed-service-host-index", type=int)
    parser.add_argument("--excluded-service-host-index", type=int)
    parser.add_argument("--commit-receipt", type=Path)
    parser.add_argument("--ranks-per-host", type=int)
    parser.add_argument("--hcas", default="mlx5_0,mlx5_1")
    parser.add_argument("--nic-idle-probe-seconds", type=float, default=1.0)
    parser.add_argument("--max-idle-nic-bytes", type=int, default=10 * 1024 * 1024)
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
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bundle(path: Path) -> ReducedCheckpoint:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, ReducedCheckpoint):
        raise ValueError(f"not a ReducedCheckpoint: {path}")
    value.validate()
    return value


def save_epoch(args, reduced, plan, endpoints, rank, epoch):
    started = time.perf_counter()
    source_host = rank // plan.ranks_per_host
    peer_host = plan.peer_host(source_host)
    records = []
    transfer_metrics = {}
    rank_value_ids = {
        item.value_id
        for item in reduced.objects
        if any(requirement.rank == rank for requirement in item.requirements)
    }
    rank_tensors = {
        value_id: reduced.tensors[value_id] for value_id in rank_value_ids
    }
    for target_host, label in ((source_host, "local_base"), (peer_host, "remote_delta")):
        assignment = plan.assignment(rank, target_host)
        shard = build_peer_delta_shard(
            reduced,
            plan,
            source_rank=rank,
            target_host=target_host,
            tensors=rank_tensors,
        )
        record, metrics = commit_data_shard(
            ReplicaServiceClient.from_endpoint(endpoints[target_host]),
            epoch=epoch,
            shard={
                **shard,
                "publisher_rank": rank,
                "host_index": target_host,
            },
            group_ranks=tuple(sorted((rank, assignment.peer_rank))),
        )
        if record is None:
            raise RuntimeError(f"{label} did not commit")
        records.append(record)
        transfer_metrics[label] = {
            "target_host": target_host,
            "logical_bytes": assignment.logical_bytes,
            "value_count": len(assignment.value_ids),
            "protocol_request_bytes": sum(item.request_bytes for item in metrics),
            "protocol_response_bytes": sum(item.response_bytes for item in metrics),
            "protocol_wire_bytes": sum(item.wire_bytes for item in metrics),
        }

    local = {
        "rank": rank,
        "epoch": epoch,
        "records": [record.to_dict() for record in records],
        **transfer_metrics,
    }
    gathered: list[dict[str, Any] | None] = [None] * plan.world_size
    dist.all_gather_object(gathered, local)
    committed = False
    control_error = None
    control_protocol_request_bytes = 0
    control_protocol_response_bytes = 0
    remote_control_protocol_request_bytes = 0
    remote_control_protocol_response_bytes = 0
    control_service_protocol = [
        {"request_bytes": 0, "response_bytes": 0, "wire_bytes": 0}
        for _ in endpoints
    ]
    if rank == 0:
        try:
            all_records = [
                DataShardRecord.from_dict(record)
                for item in gathered
                if item is not None
                for record in item["records"]
            ]
            control = build_peer_delta_control_state(reduced, plan, all_records)
            _, metrics = commit_control(
                tuple(ReplicaServiceClient.from_endpoint(item) for item in endpoints),
                epoch=epoch,
                control_state=control,
            )
            control_protocol_request_bytes = sum(item.request_bytes for item in metrics)
            control_protocol_response_bytes = sum(
                item.response_bytes for item in metrics
            )
            remote_control_protocol_request_bytes = (
                sum(
                    metrics[host].request_bytes
                    + metrics[host + len(endpoints)].request_bytes
                    for host in range(1, len(endpoints))
                )
            )
            remote_control_protocol_response_bytes = (
                sum(
                    metrics[host].response_bytes
                    + metrics[host + len(endpoints)].response_bytes
                    for host in range(1, len(endpoints))
                )
            )
            control_service_protocol = [
                {
                    "request_bytes": metrics[host].request_bytes
                    + metrics[host + len(endpoints)].request_bytes,
                    "response_bytes": metrics[host].response_bytes
                    + metrics[host + len(endpoints)].response_bytes,
                    "wire_bytes": metrics[host].wire_bytes
                    + metrics[host + len(endpoints)].wire_bytes,
                }
                for host in range(len(endpoints))
            ]
            committed = True
        except Exception as error:
            control_error = f"{type(error).__name__}: {error}"
    decision = [
        committed,
        control_error,
        control_protocol_request_bytes,
        control_protocol_response_bytes,
        remote_control_protocol_request_bytes,
        remote_control_protocol_response_bytes,
        control_service_protocol,
    ]
    dist.broadcast_object_list(decision, src=0)
    (
        committed,
        control_error,
        control_protocol_request_bytes,
        control_protocol_response_bytes,
        remote_control_protocol_request_bytes,
        remote_control_protocol_response_bytes,
        control_service_protocol,
    ) = decision
    if not committed:
        raise RuntimeError(f"peer delta control did not commit: {control_error}")
    local.update(
        {
            "committed": True,
            "control_commit_acknowledged_services": len(endpoints),
            "control_protocol_request_bytes": (
                int(control_protocol_request_bytes) if rank == 0 else 0
            ),
            "control_protocol_response_bytes": (
                int(control_protocol_response_bytes) if rank == 0 else 0
            ),
            "remote_control_protocol_request_bytes": (
                int(remote_control_protocol_request_bytes) if rank == 0 else 0
            ),
            "remote_control_protocol_response_bytes": (
                int(remote_control_protocol_response_bytes) if rank == 0 else 0
            ),
            "control_service_protocol": (
                control_service_protocol
                if rank == 0
                else [
                    {"request_bytes": 0, "response_bytes": 0, "wire_bytes": 0}
                    for _ in endpoints
                ]
            ),
            "commit_ms": (time.perf_counter() - started) * 1000,
        }
    )
    return local, gathered


def _service_protocol_accounting(
    events, ranks_per_host, stats_before, stats_before_metrics,
    stats_after, stats_after_metrics,
):
    accounting = []
    for host in range(len(stats_before)):
        workload_request = 0
        workload_response = 0
        for event in events:
            for label in ("local_base", "remote_delta"):
                transfer = event[label]
                if int(transfer["target_host"]) == host:
                    workload_request += int(transfer["protocol_request_bytes"])
                    workload_response += int(transfer["protocol_response_bytes"])
            control = event["control_service_protocol"][host]
            workload_request += int(control["request_bytes"])
            workload_response += int(control["response_bytes"])
        observed_request = (
            int(stats_after[host]["protocol_rx_bytes"])
            - int(stats_before[host]["protocol_rx_bytes"])
        )
        observed_response = (
            int(stats_after[host]["protocol_tx_bytes"])
            - int(stats_before[host]["protocol_tx_bytes"])
        )
        # A STATS response is counted after its snapshot; its request is counted before it.
        expected_request = workload_request + stats_after_metrics[host].request_bytes
        expected_response = workload_response + stats_before_metrics[host].response_bytes
        item = {
            "service_host_index": host,
            "workload_request_bytes": workload_request,
            "workload_response_bytes": workload_response,
            "expected_counter_request_delta": expected_request,
            "observed_counter_request_delta": observed_request,
            "expected_counter_response_delta": expected_response,
            "observed_counter_response_delta": observed_response,
            "request_difference_bytes": observed_request - expected_request,
            "response_difference_bytes": observed_response - expected_response,
        }
        item["validated"] = (
            item["request_difference_bytes"] == 0
            and item["response_difference_bytes"] == 0
        )
        accounting.append(item)
    if not all(item["validated"] for item in accounting):
        raise RuntimeError(f"service protocol byte accounting failed: {accounting}")
    return accounting


def save_phase(
    args, reduced, plan, endpoints, rank,
    service_stats_before, service_stats_before_metrics,
):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank % plan.ranks_per_host)))
    node_index = rank // plan.ranks_per_host
    reader = None
    nic_record = None
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
                "idle_delta": snapshot_delta(idle_before, idle_after, hcas),
            }
            nic_record["idle_threshold_bytes_per_direction"] = (
                args.max_idle_nic_bytes
            )
            nic_record["contaminated"] = any(
                int(nic_record["idle_delta"][key]) > args.max_idle_nic_bytes
                for key in ("total_xmit_bytes", "total_rcv_bytes")
            )
        except Exception as error:
            if args.require_nic_counters:
                raise
            reader = None
            nic_record = {
                "status": "unavailable",
                "node_index": node_index,
                "error": f"{type(error).__name__}: {error}",
            }
    dist.barrier()
    if reader is not None and args.warmup_epochs == 0:
        nic_record["workload_before"] = reader.read()
    events = []
    for offset in range(args.epochs):
        dist.barrier()
        event, _ = save_epoch(
            args, reduced, plan, endpoints, rank, args.epoch + offset
        )
        events.append(event)
        if offset + 1 == args.warmup_epochs:
            dist.barrier()
            if reader is not None:
                nic_record["workload_before"] = reader.read()
    dist.barrier()
    if reader is not None:
        workload_after = reader.read()
        nic_record["workload_delta"] = snapshot_delta(
            nic_record["workload_before"], workload_after, tuple(nic_record["hcas"])
        )
    nic_records: list[dict[str, Any] | None] = [None] * plan.world_size
    dist.all_gather_object(nic_records, nic_record)
    all_rank_events: list[list[dict[str, Any]] | None] = [None] * plan.world_size
    dist.all_gather_object(all_rank_events, events)
    atomic_json(args.output_dir / f"rank-{rank:05d}.json", {"events": events})
    dist.barrier()
    if rank == 0:
        service_stats_after_pairs = [
            ReplicaServiceClient.from_endpoint(endpoint).stats_with_metrics()
            for endpoint in endpoints
        ]
        service_stats_after = [item[0] for item in service_stats_after_pairs]
        service_stats_after_metrics = [item[1] for item in service_stats_after_pairs]
        last_epoch = args.epoch + args.epochs - 1
        for host_index, stats in enumerate(service_stats_after):
            expected_sources = {
                -1,
                *(
                    assignment.source_rank
                    for assignment in plan.assignments
                    if assignment.target_host == host_index
                ),
            }
            actual_sources = {
                int(item["source_rank"])
                for item in stats["records"]
                if int(item["epoch"]) == last_epoch
                and item["state"] == "committed"
            }
            if actual_sources != expected_sources:
                raise RuntimeError(
                    f"service {host_index} has incomplete epoch {last_epoch}: "
                    f"{sorted(actual_sources)}"
                )
        flattened_events = [
            event
            for rank_events in all_rank_events
            if rank_events is not None
            for event in rank_events
        ]
        remote_delta_request = sum(
            event["remote_delta"]["protocol_request_bytes"]
            for event in flattened_events
        )
        remote_delta_response = sum(
            event["remote_delta"]["protocol_response_bytes"]
            for event in flattened_events
        )
        remote_control_request = sum(
            event["remote_control_protocol_request_bytes"]
            for event in flattened_events
        )
        remote_control_response = sum(
            event["remote_control_protocol_response_bytes"]
            for event in flattened_events
        )
        service_protocol_accounting = _service_protocol_accounting(
            flattened_events,
            plan.ranks_per_host,
            service_stats_before,
            service_stats_before_metrics,
            service_stats_after,
            service_stats_after_metrics,
        )
        summary = {
            **provenance(),
            "schema_version": 1,
            "run_kind": "peer_delta_service_save",
            "source_bundle": str(args.bundle_path.resolve()),
            "source_bundle_sha256": file_sha256(args.bundle_path),
            "policy": args.policy,
            "grouping_seed": args.grouping_seed,
            "profile": args.profile,
            "profile_hash": args.profile_hash,
            "world_size": plan.world_size,
            "ranks_per_host": plan.ranks_per_host,
            "host_count": plan.host_count,
            "host_pairs": [list(item) for item in plan.host_pairs],
            "catalog_hash": plan.catalog_hash,
            "plan_hash": plan.plan_hash,
            "grouping": plan.grouping.to_dict(),
            "availability_scope": plan.availability_scope,
            "planned_pair_local_transfer_bytes": plan.pair_local_transfer_bytes,
            "host_global_control_bytes": plan.host_global_control_bytes,
            "remote_transfer_bytes": plan.remote_transfer_bytes,
            "maximum_remote_publisher_bytes": plan.maximum_remote_publisher_bytes,
            "epochs": args.epochs,
            "warmup_epochs": args.warmup_epochs,
            "measured_epochs": args.epochs - args.warmup_epochs,
            "first_epoch": args.epoch,
            "last_epoch": last_epoch,
            "last_epoch_committed": events[-1]["committed"],
            "global_commit_semantics": (
                "driver_success_requires_ack_from_all_services; service visibility "
                "is sequential and not a distributed atomic commit"
            ),
            "control_commit_acknowledged_services": len(endpoints),
            "remote_cross_host_protocol": {
                "request_bytes": remote_delta_request + remote_control_request,
                "response_bytes": remote_delta_response + remote_control_response,
                "wire_bytes": remote_delta_request
                + remote_delta_response
                + remote_control_request
                + remote_control_response,
            },
            "service_stats_before": service_stats_before,
            "service_stats_after": service_stats_after,
            "service_protocol_accounting": service_protocol_accounting,
            "last_epoch_committed_record_count_per_service": {
                str(host): 1
                + sum(
                    assignment.target_host == host
                    for assignment in plan.assignments
                )
                for host in range(plan.host_count)
            },
            "nic_records": [item for item in nic_records if item is not None],
            "runtime_failure_sets_validated": [],
        }
        atomic_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def restore_phase(args, endpoints, rank):
    if rank == 0:
        if args.commit_receipt is None:
            raise ValueError("restore requires --commit-receipt")
        receipt = json.loads(args.commit_receipt.read_text())
        host_count = len(endpoints)
        if (
            not receipt.get("last_epoch_committed")
            or int(receipt.get("last_epoch", -1)) != args.epoch
            or int(receipt.get("control_commit_acknowledged_services", 0))
            != host_count
        ):
            raise ValueError("restore receipt does not prove all-service commit ACK")
        if not 0 <= args.restore_host_index < host_count:
            raise ValueError("restore-host-index is outside the service topology")
        if (
            args.failed_service_host_index is not None
            and not 0 <= args.failed_service_host_index < host_count
        ):
            raise ValueError("failed-service-host-index is outside the service topology")
        if (
            args.excluded_service_host_index is not None
            and not 0 <= args.excluded_service_host_index < host_count
        ):
            raise ValueError("excluded-service-host-index is outside the service topology")
        if (
            args.failed_service_host_index is not None
            and args.excluded_service_host_index is not None
        ):
            raise ValueError("failed and excluded service modes are mutually exclusive")
        unavailable_host = (
            args.failed_service_host_index
            if args.failed_service_host_index is not None
            else args.excluded_service_host_index
        )
        if unavailable_host == args.restore_host_index:
            raise ValueError("restore host cannot be the unavailable service")
        failed_service_error = None
        if args.failed_service_host_index is not None:
            if args.failed_service_host_index == args.restore_host_index:
                raise ValueError("cannot restore from the service declared failed")
            try:
                ReplicaServiceClient.from_endpoint(
                    endpoints[args.failed_service_host_index], timeout_seconds=3
                ).health()
            except Exception as error:
                failed_service_error = f"{type(error).__name__}: {error}"
            else:
                raise RuntimeError("declared failed service is still reachable")
        started = time.perf_counter()
        if host_count == 2 and args.failed_service_host_index is None:
            reduced, wire_bytes = restore_peer_delta_from_service(
                ReplicaServiceClient.from_endpoint(endpoints[args.restore_host_index]),
                epoch=args.epoch,
                host_index=args.restore_host_index,
            )
            restore_hosts = [args.restore_host_index]
        else:
            clients = {
                host: ReplicaServiceClient.from_endpoint(endpoint)
                for host, endpoint in enumerate(endpoints)
                if host != unavailable_host
            }
            reduced, wire_bytes = restore_peer_delta_from_services(
                clients,
                epoch=args.epoch,
                failed_host_indices=(
                    ()
                    if unavailable_host is None
                    else (unavailable_host,)
                ),
            )
            restore_hosts = sorted(clients)
        restored_catalog_hash = reduced_catalog_hash(reduced)
        if restored_catalog_hash != receipt.get("catalog_hash"):
            raise ValueError("restore receipt catalog does not match service control")
        for template_rank in reduced.templates:
            reduced.reconstruct_rank(int(template_rank))
        atomic_json(
            args.output_dir / "restore-summary.json",
            {
                **provenance(),
                "schema_version": 1,
                "run_kind": "peer_delta_service_restore",
                "restore_host_index": args.restore_host_index,
                "restore_service_host_indices": restore_hosts,
                "restore_epoch": args.epoch,
                "catalog_hash": restored_catalog_hash,
                "catalog_valid": True,
                "all_rank_templates_reconstructed": True,
                "restore_protocol_wire_bytes": wire_bytes,
                "restore_ms": (time.perf_counter() - started) * 1000,
                "catalog_restore_smoke": True,
                "survivor_service_restore": {
                    "service_host_index": args.restore_host_index,
                    "other_service_used_for_restore": len(restore_hosts) > 1,
                    "failed_service_probe_error": failed_service_error,
                },
                "service_failure_sets_validated": (
                    [[args.failed_service_host_index]]
                    if args.failed_service_host_index is not None
                    else []
                ),
                "artifact_exclusion_sets_validated": (
                    [[args.excluded_service_host_index]]
                    if args.excluded_service_host_index is not None
                    else []
                ),
                "runtime_failure_sets_validated": [],
                "failure_semantics": (
                    "controlled_service_process_failure_simulation"
                    if args.failed_service_host_index is not None
                    else (
                        "controlled_service_artifact_exclusion"
                        if args.excluded_service_host_index is not None
                        else "no_injected_failure"
                    )
                ),
            },
        )
    dist.barrier()


def main() -> None:
    args = parse_args()
    endpoints = json.loads(args.service_endpoints)
    if (
        not isinstance(endpoints, list)
        or len(endpoints) < 2
        or len(set(endpoints)) != len(endpoints)
    ):
        raise ValueError("service-endpoints must contain distinct host services")
    if (
        args.epochs < 1
        or not 0 <= args.warmup_epochs < args.epochs
        or args.nic_idle_probe_seconds < 0
        or args.max_idle_nic_bytes < 0
    ):
        raise ValueError(
            "epochs must leave a measured epoch and NIC probe limits must be non-negative"
        )
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    ranks_per_host = args.ranks_per_host or int(os.environ["LOCAL_WORLD_SIZE"])
    if args.phase == "save" and world_size != len(endpoints) * ranks_per_host:
        raise ValueError("world size must match service count times ranks per host")
    if args.phase == "save":
        reduced = load_bundle(args.bundle_path)
        profile = None
        profile_hash = None
        if args.profile_json is not None:
            if len(endpoints) != 2:
                raise ValueError("pair profile calibration currently supports two hosts")
            profile_document = json.loads(args.profile_json.read_text())
            profile = PeerResourceProfile.from_dict(
                profile_document.get("profile", profile_document)
            )
            profile_hash = file_sha256(args.profile_json)
        if len(endpoints) == 2 and args.policy in FORMAL_PEER_GROUPING_POLICIES:
            matrix = build_peer_delta_matrix(
                reduced,
                ranks_per_host=ranks_per_host,
                catalog_hash=reduced_catalog_hash(reduced),
                profile=profile,
            )
            grouping = peer_grouping_for_policy(
                matrix, args.policy, seed=args.grouping_seed
            )
        else:
            study = build_hierarchical_study(
                reduced, ranks_per_host=ranks_per_host
            )
            study_policy = {
                "host_pooled_host": "host_pooled_opt",
            }.get(args.policy, args.policy)
            if study_policy not in study:
                raise ValueError(
                    f"policy {args.policy} is unavailable for {len(endpoints)} hosts"
                )
            grouping = study[study_policy].grouping
        availability_scope = (
            "host"
            if args.policy.startswith("host_pooled_")
            else "peer"
        )
        plan = build_peer_delta_protection_plan(
            reduced,
            grouping,
            ranks_per_host=ranks_per_host,
            availability_scope=availability_scope,
            balance_publishers=args.policy not in {
                "host_pooled_fixed",
                "host_pooled_host",
            },
        )
        service_stats_before = None
        service_stats_before_metrics = None
        if rank == 0:
            service_stats_before_pairs = [
                ReplicaServiceClient.from_endpoint(endpoint).stats_with_metrics()
                for endpoint in endpoints
            ]
            service_stats_before = [item[0] for item in service_stats_before_pairs]
            service_stats_before_metrics = [
                item[1] for item in service_stats_before_pairs
            ]
        args.profile = profile.to_dict() if profile is not None else None
        args.profile_hash = profile_hash
        save_phase(
            args, reduced, plan, endpoints, rank,
            service_stats_before, service_stats_before_metrics,
        )
    else:
        restore_phase(args, endpoints, rank)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
