#!/usr/bin/env python3
"""Commit one detector-built reduced bundle and validate fresh-job recovery."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist

from Merak.core.checkpoint.catalog import (
    deserialize_checkpoint,
    validate_checkpoint_objects,
)
from Merak.core.checkpoint.recovery import ServiceRemoteCheckpointSession
from Merak.core.checkpoint.reduced import ReducedCheckpoint
from Merak.core.checkpoint.replica_service import ReplicaServiceClient
from remote_checkpoint_harness import (
    model_checksum,
    restore_training_state,
    train_range,
)
from service_recovery_harness import build_training_objects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("save", "restore"), required=True)
    parser.add_argument("--service-endpoints", required=True)
    parser.add_argument("--bundle-path", type=Path)
    parser.add_argument("--restore-host-index", type=int, default=0)
    parser.add_argument("--save-step", type=int, default=5)
    parser.add_argument("--continue-steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_reduced(path: Path) -> ReducedCheckpoint:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, ReducedCheckpoint):
        raise ValueError(f"not a ReducedCheckpoint: {path}")
    value.validate()
    return value


def validate_bundle_world(reduced: ReducedCheckpoint, world_size: int) -> None:
    expected = set(range(world_size))
    if set(reduced.templates) != expected:
        raise ValueError(
            f"bundle rank set {sorted(reduced.templates)} does not match {sorted(expected)}"
        )


def provenance() -> dict[str, object]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).splitlines()
    return {
        "git_commit": commit,
        "tracked_worktree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                text=True,
            ).strip()
        ),
        "git_status_porcelain": status,
    }


def continue_training(args, rank, device, model, optimizer, scheduler):
    return train_range(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        start_step=args.save_step + 1,
        stop_step=args.save_step + args.continue_steps,
        rank=rank,
        width=args.width,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )


def save_phase(args, endpoints, rank, world_size, device, model, optimizer, scheduler):
    if args.bundle_path is None:
        raise ValueError("--bundle-path is required for the save phase")
    reduced = load_reduced(args.bundle_path)
    validate_bundle_world(reduced, world_size)
    state = reduced.reconstruct_rank(rank)
    if state["rank_local"].tolist() != [rank, args.save_step]:
        raise ValueError(f"rank {rank} received the wrong reduced template")
    restore_training_state(
        state=state,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    if rank == 0:
        session = ServiceRemoteCheckpointSession(
            rank=0,
            group_ranks=(0, 1),
            service_endpoint=endpoints[0],
            mirror_service_endpoint=endpoints[1] if len(endpoints) > 1 else None,
            manifest_root=args.output_dir / "manifests",
            max_local_store_bytes=1,
        )
        commit = session.save(args.save_step, reduced.to_wire_state())
        bundle_commit_path = args.bundle_path.parent / "git_commit.txt"
        atomic_json(
            args.output_dir / "save_commit.json",
            {
                **provenance(),
                "source_bundle": str(args.bundle_path.resolve()),
                "source_bundle_commit": (
                    bundle_commit_path.read_text().strip()
                    if bundle_commit_path.is_file()
                    else None
                ),
                "manifest_id": commit.manifest.manifest_id,
                "manifest_replica_count": commit.manifest.replica_count,
                "manifest_tolerated_failures": commit.manifest.tolerated_failures,
                "object_count": len(commit.manifest.objects),
                "logical_bytes": commit.manifest.logical_bytes,
                "payload_bytes": commit.manifest.payload_bytes,
                "wire_bytes": commit.transport.wire_bytes,
                "serialize_ms": commit.serialize_ms,
                "transport_ms": commit.transport.duration_ms,
                "commit_ms": commit.commit_ms,
                "service_endpoints": endpoints,
                "full_source_logical_bytes": reduced.full_logical_bytes,
            },
        )
    dist.barrier()

    losses = continue_training(args, rank, device, model, optimizer, scheduler)
    oracle_dir = args.output_dir / "oracle"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.module.state_dict().items()
        },
        oracle_dir / f"rank_{rank:05d}_final_model.pt",
    )
    atomic_json(
        args.output_dir / f"stage1_rank_{rank:05d}.json",
        {
            "rank": rank,
            "reference_losses": losses,
            "reference_checksum": model_checksum(model.module),
        },
    )
    dist.barrier()
    if rank == 0:
        commit = json.loads((args.output_dir / "save_commit.json").read_text())
        commit.update(
            {
                "run_kind": "centralized_detector_reduced_service_save",
                "world_size": world_size,
                "all_rank_references_published": all(
                    (args.output_dir / f"stage1_rank_{item:05d}.json").is_file()
                    for item in range(world_size)
                ),
            }
        )
        atomic_json(args.output_dir / "save_summary.json", commit)


def restore_phase(
    args, endpoints, rank, world_size, device, model, optimizer, scheduler
):
    try:
        endpoint = endpoints[args.restore_host_index]
    except IndexError as error:
        raise ValueError("restore-host-index has no service endpoint") from error
    client = ReplicaServiceClient.from_endpoint(endpoint)
    restore_started = time.perf_counter()
    envelope, metrics = client.get(0, args.save_step)
    wire_state = deserialize_checkpoint(
        envelope.payload, envelope.manifest.payload_checksum
    )
    validate_checkpoint_objects(wire_state, envelope.manifest.objects)
    reduced = ReducedCheckpoint.from_wire_state(wire_state)
    validate_bundle_world(reduced, world_size)
    state = reduced.reconstruct_rank(rank)
    if state["rank_local"].tolist() != [rank, args.save_step]:
        raise ValueError(f"rank {rank} restored another rank's state")
    restore_training_state(
        state=state,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )
    restore_ms = (time.perf_counter() - restore_started) * 1000
    losses = continue_training(args, rank, device, model, optimizer, scheduler)

    reference = json.loads(
        (args.output_dir / f"stage1_rank_{rank:05d}.json").read_text()
    )
    oracle = torch.load(
        args.output_dir / "oracle" / f"rank_{rank:05d}_final_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    max_loss_error = max(
        abs(expected - actual)
        for expected, actual in zip(reference["reference_losses"], losses)
    )
    parameter_exact = True
    parameter_allclose = True
    parameter_max_abs_error = 0.0
    for name, actual in model.module.state_dict().items():
        expected = oracle[name]
        actual = actual.detach().cpu()
        parameter_exact = parameter_exact and torch.equal(expected, actual)
        parameter_allclose = parameter_allclose and torch.allclose(
            expected, actual, rtol=1e-6, atol=1e-7
        )
        parameter_max_abs_error = max(
            parameter_max_abs_error,
            torch.max(torch.abs(expected - actual)).item(),
        )
    valid = max_loss_error <= 1e-6 and parameter_allclose
    if not valid:
        raise RuntimeError(
            f"rank {rank} recovery mismatch: loss={max_loss_error}, "
            f"parameter={parameter_max_abs_error}"
        )
    atomic_json(
        args.output_dir / f"stage2_rank_{rank:05d}.json",
        {
            "rank": rank,
            "service_endpoint": endpoint,
            "restore_ms": restore_ms,
            "restore_wire_bytes": metrics.wire_bytes,
            "manifest_payload_bytes": envelope.manifest.payload_bytes,
            "checkpoint_payload_and_objects_validated": True,
            "max_loss_error": max_loss_error,
            "continued_loss_exact": max_loss_error == 0,
            "final_parameter_bitwise_equal": parameter_exact,
            "final_parameter_allclose": parameter_allclose,
            "final_parameter_max_abs_error": parameter_max_abs_error,
            "final_parameter_checksum_match": (
                model_checksum(model.module) == reference["reference_checksum"]
            ),
            "recovery_valid": valid,
        },
    )
    dist.barrier()
    if rank == 0:
        results = [
            json.loads(
                (args.output_dir / f"stage2_rank_{item:05d}.json").read_text()
            )
            for item in range(world_size)
        ]
        atomic_json(
            args.output_dir / "summary.json",
            {
                **provenance(),
                "run_kind": "fresh_job_detector_reduced_service_restore",
                "world_size": world_size,
                "restore_host_index": args.restore_host_index,
                "restore_service_endpoint": endpoint,
                "all_payloads_and_objects_validated": all(
                    item["checkpoint_payload_and_objects_validated"]
                    for item in results
                ),
                "all_recoveries_valid": all(
                    item["recovery_valid"] for item in results
                ),
                "all_continued_losses_exact": all(
                    item["continued_loss_exact"] for item in results
                ),
                "all_final_parameters_bitwise_equal": all(
                    item["final_parameter_bitwise_equal"] for item in results
                ),
                "all_final_parameters_allclose": all(
                    item["final_parameter_allclose"] for item in results
                ),
                "aggregate_restore_wire_bytes": sum(
                    item["restore_wire_bytes"] for item in results
                ),
                "rank_results": results,
            },
        )


def main() -> None:
    args = parse_args()
    endpoints = json.loads(args.service_endpoints)
    if not isinstance(endpoints, list) or not 1 <= len(endpoints) <= 2:
        raise ValueError("service-endpoints must contain one or two services")
    if args.continue_steps < 1:
        raise ValueError("continue-steps must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if world_size < 2:
        raise ValueError("the reduced service harness requires at least two ranks")
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    device, model, optimizer, scheduler = build_training_objects(args, local_rank)
    if args.phase == "save":
        save_phase(
            args, endpoints, rank, world_size, device, model, optimizer, scheduler
        )
    else:
        restore_phase(
            args, endpoints, rank, world_size, device, model, optimizer, scheduler
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
