#!/usr/bin/env python3
"""Two-host ordinary checkpoint and whole-job recovery correctness harness."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

import torch
import torch.distributed as dist

from Merak.core.checkpoint.grouping import (
    CheckpointGroupingConfig,
    plan_checkpoint_groups,
)
from Merak.core.checkpoint.recovery import ServiceRemoteCheckpointSession
from Merak.core.checkpoint.service_placement import (
    plan_replica_service_placement,
)
from remote_checkpoint_harness import (
    capture_state,
    model_checksum,
    restore_training_state,
    train_range,
)
from service_recovery_harness import build_training_objects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("save", "save-and-fail", "restore"), required=True
    )
    parser.add_argument("--service-endpoints", required=True)
    parser.add_argument("--mirror-local", action="store_true")
    parser.add_argument("--restore-host-index", type=int)
    parser.add_argument(
        "--pairing",
        choices=("identity", "cross_numa", "profile_aware"),
        default="identity",
    )
    parser.add_argument("--save-step", type=int, default=5)
    parser.add_argument("--continue-steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def save_checkpoint(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    group_ranks: tuple[int, ...],
    session: ServiceRemoteCheckpointSession,
    device: torch.device,
    model,
    optimizer,
    scheduler,
) -> None:
    train_range(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        start_step=1,
        stop_step=args.save_step,
        rank=rank,
        width=args.width,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )
    state = capture_state(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        iteration=args.save_step,
        rank=rank,
        device=device,
    )
    commit = session.save(args.save_step, state)
    reference_losses = train_range(
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
            "group": list(group_ranks),
            "manifest_id": commit.manifest.manifest_id,
            "payload_bytes": commit.manifest.payload_bytes,
            "wire_bytes": commit.transport.wire_bytes,
            "serialize_ms": commit.serialize_ms,
            "transport_ms": commit.transport.duration_ms,
            "commit_ms": commit.commit_ms,
            "reference_losses": reference_losses,
            "reference_checksum": model_checksum(model.module),
        },
    )
    dist.barrier()
    if rank == 0:
        results = [
            json.loads(
                (args.output_dir / f"stage1_rank_{item:05d}.json").read_text()
            )
            for item in range(world_size)
        ]
        atomic_json(
            args.output_dir / "save_summary.json",
            {
                "run_kind": "two_host_service_checkpoint_save",
                "world_size": world_size,
                "pairing": args.pairing,
                "mirror_local": args.mirror_local,
                "all_manifests_committed": len(results) == world_size,
                "rank_results": results,
            },
        )
        if args.phase == "save-and-fail":
            atomic_json(
                args.output_dir / "failure_injected.json",
                {
                    "failure": "SIGKILL",
                    "failed_rank": 0,
                    "checkpoint_epoch": args.save_step,
                    "all_rank_references_published": True,
                    "scope": "whole distributed job subsequently terminated",
                },
            )
            os.kill(os.getpid(), signal.SIGKILL)
    if args.phase == "save-and-fail":
        time.sleep(30)
        raise RuntimeError("distributed job remained alive after rank 0 SIGKILL")


def restore_checkpoint(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    group_ranks: tuple[int, ...],
    session: ServiceRemoteCheckpointSession,
    device: torch.device,
    model,
    optimizer,
    scheduler,
) -> None:
    restore_started = time.perf_counter()
    state = session.restore_from_peer(args.save_step)
    restore_ms = (time.perf_counter() - restore_started) * 1000
    if state["rank_local"].tolist() != [rank, args.save_step]:
        raise RuntimeError(f"rank {rank} restored another rank's checkpoint")
    restore_training_state(
        state=state,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )
    losses = train_range(
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
    reference = json.loads(
        (args.output_dir / f"stage1_rank_{rank:05d}.json").read_text()
    )
    oracle = torch.load(
        args.output_dir / "oracle" / f"rank_{rank:05d}_final_model.pt",
        map_location="cpu",
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
    loss_exact = max_loss_error == 0
    loss_allclose = max_loss_error <= 1e-6
    valid = loss_allclose and parameter_allclose
    if not valid:
        raise RuntimeError(
            f"rank {rank} recovery mismatch: loss={max_loss_error}, "
            f"parameter_error={parameter_max_abs_error}"
        )
    atomic_json(
        args.output_dir / f"stage2_rank_{rank:05d}.json",
        {
            "rank": rank,
            "group": list(group_ranks),
            "restore_ms": restore_ms,
            "restore_wire_bytes": session.last_restore_wire_bytes,
            "max_loss_error": max_loss_error,
            "continued_loss_exact": loss_exact,
            "continued_loss_allclose_atol_1e-6": loss_allclose,
            "final_parameter_bitwise_equal": parameter_exact,
            "final_parameter_allclose": parameter_allclose,
            "final_parameter_max_abs_error": parameter_max_abs_error,
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
        summary = {
            "run_kind": "two_host_service_checkpoint_restore",
            "world_size": world_size,
            "pairing": args.pairing,
            "fresh_torchrun_restore": True,
            "restore_host_index": args.restore_host_index,
            "all_checkpoint_payloads_validated": True,
            "all_continued_losses_exact": all(
                item["continued_loss_exact"] for item in results
            ),
            "all_continued_losses_allclose_atol_1e-6": all(
                item["continued_loss_allclose_atol_1e-6"] for item in results
            ),
            "all_final_parameters_bitwise_equal": all(
                item["final_parameter_bitwise_equal"] for item in results
            ),
            "all_final_parameters_allclose": all(
                item["final_parameter_allclose"] for item in results
            ),
            "all_recoveries_valid": all(item["recovery_valid"] for item in results),
            "rank_results": results,
        }
        atomic_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    dist.barrier()


def main() -> None:
    args = parse_args()
    endpoints = json.loads(args.service_endpoints)
    if not isinstance(endpoints, list):
        raise ValueError("service-endpoints must be a JSON list indexed by host")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    ranks_per_node = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    if world_size != 2 * ranks_per_node:
        raise ValueError("this harness requires exactly two physical hosts")
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    nic_costs = tuple([1.0] * (ranks_per_node // 2) + [4.0] * (ranks_per_node // 2))
    plan = plan_checkpoint_groups(
        world_size,
        CheckpointGroupingConfig(
            ranks_per_node=ranks_per_node,
            group_size=2,
            scope="host",
            policy="sequential",
            rank_pairing_policy=args.pairing,
            local_nic_costs=nic_costs,
        ),
    )
    group_ranks = plan.group_for_rank(rank)
    placement = plan_replica_service_placement(
        rank=rank,
        group_ranks=group_ranks,
        ranks_per_node=ranks_per_node,
        service_endpoints=endpoints,
        mirror_local=args.mirror_local,
        restore_host_index=args.restore_host_index,
    )
    session = ServiceRemoteCheckpointSession(
        rank=rank,
        group_ranks=group_ranks,
        service_endpoint=placement.primary_endpoint,
        mirror_service_endpoint=placement.mirror_endpoint,
        manifest_root=args.output_dir / "manifests",
        max_local_store_bytes=1,
    )
    device, model, optimizer, scheduler = build_training_objects(args, local_rank)
    if args.phase in ("save", "save-and-fail"):
        save_checkpoint(
            args,
            rank,
            world_size,
            group_ranks,
            session,
            device,
            model,
            optimizer,
            scheduler,
        )
    else:
        restore_checkpoint(
            args,
            rank,
            world_size,
            group_ranks,
            session,
            device,
            model,
            optimizer,
            scheduler,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
