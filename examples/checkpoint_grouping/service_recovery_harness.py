#!/usr/bin/env python3
"""Validate checkpoint recovery across a real distributed-worker restart."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from Merak.core.checkpoint.grouping import CheckpointGroupingConfig, plan_checkpoint_groups
from Merak.core.checkpoint.recovery import ServiceRemoteCheckpointSession
from remote_checkpoint_harness import (
    capture_state,
    model_checksum,
    restore_training_state,
    train_range,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("save-and-fail", "restore"), required=True)
    parser.add_argument("--service-endpoint", required=True)
    parser.add_argument("--policy", choices=("sequential", "strided"), default="sequential")
    parser.add_argument("--save-step", type=int, default=5)
    parser.add_argument("--continue-steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def build_training_objects(args: argparse.Namespace, local_rank: int):
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    model = nn.Sequential(
        nn.Linear(args.width, args.width),
        nn.GELU(),
        nn.Linear(args.width, args.width),
    ).to(device)
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        static_graph=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
    return device, model, optimizer, scheduler


def save_and_fail(
    *,
    args: argparse.Namespace,
    rank: int,
    group_ranks: tuple[int, ...],
    session: ServiceRemoteCheckpointSession,
    device: torch.device,
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
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
        {name: tensor.detach().cpu().clone() for name, tensor in model.module.state_dict().items()},
        oracle_dir / f"rank_{rank:05d}_final_model.pt",
    )
    result = {
        "rank": rank,
        "group": list(group_ranks),
        "manifest_id": commit.manifest.manifest_id,
        "object_count": len(commit.manifest.objects),
        "payload_bytes": commit.manifest.payload_bytes,
        "wire_bytes": commit.transport.wire_bytes,
        "serialize_ms": commit.serialize_ms,
        "transport_ms": commit.transport.duration_ms,
        "commit_ms": commit.commit_ms,
        "reference_losses": reference_losses,
        "reference_checksum": model_checksum(model.module),
    }
    (args.output_dir / f"stage1_rank_{rank:05d}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    dist.barrier()
    if rank == 0:
        marker = {
            "failure": "SIGKILL",
            "failed_rank": 0,
            "checkpoint_epoch": args.save_step,
            "all_rank_references_published": True,
        }
        (args.output_dir / "failure_injected.json").write_text(
            json.dumps(marker, indent=2, sort_keys=True) + "\n"
        )
        os.kill(os.getpid(), signal.SIGKILL)
    time.sleep(30)
    raise RuntimeError("torchrun did not terminate surviving workers after rank 0 failure")


def restore_and_validate(
    *,
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    group_ranks: tuple[int, ...],
    session: ServiceRemoteCheckpointSession,
    device: torch.device,
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    restore_start = time.perf_counter()
    state = session.restore_from_peer(args.save_step)
    restore_ms = (time.perf_counter() - restore_start) * 1000
    if state["rank_local"].tolist() != [rank, args.save_step]:
        raise RuntimeError(f"rank {rank} restored another rank's checkpoint")
    restore_training_state(
        state=state,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )
    restored_losses = train_range(
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
    restored_checksum = model_checksum(model.module)
    reference = json.loads(
        (args.output_dir / f"stage1_rank_{rank:05d}.json").read_text()
    )
    oracle = torch.load(
        args.output_dir / "oracle" / f"rank_{rank:05d}_final_model.pt",
        map_location="cpu",
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
    max_loss_error = max(
        abs(expected - actual)
        for expected, actual in zip(reference["reference_losses"], restored_losses)
    )
    loss_exact = max_loss_error == 0
    valid = loss_exact and parameter_allclose
    if not valid:
        raise RuntimeError(
            f"rank {rank} restart mismatch: loss_error={max_loss_error}, "
            f"checksum={restored_checksum == reference['reference_checksum']}, "
            f"parameter_max_abs_error={parameter_max_abs_error}, "
            f"parameter_allclose={parameter_allclose}"
        )
    result = {
        "rank": rank,
        "group": list(group_ranks),
        "restore_ms": restore_ms,
        "restore_wire_bytes": session.last_restore_wire_bytes,
        "max_loss_error": max_loss_error,
        "checkpoint_payload_and_objects_validated": True,
        "continued_loss_exact": loss_exact,
        "final_parameter_checksum_match": (
            restored_checksum == reference["reference_checksum"]
        ),
        "final_parameter_bitwise_equal": parameter_exact,
        "final_parameter_allclose_rtol_1e-6_atol_1e-7": parameter_allclose,
        "final_parameter_max_abs_error": parameter_max_abs_error,
        "recovery_valid": valid,
    }
    (args.output_dir / f"stage2_rank_{rank:05d}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    dist.barrier()
    if rank == 0:
        rank_results = [
            json.loads(
                (args.output_dir / f"stage2_rank_{item:05d}.json").read_text()
            )
            for item in range(world_size)
        ]
        summary = {
            "run_kind": "out_of_process_replica_worker_restart",
            "world_size": world_size,
            "policy": args.policy,
            "failed_rank": 0,
            "fresh_torchrun_restore": True,
            "validation_criteria": {
                "checkpoint_payload_and_objects": "SHA-256 exact",
                "continued_loss": "bitwise exact",
                "final_parameters": "rtol=1e-6, atol=1e-7",
            },
            "all_checkpoint_payloads_and_objects_validated": all(
                item["checkpoint_payload_and_objects_validated"]
                for item in rank_results
            ),
            "all_continued_losses_exact": all(
                item["continued_loss_exact"] for item in rank_results
            ),
            "all_final_parameters_bitwise_equal": all(
                item["final_parameter_bitwise_equal"] for item in rank_results
            ),
            "all_final_parameters_allclose": all(
                item["final_parameter_allclose_rtol_1e-6_atol_1e-7"]
                for item in rank_results
            ),
            "all_recoveries_valid": all(
                item["recovery_valid"] for item in rank_results
            ),
            "rank_results": rank_results,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(summary, indent=2), flush=True)
    dist.barrier()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    plan = plan_checkpoint_groups(
        world_size,
        CheckpointGroupingConfig(
            ranks_per_node=world_size,
            group_size=2,
            scope="rank",
            policy=args.policy,
        ),
    )
    group_ranks = plan.group_for_rank(rank)
    session = ServiceRemoteCheckpointSession(
        rank=rank,
        group_ranks=group_ranks,
        service_endpoint=args.service_endpoint,
        manifest_root=args.output_dir / "manifests",
        max_local_store_bytes=1,
    )
    device, model, optimizer, scheduler = build_training_objects(args, local_rank)
    if args.phase == "save-and-fail":
        save_and_fail(
            args=args,
            rank=rank,
            group_ranks=group_ranks,
            session=session,
            device=device,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    else:
        restore_and_validate(
            args=args,
            rank=rank,
            world_size=world_size,
            group_ranks=group_ranks,
            session=session,
            device=device,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
