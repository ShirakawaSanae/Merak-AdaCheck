#!/usr/bin/env python3
"""Validate fixed-pair remote save, manifest commit, and peer restore."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from Merak.core.checkpoint.catalog import iter_checkpoint_objects
from Merak.core.checkpoint.grouping import (
    CheckpointGroupingConfig,
    create_checkpoint_process_groups,
    plan_checkpoint_groups,
)
from Merak.core.checkpoint.recovery import RemoteCheckpointSession


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("sequential", "strided"), default="sequential")
    parser.add_argument("--save-step", type=int, default=5)
    parser.add_argument("--continue-steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--store-mib", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def model_checksum(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for item in iter_checkpoint_objects(model.state_dict()):
        digest.update(item.path.encode("utf-8"))
        digest.update(item.checksum.encode("ascii"))
    return digest.hexdigest()


def batch_for_step(
    *, step: int, rank: int, width: int, batch_size: int, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + step * 1009 + rank * 9176)
    inputs = torch.randn(batch_size, width, generator=generator, device=device)
    targets = torch.randn(batch_size, width, generator=generator, device=device)
    return inputs, targets


def train_range(
    *,
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    start_step: int,
    stop_step: int,
    rank: int,
    width: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> list[float]:
    losses = []
    for step in range(start_step, stop_step + 1):
        inputs, targets = batch_for_step(
            step=step,
            rank=rank,
            width=width,
            batch_size=batch_size,
            seed=seed,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.detach().item())
    torch.cuda.synchronize(device)
    return losses


def capture_state(
    *,
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    iteration: int,
    rank: int,
    device: torch.device,
) -> dict:
    return {
        "checkpoint_version": 1,
        "iteration": iteration,
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device),
        "rank_local": torch.tensor([rank, iteration], dtype=torch.int64),
    }


def restore_training_state(
    *,
    state: dict,
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> None:
    model.module.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["lr_scheduler"])
    random.setstate(state["random_rng_state"])
    np.random.set_state(state["np_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state(state["cuda_rng_state"], device)


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    config = CheckpointGroupingConfig(
        ranks_per_node=world_size,
        group_size=2,
        scope="rank",
        policy=args.policy,
    )
    plan = plan_checkpoint_groups(world_size, config)
    checkpoint_group, group_ranks = create_checkpoint_process_groups(plan, backend="gloo")
    session = RemoteCheckpointSession(
        group=checkpoint_group,
        group_ranks=group_ranks,
        manifest_root=args.output_dir / "manifests",
        max_store_bytes=args.store_mib * 1024 * 1024,
    )

    model = nn.Sequential(
        nn.Linear(args.width, args.width),
        nn.GELU(),
        nn.Linear(args.width, args.width),
    ).to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

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
    checkpoint_state = capture_state(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        iteration=args.save_step,
        rank=rank,
        device=device,
    )
    commit = session.save(args.save_step, checkpoint_state)

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
    reference_checksum = model_checksum(model.module)

    session.drop_local(args.save_step)
    restore_start = time.perf_counter()
    restored = session.restore_from_peer(args.save_step)
    restore_ms = (time.perf_counter() - restore_start) * 1000
    if restored["rank_local"].tolist() != [rank, args.save_step]:
        raise RuntimeError(f"rank {rank} restored another rank's checkpoint")
    restore_training_state(
        state=restored,
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
    max_loss_error = max(
        abs(reference - recovered)
        for reference, recovered in zip(reference_losses, restored_losses)
    )
    valid = max_loss_error == 0 and reference_checksum == restored_checksum
    if not valid:
        raise RuntimeError(
            f"rank {rank} continuation mismatch: loss_error={max_loss_error}, "
            f"checksum={reference_checksum == restored_checksum}"
        )

    result = {
        "rank": rank,
        "group": list(group_ranks),
        "peer_rank": session.peer_rank,
        "manifest_id": commit.manifest.manifest_id,
        "object_count": len(commit.manifest.objects),
        "logical_bytes": commit.manifest.logical_bytes,
        "payload_bytes": commit.manifest.payload_bytes,
        "wire_bytes": commit.transport.wire_bytes,
        "serialize_ms": commit.serialize_ms,
        "transport_ms": commit.transport.duration_ms,
        "commit_ms": commit.commit_ms,
        "restore_ms": restore_ms,
        "store_bytes": commit.store_bytes,
        "max_loss_error": max_loss_error,
        "parameter_checksum_match": reference_checksum == restored_checksum,
        "continued_training_match": valid,
    }
    (args.output_dir / f"rank_{rank:05d}_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    dist.barrier()

    if rank == 0:
        rank_results = [
            json.loads((args.output_dir / f"rank_{item:05d}_result.json").read_text())
            for item in range(world_size)
        ]
        summary = {
            "run_kind": "fixed_pair_remote_checkpoint_correctness",
            "world_size": world_size,
            "policy": args.policy,
            "groups": [list(group) for group in plan.groups],
            "save_step": args.save_step,
            "continue_steps": args.continue_steps,
            "all_continued_training_match": all(
                item["continued_training_match"] for item in rank_results
            ),
            "rank_results": rank_results,
        }
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        (args.output_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n")
        print(json.dumps(summary, indent=2), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
