#!/usr/bin/env python3
"""Single-node AdaCheck grouping experiment harness.

The current AdaCheck artifact detects redundant tensors and constructs remote
communication groups but does not invoke a remote-copy data path. This harness
uses its explicit grouping module, trains an 8-rank DDP model, and protects a
versioned rank-local reduced-state shard with a peer copy. It is deliberately
honest about scope: rank-scope placement tests process/GPU-pair behavior on one
host, not host-failure tolerance.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import P2POp
from torch.nn.parallel import DistributedDataParallel

from Merak.core.checkpoint.grouping import (
    CheckpointGroupingConfig,
    create_checkpoint_process_groups,
    plan_checkpoint_groups,
)
from Merak.core.checkpoint.rchara_checkpoint import build_comm_groups_for_remote_checkpointing


def percentile(values: Iterable[float], percentile_value: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    index = round((len(ordered) - 1) * percentile_value / 100)
    return ordered[index]


def summarize(values: Iterable[float]) -> dict[str, float | int | None]:
    data = list(values)
    return {
        "count": len(data),
        "mean": sum(data) / len(data) if data else None,
        "p50": percentile(data, 50),
        "p95": percentile(data, 95),
        "p99": percentile(data, 99),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=("sequential", "strided"), required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=2)
    parser.add_argument("--payload-mib", type=int, default=16)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def peer_for_rank(group_ranks: tuple[int, ...], rank: int) -> int:
    if len(group_ranks) != 2:
        raise ValueError(f"this replication harness requires pairs, got {group_ranks}")
    return group_ranks[1] if group_ranks[0] == rank else group_ranks[0]


def main() -> None:
    args = parse_args()
    if args.steps <= args.warmup_steps or args.checkpoint_interval < 0:
        raise ValueError("steps must exceed warmup-steps and checkpoint interval must be positive")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    config = CheckpointGroupingConfig(
        ranks_per_node=world_size,
        group_size=2,
        scope="rank",
        policy=args.policy,
    )
    plan = plan_checkpoint_groups(world_size, config)
    transport_group, group_ranks = create_checkpoint_process_groups(plan, backend="nccl")
    peer_rank = peer_for_rank(group_ranks, rank)

    # Exercise the actual AdaCheck group-construction entry point as a separate
    # control-plane check. The NCCL group above is the experiment data plane.
    os.environ["CKPT_RANKS_PER_NODE"] = str(world_size)
    os.environ["CKPT_GROUP_SIZE"] = "2"
    os.environ["CKPT_GROUPING_SCOPE"] = "rank"
    os.environ["CKPT_GROUPING_POLICY"] = args.policy
    rchara_group, rchara_ranks = build_comm_groups_for_remote_checkpointing()
    del rchara_group
    if tuple(rchara_ranks) != group_ranks:
        raise RuntimeError(f"AdaCheck group mismatch: {rchara_ranks} != {group_ranks}")

    model = nn.Sequential(
        nn.Linear(args.width, args.width), nn.GELU(), nn.Linear(args.width, args.width)
    ).to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    payload_elements = args.payload_mib * 1024 * 1024 // torch.tensor([], dtype=torch.float32).element_size()
    reduced_state = torch.empty(payload_elements, dtype=torch.float32, device=device)
    received_state = torch.empty_like(reduced_state)
    model_logical_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.module.parameters())
    train_events: list[float] = []
    end_to_end_step_events: list[float] = []
    checkpoint_events: list[dict[str, float | int | bool]] = []

    for step in range(1, args.steps + 1):
        torch.cuda.synchronize(device)
        step_start = time.perf_counter()
        inputs = torch.randn(args.batch_size, args.width, device=device)
        targets = torch.randn(args.batch_size, args.width, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize(device)
        train_ms = (time.perf_counter() - step_start) * 1000
        if step > args.warmup_steps:
            train_events.append(train_ms)

        if args.checkpoint_interval and step % args.checkpoint_interval == 0:
            # This controlled tensor represents the under-protected reduced
            # optimizer/shard state that AdaCheck must place remotely.
            reduced_state.fill_(rank * 1_000_000 + step)
            torch.cuda.synchronize(device)
            checkpoint_start = time.perf_counter()
            works = dist.batch_isend_irecv(
                [
                    P2POp(dist.isend, reduced_state, peer_rank, transport_group),
                    P2POp(dist.irecv, received_state, peer_rank, transport_group),
                ]
            )
            for work in works:
                work.wait()
            torch.cuda.synchronize(device)
            commit_ms = (time.perf_counter() - checkpoint_start) * 1000
            expected = peer_rank * 1_000_000 + step
            valid = bool(torch.all(received_state == expected).item())
            if not valid:
                raise RuntimeError(f"rank {rank} received corrupt checkpoint data at step {step}")
            if step > args.warmup_steps:
                checkpoint_events.append(
                    {
                        "rank": rank,
                        "step": step,
                        "peer_rank": peer_rank,
                        "commit_ms": commit_ms,
                        "logical_bytes": args.payload_mib * 1024 * 1024,
                        "payload_bytes": args.payload_mib * 1024 * 1024,
                        "wire_bytes_estimate": args.payload_mib * 1024 * 1024,
                        "valid": valid,
                    }
                )

        torch.cuda.synchronize(device)
        end_to_end_step_ms = (time.perf_counter() - step_start) * 1000
        if step > args.warmup_steps:
            end_to_end_step_events.append(end_to_end_step_ms)

    mean_checkpoint_ms = (
        sum(event["commit_ms"] for event in checkpoint_events) / len(checkpoint_events)
        if checkpoint_events else -1.0
    )
    max_checkpoint_ms = max((event["commit_ms"] for event in checkpoint_events), default=-1.0)
    per_rank = torch.tensor(
        [
            sum(train_events) / len(train_events),
            max(train_events),
            sum(end_to_end_step_events) / len(end_to_end_step_events),
            max(end_to_end_step_events),
            mean_checkpoint_ms,
            max_checkpoint_ms,
        ],
        device=device,
        dtype=torch.float64,
    )
    gathered = [torch.empty_like(per_rank) for _ in range(world_size)]
    dist.all_gather(gathered, per_rank)
    event_tensor = torch.tensor(
        [event["commit_ms"] for event in checkpoint_events] or [-1.0],
        device=device,
        dtype=torch.float64,
    )
    gathered_event_tensors = [torch.empty_like(event_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_event_tensors, event_tensor)

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        flat_commit_ms = [
            item.item() for tensor in gathered_event_tensors for item in tensor if item.item() >= 0
        ]
        summary = {
            "run_kind": "single_host_rank_scope_adacheck_grouping_harness",
            "host": socket.gethostname(),
            "world_size": world_size,
            "gpu_count": torch.cuda.device_count(),
            "policy": args.policy,
            "grouping_scope": "rank",
            "groups": [list(group) for group in plan.groups],
            "checkpoint_group_size": 2,
            "fault_guarantee": "process/GPU peer copy only; no host-failure guarantee on one host",
            "model_logical_state_bytes": model_logical_bytes,
            "checkpoint_enabled": bool(args.checkpoint_interval),
            "ada_reduced_logical_bytes_per_checkpoint": args.payload_mib * 1024 * 1024 if args.checkpoint_interval else 0,
            "application_payload_bytes_per_checkpoint": args.payload_mib * 1024 * 1024 if args.checkpoint_interval else 0,
            "transport_wire_bytes_estimate_per_checkpoint": args.payload_mib * 1024 * 1024 if args.checkpoint_interval else 0,
            "physical_nic_counter": "unavailable: node-global counters cannot be attributed to this job",
            "queue_depth": 0,
            "checkpoint_commit_ms": summarize(flat_commit_ms),
            "rank_mean_train_step_ms": [tensor[0].item() for tensor in gathered],
            "rank_max_train_step_ms": [tensor[1].item() for tensor in gathered],
            "rank_mean_end_to_end_step_ms": [tensor[2].item() for tensor in gathered],
            "rank_max_end_to_end_step_ms": [tensor[3].item() for tensor in gathered],
            "rank_mean_checkpoint_commit_ms": [tensor[4].item() if tensor[4].item() >= 0 else None for tensor in gathered],
            "rank_max_checkpoint_commit_ms": [tensor[5].item() if tensor[5].item() >= 0 else None for tensor in gathered],
            "checkpoint_count_per_rank": len(checkpoint_events),
            "recovery_validation": "peer copy checksum/value validation passed",
            "seed": args.seed,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "checkpoint_interval": args.checkpoint_interval,
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        with (args.output_dir / "events.jsonl").open("w") as event_file:
            for event_index, commit_ms in enumerate(flat_commit_ms):
                event_file.write(json.dumps({"event": event_index, "commit_ms": commit_ms}) + "\n")
        (args.output_dir / "command.txt").write_text(" ".join(os.sys.argv) + "\n")
        print(json.dumps(summary, indent=2), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
