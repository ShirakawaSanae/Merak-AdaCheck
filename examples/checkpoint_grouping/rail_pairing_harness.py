#!/usr/bin/env python3
"""Two-host checkpoint rank-pairing, NIC-byte, tail, and overlap harness.

This is a controlled data-plane experiment.  It exercises AdaCheck's actual
group-construction entry point, transfers versioned GPU-resident checkpoint
shards between failure domains, validates them before a pair-wide commit ACK,
and measures node-global InfiniBand PMA counters around the workload.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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
from Merak.core.checkpoint.telemetry import IbPortCounters, ib_counter_delta, parse_perfquery_extended
from Merak.core.checkpoint.topology import plan_local_rank_pairing


def percentile(values: Iterable[float], percentage: float) -> float | None:
    data = sorted(values)
    if not data:
        return None
    if percentage < 0 or percentage > 100:
        raise ValueError("percentage must be in [0, 100]")
    index = max(0, math.ceil(percentage / 100 * len(data)) - 1)
    return data[index]


def summarize(values: Iterable[float]) -> dict[str, float | int | None]:
    data = list(values)
    return {
        "count": len(data),
        "mean": sum(data) / len(data) if data else None,
        "p50": percentile(data, 50),
        "p95": percentile(data, 95),
        "p99": percentile(data, 99),
        "max": max(data) if data else None,
    }


def parse_csv_numbers(text: str, converter: type[int] | type[float]) -> tuple[Any, ...]:
    values = tuple(converter(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise argparse.ArgumentTypeError("the comma-separated list cannot be empty")
    return values


def atomic_write_text(path: Path, contents: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(contents)
    temporary.replace(path)


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def peer_for_rank(group_ranks: tuple[int, ...], rank: int) -> int:
    if len(group_ranks) != 2:
        raise ValueError(f"this harness requires rank pairs, got {group_ranks}")
    return group_ranks[1] if group_ranks[0] == rank else group_ranks[0]


class PerfqueryReader:
    """Read local 64-bit InfiniBand PMA counters without resetting them."""

    def __init__(self, repo_root: Path, hcas: tuple[str, ...]) -> None:
        project_binary = repo_root / "runs/deps/ibdiags/rootfs/usr/sbin/perfquery"
        self.binary = project_binary if project_binary.is_file() else Path("/usr/sbin/perfquery")
        self.hcas = hcas
        project_libraries = (
            repo_root / "runs/deps/ibdiags/rootfs/usr/lib/x86_64-linux-gnu",
            repo_root / "runs/deps/libibmad5/rootfs/usr/lib/x86_64-linux-gnu",
        )
        library_parts = [str(path) for path in project_libraries if path.is_dir()]
        if os.environ.get("LD_LIBRARY_PATH"):
            library_parts.append(os.environ["LD_LIBRARY_PATH"])
        self.environment = os.environ.copy()
        self.environment["LD_LIBRARY_PATH"] = ":".join(library_parts)

    def read(self) -> dict[str, object]:
        if not self.binary.is_file():
            raise FileNotFoundError(
                "perfquery is unavailable; provision it under runs/deps/ibdiags or install it"
            )
        result: dict[str, object] = {
            "started_wall_time_ns": time.time_ns(),
            "hcas": {},
        }
        hca_values: dict[str, object] = result["hcas"]  # type: ignore[assignment]
        for hca in self.hcas:
            started_ns = time.time_ns()
            completed = subprocess.run(
                [str(self.binary), "-x", "-C", hca, "-P", "1"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
                env=self.environment,
            )
            counters = parse_perfquery_extended(completed.stdout)
            hca_values[hca] = {
                "sample_started_wall_time_ns": started_ns,
                "sample_finished_wall_time_ns": time.time_ns(),
                "counters": counters.as_dict(),
            }
        result["finished_wall_time_ns"] = time.time_ns()
        return result


def counters_from_snapshot(snapshot: dict[str, object], hca: str) -> IbPortCounters:
    hcas = snapshot["hcas"]
    if not isinstance(hcas, dict) or hca not in hcas:
        raise ValueError(f"snapshot has no counters for {hca}")
    record = hcas[hca]
    if not isinstance(record, dict) or not isinstance(record.get("counters"), dict):
        raise ValueError(f"snapshot counters for {hca} are malformed")
    counters = record["counters"]
    return IbPortCounters(**counters)


def snapshot_delta(
    before: dict[str, object], after: dict[str, object], hcas: tuple[str, ...]
) -> dict[str, object]:
    per_hca = {
        hca: ib_counter_delta(
            counters_from_snapshot(before, hca), counters_from_snapshot(after, hca)
        ).as_dict()
        for hca in hcas
    }
    return {
        "per_hca": per_hca,
        "total_xmit_bytes": sum(value["xmit_bytes"] for value in per_hca.values()),
        "total_rcv_bytes": sum(value["rcv_bytes"] for value in per_hca.values()),
        "total_xmit_packets": sum(value["xmit_packets"] for value in per_hca.values()),
        "total_rcv_packets": sum(value["rcv_packets"] for value in per_hca.values()),
    }


@dataclass
class CheckpointBuffers:
    send: torch.Tensor
    receive: torch.Tensor


@dataclass
class PendingCheckpoint:
    step: int
    epoch: int
    buffers: CheckpointBuffers
    works: list[Any]
    launch_started: float
    enqueue_finished: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairing",
        choices=("identity", "cross_numa", "profile_aware"),
        required=True,
    )
    parser.add_argument("--checkpoint-mode", choices=("off", "sync", "async"), required=True)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=2)
    parser.add_argument("--max-inflight", type=int, default=1)
    parser.add_argument("--checkpoint-warmup-events", type=int, default=1)
    parser.add_argument(
        "--payload-profile-mib",
        default="8,8,8,8,32,32,32,32",
        help="checkpoint payload MiB by local rank; shared by both hosts",
    )
    parser.add_argument(
        "--nic-cost-profile",
        default="1,1,1,1,4,4,4,4",
        help="measured relative GPU-to-NIC path cost by local rank",
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--hcas", default="mlx5_0,mlx5_1")
    parser.add_argument("--nic-idle-probe-seconds", type=float, default=1.0)
    parser.add_argument("--require-nic-counters", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def train_step(
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    width: int,
    device: torch.device,
) -> tuple[float, float]:
    wall_started = time.perf_counter()
    gpu_started = torch.cuda.Event(enable_timing=True)
    gpu_finished = torch.cuda.Event(enable_timing=True)
    gpu_started.record()
    inputs = torch.randn(batch_size, width, device=device)
    targets = torch.randn(batch_size, width, device=device)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()
    gpu_finished.record()
    gpu_finished.synchronize()
    return gpu_started.elapsed_time(gpu_finished), (time.perf_counter() - wall_started) * 1000


def main() -> None:
    args = parse_args()
    payload_mib = tuple(int(value) for value in parse_csv_numbers(args.payload_profile_mib, int))
    nic_costs = tuple(float(value) for value in parse_csv_numbers(args.nic_cost_profile, float))
    hcas = tuple(value.strip() for value in args.hcas.split(",") if value.strip())
    if args.steps <= args.warmup_steps:
        raise ValueError("steps must exceed warmup-steps")
    if args.checkpoint_interval < 1:
        raise ValueError("checkpoint-interval must be positive")
    if args.max_inflight < 1:
        raise ValueError("max-inflight must be positive")
    if args.checkpoint_warmup_events < 0:
        raise ValueError("checkpoint-warmup-events cannot be negative")
    if args.nic_idle_probe_seconds < 0:
        raise ValueError("nic-idle-probe-seconds cannot be negative")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    ranks_per_node = int(os.environ["LOCAL_WORLD_SIZE"])
    node_rank = rank // ranks_per_node
    if world_size != 2 * ranks_per_node:
        raise ValueError(
            f"this harness requires exactly two hosts, got world_size={world_size}, "
            f"ranks_per_node={ranks_per_node}"
        )
    if len(payload_mib) != ranks_per_node or len(nic_costs) != ranks_per_node:
        raise ValueError("payload and NIC-cost profiles must match LOCAL_WORLD_SIZE")
    if any(value <= 0 for value in payload_mib):
        raise ValueError("payload MiB values must be positive")
    if any(value < 0 for value in nic_costs):
        raise ValueError("NIC costs cannot be negative")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    payload_bytes = tuple(value * 1024 * 1024 for value in payload_mib)
    local_pairing = plan_local_rank_pairing(
        ranks_per_node,
        args.pairing,
        first_payload_bytes=payload_bytes,
        second_payload_bytes=payload_bytes,
        first_nic_costs=nic_costs,
        second_nic_costs=nic_costs,
    )
    config = CheckpointGroupingConfig(
        ranks_per_node=ranks_per_node,
        group_size=2,
        scope="host",
        policy="sequential",
        rank_pairing_policy=args.pairing,
        local_payload_bytes=payload_bytes,
        local_nic_costs=nic_costs,
    )
    plan = plan_checkpoint_groups(world_size, config)
    transport_group, group_ranks = create_checkpoint_process_groups(plan, backend="nccl")
    peer_rank = peer_for_rank(group_ranks, rank)
    peer_local_rank = peer_rank % ranks_per_node

    # Verify that the actual AdaCheck entry point consumes the same plan.
    os.environ["CKPT_RANKS_PER_NODE"] = str(ranks_per_node)
    os.environ["CKPT_GROUP_SIZE"] = "2"
    os.environ["CKPT_GROUPING_SCOPE"] = "host"
    os.environ["CKPT_GROUPING_POLICY"] = "sequential"
    os.environ["CKPT_RANK_PAIRING_POLICY"] = args.pairing
    os.environ["CKPT_LOCAL_PAYLOAD_BYTES"] = ",".join(str(value) for value in payload_bytes)
    os.environ["CKPT_LOCAL_NIC_COSTS"] = ",".join(str(value) for value in nic_costs)
    rchara_group, rchara_ranks = build_comm_groups_for_remote_checkpointing()
    del rchara_group
    if tuple(rchara_ranks) != group_ranks:
        raise RuntimeError(f"AdaCheck group mismatch: {rchara_ranks} != {group_ranks}")

    model = nn.Sequential(
        nn.Linear(args.width, args.width),
        nn.GELU(),
        nn.Linear(args.width, args.width),
    ).to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model_logical_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.module.parameters()
    )

    float_bytes = torch.empty((), dtype=torch.float32).element_size()
    local_elements = payload_bytes[local_rank] // float_bytes
    peer_elements = payload_bytes[peer_local_rank] // float_bytes
    available_buffers = [
        CheckpointBuffers(
            send=torch.empty(local_elements, dtype=torch.float32, device=device),
            receive=torch.empty(peer_elements, dtype=torch.float32, device=device),
        )
        for _ in range(args.max_inflight)
    ]
    pending: list[PendingCheckpoint] = []
    checkpoint_events: list[dict[str, object]] = []
    checkpoint_stream = torch.cuda.Stream(device=device)

    def drain_oldest() -> None:
        checkpoint = pending.pop(0)
        drain_started = time.perf_counter()
        for work in checkpoint.works:
            work.wait()
        expected = float(peer_rank * 1_000_000 + checkpoint.epoch)
        valid = bool(torch.all(checkpoint.buffers.receive == expected).item())
        ack = torch.tensor(1 if valid else 0, dtype=torch.int32, device=device)
        dist.all_reduce(ack, op=dist.ReduceOp.MIN, group=transport_group)
        globally_valid = bool(ack.item())
        committed = time.perf_counter()
        if not globally_valid:
            raise RuntimeError(
                f"pair {group_ranks} rejected corrupt checkpoint epoch {checkpoint.epoch}"
            )
        commit_ms = (committed - checkpoint.launch_started) * 1000
        exposed_stall_ms = (committed - drain_started) * 1000
        nonblocking_window_ms = max(
            0.0, (drain_started - checkpoint.enqueue_finished) * 1000
        )
        checkpoint_events.append(
            {
                "rank": rank,
                "local_rank": local_rank,
                "peer_rank": peer_rank,
                "peer_local_rank": peer_local_rank,
                "step": checkpoint.step,
                "epoch": checkpoint.epoch,
                "send_payload_bytes": payload_bytes[local_rank],
                "receive_payload_bytes": payload_bytes[peer_local_rank],
                "enqueue_ms": (checkpoint.enqueue_finished - checkpoint.launch_started) * 1000,
                "commit_ms": commit_ms,
                "exposed_stall_ms": exposed_stall_ms,
                "nonblocking_window_ms": nonblocking_window_ms,
                "nonblocking_fraction": (
                    min(1.0, nonblocking_window_ms / commit_ms) if commit_ms else 0.0
                ),
                "valid": globally_valid,
            }
        )
        available_buffers.append(checkpoint.buffers)

    def launch_checkpoint(step: int, epoch: int) -> None:
        if args.checkpoint_mode == "async" and len(pending) >= args.max_inflight:
            drain_oldest()
        if not available_buffers:
            raise RuntimeError("checkpoint buffer queue has no available slot")
        buffers = available_buffers.pop()
        launch_started = time.perf_counter()
        with torch.cuda.stream(checkpoint_stream):
            buffers.send.fill_(float(rank * 1_000_000 + epoch))
            works = dist.batch_isend_irecv(
                [
                    P2POp(dist.isend, buffers.send, peer_rank, transport_group),
                    P2POp(dist.irecv, buffers.receive, peer_rank, transport_group),
                ]
            )
        pending.append(
            PendingCheckpoint(
                step=step,
                epoch=epoch,
                buffers=buffers,
                works=works,
                launch_started=launch_started,
                enqueue_finished=time.perf_counter(),
            )
        )
        if args.checkpoint_mode == "sync":
            drain_oldest()

    for _ in range(args.warmup_steps):
        train_step(model, optimizer, args.batch_size, args.width, device)
    if args.checkpoint_mode != "off":
        for warmup_epoch in range(1, args.checkpoint_warmup_events + 1):
            launch_checkpoint(args.warmup_steps, warmup_epoch)
        while pending:
            drain_oldest()
        # Communicator setup and allocator warm-up are not experimental samples.
        checkpoint_events.clear()
    torch.cuda.synchronize(device)
    dist.barrier()

    repo_root = Path(__file__).resolve().parents[2]
    nic_record: dict[str, object] | None = None
    counter_before: dict[str, object] | None = None
    counter_error: str | None = None
    if local_rank == 0:
        try:
            reader = PerfqueryReader(repo_root, hcas)
            idle_started = reader.read()
            time.sleep(args.nic_idle_probe_seconds)
            counter_before = reader.read()
            nic_record = {
                "status": "available",
                "source": "InfiniBand PMA perfquery -x 64-bit port counters",
                "scope": "node-global; attribution requires no concurrent RDMA traffic",
                "data_counter_bytes_per_unit": 4,
                "idle_probe_seconds": args.nic_idle_probe_seconds,
                "idle_before": idle_started,
                "before": counter_before,
                "idle_delta": snapshot_delta(idle_started, counter_before, hcas),
            }
        except Exception as error:  # preserve the run and report the telemetry limitation
            counter_error = f"{type(error).__name__}: {error}"
            nic_record = {"status": "unavailable", "error": counter_error}

    counter_ok = torch.tensor(
        0 if local_rank == 0 and counter_error else 1,
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(counter_ok, op=dist.ReduceOp.MIN)
    if args.require_nic_counters and not bool(counter_ok.item()):
        raise RuntimeError("at least one host cannot read required InfiniBand PMA counters")
    dist.barrier()

    train_gpu_ms: list[float] = []
    train_wall_ms: list[float] = []
    end_to_end_step_ms: list[float] = []
    measured_started = time.perf_counter()
    checkpoint_epoch = 0
    for step in range(args.warmup_steps + 1, args.steps + 1):
        step_started = time.perf_counter()
        gpu_ms, wall_ms = train_step(model, optimizer, args.batch_size, args.width, device)
        train_gpu_ms.append(gpu_ms)
        train_wall_ms.append(wall_ms)
        if args.checkpoint_mode != "off" and step % args.checkpoint_interval == 0:
            checkpoint_epoch += 1
            launch_checkpoint(step, checkpoint_epoch)
        end_to_end_step_ms.append((time.perf_counter() - step_started) * 1000)

    while pending:
        drain_oldest()
    torch.cuda.synchronize(device)
    measured_wall_ms = (time.perf_counter() - measured_started) * 1000
    dist.barrier()

    if local_rank == 0 and nic_record is not None and counter_before is not None:
        try:
            counter_after = reader.read()
            nic_record["after"] = counter_after
            nic_record["measured_delta"] = snapshot_delta(counter_before, counter_after, hcas)
        except Exception as error:
            nic_record = {
                "status": "unavailable",
                "error": f"final counter read failed: {type(error).__name__}: {error}",
            }

    final_counter_ok = torch.tensor(
        0
        if local_rank == 0
        and (nic_record is None or nic_record.get("status") != "available")
        else 1,
        dtype=torch.int32,
        device=device,
    )
    dist.all_reduce(final_counter_ok, op=dist.ReduceOp.MIN)
    if args.require_nic_counters and not bool(final_counter_ok.item()):
        raise RuntimeError(
            "at least one host failed the final InfiniBand PMA counter read"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if local_rank == 0 and nic_record is not None:
        nic_record.update(
            {
                "host": socket.gethostname(),
                "node_rank": node_rank,
                "hcas": list(hcas),
            }
        )
        atomic_write_json(args.output_dir / f"nic-node-{node_rank}.json", nic_record)

    event_lines = "".join(json.dumps(event, sort_keys=True) + "\n" for event in checkpoint_events)
    atomic_write_text(args.output_dir / f"events-rank-{rank:04d}.jsonl", event_lines)
    rank_summary = {
        "rank": rank,
        "local_rank": local_rank,
        "node_rank": node_rank,
        "host": socket.gethostname(),
        "peer_rank": peer_rank,
        "peer_local_rank": peer_local_rank,
        "train_gpu_ms": summarize(train_gpu_ms),
        "train_wall_ms": summarize(train_wall_ms),
        "end_to_end_step_ms": summarize(end_to_end_step_ms),
        "measured_wall_ms": measured_wall_ms,
        "checkpoint_count": len(checkpoint_events),
    }
    atomic_write_json(args.output_dir / f"rank-{rank:04d}.json", rank_summary)
    dist.barrier()

    if rank == 0:
        all_events: list[dict[str, object]] = []
        rank_summaries = []
        for source_rank in range(world_size):
            rank_summaries.append(
                json.loads((args.output_dir / f"rank-{source_rank:04d}.json").read_text())
            )
            event_path = args.output_dir / f"events-rank-{source_rank:04d}.jsonl"
            all_events.extend(
                json.loads(line) for line in event_path.read_text().splitlines() if line
            )
        nic_records = []
        for source_node in range(2):
            path = args.output_dir / f"nic-node-{source_node}.json"
            if path.is_file():
                nic_records.append(json.loads(path.read_text()))

        def event_metric(name: str) -> list[float]:
            return [float(event[name]) for event in all_events]

        available_nic_records = [
            record for record in nic_records if record.get("status") == "available"
        ]
        physical_nic_xmit_bytes = (
            sum(
                int(record["measured_delta"]["total_xmit_bytes"])
                for record in available_nic_records
            )
            if len(available_nic_records) == 2
            else None
        )
        physical_nic_rcv_bytes = (
            sum(
                int(record["measured_delta"]["total_rcv_bytes"])
                for record in available_nic_records
            )
            if len(available_nic_records) == 2
            else None
        )
        summary = {
            "schema_version": 1,
            "run_kind": "two_host_adacheck_rank_pairing_data_plane",
            "condition": {
                "pairing": args.pairing,
                "checkpoint_mode": args.checkpoint_mode,
                "max_inflight": args.max_inflight,
            },
            "world_size": world_size,
            "ranks_per_node": ranks_per_node,
            "hosts": [record["host"] for record in rank_summaries if record["local_rank"] == 0],
            "groups": [list(group) for group in plan.groups],
            "local_rank_permutation": list(local_pairing.permutation),
            "planner_cost_proxy": local_pairing.estimated_cost,
            "payload_profile_mib": list(payload_mib),
            "nic_cost_profile": list(nic_costs),
            "model_logical_state_bytes": model_logical_bytes,
            "application_checkpoint_send_bytes": sum(
                int(event["send_payload_bytes"]) for event in all_events
            ),
            "checkpoint_event_count": len(all_events),
            "checkpoint_commit_ms": summarize(event_metric("commit_ms")),
            "checkpoint_enqueue_ms": summarize(event_metric("enqueue_ms")),
            "checkpoint_exposed_stall_ms": summarize(event_metric("exposed_stall_ms")),
            "checkpoint_nonblocking_window_ms": summarize(
                event_metric("nonblocking_window_ms")
            ),
            "checkpoint_nonblocking_fraction": summarize(
                event_metric("nonblocking_fraction")
            ),
            "rank_train_gpu_mean_ms": [record["train_gpu_ms"]["mean"] for record in rank_summaries],
            "rank_train_wall_mean_ms": [record["train_wall_ms"]["mean"] for record in rank_summaries],
            "rank_end_to_end_step_mean_ms": [
                record["end_to_end_step_ms"]["mean"] for record in rank_summaries
            ],
            "rank_measured_wall_ms": [record["measured_wall_ms"] for record in rank_summaries],
            "physical_nic_counter": {
                "source": "InfiniBand PMA 64-bit port counters",
                "scope": "node-global; includes DDP, checkpoint, barriers, and unrelated RDMA",
                "node_records_available": len(available_nic_records),
                "aggregate_xmit_bytes": physical_nic_xmit_bytes,
                "aggregate_rcv_bytes": physical_nic_rcv_bytes,
                "records": [f"nic-node-{node}.json" for node in range(2)],
            },
            "counter_attribution_warning": (
                "PMA counters are node-global. Use idle deltas and a no-checkpoint control; "
                "do not attribute all measured bytes to checkpoint traffic."
            ),
            "overlap_definition": (
                "Per-event nonblocking_fraction is scheduler-visible time between enqueue and "
                "drain divided by commit latency. True system overlap must be computed from "
                "runtime overhead versus matched off/sync controls."
            ),
            "fault_guarantee": (
                "replica crosses one host boundary and is value-validated before pair ACK; "
                "this harness does not itself restart a failed host"
            ),
            "recovery_validation": "all received versioned shards passed pair-wide commit ACK",
            "percentile_method": "nearest-rank",
            "seed": args.seed,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "checkpoint_interval": args.checkpoint_interval,
            "checkpoint_warmup_events": args.checkpoint_warmup_events,
            "environment": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "nccl": torch.cuda.nccl.version(),
                "NCCL_IB_HCA": os.environ.get("NCCL_IB_HCA"),
                "NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME"),
            },
        }
        atomic_write_json(args.output_dir / "summary.json", summary)
        atomic_write_text(args.output_dir / "command.txt", " ".join(os.sys.argv) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
