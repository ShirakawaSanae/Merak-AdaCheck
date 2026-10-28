#!/usr/bin/env python3
"""Fail-closed NCCL all-reduce preflight for checkpoint experiments."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    value = torch.tensor(float(rank + 1), device=f"cuda:{local_rank}")
    dist.all_reduce(value)
    expected = world_size * (world_size + 1) / 2
    if value.item() != expected:
        raise RuntimeError(f"NCCL all-reduce mismatch: {value.item()} != {expected}")
    record = {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "cuda_device": torch.cuda.get_device_name(local_rank),
        "all_reduce_value": value.item(),
    }
    atomic_json(args.output_dir / f"rank-{rank:05d}.json", record)
    dist.barrier()
    if rank == 0:
        records = [
            json.loads(path.read_text())
            for path in sorted(args.output_dir.glob("rank-*.json"))
        ]
        if len(records) != world_size or {item["rank"] for item in records} != set(
            range(world_size)
        ):
            raise RuntimeError("preflight rank records are incomplete")
        atomic_json(
            args.output_dir / "summary.json",
            {
                "schema_version": 1,
                "run_kind": "four_host_nccl_preflight",
                "world_size": world_size,
                "hostnames": sorted({item["hostname"] for item in records}),
                "all_reduce_expected": expected,
                "all_reduce_valid": all(
                    item["all_reduce_value"] == expected for item in records
                ),
                "rank_records": len(records),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
