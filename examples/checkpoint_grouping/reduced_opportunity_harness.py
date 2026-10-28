#!/usr/bin/env python3
"""Measure recoverable tensor deduplication on real distributed training state."""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist

from Merak.core.checkpoint.catalog import (
    SerializedCheckpoint,
    TensorRedundancyEvidence,
    deserialize_checkpoint,
    serialize_checkpoint,
)
from Merak.core.checkpoint.reduced import (
    ReducedCheckpoint,
    build_reduced_checkpoint,
    plan_replication_cost,
)
from Merak.core.checkpoint.redundancy_checker import RedundancyChecker
from remote_checkpoint_harness import capture_state, train_range
from service_recovery_harness import build_training_objects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-step", type=int, default=5)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--replica-count", type=int)
    parser.add_argument("--tolerated-failures", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def atomic_bytes(path: Path, value: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def serialize_reduced(reduced: ReducedCheckpoint) -> bytes:
    buffer = io.BytesIO()
    torch.save(reduced, buffer, _use_new_zipfile_serialization=False)
    return buffer.getvalue()


def validate_reconstruction(
    reduced: ReducedCheckpoint,
    serialized_by_rank: dict[int, SerializedCheckpoint],
) -> None:
    for rank, expected in serialized_by_rank.items():
        restored = reduced.reconstruct_rank(rank)
        actual = serialize_checkpoint(restored)
        if actual.objects != expected.objects:
            raise RuntimeError(f"rank {rank} tensor catalog changed during reconstruction")
        if int(restored["iteration"]) <= 0:
            raise RuntimeError(f"rank {rank} lost checkpoint progress")
        if restored["rank_local"].tolist() != [rank, restored["iteration"]]:
            raise RuntimeError(f"rank {rank} restored another rank's local state")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    hosts: list[str | None] = [None] * world_size
    dist.all_gather_object(hosts, socket.gethostname())
    rank_to_domain = {item: str(hosts[item]) for item in range(world_size)}
    domains = sorted(set(rank_to_domain.values()))
    tolerated_failures = (
        args.tolerated_failures
        if args.tolerated_failures is not None
        else min(1, len(domains) - 1)
    )
    replica_count = (
        args.replica_count
        if args.replica_count is not None
        else tolerated_failures + 1
    )

    if rank == 0:
        (args.output_dir / "source").mkdir(parents=True, exist_ok=True)
        (args.output_dir / "detector").mkdir(parents=True, exist_ok=True)
    dist.barrier()

    device, model, optimizer, scheduler = build_training_objects(args, local_rank)
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
    checker = RedundancyChecker(zero_stage=0)
    checker.comm_groups_for_redundancy_checking = False
    checker.add_model_state(state["model"])
    checker.add_optimizer_state(state["optimizer"])
    checker.get_hash_values()
    checker.pack_to_tensor()
    torch.cuda.synchronize(device)
    detector_started = time.perf_counter()
    checker.do_redundancy_checking()
    evidence = checker.tensor_redundancy_evidence(rank)
    torch.cuda.synchronize(device)
    detector_ms = (time.perf_counter() - detector_started) * 1000
    serialized = serialize_checkpoint(state)
    atomic_bytes(args.output_dir / "source" / f"rank_{rank:05d}.bin", serialized.payload)
    atomic_json(
        args.output_dir / "source" / f"rank_{rank:05d}.json",
        {
            "rank": rank,
            "host": rank_to_domain[rank],
            "payload_bytes": len(serialized.payload),
            "payload_checksum": serialized.payload_checksum,
            "logical_tensor_bytes": serialized.logical_bytes,
            "object_count": len(serialized.objects),
            "detector_ms": detector_ms,
            "detector_packed_hash_count": int(checker.packed_tensor.numel()),
            "detector_tensor_evidence_count": len(evidence),
        },
    )
    atomic_json(
        args.output_dir / "detector" / f"rank_{rank:05d}.json",
        [item.to_dict() for item in evidence],
    )
    dist.barrier()

    if rank == 0:
        serialized_by_rank = {}
        states = {}
        detector_evidence = []
        detector_metadata = []
        for source_rank in range(world_size):
            metadata = json.loads(
                (
                    args.output_dir / "source" / f"rank_{source_rank:05d}.json"
                ).read_text()
            )
            payload = (
                args.output_dir / "source" / f"rank_{source_rank:05d}.bin"
            ).read_bytes()
            if len(payload) != int(metadata["payload_bytes"]):
                raise RuntimeError(f"rank {source_rank} source payload size changed")
            state = deserialize_checkpoint(payload, metadata["payload_checksum"])
            if int(state.get("iteration", -1)) != args.save_step:
                raise RuntimeError(
                    f"rank {source_rank} state is not from epoch {args.save_step}"
                )
            recataloged = serialize_checkpoint(state)
            if recataloged.logical_bytes != int(metadata["logical_tensor_bytes"]):
                raise RuntimeError(f"rank {source_rank} logical bytes changed")
            checkpoint = SerializedCheckpoint(
                payload=payload,
                payload_checksum=metadata["payload_checksum"],
                objects=recataloged.objects,
                logical_bytes=int(metadata["logical_tensor_bytes"]),
            )
            serialized_by_rank[source_rank] = checkpoint
            states[source_rank] = state
            detector_metadata.append(metadata)
            detector_evidence.extend(
                TensorRedundancyEvidence.from_dict(item)
                for item in json.loads(
                    (
                        args.output_dir
                        / "detector"
                        / f"rank_{source_rank:05d}.json"
                    ).read_text()
                )
            )

        build_started = time.perf_counter()
        reduced = build_reduced_checkpoint(states, detector_evidence)
        build_ms = (time.perf_counter() - build_started) * 1000
        reduced_payload = serialize_reduced(reduced)
        reduced = torch.load(
            io.BytesIO(reduced_payload), map_location="cpu", weights_only=False
        )
        validate_started = time.perf_counter()
        validate_reconstruction(reduced, serialized_by_rank)
        validate_ms = (time.perf_counter() - validate_started) * 1000
        cost = plan_replication_cost(
            reduced,
            rank_to_domain,
            domains,
            replica_count=replica_count,
            tolerated_failures=tolerated_failures,
        )
        atomic_bytes(args.output_dir / "reduced_bundle.pt", reduced_payload)

        catalog_path = args.output_dir / "object_catalog.jsonl"
        catalog_lines = []
        for item in reduced.objects:
            catalog_lines.append(
                json.dumps(
                    {
                        "value_id": item.value_id,
                        "role": item.role,
                        "dtype": item.dtype,
                        "shape": list(item.shape),
                        "logical_bytes": item.logical_bytes,
                        "checksum": item.checksum,
                        "source_ranks": list(item.source_ranks),
                        "source_domains": sorted(
                            {rank_to_domain[source] for source in item.source_ranks}
                        ),
                        "requirements": [
                            requirement.__dict__ for requirement in item.requirements
                        ],
                    },
                    sort_keys=True,
                )
            )
        atomic_bytes(catalog_path, ("\n".join(catalog_lines) + "\n").encode())
        atomic_json(args.output_dir / "placement_plan.json", cost.to_dict())
        atomic_json(
            args.output_dir / "topology.json",
            {
                "hosts": domains,
                "rank_to_host": rank_to_domain,
                "world_size": world_size,
                "local_world_size": local_world_size,
                "failure_domain": "host",
            },
        )
        atomic_json(
            args.output_dir / "environment.json",
            {
                "environment_prefix": os.environ.get("CKPT_ENV_PREFIX"),
                "hostname": socket.gethostname(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "nccl": list(torch.cuda.nccl.version()),
                "gpu_count_per_host": local_world_size,
            },
        )
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        tracked_dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], text=True
            ).strip()
        )
        git_status = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).splitlines()
        atomic_bytes(args.output_dir / "git_commit.txt", (commit + "\n").encode())
        atomic_bytes(
            args.output_dir / "command.txt",
            (" ".join(os.sys.argv) + "\n").encode(),
        )

        full_payload_bytes = sum(
            len(item.payload) for item in serialized_by_rank.values()
        )
        replicated_full_payload_bytes = full_payload_bytes * replica_count
        replicated_reduced_payload_bytes = len(reduced_payload) * replica_count
        summary = {
            "schema_version": 2,
            "git_commit": commit,
            "tracked_worktree_dirty": tracked_dirty,
            "git_status_porcelain": git_status,
            "run_kind": "offline_detector_guided_reduced_checkpoint_opportunity",
            "scope": (
                "RedundancyChecker-guided model/optimizer candidates; "
                "scheduler/RNG/progress retained rank-locally"
            ),
            "detector": "Merak RedundancyChecker candidate graph",
            "detector_short_hash_used_as_object_identity": False,
            "detector_tensor_evidence_count": len(detector_evidence),
            "detector_packed_hash_count_by_rank": [
                int(item["detector_packed_hash_count"])
                for item in detector_metadata
            ],
            "detector_ms_by_rank": [
                float(item["detector_ms"]) for item in detector_metadata
            ],
            "online_transport_measured": False,
            "world_size": world_size,
            "host_count": len(domains),
            "save_step": args.save_step,
            "source_object_count": sum(
                len(item.objects) for item in serialized_by_rank.values()
            ),
            "unique_reduced_object_count": len(reduced.objects),
            "full_logical_tensor_bytes": reduced.full_logical_bytes,
            "reduced_logical_tensor_bytes": reduced.reduced_logical_bytes,
            "logical_tensor_reduction_ratio": (
                1.0 - reduced.reduced_logical_bytes / reduced.full_logical_bytes
            ),
            "full_serialized_payload_bytes": full_payload_bytes,
            "reduced_serialized_payload_bytes": len(reduced_payload),
            "serialized_payload_reduction_ratio": (
                1.0 - len(reduced_payload) / full_payload_bytes
            ),
            "replica_count": replica_count,
            "tolerated_host_failures": tolerated_failures,
            "full_replication_payload_bytes": replicated_full_payload_bytes,
            "reduced_replication_payload_bytes": replicated_reduced_payload_bytes,
            "cross_domain_tensor_bytes_lower_bound": cost.cross_domain_bytes,
            "all_rank_tensor_catalogs_reconstructed": True,
            "all_rank_progress_and_identity_reconstructed": True,
            "failure_budget_validated_by_enumeration": True,
            "build_ms": build_ms,
            "reconstruction_validation_ms": validate_ms,
            "limitations": [
                "offline NFS collection is excluded from checkpoint timing",
                "cross-domain cost counts tensor bytes and excludes protocol metadata",
                "this symmetric DDP control does not represent an asymmetric parallel plan",
                "detector candidates are confirmed with dtype, shape, and full SHA-256",
                "ZeRO tensor-range evidence is not implemented",
            ],
        }
        atomic_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
