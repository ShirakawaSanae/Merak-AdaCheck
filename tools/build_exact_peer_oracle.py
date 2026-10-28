#!/usr/bin/env python3
"""Build a diagnostic exact-value catalog from captured rank states."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

from Merak.core.checkpoint.catalog import deserialize_checkpoint, iter_checkpoint_objects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ranks-per-host", type=int)
    return parser.parse_args()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def signature_id(item) -> str:
    encoded = json.dumps(
        {
            "role": item.role,
            "dtype": item.dtype,
            "shape": list(item.shape),
            "checksum": item.checksum,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    args = parse_args()
    source_dir = args.capture_dir / "source"
    metadata_paths = sorted(source_dir.glob("rank_*.json"))
    if not metadata_paths:
        raise ValueError(f"capture contains no rank metadata: {args.capture_dir}")
    metadata = [json.loads(path.read_text()) for path in metadata_paths]
    world_sizes = {int(item["world_size"]) for item in metadata}
    local_world_sizes = {int(item["local_world_size"]) for item in metadata}
    epochs = {int(item["epoch"]) for item in metadata}
    if len(world_sizes) != 1 or len(local_world_sizes) != 1 or len(epochs) != 1:
        raise ValueError("capture ranks disagree on topology or epoch")
    world_size = world_sizes.pop()
    ranks_per_host = args.ranks_per_host or local_world_sizes.pop()
    epoch = epochs.pop()
    if world_size != 2 * ranks_per_host or len(metadata) != world_size:
        raise ValueError("exact peer oracle requires two complete equal-sized hosts")
    rank_hosts = {int(item["rank"]): str(item["host"]) for item in metadata}
    first_hosts = {rank_hosts[rank] for rank in range(ranks_per_host)}
    second_hosts = {rank_hosts[rank] for rank in range(ranks_per_host, world_size)}
    if len(first_hosts) != 1 or len(second_hosts) != 1 or first_hosts == second_hosts:
        raise ValueError("capture rank order does not describe two contiguous hosts")

    records = {}
    full_logical_bytes = 0
    input_object_count = 0
    for expected_rank, item in enumerate(metadata):
        rank = int(item["rank"])
        if rank != expected_rank:
            raise ValueError("capture rank metadata is incomplete or unordered")
        payload = (source_dir / f"rank_{rank:05d}.bin").read_bytes()
        if len(payload) != int(item["payload_bytes"]):
            raise ValueError(f"rank {rank} payload size changed")
        state = deserialize_checkpoint(payload, str(item["payload_checksum"]))
        del payload
        rank_logical_bytes = 0
        rank_object_count = 0
        for checkpoint_object in iter_checkpoint_objects(state):
            value_id = signature_id(checkpoint_object)
            record = records.setdefault(
                value_id,
                {
                    "value_id": value_id,
                    "role": checkpoint_object.role,
                    "dtype": checkpoint_object.dtype,
                    "shape": list(checkpoint_object.shape),
                    "logical_bytes": checkpoint_object.logical_bytes,
                    "checksum": checkpoint_object.checksum,
                    "source_ranks": set(),
                    "requirements": [],
                },
            )
            expected_signature = (
                record["role"],
                record["dtype"],
                tuple(record["shape"]),
                record["logical_bytes"],
                record["checksum"],
            )
            actual_signature = (
                checkpoint_object.role,
                checkpoint_object.dtype,
                checkpoint_object.shape,
                checkpoint_object.logical_bytes,
                checkpoint_object.checksum,
            )
            if actual_signature != expected_signature:
                raise ValueError(f"oracle signature collision for {value_id}")
            record["source_ranks"].add(rank)
            record["requirements"].append(
                {
                    "rank": rank,
                    "path": checkpoint_object.path,
                    "object_id": checkpoint_object.object_id,
                    "alias_id": f"rank:{rank}:{checkpoint_object.object_id}",
                }
            )
            rank_logical_bytes += checkpoint_object.logical_bytes
            rank_object_count += 1
        if rank_logical_bytes != int(item["logical_bytes"]):
            raise ValueError(f"rank {rank} logical bytes changed")
        full_logical_bytes += rank_logical_bytes
        input_object_count += rank_object_count
        del state
        gc.collect()
        print(
            f"rank={rank} objects={rank_object_count} "
            f"logical_bytes={rank_logical_bytes}",
            flush=True,
        )

    lines = []
    cross_host_objects = 0
    cross_host_unique_bytes = 0
    for value_id, record in sorted(records.items()):
        source_ranks = sorted(record.pop("source_ranks"))
        record["source_ranks"] = source_ranks
        record["requirements"] = sorted(
            record["requirements"], key=lambda value: (value["rank"], value["path"])
        )
        source_hosts = {rank_hosts[rank] for rank in source_ranks}
        if len(source_hosts) > 1:
            cross_host_objects += 1
            cross_host_unique_bytes += int(record["logical_bytes"])
        lines.append(json.dumps(record, sort_keys=True))
    catalog_payload = (("\n".join(lines) + "\n") if lines else "").encode()
    catalog_hash = hashlib.sha256(catalog_payload).hexdigest()
    atomic_bytes(args.output_dir / "object_catalog.jsonl", catalog_payload)
    summary = {
        "schema_version": 1,
        "run_kind": "exact_value_peer_oracle",
        "capture_dir": str(args.capture_dir.resolve()),
        "epoch": epoch,
        "world_size": world_size,
        "ranks_per_host": ranks_per_host,
        "rank_hosts": {str(rank): host for rank, host in sorted(rank_hosts.items())},
        "catalog_hash": catalog_hash,
        "full_logical_bytes": full_logical_bytes,
        "input_object_count": input_object_count,
        "unique_object_count": len(records),
        "cross_host_object_count": cross_host_objects,
        "cross_host_unique_bytes": cross_host_unique_bytes,
        "detector_evidence": False,
        "claim_scope": "diagnostic exact-value upper bound only",
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
