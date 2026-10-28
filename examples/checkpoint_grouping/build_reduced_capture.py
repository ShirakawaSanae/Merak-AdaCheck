#!/usr/bin/env python3
"""Build one immutable reduced bundle from an exported detector capture."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path

import torch

from Merak.core.checkpoint.catalog import (
    CheckpointObject,
    TensorRedundancyEvidence,
    deserialize_checkpoint,
    serialize_checkpoint,
)
from Merak.core.checkpoint.matched import reduced_catalog_hash
from Merak.core.checkpoint.opportunity import (
    asymmetric_plan_from_reduced,
    grouping_for_policy,
)
from Merak.core.checkpoint.reduced import (
    build_reduced_checkpoint,
    confirm_detector_evidence,
    confirm_detector_object_evidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--stability-capture-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grouping-seed", type=int, default=20260831)
    parser.add_argument("--ranks-per-host", type=int)
    parser.add_argument("--metadata-only", action="store_true")
    return parser.parse_args()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def load_capture(capture_dir: Path):
    metadata_paths = sorted(
        path
        for path in (capture_dir / "source").glob("rank_*.json")
        if not path.name.endswith(".objects.json")
    )
    if not metadata_paths:
        raise ValueError(f"capture contains no rank metadata: {capture_dir}")
    metadata = [json.loads(path.read_text()) for path in metadata_paths]
    world_sizes = {int(item["world_size"]) for item in metadata}
    local_world_sizes = {int(item["local_world_size"]) for item in metadata}
    epochs = {int(item["epoch"]) for item in metadata}
    detector_scopes = {
        str(item.get("detector_comparison_scope", "data")) for item in metadata
    }
    if (
        len(world_sizes) != 1
        or len(local_world_sizes) != 1
        or len(epochs) != 1
        or len(detector_scopes) != 1
    ):
        raise ValueError("capture ranks disagree on topology or epoch")
    world_size = world_sizes.pop()
    captured_local_world_size = local_world_sizes.pop()
    epoch = epochs.pop()
    detector_scope = detector_scopes.pop()
    if len(metadata) != world_size:
        raise ValueError(f"capture has {len(metadata)} ranks, expected {world_size}")
    states = {}
    evidence = []
    hosts = {}
    for expected_rank, item in enumerate(metadata):
        rank = int(item["rank"])
        if rank != expected_rank:
            raise ValueError("capture rank metadata is incomplete or unordered")
        payload = (capture_dir / "source" / f"rank_{rank:05d}.bin").read_bytes()
        if len(payload) != int(item["payload_bytes"]):
            raise ValueError(f"rank {rank} payload size changed")
        states[rank] = deserialize_checkpoint(payload, item["payload_checksum"])
        hosts[rank] = str(item["host"])
        evidence.extend(
            TensorRedundancyEvidence.from_dict(value)
            for value in json.loads(
                (capture_dir / "detector" / f"rank_{rank:05d}.json").read_text()
            )
        )
    return (
        states,
        evidence,
        hosts,
        world_size,
        captured_local_world_size,
        epoch,
        detector_scope,
    )


def load_metadata_capture(capture_dir: Path):
    metadata_paths = sorted(
        path
        for path in (capture_dir / "source").glob("rank_*.json")
        if not path.name.endswith(".objects.json")
    )
    if not metadata_paths:
        raise ValueError(f"capture contains no rank metadata: {capture_dir}")
    metadata = [json.loads(path.read_text()) for path in metadata_paths]
    common = {
        "world_size": {int(item["world_size"]) for item in metadata},
        "local_world_size": {int(item["local_world_size"]) for item in metadata},
        "epoch": {int(item["epoch"]) for item in metadata},
        "detector_scope": {
            str(item.get("detector_comparison_scope", "data")) for item in metadata
        },
        "evidence_roles": {
            tuple(item.get("detector_evidence_roles", ())) for item in metadata
        },
        "unique_roles": {
            tuple(item.get("conservative_unique_roles", ())) for item in metadata
        },
    }
    if any(len(values) != 1 for values in common.values()):
        raise ValueError("metadata capture ranks disagree on topology or evidence scope")
    if any(item.get("capture_format") != "metadata_only" for item in metadata):
        raise ValueError("metadata capture contains a payload-format rank")
    world_size = common["world_size"].pop()
    if len(metadata) != world_size:
        raise ValueError(f"capture has {len(metadata)} ranks, expected {world_size}")
    objects = {}
    evidence = []
    hosts = {}
    for expected_rank, item in enumerate(metadata):
        rank = int(item["rank"])
        if rank != expected_rank:
            raise ValueError("capture rank metadata is incomplete or unordered")
        values = json.loads(
            (
                capture_dir / "source" / f"rank_{rank:05d}.objects.json"
            ).read_text()
        )
        rank_objects = tuple(CheckpointObject.from_dict(value) for value in values)
        if len(rank_objects) != int(item["object_count"]):
            raise ValueError(f"rank {rank} object count changed")
        if sum(value.logical_bytes for value in rank_objects) != int(
            item["logical_bytes"]
        ):
            raise ValueError(f"rank {rank} logical byte count changed")
        for value in rank_objects:
            key = (rank, value.path)
            if key in objects:
                raise ValueError(f"duplicate checkpoint object {key}")
            objects[key] = value
        hosts[rank] = str(item["host"])
        evidence.extend(
            TensorRedundancyEvidence.from_dict(value)
            for value in json.loads(
                (capture_dir / "detector" / f"rank_{rank:05d}.json").read_text()
            )
        )
    return (
        objects,
        tuple(evidence),
        hosts,
        world_size,
        common["local_world_size"].pop(),
        common["epoch"].pop(),
        common["detector_scope"].pop(),
        common["evidence_roles"].pop(),
        common["unique_roles"].pop(),
    )


def stable_evidence(first, second):
    """Keep equal-location components stable across both confirmed snapshots."""
    def signatures(evidence):
        locations = {
            (item.rank, item.local_index): (item.rank, item.path) for item in evidence
        }
        return {
            (item.rank, item.path): tuple(
                sorted(locations[location] for location in item.equivalent_locations)
            )
            for item in evidence
        }

    first_signatures = signatures(first)
    second_signatures = signatures(second)
    result = []
    stable_components = set()
    singleton_index = max((item.local_index for item in first), default=-1) + 1
    for item in first:
        key = (item.rank, item.path)
        signature = first_signatures[key]
        if second_signatures.get(key) == signature:
            stable_components.add(signature)
            result.append(item)
        else:
            result.append(
                TensorRedundancyEvidence(
                    rank=item.rank,
                    role=item.role,
                    path=item.path,
                    local_index=singleton_index,
                    equivalent_locations=((item.rank, singleton_index),),
                )
            )
            singleton_index += 1
    return tuple(result), {
        "stable_components": len(stable_components),
        "stable_records": sum(len(component) for component in stable_components),
        "singleton_fallback_records": len(result)
        - sum(len(component) for component in stable_components),
    }


def build_metadata_object_catalog(objects, evidence, hosts):
    evidence_by_path = {(item.rank, item.path): item for item in evidence}
    if len(evidence_by_path) != len(evidence):
        raise ValueError("metadata catalog contains duplicate detector evidence")
    evidence_roles = {item.role for item in evidence}
    records = {}
    full_logical_bytes = 0
    for (rank, path), item in sorted(objects.items()):
        detector = evidence_by_path.get((rank, path))
        if detector is None and item.role in evidence_roles:
            raise ValueError(f"detector evidence is missing rank {rank} tensor {path}")
        scope = (
            {
                "kind": "detector_component",
                "locations": [list(value) for value in detector.equivalent_locations],
            }
            if detector is not None
            else {"kind": "rank_path", "rank": rank, "path": path}
        )
        identity = json.dumps(
            {
                "role": item.role,
                "dtype": item.dtype,
                "shape": list(item.shape),
                "checksum": item.checksum,
                "scope": scope,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        value_id = hashlib.sha256(identity).hexdigest()
        full_logical_bytes += item.logical_bytes
        record = records.setdefault(
            value_id,
            {
                "value_id": value_id,
                "role": item.role,
                "dtype": item.dtype,
                "shape": list(item.shape),
                "logical_bytes": item.logical_bytes,
                "checksum": item.checksum,
                "source_ranks": set(),
                "source_hosts": set(),
                "requirements": [],
            },
        )
        record["source_ranks"].add(rank)
        record["source_hosts"].add(hosts[rank])
        record["requirements"].append(
            {
                "rank": rank,
                "path": path,
                "object_id": item.object_id,
                "alias_id": f"rank:{rank}:{item.object_id}",
            }
        )
    result = []
    for value_id in sorted(records):
        record = records[value_id]
        record["source_ranks"] = sorted(record["source_ranks"])
        record["source_hosts"] = sorted(record["source_hosts"])
        record["requirements"].sort(key=lambda value: (value["rank"], value["path"]))
        result.append(record)
    return result, full_logical_bytes


def semantic_catalog_hash(full_logical_bytes, world_size, objects):
    value = {
        "full_logical_bytes": full_logical_bytes,
        "template_ranks": list(range(world_size)),
        "objects": [
            {
                key: item[key]
                for key in (
                    "value_id",
                    "role",
                    "dtype",
                    "shape",
                    "logical_bytes",
                    "checksum",
                    "source_ranks",
                    "requirements",
                )
            }
            for item in objects
        ],
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_metadata_capture(args):
    (
        objects,
        evidence,
        hosts,
        world_size,
        local_world_size,
        epoch,
        detector_scope,
        evidence_roles,
        unique_roles,
    ) = load_metadata_capture(args.capture_dir)
    ranks_per_host = args.ranks_per_host or local_world_size
    if ranks_per_host <= 0 or world_size % ranks_per_host or world_size // ranks_per_host < 2:
        raise ValueError("capture must contain at least two equal-sized hosts")
    confirmed, confirmation = confirm_detector_object_evidence(objects, evidence)
    stability = None
    stability_epoch = None
    if args.stability_capture_dir is not None:
        second = load_metadata_capture(args.stability_capture_dir)
        if (
            second[3] != world_size
            or second[4] != local_world_size
            or second[6:] != (detector_scope, evidence_roles, unique_roles)
            or second[5] == epoch
        ):
            raise ValueError("stability metadata topology/scope/roles/epoch changed")
        second_confirmed, second_confirmation = confirm_detector_object_evidence(
            second[0], second[1]
        )
        confirmed, stability = stable_evidence(confirmed, second_confirmed)
        stability["second_confirmation"] = second_confirmation.__dict__
        stability_epoch = second[5]
    catalog, full_logical_bytes = build_metadata_object_catalog(
        objects, confirmed, hosts
    )
    payload = (
        "\n".join(json.dumps(item, sort_keys=True) for item in catalog) + "\n"
    ).encode()
    atomic_bytes(args.output_dir / "object_catalog.jsonl", payload)
    summary = {
        "schema_version": 1,
        "run_kind": "matched_detector_metadata_capture_build",
        "capture_dir": str(args.capture_dir.resolve()),
        "epoch": epoch,
        "stability_epoch": stability_epoch,
        "world_size": world_size,
        "ranks_per_host": ranks_per_host,
        "captured_local_world_size": local_world_size,
        "catalog_hash": semantic_catalog_hash(
            full_logical_bytes, world_size, catalog
        ),
        "full_logical_bytes": full_logical_bytes,
        "reduced_logical_bytes": sum(item["logical_bytes"] for item in catalog),
        "object_count": len(catalog),
        "detector_confirmation": confirmation.__dict__,
        "detector_stability_steps": 2 if stability is not None else 1,
        "detector_stability": stability,
        "detector_comparison_scope": detector_scope,
        "detector_evidence_roles": list(evidence_roles),
        "conservative_unique_roles": list(unique_roles),
        "recoverable_payload_present": False,
        "all_rank_catalogs_reconstructed": False,
        "policy_hashes": {},
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.metadata_only:
        build_metadata_capture(args)
        return
    (
        states,
        evidence,
        hosts,
        world_size,
        captured_local_world_size,
        epoch,
        detector_scope,
    ) = load_capture(args.capture_dir)
    ranks_per_host = args.ranks_per_host or captured_local_world_size
    if world_size % ranks_per_host or world_size // ranks_per_host < 2:
        raise ValueError("capture must contain at least two equal-sized hosts")
    confirmed_evidence, confirmation = confirm_detector_evidence(states, evidence)
    stability = None
    stability_epoch = None
    if args.stability_capture_dir is not None:
        del states
        gc.collect()
        (
            second_states,
            second_evidence,
            _,
            second_world_size,
            second_local_world_size,
            stability_epoch,
            second_scope,
        ) = load_capture(args.stability_capture_dir)
        if (
            second_world_size != world_size
            or second_local_world_size != captured_local_world_size
            or second_scope != detector_scope
            or stability_epoch == epoch
        ):
            raise ValueError("stability capture topology/scope/epoch does not match")
        second_confirmed, second_confirmation = confirm_detector_evidence(
            second_states, second_evidence
        )
        confirmed_evidence, stability = stable_evidence(
            confirmed_evidence, second_confirmed
        )
        stability["second_confirmation"] = second_confirmation.__dict__
        del second_states
        gc.collect()
        reloaded = load_capture(args.capture_dir)
        states = reloaded[0]
        hosts = reloaded[2]
    reduced = build_reduced_checkpoint(states, confirmed_evidence)
    for rank, expected in states.items():
        actual_catalog = serialize_checkpoint(reduced.reconstruct_rank(rank)).objects
        expected_catalog = serialize_checkpoint(expected).objects
        if actual_catalog != expected_catalog:
            raise RuntimeError(f"rank {rank} reconstruction catalog changed")
    buffer = io.BytesIO()
    torch.save(reduced, buffer, _use_new_zipfile_serialization=False)
    atomic_bytes(args.output_dir / "reduced_bundle.pt", buffer.getvalue())

    opportunity = asymmetric_plan_from_reduced(
        reduced,
        name=args.capture_dir.name,
        ranks_per_host=ranks_per_host,
        metadata={"capture_dir": str(args.capture_dir.resolve()), "epoch": epoch},
    )
    policy_hashes = {}
    for policy in ("fixed", "random", "exact"):
        grouping = grouping_for_policy(
            opportunity, policy, seed=args.grouping_seed
        )
        atomic_json(args.output_dir / f"grouping-{policy}.json", grouping.to_dict())
        policy_hashes[policy] = grouping.plan_hash
    catalog_hash = reduced_catalog_hash(reduced)
    object_lines = [
        json.dumps(
            {
                "value_id": item.value_id,
                "role": item.role,
                "dtype": item.dtype,
                "shape": list(item.shape),
                "logical_bytes": item.logical_bytes,
                "checksum": item.checksum,
                "source_ranks": list(item.source_ranks),
                "source_hosts": sorted({hosts[rank] for rank in item.source_ranks}),
                "requirements": [requirement.__dict__ for requirement in item.requirements],
            },
            sort_keys=True,
        )
        for item in reduced.objects
    ]
    atomic_bytes(
        args.output_dir / "object_catalog.jsonl",
        (("\n".join(object_lines) + "\n") if object_lines else "").encode(),
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    summary = {
        "schema_version": 1,
        "run_kind": "matched_detector_capture_build",
        "git_commit": commit,
        "tracked_worktree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], text=True
            ).strip()
        ),
        "capture_dir": str(args.capture_dir.resolve()),
        "epoch": epoch,
        "stability_epoch": stability_epoch,
        "world_size": world_size,
        "ranks_per_host": ranks_per_host,
        "captured_local_world_size": captured_local_world_size,
        "catalog_hash": catalog_hash,
        "full_logical_bytes": reduced.full_logical_bytes,
        "reduced_logical_bytes": reduced.reduced_logical_bytes,
        "object_count": len(reduced.objects),
        "detector_confirmation": confirmation.__dict__,
        "detector_stability_steps": 2 if stability is not None else 1,
        "detector_stability": stability,
        "detector_comparison_scope": detector_scope,
        "policy_hashes": policy_hashes,
        "all_rank_catalogs_reconstructed": True,
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
