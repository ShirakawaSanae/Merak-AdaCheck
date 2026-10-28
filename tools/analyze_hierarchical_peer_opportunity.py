#!/usr/bin/env python3
"""Analyze four-host peer-local and host-pooled grouping opportunity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from Merak.core.checkpoint.hierarchical_peer import (
    build_hierarchical_study_from_catalog,
)
from tools.analyze_peer_delta_opportunity import (
    reduction_percent,
    validate_catalog_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opportunity-threshold-percent", type=float, default=5.0)
    parser.add_argument("--require-two-step-stability", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.opportunity_threshold_percent < 0:
        raise ValueError("opportunity threshold cannot be negative")
    summary = json.loads((args.frozen_dir / "summary.json").read_text())
    catalog_payload = (args.frozen_dir / "object_catalog.jsonl").read_bytes()
    objects = [
        json.loads(line) for line in catalog_payload.decode().splitlines() if line
    ]
    evidence_source, detector_evidence, catalog_content_hash = validate_catalog_source(
        summary, catalog_payload, objects
    )
    stability_steps = int(summary.get("detector_stability_steps", 1))
    if args.require_two_step_stability and stability_steps < 2:
        raise ValueError("formal four-host analysis requires two-step detector stability")
    world_size = int(summary["world_size"])
    ranks_per_host = int(summary["ranks_per_host"])
    study = build_hierarchical_study_from_catalog(
        objects,
        world_size=world_size,
        ranks_per_host=ranks_per_host,
        catalog_hash=str(summary["catalog_hash"]),
    )
    for policy, evaluation in study.items():
        atomic_json(args.output_dir / f"hierarchical-plan-{policy}.json", evaluation.to_dict())

    fixed = study["sequential_fixed"]
    peer_gaps = {
        policy: reduction_percent(
            fixed.peer_local_transfer_bytes,
            study[policy].peer_local_transfer_bytes,
        )
        for policy in ("host_only", "rank_only", "hierarchical_joint")
    }
    pooled_gaps = {
        "host_pooled_host": reduction_percent(
            fixed.host_pooled_transfer_bytes,
            study["host_pooled_opt"].host_pooled_transfer_bytes,
        ),
        "host_pooled_publisher": reduction_percent(
            fixed.host_pooled_max_publisher_bytes,
            study["host_pooled_publisher"].host_pooled_max_publisher_bytes,
        ),
        "host_pooled_joint": reduction_percent(
            fixed.host_pooled_max_publisher_bytes,
            study["host_pooled_joint"].host_pooled_max_publisher_bytes,
        ),
    }
    threshold = args.opportunity_threshold_percent
    eligible = []
    equivalent_treatments = {}
    identities = {}
    policy_evaluation = {
        **{policy: study[policy] for policy in peer_gaps},
        "host_pooled_host": study["host_pooled_opt"],
        "host_pooled_publisher": study["host_pooled_publisher"],
        "host_pooled_joint": study["host_pooled_joint"],
    }
    gaps = {**peer_gaps, **pooled_gaps}
    for policy in (
        "host_only",
        "rank_only",
        "hierarchical_joint",
        "host_pooled_host",
        "host_pooled_publisher",
        "host_pooled_joint",
    ):
        if gaps[policy] < threshold:
            continue
        evaluation = policy_evaluation[policy]
        identity = hashlib.sha256(
            json.dumps(
                {
                    "availability_scope": (
                        "host" if policy.startswith("host_pooled_") else "peer"
                    ),
                    "groups": evaluation.grouping.groups,
                    "publisher_policy": (
                        "least_load"
                        if policy in {"host_pooled_publisher", "host_pooled_joint"}
                        else (
                            "first_eligible"
                            if policy.startswith("host_pooled_")
                            else "rank_local"
                        )
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        if identity in identities:
            equivalent_treatments[policy] = identities[identity]
        else:
            identities[identity] = policy
            eligible.append(policy)
    result = {
        "schema_version": 1,
        "run_kind": "hierarchical_peer_offline_opportunity",
        "frozen_dir": str(args.frozen_dir.resolve()),
        "world_size": world_size,
        "ranks_per_host": ranks_per_host,
        "host_count": world_size // ranks_per_host,
        "catalog_hash": summary["catalog_hash"],
        "catalog_content_hash": catalog_content_hash,
        "evidence_source": evidence_source,
        "detector_evidence": detector_evidence,
        "detector_stability_steps": stability_steps,
        "peer_local_reduction_percent": peer_gaps,
        "host_pooled_reduction_percent": pooled_gaps,
        "opportunity_threshold_percent": threshold,
        "eligible_runtime_treatments": eligible,
        "equivalent_treatments": equivalent_treatments,
        "passes_offline_gate": bool(detector_evidence and eligible),
        "policies": {
            policy: evaluation.to_dict() for policy, evaluation in study.items()
        },
        "policy_aliases": {
            "adacheck_local_rank": {
                "equivalent_to": "sequential_fixed",
                "reason": "ordered hosts and corresponding local ranks produce the same grouping",
                "executed_separately": False,
            }
        },
        "limitations": [
            "logical bytes and publisher load are offline metrics, not latency predictions",
            "peer-local and host-pooled availability are separate treatments",
            "shared NIC, queueing, and completion latency require runtime measurement",
            "failure-domain coverage is static and not a physical power-off result",
        ],
    }
    atomic_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
