#!/usr/bin/env python3
"""Analyze pair-local checkpoint deltas from one frozen JSONL catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from functools import lru_cache
from pathlib import Path

from Merak.core.checkpoint.peer_delta import (
    PeerDeltaMatrix,
    PeerResourceProfile,
    build_peer_delta_matrix_from_catalog,
    evaluate_peer_grouping,
    peer_grouping_for_policy,
)
from Merak.core.checkpoint.opportunity import RankGrouping
from Merak.core.checkpoint.peer_delta_runtime import (
    PeerDeltaProtectionPlan,
    PeerShardAssignment,
)


HISTORICAL_POLICIES = ("fixed", "random", "redundancy_max", "profile_exact")
MOTIVATION_POLICIES = (
    "adacheck_style",
    "gemini_style",
    "random_feasible",
    "redundancy_max",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-dir", type=Path, required=True)
    parser.add_argument(
        "--base-frozen-dir",
        type=Path,
        help="optional previous epoch; only changed rank/path requirements are costed",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--opportunity-threshold-percent", type=float, default=5.0)
    parser.add_argument("--first-to-second-bps", type=float)
    parser.add_argument("--second-to-first-bps", type=float)
    parser.add_argument("--first-service-bps", type=float)
    parser.add_argument("--second-service-bps", type=float)
    parser.add_argument(
        "--policy-contract",
        choices=("historical", "motivation-v2", "motivation-opportunity"),
        default="historical",
        help="choose the historical, host-constrained, or unconstrained motivation contract",
    )
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def build_profile(args: argparse.Namespace) -> PeerResourceProfile | None:
    directional = (args.first_to_second_bps, args.second_to_first_bps)
    if all(value is None for value in directional):
        if args.first_service_bps is not None or args.second_service_bps is not None:
            raise ValueError("service rates require both directional network rates")
        return None
    if any(value is None for value in directional):
        raise ValueError("both directional network rates are required")
    return PeerResourceProfile(
        first_to_second_bytes_per_second=args.first_to_second_bps,
        second_to_first_bytes_per_second=args.second_to_first_bps,
        first_service_bytes_per_second=(
            float("inf") if args.first_service_bps is None else args.first_service_bps
        ),
        second_service_bytes_per_second=(
            float("inf") if args.second_service_bps is None else args.second_service_bps
        ),
    )


def reduction_percent(baseline: float, candidate: float) -> float:
    if baseline == 0:
        return 0.0
    return 100.0 * (baseline - candidate) / baseline


def _logical_pair(matrix: PeerDeltaMatrix, first: int, second: int) -> dict[str, int]:
    first_values = set(matrix.rank_value_ids[first])
    second_values = set(matrix.rank_value_ids[second])

    def byte_sum(values: set[str]) -> int:
        return sum(matrix.object_bytes[value] for value in values)

    first_to_second = byte_sum(first_values - second_values)
    second_to_first = byte_sum(second_values - first_values)
    return {
        "first_required_bytes": byte_sum(first_values),
        "second_required_bytes": byte_sum(second_values),
        "first_to_second_bytes": first_to_second,
        "second_to_first_bytes": second_to_first,
        "overlap_bytes": byte_sum(first_values & second_values),
        "pair_peak_bytes": max(first_to_second, second_to_first),
    }


def _logical_only_report(
    matrix: PeerDeltaMatrix, grouping: RankGrouping
) -> dict[str, object]:
    flattened = [rank for group in grouping.groups for rank in group]
    if sorted(flattened) != list(range(matrix.world_size)):
        raise ValueError("grouping must contain every rank exactly once")
    pairs = [_logical_pair(matrix, first, second) for first, second in grouping.groups]
    cross_host = all(
        first // matrix.ranks_per_host != second // matrix.ranks_per_host
        for first, second in grouping.groups
    )
    return {
        "schema_version": 1,
        "grouping": grouping.to_dict(),
        "pair_local_transfer_bytes": sum(
            item["first_to_second_bytes"] + item["second_to_first_bytes"]
            for item in pairs
        ),
        "first_to_second_bytes": sum(
            item["first_to_second_bytes"] for item in pairs
        ),
        "second_to_first_bytes": sum(
            item["second_to_first_bytes"] for item in pairs
        ),
        "aggregate_pair_peak_bytes": sum(item["pair_peak_bytes"] for item in pairs),
        "maximum_pair_peak_bytes": max(
            (item["pair_peak_bytes"] for item in pairs), default=0
        ),
        "overlap_bytes": sum(item["overlap_bytes"] for item in pairs),
        "host_global_transfer_bytes": matrix.host_global_transfer_bytes,
        "makespan_proxy": max((item["pair_peak_bytes"] for item in pairs), default=0),
        "makespan_proxy_unit": "unit_bandwidth_bytes",
        "worst_rank_state_bytes": max(
            (
                max(item["first_required_bytes"], item["second_required_bytes"])
                for item in pairs
            ),
            default=0,
        ),
        "host_failure_feasible": cross_host,
        "planned_tolerated_host_failures": 1 if cross_host else 0,
        "static_failure_sets_covered": [[0], [1]] if cross_host else [],
        "runtime_failure_sets_validated": [],
        "failure_feasibility_reason": (
            None
            if cross_host
            else "one or more rank pairs are co-located under node-major placement"
        ),
    }


def _unconstrained_random_grouping(
    matrix: PeerDeltaMatrix, seed: int
) -> RankGrouping:
    ranks = list(range(matrix.world_size))
    random.Random(seed).shuffle(ranks)
    return RankGrouping(
        "random_feasible",
        tuple(
            sorted(
                tuple(sorted(ranks[index : index + 2]))
                for index in range(0, len(ranks), 2)
            )
        ),
        seed=seed,
    )


def _unconstrained_redundancy_grouping(matrix: PeerDeltaMatrix) -> RankGrouping:
    """Exact maximum-overlap perfect matching for the 16-rank motivation slice."""

    if matrix.world_size % 2 or matrix.world_size > 20:
        raise ValueError("unconstrained exact matching requires an even world size <= 20")
    overlap = {
        (first, second): _logical_pair(matrix, first, second)["overlap_bytes"]
        for first in range(matrix.world_size)
        for second in range(first + 1, matrix.world_size)
    }

    @lru_cache(maxsize=None)
    def solve(remaining: int) -> tuple[int, tuple[tuple[int, int], ...]]:
        if remaining == 0:
            return 0, ()
        first_bit = remaining & -remaining
        first = first_bit.bit_length() - 1
        tail_mask = remaining ^ first_bit
        best: tuple[int, tuple[tuple[int, int], ...]] | None = None
        candidates = tail_mask
        while candidates:
            second_bit = candidates & -candidates
            second = second_bit.bit_length() - 1
            tail_score, tail_groups = solve(tail_mask ^ second_bit)
            candidate = (
                overlap[(first, second)] + tail_score,
                ((first, second),) + tail_groups,
            )
            if best is None or candidate[0] > best[0] or (
                candidate[0] == best[0] and candidate[1] < best[1]
            ):
                best = candidate
            candidates ^= second_bit
        assert best is not None
        return best

    _, groups = solve((1 << matrix.world_size) - 1)
    return RankGrouping("redundancy_max", groups)


def motivation_policy_reports(
    matrix: PeerDeltaMatrix, *, seed: int, constrain_failure_domains: bool = True
) -> dict[str, dict[str, object]]:
    """Evaluate the four named policies without hiding GEMINI host infeasibility."""

    if matrix.world_size % 2:
        raise ValueError("GeminiStyleAdjacent requires an even world size")
    fixed = peer_grouping_for_policy(matrix, "fixed", seed=seed)
    if constrain_failure_domains:
        random_grouping = peer_grouping_for_policy(matrix, "random", seed=seed)
        exact = peer_grouping_for_policy(matrix, "redundancy_max", seed=seed)
    else:
        random_grouping = _unconstrained_random_grouping(matrix, seed)
        exact = _unconstrained_redundancy_grouping(matrix)
    groupings = {
        "adacheck_style": RankGrouping("adacheck_style", fixed.groups),
        "gemini_style": RankGrouping(
            "gemini_style",
            tuple((rank, rank + 1) for rank in range(0, matrix.world_size, 2)),
        ),
        "random_feasible": RankGrouping(
            "random_feasible", random_grouping.groups, seed=random_grouping.seed
        ),
        "redundancy_max": exact,
    }
    reports = {}
    for policy, grouping in groupings.items():
        if policy == "gemini_style" or not constrain_failure_domains:
            reports[policy] = _logical_only_report(matrix, grouping)
            continue
        report = evaluate_peer_grouping(matrix, grouping).to_dict()
        report["host_failure_feasible"] = True
        report["failure_feasibility_reason"] = None
        reports[policy] = report
    return reports


def host_global_arms(
    matrix: PeerDeltaMatrix, objects: list[dict]
) -> dict[str, object]:
    """Compare publisher load while keeping two-host recovery bytes invariant."""

    grouping = peer_grouping_for_policy(matrix, "fixed")
    peer = {
        rank: other
        for first, second in grouping.groups
        for rank, other in ((first, second), (second, first))
    }
    object_bytes = {
        str(item["value_id"]): int(item["logical_bytes"]) for item in objects
    }
    object_holders = {
        str(item["value_id"]): {int(rank) for rank in item["source_ranks"]}
        for item in objects
    }
    required = tuple((host, tuple(sorted(object_bytes))) for host in range(2))
    failures = ((), (0,), (1,))

    def build(balance_publishers: bool) -> PeerDeltaProtectionPlan:
        value_ids_by_assignment = {
            (rank, target): []
            for rank in range(matrix.world_size)
            for target in (
                rank // matrix.ranks_per_host,
                1 - rank // matrix.ranks_per_host,
            )
        }
        loads = {rank: 0 for rank in range(matrix.world_size)}
        for item in sorted(
            objects,
            key=lambda value: (-int(value["logical_bytes"]), value["value_id"]),
        ):
            value_id = str(item["value_id"])
            holders = object_holders[value_id]
            for target_host in range(2):
                local = sorted(
                    rank
                    for rank in holders
                    if rank // matrix.ranks_per_host == target_host
                )
                eligible = local or sorted(holders)
                if not eligible:
                    raise ValueError(f"object {value_id} has no eligible publisher")
                remote = not local
                publisher = eligible[0]
                if balance_publishers and remote:
                    publisher = min(eligible, key=lambda rank: (loads[rank], rank))
                value_ids_by_assignment[(publisher, target_host)].append(value_id)
                if remote:
                    loads[publisher] += object_bytes[value_id]
        assignments = []
        for rank in range(matrix.world_size):
            source_host = rank // matrix.ranks_per_host
            for target_host in (source_host, 1 - source_host):
                value_ids = tuple(sorted(value_ids_by_assignment[(rank, target_host)]))
                assignments.append(
                    PeerShardAssignment(
                        rank,
                        target_host,
                        peer[rank],
                        "base" if source_host == target_host else "delta",
                        value_ids,
                        sum(object_bytes[value_id] for value_id in value_ids),
                    )
                )
        return PeerDeltaProtectionPlan(
            catalog_hash=matrix.catalog_hash,
            grouping=grouping,
            availability_scope="host",
            world_size=matrix.world_size,
            ranks_per_host=matrix.ranks_per_host,
            assignments=tuple(assignments),
            required_value_ids_by_host=required,
            planned_failure_sets=failures,
            pair_local_transfer_bytes=sum(loads.values()),
            remote_transfer_bytes=sum(loads.values()),
            maximum_remote_publisher_bytes=max(loads.values(), default=0),
            publisher_policy="least_load" if balance_publishers else "first_eligible",
        )

    fixed = build(False)
    m1 = build(True)
    for field in (
        "catalog_hash",
        "availability_scope",
        "world_size",
        "ranks_per_host",
        "required_value_ids_by_host",
        "planned_failure_sets",
        "remote_transfer_bytes",
    ):
        if getattr(fixed, field) != getattr(m1, field):
            raise ValueError(f"host-global A/B changes common contract field {field}")
    if fixed.remote_transfer_bytes != matrix.host_global_transfer_bytes:
        raise ValueError("host-global A/B bytes differ from the catalog lower bound")

    def report(plan: PeerDeltaProtectionPlan) -> dict[str, object]:
        return {
            "catalog_hash": plan.catalog_hash,
            "availability_scope": plan.availability_scope,
            "publisher_policy": plan.publisher_policy,
            "total_remote_bytes": plan.remote_transfer_bytes,
            "maximum_remote_publisher_bytes": plan.maximum_remote_publisher_bytes,
            "planned_failure_sets": [list(item) for item in plan.planned_failure_sets],
            "plan_hash": plan.plan_hash,
        }

    return {
        "fixed": report(fixed),
        "m1": report(m1),
        "predicted_max_publisher_reduction_percent": reduction_percent(
            fixed.maximum_remote_publisher_bytes,
            m1.maximum_remote_publisher_bytes,
        ),
    }


def semantic_reduced_catalog_hash(summary: dict, objects: list[dict]) -> str:
    value = {
        "full_logical_bytes": int(summary["full_logical_bytes"]),
        "template_ranks": list(range(int(summary["world_size"]))),
        "objects": [
            {
                "value_id": item["value_id"],
                "role": item["role"],
                "dtype": item["dtype"],
                "shape": item["shape"],
                "logical_bytes": item["logical_bytes"],
                "checksum": item["checksum"],
                "source_ranks": item["source_ranks"],
                "requirements": item["requirements"],
            }
            for item in objects
        ],
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_catalog_source(
    summary: dict, catalog_payload: bytes, objects: list[dict]
) -> tuple[str, bool, str]:
    run_kind = summary.get("run_kind")
    content_hash = hashlib.sha256(catalog_payload).hexdigest()
    if run_kind in {
        "matched_detector_capture_build",
        "matched_detector_metadata_capture_build",
    }:
        actual = semantic_reduced_catalog_hash(summary, objects)
        if actual != summary.get("catalog_hash"):
            raise ValueError("detector catalog content does not match catalog_hash")
        if run_kind == "matched_detector_metadata_capture_build":
            if summary.get("recoverable_payload_present") is not False:
                raise ValueError("metadata-only catalog must reject payload recovery")
            return "adacheck_detector_confirmed_metadata_only", True, content_hash
        return "adacheck_detector_confirmed", True, content_hash
    if run_kind == "exact_value_peer_oracle":
        if content_hash != summary.get("catalog_hash"):
            raise ValueError("exact oracle catalog content does not match catalog_hash")
        if summary.get("detector_evidence") is not False:
            raise ValueError("exact oracle must explicitly reject detector provenance")
        return "diagnostic_exact_value_upper_bound", False, content_hash
    raise ValueError(f"unsupported peer catalog producer: {run_kind!r}")


def changed_requirement_catalog(
    base_objects: list[dict], current_objects: list[dict]
) -> tuple[list[dict], dict[str, int]]:
    """Keep current requirements whose tensor value changed from the base epoch."""

    def locations(objects: list[dict]) -> dict[tuple[int, str], tuple[object, ...]]:
        result = {}
        for item in objects:
            metadata = (
                str(item["role"]),
                str(item["dtype"]),
                tuple(int(value) for value in item["shape"]),
                int(item["logical_bytes"]),
                str(item["checksum"]),
            )
            for requirement in item["requirements"]:
                key = (int(requirement["rank"]), str(requirement["path"]))
                if key in result:
                    raise ValueError(f"duplicate catalog requirement {key}")
                result[key] = metadata
        return result

    base = locations(base_objects)
    current = locations(current_objects)
    if base.keys() != current.keys():
        missing = sorted(base.keys() - current.keys())
        added = sorted(current.keys() - base.keys())
        raise ValueError(
            f"catalog rank/path sets changed: missing={missing[:3]}, added={added[:3]}"
        )

    changed = []
    changed_requirements = 0
    changed_bytes = 0
    total_requirements = len(current)
    total_bytes = sum(int(metadata[3]) for metadata in current.values())
    for item in current_objects:
        requirements = []
        for requirement in item["requirements"]:
            key = (int(requirement["rank"]), str(requirement["path"]))
            base_metadata = base[key]
            current_metadata = current[key]
            if base_metadata[:4] != current_metadata[:4]:
                raise ValueError(f"catalog tensor metadata changed at {key}")
            if base_metadata[4] != current_metadata[4]:
                requirements.append(requirement)
                changed_requirements += 1
                changed_bytes += int(item["logical_bytes"])
        if requirements:
            value = dict(item)
            value["requirements"] = requirements
            value["source_ranks"] = sorted(
                {int(requirement["rank"]) for requirement in requirements}
            )
            changed.append(value)
    return changed, {
        "total_requirement_count": total_requirements,
        "changed_requirement_count": changed_requirements,
        "unchanged_requirement_count": total_requirements - changed_requirements,
        "total_requirement_logical_bytes": total_bytes,
        "changed_requirement_logical_bytes": changed_bytes,
        "unchanged_requirement_logical_bytes": total_bytes - changed_bytes,
        "changed_object_count": len(changed),
    }


def transition_catalog_hash(
    base_catalog_hash: str, current_catalog_hash: str, objects: list[dict]
) -> str:
    encoded = json.dumps(
        {
            "method": "changed_rank_path_requirements_v1",
            "base_catalog_hash": base_catalog_hash,
            "current_catalog_hash": current_catalog_hash,
            "objects": objects,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    args = parse_args()
    if args.opportunity_threshold_percent < 0:
        raise ValueError("opportunity threshold cannot be negative")
    summary = json.loads((args.frozen_dir / "summary.json").read_text())
    if (
        args.policy_contract in {"motivation-v2", "motivation-opportunity"}
        and summary.get("detector_comparison_scope") != "world"
    ):
        raise ValueError("motivation-v2 requires a world-scope detector catalog")
    catalog_payload = (args.frozen_dir / "object_catalog.jsonl").read_bytes()
    objects = [
        json.loads(line)
        for line in catalog_payload.decode().splitlines()
        if line
    ]
    evidence_source, detector_evidence, catalog_content_hash = validate_catalog_source(
        summary, catalog_payload, objects
    )
    metadata_only = summary.get("run_kind") == "matched_detector_metadata_capture_build"
    cost_input = (
        "single_epoch_full_tensor_metadata"
        if metadata_only
        else "single_epoch_full_tensor_values"
    )
    transition = None
    if args.base_frozen_dir is not None:
        base_summary = json.loads((args.base_frozen_dir / "summary.json").read_text())
        base_payload = (args.base_frozen_dir / "object_catalog.jsonl").read_bytes()
        base_objects = [
            json.loads(line) for line in base_payload.decode().splitlines() if line
        ]
        base_source, base_detector, base_content_hash = validate_catalog_source(
            base_summary, base_payload, base_objects
        )
        if (
            int(base_summary["world_size"]) != int(summary["world_size"])
            or int(base_summary["ranks_per_host"])
            != int(summary["ranks_per_host"])
            or base_summary.get("detector_comparison_scope")
            != summary.get("detector_comparison_scope")
        ):
            raise ValueError("base/current catalog topology or detector scope changed")
        if int(summary["epoch"]) != int(base_summary["epoch"]) + 1:
            raise ValueError("base/current catalogs must describe adjacent epochs")
        objects, transition_stats = changed_requirement_catalog(base_objects, objects)
        if not objects:
            raise ValueError("no tensor requirements changed between the two epochs")
        source_catalog_hash = str(summary["catalog_hash"])
        matrix_catalog_hash = transition_catalog_hash(
            str(base_summary["catalog_hash"]), source_catalog_hash, objects
        )
        catalog_content_hash = hashlib.sha256(
            json.dumps(objects, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        evidence_source = "adjacent_adacheck_detector_confirmed"
        detector_evidence = detector_evidence and base_detector
        cost_input = "changed_rank_path_full_tensor_values"
        transition = {
            **transition_stats,
            "base_frozen_dir": str(args.base_frozen_dir.resolve()),
            "base_epoch": int(base_summary["epoch"]),
            "current_epoch": int(summary["epoch"]),
            "base_catalog_hash": str(base_summary["catalog_hash"]),
            "current_catalog_hash": source_catalog_hash,
            "base_catalog_content_hash": base_content_hash,
            "current_catalog_content_hash": hashlib.sha256(catalog_payload).hexdigest(),
            "base_evidence_source": base_source,
        }
        atomic_json(
            args.output_dir / "changed-object-catalog.json",
            {"schema_version": 1, "transition": transition, "objects": objects},
        )
    else:
        matrix_catalog_hash = str(summary["catalog_hash"])
    workload_path = args.frozen_dir.parent / "workload.json"
    workload = json.loads(workload_path.read_text()) if workload_path.exists() else None
    matrix = build_peer_delta_matrix_from_catalog(
        objects,
        world_size=int(summary["world_size"]),
        ranks_per_host=int(summary["ranks_per_host"]),
        catalog_hash=matrix_catalog_hash,
        profile=build_profile(args),
    )
    atomic_json(args.output_dir / "pair-matrix.json", matrix.to_dict())
    global_arms = host_global_arms(matrix, objects)

    evaluations = {}
    policies = (
        MOTIVATION_POLICIES
        if args.policy_contract in {"motivation-v2", "motivation-opportunity"}
        else HISTORICAL_POLICIES
    )
    if args.policy_contract in {"motivation-v2", "motivation-opportunity"}:
        reports = motivation_policy_reports(
            matrix,
            seed=args.seed,
            constrain_failure_domains=args.policy_contract == "motivation-v2",
        )
    else:
        for policy in policies:
            grouping = peer_grouping_for_policy(matrix, policy, seed=args.seed)
            evaluations[policy] = evaluate_peer_grouping(matrix, grouping)
        reports = {
            policy: evaluation.to_dict()
            for policy, evaluation in evaluations.items()
        }
    for policy in policies:
        plan_document = dict(reports[policy])
        plan_document.update(
            {
                "catalog_hash": matrix.catalog_hash,
                "catalog_content_hash": catalog_content_hash,
                "evidence_source": evidence_source,
                "profile": matrix.profile.to_dict() if matrix.profile else None,
                "runtime_reliability_validated": False,
            }
        )
        atomic_json(args.output_dir / f"peer-plan-{policy}.json", plan_document)

    baseline_name = (
        "adacheck_style"
        if args.policy_contract in {"motivation-v2", "motivation-opportunity"}
        else "fixed"
    )
    best_name = (
        "redundancy_max"
        if args.policy_contract in {"motivation-v2", "motivation-opportunity"}
        else "profile_exact"
    )
    fixed = reports[baseline_name]
    best = reports[best_name]
    delta_gap = reduction_percent(
        int(fixed["pair_local_transfer_bytes"]),
        int(best["pair_local_transfer_bytes"]),
    )
    makespan_gap = reduction_percent(
        float(fixed["makespan_proxy"]), float(best["makespan_proxy"])
    )
    threshold_pass = max(delta_gap, makespan_gap) >= args.opportunity_threshold_percent
    atomic_json(
        args.output_dir / "peer-plan-host_global.json",
        {
            "schema_version": 1,
            "policy": "host_global",
            "catalog_hash": matrix.catalog_hash,
            "catalog_content_hash": catalog_content_hash,
            "evidence_source": evidence_source,
            "profile": matrix.profile.to_dict() if matrix.profile else None,
            "first_to_second_value_ids": list(
                matrix.first_to_second_host_global_value_ids
            ),
            "second_to_first_value_ids": list(
                matrix.second_to_first_host_global_value_ids
            ),
            "transfer_bytes": matrix.host_global_transfer_bytes,
            "runtime_reliability_validated": False,
        },
    )
    runtime_eligible = False
    result = {
        "schema_version": 1,
        "run_kind": (
            "peer_delta_offline_activity_diagnostic"
            if transition is not None
            else "peer_delta_offline_opportunity"
        ),
        "cost_input": cost_input,
        "frozen_dir": str(args.frozen_dir.resolve()),
        "catalog_hash": matrix.catalog_hash,
        "catalog_content_hash": catalog_content_hash,
        "evidence_source": evidence_source,
        "detector_evidence": detector_evidence,
        "detector_comparison_scope": summary.get("detector_comparison_scope"),
        "detector_evidence_roles": summary.get("detector_evidence_roles"),
        "conservative_unique_roles": summary.get("conservative_unique_roles"),
        "recoverable_payload_present": summary.get(
            "recoverable_payload_present", True
        ),
        "transition": transition,
        "workload": workload,
        "world_size": matrix.world_size,
        "ranks_per_host": matrix.ranks_per_host,
        "profile": matrix.profile.to_dict() if matrix.profile is not None else None,
        "policy_contract": args.policy_contract,
        "policy_order": list(policies),
        "policies": reports,
        "host_global_control_bytes": int(fixed["host_global_transfer_bytes"]),
        "host_global_arms": global_arms,
        "m1_delta_reduction_percent": delta_gap,
        "m1_makespan_reduction_percent": makespan_gap,
        "opportunity_threshold_percent": args.opportunity_threshold_percent,
        "exceeds_opportunity_threshold": threshold_pass,
        "adacheck_payload_faithful": False,
        "eligible_for_runtime_gate": runtime_eligible,
        "passes_single_catalog_gate": runtime_eligible and threshold_pass,
        "limitations": [
            (
                "changed full-tensor requirements, not AdaCheck gradient records or "
                "serialized NIC wire bytes"
                if transition is not None
                else "full-state logical tensor bytes, not AdaCheck gradient/direct-state "
                "encoding or serialized NIC wire bytes"
            ),
            "detector provenance does not make this offline cost runtime eligible",
            "P99 is not predicted when profile is absent",
            "failure sets are statically covered, not runtime validated",
        ],
    }
    if args.policy_contract == "historical":
        result.update(
            {
                "profile_exact_delta_reduction_percent": delta_gap,
                "profile_exact_makespan_reduction_percent": makespan_gap,
            }
        )
    if metadata_only:
        result["limitations"].append(
            "metadata-only capture supports redundancy opportunity analysis but not "
            "checkpoint reconstruction or recovery"
        )
    if args.policy_contract == "motivation-v2":
        result["limitations"].append(
            "GeminiStyleAdjacent logical bytes are diagnostic only: node-major adjacent "
            "ranks are co-located and do not tolerate one host failure"
        )
    if args.policy_contract == "motivation-opportunity":
        result["limitations"].append(
            "rank-pair opportunity deliberately omits host-failure recovery constraints; "
            "host-infeasible policies require the later object-recovery/M2 design"
        )
    if transition is not None:
        result["limitations"].extend(
            [
                "activity diagnostic only; it does not model the dense gradients "
                "transmitted by AdaCheck and cannot enter the runtime gate",
                "the transition filter conservatively counts only sharing among "
                "changed requirements; it does not credit reuse from an unchanged "
                "peer tensor",
            ]
        )
    if not detector_evidence:
        result["limitations"].append(
            "exact-value oracle is an upper bound and cannot pass the AdaCheck gate"
        )
    atomic_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
