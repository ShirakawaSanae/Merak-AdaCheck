#!/usr/bin/env python3
"""Compare a restored Merak continuation with an uninterrupted reference run."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Mapping, Sequence


LOSS = re.compile(r"steps: (\d+) loss: ([^ ]+)")


def digests(root: Path, phase: str) -> dict[int, dict]:
    values = {}
    for path in sorted((root / "digests" / phase).glob("rank_*.json")):
        value = json.loads(path.read_text())
        values[int(value["rank"])] = value
    if not values:
        raise ValueError(f"missing {phase} state digests in {root}")
    return values


def losses(path: Path) -> dict[int, str]:
    return {int(step): loss for step, loss in LOSS.findall(path.read_text())}


def compare_logged_losses(
    recovered: Mapping[int, str],
    reference: Mapping[int, str],
    *,
    expected_steps: Sequence[int],
    rtol: float,
    atol: float,
) -> dict:
    expected = [int(step) for step in expected_steps]
    recovered_steps_exact = sorted(recovered) == expected
    reference_steps_present = all(step in reference for step in expected)
    present = recovered_steps_exact and reference_steps_present
    absolute_errors = []
    relative_errors = []
    exact = present
    finite = present
    close = present
    for step in expected:
        if step not in recovered or step not in reference:
            continue
        actual = float(recovered[step])
        target = float(reference[step])
        if not math.isfinite(actual) or not math.isfinite(target):
            finite = False
            close = False
            absolute_errors.append(math.inf)
            relative_errors.append(math.inf)
            continue
        difference = abs(actual - target)
        scale = abs(target)
        absolute_errors.append(difference)
        relative_errors.append(difference / scale if scale else difference)
        exact = exact and recovered[step] == reference[step]
        close = close and difference <= atol + rtol * scale
    return {
        "expected_steps": expected,
        "expected_steps_present": present,
        "recovered_steps_exact": recovered_steps_exact,
        "reference_steps_present": reference_steps_present,
        "all_finite": finite,
        "exact": exact,
        "numerically_close": close and finite,
        "max_absolute_error": max(absolute_errors, default=math.inf),
        "max_relative_error": max(relative_errors, default=math.inf),
    }


def component_mismatches(actual: dict[int, dict], expected: dict[int, dict]) -> dict:
    names = ("model", "optimizer", "lr_scheduler")
    return {
        name: [
            rank
            for rank in sorted(set(actual) | set(expected))
            if actual.get(rank, {}).get("component_checksums", {}).get(name)
            != expected.get(rank, {}).get("component_checksums", {}).get(name)
        ]
        for name in names
    }


def materialized_oracles(root: Path) -> dict[int, dict]:
    values = {}
    for path in sorted((root / "checkpoint" / "metrics").glob("rank_*.json")):
        value = json.loads(path.read_text())
        oracle = value.get("materialized_state_oracle")
        if oracle is not None:
            values[int(value["rank"])] = oracle
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recovered", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--loss-rtol", type=float, default=1e-4)
    parser.add_argument("--loss-atol", type=float, default=1e-4)
    parser.add_argument("--expected-start-step", type=int, required=True)
    parser.add_argument("--expected-end-step", type=int, required=True)
    args = parser.parse_args()
    if args.loss_rtol < 0 or args.loss_atol < 0:
        parser.error("loss tolerances must be non-negative")
    if args.expected_end_step < args.expected_start_step:
        parser.error("expected loss step interval is empty")

    recovered = digests(args.recovered, "restore")
    reference = digests(args.reference, "train")
    restored_boundary = digests(args.recovered, "restored-boundary")
    checkpoint_boundary = digests(args.recovered, "train")
    ranks = sorted(set(recovered) | set(reference))
    mismatches = [
        rank
        for rank in ranks
        if recovered.get(rank, {}).get("state_checksum")
        != reference.get(rank, {}).get("state_checksum")
    ]
    recovered_losses = losses(args.recovered / "logs" / "restore" / "node-0.log")
    reference_losses = losses(args.reference / "logs" / "train" / "node-0.log")
    expected_steps = list(
        range(args.expected_start_step, args.expected_end_step + 1)
    )
    loss_comparison = compare_logged_losses(
        recovered_losses,
        reference_losses,
        expected_steps=expected_steps,
        rtol=args.loss_rtol,
        atol=args.loss_atol,
    )
    checkpoint_oracles = materialized_oracles(args.recovered)
    expected_ranks = list(range(16))
    rank_sets = {
        "recovered_final": sorted(recovered),
        "reference_final": sorted(reference),
        "restored_boundary": sorted(restored_boundary),
        "checkpoint_boundary": sorted(checkpoint_boundary),
        "materialized_oracle": sorted(checkpoint_oracles),
    }
    rank_sets_complete = all(
        ranks_for_source == expected_ranks for ranks_for_source in rank_sets.values()
    )
    boundary_mismatches = component_mismatches(
        restored_boundary, checkpoint_boundary
    )
    final_component_mismatches = component_mismatches(recovered, reference)
    result = {
        "schema_version": 2,
        "recovered_run": str(args.recovered),
        "reference_run": str(args.reference),
        "rank_count": len(ranks),
        "expected_ranks": expected_ranks,
        "rank_sets": rank_sets,
        "all_rank_sets_complete": rank_sets_complete,
        "mismatched_state_ranks": mismatches,
        "state_bitwise_equal": len(ranks) == 16 and not mismatches,
        "boundary_component_mismatched_ranks": boundary_mismatches,
        "boundary_state_bitwise_equal": all(
            not values for values in boundary_mismatches.values()
        ),
        "final_component_mismatched_ranks": final_component_mismatches,
        "common_loss_steps": expected_steps,
        "recovered_losses": {
            str(step): recovered_losses[step]
            for step in expected_steps
            if step in recovered_losses
        },
        "reference_losses": {
            str(step): reference_losses[step]
            for step in expected_steps
            if step in reference_losses
        },
        "logged_loss_comparison": loss_comparison,
        "logged_losses_equal": loss_comparison["exact"],
        "logged_losses_numerically_close": loss_comparison["numerically_close"],
        "checkpoint_boundary_numerically_close": len(checkpoint_oracles) == 16
        and all(
            oracle.get("all_numerically_close", False)
            for oracle in checkpoint_oracles.values()
        ),
    }
    result["numerical_continuation_valid"] = (
        result["all_rank_sets_complete"]
        and result["checkpoint_boundary_numerically_close"]
        and result["logged_losses_numerically_close"]
    )
    result["bitwise_continuation_valid"] = (
        result["state_bitwise_equal"] and result["logged_losses_equal"]
    )
    result["acceptance_contract"] = {
        "name": "checkpoint-boundary-numerical-plus-finite-continuation-loss-v2",
        "expected_loss_steps": expected_steps,
        "loss_rtol": args.loss_rtol,
        "loss_atol": args.loss_atol,
    }
    result["valid"] = result["numerical_continuation_valid"]
    destination = args.output or args.recovered / "recovery-oracle.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(destination)
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
