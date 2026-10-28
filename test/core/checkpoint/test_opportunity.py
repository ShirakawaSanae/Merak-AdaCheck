from itertools import permutations

import pytest

from Merak.core.checkpoint.opportunity import (
    AsymmetricCheckpointPlan,
    PlanRequirement,
    PlannedCheckpointObject,
    RankGrouping,
    evaluate_grouping,
    exact_oracle_grouping,
    fixed_grouping,
    random_grouping,
)


def _object(rank: int, logical_bytes: int) -> PlannedCheckpointObject:
    return PlannedCheckpointObject(
        object_id=f"object-{rank}",
        role="model",
        dtype="torch.uint8",
        shape=(logical_bytes,),
        logical_bytes=logical_bytes,
        checksum=f"checksum-{rank}",
        holders=(rank,),
        requirements=(
            PlanRequirement(f"model.tensor_{rank}", rank, ((0, logical_bytes),)),
        ),
    )


def _plan(payloads=(1, 100, 3, 80, 90, 2, 70, 4)) -> AsymmetricCheckpointPlan:
    return AsymmetricCheckpointPlan(
        name="test-plan",
        producer="test",
        world_size=8,
        ranks_per_host=2,
        objects=tuple(_object(rank, size) for rank, size in enumerate(payloads)),
        metadata={"kind": "test"},
    )


def _host_matchings():
    return (
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    )


def _all_host_disjoint_groupings(plan):
    for host_pairs in _host_matchings():
        choices = []
        for first_host, second_host in host_pairs:
            first = tuple(
                range(
                    first_host * plan.ranks_per_host,
                    (first_host + 1) * plan.ranks_per_host,
                )
            )
            second = tuple(
                range(
                    second_host * plan.ranks_per_host,
                    (second_host + 1) * plan.ranks_per_host,
                )
            )
            choices.append(
                tuple(tuple(zip(first, order)) for order in permutations(second))
            )
        for first_groups in choices[0]:
            for second_groups in choices[1]:
                yield RankGrouping(
                    "candidate", tuple(sorted(first_groups + second_groups))
                )


def test_all_legal_policies_preserve_primary_protection_cost_and_failure_budget():
    plan = _plan()
    groupings = (
        fixed_grouping(plan),
        random_grouping(plan, 17),
        exact_oracle_grouping(plan),
    )
    evaluations = [evaluate_grouping(plan, grouping) for grouping in groupings]

    assert {item.cross_domain_transfer_bytes for item in evaluations} == {350}
    assert {item.global_full_duplex_lower_bound_bytes for item in evaluations} == {
        100
    }
    assert all(item.underprotected_bytes_after == 0 for item in evaluations)
    assert all(len(item.transfers) == len(plan.objects) for item in evaluations)


def test_exact_oracle_matches_exhaustive_small_host_pair_search():
    plan = _plan()
    oracle = evaluate_grouping(plan, exact_oracle_grouping(plan))
    exhaustive = min(
        (evaluate_grouping(plan, grouping) for grouping in _all_host_disjoint_groupings(plan)),
        key=lambda item: item.score(),
    )

    assert oracle.score() == exhaustive.score()


def test_random_and_plan_hashes_are_deterministic():
    plan = _plan()

    assert plan.plan_hash == _plan().plan_hash
    assert random_grouping(plan, 29) == random_grouping(plan, 29)
    assert random_grouping(plan, 29).plan_hash == random_grouping(plan, 29).plan_hash
    assert random_grouping(plan, 29) != random_grouping(plan, 31)


def test_evaluator_rejects_same_host_peers():
    plan = _plan()
    grouping = RankGrouping(
        "invalid",
        ((0, 1), (2, 3), (4, 5), (6, 7)),
    )

    with pytest.raises(ValueError, match="different hosts"):
        evaluate_grouping(plan, grouping)


def test_plan_rejects_requirements_that_disagree_with_holders():
    malformed = PlannedCheckpointObject(
        object_id="malformed",
        role="model",
        dtype="torch.float32",
        shape=(1,),
        logical_bytes=4,
        checksum="checksum",
        holders=(0, 2),
        requirements=(PlanRequirement("model.value", 0, ((0, 1),)),),
    )

    with pytest.raises(ValueError, match="requirements and holders"):
        AsymmetricCheckpointPlan(
            name="bad",
            producer="test",
            world_size=8,
            ranks_per_host=2,
            objects=(malformed,),
            metadata={},
        )


def test_already_cross_host_objects_require_no_transfer():
    shared = PlannedCheckpointObject(
        object_id="shared",
        role="model",
        dtype="torch.float32",
        shape=(1,),
        logical_bytes=4,
        checksum="shared-checksum",
        holders=(0, 2),
        requirements=(
            PlanRequirement("model.shared", 0, ((0, 1),)),
            PlanRequirement("model.shared", 2, ((0, 1),)),
        ),
    )
    plan = AsymmetricCheckpointPlan(
        name="shared-plan",
        producer="test",
        world_size=8,
        ranks_per_host=2,
        objects=(shared,),
        metadata={},
    )

    evaluation = evaluate_grouping(plan, fixed_grouping(plan))

    assert evaluation.cross_domain_transfer_bytes == 0
    assert evaluation.initially_underprotected_bytes == 0
    assert evaluation.transfers == ()
