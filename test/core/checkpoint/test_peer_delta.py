from itertools import permutations

import pytest
import torch

from Merak.core.checkpoint.matched import reduced_catalog_hash
from Merak.core.checkpoint.opportunity import RankGrouping
from Merak.core.checkpoint.peer_delta import (
    FORMAL_PEER_GROUPING_POLICIES,
    PeerResourceProfile,
    build_peer_delta_matrix,
    build_peer_delta_matrix_from_catalog,
    evaluate_peer_grouping,
    peer_grouping_for_policy,
)
from Merak.core.checkpoint.reduced import build_reduced_checkpoint


def _reduced():
    # Fixed pairs mismatch the large model tensors; the crossed pairs reuse them.
    states = {
        0: {"model": {"weight": torch.ones(32), "rank": torch.tensor([0])}},
        1: {"model": {"weight": torch.full((16,), 2.0), "rank": torch.tensor([1])}},
        2: {"model": {"weight": torch.full((16,), 2.0), "rank": torch.tensor([2])}},
        3: {"model": {"weight": torch.ones(32), "rank": torch.tensor([3])}},
    }
    return build_reduced_checkpoint(states)


def _matrix(profile=None):
    reduced = _reduced()
    return build_peer_delta_matrix(
        reduced,
        ranks_per_host=2,
        catalog_hash=reduced_catalog_hash(reduced),
        profile=profile,
    )


def test_peer_delta_counts_overlap_and_host_global_control():
    matrix = _matrix()
    mismatched = matrix.edge(0, 2)
    matched = matrix.edge(0, 3)

    assert mismatched.overlap_bytes == 0
    assert matched.overlap_bytes == 32 * torch.tensor([], dtype=torch.float32).element_size()
    assert matched.total_transfer_bytes < mismatched.total_transfer_bytes

    fixed = evaluate_peer_grouping(matrix, peer_grouping_for_policy(matrix, "fixed"))
    optimal = evaluate_peer_grouping(
        matrix, peer_grouping_for_policy(matrix, "redundancy_max")
    )
    assert optimal.grouping.groups == ((0, 3), (1, 2))
    assert optimal.pair_local_transfer_bytes < fixed.pair_local_transfer_bytes
    assert optimal.host_global_transfer_bytes <= optimal.pair_local_transfer_bytes
    assert optimal.static_failure_sets_covered == ((0,), (1,))
    assert optimal.runtime_failure_sets_validated == ()


def test_formal_peer_policy_registry_excludes_availability_controls_and_greedy():
    matrix = _matrix()
    assert FORMAL_PEER_GROUPING_POLICIES == (
        "fixed",
        "random",
        "redundancy_max",
        "profile_exact",
    )
    for retired_or_diagnostic in ("greedy", "host_global", "exact"):
        with pytest.raises(ValueError, match="peer policy must"):
            peer_grouping_for_policy(matrix, retired_or_diagnostic)


def test_exact_profile_policy_matches_exhaustive_search():
    matrix = _matrix(
        PeerResourceProfile(
            first_to_second_bytes_per_second=100,
            second_to_first_bytes_per_second=20,
            first_service_bytes_per_second=40,
            second_service_bytes_per_second=80,
        )
    )
    actual = evaluate_peer_grouping(
        matrix, peer_grouping_for_policy(matrix, "profile_exact")
    )
    candidates = []
    for order in permutations(matrix.second_ranks):
        grouping = RankGrouping(
            "candidate", tuple(zip(matrix.first_ranks, order))
        )
        evaluation = evaluate_peer_grouping(matrix, grouping)
        candidates.append(
            (
                evaluation.makespan_proxy,
                evaluation.pair_local_transfer_bytes,
                evaluation.maximum_pair_peak_bytes,
                grouping.groups,
            )
        )
    assert (
        actual.makespan_proxy,
        actual.pair_local_transfer_bytes,
        actual.maximum_pair_peak_bytes,
        actual.grouping.groups,
    ) == min(candidates)


def test_random_policy_is_deterministic_and_all_policies_are_recoverable():
    matrix = _matrix()
    first = peer_grouping_for_policy(matrix, "random", seed=17)
    assert first == peer_grouping_for_policy(matrix, "random", seed=17)
    for policy in ("fixed", "random", "redundancy_max", "profile_exact"):
        grouping = peer_grouping_for_policy(matrix, policy, seed=17)
        evaluation = evaluate_peer_grouping(matrix, grouping)
        assert evaluation.planned_tolerated_host_failures == 1


def test_matrix_rejects_invalid_topology_and_profile():
    reduced = _reduced()
    with pytest.raises(ValueError, match="exactly two"):
        build_peer_delta_matrix(
            reduced, ranks_per_host=1, catalog_hash="catalog"
        )
    with pytest.raises(ValueError, match="positive"):
        PeerResourceProfile(0, 1)
    with pytest.raises(ValueError, match="finite"):
        PeerResourceProfile(float("nan"), 1)


def test_profile_round_trip_preserves_infinite_service_rates():
    profile = PeerResourceProfile(10.0, 20.0)
    assert PeerResourceProfile.from_dict(profile.to_dict()) == profile


def test_json_catalog_adapter_matches_reduced_adapter():
    reduced = _reduced()
    catalog = [
        {
            "value_id": item.value_id,
            "logical_bytes": item.logical_bytes,
            "source_ranks": list(item.source_ranks),
            "requirements": [
                {"rank": requirement.rank} for requirement in item.requirements
            ],
        }
        for item in reduced.objects
    ]
    direct = _matrix()
    adapted = build_peer_delta_matrix_from_catalog(
        catalog,
        world_size=4,
        ranks_per_host=2,
        catalog_hash=reduced_catalog_hash(reduced),
    )
    assert adapted == direct


def test_exact_solver_collapses_equal_cost_matchings():
    states = {
        rank: {"model": {"rank": torch.tensor([rank])}}
        for rank in range(8)
    }
    reduced = build_reduced_checkpoint(states)
    matrix = build_peer_delta_matrix(
        reduced,
        ranks_per_host=4,
        catalog_hash=reduced_catalog_hash(reduced),
    )
    grouping = peer_grouping_for_policy(matrix, "profile_exact")
    assert grouping.groups == ((0, 4), (1, 5), (2, 6), (3, 7))


def test_host_global_control_counts_a_shared_source_object_once():
    states = {
        0: {"model": {"shared": torch.ones(8)}},
        1: {"model": {"shared": torch.ones(8)}},
        2: {"model": {"other": torch.zeros(1)}},
        3: {"model": {"other": torch.zeros(1)}},
    }
    reduced = build_reduced_checkpoint(states)
    matrix = build_peer_delta_matrix(
        reduced,
        ranks_per_host=2,
        catalog_hash=reduced_catalog_hash(reduced),
    )
    evaluation = evaluate_peer_grouping(
        matrix, peer_grouping_for_policy(matrix, "fixed")
    )
    assert evaluation.host_global_transfer_bytes == (8 + 1) * 4
    assert evaluation.pair_local_transfer_bytes == 2 * (8 + 1) * 4
