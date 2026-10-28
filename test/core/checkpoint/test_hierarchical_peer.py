import torch

from Merak.core.checkpoint.catalog import TensorRedundancyEvidence
from Merak.core.checkpoint.hierarchical_peer import (
    build_hierarchical_study,
    host_matchings,
)
from Merak.core.checkpoint.reduced import build_reduced_checkpoint
from Merak.core.checkpoint.peer_delta_runtime import build_peer_delta_protection_plan
from examples.checkpoint_grouping.build_reduced_capture import stable_evidence


def _four_host_reduced():
    # Hosts 0/2 and 1/3 share their large tensors; sequential host pairing does not.
    states = {}
    host_values = (1.0, 2.0, 1.0, 2.0)
    for host, value in enumerate(host_values):
        for local_rank in range(2):
            rank = host * 2 + local_rank
            states[rank] = {
                "model": {
                    "weight": torch.full((32,), value),
                    "local": torch.tensor([local_rank]),
                    "rank": torch.tensor([rank]),
                }
            }
    return build_reduced_checkpoint(states)


def test_four_hosts_have_three_perfect_matchings():
    assert host_matchings(4) == (
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    )


def test_hierarchical_study_is_exact_and_monotone_for_peer_bytes():
    study = build_hierarchical_study(_four_host_reduced(), ranks_per_host=2)
    fixed = study["sequential_fixed"]
    host = study["host_only"]
    rank = study["rank_only"]
    joint = study["hierarchical_joint"]

    assert host.host_pairs == ((0, 2), (1, 3))
    assert joint.peer_local_transfer_bytes <= min(
        host.peer_local_transfer_bytes, rank.peer_local_transfer_bytes
    )
    assert min(host.peer_local_transfer_bytes, rank.peer_local_transfer_bytes) <= (
        fixed.peer_local_transfer_bytes
    )
    assert study["host_pooled_opt"].host_pooled_transfer_bytes <= (
        fixed.host_pooled_transfer_bytes
    )


def test_host_pooled_bytes_do_not_depend_on_rank_permutation():
    study = build_hierarchical_study(_four_host_reduced(), ranks_per_host=2)
    fixed = study["sequential_fixed"]
    rank = study["rank_only"]
    assert rank.host_pairs == fixed.host_pairs
    assert rank.host_pooled_transfer_bytes == fixed.host_pooled_transfer_bytes


def test_offline_bytes_equal_runtime_plan_bytes_for_both_availability_scopes():
    reduced = _four_host_reduced()
    study = build_hierarchical_study(reduced, ranks_per_host=2)
    for policy in ("sequential_fixed", "host_only", "rank_only", "hierarchical_joint"):
        evaluation = study[policy]
        plan = build_peer_delta_protection_plan(
            reduced,
            evaluation.grouping,
            ranks_per_host=2,
            availability_scope="peer",
        )
        assert plan.pair_local_transfer_bytes == evaluation.peer_local_transfer_bytes
    for policy in (
        "host_pooled_fixed",
        "host_pooled_publisher",
        "host_pooled_opt",
        "host_pooled_joint",
    ):
        evaluation = study[policy]
        plan = build_peer_delta_protection_plan(
            reduced,
            evaluation.grouping,
            ranks_per_host=2,
            availability_scope="host",
            balance_publishers=policy not in {
                "host_pooled_fixed",
                "host_pooled_opt",
            },
        )
        assert plan.pair_local_transfer_bytes == evaluation.host_pooled_transfer_bytes
        assert (
            plan.maximum_remote_publisher_bytes
            == evaluation.host_pooled_max_publisher_bytes
        )


def test_odd_host_count_is_rejected():
    try:
        host_matchings(3)
    except ValueError as error:
        assert "even" in str(error)
    else:
        raise AssertionError("odd host count was accepted")


def test_detector_stability_keeps_components_and_splits_changed_relations():
    first = (
        TensorRedundancyEvidence(0, "model", "model.a", 0, ((0, 0), (1, 0))),
        TensorRedundancyEvidence(1, "model", "model.a", 0, ((0, 0), (1, 0))),
        TensorRedundancyEvidence(0, "model", "model.b", 1, ((0, 1), (1, 1))),
        TensorRedundancyEvidence(1, "model", "model.b", 1, ((0, 1), (1, 1))),
    )
    second = (
        TensorRedundancyEvidence(0, "model", "model.a", 4, ((0, 4), (1, 4))),
        TensorRedundancyEvidence(1, "model", "model.a", 4, ((0, 4), (1, 4))),
        TensorRedundancyEvidence(0, "model", "model.b", 5, ((0, 5),)),
        TensorRedundancyEvidence(1, "model", "model.b", 6, ((1, 6),)),
    )
    stable, summary = stable_evidence(first, second)
    by_path = {(item.rank, item.path): item for item in stable}
    assert len(by_path[(0, "model.a")].equivalent_locations) == 2
    assert len(by_path[(0, "model.b")].equivalent_locations) == 1
    assert len(by_path[(1, "model.b")].equivalent_locations) == 1
    assert summary == {
        "stable_components": 1,
        "stable_records": 2,
        "singleton_fallback_records": 2,
    }
