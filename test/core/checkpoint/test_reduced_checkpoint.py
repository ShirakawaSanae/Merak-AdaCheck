import io
from dataclasses import replace

import pytest
import torch

from Merak.core.checkpoint.catalog import (
    TensorRedundancyEvidence,
    checkpoint_path,
    serialize_checkpoint,
)
from Merak.core.checkpoint.reduced import (
    build_reduced_checkpoint,
    confirm_detector_evidence,
    plan_replication_cost,
)
from Merak.core.checkpoint.redundancy_checker import RedundancyChecker


def _rank_state(rank):
    return {
        "model": {
            "shared": torch.arange(8, dtype=torch.float32),
            "host_shard": torch.full((4,), rank // 2, dtype=torch.float16),
        },
        "optimizer": {
            "step": torch.tensor(7, dtype=torch.int64),
        },
        "rng": torch.tensor([rank], dtype=torch.uint8),
        "iteration": 7,
    }


def test_reduced_checkpoint_reconstructs_every_rank_without_aliasing():
    states = {rank: _rank_state(rank) for rank in range(4)}
    reduced = build_reduced_checkpoint(states)
    buffer = io.BytesIO()
    torch.save(reduced, buffer, _use_new_zipfile_serialization=False)
    buffer.seek(0)
    reduced = torch.load(buffer, map_location="cpu", weights_only=False)

    assert reduced.reduced_logical_bytes < reduced.full_logical_bytes
    for rank, expected in states.items():
        restored = reduced.reconstruct_rank(rank)
        assert serialize_checkpoint(restored).objects == serialize_checkpoint(expected).objects

    restored_zero = reduced.reconstruct_rank(0)
    restored_one = reduced.reconstruct_rank(1)
    restored_zero["model"]["shared"].zero_()
    assert torch.equal(
        restored_one["model"]["shared"], states[1]["model"]["shared"]
    )


def test_reconstruction_preserves_rank_local_tied_storage_only():
    tied = torch.arange(8, dtype=torch.float32)
    independent = tied.clone()
    reduced = build_reduced_checkpoint(
        {
            0: {
                "model": {
                    "tied_a": tied,
                    "tied_b": tied.view_as(tied),
                    "independent": independent,
                }
            }
        }
    )
    buffer = io.BytesIO()
    torch.save(reduced, buffer, _use_new_zipfile_serialization=False)
    buffer.seek(0)
    reduced = torch.load(buffer, map_location="cpu", weights_only=False)

    restored = reduced.reconstruct_rank(0)["model"]
    assert restored["tied_a"].data_ptr() == restored["tied_b"].data_ptr()
    assert restored["tied_a"].data_ptr() != restored["independent"].data_ptr()
    restored["tied_a"].zero_()
    assert torch.count_nonzero(restored["tied_b"]) == 0
    assert torch.equal(restored["independent"], independent)


def test_reduction_rejects_distinct_views_of_shared_storage():
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    with pytest.raises(ValueError, match="distinct views"):
        build_reduced_checkpoint(
            {0: {"model": {"matrix": tensor, "transpose": tensor.T}}}
        )


def test_replication_cost_is_failure_domain_aware_and_deterministic():
    reduced = build_reduced_checkpoint({rank: _rank_state(rank) for rank in range(4)})
    rank_to_domain = {0: "node17", 1: "node17", 2: "node1", 3: "node1"}

    first = plan_replication_cost(
        reduced,
        rank_to_domain,
        ["node17", "node1"],
        replica_count=2,
        tolerated_failures=1,
    )
    second = plan_replication_cost(
        reduced,
        rank_to_domain,
        ["node1", "node17"],
        replica_count=2,
        tolerated_failures=1,
    )

    assert first == second
    assert first.reduced_replication_bytes < first.full_replication_bytes
    assert first.logical_reduction_ratio > 0
    assert first.unrecoverable_values(["node17"]) == ()
    assert first.unrecoverable_values(["node1"]) == ()
    assert first.unrecoverable_values(["node17", "node1"])
    assert first.cross_domain_bytes == 20


def test_replication_cost_rejects_an_unachievable_failure_budget():
    reduced = build_reduced_checkpoint({0: _rank_state(0)})

    with pytest.raises(ValueError, match="must exceed"):
        plan_replication_cost(
            reduced,
            {0: "node17"},
            ["node17", "node1"],
            replica_count=1,
            tolerated_failures=1,
        )

    with pytest.raises(ValueError, match="only 2 are available"):
        plan_replication_cost(
            reduced,
            {0: "node17"},
            ["node17", "node1"],
            replica_count=3,
            tolerated_failures=1,
        )

    with pytest.raises(ValueError, match="finite and non-negative"):
        plan_replication_cost(
            reduced,
            {0: "node17"},
            ["node17"],
            replica_count=1,
            tolerated_failures=0,
            path_cost={("node17", "node17"): -1},
        )


def _detector_evidence(rank, shared_index, local_index, shared_locations):
    return (
        TensorRedundancyEvidence(
            rank=rank,
            role="model",
            path=checkpoint_path(("model", "shared")),
            local_index=shared_index,
            equivalent_locations=shared_locations,
        ),
        TensorRedundancyEvidence(
            rank=rank,
            role="model",
            path=checkpoint_path(("model", "local")),
            local_index=local_index,
            equivalent_locations=((rank, local_index),),
        ),
    )


def test_detector_evidence_controls_cross_rank_reduction():
    states = {
        0: {
            "model": {"shared": torch.arange(4), "local": torch.tensor([0])},
            "rng": torch.ones(2),
        },
        1: {
            "model": {"shared": torch.arange(4), "local": torch.tensor([1])},
            "rng": torch.ones(2),
        },
    }
    shared_locations = ((0, 0), (1, 4))
    evidence = (
        *_detector_evidence(0, 0, 1, shared_locations),
        *_detector_evidence(1, 4, 5, shared_locations),
    )

    reduced = build_reduced_checkpoint(states, evidence)

    shared = [item for item in reduced.objects if len(item.source_ranks) == 2]
    assert len(shared) == 1
    assert shared[0].role == "model"
    assert torch.equal(reduced.reconstruct_rank(0)["rng"], states[0]["rng"])
    assert torch.equal(reduced.reconstruct_rank(1)["rng"], states[1]["rng"])


def test_detector_evidence_fails_closed_on_false_match_or_missing_path():
    states = {
        0: {"model": {"shared": torch.tensor([0])}},
        1: {"model": {"shared": torch.tensor([1])}},
    }
    locations = ((0, 0), (1, 0))
    evidence = (
        TensorRedundancyEvidence(
            0, "model", checkpoint_path(("model", "shared")), 0, locations
        ),
        TensorRedundancyEvidence(
            1, "model", checkpoint_path(("model", "shared")), 0, locations
        ),
    )
    with pytest.raises(ValueError, match="contains unequal tensors"):
        build_reduced_checkpoint(states, evidence)

    with pytest.raises(ValueError, match="missing rank 1 tensor"):
        build_reduced_checkpoint(states, evidence[:1])


def test_detector_confirmation_splits_reused_hash_bucket_by_full_identity():
    path_a = checkpoint_path(("optimizer", "a"))
    path_b = checkpoint_path(("optimizer", "b"))
    states = {
        0: {"optimizer": {"a": torch.tensor([1]), "b": torch.tensor([2])}},
        1: {"optimizer": {"a": torch.tensor([1]), "b": torch.tensor([3])}},
    }
    bucket = ((0, 7), (1, 7))
    evidence = tuple(
        TensorRedundancyEvidence(rank, "optimizer", path, 7, bucket)
        for rank in (0, 1)
        for path in (path_a, path_b)
    )

    confirmed, stats = confirm_detector_evidence(states, evidence)
    reduced = build_reduced_checkpoint(states, confirmed)

    shared = [item for item in reduced.objects if item.source_ranks == (0, 1)]
    assert len(shared) == 1
    assert stats.candidate_pairs == 6
    assert stats.confirmed_pairs == 1
    assert stats.rejected_pairs == 5
    assert reduced.reconstruct_rank(1)["optimizer"]["b"].item() == 3


def test_reduced_validation_rejects_an_incomplete_template_graph():
    reduced = build_reduced_checkpoint({0: _rank_state(0)})
    malformed = replace(reduced, templates={0: {"iteration": 7}})

    with pytest.raises(ValueError, match="templates and object requirements"):
        malformed.validate()


def test_redundancy_checker_exports_path_and_packed_index_evidence():
    checker = RedundancyChecker(zero_stage=0)
    checker.model_state = {
        "layer": {"weight": torch.ones(2)},
        "version": 3,
        "scalar_tuple": (0.9, 0.999),
    }
    checker.optimizer_state = {"step": torch.tensor(4)}
    checker.model_state_index_mapping = {
        "layer": {"weight": 0},
        "version": 1,
        "scalar_tuple": 3,
    }
    checker.optimizer_index_mapping = {"step": 2}
    checker.redundancy_results_for_model_state = {
        "layer": {"weight": [[0, 0], [1, 7]]},
        "version": [],
        "scalar_tuple": [],
    }
    checker.redundancy_results_for_optimizer = {"step": []}

    evidence = checker.tensor_redundancy_evidence(rank=0)

    assert [item.path for item in evidence] == [
        checkpoint_path(("model", "layer", "weight")),
        checkpoint_path(("optimizer", "step")),
    ]
    assert evidence[0].equivalent_locations == ((0, 0), (1, 7))
    assert evidence[1].equivalent_locations == ((0, 2),)
