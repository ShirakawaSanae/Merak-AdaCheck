import json

import pytest

from Merak.core.checkpoint.peer_delta import build_peer_delta_matrix_from_catalog
from tools.analyze_peer_delta_results import parse_named_path, percentile
from tools.analyze_peer_delta_opportunity import (
    changed_requirement_catalog,
    host_global_arms,
    motivation_policy_reports,
)
from tools.analyze_peer_delta_formal import aggregate, load_formal, mean_ci95
from tools.analyze_peer_delta_recovery import classify_noninferiority
from tools.analyze_four_node_peer_delta import paired_bootstrap, treatment_baseline


def test_peer_delta_analysis_percentile_uses_nearest_rank():
    assert percentile([4, 1, 3, 2], 50) == 2
    assert percentile([4, 1, 3, 2], 99) == 4
    with pytest.raises(ValueError):
        percentile([], 50)


def test_peer_delta_analysis_named_paths_are_explicit():
    name, path = parse_named_path("bert=runs/example")
    assert name == "bert"
    assert str(path) == "runs/example"
    with pytest.raises(Exception):
        parse_named_path("runs/example")


def test_changed_requirement_catalog_filters_per_rank_path():
    def item(value_id, checksum, requirements, logical_bytes=8):
        return {
            "value_id": value_id,
            "role": "optimizer",
            "dtype": "torch.float32",
            "shape": [2],
            "logical_bytes": logical_bytes,
            "checksum": checksum,
            "source_ranks": sorted({rank for rank, _ in requirements}),
            "requirements": [
                {"rank": rank, "path": path} for rank, path in requirements
            ],
        }

    base = [
        item("base-shared", "same", [(0, "/a"), (2, "/a")]),
        item("base-left", "old-left", [(1, "/b")]),
        item("base-right", "old-right", [(3, "/b")]),
        item("base-mixed-left", "old-mixed", [(0, "/c")]),
        item("base-mixed-right", "new-mixed", [(2, "/c")]),
    ]
    current = [
        item("current-shared", "same", [(0, "/a"), (2, "/a")]),
        item("current-changed", "new", [(1, "/b"), (3, "/b")]),
        item("current-mixed", "new-mixed", [(0, "/c"), (2, "/c")]),
    ]

    changed, stats = changed_requirement_catalog(base, current)

    assert [value["value_id"] for value in changed] == [
        "current-changed",
        "current-mixed",
    ]
    assert changed[0]["source_ranks"] == [1, 3]
    assert changed[1]["source_ranks"] == [0]
    assert changed[1]["requirements"] == [{"rank": 0, "path": "/c"}]
    assert stats == {
        "total_requirement_count": 6,
        "changed_requirement_count": 3,
        "unchanged_requirement_count": 3,
        "total_requirement_logical_bytes": 48,
        "changed_requirement_logical_bytes": 24,
        "unchanged_requirement_logical_bytes": 24,
        "changed_object_count": 2,
    }


def test_changed_requirement_catalog_rejects_metadata_drift():
    base = [
        {
            "value_id": "base",
            "role": "model",
            "dtype": "torch.float32",
            "shape": [2],
            "logical_bytes": 8,
            "checksum": "old",
            "source_ranks": [0],
            "requirements": [{"rank": 0, "path": "/weight"}],
        }
    ]
    current = [{**base[0], "shape": [3], "logical_bytes": 12, "checksum": "new"}]

    with pytest.raises(ValueError, match="metadata changed"):
        changed_requirement_catalog(base, current)


def test_host_global_arms_isolate_publisher_load_at_equal_recovery_bytes():
    objects = [
        {
            "value_id": value_id,
            "logical_bytes": 8,
            "source_ranks": holders,
            "requirements": [{"rank": rank} for rank in holders],
        }
        for value_id, holders in (
            ("left-a", [0, 1]),
            ("left-b", [0, 1]),
            ("right-a", [2, 3]),
            ("right-b", [2, 3]),
            ("already-replicated", [0, 2]),
        )
    ]
    matrix = build_peer_delta_matrix_from_catalog(
        objects,
        world_size=4,
        ranks_per_host=2,
        catalog_hash="a" * 64,
    )

    result = host_global_arms(matrix, objects)

    assert result["fixed"]["total_remote_bytes"] == 32
    assert result["m1"]["total_remote_bytes"] == 32
    assert result["fixed"]["maximum_remote_publisher_bytes"] == 16
    assert result["m1"]["maximum_remote_publisher_bytes"] == 8
    assert result["fixed"]["publisher_policy"] == "first_eligible"
    assert result["m1"]["publisher_policy"] == "least_load"
    assert result["predicted_max_publisher_reduction_percent"] == 50.0
    assert result["fixed"]["plan_hash"] != result["m1"]["plan_hash"]


def test_motivation_policy_contract_keeps_literal_gemini_infeasible():
    objects = [
        {
            "value_id": f"rank-{rank}",
            "logical_bytes": 8,
            "source_ranks": [rank],
            "requirements": [{"rank": rank}],
        }
        for rank in range(4)
    ]
    matrix = build_peer_delta_matrix_from_catalog(
        objects,
        world_size=4,
        ranks_per_host=2,
        catalog_hash="b" * 64,
    )

    reports = motivation_policy_reports(matrix, seed=17)

    assert tuple(reports) == (
        "adacheck_style",
        "gemini_style",
        "random_feasible",
        "redundancy_max",
    )
    assert reports["adacheck_style"]["host_failure_feasible"] is True
    assert reports["random_feasible"]["host_failure_feasible"] is True
    assert reports["redundancy_max"]["host_failure_feasible"] is True
    assert reports["gemini_style"]["grouping"]["groups"] == [[0, 1], [2, 3]]
    assert reports["gemini_style"]["host_failure_feasible"] is False
    assert reports["gemini_style"]["planned_tolerated_host_failures"] == 0


def test_motivation_opportunity_finds_same_host_redundancy():
    objects = [
        {
            "value_id": "host-0-shared",
            "logical_bytes": 100,
            "source_ranks": [0, 1],
            "requirements": [{"rank": 0}, {"rank": 1}],
        },
        {
            "value_id": "host-1-shared",
            "logical_bytes": 100,
            "source_ranks": [2, 3],
            "requirements": [{"rank": 2}, {"rank": 3}],
        },
    ]
    matrix = build_peer_delta_matrix_from_catalog(
        objects,
        world_size=4,
        ranks_per_host=2,
        catalog_hash="c" * 64,
    )

    reports = motivation_policy_reports(
        matrix, seed=17, constrain_failure_domains=False
    )

    assert reports["adacheck_style"]["pair_local_transfer_bytes"] == 400
    assert reports["redundancy_max"]["pair_local_transfer_bytes"] == 0
    assert reports["redundancy_max"]["grouping"]["groups"] == [[0, 1], [2, 3]]
    assert reports["redundancy_max"]["host_failure_feasible"] is False


def test_formal_confidence_interval_requires_trials_not_rank_samples():
    mean, interval = mean_ci95([10.0, 12.0, 14.0])
    assert mean == 12.0
    assert interval > 0


def test_formal_analysis_rejects_non_preregistered_matrix_size(tmp_path):
    (tmp_path / "COMPLETE").write_text("complete\n")
    (tmp_path / "experiment-config.json").write_text(
        json.dumps(
            {
                "policies": ["fixed", "random", "redundancy_max", "profile_exact"],
                "repetitions": 3,
                "epochs_per_policy": 8,
            }
        )
    )

    with pytest.raises(ValueError, match="exactly 5 repetitions x 8 epochs"):
        load_formal("invalid", tmp_path, warmup_epochs=2)


def test_formal_analysis_excludes_contaminated_nic_trials():
    metrics = {
        "logical_delta_bytes_per_epoch": 100,
        "protocol_wire_bytes_per_epoch": 110,
        "rank_commit_p50_ms": 1,
        "rank_commit_p95_ms": 2,
        "rank_commit_p99_ms": 3,
        "checkpoint_completion_p50_ms": 2,
        "checkpoint_completion_p95_ms": 3,
        "checkpoint_completion_p99_ms": 4,
        "rank_commit_sample_count": 12,
        "checkpoint_completion_sample_count": 3,
    }
    formal = {
        "policies": ["fixed", "random"],
        "trials": [
            {**metrics, "policy": "fixed", "trial": 1, "nic_xmit_bytes_per_epoch": None},
            {**metrics, "policy": "random", "trial": 1, "nic_xmit_bytes_per_epoch": 120},
        ],
    }

    rows = {item["policy"]: item for item in aggregate(formal)}
    assert rows["fixed"]["nic_xmit_bytes_per_epoch_valid_trial_count"] == 0
    assert rows["fixed"]["nic_xmit_bytes_per_epoch_mean"] is None
    paired = rows["random"]["paired_reductions_vs_fixed"]["nic_xmit_bytes_per_epoch"]
    assert paired["valid_pair_count"] == 0
    assert paired["mean_percent"] is None


def test_recovery_noninferiority_uses_confidence_interval_upper_bound():
    assert classify_noninferiority(-2.0, 3.0, 5.0) == "pass"
    assert classify_noninferiority(4.0, 3.0, 5.0) == "inconclusive"
    assert classify_noninferiority(10.0, 3.0, 5.0) == "fail"


def test_four_host_analysis_keeps_availability_scopes_separate():
    assert treatment_baseline("hierarchical_joint") == "sequential_fixed"
    assert treatment_baseline("host_pooled_joint") == "host_pooled_fixed"
    assert treatment_baseline("sequential_fixed") is None


def test_four_host_paired_bootstrap_is_deterministic_and_trial_based():
    first = paired_bootstrap([5.0, 7.0, 9.0], seed=20260902)
    second = paired_bootstrap([5.0, 7.0, 9.0], seed=20260902)
    assert first == second
    assert first["pair_count"] == 3
    assert first["mean_percent"] == 7.0
    assert first["ci95_low_percent"] <= 7.0 <= first["ci95_high_percent"]
