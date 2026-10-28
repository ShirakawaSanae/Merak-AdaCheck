import json

import pytest

from tools.analyze_frozen_nnscaler_layout import (
    analyze_layout,
    build_static_model_slice_catalog,
)
from tools.analyze_frozen_nnscaler_remap_upper_bound import analyze_remap_upper_bound


def _layout(*, gap: bool = False):
    parameters = []
    for rank in range(8):
        holders = (rank, 15 - rank)
        parameters.append(
            {
                "original_name": f"model.parameter_{rank}",
                "dtype": "torch.float32",
                "full_shape": [1],
                "shards": [
                    {
                        "rank": holder,
                        "kind": "parameter",
                        "dtype": "torch.float32",
                        "index_ranges": [[0, 1]],
                    }
                    for holder in holders
                ],
            }
        )
    if gap:
        parameters[0]["full_shape"] = [2]
        parameters[0]["shards"] = [{
            **parameters[0]["shards"][0],
            "index_ranges": [[0, 1]],
        }]
    return {
        "world_size": 16,
        "ranks_per_host": 8,
        "optimizer_layout": {"complete": False},
        "online_runtime_eligible": False,
        "parameters": parameters,
        "layout_hash": "fixture",
        "name": "fixture",
        "model_name": "fixture-model",
    }


def test_static_slice_matrix_uses_existing_exact_cross_host_solver(tmp_path):
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps(_layout()))

    result, objects = analyze_layout(_layout(), layout_path=layout_path)

    assert len(objects) == 8
    assert result["matrix"]["edge_count"] == 64
    assert result["policies"]["fixed_mirror"]["groups"] == [
        [rank, rank + 8] for rank in range(8)
    ]
    assert result["policies"]["constrained_exact_cross_host"]["groups"] == [
        [rank, 15 - rank] for rank in range(8)
    ]
    assert result["policies"]["fixed_mirror"]["static_model_slice_completion_bytes"] == 64
    assert result["policies"]["constrained_exact_cross_host"]["static_model_slice_completion_bytes"] == 0
    assert result["fixed_to_exact_static_completion_delta_percent"] == 100.0
    assert result["evidence_scope"]["model_state_only"] is True
    assert result["evidence_scope"]["online_runtime_eligible"] is False


def test_static_slice_catalog_rejects_uncovered_parameter_cells():
    with pytest.raises(ValueError, match="uncovered static slice"):
        build_static_model_slice_catalog(_layout(gap=True))


def test_balanced_remap_upper_bound_finds_nonzero_static_overlap(tmp_path):
    parameters = []
    for index, holders in enumerate(((0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15))):
        parameters.append(
            {
                "original_name": f"model.parameter_{index}",
                "dtype": "torch.float32",
                "full_shape": [1],
                "shards": [
                    {
                        "rank": rank,
                        "kind": "parameter",
                        "dtype": "torch.float32",
                        "index_ranges": [[0, 1]],
                    }
                    for rank in holders
                ],
            }
        )
    layout = {
        **_layout(),
        "parameters": parameters,
        "rank_placement": [
            {
                "rank": rank,
                "host_index": rank // 8,
                "host_id": "17" if rank < 8 else "1",
                "local_device": rank % 8,
            }
            for rank in range(16)
        ],
    }
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps(layout))

    result = analyze_remap_upper_bound(layout, layout_path=layout_path)

    current = result["current_placement"]["constrained_exact_matching"]
    upper = result["unconstrained_balanced_remap_upper_bound"]
    assert current["static_model_slice_overlap_bytes"] == 0
    assert current["static_model_slice_completion_bytes"] == 64
    assert upper["enumerated_balanced_partitions"] == 6435
    assert upper["abstract_host_sets"]["host_0"]["ranks"] == [0, 2, 4, 6, 8, 10, 12, 14]
    assert upper["maximum_static_model_slice_overlap"]["static_model_slice_overlap_bytes"] == 32
    assert upper["maximum_static_model_slice_overlap"]["static_model_slice_completion_bytes"] == 0
    assert upper["nonzero_static_model_slice_overlap_exists"] is True
    assert result["evidence_scope"]["logical_rank_host_remap_executable"] is False


def test_balanced_remap_upper_bound_requires_explicit_balanced_host_placement(tmp_path):
    layout_path = tmp_path / "layout.json"
    layout_path.write_text(json.dumps(_layout()))

    with pytest.raises(ValueError, match="provide all 16 rank placements"):
        analyze_remap_upper_bound(_layout(), layout_path=layout_path)
