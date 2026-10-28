import torch

from Merak.core.checkpoint import state_oracle
from Merak.core.checkpoint.state_oracle import compare_checkpoint_state


def test_bfloat16_adjacent_value_is_within_normwise_contract():
    expected = torch.tensor([1.0], dtype=torch.bfloat16)
    adjacent = torch.nextafter(
        expected, torch.tensor([float("inf")], dtype=torch.bfloat16)
    )

    result = compare_checkpoint_state({"weight": adjacent}, {"weight": expected})

    assert not result["bitwise_equal"]
    assert result["numerically_close"]
    assert result["max_bfloat16_ulp"] == 1


def test_bfloat16_difference_above_pairwise_rounding_bound_is_rejected():
    expected = torch.tensor([1.0], dtype=torch.bfloat16)
    direction = torch.tensor([float("inf")], dtype=torch.bfloat16)
    two_ulp = torch.nextafter(torch.nextafter(expected, direction), direction)

    result = compare_checkpoint_state(two_ulp, expected)

    assert not result["numerically_close"]
    assert result["max_bfloat16_ulp"] == 2


def test_fp32_and_metadata_contracts_are_explicit():
    close = compare_checkpoint_state(
        {"state": [torch.tensor([1.0 + 5e-7]), 3]},
        {"state": [torch.tensor([1.0]), 3]},
    )
    different_metadata = compare_checkpoint_state({"step": 4}, {"step": 3})

    assert close["numerically_close"]
    assert not close["bitwise_equal"]
    assert not different_metadata["numerically_close"]


def test_chunking_preserves_tensor_level_bfloat16_contract(monkeypatch):
    monkeypatch.setattr(state_oracle, "_COMPARISON_CHUNK_ELEMENTS", 1)
    expected = torch.tensor([1.0, 1024.0], dtype=torch.bfloat16)
    actual = expected.clone()
    actual[0] = torch.nextafter(
        actual[0], torch.tensor(float("inf"), dtype=torch.bfloat16)
    )

    result = compare_checkpoint_state(actual, expected)

    assert result["exact_mismatch_elements"] == 1
    assert result["max_bfloat16_ulp"] == 1
    assert result["numerically_close"]
