from hashlib import blake2s

import torch
import pytest

from Merak.core.checkpoint.name_value_mapping import NVMapping
from Merak.core.checkpoint.redundancy_checker import RedundancyChecker


def test_redundancy_scope_is_explicit_and_defaults_to_data(monkeypatch):
    monkeypatch.delenv("CKPT_REDUNDANCY_SCOPE", raising=False)
    assert RedundancyChecker().comparison_scope == "data"
    monkeypatch.setenv("CKPT_REDUNDANCY_SCOPE", "world")
    checker = RedundancyChecker()
    assert checker.comparison_scope == "world"
    assert not checker.comm_groups_for_redundancy_checking
    monkeypatch.setenv("CKPT_REDUNDANCY_SCOPE", "peer")
    with pytest.raises(ValueError, match="data or world"):
        RedundancyChecker()


def test_data_parallel_ranks_follow_the_initialized_process_group(monkeypatch):
    checker = RedundancyChecker()
    group = object()
    monkeypatch.setattr(
        "Merak.core.checkpoint.redundancy_checker.mpu.get_data_parallel_group",
        lambda: group,
    )
    monkeypatch.setattr(
        "Merak.core.checkpoint.redundancy_checker.dist.get_process_group_ranks",
        lambda actual: (9, 1, 5, 13) if actual is group else (),
    )

    checker._get_data_parallel_group_ranks()

    assert checker.data_ranks == (9, 1, 5, 13)


def test_redundancy_comparison_preserves_all_duplicate_index_pairs():
    checker = RedundancyChecker()
    checker.packed_tensor = torch.tensor([11, 22, 11, 33], dtype=torch.int64)

    checker._calculate_redundancy(
        target_rank=7,
        target_tensor=torch.tensor([33, 11, 44, 11], dtype=torch.int64),
        my_rank=3,
    )

    assert checker.redundancy_results[7] == [
        [3, 0, 7, 1],
        [3, 0, 7, 3],
        [3, 2, 7, 1],
        [3, 2, 7, 3],
        [3, 3, 7, 0],
    ]


def test_name_value_hash_supports_bfloat16_without_changing_fp32_bytes():
    mapping = NVMapping()
    fp32 = torch.tensor([1.25, -2.5], dtype=torch.float32)
    expected = blake2s(fp32.numpy().tobytes() + b"optimizer", digest_size=7).hexdigest()

    assert mapping.get_hash_value(fp32) == expected
    assert mapping.get_hash_value(fp32.to(torch.bfloat16)) == mapping.get_hash_value(
        fp32.to(torch.bfloat16)
    )


def test_redundancy_hash_supports_bfloat16_without_changing_fp32_bytes():
    checker = RedundancyChecker()
    fp32 = torch.tensor([1.25, -2.5], dtype=torch.float32)
    expected = blake2s(fp32.numpy().tobytes(), digest_size=7).hexdigest()

    assert checker._get_hash_values_for_single_dict(fp32) == expected
    assert checker._get_hash_values_for_single_dict(
        fp32.to(torch.bfloat16)
    ) == checker._get_hash_values_for_single_dict(fp32.to(torch.bfloat16))
    assert checker._get_hash_values_for_single_dict(torch.tensor(1.0)) == blake2s(
        torch.tensor(1.0).numpy().tobytes(), digest_size=7
    ).hexdigest()
