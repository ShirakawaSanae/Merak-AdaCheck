from types import SimpleNamespace

import pytest
import torch

from Merak.utils.datasets import DynamicGenDataset
from Merak.utils.trainer_utils import MegatronPretrainingRandomSampler


@pytest.mark.parametrize("vocab_size", [2, 31, 1000, 32000])
def test_text_only_tokens_respect_model_vocabulary(vocab_size):
    dataset = DynamicGenDataset(
        SimpleNamespace(vocab_size=vocab_size),
        mode="text_only",
        seq_length=64,
    )

    sample = dataset[0]

    for key in ("input_ids", "labels"):
        assert sample[key].min() >= 1
        assert sample[key].max() < min(vocab_size, 1000)


def test_text_only_requires_at_least_two_vocabulary_entries():
    dataset = DynamicGenDataset(
        SimpleNamespace(vocab_size=1), mode="text_only", seq_length=4
    )

    with pytest.raises(RuntimeError, match="vocab_size >= 2"):
        dataset[0]


def test_seeded_text_samples_are_index_deterministic_without_global_rng_changes():
    dataset = DynamicGenDataset(
        SimpleNamespace(vocab_size=31),
        mode="text_only",
        seq_length=64,
        seed=73,
    )
    torch_state = torch.get_rng_state().clone()

    first = dataset[9]
    dataset[3]
    replay = dataset[9]

    assert all(torch.equal(first[key], replay[key]) for key in first)
    assert not torch.equal(first["input_ids"], dataset[10]["input_ids"])
    assert torch.equal(torch_state, torch.get_rng_state())


def test_sampler_resume_replays_next_batch_across_epoch_boundary():
    def make_sampler(consumed_samples):
        return MegatronPretrainingRandomSampler(
            total_samples=8,
            consumed_samples=consumed_samples,
            micro_batch_size=1,
            data_parallel_rank=0,
            data_parallel_size=2,
        )

    uninterrupted = make_sampler(0)
    uninterrupted_iterator = iter(uninterrupted)
    first_epoch = [next(uninterrupted_iterator) for _ in range(4)]
    assert uninterrupted.consumed_samples == 8

    resumed = make_sampler(uninterrupted.consumed_samples)
    resumed_iterator = iter(resumed)

    assert resumed.epoch == 1
    assert next(uninterrupted_iterator) == next(resumed_iterator)
    assert first_epoch == [
        [index]
        for index in torch.randperm(4, generator=torch.Generator().manual_seed(0)).tolist()
    ]
