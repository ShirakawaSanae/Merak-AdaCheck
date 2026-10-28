import pytest
import torch

from Merak.core.checkpoint.catalog import (
    deserialize_checkpoint,
    serialize_checkpoint,
    validate_checkpoint_objects,
)


def test_mixed_dtype_state_round_trip_and_object_validation():
    state = {
        "model": {
            "weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "scale": torch.arange(4, dtype=torch.float32).to(torch.bfloat16),
        },
        "optimizer": {
            0: {
                "exp_avg": torch.linspace(0, 1, 4, dtype=torch.float16),
                "step": torch.tensor(7, dtype=torch.int64),
            }
        },
        "rng": torch.tensor([1, 2, 3], dtype=torch.uint8),
        "iteration": 9,
    }

    serialized = serialize_checkpoint(state)
    restored = deserialize_checkpoint(serialized.payload, serialized.payload_checksum)
    validate_checkpoint_objects(restored, serialized.objects)

    assert {item.dtype for item in serialized.objects} == {
        "torch.bfloat16",
        "torch.float16",
        "torch.float32",
        "torch.int64",
        "torch.uint8",
    }
    assert restored["iteration"] == 9
    assert all(not tensor.is_cuda for tensor in restored["model"].values())


def test_payload_corruption_is_rejected():
    serialized = serialize_checkpoint({"model": torch.ones(4)})
    corrupted = serialized.payload[:-1] + bytes([serialized.payload[-1] ^ 0xFF])

    with pytest.raises(ValueError, match="payload checksum mismatch"):
        deserialize_checkpoint(corrupted, serialized.payload_checksum)


def test_payload_checksum_mode_preserves_schema_validation():
    state = {"gradient": torch.arange(8, dtype=torch.float32)}
    serialized = serialize_checkpoint(
        state,
        state_is_snapshot=True,
        object_checksums=False,
    )

    assert serialized.objects[0].checksum == ""
    restored = deserialize_checkpoint(serialized.payload, serialized.payload_checksum)
    validate_checkpoint_objects(restored, serialized.objects)
    with pytest.raises(ValueError, match="catalog validation failed"):
        validate_checkpoint_objects(
            {"gradient": torch.arange(7, dtype=torch.float32)},
            serialized.objects,
        )
