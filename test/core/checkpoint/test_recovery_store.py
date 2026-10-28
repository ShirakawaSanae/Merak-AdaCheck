from dataclasses import replace

import pytest
import torch

from Merak.core.checkpoint.catalog import serialize_checkpoint
from Merak.core.checkpoint.manifest import create_prepared_manifest
from Merak.core.checkpoint.recovery import ReplicaStore
from Merak.core.checkpoint.transport import ReplicaEnvelope


def _envelope(source_rank: int, epoch: int, value: int) -> ReplicaEnvelope:
    serialized = serialize_checkpoint({"model": torch.tensor([value])})
    prepared = create_prepared_manifest(
        epoch=epoch,
        source_rank=source_rank,
        source_host=f"host-{source_rank}",
        target_rank=1 - source_rank,
        target_host=f"host-{1 - source_rank}",
        group_ranks=(0, 1),
        serialized=serialized,
    )
    committed = replace(prepared, state="committed", committed_time_ns=1)
    return ReplicaEnvelope(committed, serialized.payload)


def test_replica_store_has_a_strict_capacity_and_evicts_oldest():
    first = _envelope(0, 1, 10)
    second = _envelope(1, 1, 11)
    third = _envelope(0, 2, 12)
    store = ReplicaStore(max_bytes=len(first.payload) + len(second.payload))

    store.put(first)
    store.put(second)
    store.put(third)

    assert store.bytes_used <= store.max_bytes
    with pytest.raises(KeyError, match="replica not found"):
        store.get(0, 1)
    assert store.get(0, 2) == third


def test_replica_larger_than_capacity_is_rejected():
    envelope = _envelope(0, 1, 10)
    store = ReplicaStore(max_bytes=len(envelope.payload) - 1)

    with pytest.raises(MemoryError, match="exceeds store capacity"):
        store.put(envelope)
