from pathlib import Path

import pytest
import torch

from Merak.core.checkpoint.catalog import serialize_checkpoint
from Merak.core.checkpoint.manifest import (
    ManifestRepository,
    create_prepared_manifest,
)


def _manifest():
    return create_prepared_manifest(
        epoch=4,
        source_rank=0,
        source_host="host-a",
        target_rank=1,
        target_host="host-b",
        group_ranks=(0, 1),
        serialized=serialize_checkpoint({"model": torch.arange(8)}),
    )


def test_only_committed_manifest_is_visible(tmp_path: Path):
    repository = ManifestRepository(tmp_path)
    prepared = _manifest()

    prepared_path = repository.prepare(prepared)
    assert prepared_path.exists()
    assert repository.latest_committed(0) is None

    committed = repository.commit(prepared)
    repository.record_replica(1, committed)

    assert repository.latest_committed(0) == committed
    assert repository.replica_path(1, committed).exists()


def test_holder_rejects_prepared_manifest(tmp_path: Path):
    repository = ManifestRepository(tmp_path)

    with pytest.raises(ValueError, match="only a committed replica"):
        repository.record_replica(1, _manifest())
