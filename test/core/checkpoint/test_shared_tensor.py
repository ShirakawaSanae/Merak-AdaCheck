import numpy as np
import torch

from Merak.core.checkpoint.catalog import checkpoint_state_checksum
from Merak.core.checkpoint.shared_tensor import (
    SharedTensorServer,
    _shared_snapshot,
    describe_shared_update,
)


def test_shared_update_digest_accepts_numpy_runtime_state():
    state = {
        "kind": "optimizer_update",
        "runtime_state": {
            "numpy_rng": ("MT19937", np.arange(8, dtype=np.uint32))
        },
    }
    first = describe_shared_update(state)

    assert describe_shared_update(state).payload_checksum == first.payload_checksum
    state["runtime_state"]["numpy_rng"][1][0] = 9
    assert describe_shared_update(state).payload_checksum != first.payload_checksum


def test_checkpoint_state_checksum_is_stable_and_content_sensitive():
    state = {
        "tensor": torch.arange(4),
        "scalar": torch.tensor(1.0),
        "metadata": {"epoch": 1},
    }

    assert checkpoint_state_checksum(state) == checkpoint_state_checksum(state)
    state["tensor"][0] = 9
    assert checkpoint_state_checksum(state) != checkpoint_state_checksum(
        {
            "tensor": torch.arange(4),
            "scalar": torch.tensor(1.0),
            "metadata": {"epoch": 1},
        }
    )


def test_shared_update_equivalence_excludes_only_rank_local_state():
    first_state = {
        "kind": "optimizer_update",
        "base_id": "rank-zero-base",
        "binding": {"rank": 0},
        "gradients": [torch.tensor([1.0])],
        "runtime_state": {"rank": 0},
    }
    peer_state = {
        **first_state,
        "base_id": "rank-one-base",
        "binding": {"rank": 1},
        "runtime_state": {"rank": 1},
    }

    first = describe_shared_update(first_state)
    peer = describe_shared_update(peer_state)
    assert first.equivalence_checksum == peer.equivalence_checksum
    assert first.payload_checksum != peer.payload_checksum

    peer_state["gradients"] = [torch.tensor([2.0])]
    assert describe_shared_update(peer_state).equivalence_checksum != (
        first.equivalence_checksum
    )


def test_shared_snapshot_reuses_matching_tensor_storage():
    first = {
        "gradients": [torch.arange(8, dtype=torch.float32)],
        "metadata": {"epoch": 1},
    }
    shared = _shared_snapshot(first)
    pointer = shared["gradients"][0].data_ptr()
    second = {
        "gradients": [torch.arange(8, dtype=torch.float32) + 10],
        "metadata": {"epoch": 2},
    }

    reused = _shared_snapshot(second, shared)

    assert reused["gradients"][0].data_ptr() == pointer
    assert reused["gradients"][0].is_shared()
    assert torch.equal(reused["gradients"][0], second["gradients"][0])
    assert reused["metadata"] == {"epoch": 2}


def test_shared_snapshot_reallocates_changed_tensor_shape():
    shared = _shared_snapshot(torch.arange(4, dtype=torch.float32))

    changed = _shared_snapshot(torch.arange(5, dtype=torch.float32), shared)

    assert changed.data_ptr() != shared.data_ptr()
    assert changed.is_shared()
    assert torch.equal(changed, torch.arange(5, dtype=torch.float32))
