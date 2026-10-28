import copy
import threading
import time
from io import BytesIO
from pathlib import Path

import pytest
import torch

from Merak.core.checkpoint.async_checkpoint import (
    AsyncServiceRemoteCheckpointSession,
    CheckpointBackpressureError,
)
from Merak.core.checkpoint.catalog import deserialize_checkpoint, serialize_checkpoint
from Merak.core.checkpoint.incremental_checkpoint import (
    RecoveryBinding,
    apply_optimizer_update,
    capture_optimizer_update,
)
from Merak.core.checkpoint.manifest import ManifestRepository, create_prepared_manifest
from Merak.core.checkpoint.recovery import CommitResult
from Merak.core.checkpoint.transport import TransportMetrics


class _Session:
    rank = 0

    def __init__(self, root: Path, gate: threading.Event | None = None):
        self.repository = ManifestRepository(root)
        self.gate = gate
        self.states = []

    def save(
        self,
        epoch,
        state,
        *,
        state_is_snapshot=False,
        object_checksums=True,
    ):
        if self.gate is not None:
            self.gate.wait(timeout=5)
        snapshot = serialize_checkpoint(
            state,
            state_is_snapshot=state_is_snapshot,
            object_checksums=object_checksums,
        )
        self.states.append((epoch, snapshot))
        manifest = create_prepared_manifest(
            epoch=epoch,
            source_rank=0,
            source_host="host-a",
            target_rank=1,
            target_host="host-b",
            group_ranks=(0, 1),
            serialized=snapshot,
        )
        return CommitResult(
            manifest=manifest,
            serialize_ms=0.0,
            transport=TransportMetrics(0, len(snapshot.payload), len(snapshot.payload), 0.0),
            commit_ms=0.0,
            store_bytes=len(snapshot.payload),
        )

    def restore_from_peer(self, epoch):
        raise NotImplementedError


def test_async_snapshot_is_immutable_and_fifo(tmp_path):
    session = _Session(tmp_path)
    async_session = AsyncServiceRemoteCheckpointSession(session, max_inflight=2)
    tensor = torch.tensor([1.0])

    async_session.enqueue(1, {"model": tensor})
    tensor.fill_(2.0)
    async_session.enqueue(2, {"model": tensor})
    completed = async_session.close()

    assert [item.epoch for item in completed] == [1, 2]
    assert all(item.committed_at_ns > 0 for item in completed)
    assert all(item.publisher_queue_ms >= 0 for item in completed)
    assert all(item.snapshot_ready_wait_ms >= 0 for item in completed)
    assert all(
        item.queue_ms == item.publisher_queue_ms + item.snapshot_ready_wait_ms
        for item in completed
    )
    assert all(item.enqueue_to_commit_ms >= item.queue_ms for item in completed)
    assert torch.load(BytesIO(session.states[0][1].payload), weights_only=False)["model"].item() == 1
    assert torch.load(BytesIO(session.states[1][1].payload), weights_only=False)["model"].item() == 2


def test_snapshot_slots_can_be_initialized_before_enqueue(tmp_path):
    initialized = []

    def initial_state():
        initialized.append(True)
        return {"model": torch.empty(1)}

    session = _Session(tmp_path)
    async_session = AsyncServiceRemoteCheckpointSession(
        session,
        max_inflight=2,
        initial_slot_state_factory=initial_state,
    )
    async_session.enqueue(1, {"model": torch.tensor([3.0])})
    async_session.close()

    assert len(initialized) == 2
    assert torch.load(BytesIO(session.states[0][1].payload), weights_only=False)["model"].item() == 3


def test_full_queue_fails_instead_of_waiting_or_dropping(tmp_path):
    gate = threading.Event()
    session = _Session(tmp_path, gate)
    async_session = AsyncServiceRemoteCheckpointSession(session, max_inflight=1)
    async_session.enqueue(7, {"model": torch.tensor([7])})

    started = time.perf_counter()
    with pytest.raises(CheckpointBackpressureError):
        async_session.enqueue(8, {"model": torch.tensor([8])})
    assert time.perf_counter() - started < 0.1

    gate.set()
    assert [item.epoch for item in async_session.close()] == [7]


def test_epochs_must_be_contiguous(tmp_path):
    session = _Session(tmp_path)
    async_session = AsyncServiceRemoteCheckpointSession(session, max_inflight=2)
    async_session.enqueue(3, {"model": torch.tensor([3])})
    with pytest.raises(ValueError, match="contiguous"):
        async_session.enqueue(5, {"model": torch.tensor([5])})
    async_session.close()


def test_prebuilt_cpu_snapshot_can_be_published_without_a_second_copy(tmp_path):
    session = _Session(tmp_path)
    async_session = AsyncServiceRemoteCheckpointSession(session, max_inflight=1)
    snapshot = {"runtime": torch.tensor([3], device="cpu")}

    async_session.enqueue_snapshot(1, snapshot)
    async_session.close()

    restored = deserialize_checkpoint(
        session.states[0][1].payload,
        session.states[0][1].payload_checksum,
    )
    assert restored["runtime"].item() == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_snapshot_survives_optimizer_step_and_gradient_release(tmp_path):
    session = _Session(tmp_path)
    async_session = AsyncServiceRemoteCheckpointSession(session, max_inflight=1)
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], device="cuda"))
    optimizer = torch.optim.AdamW([parameter], lr=1e-2, foreach=False)
    parameter.grad = torch.tensor([0.25, -0.5], device="cuda")

    async_session.enqueue(1, {"gradient": parameter.grad})
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    async_session.close()

    restored = torch.load(
        BytesIO(session.states[0][1].payload), map_location="cpu", weights_only=False
    )
    assert torch.equal(restored["gradient"], torch.tensor([0.25, -0.5]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_twenty_async_cuda_updates_replay_exactly(tmp_path):
    direct_parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], device="cuda"))
    direct = torch.optim.AdamW(
        [direct_parameter], lr=3e-3, foreach=False, fused=False
    )
    direct_scheduler = torch.optim.lr_scheduler.StepLR(
        direct, step_size=7, gamma=0.9
    )
    for step in range(1, 6):
        direct_parameter.grad = torch.tensor(
            [step / 10, -step / 20], device="cuda"
        )
        direct.step()
        direct_scheduler.step()
    base_parameter = direct_parameter.detach().clone()
    base_optimizer = copy.deepcopy(direct.state_dict())
    base_scheduler = copy.deepcopy(direct_scheduler.state_dict())

    binding = RecoveryBinding(0, 2, 0, 0, 0)
    base_id = "sha256:" + "0" * 64
    parameter_ids = [["parameter"]]
    session = _Session(tmp_path)
    async_session = AsyncServiceRemoteCheckpointSession(session, max_inflight=1)
    for epoch in range(1, 21):
        direct_parameter.grad = torch.tensor(
            [epoch / 100, -epoch / 200], device="cuda"
        )
        record = capture_optimizer_update(
            direct,
            parameter_ids=parameter_ids,
            binding=binding,
            base_id=base_id,
            base_epoch=0,
            epoch=epoch,
            snapshot_gradients=False,
        )
        async_session.enqueue(epoch, record)
        direct.step()
        direct_scheduler.step()
        direct.zero_grad(set_to_none=True)
        while not async_session.poll():
            time.sleep(0.001)
    async_session.close()

    replay_parameter = torch.nn.Parameter(base_parameter.clone())
    replay = torch.optim.AdamW(
        [replay_parameter], lr=3e-3, foreach=False, fused=False
    )
    replay_scheduler = torch.optim.lr_scheduler.StepLR(
        replay, step_size=7, gamma=0.9
    )
    replay.load_state_dict(base_optimizer)
    replay_scheduler.load_state_dict(base_scheduler)
    for epoch, serialized in session.states:
        record = deserialize_checkpoint(
            serialized.payload, serialized.payload_checksum
        )
        apply_optimizer_update(
            replay,
            record,
            expected_parameter_ids=parameter_ids,
            expected_binding=binding,
            expected_base_id=base_id,
            expected_base_epoch=0,
            expected_epoch=epoch,
        )
        replay_scheduler.step()

    assert torch.equal(direct_parameter, replay_parameter)
    assert direct.state_dict()["param_groups"] == replay.state_dict()["param_groups"]
    assert direct_scheduler.state_dict() == replay_scheduler.state_dict()
    for direct_state, replay_state in zip(
        direct.state_dict()["state"].values(), replay.state_dict()["state"].values()
    ):
        for key in direct_state:
            assert torch.equal(direct_state[key], replay_state[key])
