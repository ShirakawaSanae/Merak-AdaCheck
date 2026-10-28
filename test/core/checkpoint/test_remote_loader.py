from types import SimpleNamespace

import pytest
import torch

from Merak.core.checkpoint.catalog import (
    deserialize_checkpoint,
    serialize_checkpoint,
)
from Merak.core.checkpoint.rchara_checkpoint import RcharaCheckpointLoader
from Merak.core.checkpoint.rchara_checkpoint import RcharaCheckpointSaver


def _step(model, optimizer, scheduler):
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(torch.ones(2, 4)), torch.zeros(2, 4))
    loss.backward()
    optimizer.step()
    scheduler.step()


def test_remote_loader_applies_full_training_state():
    torch.manual_seed(11)
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    _step(model, optimizer, scheduler)
    expected = {
        "checkpoint_version": 3.0,
        "iteration": 7,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
    }
    serialized = serialize_checkpoint(expected)
    expected = deserialize_checkpoint(serialized.payload, serialized.payload_checksum)

    _step(model, optimizer, scheduler)
    args = SimpleNamespace(
        finetune=False,
        no_load_optim=False,
        no_load_rng=True,
        zero_stage=None,
        fp16=False,
    )
    iteration = RcharaCheckpointLoader().load_remote_state(
        expected, model, optimizer, scheduler, args
    )

    assert iteration == 7
    assert scheduler.state_dict() == expected["lr_scheduler"]
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, expected["model"][name])
    actual_optimizer = serialize_checkpoint(optimizer.state_dict())
    expected_optimizer = serialize_checkpoint(expected["optimizer"])
    assert actual_optimizer.objects == expected_optimizer.objects


def test_remote_loader_rejects_parameter_only_state():
    model = torch.nn.Linear(2, 2)
    args = SimpleNamespace()

    with pytest.raises(ValueError, match="does not contain a full model state"):
        RcharaCheckpointLoader().load_remote_state(
            {"weight": model.weight}, model, None, None, args
        )


def test_remote_saver_publishes_tracker_only_after_remote_commit(monkeypatch):
    saver = RcharaCheckpointSaver.__new__(RcharaCheckpointSaver)
    saver.args = SimpleNamespace(
        check_steps=[],
        use_differential=False,
        zero_stage=0,
    )
    saver.iteration = 7
    saver.model = None
    saver.optimizer = None
    saver.redundancy_checker = None
    saver.save_path = "unused"
    saver.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    events = []
    saver._remote_checkpoint_enabled = lambda: True
    saver._save_remote_checkpoint = lambda *args, **kwargs: events.append("remote")
    saver._save_tracker_file = lambda: events.append("tracker")
    monkeypatch.setattr(
        "Merak.core.checkpoint.rchara_checkpoint.mpu.get_data_parallel_rank",
        lambda: 1,
    )
    monkeypatch.setattr(
        "Merak.core.checkpoint.rchara_checkpoint.dist.barrier", lambda: None
    )

    saver._save_state_dict(
        {"model": {}, "optimizer": {}}, checkpoint_name="unused.pt"
    )

    assert events == ["remote", "tracker"]
