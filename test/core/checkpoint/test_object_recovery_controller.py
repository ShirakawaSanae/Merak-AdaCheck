import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from Merak.core.checkpoint.object_recovery import RestoredRankState
from Merak.core.checkpoint.object_recovery_adapter import capture_live_object_fragments
from Merak.core.checkpoint.object_recovery_controller import (
    LiveObjectRecoveryController,
    load_object_recovery_controller,
)
from Merak.merak_trainer import MerakTrainer


class _Scheduler:
    def __init__(self):
        self.value = 0

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        if set(state) != {"value"}:
            raise ValueError("invalid scheduler state")
        self.value = int(state["value"])


def _module(value):
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(
        torch.full((2, 2), value, dtype=torch.bfloat16)
    )
    return module


def _optimizer(module, *, initialized):
    master = torch.nn.Parameter(module.weight.detach().float().clone())
    inner = torch.optim.AdamW([master], lr=1e-3, foreach=False, fused=False)
    if initialized:
        master.grad = torch.ones_like(master)
        inner.step()
    return SimpleNamespace(
        fp16_groups=[[module.weight]],
        fp32_groups=[[master]],
        optimizer=inner,
        dynamic_loss_scale=False,
        cur_scale=128.0,
        cur_iter=0,
        global_step=0,
    )


def _state(source, source_optimizer):
    tensors = {
        item.address.semantic_id: item.tensor.clone()
        for item in capture_live_object_fragments(
            source,
            source_optimizer,
            rank=0,
            storage_host=1,
            epoch=7,
            tensor_parallel_rank=0,
            tensor_parallel_world_size=1,
        )
    }
    return RestoredRankState(
        tensors,
        {
            "scheduler": {"value": 7},
            "rng": {
                "random_rng_state": ("python",),
                "np_rng_state": ("numpy",),
                "torch_rng_state": torch.arange(4, dtype=torch.uint8),
                "cuda_rng_state": torch.arange(4, dtype=torch.uint8),
                "rng_tracker_states": {"model-parallel-rng": torch.arange(2)},
            },
            "sampler": {"consumed_samples": 112, "epoch": 3},
            "progress": {
                "iteration": 7,
                "optimizer": {
                    "dynamic_loss_scale": False,
                    "cur_scale": 64.0,
                    "cur_iter": 7,
                    "global_step": 7,
                },
            },
        },
    )


def test_controller_restores_fresh_live_training_state_and_runtime():
    source = _module(2)
    source_optimizer = _optimizer(source, initialized=True)
    target = _module(9)
    target_optimizer = _optimizer(target, initialized=False)
    scheduler = _Scheduler()
    sampler = SimpleNamespace(consumed_samples=0, epoch=0)
    runtime_calls = []

    def restore_runtime(state):
        runtime_calls.append(state)
        sampler.consumed_samples = state["sampler"]["consumed_samples"]
        sampler.epoch = state["sampler"]["epoch"]
        return state["iteration"]

    controller = LiveObjectRecoveryController(
        target,
        target_optimizer,
        scheduler,
        sampler=sampler,
        tensor_parallel_rank=0,
        tensor_parallel_world_size=1,
        state_loader=lambda epoch: _state(source, source_optimizer),
        runtime_restore=restore_runtime,
    )

    assert controller.restore(7) == 7
    assert torch.equal(target.weight, source.weight)
    assert scheduler.value == 7
    assert (sampler.consumed_samples, sampler.epoch) == (112, 3)
    assert (target_optimizer.global_step, target_optimizer.cur_iter) == (7, 7)
    assert target_optimizer.cur_scale == 64.0
    assert runtime_calls[0]["iteration"] == 7
    assert set(runtime_calls[0]) == {
        "iteration",
        "random_rng_state",
        "np_rng_state",
        "torch_rng_state",
        "cuda_rng_state",
        "rng_tracker_states",
        "sampler",
    }


def test_controller_rejects_incomplete_opaque_state_before_live_mutation():
    source = _module(2)
    source_optimizer = _optimizer(source, initialized=True)
    restored = _state(source, source_optimizer)
    restored = RestoredRankState(
        restored.tensors,
        {key: value for key, value in restored.opaque.items() if key != "sampler"},
    )
    target = _module(9)
    target_optimizer = _optimizer(target, initialized=False)
    scheduler = _Scheduler()
    original = target.weight.detach().clone()
    controller = LiveObjectRecoveryController(
        target,
        target_optimizer,
        scheduler,
        sampler=SimpleNamespace(consumed_samples=0, epoch=0),
        tensor_parallel_rank=0,
        tensor_parallel_world_size=1,
        state_loader=lambda epoch: restored,
        runtime_restore=lambda state: state["iteration"],
    )

    with pytest.raises(ValueError, match="requires scheduler"):
        controller.restore(7)

    assert torch.equal(target.weight, original)
    assert not target_optimizer.optimizer.state
    assert scheduler.value == 0


def test_factory_dispatch_is_explicit_and_requires_restore_controller(monkeypatch):
    plugin = ModuleType("object_recovery_test_plugin")
    expected = SimpleNamespace(restore=lambda epoch: epoch)
    plugin.build = lambda **kwargs: expected
    plugin.invalid = lambda **kwargs: object()
    monkeypatch.setitem(sys.modules, plugin.__name__, plugin)

    assert (
        load_object_recovery_controller(f"{plugin.__name__}:build", marker=1)
        is expected
    )
    with pytest.raises(TypeError, match="restore controller"):
        load_object_recovery_controller(f"{plugin.__name__}:invalid")
    with pytest.raises(ValueError, match="module:function"):
        load_object_recovery_controller(plugin.__name__)


def _empty_trainer(*, resume_from_checkpoint=None):
    trainer = object.__new__(MerakTrainer)
    trainer.engine = SimpleNamespace(module=object(), optimizer=object(), lr_scheduler=object())
    trainer.init_engine = lambda: None
    trainer.prepare_data = lambda value: value
    trainer._checkpoint_saver = object()
    trainer.train_dataloader = SimpleNamespace(batch_sampler=object())
    trainer.args = SimpleNamespace(
        resume_from_checkpoint=resume_from_checkpoint,
        num_train_epochs=0,
        lora_config=None,
    )
    trainer.tuning_params = SimpleNamespace(
        global_steps=0,
        resume=lambda iteration: 0,
        logging=lambda: None,
    )
    trainer.logger = SimpleNamespace(warning=lambda message: None)
    return trainer


def test_ep_zero1_publication_rejects_semantic_fixed(monkeypatch):
    monkeypatch.setenv("CKPT_ZERO1_OBJECT_PUBLISH_STEPS", "1")
    monkeypatch.setenv("CKPT_OBJECT_SEMANTIC_ENABLE", "1")
    trainer = _empty_trainer()
    from Merak.core.moe.expert_parallel import ExpertParallelLayout

    trainer.engine.module = SimpleNamespace(
        _merak_role_aware_zero1=True,
        _merak_ep_layout=ExpertParallelLayout(1, 2, 1, 1),
    )
    trainer.args.zero_stage = 1
    trainer.args.ada_checkpoint = False
    trainer.args.save = False

    with pytest.raises(ValueError, match="fresh role-aware ZeRO-1 job"):
        trainer.train()


def test_dense_semantic_fixed_is_not_blocked_by_zero1_guard(monkeypatch):
    monkeypatch.setenv("CKPT_OBJECT_SEMANTIC_ENABLE", "1")
    trainer = _empty_trainer()
    calls = []

    monkeypatch.setattr(
        "Merak.core.checkpoint.object_semantic_training.build_semantic_fixed_training_controller",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    trainer.train()
    assert len(calls) == 1


def test_ep_without_zero1_keeps_semantic_fixed_enabled(monkeypatch):
    monkeypatch.setenv("CKPT_OBJECT_SEMANTIC_ENABLE", "1")
    trainer = _empty_trainer()
    from Merak.core.moe.expert_parallel import ExpertParallelLayout

    trainer.engine.module = SimpleNamespace(
        _merak_ep_layout=ExpertParallelLayout(1, 2, 1, 1)
    )
    calls = []
    monkeypatch.setattr(
        "Merak.core.checkpoint.object_semantic_training.build_semantic_fixed_training_controller",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    trainer.train()
    assert len(calls) == 1


def test_trainer_object_restore_dispatch_is_opt_in_and_restore_only(monkeypatch):
    plugin = ModuleType("trainer_object_recovery_test_plugin")
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(restore=lambda epoch: epoch)

    plugin.build = build
    monkeypatch.setitem(sys.modules, plugin.__name__, plugin)
    monkeypatch.setenv("CKPT_OBJECT_RECOVERY_FACTORY", f"{plugin.__name__}:build")
    monkeypatch.setenv("CKPT_OBJECT_RESTORE_EPOCH", "7")
    barriers = []
    monkeypatch.setattr(
        "Merak.merak_trainer.dist.barrier", lambda: barriers.append("restored")
    )

    trainer = _empty_trainer()
    trainer.train()

    assert calls == [
        {
            "module": trainer.engine.module,
            "optimizer": trainer.engine.optimizer,
            "scheduler": trainer.engine.lr_scheduler,
            "sampler": trainer.train_dataloader.batch_sampler,
            "args": trainer.args,
        }
    ]
    assert barriers == ["restored"]


def test_trainer_object_restore_rejects_other_restore_modes(monkeypatch):
    monkeypatch.setenv("CKPT_OBJECT_RECOVERY_FACTORY", "unused:factory")
    monkeypatch.setenv("CKPT_OBJECT_RESTORE_EPOCH", "7")
    trainer = _empty_trainer(resume_from_checkpoint="disk-checkpoint")

    with pytest.raises(ValueError, match="mutually exclusive"):
        trainer.train()


def test_trainer_semantic_fixed_source_uses_existing_checkpoint_lifecycle(monkeypatch):
    calls = []
    lifecycle = []

    class _Controller:
        def start(self, epoch):
            lifecycle.append(("start", epoch))

        def finalize(self, epoch):
            lifecycle.append(("finalize", epoch))
            return {"valid": True, "recovery_valid": True, "cadence_valid": True}

    monkeypatch.setattr(
        "Merak.core.checkpoint.object_semantic_training.build_semantic_fixed_training_controller",
        lambda *args, **kwargs: calls.append((args, kwargs)) or _Controller(),
    )
    barriers = []
    monkeypatch.setattr("Merak.merak_trainer.dist.barrier", lambda: barriers.append("barrier"))
    monkeypatch.setattr("Merak.merak_trainer.dist.all_reduce", lambda tensor, op=None: tensor)
    original_tensor = torch.tensor
    monkeypatch.setattr(
        "Merak.merak_trainer.torch.tensor",
        lambda *args, **kwargs: original_tensor(
            *args, **{key: value for key, value in kwargs.items() if key != "device"}
        ),
    )
    monkeypatch.setattr("Merak.merak_trainer.torch.cuda.current_device", lambda: 0)
    monkeypatch.setenv("CKPT_OBJECT_SEMANTIC_ENABLE", "1")

    trainer = _empty_trainer()
    trainer.train()

    assert len(calls) == 1
    assert calls[0][0] == (
        trainer.engine.module,
        trainer.engine.optimizer,
        trainer.engine.lr_scheduler,
    )
    assert calls[0][1]["args"] is trainer.args
    assert calls[0][1]["sampler"] is trainer.train_dataloader.batch_sampler
    assert calls[0][1]["epoch_provider"]() == 0
    assert lifecycle == [("start", 0), ("finalize", 0)]
    assert barriers == ["barrier"]


def test_trainer_semantic_fixed_source_rejects_restore_mode(monkeypatch):
    monkeypatch.setenv("CKPT_OBJECT_SEMANTIC_ENABLE", "1")
    monkeypatch.setenv("CKPT_OBJECT_RECOVERY_FACTORY", "unused:factory")
    monkeypatch.setenv("CKPT_OBJECT_RESTORE_EPOCH", "7")

    with pytest.raises(ValueError, match="mutually exclusive"):
        _empty_trainer().train()
