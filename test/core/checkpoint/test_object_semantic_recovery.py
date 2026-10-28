import copy
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import asdict

import pytest
import torch

from Merak.core.checkpoint.incremental_checkpoint import (
    RecoveryBinding,
    capture_optimizer_update,
)
from Merak.core.checkpoint.catalog import checkpoint_state_checksum
from Merak.core.checkpoint.object_optimizer_update import externalize_local_optimizer_update
from Merak.core.checkpoint.object_replica_service import ObjectReplicaServiceStore
from Merak.core.checkpoint.object_semantic_plan import (
    GradientShard,
    RankBinding,
    build_active_gradient_plan,
    confirm_gradient_equivalence,
)
from Merak.core.checkpoint.object_semantic_recovery import (
    restore_semantic_fixed_object_suffix,
)
from Merak.core.checkpoint.object_update_transport import publish_object_update_epoch
from Merak.core.checkpoint.replica_service import ReplicaServiceClient, ReplicaTCPServer


class _SpyClient:
    def __init__(self, client):
        self.client = client
        self.get_source_ranks = []

    def get(self, source_rank, *args, **kwargs):
        self.get_source_ranks.append(source_rank)
        return self.client.get(source_rank, *args, **kwargs)


@contextmanager
def _services():
    with ExitStack() as stack:
        services = []
        for _ in range(2):
            server = ReplicaTCPServer(
                ("127.0.0.1", 0),
                ObjectReplicaServiceStore(8 << 20, retained_epochs_per_source=4),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            stack.callback(thread.join, timeout=5)
            stack.callback(server.server_close)
            stack.callback(server.shutdown)
            services.append((ReplicaServiceClient(*server.server_address), server))
        yield services


def _optimizer(module):
    return torch.optim.AdamW(
        module.parameters(),
        lr=0.003,
        betas=(0.8, 0.95),
        weight_decay=0.1,
        foreach=False,
        fused=False,
    )


class _HalfPrecisionOptimizer:
    """CPU stand-in with the Trainer wrapper's checkpoint/replay surface."""

    def __init__(self, module):
        self.fp16_groups = [[module.weight]]
        self.fp32_groups = [[torch.nn.Parameter(module.weight.detach().float())]]
        self.optimizer = torch.optim.AdamW(
            self.fp32_groups[0],
            lr=0.003,
            betas=(0.8, 0.95),
            weight_decay=0.1,
            foreach=False,
            fused=False,
        )
        self.dynamic_loss_scale = False
        self.cur_scale = 1.0
        self.cur_iter = self.global_step = 0

    def state_dict(self):
        return {
            "dynamic_loss_scale": self.dynamic_loss_scale,
            "cur_scale": self.cur_scale,
            "cur_iter": self.cur_iter,
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "fp32_groups": self.fp32_groups,
        }

    def load_state_dict(self, state):
        self.dynamic_loss_scale = state["dynamic_loss_scale"]
        self.cur_scale = state["cur_scale"]
        self.cur_iter = state["cur_iter"]
        self.global_step = state["global_step"]
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        for current, saved in zip(self.fp32_groups[0], state["fp32_groups"][0]):
            current.data.copy_(saved.data)

    def refresh_model_params(self):
        for master, parameter in zip(self.fp32_groups[0], self.fp16_groups[0]):
            master.grad = None
            parameter.data.copy_(master.data)


def _inner(optimizer):
    return getattr(optimizer, "optimizer", optimizer)


def _step(optimizer):
    inner = _inner(optimizer)
    inner.step()
    inner.zero_grad(set_to_none=True)
    if inner is not optimizer:
        optimizer.global_step += 1
        optimizer.cur_iter += 1
        optimizer.refresh_model_params()


def _runtime(epoch):
    return {
        "iteration": epoch,
        "random_rng_state": (3, (1, 2, epoch), None),
        "np_rng_state": ("MT19937", [epoch], 0, 0, 0.0),
        "torch_rng_state": torch.tensor([epoch], dtype=torch.uint8),
        "cuda_rng_state": torch.tensor([epoch + 1], dtype=torch.uint8),
        "rng_tracker_states": {"model": torch.tensor([epoch + 2], dtype=torch.uint8)},
        "sampler": {"consumed_samples": epoch * 8, "epoch": epoch},
    }


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _equal(first, second)
    else:
        assert left == right


def _plan():
    return build_active_gradient_plan(
        (
            GradientShard(0, "weight", "detector", (1, 2), (0, 0), (1, 2), "torch.float32"),
            GradientShard(1, "weight", "detector", (1, 2), (0, 0), (1, 2), "torch.float32"),
        ),
        (RankBinding(0, 0, 0, 0, 0), RankBinding(1, 0, 0, 1, 1)),
        arm="m1",
        detector_catalog_hash="a" * 64,
        semantic_plan_hash="b" * 64,
    )


def _base(module, optimizer, scheduler):
    binding = RecoveryBinding(0, 2, 0, 0, 0)
    return {
        "schema_version": 1,
        "kind": "merak_incremental_base",
        "iteration": 0,
        "model": copy.deepcopy(module.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "lr_scheduler": copy.deepcopy(scheduler.state_dict()),
        "parameter_ids": [["weight"]],
        "binding": asdict(binding),
        "runtime_state": _runtime(0),
    }


def _base_id(state):
    return f"sha256:{checkpoint_state_checksum(state)}"


def _records(plan, target_module, target_optimizer, base_id, epoch):
    gradient = torch.tensor([[0.25 * epoch, -0.5]], dtype=torch.float32)
    records = {}
    for rank in (0, 1):
        if rank == 0:
            module, optimizer = target_module, _inner(target_optimizer)
        else:
            module = torch.nn.Linear(2, 1, bias=False)
            optimizer = _optimizer(module)
        module.weight.grad = gradient.clone()
        if optimizer is not target_optimizer:
            optimizer.param_groups[0]["params"][0].grad = gradient.clone()
        records[rank] = capture_optimizer_update(
            optimizer,
            parameter_ids=(("weight",),),
            binding=RecoveryBinding(rank, 2, rank, 0, 0),
            base_id=base_id,
            base_epoch=0,
            epoch=epoch,
            runtime_state=_runtime(epoch),
        )
    item = plan.objects[0]
    encoding = confirm_gradient_equivalence(
        plan, {(rank, item.object_id): gradient for rank in (0, 1)}, epoch=epoch
    )
    updates = {
        rank: externalize_local_optimizer_update(plan, encoding, rank, record)
        for rank, record in records.items()
    }
    return records, encoding, updates


@pytest.mark.parametrize("wrapped", (False, True))
def test_fresh_process_restores_one_sealed_base_and_object_update_suffix(wrapped):
    plan = _plan()
    direct_module = torch.nn.Linear(2, 1, bias=False)
    direct_optimizer = (
        _HalfPrecisionOptimizer(direct_module) if wrapped else _optimizer(direct_module)
    )
    direct_scheduler = torch.optim.lr_scheduler.LambdaLR(
        _inner(direct_optimizer), lambda _: 1.0
    )
    base = _base(direct_module, direct_optimizer, direct_scheduler)
    base_id = _base_id(base)
    locators = []
    with _services() as services:
        clients = {host: value[0] for host, value in enumerate(services)}
        for epoch in (1, 2):
            records, encoding, updates = _records(
                plan, direct_module, direct_optimizer, base_id, epoch
            )
            locators.append(
                publish_object_update_epoch(
                    clients,
                    run_id="fresh-process",
                    plan=plan,
                    encoding=encoding,
                    local_updates=updates,
                ).locator
            )
            _step(direct_optimizer)
            direct_scheduler.step()

        services[0][1].shutdown()
        fresh_module = torch.nn.Linear(2, 1, bias=False)
        fresh_optimizer = (
            _HalfPrecisionOptimizer(fresh_module) if wrapped else _optimizer(fresh_module)
        )
        fresh_scheduler = torch.optim.lr_scheduler.LambdaLR(
            _inner(fresh_optimizer), lambda _: 1.0
        )
        restored_runtime = {}

        def restore_runtime(value):
            restored_runtime.update(copy.deepcopy(value))
            return value["iteration"]

        spy = _SpyClient(clients[1])
        assert restore_semantic_fixed_object_suffix(
            spy,
            surviving_host=1,
            base_state=base,
            base_id=base_id,
            base_epoch=0,
            locators=locators,
            plans_by_epoch={1: plan, 2: plan},
            logical_rank=0,
            module=fresh_module,
            optimizer=fresh_optimizer,
            scheduler=fresh_scheduler,
            runtime_restore=restore_runtime,
        ) == 2
        assert spy.get_source_ranks == [-1, 0, -1, 0]

    _equal(fresh_module.state_dict(), direct_module.state_dict())
    _equal(fresh_optimizer.state_dict(), direct_optimizer.state_dict())
    _equal(fresh_scheduler.state_dict(), direct_scheduler.state_dict())
    _equal(restored_runtime, _runtime(2))
    if wrapped:
        assert (fresh_optimizer.global_step, fresh_optimizer.cur_iter) == (2, 2)


def test_recovery_rejects_unsupported_logical_remap():
    plan = _plan()
    module = torch.nn.Linear(2, 1, bias=False)
    optimizer = _optimizer(module)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    base = _base(module, optimizer, scheduler)
    base_id = _base_id(base)
    tampered = copy.deepcopy(base)
    tampered["model"]["weight"].add_(1)
    with pytest.raises(ValueError, match="base content"):
        restore_semantic_fixed_object_suffix(
            None,
            surviving_host=0,
            base_state=tampered,
            base_id=base_id,
            base_epoch=0,
            locators=(),
            plans_by_epoch={},
            logical_rank=0,
            module=module,
            optimizer=optimizer,
            scheduler=scheduler,
            runtime_restore=lambda value: value["iteration"],
        )
    with pytest.raises(ValueError, match="cannot be remapped"):
        restore_semantic_fixed_object_suffix(
            None,
            surviving_host=0,
            base_state=base,
            base_id=base_id,
            base_epoch=0,
            locators=(),
            plans_by_epoch={},
            logical_rank=1,
            module=module,
            optimizer=optimizer,
            scheduler=scheduler,
            runtime_restore=lambda value: value["iteration"],
        )
