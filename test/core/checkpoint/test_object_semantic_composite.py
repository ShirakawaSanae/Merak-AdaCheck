"""CPU/Gloo composite gate for sealed SemanticFixed base-plus-suffix recovery."""

from __future__ import annotations

import copy
import random
import threading
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from Merak.core.checkpoint.catalog import checkpoint_state_checksum
from Merak.core.checkpoint.incremental_checkpoint import (
    RecoveryBinding,
    capture_optimizer_update,
)
from Merak.core.checkpoint.object_optimizer_update import externalize_local_optimizer_update
from Merak.core.checkpoint.object_replica_service import ObjectReplicaServiceStore
from Merak.core.checkpoint.object_semantic_base import (
    commit_distributed_object_semantic_base,
    load_object_semantic_base_locator,
    restore_object_semantic_base,
)
from Merak.core.checkpoint.object_semantic_distributed import (
    gather_active_gradient_plan,
    gather_confirmed_gradient_encoding,
)
from Merak.core.checkpoint.object_semantic_plan import (
    GradientShard,
    RankBinding,
    build_active_gradient_plan,
)
from Merak.core.checkpoint.object_semantic_recovery import restore_semantic_fixed_object_suffix
from Merak.core.checkpoint.object_update_distributed import commit_distributed_object_update_epoch
from Merak.core.checkpoint.object_update_transport import load_object_update_locator
from Merak.core.checkpoint.replica_service import ReplicaServiceClient, ReplicaTCPServer


ARMS = ("adacheck_fixed", "host_global_fixed", "m1")
WORLD_SIZE = 4


@contextmanager
def _services():
    with ExitStack() as stack:
        servers = []
        for _ in range(2):
            server = ReplicaTCPServer(
                ("127.0.0.1", 0),
                ObjectReplicaServiceStore(64 << 20, retained_epochs_per_source=4),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            stack.callback(thread.join, timeout=5)
            stack.callback(server.server_close)
            stack.callback(server.shutdown)
            servers.append(server)
        yield tuple(server.server_address for server in servers), servers


def _layout():
    return tuple(RankBinding(rank, 0, 0, rank, rank // 2) for rank in range(WORLD_SIZE))


def _module(rank):
    module = torch.nn.Module()
    module.left = torch.nn.Parameter(torch.tensor([1.0 + rank, -2.0]))
    module.right = torch.nn.Parameter(torch.tensor([3.0, -4.0 - rank]))
    return module


def _optimizer(module):
    return torch.optim.AdamW(
        [module.left, module.right],
        lr=0.003,
        betas=(0.8, 0.95),
        weight_decay=0.1,
        foreach=False,
        fused=False,
    )


def _runtime(rank, epoch):
    return {
        "iteration": epoch,
        "random_rng_state": random.Random(1000 + rank * 10 + epoch).getstate(),
        "np_rng_state": np.random.RandomState(2000 + rank * 10 + epoch).get_state(),
        "torch_rng_state": torch.Generator().manual_seed(3000 + rank * 10 + epoch).get_state(),
        "cuda_rng_state": torch.Generator().manual_seed(4000 + rank * 10 + epoch).get_state(),
        "rng_tracker_states": {
            "model": torch.Generator().manual_seed(5000 + rank * 10 + epoch).get_state()
        },
        "sampler": {"consumed_samples": epoch * 32 + rank, "epoch": epoch},
    }


def _base(rank):
    module = _module(rank)
    optimizer = _optimizer(module)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return module, optimizer, scheduler, {
        "schema_version": 1,
        "kind": "merak_incremental_base",
        "iteration": 0,
        "model": copy.deepcopy(module.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "lr_scheduler": copy.deepcopy(scheduler.state_dict()),
        "parameter_ids": [["left", "right"]],
        "binding": asdict(RecoveryBinding(rank, WORLD_SIZE, rank, 0, 0)),
        "runtime_state": _runtime(rank, 0),
    }


def _gradients(epoch):
    return {
        "left": torch.tensor([0.25 * epoch, -0.5]),
        "right": torch.tensor([-0.75, 0.125 * epoch]),
    }


def _plan(arm):
    return build_active_gradient_plan(
        tuple(
            GradientShard(rank, name, f"detector-{name}", (2,), (0,), (2,), "torch.float32")
            for rank in range(WORLD_SIZE)
            for name in ("left", "right")
        ),
        _layout(),
        arm=arm,
        detector_catalog_hash="a" * 64,
        semantic_plan_hash="b" * 64,
        fixed_pairs=((0, 2), (1, 3)),
    )


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
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


def _worker(rank, rendezvous, endpoints, output_dir, arm):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=WORLD_SIZE
    )
    clients = {host: ReplicaServiceClient(*endpoint) for host, endpoint in enumerate(endpoints)}
    try:
        module, optimizer, scheduler, base = _base(rank)
        base_id = f"sha256:{checkpoint_state_checksum(base)}"
        commit_distributed_object_semantic_base(
            clients,
            run_id=f"composite-base-{arm}",
            epoch=0,
            local_base=base,
            rank_bindings=_layout(),
            locator_path=output_dir / "base.json",
            group=dist.group.WORLD,
        )
        for epoch in (1, 2):
            binding = _layout()[rank]
            plan = gather_active_gradient_plan(
                binding,
                tuple(
                    GradientShard(rank, name, f"detector-{name}", (2,), (0,), (2,), "torch.float32")
                    for name in ("left", "right")
                ),
                arm=arm,
                detector_catalog_hash="a" * 64,
                semantic_plan_hash="b" * 64,
                fixed_pairs=((0, 2), (1, 3)),
                group=dist.group.WORLD,
            )
            gradients = _gradients(epoch)
            encoding = gather_confirmed_gradient_encoding(
                plan,
                {
                    item.object_id: gradients[item.fqn]
                    for item in plan.objects
                    if rank in item.holders
                },
                epoch=epoch,
                group=dist.group.WORLD,
            )
            for parameter, name in zip((module.left, module.right), ("left", "right")):
                parameter.grad = gradients[name].clone()
            record = capture_optimizer_update(
                optimizer,
                parameter_ids=(("left", "right"),),
                binding=RecoveryBinding(rank, WORLD_SIZE, rank, 0, 0),
                base_id=base_id,
                base_epoch=0,
                epoch=epoch,
                runtime_state=_runtime(rank, epoch),
            )
            update = externalize_local_optimizer_update(plan, encoding, rank, record)
            commit_distributed_object_update_epoch(
                clients,
                run_id=f"composite-suffix-{arm}",
                plan=plan,
                encoding=encoding,
                update=update,
                locator_path=output_dir / f"suffix-{epoch}.json",
                group=dist.group.WORLD,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
    finally:
        dist.destroy_process_group()


def _expected(rank):
    module, optimizer, scheduler, _ = _base(rank)
    for epoch in (1, 2):
        for parameter, name in zip((module.left, module.right), ("left", "right")):
            parameter.grad = _gradients(epoch)[name]
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
    return module, optimizer, scheduler


def _fresh():
    module = _module(99)
    optimizer = _optimizer(module)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    sampler = SimpleNamespace(consumed_samples=-1, epoch=-1)
    runtime = {}

    def restore_runtime(value):
        runtime.update(copy.deepcopy(value))
        sampler.consumed_samples = value["sampler"]["consumed_samples"]
        sampler.epoch = value["sampler"]["epoch"]
        return value["iteration"]

    return module, optimizer, scheduler, sampler, runtime, restore_runtime


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("failed_host", (0, 1))
def test_composite_sealed_base_two_suffixes_fresh_recovery_after_host_loss(
    tmp_path, arm, failed_host
):
    with _services() as (endpoints, servers):
        mp.spawn(
            _worker,
            args=(str(tmp_path / "rendezvous"), endpoints, tmp_path, arm),
            nprocs=WORLD_SIZE,
        )
        base_locator = load_object_semantic_base_locator(tmp_path / "base.json")
        suffixes = tuple(
            load_object_update_locator(tmp_path / f"suffix-{epoch}.json")
            for epoch in (1, 2)
        )
        servers[failed_host].shutdown()
        survivor = 1 - failed_host
        client = ReplicaServiceClient(*endpoints[survivor])
        expected_module, expected_optimizer, expected_scheduler = _expected(0)
        _, _, _, expected_base = _base(0)
        restored_base = restore_object_semantic_base(
            client,
            surviving_host=survivor,
            locator=base_locator,
            logical_rank=0,
            expected_rank_bindings=_layout(),
            expected_parameter_ids=(("left", "right"),),
            expected_base_id=f"sha256:{checkpoint_state_checksum(expected_base)}",
        )
        module, optimizer, scheduler, sampler, runtime, restore_runtime = _fresh()
        assert restore_semantic_fixed_object_suffix(
            client,
            surviving_host=survivor,
            base_state=restored_base.state,
            base_id=restored_base.base_id,
            base_epoch=0,
            locators=suffixes,
            plans_by_epoch={1: _plan(arm), 2: _plan(arm)},
            logical_rank=0,
            module=module,
            optimizer=optimizer,
            scheduler=scheduler,
            runtime_restore=restore_runtime,
        ) == 2
        _equal(module.state_dict(), expected_module.state_dict())
        _equal(optimizer.state_dict(), expected_optimizer.state_dict())
        _equal(scheduler.state_dict(), expected_scheduler.state_dict())
        _equal(runtime, _runtime(0, 2))
        assert (sampler.consumed_samples, sampler.epoch) == (64, 2)

        tampered = copy.deepcopy(restored_base.state)
        tampered["model"]["left"].add_(1)
        bad_module, bad_optimizer, bad_scheduler, _, _, bad_runtime = _fresh()
        before = copy.deepcopy(bad_module.state_dict())
        with pytest.raises(ValueError, match="base content"):
            restore_semantic_fixed_object_suffix(
                client,
                surviving_host=survivor,
                base_state=tampered,
                base_id=restored_base.base_id,
                base_epoch=0,
                locators=suffixes,
                plans_by_epoch={1: _plan(arm), 2: _plan(arm)},
                logical_rank=0,
                module=bad_module,
                optimizer=bad_optimizer,
                scheduler=bad_scheduler,
                runtime_restore=bad_runtime,
            )
        _equal(bad_module.state_dict(), before)

        gap_module, gap_optimizer, gap_scheduler, _, _, gap_runtime = _fresh()
        with pytest.raises(ValueError, match="contiguous suffix"):
            restore_semantic_fixed_object_suffix(
                client,
                surviving_host=survivor,
                base_state=restored_base.state,
                base_id=restored_base.base_id,
                base_epoch=0,
                locators=(suffixes[1],),
                plans_by_epoch={2: _plan(arm)},
                logical_rank=0,
                module=gap_module,
                optimizer=gap_optimizer,
                scheduler=gap_scheduler,
                runtime_restore=gap_runtime,
            )
