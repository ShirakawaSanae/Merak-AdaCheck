"""16-rank CPU/Gloo/loopback-TCP scale gate for SemanticFixed recovery."""

from __future__ import annotations

import copy
import json
import os
import random
import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from Merak.core.checkpoint.catalog import checkpoint_state_checksum
from Merak.core.checkpoint.incremental_checkpoint import RecoveryBinding, capture_optimizer_update
from Merak.core.checkpoint.matched import CONTROL_SOURCE_RANK
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
    confirm_gradient_equivalence,
)
from Merak.core.checkpoint.object_semantic_recovery import restore_semantic_fixed_object_suffix
from Merak.core.checkpoint.object_update_distributed import commit_distributed_object_update_epoch
from Merak.core.checkpoint.object_update_transport import load_object_update_locator
from Merak.core.checkpoint.replica_service import ReplicaServiceClient, ReplicaTCPServer


ARMS = ("adacheck_fixed", "host_global_fixed", "m1")
WORLD_SIZE = 16
EPOCHS = tuple(range(1, 11))


@contextmanager
def _services():
    with ExitStack() as stack:
        servers = []
        for _ in range(2):
            server = ReplicaTCPServer(
                ("127.0.0.1", 0),
                ObjectReplicaServiceStore(64 << 20, retained_epochs_per_source=16),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            stack.callback(thread.join, timeout=5)
            stack.callback(server.server_close)
            stack.callback(server.shutdown)
            servers.append(server)
        yield tuple(server.server_address for server in servers), servers


def _layout():
    return tuple(
        RankBinding(rank, rank // 4, 0, rank % 4, rank // (WORLD_SIZE // 2))
        for rank in range(WORLD_SIZE)
    )


def _stage(rank):
    return f"stage_{rank // 4}"


def _parameter_ids(rank):
    return tuple(f"{_stage(rank)}_{slot}" for slot in range(4))


def _recovery_binding(rank):
    binding = _layout()[rank]
    return RecoveryBinding(
        rank,
        WORLD_SIZE,
        binding.data_rank,
        binding.pipeline_rank,
        binding.tensor_rank,
    )


def _module(rank):
    module = torch.nn.Module()
    stage = rank // 4
    for slot in range(4):
        setattr(
            module,
            f"stage_{stage}_{slot}",
            torch.nn.Parameter(torch.tensor([1.0 + stage + slot, -2.0 - stage - slot])),
        )
    return module


def _optimizer(module):
    return torch.optim.AdamW(
        list(module.parameters()),
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
        "np_rng_state": ("MT19937", [2000 + rank * 10 + epoch], 0, 0, 0.0),
        "torch_rng_state": torch.Generator().manual_seed(3000 + rank * 10 + epoch).get_state(),
        "cuda_rng_state": torch.Generator().manual_seed(4000 + rank * 10 + epoch).get_state(),
        "rng_tracker_states": {"model": torch.Generator().manual_seed(5000 + rank * 10 + epoch).get_state()},
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
        "parameter_ids": [list(_parameter_ids(rank))],
        "binding": asdict(_recovery_binding(rank)),
        "runtime_state": _runtime(rank, 0),
    }


def _gradients(epoch):
    return {
        f"stage_{stage}_{slot}": torch.tensor([0.25 * epoch + stage + slot, -0.5 - stage - slot])
        for stage in range(4)
        for slot in range(4)
    }


def _pairs():
    return tuple((rank, rank + WORLD_SIZE // 2) for rank in range(WORLD_SIZE // 2))


def _plan(arm):
    return build_active_gradient_plan(
        tuple(
            GradientShard(rank, name, f"detector-{name}", (2,), (0,), (2,), "torch.float32")
            for rank in range(WORLD_SIZE)
            for name in _parameter_ids(rank)
        ),
        _layout(),
        arm=arm,
        detector_catalog_hash="a" * 64,
        semantic_plan_hash="b" * 64,
        fixed_pairs=_pairs(),
    )


def _encoding(plan, epoch):
    values = _gradients(epoch)
    return confirm_gradient_equivalence(
        plan,
        {
            (rank, item.object_id): values[item.fqn]
            for rank in range(WORLD_SIZE)
            for item in plan.objects
            if rank in item.holders
        },
        epoch=epoch,
    )


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


def _metrics(value):
    return {
        "remote_logical_bytes": int(value.remote_gradient_logical_bytes),
        "remote_application_bytes": int(value.total_remote_payload_bytes),
        "remote_data_payload_bytes": int(value.remote_data_payload_bytes),
        "remote_control_payload_bytes": int(value.remote_control_payload_bytes),
        "remote_wire_bytes": int(value.remote_wire_bytes),
        "maximum_remote_put_ms": float(value.maximum_remote_put_ms),
    }


def _base_metrics(value):
    return {
        "remote_logical_bytes": 0,
        "remote_application_bytes": int(value.total_remote_payload_bytes),
        "remote_data_payload_bytes": int(value.remote_data_payload_bytes),
        "remote_control_payload_bytes": int(value.remote_control_payload_bytes),
        "remote_wire_bytes": int(value.remote_wire_bytes),
        "maximum_remote_put_ms": float(value.maximum_remote_put_ms),
    }


def _aggregate(metrics):
    return {
        "remote_logical_bytes": sum(item["remote_logical_bytes"] for item in metrics),
        "remote_application_bytes": sum(item["remote_application_bytes"] for item in metrics),
        "remote_data_payload_bytes": sum(item["remote_data_payload_bytes"] for item in metrics),
        "remote_control_payload_bytes": sum(item["remote_control_payload_bytes"] for item in metrics),
        "remote_wire_bytes": sum(item["remote_wire_bytes"] for item in metrics),
        "maximum_remote_put_ms": max(item["maximum_remote_put_ms"] for item in metrics),
    }


def _worker(rank, rendezvous, endpoints, output_dir, arm):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=WORLD_SIZE)
    clients = {host: ReplicaServiceClient(*endpoint) for host, endpoint in enumerate(endpoints)}
    try:
        module, optimizer, scheduler, base = _base(rank)
        base_id = f"sha256:{checkpoint_state_checksum(base)}"
        published_base = commit_distributed_object_semantic_base(
            clients,
            run_id=f"service-scale-base-{arm}",
            epoch=0,
            local_base=base,
            rank_bindings=_layout(),
            locator_path=output_dir / "base.json",
            group=dist.group.WORLD,
        )
        suffix_metrics = []
        for epoch in EPOCHS:
            binding = _layout()[rank]
            plan = gather_active_gradient_plan(
                binding,
                tuple(
                    GradientShard(rank, name, f"detector-{name}", (2,), (0,), (2,), "torch.float32")
                    for name in _parameter_ids(rank)
                ),
                arm=arm,
                detector_catalog_hash="a" * 64,
                semantic_plan_hash="b" * 64,
                fixed_pairs=_pairs(),
                group=dist.group.WORLD,
            )
            gradients = _gradients(epoch)
            encoding = gather_confirmed_gradient_encoding(
                plan,
                {item.object_id: gradients[item.fqn] for item in plan.objects if rank in item.holders},
                epoch=epoch,
                group=dist.group.WORLD,
            )
            for name in _parameter_ids(rank):
                getattr(module, name).grad = gradients[name].clone()
            update = externalize_local_optimizer_update(
                plan,
                encoding,
                rank,
                capture_optimizer_update(
                    optimizer,
                    parameter_ids=(_parameter_ids(rank),),
                    binding=_recovery_binding(rank),
                    base_id=base_id,
                    base_epoch=0,
                    epoch=epoch,
                    runtime_state=_runtime(rank, epoch),
                ),
            )
            published = commit_distributed_object_update_epoch(
                clients,
                run_id=f"service-scale-suffix-{arm}",
                plan=plan,
                encoding=encoding,
                update=update,
                locator_path=output_dir / f"suffix-{epoch}.json",
                group=dist.group.WORLD,
            )
            suffix_metrics.append(_metrics(published.metrics))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        if rank == 0:
            (output_dir / "training-metrics.json").write_text(json.dumps({
                "base": _base_metrics(published_base.metrics), "suffix": _aggregate(suffix_metrics),
            }, indent=2, sort_keys=True))
    finally:
        dist.destroy_process_group()


def _expected(rank):
    module, optimizer, scheduler, _ = _base(rank)
    for epoch in EPOCHS:
        for name in _parameter_ids(rank):
            getattr(module, name).grad = _gradients(epoch)[name]
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
    return module, optimizer, scheduler


def _fresh(rank):
    module = _module(rank)
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


def _target_sources(plan, encoding, logical_rank, surviving_host):
    references = {}
    for action in encoding.actions:
        references.setdefault(
            (action.object_id, action.source_rank if action.action == "direct_fallback" else None),
            action,
        )
    publishers = set()
    for item in plan.objects:
        if logical_rank not in item.holders:
            continue
        reference = references[(item.object_id, logical_rank if item.object_id in encoding.mismatched_object_ids else None)]
        publishers.add(next(
            action.source_rank for action in encoding.actions
            if action.object_id == item.object_id
            and action.target_host == surviving_host
            and action.version_id == reference.version_id
            and action.action == reference.action
            and (reference.action != "direct_fallback" or action.source_rank == reference.source_rank)
        ))
    return sorted({logical_rank, *publishers})


class _AuditClient:
    def __init__(self, client):
        self.client = client
        self.calls = []

    def get(self, source_rank, epoch, *args, **kwargs):
        started = time.perf_counter()
        envelope, metrics = self.client.get(source_rank, epoch, *args, **kwargs)
        self.calls.append({
            "source_rank": source_rank,
            "epoch": epoch,
            "application_bytes": len(envelope.payload),
            "wire_bytes": metrics.wire_bytes,
            "get_ms": (time.perf_counter() - started) * 1000.0,
        })
        return envelope, metrics


def _recovery_metrics(calls):
    return {
        "get_count": len(calls),
        "application_bytes": sum(item["application_bytes"] for item in calls),
        "wire_bytes": sum(item["wire_bytes"] for item in calls),
        "maximum_get_ms": max((item["get_ms"] for item in calls), default=0.0),
    }


def _retained(stats):
    base = [item for item in stats["records"] if item["epoch"] == 0]
    suffix = [item for item in stats["records"] if item["epoch"] in EPOCHS]
    assert len(base) == WORLD_SIZE + 1
    assert len(suffix) == len(EPOCHS) * (WORLD_SIZE + 1)
    assert stats["record_count"] == len(base) + len(suffix)
    assert stats["bytes_used"] == sum(item["payload_bytes"] for item in stats["records"])
    return {
        "bytes_used": int(stats["bytes_used"]),
        "record_count": int(stats["record_count"]),
        "base": {"bytes": sum(item["payload_bytes"] for item in base), "entries": len(base)},
        "suffix": {"bytes": sum(item["payload_bytes"] for item in suffix), "entries": len(suffix)},
    }


def _run_case(tmp_path, arm, failed_host):
    case_dir = tmp_path / f"{arm}-failed-{failed_host}"
    case_dir.mkdir()
    with _services() as (endpoints, servers):
        mp.spawn(_worker, args=(str(case_dir / "rendezvous"), endpoints, case_dir, arm), nprocs=WORLD_SIZE)
        training = json.loads((case_dir / "training-metrics.json").read_text())
        plan = _plan(arm)
        plans = {epoch: plan for epoch in EPOCHS}
        encodings = {epoch: _encoding(plans[epoch], epoch) for epoch in EPOCHS}
        suffixes = tuple(load_object_update_locator(case_dir / f"suffix-{epoch}.json") for epoch in EPOCHS)
        base_locator = load_object_semantic_base_locator(case_dir / "base.json")
        before_loss = {str(host): _retained(ReplicaServiceClient(*endpoints[host]).stats()) for host in (0, 1)}
        servers[failed_host].shutdown()
        surviving_host = 1 - failed_host
        recovery = {"base": [], "suffix": [], "rank_epoch_sources": {}}
        for rank in range(WORLD_SIZE):
            spy = _AuditClient(ReplicaServiceClient(*endpoints[surviving_host]))
            _, _, _, expected_base = _base(rank)
            restored_base = restore_object_semantic_base(
                spy,
                surviving_host=surviving_host,
                locator=base_locator,
                logical_rank=rank,
                expected_rank_bindings=_layout(),
                expected_parameter_ids=(_parameter_ids(rank),),
                expected_base_id=f"sha256:{checkpoint_state_checksum(expected_base)}",
            )
            base_calls = spy.calls[:]
            assert [(item["source_rank"], item["epoch"]) for item in base_calls] == [
                (CONTROL_SOURCE_RANK, 0), (rank, 0)
            ]
            module, optimizer, scheduler, sampler, runtime, restore_runtime = _fresh(rank)
            assert restore_semantic_fixed_object_suffix(
                spy,
                surviving_host=surviving_host,
                base_state=restored_base.state,
                base_id=restored_base.base_id,
                base_epoch=0,
                locators=suffixes,
                plans_by_epoch=plans,
                logical_rank=rank,
                module=module,
                optimizer=optimizer,
                scheduler=scheduler,
                runtime_restore=restore_runtime,
            ) == EPOCHS[-1]
            expected_module, expected_optimizer, expected_scheduler = _expected(rank)
            _equal(module.state_dict(), expected_module.state_dict())
            _equal(optimizer.state_dict(), expected_optimizer.state_dict())
            _equal(scheduler.state_dict(), expected_scheduler.state_dict())
            _equal(runtime, _runtime(rank, EPOCHS[-1]))
            assert (sampler.consumed_samples, sampler.epoch) == (EPOCHS[-1] * 32 + rank, EPOCHS[-1])
            suffix_calls = spy.calls[len(base_calls):]
            recovery["base"].extend(base_calls)
            recovery["suffix"].extend(suffix_calls)
            for epoch in EPOCHS:
                observed = [item["source_rank"] for item in suffix_calls if item["epoch"] == epoch]
                expected = [CONTROL_SOURCE_RANK, *_target_sources(plans[epoch], encodings[epoch], rank, surviving_host)]
                assert observed == expected
                assert observed.count(CONTROL_SOURCE_RANK) == 1
                recovery["rank_epoch_sources"][f"{rank}:{epoch}"] = {"expected": expected, "observed": observed}
        return {
            "failed_host": failed_host,
            "surviving_host": surviving_host,
            "plan": {
                "publisher_policy": plan.publisher_policy,
                "total_remote_logical_bytes": plan.total_remote_logical_bytes,
                "maximum_remote_publisher_bytes": plan.maximum_remote_publisher_bytes,
            },
            "service_retained_before_loss": before_loss,
            "recovery": {
                "base": _recovery_metrics(recovery["base"]),
                "suffix": _recovery_metrics(recovery["suffix"]),
                "rank_epoch_sources": recovery["rank_epoch_sources"],
            },
            "training": training,
        }


def test_service_scale_16_rank_three_arm_ten_suffixes_after_each_host_loss(tmp_path):
    results = {arm: [_run_case(tmp_path, arm, failed_host) for failed_host in (0, 1)] for arm in ARMS}
    for failed_host in range(2):
        ada = results["adacheck_fixed"][failed_host]["training"]["suffix"]
        fixed = results["host_global_fixed"][failed_host]["training"]["suffix"]
        m1 = results["m1"][failed_host]["training"]["suffix"]
        assert ada["remote_logical_bytes"] > fixed["remote_logical_bytes"]
        assert fixed["remote_logical_bytes"] == m1["remote_logical_bytes"]
        assert (
            results["m1"][failed_host]["plan"]["maximum_remote_publisher_bytes"]
            < results["host_global_fixed"][failed_host]["plan"]["maximum_remote_publisher_bytes"]
        )
    summary_path = os.environ.get("OBJECT_SEMANTIC_SCALE_SUMMARY")
    if summary_path:
        destination = Path(summary_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({
            "gate": "object-semantic-service-scale-r120",
            "environment": "CPU/Gloo/loopback-TCP only",
            "world_size": WORLD_SIZE,
            "epochs": len(EPOCHS),
            "arms": results,
        }, indent=2, sort_keys=True) + "\n")
