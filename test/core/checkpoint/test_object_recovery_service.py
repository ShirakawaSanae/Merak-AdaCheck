from contextlib import ExitStack, contextmanager
import hashlib
import json
import random
from types import SimpleNamespace
import threading

import numpy as np
import pytest
import torch

from Merak.core.amp.amp_optimizer import HalfPrecisionOptimizer
from Merak.core.checkpoint.matched import (
    commit_control,
    commit_data_shard,
    reduced_catalog_hash,
)
from Merak.core.checkpoint.object_protection_runtime import (
    build_protected_object_runtime,
)
from Merak.core.checkpoint.object_recovery_service import (
    SameLayoutServiceMetadata,
    build_same_layout_service_controller,
    load_same_layout_service_metadata,
    write_same_layout_service_metadata,
)
from Merak.core.checkpoint.peer_delta import (
    build_peer_delta_matrix,
    peer_grouping_for_policy,
)
from Merak.core.checkpoint.peer_delta_runtime import (
    build_peer_delta_control_state,
    build_peer_delta_protection_plan,
    build_peer_delta_shard,
)
from Merak.core.checkpoint.reduced import build_reduced_checkpoint
from Merak.core.checkpoint.replica_service import (
    ReplicaServiceClient,
    ReplicaTCPServer,
    ServiceReplicaStore,
)


class _CPUHalfPrecision:
    def __init__(self, module, *, initialized):
        master = torch.nn.Parameter(module.weight.detach().float().clone())
        self.fp16_groups = [[module.weight]]
        self.fp32_groups = [[master]]
        self.optimizer = torch.optim.AdamW(
            [master], lr=1e-3, foreach=False, fused=False
        )
        self.dynamic_loss_scale = False
        self.cur_scale = 128.0
        self.cur_iter = 7 if initialized else 0
        self.global_step = 7 if initialized else 0
        if initialized:
            master.grad = torch.ones_like(master)
            self.optimizer.step()

    def state_dict(self):
        return {
            "dynamic_loss_scale": self.dynamic_loss_scale,
            "cur_scale": self.cur_scale,
            "cur_iter": self.cur_iter,
            "global_step": self.global_step,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "fp32_groups": self.fp32_groups,
        }

    def load_state_dict(self, state, load_optimizer_states=True):
        return HalfPrecisionOptimizer.load_state_dict(
            self, state, load_optimizer_states=load_optimizer_states
        )


class _Scheduler:
    def __init__(self, value=0):
        self.value = value

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state):
        if set(state) != {"value"}:
            raise ValueError("invalid scheduler state")
        self.value = int(state["value"])


@contextmanager
def _service():
    server = ReplicaTCPServer(("127.0.0.1", 0), ServiceReplicaStore(max_bytes=8 << 20))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, thread, ReplicaServiceClient(*server.server_address)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _module(value):
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(
        torch.full((2, 2), value, dtype=torch.bfloat16)
    )
    return module


def _rank_state(rank):
    module = _module(rank + 1)
    optimizer = _CPUHalfPrecision(module, initialized=True)
    scheduler = _Scheduler(7)
    return {
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(),
        "iteration": 7,
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.arange(4, dtype=torch.uint8),
        "rng_tracker_states": {"model-parallel-rng": torch.arange(4, dtype=torch.uint8)},
        "sampler": {"consumed_samples": 112 + rank, "epoch": 3},
        "checkpoint_version": 0,
    }


def _runtime(states, *, arm="adacheck_object_fixed"):
    reduced = build_reduced_checkpoint(states)
    matrix = build_peer_delta_matrix(
        reduced, ranks_per_host=1, catalog_hash=reduced_catalog_hash(reduced)
    )
    protection = build_peer_delta_protection_plan(
        reduced,
        peer_grouping_for_policy(matrix, "fixed"),
        ranks_per_host=1,
        availability_scope="host",
        balance_publishers=arm == "m1_object",
    )
    runtime = build_protected_object_runtime(
        reduced,
        protection,
        arm=arm,
        epoch=7,
        semantic_plan_hash="s" * 64,
        encoder_name="semantic_fixed_same_layout_v1",
        source_layout_hash="pp1-tp1-dp2",
    )
    return build_protected_object_runtime(
        reduced,
        protection,
        arm=arm,
        epoch=7,
        semantic_plan_hash="s" * 64,
        encoder_name="semantic_fixed_same_layout_v1",
        source_layout_hash=runtime.layout.layout_hash,
    )


def _commit(runtime, clients):
    records = []
    reduced, protection = runtime.reduced, runtime.protection
    holders = {
        rank: {
            item.value_id for item in reduced.objects if rank in item.source_ranks
        }
        for rank in reduced.templates
    }
    for assignment in protection.assignments:
        shard = build_peer_delta_shard(
            reduced,
            protection,
            source_rank=assignment.source_rank,
            target_host=assignment.target_host,
            tensors={key: reduced.tensors[key] for key in holders[assignment.source_rank]},
        )
        record, _ = commit_data_shard(
            clients[assignment.target_host],
            epoch=7,
            shard={
                **shard,
                "publisher_rank": assignment.source_rank,
                "host_index": assignment.target_host,
            },
            group_ranks=(0, 1),
        )
        records.append(record)
    commit_control(
        clients,
        epoch=7,
        control_state=build_peer_delta_control_state(
            reduced, protection, tuple(records)
        ),
    )


def test_tcp_factory_restores_fresh_same_rank_state_from_only_surviving_service(
    tmp_path, monkeypatch
):
    states = {rank: _rank_state(rank) for rank in range(2)}
    runtime = _runtime(states)
    metadata_path = tmp_path / "object-service-metadata.json"
    write_same_layout_service_metadata(metadata_path, runtime)
    metadata_json = json.loads(metadata_path.read_text())
    assert set(metadata_json) == {
        "schema_version",
        "scope",
        "control",
        "arm",
        "protection_plan",
        "semantic_plan_hash",
        "encoder_name",
        "source_layout_hash",
        "target_layout_hash",
        "recovery_plan_hash",
        "metadata_hash",
    }
    assert "tensors" not in metadata_path.read_text()

    with ExitStack() as stack:
        services = [stack.enter_context(_service()) for _ in range(2)]
        clients = [service[2] for service in services]
        _commit(runtime, clients)
        endpoints = [
            f"{server.server_address[0]}:{server.server_address[1]}"
            for server, _, _ in services
        ]
        failed_server, failed_thread, _ = services[0]
        failed_server.shutdown()
        failed_server.server_close()
        failed_thread.join(timeout=5)

        target = _module(9)
        optimizer = _CPUHalfPrecision(target, initialized=False)
        scheduler = _Scheduler()
        sampler = SimpleNamespace(consumed_samples=0, epoch=0)
        args = SimpleNamespace(
            fp16=True,
            finetune=False,
            no_load_optim=False,
            zero_stage=None,
            no_load_rng=False,
        )
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(
            "Merak.core.checkpoint.object_recovery_service.dist.get_rank", lambda: 0
        )
        monkeypatch.setenv("CKPT_OBJECT_RECOVERY_METADATA", str(metadata_path))
        monkeypatch.setenv("CKPT_OBJECT_RECOVERY_ARM", "fixed")
        monkeypatch.setenv("CKPT_REPLICA_ENDPOINTS", json.dumps(endpoints))
        monkeypatch.setenv("CKPT_OBJECT_FAILED_HOSTS", "[0]")

        controller = build_same_layout_service_controller(
            module=target,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            args=args,
        )
        assert set(controller.clients) == {1}
        assert controller.restore(7) == 7

    expected = states[0]
    assert torch.equal(target.weight, expected["model"]["weight"])
    assert scheduler.state_dict() == expected["lr_scheduler"]
    assert (sampler.consumed_samples, sampler.epoch) == (112, 3)
    assert optimizer.global_step == 7
    target_master = optimizer.fp32_groups[0][0]
    expected_master = expected["optimizer"]["fp32_groups"][0][0]
    assert torch.equal(target_master, expected_master)
    assert set(optimizer.optimizer.state[target_master]) == {
        "step",
        "exp_avg",
        "exp_avg_sq",
    }


def test_metadata_and_factory_reject_unknown_keys_and_arms(tmp_path, monkeypatch):
    states = {rank: _rank_state(rank) for rank in range(2)}
    metadata = SameLayoutServiceMetadata.from_runtime(_runtime(states))
    m1_metadata = SameLayoutServiceMetadata.from_runtime(
        _runtime(states, arm="m1_object")
    )
    assert set(metadata.to_dict()) == set(m1_metadata.to_dict())
    assert m1_metadata.arm == "m1_object"
    changed = metadata.to_dict()
    changed["unknown"] = True
    with pytest.raises(ValueError, match="unknown or missing"):
        SameLayoutServiceMetadata.from_dict(changed)

    path = tmp_path / "metadata.json"
    write_same_layout_service_metadata(path, _runtime(states))
    assert load_same_layout_service_metadata(path).arm == "adacheck_object_fixed"
    monkeypatch.setenv("CKPT_OBJECT_RECOVERY_METADATA", str(path))
    monkeypatch.setenv("CKPT_OBJECT_RECOVERY_ARM", "other")
    monkeypatch.setenv("CKPT_REPLICA_ENDPOINTS", '["127.0.0.1:1", "127.0.0.1:2"]')
    monkeypatch.setenv("CKPT_OBJECT_FAILED_HOSTS", "[0]")
    with pytest.raises(ValueError, match="must be fixed or m1"):
        build_same_layout_service_controller(
            module=object(),
            optimizer=object(),
            scheduler=object(),
            sampler=object(),
            args=object(),
        )


def test_factory_rejects_different_layout_before_restore(tmp_path, monkeypatch):
    metadata = SameLayoutServiceMetadata.from_runtime(
        _runtime({rank: _rank_state(rank) for rank in range(2)})
    )
    changed = metadata.to_dict()
    changed["target_layout_hash"] = "different-layout"
    body = {key: value for key, value in changed.items() if key != "metadata_hash"}
    changed["metadata_hash"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "mismatched-layout.json"
    path.write_text(json.dumps(changed))

    target = _module(9)
    before = target.weight.detach().clone()
    monkeypatch.setenv("CKPT_OBJECT_RECOVERY_METADATA", str(path))
    monkeypatch.setenv("CKPT_OBJECT_RECOVERY_ARM", "fixed")
    monkeypatch.setenv("CKPT_REPLICA_ENDPOINTS", '["127.0.0.1:1", "127.0.0.1:2"]')
    monkeypatch.setenv("CKPT_OBJECT_FAILED_HOSTS", "[0]")
    with pytest.raises(ValueError, match="identical source and target layouts"):
        build_same_layout_service_controller(
            module=target,
            optimizer=object(),
            scheduler=object(),
            sampler=object(),
            args=object(),
        )
    assert torch.equal(target.weight, before)


def test_metadata_rejects_mislabeled_publisher_policy():
    states = {rank: _rank_state(rank) for rank in range(2)}
    fixed = _runtime(states)
    mislabeled = build_protected_object_runtime(
        fixed.reduced,
        build_peer_delta_protection_plan(
            fixed.reduced,
            fixed.protection.grouping,
            ranks_per_host=1,
            availability_scope="host",
            balance_publishers=True,
        ),
        arm="adacheck_object_fixed",
        epoch=7,
        semantic_plan_hash="s" * 64,
        encoder_name="semantic_fixed_same_layout_v1",
        source_layout_hash="pp1-tp1-dp2",
    )

    with pytest.raises(ValueError, match="arm and publisher policy"):
        SameLayoutServiceMetadata.from_runtime(mislabeled)
