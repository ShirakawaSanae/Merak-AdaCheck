import copy
import json
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from Merak.core.checkpoint.catalog import checkpoint_state_checksum
from Merak.core.checkpoint.global_commit import manifest_evidence
from Merak.core.checkpoint.incremental_checkpoint import (
    RecoveryBinding,
    capture_optimizer_update,
)
from Merak.core.checkpoint.matched import CONTROL_SOURCE_RANK
from Merak.core.checkpoint.object_optimizer_update import externalize_local_optimizer_update
from Merak.core.checkpoint.object_replica_service import ObjectReplicaServiceStore
from Merak.core.checkpoint.object_semantic_base import (
    ObjectSemanticBaseLocator,
    write_object_semantic_base_locator,
)
from Merak.core.checkpoint.object_semantic_plan import (
    GradientShard,
    RankBinding,
    build_active_gradient_plan,
    confirm_gradient_equivalence,
)
from Merak.core.checkpoint.object_semantic_restore import (
    SemanticFixedObjectRestoreController,
    build_object_semantic_restore_controller,
)
from Merak.core.checkpoint.object_update_transport import (
    commit_object_update_state,
    publish_object_update_epoch,
    seal_object_update_host,
    write_object_update_locator,
)
from Merak.core.checkpoint.replica_service import ReplicaServiceClient, ReplicaTCPServer


@contextmanager
def _services():
    with ExitStack() as stack:
        services = []
        for _ in range(2):
            server = ReplicaTCPServer(
                ("127.0.0.1", 0),
                ObjectReplicaServiceStore(16 << 20, retained_epochs_per_source=4),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            stack.callback(thread.join, timeout=5)
            stack.callback(server.server_close)
            stack.callback(server.shutdown)
            services.append((ReplicaServiceClient(*server.server_address), server))
        yield services


def _optimizer(module):
    return torch.optim.AdamW(module.parameters(), lr=0.003, foreach=False, fused=False)


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


def _runtime(epoch):
    return {
        "iteration": epoch,
        "random_rng_state": (3, (epoch,), None),
        "np_rng_state": ("MT19937", [epoch], 0, 0, 0.0),
        "torch_rng_state": torch.tensor([epoch], dtype=torch.uint8),
        "cuda_rng_state": torch.tensor([epoch], dtype=torch.uint8),
        "rng_tracker_states": {"model": torch.tensor([epoch], dtype=torch.uint8)},
        "sampler": {"consumed_samples": epoch * 2, "epoch": epoch},
    }


def _base(module, optimizer, scheduler, rank):
    return {
        "schema_version": 1,
        "kind": "merak_incremental_base",
        "iteration": 0,
        "model": copy.deepcopy(module.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "lr_scheduler": copy.deepcopy(scheduler.state_dict()),
        "parameter_ids": [["weight"]],
        "binding": asdict(RecoveryBinding(rank, 2, rank, 0, 0)),
        "runtime_state": _runtime(0),
        "run_id": "fresh-semantic",
        "adacheck_semantic": {"catalog_hash": "a" * 64},
        "adacheck_semantic_encoder": {
            "schema_version": 1,
            "kind": "adacheck_semantic_fixed",
            "plan_hash": "b" * 64,
            "trainable_action": "gradient_bf16",
            "gradient_dtype": "torch.bfloat16",
            "gradient_snapshot_point": "raw_bf16_pre_unscale",
            "gradient_transform": "upcast_fp32_times_exact_step_scale",
            "metadata_action": "metadata_exact",
            "protection_action": "object_semantic_runtime",
            "remote_trainable_payload": "semantic_gradient_bf16",
            "peer_equivalence": "post_allreduce_dp_invariant",
            "object_partition": "fqn_global_slice_pp_tp",
            "dp_equivalence_basis": "post_allreduce_training_invariant",
            "integrity": "semantic_tensor_sha256_plus_serialized_payload_sha256",
            "rank_action_count": 1,
            "global_action_count": 2,
        },
        "adacheck_detector_evidence": {"frozen": "yes"},
    }


def _layout():
    return (RankBinding(0, 0, 0, 0, 0), RankBinding(1, 0, 0, 1, 1))


def _publish_base(clients, root, states):
    manifests = {host: {} for host in (0, 1)}
    ids = {}
    for rank, state in states.items():
        ids[rank] = f"sha256:{checkpoint_state_checksum(state)}"
        for host in (0, 1):
            bundle = {
                "schema_version": 1,
                "kind": "object_semantic_base_snapshot",
                "run_id": "fresh-semantic",
                "epoch": 0,
                "source_rank": rank,
                "source_host": rank,
                "binding": state["binding"],
                "parameter_ids": state["parameter_ids"],
                "base_id": ids[rank],
                "snapshot": state,
            }
            manifests[host][rank] = commit_object_update_state(
                clients[host], bundle, source_rank=rank, source_host=rank,
                target_host=host, target_rank=host, group_ranks=(0, 1),
            )[0]
    bases_by_host = []
    evidence_by_host = {}
    for host in (0, 1):
        records = []
        evidence_by_host[host] = []
        for rank in (0, 1):
            evidence = manifest_evidence(manifests[host][rank])
            evidence_by_host[host].append(evidence)
            records.append({
                **evidence, "rank": rank, "base_id": ids[rank],
                "binding": states[rank]["binding"],
                "parameter_ids": states[rank]["parameter_ids"],
            })
        bases_by_host.append([host, records])
    control = {
        "schema_version": 1,
        "kind": "object_semantic_base_control",
        "run_id": "fresh-semantic",
        "epoch": 0,
        "world_size": 2,
        "rank_bindings": [asdict(item) for item in _layout()],
        "bases_by_host": bases_by_host,
    }
    controls = {}
    for host in (0, 1):
        controls[host] = commit_object_update_state(
            clients[host], control, source_rank=CONTROL_SOURCE_RANK, source_host=0,
            target_host=host, target_rank=host, group_ranks=(0, 1),
        )[0]
        seal_object_update_host(
            clients[host], epoch=0,
            control_checksum=controls[host].payload_checksum,
            control_evidence=manifest_evidence(controls[host]),
            bundle_evidence=evidence_by_host[host],
        )
    locator = ObjectSemanticBaseLocator.create(
        run_id="fresh-semantic", epoch=0, layout=_layout(),
        control_payload_checksum=controls[0].payload_checksum,
        control_manifests_by_host=controls,
    )
    write_object_semantic_base_locator(root / "base" / "epoch-00000000.json", locator)
    return ids


def _plan(arm):
    return build_active_gradient_plan(
        tuple(
            GradientShard(rank, "weight", "detector", (1, 2), (0, 0), (1, 2), "torch.float32")
            for rank in (0, 1)
        ),
        _layout(), arm=arm, detector_catalog_hash="a" * 64,
        semantic_plan_hash="b" * 64,
        fixed_pairs=((0, 1),) if arm == "adacheck_fixed" else (),
    )


def _context(root, clients, plan):
    semantic_plan = SimpleNamespace(
        plan_hash="b" * 64,
        actions=(object(), object()),
        actions_for_rank=lambda rank: (object(),),
    )
    return SimpleNamespace(
        arm=plan.arm, binding=RecoveryBinding(0, 2, 0, 0, 0),
        rank_bindings=_layout(), parameter_ids=(("weight",),),
        semantic_binding=SimpleNamespace(to_dict=lambda: {"catalog_hash": "a" * 64}),
        semantic_plan=semantic_plan, detector_evidence={"frozen": "yes"}, plan=plan,
        clients=clients, root=root, run_id="fresh-semantic", mapping_hash="c" * 64,
    )


def test_factory_rejects_the_semantic_source_switch(monkeypatch):
    monkeypatch.setenv("CKPT_OBJECT_SEMANTIC_ENABLE", "1")
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_object_semantic_restore_controller(
            object(), object(), object(), sampler=None, args=object()
        )


@pytest.mark.parametrize("arm", ("adacheck_fixed", "host_global_fixed", "m1"))
@pytest.mark.parametrize("surviving_host", (0, 1))
def test_fresh_controller_restores_three_arms_after_single_host_loss(
    monkeypatch, tmp_path, arm, surviving_host
):
    plan = _plan(arm)
    direct = torch.nn.Linear(2, 1, bias=False)
    direct_optimizer = _optimizer(direct)
    direct_scheduler = torch.optim.lr_scheduler.LambdaLR(direct_optimizer, lambda _: 1.0)
    peer = torch.nn.Linear(2, 1, bias=False)
    peer_optimizer = _optimizer(peer)
    peer_scheduler = torch.optim.lr_scheduler.LambdaLR(peer_optimizer, lambda _: 1.0)
    with _services() as services:
        clients = {host: item[0] for host, item in enumerate(services)}
        ids = _publish_base(clients, tmp_path, {
            0: _base(direct, direct_optimizer, direct_scheduler, 0),
            1: _base(peer, peer_optimizer, peer_scheduler, 1),
        })
        for epoch in (1, 2):
            gradient = torch.tensor([[0.25 * epoch, -0.5]])
            records, updates = {}, {}
            for rank, (module, optimizer, base_id) in enumerate(((direct, direct_optimizer, ids[0]), (peer, peer_optimizer, ids[1]))):
                module.weight.grad = gradient.clone()
                records[rank] = capture_optimizer_update(
                    optimizer, parameter_ids=(("weight",),),
                    binding=RecoveryBinding(rank, 2, rank, 0, 0), base_id=base_id,
                    base_epoch=0, epoch=epoch, runtime_state=_runtime(epoch),
                )
            item = plan.objects[0]
            encoding = confirm_gradient_equivalence(
                plan, {(rank, item.object_id): gradient for rank in (0, 1)}, epoch=epoch
            )
            for rank in (0, 1):
                updates[rank] = externalize_local_optimizer_update(plan, encoding, rank, records[rank])
            locator = publish_object_update_epoch(
                clients, run_id="fresh-semantic", plan=plan, encoding=encoding,
                local_updates=updates,
            ).locator
            write_object_update_locator(tmp_path / "updates" / f"epoch-{epoch:08d}.json", locator)
            direct_optimizer.step()
            direct_optimizer.zero_grad(set_to_none=True)
            direct_scheduler.step()
        services[1 - surviving_host][1].shutdown()
        fresh = torch.nn.Linear(2, 1, bias=False)
        fresh_optimizer = _optimizer(fresh)
        fresh_scheduler = torch.optim.lr_scheduler.LambdaLR(fresh_optimizer, lambda _: 1.0)
        sampler = SimpleNamespace(consumed_samples=0, epoch=0)
        monkeypatch.setattr(
            "Merak.core.checkpoint.object_semantic_restore.restore_runtime_state",
            lambda state, sampler: (
                setattr(sampler, "consumed_samples", state["sampler"]["consumed_samples"])
                or setattr(sampler, "epoch", state["sampler"]["epoch"])
                or state["iteration"]
            ),
        )
        controller = SemanticFixedObjectRestoreController(
            fresh, fresh_optimizer, fresh_scheduler, sampler=sampler,
            context=_context(tmp_path, clients, plan), base_epoch=0,
            surviving_host=surviving_host,
        )
        assert controller.restore(2) == 2
    assert torch.equal(fresh.weight, direct.weight)
    _equal(fresh_optimizer.state_dict(), direct_optimizer.state_dict())
    assert fresh_scheduler.state_dict() == direct_scheduler.state_dict()
    assert (sampler.consumed_samples, sampler.epoch) == (4, 2)
    metrics = json.loads((tmp_path / "recovery_metrics" / "rank_00000.json").read_text())
    assert metrics["rank"] == 0
    assert metrics["requested_epoch"] == metrics["final_epoch"] == 2
    assert metrics["base_epoch"] == 0
    assert metrics["arm"] == arm
    assert metrics["surviving_host"] == surviving_host
    assert metrics["suffix_get_metrics"] == "unavailable"
    assert metrics["total_restore_wall_ms"] >= 0
    assert metrics["checksum_diagnostic_only"] is True
    assert set(metrics["component_checksums"]) == {"model", "optimizer", "lr_scheduler"}
    assert len(metrics["boundary_checksum"]) == 64


@pytest.mark.parametrize("fault", ("tamper", "gap"))
def test_fresh_controller_fails_closed_before_live_mutation(monkeypatch, tmp_path, fault):
    root = tmp_path
    plan = _plan("m1")
    context = _context(root, {0: object(), 1: object()}, plan)
    controller = SemanticFixedObjectRestoreController(
        torch.nn.Linear(2, 1, bias=False), _optimizer(torch.nn.Linear(2, 1, bias=False)),
        object(), sampler=None, context=context, base_epoch=0, surviving_host=1,
    )
    (root / "base").mkdir()
    (root / "updates").mkdir()
    if fault == "tamper":
        (root / "base" / "epoch-00000000.json").write_text("{")
        with pytest.raises(ValueError):
            controller.restore(1)
    else:
        source = torch.nn.Linear(2, 1, bias=False)
        source_optimizer = _optimizer(source)
        source_scheduler = torch.optim.lr_scheduler.LambdaLR(source_optimizer, lambda _: 1.0)
        base = _base(source, source_optimizer, source_scheduler, 0)
        monkeypatch.setattr(
            "Merak.core.checkpoint.object_semantic_restore.load_object_semantic_base_locator",
            lambda _path: SimpleNamespace(run_id="fresh-semantic", epoch=0),
        )
        monkeypatch.setattr(
            "Merak.core.checkpoint.object_semantic_restore.restore_object_semantic_base",
            lambda *args, **kwargs: SimpleNamespace(
                state=base, base_id=f"sha256:{checkpoint_state_checksum(base)}"
            ),
        )
        with pytest.raises(ValueError, match="contiguous"):
            controller.restore(2)
