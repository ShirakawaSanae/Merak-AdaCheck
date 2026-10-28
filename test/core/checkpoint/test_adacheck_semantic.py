import copy
import hashlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from Merak.core.checkpoint.adacheck_semantic import AdaCheckSemanticBinding
from Merak.core.checkpoint.adacheck_semantic import (
    SemanticAction,
    SemanticEncoderPlan,
    build_gradient_semantic_plan,
    encode_mixed_action_record,
    restore_mixed_action_record,
    semantic_binding_from_detector_config,
)
from Merak.core.checkpoint.global_commit import create_global_epoch, validate_global_epoch
from Merak.core.checkpoint.incremental_checkpoint import (
    RecoveryBinding,
    apply_optimizer_update,
    capture_optimizer_update,
)


def _digest(label):
    return hashlib.sha256(label.encode()).hexdigest()


def test_stable_detector_config_binds_live_topology_and_gradient_plan(tmp_path):
    summary = {
        "catalog_hash": _digest("catalog"),
        "detector_stability_steps": 2,
        "world_size": 2,
        "ranks_per_host": 1,
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True))
    config = {
        "run_kind": "matched_detector_capture",
        "status_at_creation": "capture_and_freeze_complete",
        "model_argument": "gpt_6b",
        "detector_comparison_scope": "world",
        "precision": "bf16",
        "parallelism": {
            "world_size": 2,
            "hosts": 2,
            "ranks_per_host": 1,
            "pipeline": 1,
            "tensor": 1,
            "data": 2,
        },
        "workload": {
            "rank_topology": [
                {"rank": 0, "data": 0, "pipeline": 0, "tensor": 0},
                {"rank": 1, "data": 1, "pipeline": 0, "tensor": 0},
            ]
        },
        "frozen_catalog": {
            "summary_path": str(summary_path),
            "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "catalog_hash": summary["catalog_hash"],
            "detector_stability_steps": 2,
        },
    }
    config_path = tmp_path / "experiment-config.json"
    config_path.write_text(json.dumps(config))
    rank_bindings = [
        {
            "rank": rank,
            "world_size": 2,
            "data_parallel_rank": rank,
            "pipeline_parallel_rank": 0,
            "tensor_parallel_rank": 0,
        }
        for rank in range(2)
    ]
    binding, evidence = semantic_binding_from_detector_config(
        config_path,
        expected_model="gpt_6b",
        world_size=2,
        ranks_per_host=1,
        rank_bindings=rank_bindings,
        pairs=((0, 1),),
    )
    plan = build_gradient_semantic_plan(
        binding,
        world_size=2,
        pairs=((0, 1),),
        parameter_ids_by_rank=((('left',),), (('right',),)),
    )

    assert binding.policy == "host-f1"
    assert evidence["detector_stability_steps"] == 2
    assert [item.action for item in plan.actions] == [
        "gradient_bf16",
        "gradient_bf16",
    ]

    config["workload"]["rank_topology"][1]["data"] = 0
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="rank topology"):
        semantic_binding_from_detector_config(
            config_path,
            expected_model="gpt_6b",
            world_size=2,
            ranks_per_host=1,
            rank_bindings=rank_bindings,
            pairs=((0, 1),),
        )


def _rank_evidence(rank, epoch, semantic):
    peer = 1 - rank
    group = [0, 1]
    return {
        "rank": rank,
        "epoch": epoch,
        "mode": "peer_delta",
        "adacheck_semantic": semantic.to_dict(),
        "paths": {
            "local_full": {
                "manifest_id": f"full-{rank}-{epoch}",
                "payload_checksum": f"full-payload-{rank}-{epoch}",
                "epoch": epoch,
                "source_rank": rank,
                "target_rank": rank,
                "group_ranks": group,
            },
            "remote_delta": {
                "manifest_id": f"delta-{rank}-{epoch}",
                "payload_checksum": f"delta-payload-{rank}-{epoch}",
                "epoch": epoch,
                "source_rank": rank,
                "target_rank": peer,
                "group_ranks": group,
            },
        },
    }


def _runtime(epoch, sampler):
    return {
        "iteration": epoch,
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "sampler": {"consumed_samples": sampler.consumed_samples, "epoch": sampler.epoch},
    }


def _restore_runtime(value, sampler):
    random.setstate(value["random_rng_state"])
    np.random.set_state(value["np_rng_state"])
    torch.set_rng_state(value["torch_rng_state"])
    sampler.consumed_samples = value["sampler"]["consumed_samples"]
    sampler.epoch = value["sampler"]["epoch"]
    return value["iteration"]


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def test_semantic_binding_recovers_complete_cpu_training_state():
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    semantic = AdaCheckSemanticBinding(
        "host-f1", _digest("catalog"), _digest("parallel"), _digest("protection")
    )
    binding = RecoveryBinding(0, 2, 0, 0, 0)
    module = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.01, foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    sampler = SimpleNamespace(consumed_samples=0, epoch=0)
    parameter_ids = (("weight",),)
    base_id = "sha256:" + _digest("base")
    base = {
        "model": copy.deepcopy(module.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "lr_scheduler": copy.deepcopy(scheduler.state_dict()),
        "runtime_state": _runtime(0, sampler),
        "adacheck_semantic": semantic.to_dict(),
    }
    semantic.validate_base(base)

    records = []
    for epoch in range(1, 4):
        sampler.consumed_samples += 8
        sampler.epoch = epoch
        module.weight.grad = torch.tensor([[epoch / 10, -epoch / 20]])
        records.append(
            capture_optimizer_update(
                optimizer,
                parameter_ids=parameter_ids,
                binding=binding,
                base_id=base_id,
                base_epoch=0,
                epoch=epoch,
                metadata={"adacheck_semantic": semantic.to_dict()},
                runtime_state=_runtime(epoch, sampler),
            )
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
    expected_model = copy.deepcopy(module.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_scheduler = copy.deepcopy(scheduler.state_dict())
    expected_runtime = records[-1]["runtime_state"]
    random.setstate(expected_runtime["random_rng_state"])
    expected_random = random.random()
    np.random.set_state(expected_runtime["np_rng_state"])
    expected_numpy = np.random.rand()
    torch.set_rng_state(expected_runtime["torch_rng_state"])
    expected_torch = torch.rand(())

    restored_module = torch.nn.Linear(2, 1, bias=False)
    restored_optimizer = torch.optim.AdamW(
        restored_module.parameters(), lr=0.01, foreach=False, fused=False
    )
    restored_scheduler = torch.optim.lr_scheduler.StepLR(
        restored_optimizer, step_size=2, gamma=0.9
    )
    restored_module.load_state_dict(base["model"])
    restored_optimizer.load_state_dict(base["optimizer"])
    restored_scheduler.load_state_dict(base["lr_scheduler"])
    for record in records:
        semantic.validate_update(record)
    for epoch, record in enumerate(records, start=1):
        apply_optimizer_update(
            restored_optimizer,
            record,
            expected_parameter_ids=parameter_ids,
            expected_binding=binding,
            expected_base_id=base_id,
            expected_base_epoch=0,
            expected_epoch=epoch,
        )
        restored_scheduler.step()
    restored_sampler = SimpleNamespace(consumed_samples=-1, epoch=-1)
    assert _restore_runtime(expected_runtime, restored_sampler) == 3

    assert restored_module.state_dict().keys() == expected_model.keys()
    assert torch.equal(restored_module.weight, expected_model["weight"])
    assert _equal(restored_optimizer.state_dict(), expected_optimizer)
    assert restored_scheduler.state_dict() == expected_scheduler
    assert vars(restored_sampler) == expected_runtime["sampler"]
    assert random.random() == expected_random
    assert np.random.rand() == expected_numpy
    assert torch.equal(torch.rand(()), expected_torch)


def test_semantic_binding_rejects_partial_stale_and_mismatched_commit():
    semantic = AdaCheckSemanticBinding(
        "rank-k2", _digest("catalog"), _digest("parallel"), _digest("protection")
    )
    stale = AdaCheckSemanticBinding(
        "rank-k2", _digest("catalog"), _digest("parallel"), _digest("new-protection")
    )
    with pytest.raises(ValueError, match="mismatch"):
        semantic.validate_base({"adacheck_semantic": stale.to_dict()})
    with pytest.raises(ValueError, match="lacks"):
        semantic.validate_update({"metadata": {}})

    ranks = [_rank_evidence(0, 3, semantic), _rank_evidence(1, 3, semantic)]
    control = create_global_epoch(run_id="semantic", epoch=3, world_size=2, ranks=ranks)
    assert control["adacheck_semantic"] == semantic.to_dict()
    assert validate_global_epoch(control, run_id="semantic", epoch=3, world_size=2) == control

    partial = copy.deepcopy(ranks)
    partial[1].pop("adacheck_semantic")
    with pytest.raises(ValueError, match="partial"):
        create_global_epoch(run_id="semantic", epoch=3, world_size=2, ranks=partial)
    mismatched = copy.deepcopy(ranks)
    mismatched[1]["adacheck_semantic"] = stale.to_dict()
    with pytest.raises(ValueError, match="inconsistent"):
        create_global_epoch(run_id="semantic", epoch=3, world_size=2, ranks=mismatched)


def _semantic_plan(*, peer_fallback=False):
    fallback = "direct_poststep" if peer_fallback else None
    return SemanticEncoderPlan(
        policy="host-f1",
        catalog_hash=_digest("catalog"),
        parallel_plan_hash=_digest("parallel"),
        protection_plan_hash=_digest("protection"),
        world_size=2,
        pairs=((0, 1),),
        actions=(
            SemanticAction(0, "gradient", "gradient_bf16", 0),
            SemanticAction(0, "metadata", "metadata_exact", 0),
            SemanticAction(0, "unique", "direct_poststep", 0),
            SemanticAction(1, "gradient", "peer_derived", 0, fallback),
            SemanticAction(1, "metadata", "metadata_exact", 1),
            SemanticAction(1, "unique", "direct_poststep", 1),
        ),
    )


def _values(rank):
    return {
        "gradient": torch.tensor([1.25, -0.5], dtype=torch.float32),
        "metadata": {"rank": rank, "step": torch.tensor(rank + 3)},
        "unique": torch.tensor([rank + 7], dtype=torch.float32),
    }


def test_semantic_encoder_plan_is_stable_and_rejects_canonical_cycles():
    first = _semantic_plan()
    second = _semantic_plan()
    assert first.plan_hash == second.plan_hash
    assert first.to_dict()["plan_hash"] == first.plan_hash

    with pytest.raises(ValueError, match="terminal canonical owner"):
        SemanticEncoderPlan(
            policy="rank-k2",
            catalog_hash=_digest("catalog"),
            parallel_plan_hash=_digest("parallel"),
            protection_plan_hash=_digest("protection"),
            world_size=2,
            pairs=((0, 1),),
            actions=(
                SemanticAction(0, "value", "peer_derived", 1),
                SemanticAction(1, "value", "peer_derived", 0),
            ),
        )


def test_mixed_action_record_restores_direct_gradient_peer_and_metadata():
    plan = _semantic_plan()
    first = encode_mixed_action_record(
        plan,
        rank=0,
        epoch=4,
        values=_values(0),
        metadata={"iteration": 4, "sampler": {"cursor": 16}},
    )
    first_values, first_metadata = restore_mixed_action_record(
        plan, first, rank=0, epoch=4, canonical_values={}
    )
    assert first_values["gradient"].dtype == torch.bfloat16
    assert torch.equal(first_values["unique"], torch.tensor([7.0]))
    assert first_metadata == {"iteration": 4, "sampler": {"cursor": 16}}

    second = encode_mixed_action_record(
        plan,
        rank=1,
        epoch=4,
        values=_values(1),
        metadata={"iteration": 4, "sampler": {"cursor": 24}},
    )
    second_values, second_metadata = restore_mixed_action_record(
        plan,
        second,
        rank=1,
        epoch=4,
        canonical_values={(0, "gradient"): first_values["gradient"]},
    )
    assert torch.equal(second_values["gradient"], first_values["gradient"])
    assert torch.equal(second_values["unique"], torch.tensor([8.0]))
    assert second_metadata["sampler"]["cursor"] == 24


def test_mixed_action_record_fails_closed_for_missing_peer_corruption_and_fallback():
    plan = _semantic_plan(peer_fallback=True)
    record = encode_mixed_action_record(
        plan,
        rank=1,
        epoch=5,
        values=_values(1),
        metadata={"iteration": 5},
        fallback_value_ids=("gradient",),
    )
    restored, _ = restore_mixed_action_record(
        plan, record, rank=1, epoch=5, canonical_values={}
    )
    assert restored["gradient"].dtype == torch.float32

    no_fallback = encode_mixed_action_record(
        _semantic_plan(),
        rank=1,
        epoch=5,
        values=_values(1),
        metadata={"iteration": 5},
    )
    with pytest.raises(ValueError, match="unavailable"):
        restore_mixed_action_record(
            _semantic_plan(), no_fallback, rank=1, epoch=5, canonical_values={}
        )

    corrupted = copy.deepcopy(record)
    corrupted["actions"][0]["payload"] += 1
    with pytest.raises(ValueError, match="corrupt"):
        restore_mixed_action_record(
            plan, corrupted, rank=1, epoch=5, canonical_values={}
        )

    with pytest.raises(ValueError, match="values do not match"):
        encode_mixed_action_record(
            plan,
            rank=0,
            epoch=5,
            values={"gradient": torch.ones(1)},
            metadata={"iteration": 5},
        )
