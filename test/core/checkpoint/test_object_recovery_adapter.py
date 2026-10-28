from types import SimpleNamespace

import pytest
import torch

from Merak.core.checkpoint.object_recovery import (
    ObjectRuntimeIdentity,
    OpaqueRequest,
    SliceRequest,
    TargetLayout,
    build_object_recovery_plan,
    restore_object_layout,
)
from Merak.core.checkpoint.object_recovery_adapter import (
    apply_live_object_tensors,
    build_live_object_catalog,
    capture_live_object_fragments,
    capture_opaque_records,
)


def _module(row_start, *, partitioned):
    module = torch.nn.Module()
    rows = 4 if not partitioned else 2
    module.weight = torch.nn.Parameter(
        torch.arange(row_start, row_start + rows * 3, dtype=torch.bfloat16).reshape(rows, 3)
    )
    if partitioned:
        module.weight.model_parallel = True
        module.weight.partition_dim = 0
        module.weight.partition_stride = 1
    module.bias = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
    return module


def _wrapped_optimizer(module):
    masters = [
        torch.nn.Parameter(module.weight.detach().float().clone()),
        torch.nn.Parameter(module.bias.detach().float().clone()),
    ]
    inner = torch.optim.AdamW(masters, lr=1e-3, foreach=False, fused=False)
    for master in masters:
        master.grad = torch.ones_like(master)
    inner.step()
    return SimpleNamespace(fp16_groups=[[module.weight, module.bias]], fp32_groups=[masters], optimizer=inner)


def _fresh_wrapped_optimizer(module):
    masters = [
        torch.nn.Parameter(module.weight.detach().float().clone()),
        torch.nn.Parameter(module.bias.detach().float().clone()),
    ]
    inner = torch.optim.AdamW(masters, lr=1e-3, foreach=False, fused=False)
    return SimpleNamespace(fp16_groups=[[module.weight, module.bias]], fp32_groups=[masters], optimizer=inner)


def _capture(module, optimizer, *, rank, host, tp_rank):
    return capture_live_object_fragments(
        module, optimizer, rank=rank, storage_host=host, epoch=7,
        tensor_parallel_rank=tp_rank, tensor_parallel_world_size=2,
        base_epoch=7,
    )


def _runtime(layout):
    return ObjectRuntimeIdentity(
        detector_catalog_hash="d" * 64,
        semantic_plan_hash="s" * 64,
        encoder_name="semantic_fixed_gradient_bf16",
        source_layout_hash="tp2-source",
        target_layout_hash=layout.layout_hash,
        epoch=7,
        failure_sets=((), (0,)),
    )


def test_adapter_round_trips_core_catalog_tp_reshard_host_loss_and_live_apply():
    left, right = _module(0, partitioned=True), _module(6, partitioned=True)
    left_optimizer, right_optimizer = _wrapped_optimizer(left), _wrapped_optimizer(right)
    fragments = (
        *_capture(left, left_optimizer, rank=0, host=0, tp_rank=0),
        *_capture(right, right_optimizer, rank=1, host=0, tp_rank=1),
        *_capture(left, left_optimizer, rank=8, host=1, tp_rank=0),
        *_capture(right, right_optimizer, rank=9, host=1, tp_rank=1),
    )
    opaque = capture_opaque_records(
        7,
        {"scheduler": {"step": 7}, "rng": {"seed": 3}, "sampler": {"cursor": 16}, "progress": {"iteration": 7}},
        owner_rank=0,
        storage_hosts=(0, 1),
    )
    catalog = build_live_object_catalog(7, fragments, opaque)
    primary = next(
        item for item in fragments
        if item.address.semantic_id == "weight::optimizer.master" and item.source_rank == 0
    )
    replica = next(
        item for item in fragments
        if item.address.semantic_id == "weight::optimizer.master" and item.source_rank == 8
    )
    assert (primary.shard_id, primary.version_id) == (replica.shard_id, replica.version_id)
    assert (primary.source_rank, primary.storage_host) == (0, 0)
    assert (replica.source_rank, replica.storage_host) == (8, 1)

    target = _module(0, partitioned=False)
    target_optimizer = _wrapped_optimizer(target)
    target_fragments = capture_live_object_fragments(
        target, target_optimizer, rank=0, storage_host=1, epoch=7,
        tensor_parallel_rank=0, tensor_parallel_world_size=1,
        base_epoch=7,
    )
    layout = TargetLayout(
        slices=tuple(
            SliceRequest(0, item.address, item.offsets, item.lengths)
            for item in target_fragments
        ),
        opaque=tuple(OpaqueRequest(0, 0, key) for key in ("scheduler", "rng", "sampler", "progress")),
    )
    plan = build_object_recovery_plan(catalog, layout, arm="m1_object", runtime=_runtime(layout))
    restored = restore_object_layout(catalog, plan, layout, failed_hosts=(0,))[0]
    apply_live_object_tensors(
        target, target_optimizer, restored.tensors,
        tensor_parallel_rank=0, tensor_parallel_world_size=1,
    )
    assert torch.equal(target.weight, torch.cat((left.weight, right.weight)))
    assert torch.equal(target_optimizer.fp32_groups[0][0], torch.cat((left_optimizer.fp32_groups[0][0], right_optimizer.fp32_groups[0][0])))
    target_state = target_optimizer.optimizer.state[target_optimizer.fp32_groups[0][0]]
    left_state = left_optimizer.optimizer.state[left_optimizer.fp32_groups[0][0]]
    right_state = right_optimizer.optimizer.state[right_optimizer.fp32_groups[0][0]]
    assert torch.equal(target_state["exp_avg"], torch.cat((left_state["exp_avg"], right_state["exp_avg"])))
    assert torch.equal(target_state["exp_avg_sq"], torch.cat((left_state["exp_avg_sq"], right_state["exp_avg_sq"])))
    assert target_state["step"].shape == ()
    assert target_state["step"].item() == left_state["step"].item()
    assert restored.opaque["progress"] == {"iteration": 7}


def test_adapter_rejects_missing_binding_stride_slot_shape_and_partial_apply():
    module = _module(0, partitioned=True)
    optimizer = _wrapped_optimizer(module)
    optimizer.fp16_groups[0] = [module.weight]
    optimizer.fp32_groups[0] = [optimizer.fp32_groups[0][0]]
    optimizer.optimizer.param_groups[0]["params"] = optimizer.fp32_groups[0]
    with pytest.raises(ValueError, match="lack optimizer bindings"):
        _capture(module, optimizer, rank=0, host=0, tp_rank=0)

    optimizer = _wrapped_optimizer(module)
    module.weight.partition_stride = 2
    with pytest.raises(ValueError, match="contiguous TP"):
        _capture(module, optimizer, rank=0, host=0, tp_rank=0)
    module.weight.partition_stride = 1
    optimizer.optimizer.state[optimizer.fp32_groups[0][0]]["exp_avg"] = torch.zeros(1)
    with pytest.raises(ValueError, match="shape conflicts"):
        _capture(module, optimizer, rank=0, host=0, tp_rank=0)

    optimizer = _wrapped_optimizer(module)
    with pytest.raises(ValueError, match="do not match"):
        apply_live_object_tensors(
            module, optimizer, {}, tensor_parallel_rank=0, tensor_parallel_world_size=2
        )


def test_opaque_records_need_exact_roots_owner_and_storage_hosts():
    records = capture_opaque_records(
        7,
        {"scheduler": {}, "rng": {}, "sampler": {}, "progress": {}},
        owner_rank=4,
        storage_hosts=(0, 1),
    )
    assert {(record.owner_rank, record.key, record.epoch) for record in records} == {
        (4, "scheduler", 7), (4, "rng", 7), (4, "sampler", 7), (4, "progress", 7)
    }
    with pytest.raises(ValueError, match="requires scheduler"):
        capture_opaque_records(7, {"scheduler": {}, "rng": {}, "sampler": {}}, owner_rank=4, storage_hosts=(0,))


def test_apply_bootstraps_exact_adam_slots_in_a_fresh_optimizer():
    source = _module(0, partitioned=False)
    source_optimizer = _wrapped_optimizer(source)
    tensors = {
        item.address.semantic_id: item.tensor.clone()
        for item in capture_live_object_fragments(
            source,
            source_optimizer,
            rank=0,
            storage_host=0,
            epoch=7,
            tensor_parallel_rank=0,
            tensor_parallel_world_size=1,
        )
    }
    target = _module(100, partitioned=False)
    target_optimizer = _fresh_wrapped_optimizer(target)

    apply_live_object_tensors(
        target,
        target_optimizer,
        tensors,
        tensor_parallel_rank=0,
        tensor_parallel_world_size=1,
    )

    assert torch.equal(target.weight, source.weight)
    for source_master, target_master in zip(
        source_optimizer.fp32_groups[0], target_optimizer.fp32_groups[0]
    ):
        assert torch.equal(target_master, source_master)
        assert set(target_optimizer.optimizer.state[target_master]) == {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }
        for slot in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(
                target_optimizer.optimizer.state[target_master][slot],
                source_optimizer.optimizer.state[source_master][slot],
            )


def test_fresh_adam_bootstrap_rejects_partial_slots_before_mutation():
    source = _module(0, partitioned=False)
    source_optimizer = _wrapped_optimizer(source)
    tensors = {
        item.address.semantic_id: item.tensor.clone()
        for item in capture_live_object_fragments(
            source,
            source_optimizer,
            rank=0,
            storage_host=0,
            epoch=7,
            tensor_parallel_rank=0,
            tensor_parallel_world_size=1,
        )
    }
    tensors.pop("weight::optimizer.exp_avg_sq")
    target = _module(100, partitioned=False)
    target_optimizer = _fresh_wrapped_optimizer(target)
    original = target.weight.detach().clone()

    with pytest.raises(ValueError, match="requires step"):
        apply_live_object_tensors(
            target,
            target_optimizer,
            tensors,
            tensor_parallel_rank=0,
            tensor_parallel_world_size=1,
        )

    assert torch.equal(target.weight, original)
    assert not target_optimizer.optimizer.state


def test_fresh_adam_bootstrap_rejects_amsgrad_before_mutation():
    source = _module(0, partitioned=False)
    source_optimizer = _wrapped_optimizer(source)
    tensors = {
        item.address.semantic_id: item.tensor.clone()
        for item in capture_live_object_fragments(
            source,
            source_optimizer,
            rank=0,
            storage_host=0,
            epoch=7,
            tensor_parallel_rank=0,
            tensor_parallel_world_size=1,
        )
    }
    target = _module(100, partitioned=False)
    target_optimizer = _fresh_wrapped_optimizer(target)
    target_optimizer.optimizer.param_groups[0]["amsgrad"] = True
    original = target.weight.detach().clone()

    with pytest.raises(ValueError, match="does not support amsgrad"):
        apply_live_object_tensors(
            target,
            target_optimizer,
            tensors,
            tensor_parallel_rank=0,
            tensor_parallel_world_size=1,
        )

    assert torch.equal(target.weight, original)
    assert not target_optimizer.optimizer.state
