from dataclasses import replace

import pytest
import torch

from Merak.core.checkpoint.object_recovery import (
    ObjectAddress,
    ObjectCatalog,
    ObjectFragment,
    ObjectRuntimeIdentity,
    OpaqueRecord,
    OpaqueRequest,
    SliceRequest,
    TargetLayout,
    assert_fair_object_ab,
    build_object_recovery_plan,
    restore_object_layout,
)


def _runtime(layout, *, epoch=7, failures=((), (0,), (1,))):
    return ObjectRuntimeIdentity(
        detector_catalog_hash="d" * 64,
        semantic_plan_hash="s" * 64,
        encoder_name="semantic_fixed_gradient_bf16",
        source_layout_hash="source-layout-v1",
        target_layout_hash=layout.layout_hash,
        epoch=epoch,
        failure_sets=failures,
    )


def _fragment(address, value, *, offset, source_rank, host, epoch=7):
    return ObjectFragment.from_tensor(
        address,
        epoch=epoch,
        offsets=offset,
        tensor=torch.tensor(value, dtype=getattr(torch, address.dtype.split(".")[-1])),
        source_rank=source_rank,
        storage_host=host,
    )


def test_same_layout_common_runtime_restores_full_training_state_after_host_failure():
    model = ObjectAddress("model.weight", "model", (4,), "torch.float32")
    optimizer = ObjectAddress("model.weight.exp_avg", "optimizer", (4,), "torch.float32")
    layout = TargetLayout(
        slices=(
            SliceRequest(0, model, (0,), (4,)),
            SliceRequest(0, optimizer, (0,), (4,)),
        ),
        opaque=(
            OpaqueRequest(0, 0, "scheduler"),
            OpaqueRequest(0, 0, "rng"),
            OpaqueRequest(0, 0, "sampler"),
            OpaqueRequest(0, 0, "progress"),
        ),
    )
    catalog = ObjectCatalog(
        7,
        (
            _fragment(model, [1, 2, 3, 4], offset=(0,), source_rank=0, host=0),
            _fragment(model, [1, 2, 3, 4], offset=(0,), source_rank=8, host=1),
            _fragment(optimizer, [10, 20, 30, 40], offset=(0,), source_rank=0, host=0),
            _fragment(optimizer, [10, 20, 30, 40], offset=(0,), source_rank=8, host=1),
        ),
        (
            OpaqueRecord.from_value(owner_rank=0, key="scheduler", epoch=7, value={"step": 7}, storage_hosts=(0, 1)),
            OpaqueRecord.from_value(owner_rank=0, key="rng", epoch=7, value=torch.arange(4, dtype=torch.uint8), storage_hosts=(0, 1)),
            OpaqueRecord.from_value(owner_rank=0, key="sampler", epoch=7, value={"cursor": 128}, storage_hosts=(0, 1)),
            OpaqueRecord.from_value(owner_rank=0, key="progress", epoch=7, value={"iteration": 7}, storage_hosts=(0, 1)),
        ),
    )
    runtime = _runtime(layout)
    fixed = build_object_recovery_plan(catalog, layout, arm="adacheck_object_fixed", runtime=runtime)
    m1 = build_object_recovery_plan(catalog, layout, arm="m1_object", runtime=runtime)
    assert_fair_object_ab(fixed, m1)
    restored = restore_object_layout(catalog, m1, layout, failed_hosts=(0,))[0]
    assert torch.equal(restored.tensors["model.weight"], torch.tensor([1, 2, 3, 4.0]))
    assert torch.equal(restored.tensors["model.weight.exp_avg"], torch.tensor([10, 20, 30, 40.0]))
    assert restored.opaque["scheduler"] == {"step": 7}
    assert torch.equal(restored.opaque["rng"], torch.arange(4, dtype=torch.uint8))
    assert restored.opaque["sampler"] == {"cursor": 128}
    assert restored.opaque["progress"] == {"iteration": 7}


def test_multisource_pp_tp_style_reshard_to_changed_layout():
    weight = ObjectAddress("model.layers.0.weight", "model", (4, 4), "torch.float32")
    moment = ObjectAddress("model.layers.0.weight.exp_avg", "optimizer", (4, 4), "torch.float32")
    full_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    full_moment = full_weight + 100
    fragments = []
    for address, value in ((weight, full_weight), (moment, full_moment)):
        for row in range(2):
            for column in range(2):
                part = value[row * 2 : row * 2 + 2, column * 2 : column * 2 + 2]
                fragments.extend(
                    (
                        ObjectFragment.from_tensor(address, epoch=7, offsets=(row * 2, column * 2), tensor=part, source_rank=row * 2 + column, storage_host=0),
                        ObjectFragment.from_tensor(address, epoch=7, offsets=(row * 2, column * 2), tensor=part, source_rank=8 + row * 2 + column, storage_host=1),
                    )
                )
    layout = TargetLayout(
        slices=(
            SliceRequest(0, weight, (0, 0), (4, 2)),
            SliceRequest(1, weight, (0, 2), (4, 2)),
            SliceRequest(0, moment, (0, 0), (4, 2)),
            SliceRequest(1, moment, (0, 2), (4, 2)),
        )
    )
    catalog = ObjectCatalog(7, tuple(fragments))
    plan = build_object_recovery_plan(catalog, layout, arm="m1_object", runtime=_runtime(layout))
    restored = restore_object_layout(catalog, plan, layout, failed_hosts=(1,))
    assert torch.equal(restored[0].tensors[weight.semantic_id], full_weight[:, :2])
    assert torch.equal(restored[1].tensors[weight.semantic_id], full_weight[:, 2:])
    assert torch.equal(restored[0].tensors[moment.semantic_id], full_moment[:, :2])
    assert torch.equal(restored[1].tensors[moment.semantic_id], full_moment[:, 2:])


@pytest.mark.parametrize(
    "fragments, pattern",
    [
        ([(0, [1, 2])], "coverage gap"),
        ([(0, [1, 2, 3]), (2, [3, 4])], "logical source fragments overlap"),
        ([(0, [1, 2, 3, 4])], "stale fragment"),
    ],
)
def test_object_recovery_fails_closed_for_gap_overlap_and_stale_epoch(fragments, pattern):
    address = ObjectAddress("model.weight", "model", (4,), "torch.float32")
    layout = TargetLayout(slices=(SliceRequest(0, address, (0,), (4,)),))
    values = []
    for index, (offset, tensor) in enumerate(fragments):
        values.append(_fragment(address, tensor, offset=(offset,), source_rank=index, host=0, epoch=6 if pattern == "stale fragment" else 7))
    catalog = ObjectCatalog(7, tuple(values))
    with pytest.raises(ValueError, match=pattern):
        build_object_recovery_plan(catalog, layout, arm="m1_object", runtime=_runtime(layout))


def test_missing_source_error_reports_target_and_same_semantic_address_mismatch():
    requested = ObjectAddress("expert.0.w1", "model", (4,), "torch.float32")
    captured = ObjectAddress("expert.0.w1", "model", (8,), "torch.float32")
    layout = TargetLayout(slices=(SliceRequest(3, requested, (0,), (4,)),))
    catalog = ObjectCatalog(
        7,
        (_fragment(captured, [1, 2, 3, 4], offset=(0,), source_rank=2, host=1),),
    )

    with pytest.raises(ValueError, match="coverage gap") as error:
        build_object_recovery_plan(catalog, layout, arm="m1_object", runtime=_runtime(layout))

    message = str(error.value)
    assert "semantic_id='expert.0.w1'" in message
    assert "global_shape=(8,)" in message
    assert "fragments=[((0,), (4,), 1, 2)]" in message


def test_object_recovery_rejects_conflict_and_opaque_remap():
    address = ObjectAddress("model.weight", "model", (2,), "torch.float32")
    layout = TargetLayout(
        slices=(SliceRequest(0, address, (0,), (2,)),),
        opaque=(OpaqueRequest(1, 0, "rng"),),
    )
    first = _fragment(address, [1, 2], offset=(0,), source_rank=0, host=0)
    replica = _fragment(address, [1, 2], offset=(0,), source_rank=8, host=1)
    catalog = ObjectCatalog(
        7,
        (first, replica),
        (OpaqueRecord.from_value(owner_rank=0, key="rng", epoch=7, value={"x": 1}, storage_hosts=(0, 1)),),
    )
    with pytest.raises(ValueError, match="opaque state cannot be remapped"):
        build_object_recovery_plan(catalog, layout, arm="m1_object", runtime=_runtime(layout))

    no_opaque_layout = TargetLayout(slices=(SliceRequest(0, address, (0,), (2,)),))
    conflicting = _fragment(address, [2, 3], offset=(0,), source_rank=8, host=1)
    conflicting_catalog = ObjectCatalog(7, (first, conflicting))
    with pytest.raises(ValueError, match="replicas of one logical fragment conflict"):
        build_object_recovery_plan(conflicting_catalog, no_opaque_layout, arm="m1_object", runtime=_runtime(no_opaque_layout))


def test_catalog_content_identity_is_independent_of_physical_availability():
    address = ObjectAddress("model.weight", "model", (2,), "torch.float32")
    first = ObjectCatalog(7, (_fragment(address, [1, 2], offset=(0,), source_rank=0, host=0),))
    moved = ObjectCatalog(7, (_fragment(address, [1, 2], offset=(0,), source_rank=8, host=1),))
    assert first.content_hash == moved.content_hash
    assert first.availability_hash != moved.availability_hash


def test_planner_accepts_metadata_but_restore_requires_selected_payload():
    address = ObjectAddress("model.weight", "model", (2,), "torch.float32")
    layout = TargetLayout(slices=(SliceRequest(0, address, (0,), (2,)),))
    fragment = replace(
        _fragment(address, [1, 2], offset=(0,), source_rank=0, host=0),
        tensor=None,
    )
    catalog = ObjectCatalog(7, (fragment,))
    plan = build_object_recovery_plan(
        catalog, layout, arm="m1_object", runtime=_runtime(layout, failures=((),))
    )
    with pytest.raises(ValueError, match="payload is unavailable"):
        restore_object_layout(catalog, plan, layout)


def test_sealed_plan_metrics_and_selection_cannot_be_recomputed_or_tampered():
    left = ObjectAddress("model.left", "model", (2,), "torch.float32")
    right = ObjectAddress("model.right", "optimizer", (2,), "torch.float32")
    layout = TargetLayout(
        slices=(SliceRequest(0, left, (0,), (2,)), SliceRequest(0, right, (0,), (2,)))
    )
    fragments = (
        _fragment(left, [1, 2], offset=(0,), source_rank=0, host=0),
        _fragment(left, [1, 2], offset=(0,), source_rank=1, host=1),
        _fragment(right, [3, 4], offset=(0,), source_rank=0, host=0),
        _fragment(right, [3, 4], offset=(0,), source_rank=1, host=1),
    )
    catalog = ObjectCatalog(7, fragments)
    runtime = _runtime(layout)
    fixed = build_object_recovery_plan(catalog, layout, arm="adacheck_object_fixed", runtime=runtime)
    m1 = build_object_recovery_plan(catalog, layout, arm="m1_object", runtime=runtime)
    assert fixed.total_selected_bytes == m1.total_selected_bytes == 16
    assert fixed.maximum_publisher_bytes == 16
    assert m1.maximum_publisher_bytes == 8
    assert fixed.plan_hash != m1.plan_hash
    assert len(fixed.plan_hash) == 64
    normal = m1.failure_plan(())
    with pytest.raises(ValueError, match="sealed object plan hash mismatch"):
        replace(m1, failure_plans=(replace(normal, selections=()), *m1.failure_plans[1:]))


def test_opaque_epoch_and_arm_validation_fail_closed():
    address = ObjectAddress("model.weight", "model", (2,), "torch.float32")
    layout = TargetLayout(slices=(SliceRequest(0, address, (0,), (2,)),))
    catalog = ObjectCatalog(
        7,
        (_fragment(address, [1, 2], offset=(0,), source_rank=0, host=0),),
        (OpaqueRecord.from_value(owner_rank=0, key="rng", epoch=6, value={"x": 1}, storage_hosts=(0,)),),
    )
    with pytest.raises(ValueError, match="stale opaque"):
        build_object_recovery_plan(catalog, layout, arm="m1_object", runtime=_runtime(layout, failures=((),)))
    valid = ObjectCatalog(7, (_fragment(address, [1, 2], offset=(0,), source_rank=0, host=0),))
    with pytest.raises(ValueError, match="unsupported object recovery arm"):
        build_object_recovery_plan(valid, layout, arm="not-an-arm", runtime=_runtime(layout, failures=((),)))


def test_unmaterialized_dependency_chain_fails_closed():
    address = ObjectAddress("model.weight", "model", (2,), "torch.float32")
    layout = TargetLayout(slices=(SliceRequest(0, address, (0,), (2,)),))
    fragment = ObjectFragment.from_tensor(
        address,
        epoch=7,
        offsets=(0,),
        tensor=torch.tensor([1.0, 2.0]),
        source_rank=0,
        storage_host=0,
        base_epoch=6,
        dependency_epoch=6,
    )
    with pytest.raises(ValueError, match="materialized final snapshot"):
        build_object_recovery_plan(
            ObjectCatalog(7, (fragment,)),
            layout,
            arm="m1_object",
            runtime=_runtime(layout, failures=((),)),
        )


def test_build_plan_never_allocates_target_tensor_coverage(monkeypatch):
    address = ObjectAddress("model.weight", "model", (2,), "torch.float32")
    layout = TargetLayout(slices=(SliceRequest(0, address, (0,), (2,)),))
    catalog = ObjectCatalog(
        7,
        (
            _fragment(address, [1, 2], offset=(0,), source_rank=0, host=0),
            _fragment(address, [1, 2], offset=(0,), source_rank=1, host=1),
        ),
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("planner allocated a target tensor")

    monkeypatch.setattr(torch, "empty", fail)
    monkeypatch.setattr(torch, "zeros", fail)
    plan = build_object_recovery_plan(catalog, layout, arm="m1_object", runtime=_runtime(layout))
    assert plan.total_selected_bytes == 8


def test_shuffled_catalog_has_deterministic_plan_and_restore_never_allocates_zeros(monkeypatch):
    left = ObjectAddress("model.left", "model", (2,), "torch.float32")
    right = ObjectAddress("model.right", "optimizer", (2,), "torch.float32")
    layout = TargetLayout(
        slices=(SliceRequest(0, left, (0,), (2,)), SliceRequest(0, right, (0,), (2,)))
    )
    fragments = (
        _fragment(left, [1, 2], offset=(0,), source_rank=0, host=0),
        _fragment(left, [1, 2], offset=(0,), source_rank=1, host=1),
        _fragment(right, [3, 4], offset=(0,), source_rank=0, host=0),
        _fragment(right, [3, 4], offset=(0,), source_rank=1, host=1),
    )
    runtime = _runtime(layout)
    first = build_object_recovery_plan(ObjectCatalog(7, fragments), layout, arm="m1_object", runtime=runtime)
    second_catalog = ObjectCatalog(7, tuple(reversed(fragments)))
    second = build_object_recovery_plan(second_catalog, layout, arm="m1_object", runtime=runtime)
    assert first.plan_hash == second.plan_hash

    def fail(*_args, **_kwargs):
        raise AssertionError("restore allocated a coverage mask")

    monkeypatch.setattr(torch, "zeros", fail)
    restored = restore_object_layout(second_catalog, second, layout)
    assert torch.equal(restored[0].tensors[left.semantic_id], torch.tensor([1.0, 2.0]))
