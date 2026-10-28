import pytest
import torch

from Merak.core.checkpoint.changed_layout_cpu_gate import (
    apply_sealed_adamw_objects,
    capture_materialized_adamw_fragments,
    live_adamw_objects,
)
from Merak.core.checkpoint.object_recovery import (
    ObjectAddress,
    ObjectCatalog,
    ObjectFragment,
    ObjectRuntimeIdentity,
    SliceRequest,
    TargetLayout,
    build_object_recovery_plan,
    restore_object_layout,
)


EPOCH = 11
FQNS = tuple(f"layers.{index}.weight" for index in range(4))
ROLES = ("model", "optimizer.master", "optimizer.exp_avg", "optimizer.exp_avg_sq", "optimizer.step")


def _stage(indices):
    module = torch.nn.Module()
    module.layers = torch.nn.ModuleDict(
        {str(index): torch.nn.Linear(2, 2, bias=False) for index in indices}
    )
    masters = {}
    for fqn, parameter in module.named_parameters():
        index = int(fqn.split(".")[1])
        with torch.no_grad():
            parameter.copy_(torch.arange(4, dtype=torch.float32).reshape(2, 2) + index * 10)
        masters[fqn] = torch.nn.Parameter(parameter.detach().clone() + 100)
    optimizer = torch.optim.AdamW(tuple(masters.values()), lr=0.01, foreach=False, fused=False)
    for index, master in enumerate(masters.values()):
        master.grad = torch.full_like(master, float(index + 1))
    optimizer.step()  # Creates real scalar AdamW ``step`` and both moment slots.
    optimizer.zero_grad(set_to_none=True)
    return module, optimizer, masters


def _source_catalog():
    fragments = []
    oracle = {}
    for pp_rank, fqn in enumerate(FQNS):
        for dp_rank in range(4):
            module, optimizer, masters = _stage((pp_rank,))
            captured = capture_materialized_adamw_fragments(
                module, optimizer, masters, epoch=EPOCH,
                source_rank=pp_rank * 4 + dp_rank, storage_host=dp_rank % 2,
            )
            fragments.extend(captured)
            if dp_rank == 0:
                oracle.update({item.address.semantic_id: item.tensor.detach().clone().reshape(1) if item.scalar else item.tensor.detach().clone() for item in live_adamw_objects(module, optimizer, masters)})
    return ObjectCatalog(EPOCH, tuple(fragments)), oracle


def _target_layout():
    slices = []
    for pp_rank, indices in enumerate(((0, 1), (2, 3))):
        for dp_rank in range(8):
            target_rank = pp_rank * 8 + dp_rank
            module, optimizer, masters = _stage(indices)
            for item in live_adamw_objects(module, optimizer, masters):
                slices.append(SliceRequest(target_rank, item.address, (0,) * len(item.address.global_shape), item.address.global_shape))
    return TargetLayout(slices=tuple(slices))


def _runtime(layout, *, epoch=EPOCH, layout_hash=None):
    return ObjectRuntimeIdentity(
        detector_catalog_hash="d" * 64,
        semantic_plan_hash="s" * 64,
        encoder_name="changed_layout_cpu_gate",
        source_layout_hash="pp4-tp1-dp4",
        target_layout_hash=layout.layout_hash if layout_hash is None else layout_hash,
        epoch=epoch,
        failure_sets=((), (0,), (1,)),
    )


def _snapshot(module, optimizer, masters):
    return {item.address.semantic_id: item.tensor.detach().clone() for item in live_adamw_objects(module, optimizer, masters)}


def _same_tensors(left, right):
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def test_pp4_dp4_materialized_adamw_objects_restore_to_pp2_dp8_global_oracle():
    catalog, oracle = _source_catalog()
    layout = _target_layout()
    assert {item.source_rank for item in catalog.fragments} == set(range(16))
    assert all(item.epoch == item.base_epoch == EPOCH and item.dependency_epoch is None for item in catalog.fragments)
    assert all(item.offsets == (0,) * len(item.address.global_shape) for item in catalog.fragments)
    plan = build_object_recovery_plan(catalog, layout, arm="m1_object", runtime=_runtime(layout))
    restored = restore_object_layout(catalog, plan, layout, failed_hosts=(0,))

    assert set(restored) == set(range(16))
    for target_rank, state in restored.items():
        indices = (0, 1) if target_rank < 8 else (2, 3)
        assert set(state.tensors) == {f"layers.{index}.weight::{role}" for index in indices for role in ROLES}
        module, optimizer, masters = _stage(indices)
        apply_sealed_adamw_objects(module, optimizer, masters, state.tensors)
        for item in live_adamw_objects(module, optimizer, masters):
            expected = oracle[item.address.semantic_id]
            actual = item.tensor.reshape(1) if item.scalar else item.tensor
            assert torch.equal(actual, expected)
        for fqn in (f"layers.{index}.weight" for index in indices):
            master = masters[fqn]
            state_after = optimizer.state[master]
            assert torch.equal(dict(module.named_parameters())[fqn], oracle[f"{fqn}::model"])
            assert torch.equal(master, oracle[f"{fqn}::optimizer.master"])
            assert torch.equal(state_after["exp_avg"], oracle[f"{fqn}::optimizer.exp_avg"])
            assert torch.equal(state_after["exp_avg_sq"], oracle[f"{fqn}::optimizer.exp_avg_sq"])
            assert torch.equal(state_after["step"].reshape(1), oracle[f"{fqn}::optimizer.step"])


def test_changed_layout_rejections_happen_before_live_mutation():
    catalog, _ = _source_catalog()
    layout = _target_layout()
    module, optimizer, masters = _stage((0, 1))
    before = _snapshot(module, optimizer, masters)

    for runtime in (_runtime(layout, epoch=EPOCH + 1), _runtime(layout, layout_hash="wrong-layout")):
        with pytest.raises(ValueError, match="epoch|layout"):
            build_object_recovery_plan(catalog, layout, arm="m1_object", runtime=runtime)
        assert _same_tensors(_snapshot(module, optimizer, masters), before)

    plan = build_object_recovery_plan(catalog, layout, arm="m1_object", runtime=_runtime(layout))
    valid = restore_object_layout(catalog, plan, layout)[0].tensors
    for unknown in ("unknown.weight::model", "layers.0.weight::unknown"):
        invalid = dict(valid)
        invalid[unknown] = invalid.pop("layers.0.weight::model")
        with pytest.raises(ValueError, match="FQN/role"):
            apply_sealed_adamw_objects(module, optimizer, masters, invalid)
        assert _same_tensors(_snapshot(module, optimizer, masters), before)


def test_changed_layout_catalog_rejects_missing_and_overlapping_slices_before_apply():
    address = ObjectAddress("layers.0.weight::model", "model", (3, 2), "torch.float32")
    layout = TargetLayout(slices=(SliceRequest(0, address, (0, 0), (3, 2)),))
    runtime = _runtime(layout)
    missing = ObjectCatalog(
        EPOCH,
        (ObjectFragment.from_tensor(address, epoch=EPOCH, offsets=(0, 0), tensor=torch.ones(1, 2), source_rank=0, storage_host=0),),
    )
    overlapping = ObjectCatalog(
        EPOCH,
        (
            ObjectFragment.from_tensor(address, epoch=EPOCH, offsets=(0, 0), tensor=torch.ones(2, 2), source_rank=0, storage_host=0),
            ObjectFragment.from_tensor(address, epoch=EPOCH, offsets=(1, 0), tensor=torch.ones(2, 2), source_rank=1, storage_host=0),
        ),
    )
    module, optimizer, masters = _stage((0,))
    before = _snapshot(module, optimizer, masters)
    for catalog, message in ((missing, "coverage gap"), (overlapping, "overlap")):
        with pytest.raises(ValueError, match=message):
            build_object_recovery_plan(catalog, layout, arm="m1_object", runtime=runtime)
        assert _same_tensors(_snapshot(module, optimizer, masters), before)
