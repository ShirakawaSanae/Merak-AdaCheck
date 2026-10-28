import copy
import hashlib
from dataclasses import replace

import pytest
import torch

from Merak.core.checkpoint.incremental_checkpoint import (
    RecoveryBinding,
    apply_optimizer_update,
    capture_optimizer_update,
)
from Merak.core.checkpoint.object_optimizer_update import (
    externalize_local_optimizer_update,
    externalize_owned_snapshot_optimizer_update,
    externalize_optimizer_updates,
    materialize_optimizer_update,
)
from Merak.core.checkpoint.object_semantic_plan import (
    GradientShard,
    RankBinding,
    build_active_gradient_plan,
    confirm_gradient_equivalence,
)


BASE_ID = "sha256:" + hashlib.sha256(b"base").hexdigest()


def _adam(parameter):
    return torch.optim.AdamW(
        [parameter], lr=0.003, betas=(0.8, 0.95), weight_decay=0.1,
        foreach=False, fused=False,
    )


def _case(left_gradient, right_gradient):
    plan = build_active_gradient_plan(
        (
            GradientShard(0, "weight", "dp", (2,), (0,), (2,), "torch.float32"),
            GradientShard(1, "weight", "dp", (2,), (0,), (2,), "torch.float32"),
        ),
        (
            RankBinding(0, 0, 0, 0, 0),
            RankBinding(1, 0, 0, 1, 1),
        ),
        arm="m1",
        detector_catalog_hash="a" * 64,
        semantic_plan_hash="b" * 64,
    )
    item = plan.objects[0]
    tensors = {(0, item.object_id): left_gradient, (1, item.object_id): right_gradient}
    encoding = confirm_gradient_equivalence(plan, tensors, epoch=1)
    records = {}
    for rank, gradient in enumerate((left_gradient, right_gradient)):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        optimizer = _adam(parameter)
        parameter.grad = gradient.clone()
        records[rank] = capture_optimizer_update(
            optimizer,
            parameter_ids=(("weight",),),
            binding=RecoveryBinding(rank, 2, rank, 0, 0),
            base_id=BASE_ID,
            base_epoch=0,
            epoch=1,
            metadata={"rank": rank},
            runtime_state={"loss_scale": 128.0 + rank},
        )
    return plan, encoding, records


def _four_rank_case(arm, *, mismatch=False):
    bindings = tuple(
        RankBinding(rank, 0, 0, rank, rank // 2) for rank in range(4)
    )
    plan = build_active_gradient_plan(
        tuple(
            GradientShard(
                rank, "weight", "dp", (2,), (0,), (2,), "torch.float32"
            )
            for rank in range(4)
        ),
        bindings,
        arm=arm,
        detector_catalog_hash="a" * 64,
        semantic_plan_hash="b" * 64,
        fixed_pairs=((0, 2), (1, 3)),
    )
    gradients = [torch.tensor([0.25, -0.5]) for _ in range(4)]
    if mismatch:
        for rank, gradient in enumerate(gradients):
            gradient.add_(rank)
    item = plan.objects[0]
    encoding = confirm_gradient_equivalence(
        plan,
        {(rank, item.object_id): gradient for rank, gradient in enumerate(gradients)},
        epoch=1,
    )
    records = {}
    for rank, gradient in enumerate(gradients):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        optimizer = _adam(parameter)
        parameter.grad = gradient
        records[rank] = capture_optimizer_update(
            optimizer,
            parameter_ids=(("weight",),),
            binding=RecoveryBinding(rank, 4, rank, 0, 0),
            base_id=BASE_ID,
            base_epoch=0,
            epoch=1,
        )
    return plan, encoding, records


def _materialize(update, plan, encoding, rank, **kwargs):
    return materialize_optimizer_update(
        update,
        rank,
        expected_epoch=1,
        detector_catalog_hash=plan.detector_catalog_hash,
        semantic_plan_hash=plan.semantic_plan_hash,
        plan_hash=plan.plan_hash,
        canonical_hash=plan.canonical_hash,
        encoding_hash=encoding.encoding_hash,
        **kwargs,
    )


def _apply(record, rank):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = _adam(parameter)
    apply_optimizer_update(
        optimizer,
        record,
        expected_parameter_ids=(("weight",),),
        expected_binding=RecoveryBinding(rank, 2, rank, 0, 0),
        expected_base_id=BASE_ID,
        expected_base_epoch=0,
        expected_epoch=1,
    )
    return parameter.detach(), optimizer.state_dict()


def _assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_nested_equal(a, b)
    else:
        assert left == right


def test_one_confirmed_payload_rebuilds_two_rank_records_and_adamw_replay():
    gradient = torch.tensor([0.25, -0.5])
    plan, encoding, records = _case(gradient.clone(), gradient.clone())
    update = externalize_optimizer_updates(plan, encoding, records)

    assert len(update.payloads) == 1
    for rank in (0, 1):
        rebuilt = _materialize(update, plan, encoding, rank)
        assert rebuilt["metadata"] == {"rank": rank}
        assert rebuilt["runtime_state"] == {"loss_scale": 128.0 + rank}
        assert rebuilt["param_group_options"] == records[rank]["param_group_options"]
        _assert_nested_equal(_apply(rebuilt, rank), _apply(records[rank], rank))


def test_mismatch_uses_rank_payload_and_replays_like_direct_record():
    plan, encoding, records = _case(
        torch.tensor([0.25, -0.5]), torch.tensor([-0.75, 0.125])
    )
    update = externalize_optimizer_updates(plan, encoding, records)

    assert encoding.mismatched_object_ids == (plan.objects[0].object_id,)
    assert len(update.payloads) == 2
    assert len({payload.version_id for payload in update.payloads}) == 2
    for rank in (0, 1):
        rebuilt = _materialize(update, plan, encoding, rank)
        assert torch.equal(rebuilt["gradients"][0][0], records[rank]["gradients"][0][0])
        _assert_nested_equal(_apply(rebuilt, rank), _apply(records[rank], rank))


def test_materialization_rejects_provenance_payload_and_reference_tampering():
    gradient = torch.tensor([0.25, -0.5])
    plan, encoding, records = _case(gradient.clone(), gradient.clone())
    update = externalize_optimizer_updates(plan, encoding, records)
    payload = update.payloads[0]

    with pytest.raises(ValueError, match="unexpected epoch"):
        materialize_optimizer_update(
            update, 0, expected_epoch=2,
            detector_catalog_hash=plan.detector_catalog_hash,
            semantic_plan_hash=plan.semantic_plan_hash,
            plan_hash=plan.plan_hash, canonical_hash=plan.canonical_hash,
            encoding_hash=encoding.encoding_hash,
        )
    with pytest.raises(ValueError, match="different canonical_hash"):
        materialize_optimizer_update(
            update, 0, expected_epoch=1,
            detector_catalog_hash=plan.detector_catalog_hash,
            semantic_plan_hash=plan.semantic_plan_hash,
            plan_hash=plan.plan_hash, canonical_hash="c" * 64,
            encoding_hash=encoding.encoding_hash,
        )
    with pytest.raises(ValueError, match="missing or unknown"):
        _materialize(update, plan, encoding, 0, payloads={})
    with pytest.raises(ValueError, match="missing or unknown"):
        unknown = replace(payload, version_id="d" * 64)
        _materialize(
            update, plan, encoding, 0,
            payloads={payload.version_id: payload, unknown.version_id: unknown},
        )

    bad_payload = replace(payload, tensor=payload.tensor + 1)
    with pytest.raises(ValueError, match="tensor failed validation"):
        _materialize(
            update, plan, encoding, 0,
            payloads={payload.version_id: bad_payload},
        )

    records_copy = copy.deepcopy(dict(update.records))
    records_copy[0]["gradients"][0][0]["parameter_id"] = "other"
    bad_update = replace(update, records=tuple(sorted(records_copy.items())))
    with pytest.raises(ValueError, match="reference and payload disagree"):
        _materialize(bad_update, plan, encoding, 0)


def test_externalization_rejects_encoding_hash_and_parameter_fqn_mismatch():
    gradient = torch.tensor([0.25, -0.5])
    plan, encoding, records = _case(gradient.clone(), gradient.clone())
    with pytest.raises(ValueError, match="encoding hash"):
        externalize_optimizer_updates(
            plan, replace(encoding, encoding_hash="f" * 64), records
        )
    changed = copy.deepcopy(records)
    changed[0]["parameter_ids"][0][0] = "different.weight"
    with pytest.raises(ValueError, match="FQN/parameter ids"):
        externalize_optimizer_updates(plan, encoding, changed)


@pytest.mark.parametrize(
    "arm,mismatch,expected_publishers",
    (
        ("adacheck_fixed", False, {0, 1, 2, 3}),
        ("host_global_fixed", False, {0, 2}),
        ("m1", True, {0, 1, 2, 3}),
    ),
)
def test_local_externalization_matches_oracle_and_publisher_contract(
    arm, mismatch, expected_publishers
):
    plan, encoding, records = _four_rank_case(arm, mismatch=mismatch)
    oracle = externalize_optimizer_updates(plan, encoding, records)
    locals_by_rank = {
        rank: externalize_local_optimizer_update(plan, encoding, rank, record)
        for rank, record in records.items()
    }

    assert {
        rank: local.records[0][1] for rank, local in locals_by_rank.items()
    } == dict(oracle.records)
    assert {
        rank for rank, local in locals_by_rank.items() if local.payloads
    } == expected_publishers
    produced = {
        (rank, payload.version_id)
        for rank, local in locals_by_rank.items()
        for payload in local.payloads
    }
    required = {
        (action.source_rank, action.version_id) for action in encoding.actions
    }
    assert produced == required
    assert all(len(local.payloads) <= 1 for local in locals_by_rank.values())

    if mismatch:
        assert len({payload.version_id for local in locals_by_rank.values()
                    for payload in local.payloads}) == 4
    else:
        assert len({version for _, version in produced}) == 1


def test_local_externalization_does_not_run_all_rank_confirmation(monkeypatch):
    plan, encoding, records = _four_rank_case("host_global_fixed")

    def fail(*args, **kwargs):
        raise AssertionError("local path must not confirm all-rank tensors")

    monkeypatch.setattr(
        "Merak.core.checkpoint.object_optimizer_update.confirm_gradient_equivalence",
        fail,
    )
    local = externalize_local_optimizer_update(plan, encoding, 1, records[1])
    assert len(local.records) == 1
    assert not local.payloads


def test_owned_snapshot_externalization_reuses_only_private_cpu_storage():
    gradient = torch.tensor([0.25, -0.5])
    plan, encoding, records = _case(gradient.clone(), gradient.clone())
    record = records[0]
    default = externalize_local_optimizer_update(plan, encoding, 0, record)
    owned = externalize_owned_snapshot_optimizer_update(plan, encoding, 0, record)

    assert default.records == owned.records
    assert torch.equal(default.payloads[0].tensor, owned.payloads[0].tensor)
    source = record["gradients"][0][0]
    assert default.payloads[0].tensor.data_ptr() != source.data_ptr()
    assert owned.payloads[0].tensor.data_ptr() == source.data_ptr()

    source.add_(1)
    assert torch.equal(default.payloads[0].tensor, gradient)
    assert not torch.equal(owned.payloads[0].tensor, gradient)


@pytest.mark.parametrize(
    "mutation,message",
    (
        (lambda record: record["binding"].update(rank=1), "different planned rank"),
        (lambda record: record["parameter_ids"][0].__setitem__(0, "other"), "FQN"),
        (
            lambda record: record["gradients"][0].__setitem__(0, torch.ones(3)),
            "shape or dtype",
        ),
        (lambda record: record["gradients"][0][0].add_(1), "checksum"),
    ),
)
def test_local_externalization_validates_rank_identity_and_tensor(mutation, message):
    plan, encoding, records = _four_rank_case("host_global_fixed")
    record = copy.deepcopy(records[0])
    mutation(record)
    with pytest.raises(ValueError, match=message):
        externalize_local_optimizer_update(plan, encoding, 0, record)
