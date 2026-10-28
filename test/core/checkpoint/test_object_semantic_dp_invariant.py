import pytest
import torch

from Merak.core.checkpoint.catalog import tensor_checksum
from Merak.core.checkpoint.incremental_checkpoint import (
    RecoveryBinding,
    capture_optimizer_update,
)
from Merak.core.checkpoint.object_optimizer_update import (
    externalize_sparse_owned_snapshot_optimizer_update,
)
from Merak.core.checkpoint.object_semantic_dp_invariant import (
    gather_dp_invariant_gradient_encoding,
    staged_object_ids_by_rank,
    verifier_object_ids_by_rank,
)
from Merak.core.checkpoint.object_semantic_plan import (
    GradientShard,
    RankBinding,
    build_active_gradient_plan,
)


def _case():
    bindings = (
        RankBinding(0, 0, 0, 0, 0),
        RankBinding(1, 0, 1, 0, 0),
        RankBinding(2, 0, 0, 1, 1),
        RankBinding(3, 0, 1, 1, 1),
    )
    plan = build_active_gradient_plan(
        tuple(
            GradientShard(
                rank,
                "weight",
                "replicated-tp-and-dp",
                (2,),
                (0,),
                (2,),
                "torch.float32",
            )
            for rank in range(4)
        ),
        bindings,
        arm="m1",
        detector_catalog_hash="a" * 64,
        semantic_plan_hash="b" * 64,
    )
    return plan


def test_verifiers_cover_each_tp_coordinate_and_staging_includes_publishers():
    plan = _case()

    verifiers = verifier_object_ids_by_rank(plan)
    staged = staged_object_ids_by_rank(plan)

    for item in plan.objects:
        assert len(item.tensor_ranks) == 1
        assert len(
            {rank for rank, values in verifiers.items() if item.object_id in values}
        ) == 1
        publishers = {
            placement.publisher_rank
            for placement in plan.placements
            if placement.object_id == item.object_id
        }
        assert len(publishers) == 1
        assert all(item.object_id in staged[rank] for rank in publishers)
    assert sum(
        item.logical_bytes
        for rank, object_ids in staged.items()
        for item in plan.objects
        if item.object_id in object_ids
    ) == sum(item.logical_bytes for item in plan.objects)


def test_dp_invariant_keeps_different_tp_gradients_separate(monkeypatch):
    plan = _case()
    values = {
        item.object_id: torch.tensor([0.25, -0.5]) + item.tensor_ranks[0]
        for item in plan.objects
    }
    checksums = {
        object_id: tensor_checksum(value) for object_id, value in values.items()
    }
    assignments = verifier_object_ids_by_rank(plan)

    monkeypatch.setattr("torch.distributed.is_available", lambda: True)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.get_backend", lambda _group: "gloo")
    monkeypatch.setattr("torch.distributed.get_rank", lambda _group=None: 0)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda _group=None: 4)

    def gather(output, local, *, group):
        gathered = []
        for rank in range(4):
            gathered.append(
                [
                    {
                        "rank": rank,
                        "object_id": object_id,
                        "checksum": checksums[object_id],
                    }
                    for object_id in assignments[rank]
                ]
            )
        assert gathered[0] == local
        output[:] = gathered

    monkeypatch.setattr("torch.distributed.all_gather_object", gather)
    encoding = gather_dp_invariant_gradient_encoding(
        plan,
        {object_id: values[object_id] for object_id in assignments[0]},
        epoch=1,
        group=object(),
    )
    assert encoding.all_equivalences_confirmed
    assert {action.checksum for action in encoding.actions} == set(checksums.values())


def test_sparse_externalization_allows_references_but_requires_publisher_payloads(
    monkeypatch,
):
    plan = _case()
    item = plan.objects[0]
    value = torch.tensor([0.25, -0.5])
    checksum = tensor_checksum(value)
    from Merak.core.checkpoint.object_semantic_plan import confirm_gradient_checksums

    encoding = confirm_gradient_checksums(
        plan,
        {
            (rank, candidate.object_id): (
                checksum
                if candidate.object_id == item.object_id
                else tensor_checksum(value + 1)
            )
            for candidate in plan.objects
            for rank in candidate.holders
        },
        epoch=1,
    )

    def record(rank, gradient):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        parameter.grad = value.clone()
        optimizer = torch.optim.AdamW([parameter], foreach=False, fused=False)
        result = capture_optimizer_update(
            optimizer,
            parameter_ids=(("weight",),),
            binding=RecoveryBinding(rank, 4, rank // 2, 0, rank % 2),
            base_id="sha256:" + "c" * 64,
            base_epoch=0,
            epoch=1,
        )
        result["gradients"] = [[gradient]]
        return result

    nonpublisher = next(
        rank
        for rank in item.holders
        if all(action.source_rank != rank for action in encoding.actions)
    )
    update = externalize_sparse_owned_snapshot_optimizer_update(
        plan, encoding, nonpublisher, record(nonpublisher, None)
    )
    assert not update.payloads
    assert update.records[0][1]["gradients"][0][0]["checksum"] == checksum

    publisher = next(action.source_rank for action in encoding.actions)
    with pytest.raises(ValueError, match="omits a planned publication"):
        externalize_sparse_owned_snapshot_optimizer_update(
            plan, encoding, publisher, record(publisher, None)
        )

    monkeypatch.setattr(
        "Merak.core.checkpoint.object_optimizer_update.tensor_checksum",
        lambda _tensor: (_ for _ in ()).throw(AssertionError("redundant hash")),
    )
    update = externalize_sparse_owned_snapshot_optimizer_update(
        plan, encoding, publisher, record(publisher, value)
    )
    assert update.payloads[0].checksum == checksum
