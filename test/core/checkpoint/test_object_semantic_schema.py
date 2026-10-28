import copy

import pytest

from Merak.core.checkpoint.object_semantic_plan import (
    GradientShard,
    RankBinding,
    build_active_gradient_plan,
    confirm_gradient_checksums,
)
from Merak.core.checkpoint.object_semantic_schema import (
    export_gradient_encoding_metadata,
    recover_gradient_encoding,
)


def _plan(arm):
    bindings = tuple(
        RankBinding(rank, 0, 0, rank, 0 if rank < 2 else 1)
        for rank in range(4)
    )
    return build_active_gradient_plan(
        tuple(
            GradientShard(
                rank, "weight", "class-a", (8,), (0,), (8,), "torch.float32"
            )
            for rank in range(4)
        ),
        bindings,
        arm=arm,
        detector_catalog_hash="a" * 64,
        semantic_plan_hash="b" * 64,
        fixed_pairs=((0, 2), (1, 3)),
    )


def _encoding(plan, *, mismatch=False):
    checksums = {
        (rank, item.object_id): "c" * 64
        for item in plan.objects
        for rank in item.holders
    }
    if mismatch:
        item = plan.objects[0]
        checksums[(item.holders[-1], item.object_id)] = "d" * 64
    return confirm_gradient_checksums(plan, checksums, epoch=7)


@pytest.mark.parametrize("arm", ("adacheck_fixed", "host_global_fixed", "m1"))
@pytest.mark.parametrize("mismatch", (False, True))
def test_per_epoch_metadata_round_trips_all_arms(arm, mismatch):
    plan = _plan(arm)
    encoding = _encoding(plan, mismatch=mismatch)

    control = export_gradient_encoding_metadata(plan, encoding)

    assert recover_gradient_encoding(plan, control, epoch=7) == encoding
    assert all(
        set(item) == {"rank", "object_id", "checksum"}
        for item in control["evidence"]
    )
    by_object = {}
    for item in control["evidence"]:
        by_object.setdefault(item["object_id"], {})[item["rank"]] = item["checksum"]
    for item in plan.objects:
        assert set(by_object[item.object_id]) == set(item.holders)
        if item.object_id not in encoding.mismatched_object_ids:
            assert len(set(by_object[item.object_id].values())) == 1
    for action in encoding.actions:
        if action.action == "direct_fallback":
            assert by_object[action.object_id][action.source_rank] == action.checksum


def test_metadata_fails_closed_for_invalid_or_tampered_control():
    plan = _plan("m1")
    control = export_gradient_encoding_metadata(plan, _encoding(plan, mismatch=True))
    changed = []

    missing = copy.deepcopy(control)
    missing["evidence"].pop()
    changed.append(missing)

    duplicate = copy.deepcopy(control)
    duplicate["evidence"].append(copy.deepcopy(duplicate["evidence"][0]))
    changed.append(duplicate)

    unknown_rank = copy.deepcopy(control)
    unknown_rank["evidence"][0]["rank"] = 99
    changed.append(unknown_rank)

    unknown_object = copy.deepcopy(control)
    unknown_object["evidence"][0]["object_id"] = "e" * 64
    changed.append(unknown_object)

    invalid_digest = copy.deepcopy(control)
    invalid_digest["evidence"][0]["checksum"] = "C" * 64
    changed.append(invalid_digest)

    for value in changed:
        with pytest.raises(ValueError):
            recover_gradient_encoding(plan, value, epoch=7)

    for field, value in (
        ("encoding_hash", "f" * 64),
        ("canonical_hash", "0" * 64),
        ("failure_sets", [[], [0]]),
    ):
        tampered = copy.deepcopy(control)
        tampered["provenance"][field] = value
        with pytest.raises(ValueError):
            recover_gradient_encoding(plan, tampered, epoch=7)

    with pytest.raises(ValueError):
        recover_gradient_encoding(plan, control, epoch=8)
    with pytest.raises(ValueError):
        recover_gradient_encoding(_plan("host_global_fixed"), control, epoch=7)
