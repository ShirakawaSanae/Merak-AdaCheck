import pytest
import torch

from Merak.core.checkpoint.object_semantic_plan import (
    GradientShard,
    RankBinding,
    assert_fair_host_global_ab,
    build_active_gradient_plan,
    confirm_gradient_checksums,
    confirm_gradient_equivalence,
)


DETECTOR_HASH = "a" * 64
SEMANTIC_HASH = "b" * 64


def _bindings():
    return tuple(
        RankBinding(
            rank=rank,
            pipeline_rank=rank // 4,
            tensor_rank=0,
            data_rank=rank % 4,
            host=rank // 8,
        )
        for rank in range(16)
    )


def _shards():
    values = []
    for pipeline in range(4):
        ranks = range(4 * pipeline, 4 * (pipeline + 1))
        for index in range(4):
            for rank in ranks:
                values.append(
                    GradientShard(
                        rank,
                        f"stage{pipeline}.weight{index}",
                        f"stage-{pipeline}-class-{index}",
                        (16,),
                        (0,),
                        (16,),
                        "torch.bfloat16",
                    )
                )
    return tuple(values)


def _plan(arm):
    return build_active_gradient_plan(
        _shards(),
        _bindings(),
        arm=arm,
        detector_catalog_hash=DETECTOR_HASH,
        semantic_plan_hash=SEMANTIC_HASH,
        fixed_pairs=tuple((rank, rank + 8) for rank in range(8)),
    )


def test_host_global_m1_balances_publishers_at_equal_remote_bytes():
    adacheck = _plan("adacheck_fixed")
    fixed = _plan("host_global_fixed")
    m1 = _plan("m1")

    assert_fair_host_global_ab(fixed, m1)
    assert fixed.canonical_hash == m1.canonical_hash == adacheck.canonical_hash
    assert fixed.total_remote_logical_bytes == m1.total_remote_logical_bytes
    assert (
        fixed.total_all_publication_logical_bytes
        == m1.total_all_publication_logical_bytes
    )
    assert adacheck.total_remote_logical_bytes == 4 * fixed.total_remote_logical_bytes
    assert m1.maximum_remote_publisher_bytes < fixed.maximum_remote_publisher_bytes
    assert fixed.required_objects_by_host == m1.required_objects_by_host
    assert fixed.failure_sets == m1.failure_sets == ((), (0,), (1,))
    assert fixed.publisher_policy == "host_global_first_eligible"
    assert (
        m1.publisher_policy
        == "host_global_single_publisher_all_destinations_least_load"
    )
    assert (
        m1.maximum_all_publication_publisher_bytes
        < fixed.maximum_all_publication_publisher_bytes
    )
    for placement in m1.placements:
        holders = next(
            item.holders for item in m1.objects if item.object_id == placement.object_id
        )
        if any(
            binding.host == placement.target_host
            for binding in m1.rank_bindings
            if binding.rank in holders
        ):
            assert placement.source_host == placement.target_host
    assert all(
        len(
            {
                placement.publisher_rank
                for placement in m1.placements
                if placement.object_id == item.object_id
            }
        )
        == 1
        for item in m1.objects
    )
    assert fixed.plan_hash != m1.plan_hash


@pytest.mark.parametrize(
    "replacement, message",
    (
        (
            GradientShard(
                1, "weight", "class-a", (8,), (0,), (8,), "torch.float32"
            ),
            "global shape or dtype",
        ),
        (
            GradientShard(
                4, "weight", "class-a", (8,), (0,), (8,), "torch.bfloat16"
            ),
            "pipeline coordinates",
        ),
        (
            GradientShard(
                1, "weight", "class-a", (8,), (0,), (4,), "torch.bfloat16"
            ),
            "slice schema",
        ),
    ),
)
def test_conflicting_equivalence_metadata_fails_closed(replacement, message):
    shards = (
        GradientShard(
            0, "weight", "class-a", (8,), (0,), (8,), "torch.bfloat16"
        ),
        replacement,
    )
    with pytest.raises(ValueError, match=message):
        build_active_gradient_plan(
            shards,
            _bindings(),
            arm="host_global_fixed",
            detector_catalog_hash=DETECTOR_HASH,
            semantic_plan_hash=SEMANTIC_HASH,
        )


def test_full_slice_replicated_across_tp_ranks_stays_tp_local():
    bindings = tuple(
        RankBinding(rank, 0, rank % 2, rank // 2, rank // 2)
        for rank in range(4)
    )
    shards = tuple(
        GradientShard(
            rank, "layer_norm.weight", "class-a", (8,), (0,), (8,),
            "torch.bfloat16",
        )
        for rank in range(4)
    )

    plan = build_active_gradient_plan(
        shards,
        bindings,
        arm="m1",
        detector_catalog_hash=DETECTOR_HASH,
        semantic_plan_hash=SEMANTIC_HASH,
    )

    assert len(plan.objects) == 2
    assert {item.pipeline_rank for item in plan.objects} == {0}
    assert {item.tensor_ranks for item in plan.objects} == {(0,), (1,)}
    assert {item.holders for item in plan.objects} == {(0, 2), (1, 3)}


def test_runtime_checksum_mismatch_uses_predeclared_direct_fallbacks():
    plan = _plan("m1")
    tensors = {
        (rank, item.object_id): torch.ones(item.lengths, dtype=torch.bfloat16)
        for item in plan.objects
        for rank in item.holders
    }
    matched = confirm_gradient_equivalence(plan, tensors, epoch=7)
    assert matched.all_equivalences_confirmed
    assert not matched.mismatched_object_ids
    assert {item.action for item in matched.actions} == {"confirmed_object"}
    assert matched.total_remote_logical_bytes == plan.total_remote_logical_bytes
    assert (
        matched.all_publication_publisher_bytes
        == plan.all_publication_publisher_bytes
    )
    assert (
        matched.total_all_publication_logical_bytes
        == plan.total_all_publication_logical_bytes
    )

    changed = plan.objects[0]
    tensors[(changed.holders[-1], changed.object_id)].fill_(2)
    fallback = confirm_gradient_equivalence(plan, tensors, epoch=7)
    assert not fallback.all_equivalences_confirmed
    assert fallback.mismatched_object_ids == (changed.object_id,)
    actions = [
        item for item in fallback.actions if item.object_id == changed.object_id
    ]
    assert {item.action for item in actions} == {"direct_fallback"}
    assert len({item.version_id for item in actions}) == len(changed.holders)
    assert len(actions) == len(changed.holders) * 2
    assert fallback.total_remote_logical_bytes > plan.total_remote_logical_bytes
    fallback_loads = tuple(
        (
            rank,
            sum(
                action.logical_bytes
                for action in fallback.actions
                if action.source_rank == rank
            ),
        )
        for rank in range(len(plan.rank_bindings))
    )
    assert fallback.all_publication_publisher_bytes == fallback_loads
    assert fallback.total_all_publication_logical_bytes == sum(
        action.logical_bytes for action in fallback.actions
    )
    assert fallback.maximum_all_publication_publisher_bytes == max(
        value for _, value in fallback_loads
    )
    required = dict(fallback.required_versions_by_host)
    assert required[0] == required[1]
    assert fallback.failure_sets == ((), (0,), (1,))
    for failed_host in (0, 1):
        surviving = {
            version_id
            for host, version_ids in required.items()
            if host != failed_host
            for version_id in version_ids
        }
        assert surviving == set(required[0])
    assert len(fallback.encoding_hash) == 64


def test_runtime_confirmation_requires_every_candidate_and_exact_schema():
    plan = _plan("host_global_fixed")
    with pytest.raises(ValueError, match="cover the planned candidates"):
        confirm_gradient_equivalence(plan, {}, epoch=7)

    tensors = {
        (rank, item.object_id): torch.ones(item.lengths, dtype=torch.bfloat16)
        for item in plan.objects
        for rank in item.holders
    }
    first = next(iter(tensors))
    tensors[first] = tensors[first].float()
    with pytest.raises(ValueError, match="planned schema"):
        confirm_gradient_equivalence(plan, tensors, epoch=7)


def test_runtime_confirmation_can_gather_checksums_without_tensor_payloads():
    plan = _plan("m1")
    tensors = {
        (rank, item.object_id): torch.ones(item.lengths, dtype=torch.bfloat16)
        for item in plan.objects
        for rank in item.holders
    }
    expected = confirm_gradient_equivalence(plan, tensors, epoch=7)
    checksum = next(iter(expected.actions)).checksum
    checksums = {key: checksum for key in tensors}

    assert confirm_gradient_checksums(plan, checksums, epoch=7) == expected
    with pytest.raises(ValueError, match="cover the planned candidates"):
        confirm_gradient_checksums(plan, {}, epoch=7)
    checksums[next(iter(checksums))] = "not-a-hash"
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        confirm_gradient_checksums(plan, checksums, epoch=7)


def test_two_host_failure_contract_cannot_be_weakened():
    with pytest.raises(ValueError, match="either host loss"):
        build_active_gradient_plan(
            _shards(),
            _bindings(),
            arm="m1",
            detector_catalog_hash=DETECTOR_HASH,
            semantic_plan_hash=SEMANTIC_HASH,
            failure_sets=((), (0,)),
        )


def test_detector_classes_split_same_fqn_into_rank_specific_objects():
    shards = (
        GradientShard(
            0, "weight", "detector-a", (8,), (0,), (8,), "torch.bfloat16"
        ),
        GradientShard(
            1, "weight", "detector-b", (8,), (0,), (8,), "torch.bfloat16"
        ),
    )
    plan = build_active_gradient_plan(
        shards,
        _bindings(),
        arm="host_global_fixed",
        detector_catalog_hash=DETECTOR_HASH,
        semantic_plan_hash=SEMANTIC_HASH,
    )

    assert len(plan.objects) == 2
    assert {item.detector_class_id for item in plan.objects} == {
        "detector-a",
        "detector-b",
    }
    assert {item.holders for item in plan.objects} == {(0,), (1,)}
    assert len({item.object_id for item in plan.objects}) == 2


def test_epoch_changes_every_confirmed_version_and_encoding_hash():
    plan = _plan("m1")
    tensors = {
        (rank, item.object_id): torch.ones(item.lengths, dtype=torch.bfloat16)
        for item in plan.objects
        for rank in item.holders
    }
    first = confirm_gradient_equivalence(plan, tensors, epoch=7)
    second = confirm_gradient_equivalence(plan, tensors, epoch=8)

    assert (first.epoch, second.epoch) == (7, 8)
    assert {item.version_id for item in first.actions}.isdisjoint(
        item.version_id for item in second.actions
    )
    assert first.required_versions_by_host != second.required_versions_by_host
    assert first.encoding_hash != second.encoding_hash


def test_detector_and_semantic_provenance_is_required_and_hashed():
    with pytest.raises(ValueError, match="detector_catalog_hash"):
        build_active_gradient_plan(
            _shards(),
            _bindings(),
            arm="m1",
            detector_catalog_hash="not-a-hash",
            semantic_plan_hash=SEMANTIC_HASH,
        )
    with pytest.raises(ValueError, match="semantic_plan_hash"):
        build_active_gradient_plan(
            _shards(),
            _bindings(),
            arm="m1",
            detector_catalog_hash=DETECTOR_HASH,
            semantic_plan_hash="not-a-hash",
        )
    changed = build_active_gradient_plan(
        _shards(),
        _bindings(),
        arm="m1",
        detector_catalog_hash="c" * 64,
        semantic_plan_hash=SEMANTIC_HASH,
    )
    assert changed.canonical_hash != _plan("m1").canonical_hash
