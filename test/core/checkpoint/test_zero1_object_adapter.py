from types import SimpleNamespace

import pytest
import torch

from Merak.core.checkpoint.object_recovery import (
    ObjectCatalog,
    ObjectRuntimeIdentity,
    SliceRequest,
    TargetLayout,
    build_object_recovery_plan,
    restore_object_layout,
)
from Merak.core.checkpoint.zero1_adapter import (
    apply_zero1_model_tensors,
    apply_zero1_fragments,
    capture_zero1_fragments,
    capture_zero1_model_fragments,
    capture_zero1_step_fragments,
    iter_zero1_shards,
    zero1_model_requests,
    zero1_optimizer_requests,
)
from Merak.core.checkpoint.zero1_service import select_zero1_publications


def _optimizer(model, *, initialized):
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    second = torch.nn.Parameter(torch.tensor([4.0, 5.0, 0.0]))
    inner = torch.optim.AdamW([first, second], foreach=False)
    if initialized:
        for chunk in (first, second):
            chunk.grad = torch.ones_like(chunk)
        inner.step()
    return SimpleNamespace(
        optimizer=inner,
        fp16_groups=[[model.a, model.b]],
        params_in_rank_sub_partitions=[[[[model.a], [model.b]]]],
        params_in_rank_sub_partitions_offsets=[[[[0], [0]]]],
        local_sub_partitions_of_fp32_groups=[[first, second]],
        local_rank=0,
        group_roles=("shared",),
    )


def test_zero1_local_chunks_capture_and_restore_without_partial_apply():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.zeros(3))
    model.b = torch.nn.Parameter(torch.zeros(3))
    source = _optimizer(model, initialized=True)
    fragments = capture_zero1_fragments(model, source, rank=0, storage_host=0, epoch=7)
    step_fragments = capture_zero1_step_fragments(model, source, rank=0, storage_host=0, epoch=7)
    ObjectCatalog(7, fragments + step_fragments).validate()
    steps = {item.address.semantic_id: item.tensor for item in step_fragments}
    tensors = {
        f.address.semantic_id: torch.zeros(f.address.global_shape, dtype=f.tensor.dtype)
        for f in fragments
    }
    for fragment in fragments:
        start = fragment.offsets[0]
        tensors[fragment.address.semantic_id][start:start + fragment.lengths[0]] = fragment.tensor
    recovered = _optimizer(model, initialized=False)
    with pytest.raises(ValueError, match="missing"):
        apply_zero1_fragments(model, recovered, {}, steps=steps)
    assert not recovered.optimizer.state
    with pytest.raises(ValueError, match="every exact named AdamW step scalar"):
        apply_zero1_fragments(model, recovered, tensors, steps={})
    apply_zero1_fragments(model, recovered, tensors, steps=steps)
    for left, right in zip(
        source.local_sub_partitions_of_fp32_groups[0],
        recovered.local_sub_partitions_of_fp32_groups[0],
    ):
        assert torch.equal(left, right)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(source.optimizer.state[left][key], recovered.optimizer.state[right][key])


def test_zero1_capture_rejects_missing_optimizer_moments():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.zeros(3))
    model.b = torch.nn.Parameter(torch.zeros(3))
    with pytest.raises(ValueError, match="moments are incomplete"):
        capture_zero1_fragments(model, _optimizer(model, initialized=False), rank=0, storage_host=0, epoch=0)


def test_zero1_manifest_rejects_overlapping_model_parameter_ranges():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.zeros(4))
    model.b = torch.nn.Parameter(torch.zeros(2))
    optimizer = _optimizer(model, initialized=True)
    optimizer.params_in_rank_sub_partitions_offsets[0][0][1][0] = 2
    with pytest.raises(ValueError, match="empty range"):
        capture_zero1_fragments(model, optimizer, rank=0, storage_host=0, epoch=7)


def test_zero1_alignment_padding_is_not_a_trainable_object():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.zeros(3))
    model.b = torch.nn.Parameter(torch.zeros(2))
    source = _optimizer(model, initialized=True)
    fragments = capture_zero1_fragments(model, source, rank=0, storage_host=0, epoch=3)
    assert sum(item.lengths[0] for item in fragments if item.address.role == "optimizer.master") == 5
    tensors = {
        item.address.semantic_id: torch.zeros(item.address.global_shape, dtype=item.tensor.dtype)
        for item in fragments
    }
    for item in fragments:
        start = item.offsets[0]
        tensors[item.address.semantic_id][start:start + item.lengths[0]] = item.tensor
    steps = {
        item.address.semantic_id: item.tensor
        for item in capture_zero1_step_fragments(model, source, rank=0, storage_host=0, epoch=3)
    }
    restored = _optimizer(model, initialized=False)
    before_padding = restored.local_sub_partitions_of_fp32_groups[0][1][2].item()
    apply_zero1_fragments(model, restored, tensors, steps=steps)
    assert restored.local_sub_partitions_of_fp32_groups[0][1][2].item() == before_padding


def test_role_aware_zero1_manifest_maps_rank_local_chunks_across_intervals():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.arange(9, dtype=torch.float32))
    model.b = torch.nn.Parameter(torch.arange(7, dtype=torch.float32) + 10)
    model._merak_role_aware_zero1 = True
    model.a._merak_expert_parallel = True
    model.a._merak_expert_id = 0
    first = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    second = torch.nn.Parameter(torch.arange(4, dtype=torch.float32) + 4)
    inner = torch.optim.AdamW([first, second], foreach=False)
    for chunk in (first, second):
        chunk.grad = torch.ones_like(chunk)
    inner.step()
    source = SimpleNamespace(
        optimizer=inner,
        fp16_groups=[[model.a, model.b]],
        params_in_rank_sub_partitions=[
            [[[model.a], [model.a]], [[model.a], [model.b]]]
        ],
        params_in_rank_sub_partitions_offsets=[
            [[[0], [8]], [[4], [0]]]
        ],
        zero1_shard_ranges=[
            (
                ((model.a, 0, 4, 0),),
                ((model.a, 8, 1, 0), (model.b, 0, 3, 1)),
            )
        ],
        local_sub_partitions_of_fp32_groups=[[first, second]],
        local_ranks=[0],
        group_roles=("shared",),
        partition_counts=[2],
        sub_partition_sizes=[4],
        dp_process_groups=[None],
    )

    shards = iter_zero1_shards(model, source)
    assert [
        (shard.parameter_id, shard.parameter_offset, shard.length,
         shard.local_chunk, shard.chunk_offset)
        for shard in shards
    ] == [
        ("a", 0, 4, 0, 0),
        ("a", 8, 1, 1, 0),
        ("b", 0, 3, 1, 1),
    ]


def test_semantic_zero1_expert_model_identity_matches_capture_request_and_apply():
    model = torch.nn.Module()
    model.expert = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    model._merak_role_aware_zero1 = True
    model._merak_semantic_zero1 = True
    model.expert._merak_expert_parallel = True
    model.expert._merak_expert_id = 3
    optimizer = SimpleNamespace(fp16_groups=[[model.expert]])

    captured = capture_zero1_model_fragments(
        model,
        optimizer,
        rank=2,
        storage_host=0,
        epoch=1,
        detector_evidence={"expert": model.expert.detach().clone()},
    )
    requested = zero1_model_requests(model, rank=2)
    assert captured[0].address == requested[0].address
    assert captured[0].address.semantic_id == "expert::model"
    model.expert.data.zero_()
    apply_zero1_model_tensors(
        model,
        {captured[0].address.semantic_id: captured[0].tensor},
        rank=2,
    )
    torch.testing.assert_close(model.expert, torch.arange(4, dtype=torch.float32))


def test_semantic_zero1_model_capture_requires_detector_evidence_for_each_parameter():
    model = torch.nn.Module()
    model.expert = torch.nn.Parameter(torch.ones(2))
    model._merak_role_aware_zero1 = True
    model._merak_semantic_zero1 = True
    model.expert._merak_expert_parallel = True
    model.expert._merak_expert_id = 0

    with pytest.raises(ValueError, match="lacks matching detector value"):
        capture_zero1_model_fragments(
            model,
            SimpleNamespace(fp16_groups=[[model.expert]]),
            rank=0,
            storage_host=0,
            epoch=1,
        )


def test_role_aware_zero1_padding_only_chunk_is_captured_and_restored():
    model = torch.nn.Module()
    model.expert = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    model._merak_role_aware_zero1 = True
    model.expert._merak_expert_parallel = True
    model.expert._merak_expert_id = 0

    chunks = [torch.nn.Parameter(torch.full((3,), fill)) for fill in (1.0, 2.0)]
    inner = torch.optim.AdamW(chunks, foreach=False)
    for chunk in chunks:
        chunk.grad = torch.ones_like(chunk)
    inner.step()
    optimizer = SimpleNamespace(
        optimizer=inner,
        fp16_groups=[[model.expert]],
        params_in_rank_sub_partitions=[[[[model.expert]], [[]]]],
        params_in_rank_sub_partitions_offsets=[[[[2]], [[]]]],
        zero1_shard_ranges=[(((model.expert, 2, 2, 0),), ())],
        local_sub_partitions_of_fp32_groups=[chunks],
        local_ranks=[1],
        group_roles=("expert",),
        partition_counts=[2],
        sub_partition_sizes=[3],
        dp_process_groups=[None],
    )

    shards = iter_zero1_shards(model, optimizer)
    assert [(item.local_chunk, item.parameter_offset, item.length) for item in shards] == [(0, 2, 2)]
    fragments = capture_zero1_fragments(model, optimizer, rank=5, storage_host=1, epoch=1)
    step_fragments = capture_zero1_step_fragments(
        model, optimizer, rank=5, storage_host=1, epoch=1
    )
    requests = zero1_optimizer_requests(model, optimizer, rank=5)
    catalog = ObjectCatalog(1, fragments + step_fragments)
    catalog.validate()
    layout = TargetLayout(requests)
    runtime = ObjectRuntimeIdentity(
        detector_catalog_hash="d" * 64,
        semantic_plan_hash="s" * 64,
        encoder_name="zero1_padding_only_cpu_gate",
        source_layout_hash="source",
        target_layout_hash=layout.layout_hash,
        epoch=1,
        failure_sets=((),),
    )
    plan = build_object_recovery_plan(catalog, layout, arm="m1_object", runtime=runtime)
    recovered = restore_object_layout(catalog, plan, layout)[5].tensors

    target_chunks = [torch.nn.Parameter(torch.zeros(3)) for _ in range(2)]
    target_inner = torch.optim.AdamW(target_chunks, foreach=False)
    target = SimpleNamespace(
        optimizer=target_inner,
        fp16_groups=[[model.expert]],
        params_in_rank_sub_partitions=[[[[model.expert]], [[]]]],
        params_in_rank_sub_partitions_offsets=[[[[2]], [[]]]],
        zero1_shard_ranges=[(((model.expert, 2, 2, 0),), ())],
        local_sub_partitions_of_fp32_groups=[target_chunks],
        local_ranks=[1],
        group_roles=("expert",),
        partition_counts=[2],
        sub_partition_sizes=[3],
        dp_process_groups=[None],
    )
    apply_zero1_fragments(
        model,
        target,
        {key: value for key, value in recovered.items() if not key.endswith("::optimizer.step")},
        steps={key: value for key, value in recovered.items() if key.endswith("::optimizer.step")},
        rank=5,
    )
    for source_chunk, target_chunk in zip(chunks, target_chunks):
        torch.testing.assert_close(target_chunk, source_chunk)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                target_inner.state[target_chunk][key], inner.state[source_chunk][key]
            )


def test_zero1_step_identity_survives_local_chunk_reordering():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.zeros(3))
    model.b = torch.nn.Parameter(torch.zeros(3))
    source = _optimizer(model, initialized=True)
    source.optimizer.state[source.local_sub_partitions_of_fp32_groups[0][1]]["step"].fill_(4)
    fragments = capture_zero1_fragments(model, source, rank=0, storage_host=0, epoch=2)
    step_fragments = capture_zero1_step_fragments(model, source, rank=0, storage_host=0, epoch=2)
    ObjectCatalog(2, fragments + step_fragments).validate()
    tensors = {fragment.address.semantic_id: fragment.tensor for fragment in fragments}
    steps = {fragment.address.semantic_id: fragment.tensor for fragment in step_fragments}

    target = _optimizer(model, initialized=False)
    target.local_sub_partitions_of_fp32_groups[0].reverse()
    target.params_in_rank_sub_partitions[0][0].reverse()
    target.params_in_rank_sub_partitions_offsets[0][0].reverse()
    apply_zero1_fragments(model, target, tensors, steps=steps)
    for index, key in enumerate(("b", "a")):
        chunk = target.local_sub_partitions_of_fp32_groups[0][index]
        reference = source.local_sub_partitions_of_fp32_groups[0][1 - index]
        for role, state_key in (("optimizer.master", None), ("optimizer.exp_avg", "exp_avg"),
                                ("optimizer.exp_avg_sq", "exp_avg_sq")):
            restored = chunk if state_key is None else target.optimizer.state[chunk][state_key]
            torch.testing.assert_close(restored, tensors[f"{key}::{role}"])
        torch.testing.assert_close(target.optimizer.state[chunk]["step"], source.optimizer.state[reference]["step"])


def test_zero1_step_capture_rejects_one_parameter_with_unequal_chunk_steps():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.zeros(6))
    model.b = torch.nn.Parameter(torch.zeros(3))
    source = _optimizer(model, initialized=True)
    source.fp16_groups = [[model.a]]
    source.params_in_rank_sub_partitions = [[[[model.a], [model.a]]]]
    source.params_in_rank_sub_partitions_offsets = [[[[0], [3]]]]
    source.optimizer.state[source.local_sub_partitions_of_fp32_groups[0][1]]["step"].fill_(4)
    with pytest.raises(ValueError, match="spans chunks with unequal AdamW steps"):
        capture_zero1_step_fragments(model, source, rank=0, storage_host=0, epoch=2)


def test_zero1_restore_rejects_mixed_step_values_in_one_target_chunk():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.zeros(3))
    model.b = torch.nn.Parameter(torch.zeros(3))
    source = _optimizer(model, initialized=True)
    tensors = {
        fragment.address.semantic_id: fragment.tensor
        for fragment in capture_zero1_fragments(model, source, rank=0, storage_host=0, epoch=2)
    }
    steps = {
        fragment.address.semantic_id: fragment.tensor
        for fragment in capture_zero1_step_fragments(model, source, rank=0, storage_host=0, epoch=2)
    }
    steps["b::optimizer.step"] = torch.full((1,), 2.0)
    target = _optimizer(model, initialized=False)
    target.local_sub_partitions_of_fp32_groups = [[torch.nn.Parameter(torch.zeros(6))]]
    target.params_in_rank_sub_partitions = [[[[model.a, model.b]]]]
    target.params_in_rank_sub_partitions_offsets = [[[[0, 0]]]]
    with pytest.raises(ValueError, match="mixes unequal AdamW step"):
        apply_zero1_fragments(model, target, tensors, steps=steps)
    assert not target.optimizer.state


def test_zero1_fragments_use_existing_object_recovery_after_host_loss():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.zeros(3))
    model.b = torch.nn.Parameter(torch.zeros(3))
    source = _optimizer(model, initialized=True)
    source.optimizer.state[source.local_sub_partitions_of_fp32_groups[0][1]]["step"].fill_(4)
    fragments = tuple(
        fragment
        for host in (0, 1)
        for fragment in (
            *capture_zero1_fragments(model, source, rank=8 * host, storage_host=host, epoch=5),
            *capture_zero1_step_fragments(model, source, rank=8 * host, storage_host=host, epoch=5),
        )
    )
    catalog = ObjectCatalog(5, fragments)
    source_fragments = fragments[: len(fragments) // 2]
    requests = tuple(
        SliceRequest(0, item.address, (0,), item.address.global_shape)
        for item in source_fragments
    )
    layout = TargetLayout(requests)
    identity = ObjectRuntimeIdentity(
        detector_catalog_hash="d" * 64,
        semantic_plan_hash="s" * 64,
        encoder_name="zero1_cpu_gate",
        source_layout_hash="source-layout",
        target_layout_hash=layout.layout_hash,
        epoch=5,
        failure_sets=((), (0,), (1,)),
    )
    plan = build_object_recovery_plan(catalog, layout, arm="m1_object", runtime=identity)
    recovered = restore_object_layout(catalog, plan, layout, failed_hosts=(0,))[0].tensors
    target = _optimizer(model, initialized=False)
    target.local_sub_partitions_of_fp32_groups[0].reverse()
    target.params_in_rank_sub_partitions[0][0].reverse()
    target.params_in_rank_sub_partitions_offsets[0][0].reverse()
    apply_zero1_fragments(
        model, target,
        {key: value for key, value in recovered.items() if not key.endswith("::optimizer.step")},
        steps={key: value for key, value in recovered.items() if key.endswith("::optimizer.step")},
    )
    for index, source_index in enumerate((1, 0)):
        original = source.local_sub_partitions_of_fp32_groups[0][source_index]
        rebuilt = target.local_sub_partitions_of_fp32_groups[0][index]
        torch.testing.assert_close(rebuilt, original)
        for slot in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                target.optimizer.state[rebuilt][slot], source.optimizer.state[original][slot]
            )


def test_zero1_model_router_buffer_and_optimizer_recover_from_surviving_host():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    model.b = torch.nn.Parameter(torch.tensor([4.0, 5.0, 6.0]))
    model.register_buffer("router_count", torch.tensor([9], dtype=torch.int64))
    source = _optimizer(model, initialized=True)
    fragments = tuple(
        fragment
        for host in (0, 1)
        for fragment in (
            *capture_zero1_model_fragments(model, source, rank=8 * host, storage_host=host, epoch=5),
            *capture_zero1_fragments(model, source, rank=8 * host, storage_host=host, epoch=5),
            *capture_zero1_step_fragments(model, source, rank=8 * host, storage_host=host, epoch=5),
        )
    )
    catalog = ObjectCatalog(5, fragments)
    target_model = torch.nn.Module()
    target_model.a = torch.nn.Parameter(torch.zeros(3))
    target_model.b = torch.nn.Parameter(torch.zeros(3))
    target_model.register_buffer("router_count", torch.zeros(1, dtype=torch.int64))
    target = _optimizer(target_model, initialized=False)
    model_requests = zero1_model_requests(target_model, rank=0)
    optimizer_requests = tuple(
        SliceRequest(0, item.address, (0,), item.address.global_shape)
        for item in fragments[: len(fragments) // 2]
        if item.address.role.startswith("optimizer.")
    )
    layout = TargetLayout(model_requests + optimizer_requests)
    identity = ObjectRuntimeIdentity(
        detector_catalog_hash="d" * 64,
        semantic_plan_hash="s" * 64,
        encoder_name="zero1_cpu_gate",
        source_layout_hash="source-layout",
        target_layout_hash=layout.layout_hash,
        epoch=5,
        failure_sets=((), (0,), (1,)),
    )
    plan = build_object_recovery_plan(catalog, layout, arm="m1_object", runtime=identity)
    restored = restore_object_layout(catalog, plan, layout, failed_hosts=(0,))[0]
    model_values = {key: value for key, value in restored.tensors.items() if key.endswith("::model") or key.endswith("::model.buffer")}
    apply_zero1_model_tensors(target_model, model_values)
    apply_zero1_fragments(
        target_model, target,
        {key: value for key, value in restored.tensors.items() if key.endswith(("::optimizer.master", "::optimizer.exp_avg", "::optimizer.exp_avg_sq"))},
        steps={key: value for key, value in restored.tensors.items() if key.endswith("::optimizer.step")},
    )
    assert torch.equal(target_model.a, model.a)
    assert torch.equal(target_model.b, model.b)
    assert torch.equal(target_model.router_count, model.router_count)
    for left, right in zip(source.local_sub_partitions_of_fp32_groups[0], target.local_sub_partitions_of_fp32_groups[0]):
        torch.testing.assert_close(left, right)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(source.optimizer.state[left][key], target.optimizer.state[right][key])


def test_zero1_model_apply_rejects_missing_buffer_before_changing_weights():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    model.register_buffer("router_count", torch.tensor([1], dtype=torch.int64))
    with pytest.raises(ValueError, match="complete model or buffer"):
        apply_zero1_model_tensors(model, {"a::model": torch.zeros(2)})
    torch.testing.assert_close(model.a, torch.tensor([1.0, 2.0]))


def test_role_aware_rotary_buffers_use_rank_local_identity_and_shape():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.ones(3))
    model.b = torch.nn.Parameter(torch.ones(3))
    rotary = torch.nn.Module()
    rotary.register_buffer("cos_cached", torch.ones(4, 4, dtype=torch.bfloat16))
    model.rotary = rotary
    model._merak_role_aware_zero1 = True
    source = _optimizer(model, initialized=True)

    captured = capture_zero1_model_fragments(
        model, source, rank=3, storage_host=0, epoch=2,
        tensor_parallel_rank=0, tensor_parallel_world_size=1,
    )
    captured_buffer = next(item for item in captured if item.address.role == "model.buffer")

    target = torch.nn.Module()
    target.a = torch.nn.Parameter(torch.zeros(3))
    target.b = torch.nn.Parameter(torch.zeros(3))
    for parameter in target.parameters():
        parameter.model_parallel = True
    target.rotary = torch.nn.Module()
    target.rotary.register_buffer("cos_cached", torch.zeros(4, 4, dtype=torch.bfloat16))
    target._merak_role_aware_zero1 = True
    requests = zero1_model_requests(
        target, rank=3, tensor_parallel_rank=0, tensor_parallel_world_size=1
    )
    requested_buffer = next(item for item in requests if item.address.role == "model.buffer")

    assert captured_buffer.address == requested_buffer.address
    assert captured_buffer.address.semantic_id.endswith("@rank=3::model.buffer")
    assert captured_buffer.address.global_shape == (4, 4)


def test_role_aware_model_and_optimizer_ids_remain_rank_local():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.ones(3))
    model.b = torch.nn.Parameter(torch.ones(3))
    model._merak_role_aware_zero1 = True
    optimizer = _optimizer(model, initialized=True)

    model_objects = capture_zero1_model_fragments(
        model, optimizer, rank=3, storage_host=0, epoch=2
    )
    optimizer_objects = capture_zero1_fragments(
        model,
        SimpleNamespace(
            **vars(optimizer),
            zero1_shard_ranges=[(((model.a, 0, 3, 0),), ((model.b, 0, 3, 0),))],
        ),
        rank=3,
        storage_host=0,
        epoch=2,
    )
    requests = zero1_model_requests(model, rank=3)

    assert {item.address.semantic_id for item in model_objects} == {
        item.address.semantic_id for item in requests
    }
    assert "a@rank=3::model" in {
        item.address.semantic_id for item in model_objects
    }
    assert "b@rank=3::model" in {
        item.address.semantic_id for item in model_objects
    }
    assert optimizer_objects
    assert all("@rank=3" in item.address.semantic_id for item in optimizer_objects)


def test_nonpersistent_rotary_cache_is_rebuilt_not_recovered():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.ones(3))
    model.b = torch.nn.Parameter(torch.ones(3))
    rotary = torch.nn.Module()
    rotary.register_buffer("cos_cached", torch.ones(4, 4), persistent=False)
    rotary.register_buffer("sin_cached", torch.ones(4, 4), persistent=False)
    rotary.register_buffer("inv_freq", torch.arange(4, dtype=torch.float32))
    model.rotary = rotary
    model._merak_role_aware_zero1 = True
    optimizer = _optimizer(model, initialized=True)

    fragments = capture_zero1_model_fragments(
        model, optimizer, rank=1, storage_host=0, epoch=2
    )
    target = torch.nn.Module()
    target.a = torch.nn.Parameter(torch.zeros(3))
    target.b = torch.nn.Parameter(torch.zeros(3))
    target.rotary = torch.nn.Module()
    target.rotary.register_buffer("cos_cached", torch.zeros(16, 4), persistent=False)
    target.rotary.register_buffer("sin_cached", torch.zeros(16, 4), persistent=False)
    target.rotary.register_buffer("inv_freq", torch.zeros(4))
    target._merak_role_aware_zero1 = True
    requests = zero1_model_requests(target, rank=1)

    captured_ids = {item.address.semantic_id for item in fragments}
    requested_ids = {item.address.semantic_id for item in requests}
    assert captured_ids == requested_ids
    assert not any("cos_cached" in item for item in captured_ids)
    assert not any("sin_cached" in item for item in captured_ids)
    assert any("inv_freq" in item for item in captured_ids)

    apply_zero1_model_tensors(
        target,
        {item.address.semantic_id: item.tensor for item in fragments},
        rank=1,
    )
    torch.testing.assert_close(target.rotary.inv_freq, model.rotary.inv_freq)
    assert tuple(target.rotary.cos_cached.shape) == (16, 4)


def test_tp1_stage_model_parallel_marker_does_not_expand_parameter_address():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.ones(3))
    model.b = torch.nn.Parameter(torch.ones(3))
    for parameter in model.parameters():
        parameter.model_parallel = True
    source = _optimizer(model, initialized=True)

    captured = capture_zero1_model_fragments(
        model, source, rank=1, storage_host=0, epoch=2,
        tensor_parallel_rank=0, tensor_parallel_world_size=1,
    )
    requests = zero1_model_requests(
        model, rank=1, tensor_parallel_rank=0, tensor_parallel_world_size=1
    )

    assert [item.address for item in captured] == [item.address for item in requests]


def test_zero1_fixed_pair_and_host_grouping_share_exact_object_versions():
    from Merak.core.checkpoint.object_recovery import ObjectAddress, ObjectFragment

    original = torch.tensor([3.0, -2.0])
    changed = torch.tensor([3.0, -1.0])
    address = ObjectAddress("block::model", "model", (2,), "torch.float32")
    ranks = {rank: rank // 2 for rank in range(4)}
    fragments = tuple(
        ObjectFragment.from_tensor(
            address, epoch=2, offsets=(0,), tensor=changed if rank == 3 else original,
            source_rank=rank, storage_host=ranks[rank],
        )
        for rank in ranks
    )
    fixed = select_zero1_publications(fragments, rank_hosts=ranks, arm="adacheck_fixed")
    pooled = select_zero1_publications(fragments, rank_hosts=ranks, arm="host_global_fixed")
    m1 = select_zero1_publications(fragments, rank_hosts=ranks, arm="m1")
    assert (len(fixed[0]), len(fixed[1])) == (3, 3)
    assert (len(pooled[0]), len(pooled[1])) == (2, 2)
    assert (len(m1[0]), len(m1[1])) == (2, 2)
    assert any(torch.equal(item.tensor, changed) for item in m1[0])
    assert all(item.storage_host == host for host in (0, 1) for item in m1[host])
    for host in (0, 1):
        with pytest.raises(ValueError, match="replicas of one logical fragment conflict"):
            ObjectCatalog(2, m1[host]).validate()
    # Two versions under one global slice ID are not an admissible checkpoint,
    # irrespective of whether their per-version coverage looks complete.
    admissible = tuple(
        ObjectFragment.from_tensor(
            address, epoch=2, offsets=(0,), tensor=original,
            source_rank=rank, storage_host=ranks[rank],
        )
        for rank in ranks
    )
    fixed = select_zero1_publications(admissible, rank_hosts=ranks, arm="adacheck_fixed")
    pooled = select_zero1_publications(admissible, rank_hosts=ranks, arm="host_global_fixed")
    m1 = select_zero1_publications(admissible, rank_hosts=ranks, arm="m1")
    assert all(len(fixed[host]) == 2 and len(pooled[host]) == len(m1[host]) == 1
               for host in (0, 1))
    for host in (0, 1):
        ObjectCatalog(2, m1[host]).validate()
    with pytest.raises(ValueError, match="source mapping"):
        select_zero1_publications(
            fragments, rank_hosts={0: 0, 1: 1, 2: 1, 3: 1}, arm="adacheck_fixed"
        )


def test_zero1_cross_host_bytes_include_sealed_control():
    from contextlib import ExitStack
    import threading

    from Merak.core.checkpoint.object_recovery import ObjectAddress, ObjectFragment, OpaqueRecord
    from Merak.core.checkpoint.object_replica_service import ObjectReplicaServiceStore
    from Merak.core.checkpoint.replica_service import ReplicaServiceClient, ReplicaTCPServer
    from Merak.core.checkpoint.zero1_service import publish_zero1_epoch

    rank_hosts = {rank: rank // 2 for rank in range(4)}
    address = ObjectAddress("block::model", "model", (1024,), "torch.float32")
    fragments = tuple(
        ObjectFragment.from_tensor(
            address, epoch=2, offsets=(0,), tensor=torch.ones(1024),
            source_rank=rank, storage_host=rank_hosts[rank],
        ) for rank in rank_hosts
    )
    opaque = tuple(
        OpaqueRecord.from_value(
            owner_rank=rank, key=key, epoch=2, value={"step": 2}, storage_hosts=(0, 1),
        ) for rank in rank_hosts for key in ("scheduler", "rng", "sampler", "progress")
    )
    measured = {}
    for arm in ("adacheck_fixed", "host_global_fixed", "m1"):
        with ExitStack() as stack:
            clients = {}
            for host in (0, 1):
                server = ReplicaTCPServer(("127.0.0.1", 0), ObjectReplicaServiceStore(8 << 20))
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                stack.callback(thread.join, timeout=5)
                stack.callback(server.server_close)
                stack.callback(server.shutdown)
                clients[host] = ReplicaServiceClient(*server.server_address)
            sealed = publish_zero1_epoch(
                clients, epoch=2, fragments=fragments, opaque=opaque,
                rank_hosts=rank_hosts, arm=arm,
            )
            remote_data = sum(
                clients[host].get(
                    rank, 2, control_checksum=sealed.control_checksum,
                    expected_manifest=sealed.data_evidence[host][rank],
                )[0].manifest.payload_bytes
                for host in (0, 1) for rank in rank_hosts if rank_hosts[rank] != host
            )
            remote_control = clients[1].get(
                -1, 2, control_checksum=sealed.control_checksum,
                expected_manifest=sealed.control_evidence[1],
            )[0].manifest.payload_bytes
            assert remote_control > 0
            assert sealed.remote_payload_bytes == remote_data + remote_control
            assert sealed.maximum_remote_put_ms > 0
            measured[arm] = sealed.remote_payload_bytes
    assert all(value > 0 for value in measured.values())


def test_zero1_alias_manifest_is_sealed_and_rejects_rank_local_collision(tmp_path):
    from contextlib import ExitStack
    import json
    import threading

    from Merak.core.checkpoint.object_recovery import (
        ObjectAddress, ObjectFragment, OpaqueRecord,
    )
    from Merak.core.checkpoint.object_replica_service import ObjectReplicaServiceStore
    from Merak.core.checkpoint.replica_service import ReplicaServiceClient, ReplicaTCPServer
    from Merak.core.checkpoint.zero1_service import (
        load_zero1_locator, publish_zero1_epoch, write_zero1_locator,
    )

    rank_hosts = {0: 0, 1: 1}
    canonical = "moe/model/test-canonical"
    value = torch.tensor([1.0, 2.0])
    aliases = ({"left.weight": canonical}, {"right.weight": canonical})
    fragments = tuple(
        ObjectFragment.from_tensor(
            ObjectAddress(f"{canonical}::model", "model", (2,), "torch.float32"),
            epoch=1, offsets=(0,), tensor=value,
            source_rank=rank, storage_host=rank_hosts[rank],
        ) for rank in rank_hosts
    )
    opaque = tuple(
        OpaqueRecord.from_value(
            owner_rank=rank, key=key, epoch=1, value={"step": 1},
            storage_hosts=(0, 1),
        )
        for rank in rank_hosts for key in ("scheduler", "rng", "sampler", "progress")
    )
    with ExitStack() as stack:
        clients = {}
        for host in (0, 1):
            server = ReplicaTCPServer(
                ("127.0.0.1", 0), ObjectReplicaServiceStore(8 << 20)
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            stack.callback(thread.join, timeout=5)
            stack.callback(server.server_close)
            stack.callback(server.shutdown)
            clients[host] = ReplicaServiceClient(*server.server_address)
        sealed = publish_zero1_epoch(
            clients, epoch=1, fragments=fragments, opaque=opaque,
            rank_hosts=rank_hosts, arm="m1", model_aliases=aliases,
        )
        assert sealed.model_aliases == aliases
        assert sealed.model_aliases_applied is True
        path = tmp_path / "locator.json"
        write_zero1_locator(path, sealed)
        assert load_zero1_locator(path) == sealed

        legacy = json.loads(path.read_text())
        legacy.pop("model_aliases_applied")
        path.write_text(json.dumps(legacy))
        assert load_zero1_locator(path).model_aliases_applied is False

        with pytest.raises(ValueError, match="duplicate model alias"):
            publish_zero1_epoch(
                clients, epoch=2, fragments=tuple(
                    ObjectFragment.from_tensor(
                        ObjectAddress(f"{canonical}::model", "model", (2,), "torch.float32"),
                        epoch=2, offsets=(0,), tensor=value,
                        source_rank=rank, storage_host=rank_hosts[rank],
                    ) for rank in rank_hosts
                ), opaque=tuple(
                    OpaqueRecord.from_value(
                        owner_rank=rank, key=key, epoch=2, value={"step": 2},
                        storage_hosts=(0, 1),
                    ) for rank in rank_hosts
                    for key in ("scheduler", "rng", "sampler", "progress")
                ),
                rank_hosts=rank_hosts, arm="m1",
                model_aliases=(
                    {"left.weight": canonical, "right.weight": canonical},
                    {"right.weight": canonical},
                ),
            )


def test_zero1_alias_manifest_allows_same_host_replicas():
    from Merak.core.checkpoint.object_recovery import ObjectAddress, ObjectFragment
    from Merak.core.checkpoint.zero1_service import _validate_zero1_model_aliases

    canonical = "moe/model/host-local"
    value = torch.tensor([1.0, 2.0])
    fragments = tuple(
        ObjectFragment.from_tensor(
            ObjectAddress(f"{canonical}::model", "model", (2,), "torch.float32"),
            epoch=1,
            offsets=(0,),
            tensor=value,
            source_rank=rank,
            storage_host=0,
        )
        for rank in (0, 1)
    )

    aliases = _validate_zero1_model_aliases(
        ({"left.weight": canonical}, {"right.weight": canonical}, {}, {}),
        fragments,
        {0: 0, 1: 0, 2: 1, 3: 1},
    )
    assert aliases == (
        {"left.weight": canonical},
        {"right.weight": canonical},
        {},
        {},
    )
