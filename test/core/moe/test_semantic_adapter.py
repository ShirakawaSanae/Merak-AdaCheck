from types import SimpleNamespace
import json

import pytest
import torch

from Merak.core.checkpoint.catalog import TensorRedundancyEvidence, checkpoint_path
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
from Merak.core.checkpoint.reduced import build_reduced_checkpoint, plan_replication_cost
from Merak.core.moe.expert_parallel import ExpertParallelLayout
from Merak.core.moe.semantic_adapter import (
    MoERank,
    build_moe_optimizer_manifest,
    filter_moe_detector_evidence,
    moe_detector_audit_report,
)
from Merak.core.checkpoint.zero1_service import load_zero1_locator


def test_confirmed_moe_groups_keep_topology_specific_detector_sets():
    layout = ExpertParallelLayout(1, 2, 2, 1)
    ranks = tuple(
        MoERank(rank, 0, layout.coordinates(rank)["ep"], layout.coordinates(rank)["dp"], 0)
        for rank in range(layout.world_size)
    )
    shared_path = checkpoint_path(("model", "shared"))
    expert_path = checkpoint_path(("model", "expert"))
    records = []
    states = {}
    classes = {}
    expert_ids = {}
    for row in ranks:
        states[row.rank] = {"model": {"shared": torch.ones(2), "expert": torch.ones(2)}}
        shared_index = row.rank * 2
        expert_index = shared_index + 1
        records.extend((
            TensorRedundancyEvidence(
                row.rank, "model", shared_path, shared_index,
                tuple((peer.rank, peer.rank * 2) for peer in ranks),
            ),
            TensorRedundancyEvidence(
                row.rank, "model", expert_path, expert_index,
                tuple(
                    (peer.rank, peer.rank * 2 + 1)
                    for peer in ranks if peer.expert == row.expert
                ),
            ),
        ))
        classes[row.rank, shared_path] = "shared"
        classes[row.rank, expert_path] = "expert"
        expert_ids[row.rank, expert_path] = row.expert
    groups = filter_moe_detector_evidence(
        ranks,
        records,
        values_by_rank=states,
        replica_class_by_path=classes,
        expert_id_by_path=expert_ids,
        host_by_rank={rank.rank: rank.rank // 4 for rank in ranks},
    )

    assert sorted(len(group) for group in groups.groups) == [2, 2, 4]
    from Merak.core.checkpoint.catalog import iter_checkpoint_objects
    summarized = filter_moe_detector_evidence(
        ranks, records,
        objects_by_rank={rank: tuple(iter_checkpoint_objects(value))
                         for rank, value in states.items()},
        replica_class_by_path=classes,
        expert_id_by_path=expert_ids,
        host_by_rank={rank.rank: rank.rank // 4 for rank in ranks},
    )
    assert summarized == groups
    assert all(len({item.pipeline for item in group}) == 1 for group in groups.groups)
    assert all(len({item.expert for item in group}) == 1 for group in groups.groups)


def test_stage_local_fqns_remain_local_checkpoint_paths():
    layout = ExpertParallelLayout(1, 2, 1, 1)
    ranks = tuple(
        MoERank(rank, 0, rank % 2, 0, 0) for rank in range(layout.world_size)
    )
    detector_paths = [checkpoint_path(("model", f"stage{rank}.weight")) for rank in range(2)]
    records = tuple(
        TensorRedundancyEvidence(
            rank, "model", detector_paths[rank], rank,
            ((0, 0), (1, 1)),
        )
        for rank in range(2)
    )
    groups = filter_moe_detector_evidence(
        ranks,
        records,
        values_by_rank={
            rank: {"model": {f"local{rank}.weight": torch.ones(2)}}
            for rank in range(2)
        },
        replica_class_by_path={(rank, path): "shared" for rank, path in enumerate(detector_paths)},
        expert_id_by_path={},
        host_by_rank={0: 0, 1: 1},
        state_path_by_detector_path={
            (rank, detector_paths[rank]): checkpoint_path(("model", f"local{rank}.weight"))
            for rank in range(2)
        },
    )
    assert len(groups.groups) == 1
    assert {item.name for item in groups.groups[0]} == set(detector_paths)
    assert len({item.semantic_key for item in groups.groups[0]}) == 1
    aliases = groups.model_aliases()
    assert set(aliases) == {0, 1}
    assert aliases[0]["local0.weight"] == aliases[1]["local1.weight"]
    assert aliases[0]["local0.weight"].startswith("moe/model/")


def test_confirmed_stage_local_shared_alias_recovers_after_either_host_loss():
    layout = ExpertParallelLayout(1, 2, 1, 1)
    ranks = tuple(
        MoERank(rank, 0, layout.coordinates(rank)["ep"], 0, 0)
        for rank in range(layout.world_size)
    )
    detector_paths = tuple(
        checkpoint_path(("model", f"stage{rank}.shared.weight"))
        for rank in range(layout.world_size)
    )
    values = torch.tensor([1.0, 2.0, 3.0])
    evidence = tuple(
        TensorRedundancyEvidence(
            rank, "model", detector_paths[rank], rank,
            tuple((peer, peer) for peer in range(layout.world_size)),
        )
        for rank in range(layout.world_size)
    )
    groups = filter_moe_detector_evidence(
        ranks,
        evidence,
        values_by_rank={rank: {"model": {f"local{rank}.weight": values.clone()}}
                        for rank in range(layout.world_size)},
        replica_class_by_path={(rank, path): "shared"
                               for rank, path in enumerate(detector_paths)},
        expert_id_by_path={},
        host_by_rank={0: 0, 1: 1},
        state_path_by_detector_path={
            (rank, path): checkpoint_path(("model", f"local{rank}.weight"))
            for rank, path in enumerate(detector_paths)
        },
    )
    assert len(groups.groups) == 1

    group = groups.groups[0]
    canonical = ObjectAddress(
        f"moe/{group[0].replica_class}/pp{group[0].pipeline}/t{group[0].tensor}/"
        f"{group[0].role}/{group[0].shape}",
        group[0].role,
        group[0].shape,
        group[0].dtype,
    )
    fragments = tuple(
        ObjectFragment.from_tensor(
            canonical,
            epoch=1,
            offsets=(0,),
            tensor=values,
            source_rank=rank,
            storage_host=rank,
        )
        for rank in (item.rank for item in group)
    )
    requests = tuple(
        SliceRequest(
            rank,
            ObjectAddress(f"local{rank}.weight", "model", (3,), "torch.float32"),
            (0,),
            (3,),
            source_address=canonical,
        )
        for rank in (item.rank for item in group)
    )
    target = TargetLayout(requests)
    catalog = ObjectCatalog(1, fragments)
    runtime = ObjectRuntimeIdentity(
        catalog.content_hash, "detector-confirmed-shared", "full", "source", target.layout_hash,
        1, ((), (0,), (1,)),
    )
    for arm in ("adacheck_object_fixed", "m1_object"):
        plan = build_object_recovery_plan(catalog, target, arm=arm, runtime=runtime)
        for failed_host in (0, 1):
            restored = restore_object_layout(
                catalog, plan, target, failed_hosts=(failed_host,)
            )
            for rank in range(2):
                assert torch.equal(restored[rank].tensors[f"local{rank}.weight"], values)


def test_expert_detector_components_do_not_alias_across_expert_ids():
    ranks = (
        MoERank(0, 0, 0, 0, 0),
        MoERank(1, 0, 1, 0, 0),
    )
    path = checkpoint_path(("model", "local.expert.weight"))
    evidence = tuple(
        TensorRedundancyEvidence(rank.rank, "model", path, rank.rank, ((0, 0), (1, 1)))
        for rank in ranks
    )
    groups = filter_moe_detector_evidence(
        ranks,
        evidence,
        values_by_rank={rank.rank: {"model": {"local.expert.weight": torch.ones(2)}}
                        for rank in ranks},
        replica_class_by_path={(rank.rank, path): "expert" for rank in ranks},
        expert_id_by_path={(rank.rank, path): rank.expert for rank in ranks},
        host_by_rank={0: 0, 1: 1},
    )
    assert len(groups.groups) == 2
    assert all(len(group) == 1 for group in groups.groups)
    assert {group[0].expert for group in groups.groups} == {0, 1}


def test_slice_request_alias_must_match_target_tensor_metadata():
    target = ObjectAddress("local.weight", "model", (3,), "torch.float32")
    source = ObjectAddress("canonical.weight", "model", (4,), "torch.float32")
    with pytest.raises(ValueError, match="incompatible"):
        SliceRequest(0, target, (0,), (3,), source_address=source)


@pytest.mark.parametrize("coordinate", [-1, 1.5, True])
def test_invalid_topology_coordinate_is_rejected(coordinate):
    with pytest.raises(ValueError, match="non-negative integers"):
        MoERank(0, pipeline=coordinate, expert=0, replica_dp=0, tensor=0)


def _single_rank_filter(*, replica_class="expert", expert_ids=None, evidence_rank=0):
    rank = MoERank(0, pipeline=0, expert=0, replica_dp=0, tensor=0)
    path = checkpoint_path(("model", "experts.0.weight"))
    evidence = TensorRedundancyEvidence(
        rank=evidence_rank,
        role="model",
        path=path,
        local_index=0,
        equivalent_locations=((evidence_rank, 0),),
    )
    return filter_moe_detector_evidence(
        (rank,),
        (evidence,),
        values_by_rank={0: {"model": {"experts.0.weight": torch.ones(2)}}},
        replica_class_by_path={(evidence_rank, path): replica_class},
        expert_id_by_path=expert_ids or {},
        host_by_rank={0: 0},
    )


def test_expert_detector_candidate_requires_explicit_expert_id():
    with pytest.raises(ValueError, match="expert ID"):
        _single_rank_filter()


def test_detector_candidate_with_unknown_rank_is_rejected():
    with pytest.raises(ValueError, match="unknown detector rank/path"):
        _single_rank_filter(
            expert_ids={(0, checkpoint_path(("model", "experts.0.weight"))): 0},
            evidence_rank=1,
        )


def test_detector_paths_cannot_alias_one_state_tensor():
    rank = MoERank(0, pipeline=0, expert=0, replica_dp=0, tensor=0)
    detector_paths = (
        checkpoint_path(("model", "expert.weight")),
        checkpoint_path(("model", "shared.weight")),
    )
    evidence = tuple(
        TensorRedundancyEvidence(0, "model", path, index, ((0, index),))
        for index, path in enumerate(detector_paths)
    )
    with pytest.raises(ValueError, match="alias"):
        filter_moe_detector_evidence(
            (rank,),
            evidence,
            values_by_rank={0: {"model": {"weight": torch.ones(2)}}},
            replica_class_by_path={(0, path): "shared" for path in detector_paths},
            expert_id_by_path={},
            host_by_rank={0: 0},
            state_path_by_detector_path={
                (0, path): checkpoint_path(("model", "weight")) for path in detector_paths
            },
        )


def test_distinct_paths_may_share_a_detector_hash_bucket():
    ranks = (
        MoERank(0, pipeline=0, expert=0, replica_dp=0, tensor=0),
        MoERank(1, pipeline=0, expert=0, replica_dp=1, tensor=0),
    )
    paths = (
        checkpoint_path(("model", "shared.weight")),
        checkpoint_path(("model", "shared.bias")),
    )
    evidence = tuple(
        TensorRedundancyEvidence(rank, "model", path, 7, ((0, 7), (1, 7)))
        for rank in range(2)
        for path in paths
    )
    states = {
        rank: {"model": {"shared.weight": torch.ones(2), "shared.bias": torch.ones(2)}}
        for rank in range(2)
    }
    groups = filter_moe_detector_evidence(
        ranks,
        evidence,
        values_by_rank=states,
        replica_class_by_path={(rank, path): "shared" for rank in range(2) for path in paths},
        expert_id_by_path={},
        host_by_rank={0: 0, 1: 1},
    )

    assert groups.groups == ()
    assert groups.model_aliases() == {}
    assert groups.ambiguous_proposals_dropped == 4
    assert groups.evidence == ()
    assert moe_detector_audit_report(groups)["ambiguous_proposals_dropped"] == 4


def test_ambiguous_detector_component_with_different_local_names_does_not_alias():
    ranks = tuple(
        MoERank(rank, pipeline=0, expert=0, replica_dp=rank, tensor=0)
        for rank in range(2)
    )
    paths = tuple(
        checkpoint_path(("model", f"stage{rank}.weight")) for rank in range(2)
    )
    evidence = tuple(
        TensorRedundancyEvidence(
            rank, "model", path, rank, ((0, 0), (1, 1))
        )
        for rank, path in enumerate(paths)
    )
    groups = filter_moe_detector_evidence(
        ranks,
        evidence,
        values_by_rank={
            rank: {"model": {f"local{rank}.weight": torch.ones(2)}}
            for rank in range(2)
        },
        replica_class_by_path={(rank, path): "shared" for rank, path in enumerate(paths)},
        expert_id_by_path={},
        host_by_rank={0: 0, 1: 1},
        state_path_by_detector_path={
            (rank, path): checkpoint_path(("model", f"local{rank}.weight"))
            for rank, path in enumerate(paths)
        },
    )

    assert groups.ambiguous_proposals_dropped == 0
    assert sorted(len(group) for group in groups.groups) == [2]
    assert groups.model_aliases()[0]["local0.weight"] == groups.model_aliases()[1]["local1.weight"]


def test_legacy_zero1_locator_without_detector_audit_remains_readable(tmp_path):
    value = {
        "epoch": 2,
        "control_checksum": "a" * 64,
        "control_evidence": [{}, {}],
        "data_evidence": [[], []],
        "remote_payload_bytes": 0,
        "maximum_remote_put_ms": 0.0,
        "arm": "adacheck_fixed",
    }
    path = tmp_path / "legacy-locator.json"
    path.write_text(json.dumps(value))

    assert load_zero1_locator(path).detector_audit is None


def test_detector_audit_report_marks_positive_only_evidence_and_not_placement():
    rank = MoERank(0, pipeline=0, expert=0, replica_dp=0, tensor=0)
    path = checkpoint_path(("model", "shared.weight"))
    evidence = TensorRedundancyEvidence(0, "model", path, 0, ((0, 0),))
    groups = filter_moe_detector_evidence(
        (rank,),
        (evidence,),
        values_by_rank={0: {"model": {"shared.weight": torch.ones(2)}}},
        replica_class_by_path={(0, path): "shared"},
        expert_id_by_path={},
        host_by_rank={0: 0},
    )
    report = moe_detector_audit_report(groups)
    assert report["coverage"] == "positive detector proposals only; singleton negatives are not exported"
    assert report["input_records"] == 1
    assert report["placement_applied"] is False
    assert len(report["evidence_sha256"]) == 64


def test_detector_fqn_can_map_to_a_distinct_local_checkpoint_path():
    rank = MoERank(0, pipeline=0, expert=0, replica_dp=0, tensor=0)
    detector_path = checkpoint_path(("model", "layers.0.experts.0.weight"))
    local_path = checkpoint_path(("model", "forward_funcs.1.experts.0.weight"))
    evidence = TensorRedundancyEvidence(
        rank=0,
        role="model",
        path=detector_path,
        local_index=0,
        equivalent_locations=((0, 0),),
    )
    states = {0: {"model": {"forward_funcs.1.experts.0.weight": torch.ones(2)}}}
    result = filter_moe_detector_evidence(
        (rank,),
        (evidence,),
        values_by_rank=states,
        replica_class_by_path={(0, detector_path): "expert"},
        expert_id_by_path={(0, detector_path): 0},
        host_by_rank={0: 0},
        state_path_by_detector_path={(0, detector_path): local_path},
    )
    assert result.groups[0][0].name == detector_path
    assert result.evidence[0].path == local_path


def test_zero1_manifest_keeps_shared_and_expert_shards_in_their_real_groups():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Parameter(torch.ones(3))
            self.expert = torch.nn.Parameter(torch.ones(2))
            self.expert._merak_expert_parallel = True
            self.expert._merak_expert_id = 5
            self._merak_role_aware_zero1 = True

    model = Model()
    model._merak_role_aware_zero1 = True
    optimizer = SimpleNamespace(
        fp16_groups=[[model.shared], [model.expert]],
        group_roles=("shared", "expert"),
        zero1_shard_ranges=[
            [[(model.shared, 0, 3, 0)]],
            [[(model.expert, 1, 1, 0)]],
        ],
        local_sub_partitions_of_fp32_groups=[[torch.ones(3)], [torch.ones(1)]],
        params_in_rank_sub_partitions=[],
        params_in_rank_sub_partitions_offsets=[],
    )
    shards = build_moe_optimizer_manifest(
        model, optimizer, rank=2, shared_group=(0, 1, 2, 3), expert_group=(0, 2)
    )
    assert [(item.parameter_id, item.expert_id, item.process_group, item.parameter_offset, item.length) for item in shards] == [
        ("shared", None, (0, 1, 2, 3), 0, 3),
        ("expert", 5, (0, 2), 1, 1),
    ]


def test_zero1_manifest_rejects_role_metadata_mismatch():
    model = torch.nn.Linear(2, 2, bias=False)
    model._merak_role_aware_zero1 = True
    parameter = model.weight
    optimizer = SimpleNamespace(
        fp16_groups=[[parameter]],
        group_roles=("expert",),
        zero1_shard_ranges=[[[ (parameter, 0, parameter.numel(), 0) ]]],
        local_sub_partitions_of_fp32_groups=[[torch.ones(parameter.numel())]],
        params_in_rank_sub_partitions=[],
        params_in_rank_sub_partitions_offsets=[],
    )
    with pytest.raises(ValueError, match="group role disagrees"):
        build_moe_optimizer_manifest(
            model, optimizer, rank=0, shared_group=(0,), expert_group=(0,)
        )


def test_sixteen_rank_ep_detector_filter_and_host_failure_reconstruction():
    layout = ExpertParallelLayout(4, 2, 2, 1, order=("dp", "ep", "pp", "tp"))
    ranks = tuple(
        MoERank(
            rank,
            pipeline=layout.coordinates(rank)["pp"],
            expert=layout.coordinates(rank)["ep"],
            replica_dp=layout.coordinates(rank)["dp"],
            tensor=layout.coordinates(rank)["tp"],
        )
        for rank in range(layout.world_size)
    )
    host_by_rank = {rank: int(rank >= 8) for rank in range(layout.world_size)}
    states = {}
    evidence = []
    replica_class_by_path = {}
    expert_id_by_path = {}
    locations_by_signature = {}
    path_by_rank_and_name = {}

    for coordinate in range(layout.world_size):
        coord = layout.coordinates(coordinate)
        rank = layout.rank(pp=coord["pp"], ep=coord["ep"], dp=coord["dp"], tp=coord["tp"])
        shared = torch.tensor([10.0, 20.0]) + coord["pp"]
        expert_id = coord["ep"]
        expert = torch.tensor([30.0, 40.0]) + coord["pp"] + expert_id
        optimizer = torch.tensor([float(rank)])
        state = {
            "model": {"shared": shared, f"experts.{expert_id}.weight": expert},
            "optimizer": {"local_master": optimizer},
            "runtime": {"progress": torch.tensor([3, rank], dtype=torch.int64)},
        }
        states[rank] = state
        for name, tensor, replica_class in (
            ("shared", shared, "shared"),
            (f"experts.{expert_id}.weight", expert, "expert"),
        ):
            path = checkpoint_path(("model", name))
            path_by_rank_and_name[(rank, name)] = path
            replica_class_by_path[(rank, path)] = replica_class
            if replica_class == "expert":
                expert_id_by_path[(rank, path)] = expert_id
            signature = (name, coord["pp"], expert_id if replica_class == "expert" else None)
            locations_by_signature.setdefault(signature, []).append(rank)

    for index, ((name, pp, expert_id), members) in enumerate(sorted(locations_by_signature.items())):
        del pp, expert_id
        members = tuple(sorted(members))
        for rank in members:
            path = path_by_rank_and_name[(rank, name)]
            evidence.append(
                TensorRedundancyEvidence(
                    rank=rank,
                    role="model",
                    path=path,
                    local_index=index * layout.world_size + rank,
                    equivalent_locations=tuple(
                        (other, index * layout.world_size + other) for other in members
                    ),
                )
            )

    candidates = filter_moe_detector_evidence(
        ranks,
        evidence,
        values_by_rank=states,
        replica_class_by_path=replica_class_by_path,
        expert_id_by_path=expert_id_by_path,
        host_by_rank=host_by_rank,
    )
    groups = [tuple(item.rank for item in group) for group in candidates.groups]
    assert len(groups) == 12
    assert candidates.detector_confirmation.confirmed_pairs == 32
    assert candidates.detector_confirmation.rejected_pairs == 0

    shared_groups = [group for group in candidates.groups if group[0].replica_class == "shared"]
    expert_groups = [group for group in candidates.groups if group[0].replica_class == "expert"]
    assert len(shared_groups) == 4
    assert all(len(group) == 4 for group in shared_groups)
    assert len(expert_groups) == 8
    assert all(len(group) == 2 for group in expert_groups)
    assert len(candidates.evidence) == len(evidence)
    assert all(
        {host_by_rank[item.rank] for item in group} == {0, 1}
        for group in candidates.groups
    )

    # The reducer consumes only semantic-filtered evidence. Rank-local
    # optimizer/runtime tensors remain complete in each restoration template.
    reduced = build_reduced_checkpoint(states, candidates.evidence)
    for rank, state in states.items():
        restored = reduced.reconstruct_rank(rank)
        assert torch.equal(restored["optimizer"]["local_master"], state["optimizer"]["local_master"])
        assert torch.equal(restored["runtime"]["progress"], state["runtime"]["progress"])
        for key, tensor in state["model"].items():
            assert torch.equal(restored["model"][key], tensor)

    cost = plan_replication_cost(
        reduced,
        host_by_rank,
        ("host0", "host1"),
        replica_count=2,
        tolerated_failures=1,
    )
    assert cost.unrecoverable_values(("host0",)) == ()
    assert cost.unrecoverable_values(("host1",)) == ()
    assert cost.reduced_replication_bytes < cost.full_replication_bytes
