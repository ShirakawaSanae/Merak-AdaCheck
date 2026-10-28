import copy
import io
import json
import os
import random
import tempfile
import threading
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from Merak.core.checkpoint.catalog import TensorRedundancyEvidence, checkpoint_path
from Merak.core.checkpoint.zero1_adapter import (
    apply_zero1_fragments,
    apply_zero1_model_tensors,
    capture_zero1_fragments,
    capture_zero1_model_fragments,
    capture_zero1_step_fragments,
    iter_zero1_shards,
    zero1_model_requests,
    zero1_optimizer_requests,
)
from Merak.core.checkpoint.object_recovery import (
    ObjectCatalog, ObjectRuntimeIdentity, SliceRequest, TargetLayout,
    build_object_recovery_plan, restore_object_layout,
)
from Merak.core.checkpoint.zero1_recovery_controller import (
    apply_live_zero1_state, capture_zero1_opaque, capture_zero1_rank,
    build_zero1_object_restore_controller, zero1_rank_metadata,
    zero1_target_layout,
)
from Merak.core.checkpoint.zero1_service import (
    load_zero1_locator, publish_distributed_zero1_epoch, publish_zero1_epoch,
    restore_zero1_from_host, select_zero1_publications, write_zero1_locator,
)
from Merak.core.mpu import initialize as mpu_initialize
from Merak.core.checkpoint.object_replica_service import ObjectReplicaServiceStore
from Merak.core.checkpoint.replica_service import ReplicaServiceClient, ReplicaTCPServer
from Merak.core.moe.expert_parallel import ExpertParallelLayout, create_expert_parallel_groups
from Merak.core.moe.semantic_adapter import MoERank, filter_moe_detector_evidence
from Merak.core.moe.zero1_optimizer import RoleAwareZeroOptimizerStage1
from Merak.utils.trainer_utils import MegatronPretrainingRandomSampler


class _TinyRoleModel(nn.Module):
    def __init__(self, expert_rank):
        super().__init__()
        self.shared = nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
        self.router = nn.Parameter(torch.tensor([0.5, -0.5, 1.5, -1.5, 2.5]))
        self.experts = nn.ParameterDict()
        self.experts[str(expert_rank)] = nn.Parameter(
            torch.tensor([2.0, -1.0, 0.25, -0.75]) + float(expert_rank)
        )
        self.expert._merak_expert_parallel = True
        self.expert._merak_expert_id = expert_rank

    @property
    def expert(self):
        return self.experts[str(next(iter(self.experts)))]


def _average(tensor, group):
    result = tensor.clone()
    dist.all_reduce(result, group=group)
    return result / dist.get_world_size(group=group)


def _worker(rank, init_method):
    dist.init_process_group("gloo", rank=rank, world_size=4, init_method=init_method)
    try:
        layout = ExpertParallelLayout(1, 2, 2, 1, order=("dp", "ep", "pp", "tp"))
        groups = create_expert_parallel_groups(layout)
        model = _TinyRoleModel(groups.ep_rank)
        reference = copy.deepcopy(model)
        base = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.1, foreach=False)
        optimizer = RoleAwareZeroOptimizerStage1(
            SimpleNamespace(ada_checkpoint=False, zero_reshard=""),
            base,
            model=model,
            shared_process_group=dist.group.WORLD,
            expert_process_group=groups.dp_group,
            dynamic_loss_scale=True,
            dynamic_loss_args={
                "INITIAL_LOSS_SCALE": 8.0,
                "SCALE_WINDOW": 1000,
                "MIN_LOSS_SCALE": 1.0,
            },
            clip_grad=0.4,
            max_elements_per_comm=64,
            communication_data_type=torch.float32,
        )

        assert optimizer.group_roles == ["shared", "expert"]
        assert optimizer.partition_counts == [4, 2]
        assert sum(chunk.numel() for group in optimizer.local_sub_partitions_of_fp32_groups for chunk in group) == 4
        assert [group["_merak_zero1_role"] for group in base.param_groups] == ["shared", "expert"]
        scheduler = torch.optim.lr_scheduler.StepLR(base, step_size=1, gamma=0.5)
        model._merak_zero1_shared_process_group = dist.group.WORLD
        model._merak_zero1_expert_process_group = groups.dp_group
        model._merak_ep_layout = layout
        model._merak_role_aware_zero1 = True
        observations = []
        optimizer.set_pre_optimizer_step_callback(
            lambda step, inner, scale: observations.append(
                (step, inner, scale, [
                    gradient.detach().clone()
                    for group in optimizer._reduced_gradients
                    for gradient in group
                ])
            )
        )

        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        model.shared.grad = torch.ones_like(model.shared) * optimizer.loss_scale
        model.router.grad = torch.ones_like(model.router) * optimizer.loss_scale
        model.expert.grad = torch.ones_like(model.expert) * optimizer.loss_scale
        if rank == 3:
            model.expert.grad[0] = float("inf")
        optimizer.reduce_gradients()
        assert optimizer.step() is True
        assert observations == []
        assert optimizer.loss_scale == 4.0
        for name, value in model.named_parameters():
            assert torch.equal(value, before[name])

        shared_grad = torch.tensor([1.0 + rank, 2.0, -1.0])
        router_grad = torch.tensor([0.25, -0.5 * rank, 1.0, -2.0, 0.75])
        expert_grad = (
            torch.tensor([2.0 + rank, -1.0, 0.5, 3.0])
            if groups.ep_rank == 0
            else torch.zeros(4)
        )
        model.shared.grad = shared_grad * optimizer.loss_scale
        model.router.grad = router_grad * optimizer.loss_scale
        model.expert.grad = None if groups.ep_rank == 1 else expert_grad * optimizer.loss_scale

        expected_shared = _average(shared_grad, dist.group.WORLD)
        expected_router = _average(router_grad, dist.group.WORLD)
        expected_expert = _average(expert_grad, groups.dp_group)
        local_norm = torch.tensor(0.0)
        if groups.dp_rank == 0:
            local_norm += expected_expert.square().sum()
        if rank == 0:
            local_norm += expected_shared.square().sum() + expected_router.square().sum()
        dist.all_reduce(local_norm)
        clip_scale = max(1.0, (local_norm.sqrt().item() + 1e-6) / 0.4)
        expected_shared /= clip_scale
        expected_router /= clip_scale
        expected_expert /= clip_scale

        reference_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=0.05, weight_decay=0.1, foreach=False
        )
        reference.shared.grad = expected_shared
        reference.router.grad = expected_router
        reference.expert.grad = expected_expert
        reference_optimizer.step()

        optimizer.reduce_gradients()
        assert optimizer.step() is False
        assert len(observations) == 1
        assert observations[0][0:2] == (2, base)
        assert observations[0][2] > 0
        torch.testing.assert_close(model.shared, reference.shared, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(model.router, reference.router, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(model.expert, reference.expert, atol=1e-6, rtol=1e-6)
        assert sum(shard.length for shard in iter_zero1_shards(model, optimizer)) == 4

        parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
        for group_index, parameters in enumerate(optimizer.fp16_groups):
            process_group = optimizer.dp_process_groups[group_index]
            local_parameters = tuple(
                (parameter_names[id(parameter)], parameter.detach().clone())
                for parameter in parameters
            )
            replicas = [None] * dist.get_world_size(group=process_group)
            dist.all_gather_object(replicas, local_parameters, group=process_group)
            for replica in replicas[1:]:
                assert [name for name, _ in replica] == [name for name, _ in local_parameters]
                assert all(
                    torch.equal(value, replica[index][1])
                    for index, (_, value) in enumerate(local_parameters)
                )

            local_ranges = [
                (parameter_names[id(parameter)], start, start + length)
                for shard in optimizer.zero1_shard_ranges[group_index]
                for parameter, start, length, _ in shard
            ]
            gathered_ranges = [None] * dist.get_world_size(group=process_group)
            dist.all_gather_object(gathered_ranges, local_ranges, group=process_group)
            for parameter in parameters:
                name = parameter_names[id(parameter)]
                intervals = sorted(
                    (start, stop)
                    for rank_ranges in gathered_ranges
                    for range_name, start, stop in rank_ranges
                    if range_name == name
                )
                cursor = 0
                for start, stop in intervals:
                    assert start == cursor
                    cursor = stop
                assert cursor == parameter.numel()

        scheduler.step()
        assert [group["lr"] for group in optimizer.param_groups] == [0.025, 0.025]
        assert len(optimizer.optimizer.state) == 2
        assert all(
            state["step"].item() == 1
            for state in optimizer.optimizer.state.values()
        )
        assert optimizer.global_step == 2  # overflow also advances the wrapper counter

        sampler = MegatronPretrainingRandomSampler(
            total_samples=256,
            consumed_samples=0,
            micro_batch_size=1,
            data_parallel_rank=groups.dp_rank,
            data_parallel_size=layout.dp,
        )
        sampler_iterator = iter(sampler)
        for _ in range(3):
            next(sampler_iterator)
        assert sampler.consumed_samples == 6
        random.seed(901 + rank)
        np.random.seed(1201 + rank)
        torch.manual_seed(1501 + rank)
        state_rows = [None] * 4
        dist.all_gather_object(
            state_rows,
            {"model": {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
            }},
        )
        detector_rows = []
        replica_classes = {}
        expert_ids = {}
        for owner in range(4):
            coordinate = layout.coordinates(owner)
            candidates = (
                ("shared", "shared", None, range(4)),
                ("router", "shared", None, range(4)),
                (f"experts.{coordinate['ep']}", "expert", coordinate["ep"],
                 layout.group_ranks("dp", pp=0, ep=coordinate["ep"], tp=0)),
            )
            for index, (name, replica_class, expert_id, peers) in enumerate(candidates):
                path = checkpoint_path(("model", name))
                detector_rows.append(TensorRedundancyEvidence(
                    owner, "model", path, owner * 3 + index,
                    tuple((peer, peer * 3 + index) for peer in peers),
                ))
                replica_classes[owner, path] = replica_class
                if expert_id is not None:
                    expert_ids[owner, path] = expert_id
                    replica_classes[owner, path] = replica_class
        model_alias_rows = filter_moe_detector_evidence(
            tuple(
                MoERank(
                    owner, 0, layout.coordinates(owner)["ep"],
                    layout.coordinates(owner)["dp"], 0,
                )
                for owner in range(4)
            ),
            detector_rows,
            values_by_rank={owner: value for owner, value in enumerate(state_rows)},
            replica_class_by_path=replica_classes,
            expert_id_by_path=expert_ids,
            host_by_rank={owner: owner // 2 for owner in range(4)},
        ).model_aliases()
        assert set(model_alias_rows) == set(range(4))
        with pytest.raises(ValueError, match="epoch differs"):
            capture_zero1_opaque(model, optimizer, scheduler, epoch=1, sampler=None)
        fragments, opaque = capture_zero1_rank(
            model, optimizer, scheduler, rank=rank, host=rank // 2,
            epoch=2, sampler=sampler,
            storage_hosts=(0, 1),
            model_aliases=model_alias_rows[rank],
            detector_audit={
                "schema_version": 1,
                "evidence_sha256": "a" * 64,
                "coverage": "positive detector proposals only; singleton negatives are not exported",
                "placement_applied": False,
            },
        )
        non_alias_fragments = capture_zero1_model_fragments(
            model, optimizer, rank=rank, storage_host=rank // 2, epoch=2
        ) + capture_zero1_fragments(
            model, optimizer, rank=rank, storage_host=rank // 2, epoch=2
        ) + capture_zero1_step_fragments(
            model, optimizer, rank=rank, storage_host=rank // 2, epoch=2
        )
        assert {item.key for item in opaque} == {"scheduler", "rng", "sampler", "progress"}
        assert set(next(item.value for item in opaque if item.key == "progress")) == {
            "iteration", "optimizer"
        }
        opaque_by_key = {item.key: item.value for item in opaque}
        assert opaque_by_key["sampler"] == {
            "consumed_samples": 6,
            "epoch": sampler.epoch,
        }
        expected_next_sample = tuple(next(sampler_iterator))
        expected_rng = (random.random(), float(np.random.random()), torch.rand(3))
        gathered = [None] * 4
        dist.all_gather_object(gathered, fragments)
        all_source = tuple(item for group in gathered for item in group)
        rank_hosts = {owner: owner // 2 for owner in range(4)}
        model_aliases = model_alias_rows[rank]
        fixed = select_zero1_publications(all_source, rank_hosts=rank_hosts, arm="adacheck_fixed")
        pooled = select_zero1_publications(all_source, rank_hosts=rank_hosts, arm="host_global_fixed")
        planned = select_zero1_publications(all_source, rank_hosts=rank_hosts, arm="m1")
        alias_ids = {
            f"{canonical}::model"
            for row in model_alias_rows.values() for canonical in row.values()
        }
        def alias_bytes(items):
            return sum(
                item.tensor.numel() * item.tensor.element_size()
                for item in items if item.address.semantic_id in alias_ids
            )

        assert all(sum(item.tensor.numel() for item in planned[host]) <= sum(
            item.tensor.numel() for item in fixed[host]
        ) for host in (0, 1))
        assert all(sum(item.tensor.numel() for item in planned[host]) == sum(
            item.tensor.numel() for item in pooled[host]
        ) for host in (0, 1))
        assert all(alias_bytes(planned[host]) <= alias_bytes(all_source) for host in (0, 1))
        assert all(alias_bytes(planned[host]) < alias_bytes(fixed[host]) for host in (0, 1))
        assert all(
            item.address.role != "model"
            for item in all_source
            if item.address.role.startswith("optimizer.")
        )
        optimizer_keys = {
            (item.address.key, item.shard_id, item.version_id)
            for item in all_source if item.address.role.startswith("optimizer.")
        }
        assert all(
            optimizer_keys <= {
                (item.address.key, item.shard_id, item.version_id)
                for item in planned[host]
            }
            for host in (0, 1)
        )
        snapshot = {
            name: value.detach().clone() for name, value in model.named_parameters()
        }
        local_chunks = [
            (chunk.detach().clone(), {
                key: value.detach().clone()
                for key, value in optimizer.optimizer.state[chunk].items()
            })
            for row in optimizer.local_sub_partitions_of_fp32_groups for chunk in row
        ]
        saved_step = optimizer.global_step

        # Take an uninterrupted second step, then replay it after restoring
        # the checkpoint into the same live ZeRO runtime.
        def next_step(target, wrapped, lr_scheduler):
            target.shared.grad = torch.tensor([1.5, -0.25, 0.75]) * wrapped.loss_scale
            target.router.grad = torch.tensor([0.2, -0.3, 0.4, -0.5, 0.6]) * wrapped.loss_scale
            target.expert.grad = torch.tensor([0.5, -0.5, 0.25, -0.25]) * wrapped.loss_scale
            wrapped.reduce_gradients()
            assert wrapped.step() is False
            lr_scheduler.step()

        next_step(model, optimizer, scheduler)
        uninterrupted = {
            name: value.detach().clone() for name, value in model.named_parameters()
        }
        uninterrupted_chunks = [
            (chunk.detach().clone(), {
                key: value.detach().clone()
                for key, value in optimizer.optimizer.state[chunk].items()
            })
            for row in optimizer.local_sub_partitions_of_fp32_groups for chunk in row
        ]
        for value in model.parameters():
            value.data.zero_()
        for row in optimizer.local_sub_partitions_of_fp32_groups:
            for chunk in row:
                chunk.data.zero_()
                optimizer.optimizer.state[chunk]["exp_avg"].zero_()
                optimizer.optimizer.state[chunk]["exp_avg_sq"].zero_()
                optimizer.optimizer.state[chunk]["step"].zero_()
        local = tuple(item for group in gathered for item in group)
        protected_local = ObjectCatalog(2, local + tuple(
            replace(item, storage_host=1 - item.storage_host) for item in local
        ), opaque)
        target_layout = zero1_target_layout(
            model, optimizer, rank=rank, model_aliases=model_aliases
        )
        target_runtime = ObjectRuntimeIdentity(
            detector_catalog_hash="d" * 64, semantic_plan_hash="s" * 64,
            encoder_name="zero1_cpu_gate", source_layout_hash="ep2-dp2",
            target_layout_hash=target_layout.layout_hash, epoch=2,
            failure_sets=((), (0,), (1,)),
        )
        recovery_plan = build_object_recovery_plan(
            protected_local, target_layout, arm="m1_object", runtime=target_runtime
        )
        rebuilt = restore_object_layout(
            protected_local, recovery_plan, target_layout, failed_hosts=(rank // 2,)
        )[rank]
        broken = copy.deepcopy(dict(rebuilt.opaque))
        broken["progress"]["optimizer"]["groups"][0]["_merak_zero1_role"] = "wrong"
        with pytest.raises(ValueError, match="topology differs"):
            apply_live_zero1_state(
                replace(rebuilt, opaque=broken), model, optimizer, scheduler,
                sampler=MegatronPretrainingRandomSampler(
                    total_samples=256,
                    consumed_samples=0,
                    micro_batch_size=1,
                    data_parallel_rank=groups.dp_rank,
                    data_parallel_size=layout.dp,
                ),
                expected_iteration=2,
                runtime_restore=lambda value: value["iteration"],
            )
        assert all(torch.count_nonzero(value) == 0 for value in model.parameters())
        assert all(torch.count_nonzero(chunk) == 0 for row in optimizer.local_sub_partitions_of_fp32_groups for chunk in row)
        restored_sampler = MegatronPretrainingRandomSampler(
            total_samples=256,
            consumed_samples=0,
            micro_batch_size=1,
            data_parallel_rank=groups.dp_rank,
            data_parallel_size=layout.dp,
        )
        assert apply_live_zero1_state(
            rebuilt, model, optimizer, scheduler,
            sampler=restored_sampler,
            expected_iteration=2,
        ) == 2
        assert restored_sampler.consumed_samples == 6
        assert tuple(next(iter(restored_sampler))) == expected_next_sample
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, snapshot[name])
        for (before_chunk, before_slots), chunk in zip(
            local_chunks,
            (value for group in optimizer.local_sub_partitions_of_fp32_groups for value in group),
        ):
            torch.testing.assert_close(chunk, before_chunk)
            for key, value in before_slots.items():
                torch.testing.assert_close(optimizer.optimizer.state[chunk][key], value)
        next_step(model, optimizer, scheduler)
        assert optimizer.global_step == saved_step + 1
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, uninterrupted[name])
        for (expected_chunk, expected_slots), chunk in zip(
            uninterrupted_chunks,
            (value for row in optimizer.local_sub_partitions_of_fp32_groups for value in row),
        ):
            torch.testing.assert_close(chunk, expected_chunk)
            for key, value in expected_slots.items():
                torch.testing.assert_close(optimizer.optimizer.state[chunk][key], value)
        # The surviving CPU objects must also restart a *new* optimizer, not
        # merely repair a dirty optimizer already holding AdamW step state.
        fresh = _TinyRoleModel(groups.ep_rank)
        fresh._merak_zero1_shared_process_group = dist.group.WORLD
        fresh._merak_zero1_expert_process_group = groups.dp_group
        fresh._merak_ep_layout = layout
        fresh._merak_role_aware_zero1 = True
        fresh_base = torch.optim.AdamW(fresh.parameters(), lr=0.05, weight_decay=0.1, foreach=False)
        fresh_optimizer = RoleAwareZeroOptimizerStage1(
            SimpleNamespace(ada_checkpoint=False, zero_reshard=""),
            fresh_base,
            model=fresh,
            shared_process_group=dist.group.WORLD,
            expert_process_group=groups.dp_group,
            dynamic_loss_scale=True,
            dynamic_loss_args={
                "INITIAL_LOSS_SCALE": 8.0,
                "SCALE_WINDOW": 1000,
                "MIN_LOSS_SCALE": 1.0,
            },
            clip_grad=0.4,
            max_elements_per_comm=64,
            communication_data_type=torch.float32,
        )
        fresh_scheduler = torch.optim.lr_scheduler.StepLR(fresh_base, step_size=1, gamma=0.5)
        assert not fresh_optimizer.optimizer.state
        fresh_sampler = MegatronPretrainingRandomSampler(
            total_samples=256,
            consumed_samples=0,
            micro_batch_size=1,
            data_parallel_rank=groups.dp_rank,
            data_parallel_size=layout.dp,
        )
        assert apply_live_zero1_state(
            rebuilt, fresh, fresh_optimizer, fresh_scheduler,
            sampler=fresh_sampler, expected_iteration=2,
        ) == 2
        assert fresh_sampler.consumed_samples == 6
        assert fresh_sampler.epoch == sampler.epoch
        assert tuple(next(iter(fresh_sampler))) == expected_next_sample
        assert random.random() == expected_rng[0]
        assert float(np.random.random()) == expected_rng[1]
        torch.testing.assert_close(torch.rand(3), expected_rng[2], rtol=0, atol=0)
        assert fresh_optimizer.global_step == saved_step
        assert fresh_optimizer.loss_scale == optimizer.loss_scale
        assert fresh_scheduler.state_dict()["last_epoch"] == 1
        for name, value in fresh.named_parameters():
            torch.testing.assert_close(value, snapshot[name])
        for (before_chunk, before_slots), chunk in zip(
            local_chunks,
            (value for row in fresh_optimizer.local_sub_partitions_of_fp32_groups for value in row),
        ):
            torch.testing.assert_close(chunk, before_chunk)
            for key, value in before_slots.items():
                torch.testing.assert_close(fresh_optimizer.optimizer.state[chunk][key], value)
        next_step(fresh, fresh_optimizer, fresh_scheduler)
        assert fresh_scheduler.state_dict() == scheduler.state_dict()
        for name, value in fresh.named_parameters():
            torch.testing.assert_close(value, uninterrupted[name])
        for (expected_chunk, expected_slots), chunk in zip(
            uninterrupted_chunks,
            (value for row in fresh_optimizer.local_sub_partitions_of_fp32_groups for value in row),
        ):
            torch.testing.assert_close(chunk, expected_chunk)
            for key, value in expected_slots.items():
                torch.testing.assert_close(fresh_optimizer.optimizer.state[chunk][key], value)
        # Send the captured CPU objects through two actual TCP services; shut
        # one down and restart another fresh optimizer from its survivor.
        with ExitStack() as services:
            clients = {}
            servers = {}
            if rank == 0:
                for mode in ("centralized", "distributed"):
                    for host in (0, 1):
                        server = ReplicaTCPServer(
                            ("127.0.0.1", 0), ObjectReplicaServiceStore(8 << 20),
                        )
                        thread = threading.Thread(target=server.serve_forever, daemon=True)
                        thread.start()
                        services.callback(thread.join, timeout=5)
                        services.callback(server.server_close)
                        services.callback(server.shutdown)
                        servers[mode, host] = server
                        clients[mode, host] = ReplicaServiceClient(*server.server_address)
            endpoints = [tuple(client.endpoint for client in clients.values()) if rank == 0 else None]
            dist.broadcast_object_list(endpoints, src=0)
            clients = {
                (mode, host): ReplicaServiceClient.from_endpoint(endpoints[0][index])
                for index, (mode, host) in enumerate(
                    (mode, host) for mode in ("centralized", "distributed") for host in (0, 1)
                )
            }
            central = {host: clients["centralized", host] for host in (0, 1)}
            peers = {host: clients["distributed", host] for host in (0, 1)}
            all_opaque = [None] * 4
            dist.all_gather_object(all_opaque, opaque)
            sealed = None
            if rank == 0:
                sealed = publish_zero1_epoch(
                    central, epoch=2,
                    fragments=all_source,
                    opaque=tuple(item for group in all_opaque for item in group),
                    rank_hosts=rank_hosts,
                    arm="m1",
                )
            published = [sealed]
            dist.broadcast_object_list(published, src=0)
            sealed = published[0]
            assert sealed.remote_payload_bytes > 0
            missing_rank, missing_name, missing_alias = next(
                (owner, name, canonical)
                for owner, row in model_alias_rows.items()
                for name, canonical in row.items()
            )
            broken_fragments = fragments
            if rank == missing_rank:
                missing_owner_fragment = next(
                    item for item in fragments
                    if item.address.semantic_id == f"{missing_alias}::model"
                    and item.address.role == "model"
                )
                broken_fragments = tuple(
                    item for item in fragments if item is not missing_owner_fragment
                )
            with pytest.raises(ValueError, match="must bind one model fragment"):
                publish_distributed_zero1_epoch(
                peers, epoch=2, fragments=broken_fragments, opaque=opaque,
                rank_hosts=rank_hosts, arm="m1", group=dist.group.WORLD,
                    model_aliases=model_alias_rows[rank],
                    rank_metadata=zero1_rank_metadata(
                        model, optimizer, rank=rank, host=rank_hosts[rank],
                        layout=layout, model_aliases=model_aliases,
                    ),
                )
            publication_snapshot = [fragments, opaque]
            distributed = publish_distributed_zero1_epoch(
                peers, epoch=2, captured_snapshot=publication_snapshot,
                rank_hosts=rank_hosts, arm="m1", group=dist.group.WORLD,
                model_aliases=model_alias_rows[rank],
                rank_metadata=zero1_rank_metadata(
                    model,
                    optimizer,
                    rank=rank,
                    host=rank_hosts[rank],
                    layout=layout,
                    model_aliases=model_aliases,
                ),
            )
            assert publication_snapshot == []
            assert distributed.epoch == sealed.epoch
            assert distributed.arm == sealed.arm
            assert tuple(dict(row) for row in distributed.model_aliases) == tuple(
                model_alias_rows[owner] for owner in range(4)
            )
            assert distributed.remote_payload_bytes > 0
            observation = distributed.publication_observation
            assert observation.committed_inventory is not None
            assert observation.committed_inventory["planned_equals_committed"]
            assert observation.committed_inventory["host_catalog_hashes_match"]
            assert len(observation.committed_inventory["hosts"]) == 2
            assert observation.remote_payload_bytes == distributed.remote_payload_bytes
            assert observation.remote_wire_bytes >= observation.remote_payload_bytes
            assert observation.remote_logical_bytes >= observation.remote_control_logical_bytes
            assert observation.remote_logical_bytes > observation.remote_control_logical_bytes
            assert observation.remote_edge_payload_bytes > 0
            assert observation.remote_control_payload_bytes > 0
            assert observation.remote_control_wire_bytes >= observation.remote_control_payload_bytes
            assert observation.maximum_remote_put_ms > 0
            assert observation.maximum_remote_edge_ack_ms > 0
            locator_path = Path(init_method.removeprefix("file://")).with_suffix(
                ".zero1-locator.json"
            )
            if rank == 0:
                write_zero1_locator(locator_path, distributed)
                assert load_zero1_locator(locator_path) == distributed
            os.environ["CKPT_REPLICA_ENDPOINTS"] = json.dumps(
                [peers[0].endpoint, peers[1].endpoint]
            )
            os.environ["NODE_RANK"] = str(rank_hosts[rank])
            os.environ["CKPT_OBJECT_SURVIVING_HOST"] = "1"
            os.environ["CKPT_ZERO1_LOCATOR"] = str(locator_path)
            dist.barrier()
            source_reports = [None] * 4
            dist.all_gather_object(
                source_reports,
                zero1_rank_metadata(
                    model,
                    optimizer,
                    rank=rank,
                    host=rank_hosts[rank],
                    layout=layout,
                    model_aliases=model_alias_rows[rank],
                ),
            )
            for host in (0, 1):
                distributed_state = restore_zero1_from_host(
                    peers[host], host=host, sealed=distributed, rank=rank,
                    layout=zero1_target_layout(
                        fresh, fresh_optimizer, rank=rank,
                        model_aliases=model_aliases,
                    ),
                    expected_rank_metadata_rows=source_reports,
                )
                assert set(distributed_state.tensors) == {
                    request.address.semantic_id
                    for request in zero1_target_layout(
                        fresh, fresh_optimizer, rank=rank,
                        model_aliases=model_aliases,
                    ).slices
                }
                assert set(distributed_state.opaque) == {
                    "scheduler", "rng", "sampler", "progress"
                }
            with pytest.raises((RuntimeError, ValueError), match="sealed|control"):
                restore_zero1_from_host(
                    central[1], host=1,
                    sealed=replace(sealed, control_checksum="0" * 64),
                    rank=rank,
                    layout=zero1_target_layout(
                        fresh, fresh_optimizer, rank=rank,
                        model_aliases=model_aliases,
                    ),
                )
            survivor = restore_zero1_from_host(
                peers[1], host=1, sealed=distributed, rank=rank,
                layout=zero1_target_layout(
                    fresh, fresh_optimizer, rank=rank,
                    model_aliases=model_aliases,
                ),
            )
            assert apply_live_zero1_state(
                survivor, fresh, fresh_optimizer, fresh_scheduler,
                sampler=fresh_sampler, expected_iteration=2,
                model_aliases=model_aliases,
            ) == 2
            for name, value in fresh.named_parameters():
                torch.testing.assert_close(value, snapshot[name])
            for (before_chunk, before_slots), chunk in zip(
                local_chunks,
                (value for row in fresh_optimizer.local_sub_partitions_of_fp32_groups for value in row),
            ):
                torch.testing.assert_close(chunk, before_chunk)
                for key, value in before_slots.items():
                    torch.testing.assert_close(fresh_optimizer.optimizer.state[chunk][key], value)
            next_step(fresh, fresh_optimizer, fresh_scheduler)
            assert fresh_scheduler.state_dict() == scheduler.state_dict()
            for name, value in fresh.named_parameters():
                torch.testing.assert_close(value, uninterrupted[name])
            for (expected_chunk, expected_slots), chunk in zip(
                uninterrupted_chunks,
                (value for row in fresh_optimizer.local_sub_partitions_of_fp32_groups for value in row),
            ):
                torch.testing.assert_close(chunk, expected_chunk)
                for key, value in expected_slots.items():
                    torch.testing.assert_close(fresh_optimizer.optimizer.state[chunk][key], value)

            if rank == 0:
                servers["distributed", 0].shutdown()
                servers["distributed", 0].server_close()
            dist.barrier()

            controller_target = _TinyRoleModel(groups.ep_rank)
            controller_base = torch.optim.AdamW(
                controller_target.parameters(), lr=0.05, weight_decay=0.1,
                foreach=False,
            )
            controller_optimizer = RoleAwareZeroOptimizerStage1(
                SimpleNamespace(ada_checkpoint=False, zero_reshard=""),
                controller_base,
                model=controller_target,
                shared_process_group=dist.group.WORLD,
                expert_process_group=groups.dp_group,
                dynamic_loss_scale=True,
                dynamic_loss_args={
                    "INITIAL_LOSS_SCALE": 8.0,
                    "SCALE_WINDOW": 1000,
                    "MIN_LOSS_SCALE": 1.0,
                },
                clip_grad=0.4,
                max_elements_per_comm=64,
                communication_data_type=torch.float32,
            )
            controller_scheduler = torch.optim.lr_scheduler.StepLR(
                controller_base, step_size=1, gamma=0.5
            )
            controller_target._merak_zero1_shared_process_group = dist.group.WORLD
            controller_target._merak_zero1_expert_process_group = groups.dp_group
            controller_target._merak_ep_layout = layout
            controller_target._merak_role_aware_zero1 = True
            controller_sampler = MegatronPretrainingRandomSampler(
                total_samples=256,
                consumed_samples=0,
                micro_batch_size=1,
                data_parallel_rank=groups.dp_rank,
                data_parallel_size=layout.dp,
            )
            mpu_initialize.set_model_parallel_group(dist.group.WORLD)
            mpu_initialize.set_data_parallel_group(dist.group.WORLD)
            mpu_initialize.set_pipe_parallel_group(dist.group.WORLD)
            mpu_initialize.set_sequence_parallel_group(dist.group.WORLD)
            controller = build_zero1_object_restore_controller(
                controller_target,
                controller_optimizer,
                controller_scheduler,
                sampler=controller_sampler,
                args=SimpleNamespace(zero_stage=1, resume_from_checkpoint=None),
                control_group=dist.group.WORLD,
                rank_hosts=rank_hosts,
            )
            resumed = torch.tensor(controller.restore(2), dtype=torch.int64)
            dist.all_reduce(resumed, op=dist.ReduceOp.MIN)
            assert resumed.item() == 2
            assert controller_sampler.consumed_samples == 6
            assert tuple(next(iter(controller_sampler))) == expected_next_sample
            for name, value in controller_target.named_parameters():
                torch.testing.assert_close(value, snapshot[name])
            next_step(controller_target, controller_optimizer, controller_scheduler)
            for name, value in controller_target.named_parameters():
                torch.testing.assert_close(value, uninterrupted[name])
            for (expected_chunk, expected_slots), chunk in zip(
                uninterrupted_chunks,
                (value for row in controller_optimizer.local_sub_partitions_of_fp32_groups for value in row),
            ):
                torch.testing.assert_close(chunk, expected_chunk)
                for key, value in expected_slots.items():
                    torch.testing.assert_close(controller_optimizer.optimizer.state[chunk][key], value)
            dist.barrier()
        non_alias_rows = [None] * 4
        dist.all_gather_object(non_alias_rows, non_alias_fragments)
        if rank == 0:
            source = tuple(item for group in non_alias_rows for item in group)
            catalog = ObjectCatalog(2, source)
            addresses = {item.address.key: item.address for item in catalog.fragments}
            layout = TargetLayout(tuple(
                SliceRequest(0, address, (0,) * len(address.global_shape), address.global_shape)
                for address in addresses.values()
            ))
            runtime = ObjectRuntimeIdentity(
                detector_catalog_hash="d" * 64, semantic_plan_hash="s" * 64,
                encoder_name="zero1_cpu_gate", source_layout_hash="ep2-dp2",
                target_layout_hash=layout.layout_hash, epoch=2,
                failure_sets=((), (0,), (1,)),
            )
            with pytest.raises(ValueError, match="surviving source|coverage gap"):
                build_object_recovery_plan(catalog, layout, arm="m1_object", runtime=runtime)
            # Simulate actual byte-for-byte cross-host protection of *all*
            # complementing local shards, not a duplicate rank-local snapshot.
            protected = ObjectCatalog(2, source + tuple(
                replace(item, storage_host=1 - item.storage_host) for item in source
            ))
            expected = {}
            for item in source:
                tensor = expected.setdefault(
                    item.address.semantic_id,
                    torch.empty(item.address.global_shape, dtype=item.tensor.dtype),
                )
                tensor[tuple(slice(offset, offset + length) for offset, length in zip(item.offsets, item.lengths))] = item.tensor
            fixed = build_object_recovery_plan(protected, layout, arm="adacheck_object_fixed", runtime=runtime)
            m1 = build_object_recovery_plan(protected, layout, arm="m1_object", runtime=runtime)
            for failed in ((0,), (1,)):
                for plan in (fixed, m1):
                    restored = restore_object_layout(protected, plan, layout, failed_hosts=failed)[0].tensors
                    assert restored.keys() == expected.keys()
                    for key, tensor in expected.items():
                        torch.testing.assert_close(restored[key], tensor)
            rank0 = {
                item.address.semantic_id: item.tensor
                for item in non_alias_rows[0]
                if item.address.role in {"model", "model.buffer"}
            }
            rank0_layout = TargetLayout(
                zero1_model_requests(model, rank=0) + zero1_optimizer_requests(model, optimizer, rank=0)
            )
            rank0_runtime = replace(runtime, target_layout_hash=rank0_layout.layout_hash)
            rank0_plan = build_object_recovery_plan(protected, rank0_layout, arm="m1_object", runtime=rank0_runtime)
            recovered = restore_object_layout(protected, rank0_plan, rank0_layout, failed_hosts=(0,))[0].tensors
            recovered_model = {key: value for key, value in recovered.items() if key.endswith(("::model", "::model.buffer"))}
            assert recovered_model.keys() == rank0.keys()
            for key, value in rank0.items():
                torch.testing.assert_close(recovered_model[key], value)
            for key in recovered.keys() - recovered_model.keys():
                torch.testing.assert_close(recovered[key], expected[key])
    finally:
        dist.destroy_process_group()


def _reordered_replicas_worker(rank, init_method):
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=init_method)
    try:
        class ReorderedStage(nn.Module):
            def __init__(self, reverse):
                super().__init__()
                names = ("second", "first") if reverse else ("first", "second")
                for name in names:
                    self.register_parameter(name, nn.Parameter(torch.ones(2)))

        stage = ReorderedStage(reverse=bool(rank))
        optimizer = torch.optim.AdamW(stage.parameters(), lr=0.01, foreach=False)
        with pytest.raises(ValueError, match="matching parameters"):
            RoleAwareZeroOptimizerStage1(
                SimpleNamespace(ada_checkpoint=False, zero_reshard=""),
                optimizer,
                model=stage,
                shared_process_group=dist.group.WORLD,
                expert_process_group=dist.group.WORLD,
                max_elements_per_comm=2,
                communication_data_type=torch.float32,
            )
    finally:
        dist.destroy_process_group()


def test_role_aware_zero1_rejects_reordered_same_shape_replicas():
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            _reordered_replicas_worker,
            args=(f"file://{directory}/reordered-init",),
            nprocs=2,
            join=True,
        )


def test_target_restore_prefers_stage_peers_and_keeps_global_fallback(monkeypatch):
    from Merak.core.checkpoint.object_recovery import ObjectAddress, ObjectFragment, OpaqueRecord, OpaqueRequest
    from Merak.core.checkpoint.matched import CONTROL_SOURCE_RANK

    rank_hosts = {rank: rank // 2 for rank in range(4)}
    addresses = [ObjectAddress(f"stage{stage}", "model", (2,), "torch.float32") for stage in range(2)]
    values = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    fragments = tuple(ObjectFragment.from_tensor(
        addresses[rank // 2], epoch=1, offsets=(0,), tensor=values[rank // 2],
        source_rank=rank, storage_host=rank_hosts[rank],
    ) for rank in range(4))
    roots = tuple(OpaqueRecord.from_value(
        owner_rank=rank, key=key, epoch=1, value={"rank": rank}, storage_hosts=(0, 1),
    ) for rank in range(4) for key in ("scheduler", "rng", "sampler", "progress"))
    metadata = [{"rank": rank, "host": rank_hosts[rank], "layout_hash": "layout",
                 "topology_hash": "topology", "coordinates": {"pp": rank // 2, "tp": 0}}
                for rank in range(4)]
    with ExitStack() as services:
        clients = {}
        for host in (0, 1):
            server = ReplicaTCPServer(("127.0.0.1", 0), ObjectReplicaServiceStore(8 << 20))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            services.callback(thread.join, timeout=5)
            services.callback(server.server_close)
            services.callback(server.shutdown)
            clients[host] = ReplicaServiceClient(*server.server_address)
        sealed = publish_zero1_epoch(clients, epoch=1, fragments=fragments,
                                     opaque=roots, rank_hosts=rank_hosts,
                                     arm="host_global_fixed", rank_metadata=metadata)
        reads = []
        original_get = clients[0].get

        def record_get(source_rank, *args, **kwargs):
            if source_rank != CONTROL_SOURCE_RANK:
                reads.append(source_rank)
            return original_get(source_rank, *args, **kwargs)

        monkeypatch.setattr(clients[0], "get", record_get)
        for stage, expected_reads in ((1, [3, 2]), (0, [3, 2, 0])):
            reads.clear()
            layout = TargetLayout(
                (SliceRequest(3, addresses[stage], (0,), (2,)),),
                (OpaqueRequest(3, 3, "progress"),),
            )
            result = restore_zero1_from_host(clients[0], host=0, sealed=sealed,
                                            rank=3, layout=layout, target_only=True)
            assert torch.equal(result.tensors[addresses[stage].semantic_id], values[stage])
            assert result.opaque["progress"] == {"rank": 3}
            assert reads == expected_reads


def _bf16_master_reconstruction_worker(rank, init_method):
    dist.init_process_group("gloo", rank=rank, world_size=4, init_method=init_method)
    try:
        layout = ExpertParallelLayout(1, 2, 2, 1, order=("dp", "ep", "pp", "tp"))
        groups = create_expert_parallel_groups(layout)
        model = _TinyRoleModel(groups.ep_rank).to(torch.bfloat16)
        model._merak_role_aware_zero1 = True
        base = torch.optim.AdamW(model.parameters(), lr=0.013, foreach=False)
        optimizer = RoleAwareZeroOptimizerStage1(
            SimpleNamespace(ada_checkpoint=False, zero_reshard=""), base,
            model=model, shared_process_group=dist.group.WORLD,
            expert_process_group=groups.dp_group,
            max_elements_per_comm=64, communication_data_type=torch.float32,
        )
        for step in range(3):
            for parameter in model.parameters():
                parameter.grad = torch.full_like(parameter, (rank + step + 1) / 16)
            optimizer.reduce_gradients()
            assert optimizer.step() is False
            parameters = dict(model.named_parameters())
            expected = {name: value.detach().clone() for name, value in parameters.items()}
            fragments = capture_zero1_fragments(
                model, optimizer, rank=rank, storage_host=rank // 2, epoch=step + 1,
            ) + capture_zero1_step_fragments(
                model, optimizer, rank=rank, storage_host=rank // 2, epoch=step + 1,
            )
            assert all(item.address.role.startswith("optimizer.") for item in fragments)
            stream = io.BytesIO()
            torch.save({item.address.semantic_id: item.tensor for item in fragments}, stream)
            for parameter in parameters.values():
                parameter.data.zero_()
            for chunks in optimizer.local_sub_partitions_of_fp32_groups:
                for chunk in chunks:
                    chunk.data.zero_()
                    for value in optimizer.optimizer.state[chunk].values():
                        value.zero_()
            stream.seek(0)
            saved = torch.load(stream, weights_only=True)
            apply_zero1_fragments(
                model, optimizer,
                {key: value for key, value in saved.items() if not key.endswith("::optimizer.step")},
                steps={key: value for key, value in saved.items() if key.endswith("::optimizer.step")},
                rank=rank,
            )
            local = []
            for shard in iter_zero1_shards(model, optimizer):
                master = optimizer.local_sub_partitions_of_fp32_groups[
                    shard.group_index
                ][shard.local_chunk]
                local.append((
                    shard.parameter_id, shard.parameter_offset,
                    master.detach().narrow(0, shard.chunk_offset, shard.length)
                    .to(torch.bfloat16).clone(),
                ))
            # Reconstruct from master ranges across actual owner groups only.
            for group_index, group in enumerate(optimizer.fp16_groups):
                names = {name for name, value in parameters.items()
                         if any(value is parameter for parameter in group)}
                process_group = optimizer.dp_process_groups[group_index]
                pieces = [None] * dist.get_world_size(process_group)
                dist.all_gather_object(
                    pieces, [row for row in local if row[0] in names],
                    group=process_group,
                )
                for name in names:
                    target = expected[name].flatten()
                    rebuilt = torch.empty_like(target)
                    coverage = torch.zeros(target.numel(), dtype=torch.int32)
                    for owner in pieces:
                        for identity, offset, value in owner:
                            if identity == name:
                                rebuilt[offset:offset + value.numel()] = value
                                coverage[offset:offset + value.numel()] += 1
                    assert torch.all(coverage == 1)
                    assert torch.equal(rebuilt, target)
                    parameters[name].data.copy_(rebuilt.reshape_as(parameters[name]))
    finally:
        dist.destroy_process_group()


def test_bf16_model_reconstructs_exactly_from_role_sharded_fp32_masters():
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_bf16_master_reconstruction_worker,
                 args=(f"file://{directory}/master-init",), nprocs=4, join=True)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_role_aware_zero1_matches_adamw_with_clipping_overflow_and_empty_routes():
    with tempfile.TemporaryDirectory() as directory:
        os.environ["CKPT_ZERO1_EXPORT_INVENTORY"] = "1"
        mp.spawn(
            _worker,
            args=(f"file://{directory}/init",),
            nprocs=4,
            join=True,
        )
