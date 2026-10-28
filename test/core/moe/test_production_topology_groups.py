import ast
import os
import traceback
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from Merak.core.moe.expert_parallel import (
    ExpertParallelLayout,
    create_expert_parallel_groups,
    create_shared_parameter_group,
)
from Merak.utils.trainer_utils import MegatronPretrainingRandomSampler


def _group_ranks(group):
    if hasattr(dist, "get_process_group_ranks"):
        return tuple(dist.get_process_group_ranks(group))
    world_size = dist.get_world_size(group)
    return tuple(dist.get_global_rank(group, index) for index in range(world_size))


def _production_grid_worker(rank, init_method, order, result_path, pp_first=False):
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    from Merak.core import mpu

    os.environ.update(
        DP="4" if pp_first else "2",
        PP="4",
        TP="1",
        SP="1",
        MERAK_MESH_ORDER=order,
        MERAK_MOE_EP="2" if pp_first else "1",
        MERAK_MOE_REPLICA_DP="2" if pp_first else "1",
    )
    world_size = 16 if pp_first else 8
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    try:
        import Merak.core.mpu.initialize as initialize

        def cpu_mesh(device_type_arg, shape, *, mesh_dim_names):
            del device_type_arg
            return init_device_mesh("cpu", shape, mesh_dim_names=mesh_dim_names)

        initialize.init_device_mesh = cpu_mesh
        grid = mpu.set_topo_grid_communication()
        layout = ExpertParallelLayout(
            pp=4,
            ep=2,
            dp=2 if pp_first else 1,
            tp=1,
            order=("pp", "dp", "ep", "tp")
            if pp_first
            else ("dp", "ep", "pp", "tp"),
        )
        groups = create_expert_parallel_groups(layout, backend="gloo")
        shared_group = create_shared_parameter_group(layout, backend="gloo")
        coordinates = layout.coordinates(rank)
        values = {
            "rank": rank,
            "pipe_rank": grid.get_pipe_parallel_rank(),
            "pipe_group": _group_ranks(grid.get_pipe_parallel_group()),
            "stage_ranks": tuple(
                grid.stage_to_global(stage) for stage in range(grid.get_pipe_parallel_world_size())
            ),
            "data_group": _group_ranks(mpu.get_data_parallel_group()),
            "data_world_size": mpu.get_data_parallel_world_size(),
            "data_rank": mpu.get_data_parallel_rank(),
            "ep_coordinates": layout.coordinates(rank),
            "ep_group": _group_ranks(groups.ep_group),
            "replica_dp_group": _group_ranks(groups.dp_group),
            "layout_pp_group": _group_ranks(groups.pp_group),
            "shared_group": _group_ranks(shared_group),
            "sample": tuple(next(iter(MegatronPretrainingRandomSampler(
                total_samples=64,
                consumed_samples=0,
                micro_batch_size=1,
                data_parallel_rank=layout.coordinates(rank)["dp"],
                data_parallel_size=layout.dp,
            )))),
        }
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, values)
        if rank == 0:
            with open(result_path, "w", encoding="utf-8") as output:
                output.write(repr(gathered))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("order", "expected_pp_match", "expected_data_match"),
    [
        ("dp,pp,sp,tp", True, True),
        ("pp,dp,sp,tp", False, False),
    ],
)
def test_device_mesh_grid_group_report(
    tmp_path, order, expected_pp_match, expected_data_match
):
    result_path = tmp_path / f"groups-{order.replace(',', '-')}.txt"
    mp.spawn(
        _production_grid_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", order, str(result_path)),
        nprocs=8,
        join=True,
    )
    records = ast.literal_eval(result_path.read_text(encoding="utf-8"))
    assert len(records) == 8
    assert [record["rank"] for record in records] == list(range(8))
    assert all(
        (set(record["pipe_group"]) == set(record["layout_pp_group"]))
        == expected_pp_match
        for record in records
    )
    assert all(record["rank"] == record["stage_ranks"][record["pipe_rank"]] for record in records)
    assert all(
        (record["data_group"] == record["ep_group"])
        == expected_data_match
        for record in records
    )
    assert all(record["pipe_group"] == record["stage_ranks"] for record in records)
    assert all(record["data_world_size"] == 2 for record in records)


def test_pp_first_moe_groups_match_rank_formula(tmp_path):
    result_path = tmp_path / "groups-pp-first-moe.txt"
    mp.spawn(
        _production_grid_worker,
        args=(
            f"file://{tmp_path / 'pp-first-moe-rendezvous'}",
            "pp,dp,sp,tp",
            str(result_path),
            True,
        ),
        nprocs=16,
        join=True,
    )
    records = ast.literal_eval(result_path.read_text(encoding="utf-8"))
    assert len(records) == 16
    assert [record["rank"] for record in records] == list(range(16))
    for record in records:
        rank = record["rank"]
        coordinates = record["ep_coordinates"]
        assert rank == 4 * coordinates["pp"] + 2 * coordinates["dp"] + coordinates["ep"]
        assert record["pipe_group"] == record["layout_pp_group"]
        assert record["pipe_group"][record["pipe_rank"]] == rank
        assert len(record["ep_group"]) == 2
        assert len(record["replica_dp_group"]) == 2
        assert len(record["shared_group"]) == 4
        assert record["data_group"] == record["replica_dp_group"]
    for pp in range(4):
        samples_by_dp = {}
        for dp in range(2):
            peers = [
                record for record in records
                if record["ep_coordinates"]["pp"] == pp
                and record["ep_coordinates"]["dp"] == dp
            ]
            assert len(peers) == 2
            assert len({record["sample"] for record in peers}) == 1
            samples_by_dp[dp] = peers[0]["sample"]
        assert samples_by_dp[0] != samples_by_dp[1]


def test_m1_prefers_local_holders_before_balancing_load():
    from Merak.core.checkpoint.object_recovery import ObjectAddress, ObjectFragment
    from Merak.core.checkpoint.zero1_service import select_zero1_publications

    rank_hosts = {0: 0, 1: 0, 2: 1, 3: 1}
    fragments = tuple(
        ObjectFragment.from_tensor(
            ObjectAddress(f"shared-{index}", "model", (8,), "torch.float32"),
            epoch=1, offsets=(0,), tensor=torch.ones(8),
            source_rank=rank, storage_host=rank_hosts[rank],
        )
        for index in range(4) for rank in rank_hosts
    )
    selected = select_zero1_publications(fragments, rank_hosts=rank_hosts, arm="m1")
    for host, items in enumerate(selected):
        assert len(items) == 4
        assert all(rank_hosts[item.source_rank] == host for item in items)
        assert {item.source_rank for item in items} == {
            rank for rank in rank_hosts if rank_hosts[rank] == host
        }


def test_pp_first_selector_ledger_counts_cross_host_republication():
    from Merak.core.checkpoint.object_recovery import ObjectAddress, ObjectFragment
    from Merak.core.checkpoint.zero1_service import select_zero1_publications

    layout = ExpertParallelLayout(4, 2, 2, 1, order=("pp", "dp", "ep", "tp"))
    rank_hosts = {rank: rank // 8 for rank in range(layout.world_size)}
    fragments = []

    def add_replicas(semantic_id, role, ranks, elements):
        address = ObjectAddress(
            semantic_id, role, (elements,), "torch.float32"
        )
        value = torch.arange(elements, dtype=torch.float32)
        fragments.extend(
            ObjectFragment.from_tensor(
                address,
                epoch=1,
                offsets=(0,),
                tensor=value,
                source_rank=rank,
                storage_host=rank_hosts[rank],
            )
            for rank in ranks
        )

    for pp in range(layout.pp):
        stage_ranks = tuple(
            layout.rank(pp=pp, dp=dp, ep=ep, tp=0)
            for dp in range(layout.dp) for ep in range(layout.ep)
        )
        add_replicas(f"shared-{pp}", "model", stage_ranks, 100)
        for ep in range(layout.ep):
            expert_ranks = tuple(
                layout.rank(pp=pp, dp=dp, ep=ep, tp=0)
                for dp in range(layout.dp)
            )
            add_replicas(f"expert-{pp}-{ep}", "model", expert_ranks, 40)

    placements = {
        arm: select_zero1_publications(
            fragments, rank_hosts=rank_hosts, arm=arm
        )
        for arm in ("adacheck_fixed", "host_global_fixed", "m1")
    }
    selected_model_bytes = {
        arm: sum(
            item.tensor.numel() * item.tensor.element_size()
            for host_fragments in hosts for item in host_fragments
        )
        for arm, hosts in placements.items()
    }
    remote_model_logical_bytes = {
        arm: sum(
            item.tensor.numel() * item.tensor.element_size()
            for host, host_fragments in enumerate(hosts)
            for item in host_fragments
            if rank_hosts[item.source_rank] != host
        )
        for arm, hosts in placements.items()
    }

    assert selected_model_bytes == {
        "adacheck_fixed": 2 * 4 * (4 * 100 + 2 * 2 * 40) * 4,
        "host_global_fixed": 2 * (4 * 100 + 4 * 2 * 40) * 4,
        "m1": 2 * (4 * 100 + 4 * 2 * 40) * 4,
    }
    assert remote_model_logical_bytes == {
        "adacheck_fixed": 4 * (4 * 100 + 2 * 2 * 40) * 4,
        "host_global_fixed": (4 * 100 + 4 * 2 * 40) * 4,
        "m1": (4 * 100 + 4 * 2 * 40) * 4,
    }
    assert selected_model_bytes["adacheck_fixed"] == 17_920
    assert selected_model_bytes["m1"] == 5_760
    assert remote_model_logical_bytes["adacheck_fixed"] == 8_960
    assert remote_model_logical_bytes["m1"] == 2_880


def test_pp_first_selector_ledger_exposes_logical_opportunity():
    from Merak.core.checkpoint.object_recovery import ObjectAddress, ObjectFragment
    from Merak.core.checkpoint.zero1_service import select_zero1_publications

    layout = ExpertParallelLayout(4, 2, 2, 1, order=("pp", "dp", "ep", "tp"))
    rank_hosts = {rank: rank // 8 for rank in range(layout.world_size)}
    fragments = []
    for pp in range(layout.pp):
        holders = tuple(
            layout.rank(pp=pp, dp=dp, ep=ep, tp=0)
            for dp in range(layout.dp) for ep in range(layout.ep)
        )
        address = ObjectAddress(f"shared-{pp}", "model", (100,), "torch.float32")
        fragments.extend(
            ObjectFragment.from_tensor(
                address, epoch=1, offsets=(0,),
                tensor=torch.ones(100), source_rank=rank,
                storage_host=rank_hosts[rank],
            )
            for rank in holders
        )

    planned = {
        arm: select_zero1_publications(fragments, rank_hosts=rank_hosts, arm=arm)
        for arm in ("adacheck_fixed", "host_global_fixed", "m1")
    }
    cross_host_logical_bytes = {
        arm: sum(
            item.tensor.numel() * item.tensor.element_size()
            for host, items in enumerate(hosts)
            for item in items
            if rank_hosts[item.source_rank] != host
        )
        for arm, hosts in planned.items()
    }
    assert cross_host_logical_bytes == {
        "adacheck_fixed": 4 * 4 * 100 * 4,
        "host_global_fixed": 4 * 100 * 4,
        "m1": 4 * 100 * 4,
    }
    assert cross_host_logical_bytes["m1"] < cross_host_logical_bytes["adacheck_fixed"]


def _role_aware_zero1_16rank_worker(rank, init_method, result_path, backend="gloo"):
    import os
    import random
    from types import SimpleNamespace

    import numpy as np
    from Merak.core import mpu
    from Merak.core.checkpoint.merak_incremental import (
        capture_runtime_state,
        restore_runtime_state,
    )
    from Merak.core.checkpoint.zero1_recovery_controller import zero1_target_layout
    from Merak.core.moe.zero1_optimizer import RoleAwareZeroOptimizerStage1
    from Merak.core.moe.expert_parallel import create_shared_parameter_group

    os.environ.update(
        DP="4",
        PP="4",
        TP="1",
        SP="1",
        MERAK_MOE_EP="2",
        MERAK_MOE_REPLICA_DP="2",
        MERAK_MESH_ORDER="pp,dp,sp,tp",
    )
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        backend,
        init_method=init_method,
        rank=rank,
        world_size=16,
        timeout=timedelta(seconds=180),
    )
    try:
        import Merak.core.mpu.initialize as initialize
        def cpu_mesh(device_type, shape, *, mesh_dim_names):
            return init_device_mesh(device_type, shape, mesh_dim_names=mesh_dim_names)

        if backend == "gloo":
            initialize.init_device_mesh = cpu_mesh
        grid = mpu.set_topo_grid_communication()
        topology = grid.topology()
        initialize.set_model_parallel_world_size(1)
        initialize.set_model_parallel_rank(0)
        layout = ExpertParallelLayout(
            pp=4, ep=2, dp=2, tp=1, order=("pp", "dp", "ep", "tp")
        )
        groups = create_expert_parallel_groups(layout, backend="gloo")
        shared_group = create_shared_parameter_group(layout, backend="gloo")
        control_group = dist.new_group(ranks=list(range(16)), backend="gloo")
        module = torch.nn.Module()
        module._merak_zero1_control_group = control_group
        from Merak.core.pipeline.module import PipelineModule

        pipeline = PipelineModule.__new__(PipelineModule)
        pipeline._merak_zero1_control_group = control_group
        pipeline.args = SimpleNamespace(get_communication_grid=lambda: grid, local_rank=rank)
        pipeline._initialize_parallel_properties()
        coordinates = layout.coordinates(rank)
        if hasattr(topology, "get_coord"):
            production_coordinates = topology.get_coord(rank)
            actual_coordinates = {
                "pp": production_coordinates.pipe,
                "dp": production_coordinates.data,
                "ep": production_coordinates.ep,
                "tp": production_coordinates.model,
            }
        else:
            pp_rank = grid.pp_mesh.get_local_rank()
            replica_dp_rank = grid.dp_mesh.get_local_rank()
            actual_coordinates = {
                "pp": pp_rank,
                "dp": replica_dp_rank,
                "ep": rank % layout.ep,
                "tp": grid.tp_mesh.get_local_rank(),
            }
        if actual_coordinates != coordinates:
            raise AssertionError("production MoE topology coordinates differ from PP-first layout")
        initialize.set_model_parallel_world_size(1)
        initialize.set_model_parallel_rank(0)
        if hasattr(topology, "get_axis_names") and topology.get_axis_names() != [
            "pipe", "data", "ep", "model"
        ]:
            raise AssertionError("production MoE topology axes are not PP-first")
        expected_grid_model = tuple(
            sorted(topology.get_axis_list("data", coordinates["dp"]))
        )
        if tuple(sorted(_group_ranks(grid.get_model_parallel_group()))) != expected_grid_model:
            raise AssertionError("DeepSpeed grid model group differs from its fixed data axis")
        if tuple(sorted(_group_ranks(mpu.get_model_parallel_group()))) != (rank,):
            raise AssertionError("MPU model group differs from the TP slice group")
        expected_pipeline = tuple(
            layout.group_ranks(
                "pp", dp=coordinates["dp"], ep=coordinates["ep"], tp=0
            )
        )
        if tuple(grid.stage_to_global(stage) for stage in range(4)) != expected_pipeline:
            raise AssertionError("production MoE pipeline ranks differ from EP layout")
        expected_shared = tuple(
            sorted(
                layout.rank(pp=coordinates["pp"], ep=ep, dp=dp, tp=0)
                for ep in range(layout.ep)
                for dp in range(layout.dp)
            )
        )
        expected_replica_dp = tuple(
            layout.group_ranks(
                "dp", pp=coordinates["pp"], ep=coordinates["ep"], tp=0
            )
        )
        if tuple(sorted(_group_ranks(mpu.get_data_parallel_group()))) != expected_replica_dp:
            raise AssertionError("production MoE data group differs from replica-DP axis")
        expected_composite_dp = tuple(
            sorted(
                layout.rank(pp=coordinates["pp"], ep=ep, dp=dp, tp=0)
                for ep in range(layout.ep)
                for dp in range(layout.dp)
            )
        )
        if expected_shared != expected_composite_dp:
            raise AssertionError("logical shared optimizer group differs from composite DP peers")
        if tuple(sorted(_group_ranks(shared_group))) != expected_composite_dp:
            raise AssertionError("MoE shared/ZeRO group differs from composite DP contract")
        if tuple(sorted(_group_ranks(grid.get_slice_parallel_group()))) != (
            rank,
        ):
            raise AssertionError("production MoE TP group differs from EP layout")
        expected_ep = tuple(
            layout.group_ranks(
                "ep", pp=coordinates["pp"], dp=coordinates["dp"], tp=0
            )
        )
        if tuple(sorted(_group_ranks(groups.ep_group))) != expected_ep:
            raise AssertionError("production MoE dispatch group differs from EP layout")

        replica_sampler = MegatronPretrainingRandomSampler(
            total_samples=256,
            consumed_samples=0,
            micro_batch_size=1,
            data_parallel_rank=coordinates["dp"],
            data_parallel_size=layout.dp,
        )
        replica_sample = tuple(next(iter(replica_sampler)))
        cursor_sampler = MegatronPretrainingRandomSampler(
            total_samples=256,
            consumed_samples=0,
            micro_batch_size=1,
            data_parallel_rank=coordinates["dp"],
            data_parallel_size=layout.dp,
        )
        cursor_iterator = iter(cursor_sampler)
        consumed_samples = [tuple(next(cursor_iterator)) for _ in range(3)]
        saved_cursor = cursor_sampler.consumed_samples
        random.seed(1000 + rank)
        np.random.seed(2000 + rank)
        torch.manual_seed(3000 + rank)
        expected_rng = (random.random(), float(np.random.random()), torch.rand(3))
        random.seed(1000 + rank)
        np.random.seed(2000 + rank)
        torch.manual_seed(3000 + rank)
        runtime_state = capture_runtime_state(2, cursor_sampler)
        oracle_sampler = MegatronPretrainingRandomSampler(
            total_samples=256,
            consumed_samples=saved_cursor,
            micro_batch_size=1,
            data_parallel_rank=coordinates["dp"],
            data_parallel_size=layout.dp,
        )
        expected_next_sample = tuple(next(iter(oracle_sampler)))
        restored_sampler = MegatronPretrainingRandomSampler(
            total_samples=256,
            consumed_samples=0,
            micro_batch_size=1,
            data_parallel_rank=coordinates["dp"],
            data_parallel_size=layout.dp,
        )
        restore_runtime_state(runtime_state, restored_sampler)
        restored_next_sample = tuple(next(iter(restored_sampler)))
        if restored_next_sample != expected_next_sample:
            raise AssertionError("replica-DP sampler cursor did not restore its next sample")
        restored_rng = (random.random(), float(np.random.random()), torch.rand(3))
        if restored_rng[:2] != expected_rng[:2] or not torch.equal(
            restored_rng[2], expected_rng[2]
        ):
            raise AssertionError("rank-local RNG state did not restore exactly")

        device = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
        shared = torch.nn.Parameter(torch.arange(16, dtype=torch.float32, device=device))
        expert = torch.nn.Parameter(
            torch.arange(16, dtype=torch.float32, device=device) + coordinates["ep"]
        )
        expert._merak_expert_parallel = True
        expert._merak_expert_id = coordinates["ep"]
        module = torch.nn.Module()
        module.register_parameter("shared", shared)
        module.register_parameter("expert", expert)
        base_optimizer = torch.optim.AdamW(
            [shared, expert], lr=1e-3, foreach=False
        )
        optimizer = RoleAwareZeroOptimizerStage1(
            SimpleNamespace(ada_checkpoint=False, zero_reshard=""),
            base_optimizer,
            model=module,
            shared_process_group=shared_group,
            expert_process_group=groups.dp_group,
            global_process_group=dist.group.WORLD,
            static_loss_scale=1.0,
            clip_grad=1.0,
            max_elements_per_comm=4096,
            communication_data_type=torch.float32,
        )
        expected_shared = tuple(
            sorted(
                layout.rank(pp=coordinates["pp"], ep=ep, dp=dp, tp=0)
                for dp in range(2)
                for ep in range(2)
            )
        )
        expected_expert = tuple(
            sorted(
                layout.group_ranks(
                    "dp", pp=coordinates["pp"], ep=coordinates["ep"], tp=0
                )
            )
        )
        coordinates = layout.coordinates(rank)
        module = torch.nn.Module()
        module.register_parameter("weight", torch.nn.Parameter(torch.ones(4)))
        module.weight.partition_dim = 0
        module.weight.model_parallel = True
        module._merak_ep_layout = layout
        with pytest.raises(ValueError, match="TP1 UCP"):
            zero1_target_layout(
                module, optimizer, rank=rank,
                tensor_parallel_rank=coordinates["tp"],
                tensor_parallel_world_size=layout.tp,
            )

        assert tuple(sorted(_group_ranks(shared_group))) == expected_shared
        assert tuple(sorted(_group_ranks(groups.dp_group))) == expected_expert
        assert optimizer.group_roles == ["shared", "expert"]
        assert optimizer.partition_counts == [4, 2]
        assert optimizer.dp_process_groups == [shared_group, groups.dp_group]
        assert optimizer.global_process_group is dist.group.WORLD
        assert optimizer.sub_partition_sizes == [4, 8]
        assert [
            sum(chunk.numel() for chunk in role)
            for role in optimizer.local_sub_partitions_of_fp32_groups
        ] == [4, 8]

        stage_lr = optimizer.optimizer.param_groups[0]["lr"]
        shared.grad = torch.ones_like(shared)
        expert.grad = torch.ones_like(expert)
        optimizer.reduce_gradients()
        if rank == 15:
            optimizer._reduced_gradients[0][0][0] = float("inf")
        before = (shared.detach().clone(), expert.detach().clone())
        assert optimizer.step() is True
        assert optimizer.overflow is True
        torch.testing.assert_close(shared, before[0])
        torch.testing.assert_close(expert, before[1])
        assert optimizer.optimizer.param_groups[0]["lr"] == stage_lr

        shared.grad = torch.full_like(shared, 1.0 + rank)
        expert.grad = torch.full_like(expert, 1.0 + coordinates["dp"])
        optimizer.reduce_gradients()
        assert optimizer.step() is False
        assert optimizer.overflow is False
        assert optimizer.global_step == 2

        record = {
            "rank": rank,
            "pipe_group": _group_ranks(grid.get_pipe_parallel_group()),
            "ep_pp_group": _group_ranks(groups.pp_group),
            "shared_group": tuple(sorted(_group_ranks(shared_group))),
            "expert_group": tuple(sorted(_group_ranks(groups.dp_group))),
            "dispatch_group": tuple(sorted(_group_ranks(groups.ep_group))),
            "partition_counts": tuple(optimizer.partition_counts),
            "shard_sizes": tuple(optimizer.sub_partition_sizes),
            "global_step": optimizer.global_step,
            "coordinates": coordinates,
            "production_coordinates": actual_coordinates,
            "production_pipe_group": tuple(sorted(_group_ranks(grid.get_pipe_parallel_group()))),
            "grid_model_group": tuple(sorted(_group_ranks(grid.get_model_parallel_group()))),
            "mpu_model_group": tuple(sorted(_group_ranks(mpu.get_model_parallel_group()))),
            "sampler_data_rank": replica_sampler.data_parallel_rank,
            "sampler_data_size": replica_sampler.data_parallel_size,
            "replica_sample": replica_sample,
            "consumed_samples": tuple(consumed_samples),
            "sampler_cursor": saved_cursor,
            "uninterrupted_next_sample": expected_next_sample,
            "restored_next_sample": restored_next_sample,
            "restored_rng_equal": True,
            "production_data_group": tuple(
                sorted(_group_ranks(mpu.get_data_parallel_group()))
            ),
            "expected_replica_dp_group": expected_replica_dp,
            "expected_shared_group": expected_shared,
        }
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, record)
        if rank == 0:
            with open(result_path, "w", encoding="utf-8") as output:
                output.write(repr(gathered))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_16rank_pp_first_role_aware_zero1_groups(tmp_path, backend):
    if backend == "nccl" and (
        not torch.cuda.is_available()
        or not dist.is_nccl_available()
        or torch.cuda.device_count() < 16
    ):
        pytest.skip("sixteen CUDA devices are required for the NCCL gate")
    result_path = tmp_path / "role-aware-zero1-groups.txt"
    mp.spawn(
        _role_aware_zero1_16rank_worker,
        args=(f"file://{tmp_path / f'role-aware-zero1-{backend}-rendezvous'}", str(result_path), backend),
        nprocs=16,
        join=True,
    )
    records = ast.literal_eval(result_path.read_text(encoding="utf-8"))
    assert len(records) == 16
    assert [record["rank"] for record in records] == list(range(16))
    for record in records:
        assert record["production_coordinates"] == record["coordinates"]
        assert record["production_pipe_group"] == record["pipe_group"]
        assert len(record["grid_model_group"]) == 8
        assert record["mpu_model_group"] == (record["rank"],)
        assert record["sampler_data_rank"] == record["coordinates"]["dp"]
        assert record["sampler_data_size"] == 2
        assert record["pipe_group"] == record["ep_pp_group"]
        assert len(record["shared_group"]) == 4
        assert len(record["expert_group"]) == 2
        assert len(record["dispatch_group"]) == 2
        assert record["production_data_group"] == record["expected_replica_dp_group"]
        assert record["partition_counts"] == (4, 2)
        assert record["shard_sizes"] == (4, 8)
        assert record["global_step"] == 2
        assert record["sampler_cursor"] == 6
        assert record["uninterrupted_next_sample"] == record["restored_next_sample"]
    for pp in range(4):
        for dp in range(2):
            ep_peers = [
                record
                for record in records
                if record["coordinates"]["pp"] == pp
                and record["coordinates"]["dp"] == dp
            ]
            assert len({record["replica_sample"] for record in ep_peers}) == 1
            assert len({record["consumed_samples"] for record in ep_peers}) == 1
            assert len({record["uninterrupted_next_sample"] for record in ep_peers}) == 1
            assert len({record["restored_next_sample"] for record in ep_peers}) == 1
        replica_peers = [
            record
            for record in records
            if record["coordinates"]["pp"] == pp
            and record["coordinates"]["ep"] == 0
        ]
        assert len({record["replica_sample"] for record in replica_peers}) == 2


def _stage_local_zero1_ucp_worker(rank, init_method, result_path):
    import os
    from dataclasses import replace
    from types import SimpleNamespace

    from torch import nn

    from Merak.core.checkpoint.zero1_recovery_controller import (
        apply_live_zero1_state,
        capture_zero1_rank,
        zero1_rank_metadata,
    )
    from Merak.core.checkpoint.object_recovery import (
        ObjectCatalog,
        ObjectRuntimeIdentity,
        TargetLayout,
        build_object_recovery_plan,
        restore_object_layout,
    )
    from Merak.core.checkpoint.zero1_recovery_controller import zero1_target_layout
    from Merak.core.moe.zero1_optimizer import RoleAwareZeroOptimizerStage1
    from Merak.core.checkpoint.zero1_adapter import _model_object_id

    os.environ.update(
        DP="2", PP="4", TP="1", SP="1", MERAK_MESH_ORDER="dp,pp,sp,tp"
    )
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=8,
        timeout=timedelta(seconds=180),
    )
    try:
        layout = ExpertParallelLayout(
            pp=4, ep=2, dp=1, tp=1, order=("dp", "ep", "pp", "tp")
        )
        groups = create_expert_parallel_groups(layout, backend="gloo")
        shared_group = create_shared_parameter_group(layout, backend="gloo")
        coordinates = layout.coordinates(rank)

        class Stage(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = nn.Parameter(
                    torch.tensor([1.0, -2.0, 3.0])
                )
                if coordinates["pp"] == 1:
                    self.expert = nn.Parameter(
                        torch.tensor([0.5, -1.5]) + coordinates["ep"]
                    )
                    self.expert._merak_expert_parallel = True
                    self.expert._merak_expert_id = coordinates["ep"]

        def make_runtime():
            torch.manual_seed(41)
            module = Stage()
            module._merak_zero1_shared_process_group = shared_group
            module._merak_zero1_expert_process_group = groups.dp_group
            module._merak_ep_layout = layout
            module._merak_role_aware_zero1 = True
            base = torch.optim.AdamW(module.parameters(), lr=0.01, foreach=False)
            optimizer = RoleAwareZeroOptimizerStage1(
                SimpleNamespace(ada_checkpoint=False, zero_reshard=""),
                base,
                model=module,
                shared_process_group=shared_group,
                expert_process_group=groups.dp_group,
                static_loss_scale=1.0,
                clip_grad=1.0,
                max_elements_per_comm=64,
                communication_data_type=torch.float32,
            )
            scheduler = torch.optim.lr_scheduler.StepLR(base, step_size=1)
            return module, optimizer, scheduler

        module, optimizer, scheduler = make_runtime()
        expected_roles = ["shared", "expert"] if coordinates["pp"] == 1 else ["shared"]
        assert optimizer.group_roles == expected_roles
        assert optimizer.partition_counts == ([2, 1] if "expert" in expected_roles else [2])

        def update(target, wrapped, lr_scheduler):
            for parameter in target.parameters():
                parameter.grad = torch.ones_like(parameter) * wrapped.loss_scale
            wrapped.reduce_gradients()
            assert wrapped.step() is False
            lr_scheduler.step()

        update(module, optimizer, scheduler)
        source = {name: value.detach().clone() for name, value in module.named_parameters()}
        source_model_ids = {
            _model_object_id(name, module, rank, parameter)
            for name, parameter in module.named_parameters()
        }
        fragments, opaque = capture_zero1_rank(
            module, optimizer, scheduler, rank=rank, host=rank // 4,
            epoch=1, sampler=None, storage_hosts=(0, 1),
        )
        metadata = zero1_rank_metadata(
            module, optimizer, rank=rank, host=rank // 4, layout=layout
        )
        assert metadata["optimizer_roles"] == tuple(expected_roles)
        if "expert" in expected_roles:
            assert any("@expert=" in name for name in source_model_ids)
        optimizer_ids = {
            item.address.semantic_id
            for item in fragments
            if item.address.role.startswith("optimizer.")
        }
        assert optimizer_ids and all(f"@rank={rank}" in name for name in optimizer_ids)
        metadata_rows = [None] * 8
        dist.all_gather_object(metadata_rows, metadata)
        assert {tuple(row["optimizer_roles"]) for row in metadata_rows} == {
            ("shared",), ("shared", "expert")
        }

        gathered_fragments = [None] * 8
        gathered_opaque = [None] * 8
        dist.all_gather_object(gathered_fragments, fragments)
        dist.all_gather_object(gathered_opaque, opaque)
        global_fragments = tuple(
            item for row in gathered_fragments for item in row
        )
        global_opaque = tuple(item for row in gathered_opaque for item in row)
        target_layout = zero1_target_layout(module, optimizer, rank=rank)
        owned = global_fragments
        roots = tuple(
            replace(item, storage_hosts=(0, 1))
            for item in global_opaque
        )
        host_catalogs = {
            host: ObjectCatalog(
                1,
                tuple(replace(item, storage_host=host) for item in owned),
                roots,
            )
            for host in (0, 1)
        }
        for catalog in host_catalogs.values():
            catalog.validate()
        layout_rows = [None] * 8
        dist.all_gather_object(
            layout_rows,
            zero1_target_layout(module, optimizer, rank=rank),
        )
        recovered = {}
        for target_rank, rank_layout in enumerate(layout_rows):
            surviving_host = 1 - target_rank // 4
            catalog = host_catalogs[surviving_host]
            runtime = ObjectRuntimeIdentity(
                detector_catalog_hash="d" * 64,
                semantic_plan_hash="s" * 64,
                encoder_name="stage_local_zero1_ucp_test",
                source_layout_hash="pp4-ep2",
                target_layout_hash=rank_layout.layout_hash,
                epoch=1,
                failure_sets=((1 - surviving_host,),),
            )
            rank_state = restore_object_layout(
                catalog,
                build_object_recovery_plan(
                    catalog, rank_layout, arm="m1_object", runtime=runtime
                ),
                rank_layout,
                failed_hosts=(1 - surviving_host,),
            )
            recovered[target_rank] = rank_state[target_rank]

        fresh, fresh_optimizer, fresh_scheduler = make_runtime()
        assert apply_live_zero1_state(
            recovered[rank], fresh, fresh_optimizer, fresh_scheduler,
            sampler=None, expected_iteration=1,
            runtime_restore=lambda state: state["iteration"],
        ) == 1
        for name, value in fresh.named_parameters():
            torch.testing.assert_close(value, source[name])
        update(module, optimizer, scheduler)
        update(fresh, fresh_optimizer, fresh_scheduler)
        for name, value in fresh.named_parameters():
            torch.testing.assert_close(value, dict(module.named_parameters())[name])

        record = {
            "rank": rank,
            "roles": tuple(optimizer.group_roles),
            "step": fresh_optimizer.global_step,
            "shared_group": _group_ranks(shared_group),
            "expert_group": _group_ranks(groups.dp_group),
        }
        gathered = [None] * 8
        dist.all_gather_object(gathered, record)
        if rank == 0:
            with open(result_path, "w", encoding="utf-8") as output:
                output.write(repr(gathered))
    finally:
        dist.destroy_process_group()


def test_pp4_stage_local_zero1_roles_and_ucp_restore(tmp_path):
    result_path = tmp_path / "stage-local-zero1-ucp.txt"
    mp.spawn(
        _stage_local_zero1_ucp_worker,
        args=(f"file://{tmp_path / 'stage-local-zero1-rendezvous'}", str(result_path)),
        nprocs=8,
        join=True,
    )
    records = ast.literal_eval(result_path.read_text(encoding="utf-8"))
    assert len(records) == 8
    assert {record["roles"] for record in records} == {
        ("shared",), ("shared", "expert")
    }
    assert all(record["step"] == 2 for record in records)


def _cuda_pp4_ep2_group_worker(rank, init_method, result_path):
    import os

    from Merak.core import mpu

    torch.cuda.set_device(rank)
    os.environ.update(
        DP="2",
        PP="4",
        TP="1",
        SP="1",
        LOCAL_RANK=str(rank),
        MERAK_MESH_ORDER="dp,pp,sp,tp",
    )
    dist.init_process_group(
        "nccl",
        init_method=init_method,
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    try:
        grid = mpu.set_topo_grid_communication()
        layout = ExpertParallelLayout(
            pp=4, ep=2, dp=1, tp=1, order=("dp", "ep", "pp", "tp")
        )
        groups = create_expert_parallel_groups(layout)
        shared_group = create_shared_parameter_group(layout)
        coordinates = layout.coordinates(rank)
        record = {
            "rank": rank,
            "coordinates": coordinates,
            "pipe_group": _group_ranks(grid.get_pipe_parallel_group()),
            "layout_pipe_group": layout.group_ranks(
                "pp", dp=0, ep=coordinates["ep"], tp=0
            ),
            "shared_group": _group_ranks(shared_group),
            "expert_group": _group_ranks(groups.dp_group),
            "dispatch_group": _group_ranks(groups.ep_group),
        }
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, record)
        if rank == 0:
            with open(result_path, "w", encoding="utf-8") as output:
                output.write(repr(gathered))
    finally:
        dist.destroy_process_group()


class _TinyDistributedQwenDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        ids = torch.tensor([1, 4, 7, 2]) if index % 2 == 0 else torch.tensor([3, 5, 8, 6])
        return {"input_ids": ids, "labels": ids.clone()}


def _pp4_ep2_trainer_worker(
    rank, init_method, output_dir, backend, pp_first=False, train_only=False
):
    import copy
    import os
    from datetime import timedelta

    from Merak import MerakArguments, MerakTrainer
    from Merak.core import mpu
    from Merak.core.moe.expert_parallel import (
        Qwen2MoeExpertParallelBlock,
        create_shared_parameter_group,
    )
    from Merak.core.moe.zero1_optimizer import RoleAwareZeroOptimizerStage1
    from examples.checkpoint_grouping.workloads import Workload
    from transformers.trainer_pt_utils import get_parameter_names
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

    torch.set_num_threads(1)
    use_cpu = backend == "gloo"
    local_rank = rank % 8
    if not use_cpu:
        torch.cuda.set_device(local_rank)
    os.environ.update(
        DP="2",
        PP="4",
        TP="1",
        SP="1",
        LOCAL_RANK=str(rank),
        MERAK_MESH_ORDER="pp,dp,sp,tp" if pp_first else "dp,pp,sp,tp",
        MERAK_MOE_EP="2" if pp_first else "1",
        MERAK_MOE_REPLICA_DP="1" if pp_first else "1",
    )
    dist.init_process_group(
        backend,
        init_method=init_method,
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=180),
    )
    try:
        grid = mpu.set_topo_grid_communication()
        layout = ExpertParallelLayout(
            pp=4,
            ep=2,
            dp=1,
            tp=1,
            order=("pp", "dp", "ep", "tp")
            if pp_first
            else ("dp", "ep", "pp", "tp"),
        )
        groups = create_expert_parallel_groups(layout)
        shared_group = create_shared_parameter_group(layout)
        coordinates = layout.coordinates(rank)
        assert mpu.get_pipe_parallel_rank() == coordinates["pp"]
        assert _group_ranks(grid.get_pipe_parallel_group()) == layout.group_ranks(
            "pp", dp=coordinates["dp"], ep=coordinates["ep"], tp=0
        )
        assert _group_ranks(mpu.get_data_parallel_group()) == layout.group_ranks(
            "dp", pp=coordinates["pp"], ep=coordinates["ep"], tp=0
        )
        assert _group_ranks(shared_group) == tuple(sorted(
            layout.rank(pp=coordinates["pp"], ep=ep, dp=dp, tp=0)
            for ep in range(layout.ep) for dp in range(layout.dp)
        ))
        assert _group_ranks(groups.dp_group) == layout.group_ranks(
            "dp", pp=coordinates["pp"], ep=coordinates["ep"], tp=0
        )

        torch.manual_seed(73)
        config = Qwen2MoeConfig(
            vocab_size=31,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=16,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=16,
            decoder_sparse_step=1,
            use_cache=False,
            output_router_logits=False,
            tie_word_embeddings=False,
        )
        config._attn_implementation = "eager"
        workload = Workload(
            "qwen2_moe_14b", "tiny Qwen2-MoE", config, Qwen2MoeForCausalLM
        )
        model = workload.build_model(empty=False, expert_parallel_groups=groups)
        model.to(torch.device("cpu") if use_cpu else torch.device("cuda", local_rank))
        model._merak_zero1_shared_process_group = shared_group
        model._merak_zero1_expert_process_group = groups.dp_group
        model._merak_ep_layout = layout
        model._merak_role_aware_zero1 = True
        model._merak_sampler_data_parallel_rank = coordinates["dp"]
        model._merak_sampler_data_parallel_size = layout.dp
        model._merak_replica_data_parallel_group = groups.dp_group
        model._merak_pipeline_input_to_shard = {"input_ids": 0}
        model._merak_init_weights = model._init_weights

        args = MerakArguments(
            output_dir=output_dir,
            use_cpu=use_cpu,
            local_rank=rank,
            max_steps=1,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            lr_scheduler_type="constant",
            logging_strategy="no",
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            dataloader_num_workers=0,
            seed=73,
            data_seed=73,
            seq_length=4,
            num_layers=1,
            hidden_size=16,
            num_heads=4,
            zero_stage=1,
            zero_reduce_bucket_size=256,
            zero_allgather_bucket_size=256,
            save=False,
            ada_checkpoint=False,
            activation_checkpointing=True,
            checkpoint_num_layers=1,
            no_tie_modules=True,
            parallel_vocab=False,
            split_method="layer_split",
            partition_method="custom",
            custom_partition=["0,1,2,3,4"],
            trace_method="fx",
            half_precision_backend=None,
            input_names=["input_ids"],
        )
        if use_cpu:
            args.__dict__["_setup_devices"] = torch.device("cpu")
        args.get_communication_grid = lambda: grid
        trainer = MerakTrainer(
            model, args, train_dataset=_TinyDistributedQwenDataset()
        )
        assert trainer.model._merak_pipeline_model
        assert len(trainer.model._layer_specs) == 4
        assert trainer.model._merak_ep_layout == layout
        assert trainer.model._merak_zero1_shared_process_group is shared_group
        tracked_parameter = next(
            parameter
            for parameter in trainer.model.parameters()
            if parameter.requires_grad
        )
        before = tracked_parameter.detach().clone()
        try:
            trainer.train()
        except BaseException:
            device_report = {
                "args_device": str(args.device),
                "pipeline_device": str(trainer.model.device),
                "parameter_devices": sorted(
                    {str(parameter.device) for parameter in trainer.model.parameters()}
                ),
                "layer_spec_devices": [
                    sorted({str(parameter.device) for parameter in layer.parameters()})
                    if isinstance(layer, torch.nn.Module) else []
                    for layer in trainer.model._layer_specs
                ],
            }
            with open(os.path.join(output_dir, f"worker-rank-{rank}.err"), "w", encoding="utf-8") as error_file:
                error_file.write(repr(device_report) + "\n" + traceback.format_exc())
            raise

        optimizer = trainer.engine.optimizer
        assert isinstance(optimizer, RoleAwareZeroOptimizerStage1)
        module = trainer.engine.module
        has_experts = any(
            bool(getattr(parameter, "_merak_expert_parallel", False))
            for parameter in module.parameters()
        )
        decay_names = set(get_parameter_names(module, [torch.nn.LayerNorm]))
        decay_names = {name for name in decay_names if "bias" not in name}
        expected_groups = []
        named_parameters = tuple(module.named_parameters())
        for source_index, decay_group in enumerate((True, False)):
            for role in ("shared", "expert"):
                present = any(
                    parameter.requires_grad
                    and ((name in decay_names) == decay_group)
                    and bool(getattr(parameter, "_merak_expert_parallel", False))
                    == (role == "expert")
                    for name, parameter in named_parameters
                )
                if present:
                    expected_groups.append((source_index, role))
        actual_groups = [
            (group["_merak_source_group"], group["_merak_zero1_role"])
            for group in optimizer.optimizer.param_groups
        ]
        assert actual_groups == expected_groups
        expected_roles = [role for _, role in expected_groups]
        assert optimizer.group_roles == expected_roles
        assert optimizer.partition_counts == [
            2 if role == "shared" else 1 for role in expected_roles
        ]
        assert optimizer.step_count == trainer.tuning_params.global_steps == 1
        assert torch.isfinite(trainer.engine.agg_train_loss).all()
        after = tracked_parameter.detach()
        assert not torch.equal(before, after)

        expert_blocks = [
            block
            for block in module.modules()
            if isinstance(block, Qwen2MoeExpertParallelBlock)
        ]
        if has_experts:
            assert len(expert_blocks) == 1
            assert expert_blocks[0].last_routing is not None
            assert expert_blocks[0].last_routing.sent_tokens == 8
            assert expert_blocks[0].last_routing.received_tokens > 0
        else:
            assert not expert_blocks

        if train_only:
            reports = [None] * dist.get_world_size()
            dist.all_gather_object(
                reports,
                {
                    "rank": rank,
                    "step": optimizer.step_count,
                    "loss": float(trainer.engine.agg_train_loss),
                    "roles": tuple(optimizer.group_roles),
                },
            )
            if rank == 0:
                with open(
                    os.path.join(output_dir, "pp-first-train-gate.txt"),
                    "w",
                    encoding="utf-8",
                ) as output:
                    output.write(repr(reports))
            return

        from dataclasses import replace

        from Merak.core.checkpoint.merak_incremental import restore_runtime_state
        from Merak.core.checkpoint.object_recovery import (
            ObjectCatalog,
            ObjectRuntimeIdentity,
            build_object_recovery_plan,
            restore_object_layout,
        )
        from Merak.core.checkpoint.zero1_recovery_controller import (
            apply_live_zero1_state,
            capture_zero1_rank,
            zero1_target_layout,
        )
        fragments, opaque = capture_zero1_rank(
            module,
            optimizer,
            trainer.engine.lr_scheduler,
            rank=rank,
            host=rank // 4,
            epoch=1,
            sampler=None,
            storage_hosts=(0, 1),
        )
        gathered_fragments = [None] * dist.get_world_size()
        gathered_opaque = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_fragments, fragments)
        dist.all_gather_object(gathered_opaque, opaque)
        source_fragments = tuple(item for row in gathered_fragments for item in row)
        source_opaque = tuple(item for row in gathered_opaque for item in row)
        mirrored_fragments = tuple(
            replace(item, storage_host=1 - item.storage_host)
            for item in source_fragments
        )
        catalog = ObjectCatalog(
            1,
            source_fragments + mirrored_fragments,
            source_opaque,
        )
        catalog.validate()

        local_layout = zero1_target_layout(module, optimizer, rank=rank)
        all_layouts = [None] * dist.get_world_size()
        dist.all_gather_object(all_layouts, local_layout)
        recovered_layout = all_layouts[rank]
        failed_hosts = (rank // 4,)
        runtime = ObjectRuntimeIdentity(
            detector_catalog_hash="d" * 64,
            semantic_plan_hash="s" * 64,
            encoder_name="single_node_qwen_trainer_ucp_test",
            source_layout_hash="single-node-pp4-ep2",
            target_layout_hash=recovered_layout.layout_hash,
            epoch=1,
            failure_sets=(failed_hosts,),
        )
        try:
            plan = build_object_recovery_plan(
                catalog, recovered_layout, arm="m1_object", runtime=runtime
            )
        except ValueError as error:
            catalog_keys = {item.address.key for item in catalog.fragments}
            missing = [
                {
                    "semantic_id": request.address.semantic_id,
                    "role": request.address.role,
                    "shape": request.address.global_shape,
                    "dtype": request.address.dtype,
                    "candidate_hosts": sorted({
                        item.storage_host for item in catalog.fragments
                        if item.address.key == request.address.key
                    }),
                }
                for request in recovered_layout.slices
                if request.address.key not in catalog_keys
            ]
            with open(
                os.path.join(output_dir, f"ucp-plan-rank-{rank}.err"),
                "w",
                encoding="utf-8",
            ) as diagnostic:
                diagnostic.write(repr({"error": str(error), "missing": missing}))
            raise
        recovered = restore_object_layout(
            catalog, plan, recovered_layout, failed_hosts=failed_hosts
        )[rank]

        torch.manual_seed(101 + rank)
        fresh_model = workload.build_model(empty=False, expert_parallel_groups=groups)
        fresh_model.to(torch.device("cpu") if use_cpu else torch.device("cuda", rank))
        fresh_model._merak_zero1_shared_process_group = shared_group
        fresh_model._merak_zero1_expert_process_group = groups.dp_group
        fresh_model._merak_ep_layout = layout
        fresh_model._merak_role_aware_zero1 = True
        fresh_model._merak_sampler_data_parallel_rank = coordinates["dp"]
        fresh_model._merak_sampler_data_parallel_size = layout.dp
        fresh_model._merak_replica_data_parallel_group = groups.dp_group
        fresh_model._merak_pipeline_input_to_shard = {"input_ids": 0}
        fresh_model._merak_init_weights = fresh_model._init_weights
        fresh_args = MerakArguments(
            output_dir=os.path.join(output_dir, "fresh"),
            use_cpu=False,
            local_rank=rank,
            max_steps=1,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            lr_scheduler_type="constant",
            logging_strategy="no",
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            dataloader_num_workers=0,
            seed=73,
            data_seed=73,
            seq_length=4,
            num_layers=1,
            hidden_size=16,
            num_heads=4,
            zero_stage=1,
            zero_reduce_bucket_size=256,
            zero_allgather_bucket_size=256,
            save=False,
            ada_checkpoint=False,
            activation_checkpointing=True,
            checkpoint_num_layers=1,
            no_tie_modules=True,
            parallel_vocab=False,
            split_method="layer_split",
            partition_method="custom",
            custom_partition=["0,1,2,3,4"],
            trace_method="fx",
            half_precision_backend=None,
            input_names=["input_ids"],
        )
        fresh_args.get_communication_grid = lambda: grid
        fresh_trainer = MerakTrainer(
            fresh_model, fresh_args, train_dataset=_TinyDistributedQwenDataset()
        )
        fresh_trainer.init_engine()
        fresh_optimizer = fresh_trainer.engine.optimizer
        assert isinstance(fresh_optimizer, RoleAwareZeroOptimizerStage1)
        assert apply_live_zero1_state(
            recovered,
            fresh_trainer.engine.module,
            fresh_optimizer,
            fresh_trainer.engine.lr_scheduler,
            sampler=None,
            expected_iteration=1,
            runtime_restore=lambda state: restore_runtime_state(state, None),
        ) == 1
        fresh_module = fresh_trainer.engine.module
        for name, value in module.named_parameters():
            torch.testing.assert_close(
                value,
                dict(fresh_module.named_parameters())[name],
                rtol=0,
                atol=0,
            )

        checkpoint_state = copy.deepcopy(recovered)

        def run_next_step():
            input_ids = torch.tensor([[1, 4, 7, 2]], device=fresh_trainer.engine.device)
            labels = input_ids.clone()
            iterator = iter([
                {"input_ids": input_ids, "labels": labels},
                {"input_ids": input_ids.clone(), "labels": labels.clone()},
            ])
            fresh_trainer.engine.batch_fn = fresh_trainer.prepare_data
            fresh_trainer.engine.train_batch(iterator)
            assert fresh_optimizer.global_step == 2

        run_next_step()
        reference_model = {
            name: value.detach().clone()
            for name, value in fresh_module.named_parameters()
        }
        reference_chunks = [
            (
                chunk.detach().clone(),
                {key: value.detach().clone() for key, value in fresh_optimizer.optimizer.state[chunk].items()},
            )
            for group in fresh_optimizer.local_sub_partitions_of_fp32_groups
            for chunk in group
        ]
        reference_scheduler = copy.deepcopy(fresh_trainer.engine.lr_scheduler.state_dict())
        reference_groups = copy.deepcopy([
            {key: value for key, value in group.items() if key != "params"}
            for group in fresh_optimizer.optimizer.param_groups
        ])
        reference_scaler = copy.deepcopy(vars(fresh_optimizer.loss_scaler))
        reference_step = fresh_optimizer.global_step
        assert reference_step == 2

        assert apply_live_zero1_state(
            checkpoint_state,
            fresh_module,
            fresh_optimizer,
            fresh_trainer.engine.lr_scheduler,
            sampler=None,
            expected_iteration=1,
            runtime_restore=lambda state: restore_runtime_state(state, None),
        ) == 1
        run_next_step()
        assert fresh_optimizer.global_step == reference_step
        for name, value in fresh_module.named_parameters():
            torch.testing.assert_close(value, reference_model[name], rtol=1e-6, atol=1e-7)
        restored_chunks = [
            (chunk, fresh_optimizer.optimizer.state[chunk])
            for group in fresh_optimizer.local_sub_partitions_of_fp32_groups
            for chunk in group
        ]
        assert len(restored_chunks) == len(reference_chunks)
        for (chunk, state), (expected_chunk, expected_state) in zip(
            restored_chunks, reference_chunks
        ):
            torch.testing.assert_close(chunk, expected_chunk, rtol=1e-6, atol=1e-7)
            for key, value in state.items():
                torch.testing.assert_close(value, expected_state[key], rtol=1e-6, atol=1e-7)
        assert fresh_trainer.engine.lr_scheduler.state_dict() == reference_scheduler
        assert fresh_optimizer.global_step == 2
        assert vars(fresh_optimizer.loss_scaler) == reference_scaler
        assert fresh_optimizer.overflow is False
        assert [
            {key: value for key, value in group.items() if key != "params"}
            for group in fresh_optimizer.optimizer.param_groups
        ] == reference_groups

        record = {
            "rank": rank,
            "step": fresh_optimizer.global_step,
            "roles": tuple(fresh_optimizer.group_roles),
            "catalog_objects": len(catalog.fragments),
            "recovered_iteration": 1,
            "continued_step": 2,
        }
        records = [None] * dist.get_world_size()
        dist.all_gather_object(records, record)
        if rank == 0:
            with open(
                os.path.join(output_dir, "trainer-ucp-restore.txt"),
                "w",
                encoding="utf-8",
            ) as output:
                output.write(repr(records))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not dist.is_nccl_available(),
    reason="CUDA/NCCL is unavailable",
)
def test_cuda_qwen_tiny_pp4_ep2_dp1_role_aware_zero1_ucp_restore(tmp_path):
    if torch.cuda.device_count() < 8:
        pytest.skip("eight CUDA devices are required")
    mp.spawn(
        _pp4_ep2_trainer_worker,
        args=(f"file://{tmp_path / 'cuda-pp4-ep2-trainer-rendezvous'}", str(tmp_path), "nccl"),
        nprocs=8,
        join=True,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not dist.is_nccl_available(),
    reason="CUDA/NCCL is unavailable",
)
def test_cuda_qwen_tiny_pp_first_role_aware_zero1_step(tmp_path):
    if torch.cuda.device_count() < 8:
        pytest.skip("eight CUDA devices are required")
    mp.spawn(
        _pp4_ep2_trainer_worker,
        args=(
            f"file://{tmp_path / 'cuda-pp-first-trainer-rendezvous'}",
            str(tmp_path),
            "nccl",
            True,
            True,
        ),
        nprocs=8,
        join=True,
    )
    records = ast.literal_eval(
        (tmp_path / "pp-first-train-gate.txt").read_text(encoding="utf-8")
    )
    assert len(records) == 8
    assert [record["rank"] for record in records] == list(range(8))
    assert all(record["step"] == 1 for record in records)
    assert all(torch.isfinite(torch.tensor(record["loss"])) for record in records)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not dist.is_nccl_available(),
    reason="CUDA/NCCL is unavailable",
)
def test_cuda_qwen_tiny_pp_first_role_aware_zero1_ucp_restore(tmp_path):
    if torch.cuda.device_count() < 8:
        pytest.skip("eight CUDA devices are required")
    mp.spawn(
        _pp4_ep2_trainer_worker,
        args=(
            f"file://{tmp_path / 'cuda-pp-first-ucp-rendezvous'}",
            str(tmp_path),
            "nccl",
            True,
            False,
        ),
        nprocs=8,
        join=True,
    )
    records = ast.literal_eval(
        (tmp_path / "trainer-ucp-restore.txt").read_text(encoding="utf-8")
    )
    assert len(records) == 8
    assert [record["rank"] for record in records] == list(range(8))
    assert all(record["recovered_iteration"] == 1 for record in records)
    assert all(record["continued_step"] == 2 for record in records)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not dist.is_nccl_available(),
    reason="CUDA/NCCL is unavailable",
)
def test_cuda_device_mesh_pp4_ep2_group_contract(tmp_path):
    if torch.cuda.device_count() < 8:
        pytest.skip("eight CUDA devices are required")
    result_path = tmp_path / "cuda-pp4-ep2-groups.txt"
    mp.spawn(
        _cuda_pp4_ep2_group_worker,
        args=(f"file://{tmp_path / 'cuda-pp4-ep2-rendezvous'}", str(result_path)),
        nprocs=8,
        join=True,
    )
    records = ast.literal_eval(result_path.read_text(encoding="utf-8"))
    assert len(records) == 8
    for record in records:
        assert record["pipe_group"] == record["layout_pipe_group"]
        assert len(record["shared_group"]) == 2
        assert len(record["expert_group"]) == 1
        assert len(record["dispatch_group"]) == 2
