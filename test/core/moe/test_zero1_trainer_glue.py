import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from Merak.core.amp import MixedPrecisionConfig
from Merak.core.merak_engine import PipelineEngine
from Merak.core.moe.expert_parallel import ExpertParallelLayout, create_expert_parallel_groups
from Merak.core.moe.zero1_optimizer import RoleAwareZeroOptimizerStage1
from Merak.core.pipeline.module import PipelineModule
from Merak.core.tensor_parallel.utils import reset_module_tensor
from Merak.merak_trainer import (
    _validate_moe_adacheck_grouping,
    _zero1_tensor_coordinates,
)
from examples.checkpoint_grouping.run_workload import (
    expert_layout_for_experiment,
    expert_parallel_group_report,
    validate_expert_parallel_options,
)


class _TinyStage(nn.Module):
    def __init__(self, rank, expert_id, expert_dp_rank):
        super().__init__()
        self.shared = nn.Parameter(torch.full((4,), float(rank + 1)))
        self.expert = nn.Parameter(
            torch.full((4,), float(10 * expert_id + expert_dp_rank))
        )
        self.expert._merak_expert_parallel = True
        self.expert._merak_expert_id = expert_id


def _worker(rank, init_method):
    dist.init_process_group("gloo", rank=rank, world_size=4, init_method=init_method)
    try:
        groups = create_expert_parallel_groups(
            ExpertParallelLayout(1, 2, 2, 1, order=("dp", "ep", "pp", "tp"))
        )
        report = expert_parallel_group_report(groups.layout, groups, dist.group.WORLD)
        assert len(report["shared_zero1_ranks"]) == 4
        assert len(report["expert_zero1_ranks"]) == 2
        assert len(report["expert_dispatch_ranks"]) == 2
        args = SimpleNamespace(
            lora_config=None,
            zero_stage=1,
            zero_reshard="",
            fp16=False,
            bf16=False,
            max_grad_norm=1.0,
            zero_allgather_bucket_size=64,
            zero_reduce_bucket_size=64,
            prescale_gradients=False,
            gradient_predivide_factor=1.0,
            half_precision_backend=None,
        )
        stage = _TinyStage(rank, groups.ep_rank, groups.dp_rank)
        stage._merak_role_aware_zero1 = True
        stage._merak_zero1_shared_process_group = dist.group.WORLD
        stage._merak_zero1_expert_process_group = groups.dp_group
        PipelineModule._broadcast_model(stage)
        assert torch.equal(stage.shared, torch.ones_like(stage.shared))
        assert torch.equal(
            stage.expert,
            torch.full_like(stage.expert, float(10 * groups.ep_rank)),
        )
        expert_values = [None] * dist.get_world_size()
        dist.all_gather_object(expert_values, stage.expert[0].item())
        assert expert_values == [0.0, 10.0, 0.0, 10.0]

        engine = PipelineEngine.__new__(PipelineEngine)
        engine.args = args
        engine.module = stage
        engine.device = torch.device("cpu")
        engine.optimizer = lambda module: torch.optim.AdamW(
            module.parameters(), lr=0.01, foreach=False
        )
        engine.lr_scheduler = lambda optimizer: torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1
        )
        engine.amp_engine = MixedPrecisionConfig(args)

        engine._configure_optimizer()

        assert isinstance(engine.optimizer, RoleAwareZeroOptimizerStage1)
        assert engine.optimizer.group_roles == ["shared", "expert"]
        assert engine.lr_scheduler.optimizer is engine.optimizer.optimizer
    finally:
        dist.destroy_process_group()


def test_ep_launch_options_fail_closed():
    valid = SimpleNamespace(
        expert_parallel_size=2,
        dp=4,
        role_aware_zero1=True,
        model="qwen2_moe_14b",
        tp=1,
        mesh_order="dp-pp",
        tiny_qwen_layers=1,
        tiny_deepseek_layers=1,
        dataset_size=4,
    )
    validate_expert_parallel_options(valid)
    validate_expert_parallel_options(
        SimpleNamespace(**{**vars(valid), "model": "deepseek_v2_lite"})
    )
    with pytest.raises(ValueError, match="Qwen2-MoE or DeepSeek"):
        validate_expert_parallel_options(
            SimpleNamespace(**{**vars(valid), "model": "gpt_6b"})
        )
    with pytest.raises(ValueError, match="requires --role-aware-zero1"):
        validate_expert_parallel_options(
            SimpleNamespace(**{**vars(valid), "role_aware_zero1": False})
        )
    with pytest.raises(ValueError, match="TP1"):
        validate_expert_parallel_options(
            SimpleNamespace(**{**vars(valid), "tp": 2})
        )


@pytest.mark.parametrize(
    ("mesh_order", "expected_order", "rank"),
    [
        ("pp-dp", ("pp", "dp", "ep", "tp"), 6),
        ("dp-pp", ("dp", "ep", "pp", "tp"), 9),
    ],
)
def test_experiment_layout_matches_rank_mesh_order(mesh_order, expected_order, rank):
    experiment = SimpleNamespace(
        pp=4, expert_parallel_size=2, dp=4, tp=1, mesh_order=mesh_order
    )
    layout = expert_layout_for_experiment(experiment)
    assert layout.order == expected_order
    assert layout.coordinates(rank) == (
        {"pp": 1, "dp": 1, "ep": 0, "tp": 0}
        if mesh_order == "pp-dp"
        else {"dp": 1, "ep": 0, "pp": 1, "tp": 0}
    )


def test_moe_grouping_requires_zero1_ep_replica_dp_and_tp1():
    layout = ExpertParallelLayout(1, 2, 2, 1)
    _validate_moe_adacheck_grouping(
        enabled=True, zero1_enabled=True, layout=layout, zero_stage=1
    )
    with pytest.raises(ValueError, match="UCP ZeRO-1"):
        _validate_moe_adacheck_grouping(
            enabled=True, zero1_enabled=False, layout=layout, zero_stage=1
        )
    with pytest.raises(ValueError, match="EP>1, replica DP>1, and TP1"):
        _validate_moe_adacheck_grouping(
            enabled=True, zero1_enabled=True,
            layout=ExpertParallelLayout(1, 2, 4, 2), zero_stage=1,
        )
    with pytest.raises(ValueError, match="replica DP>1"):
        _validate_moe_adacheck_grouping(
            enabled=True, zero1_enabled=True,
            layout=ExpertParallelLayout(1, 2, 1, 1), zero_stage=1,
        )


def test_meta_materialization_keeps_expert_owner_marker():
    module = nn.Module()
    module.expert = nn.Linear(2, 2, device="meta")
    old = module.expert.weight
    old._merak_expert_parallel = True
    old._merak_expert_id = 7
    reset_module_tensor(module, "expert.weight", torch.device("cpu"), torch.zeros(2, 2))
    assert module.expert.weight is not old
    assert module.expert.weight._merak_expert_parallel is True
    assert module.expert.weight._merak_expert_id == 7


@pytest.mark.parametrize("tp_size", [1, 2])
def test_zero1_tensor_coordinates_use_pipeline_grid_slice_group(tp_size):
    layout = ExpertParallelLayout(
        pp=1, ep=2, dp=1, tp=tp_size, order=("dp", "ep", "pp", "tp")
    )

    class Grid:
        def get_slice_parallel_group(self):
            return object()

    class Module:
        _merak_ep_layout = layout
        _grid = Grid()

    group_ranks = layout.group_ranks("tp", pp=0, ep=0, dp=0)
    for rank in range(layout.world_size):
        coordinates = layout.coordinates(rank)
        expected = layout.group_ranks(
            "tp", pp=coordinates["pp"], ep=coordinates["ep"], dp=coordinates["dp"]
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(dist, "get_process_group_ranks", lambda group: expected)
            assert _zero1_tensor_coordinates(Module(), rank) == (
                coordinates["tp"],
                tp_size,
            )


def test_zero1_tensor_coordinates_reject_mismatched_grid_group(monkeypatch):
    layout = ExpertParallelLayout(1, 2, 1, 1, order=("dp", "ep", "pp", "tp"))

    class Grid:
        def get_slice_parallel_group(self):
            return object()

    class Module:
        _merak_ep_layout = layout
        _grid = Grid()

    monkeypatch.setattr(dist, "get_process_group_ranks", lambda group: [0, 1])
    with pytest.raises(ValueError, match="tensor group differs"):
        _zero1_tensor_coordinates(Module(), rank=0)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_pipeline_engine_selects_role_aware_zero1_only_when_opted_in():
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            _worker,
            args=(f"file://{directory}/init",),
            nprocs=4,
            join=True,
        )
