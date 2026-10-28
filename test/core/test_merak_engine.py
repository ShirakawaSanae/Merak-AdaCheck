from types import SimpleNamespace

import torch

from Merak.core import merak_engine


def _set_mpu_groups(monkeypatch):
    monkeypatch.setattr(merak_engine.mpu, "get_pipe_parallel_group", lambda: "pipe")
    monkeypatch.setattr(merak_engine.mpu, "get_data_parallel_group", lambda: None)
    monkeypatch.setattr(merak_engine.mpu, "get_data_parallel_world_size", lambda: 1)


def _engine(last_stage, data_parallel=False):
    engine = merak_engine.CommunicationEngine.__new__(
        merak_engine.CommunicationEngine
    )
    engine.pipeline = SimpleNamespace(
        is_last_stage=lambda: last_stage,
        micro_batches=1,
    )
    engine.is_data_parallel = data_parallel
    engine.device = torch.device("cpu")
    engine.global_rank = 12 if last_stage else 0
    engine.num_stages = 4
    engine.grid = SimpleNamespace(
        pp_group=[0, 4, 8, 12],
        get_pipe_parallel_group=lambda: "pipe",
        get_data_parallel_group=lambda: "data",
        get_data_parallel_world_size=lambda: 2,
        stage_to_global=lambda stage: stage * 4,
    )
    return engine


def test_aggregate_loss_uses_moe_replica_group_without_changing_dense_path(monkeypatch):
    calls = []
    _set_mpu_groups(monkeypatch)
    replica_group = object()
    monkeypatch.setattr(merak_engine.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(merak_engine.dist, "all_reduce", lambda tensor, group: calls.append(group))
    monkeypatch.setattr(merak_engine.dist, "broadcast", lambda tensor, src, group: None)

    engine = _engine(last_stage=True, data_parallel=True)
    engine.pipeline.module = SimpleNamespace(
        _merak_replica_data_parallel_group=replica_group
    )
    engine.data_parallel_group = replica_group
    engine.data_parallel_world_size = 2

    result = engine._aggregate_total_loss(torch.tensor(3.0))

    assert result.item() == 1.5
    assert calls == [replica_group]


def test_aggregate_loss_uses_world_group(monkeypatch):
    calls = []
    _set_mpu_groups(monkeypatch)
    monkeypatch.setattr(
        merak_engine.dist,
        "broadcast",
        lambda tensor, src, group: tensor.fill_(7.0),
    )

    def all_reduce(tensor, group):
        calls.append(group)
        tensor.fill_(7.0)

    monkeypatch.setattr(merak_engine.dist, "all_reduce", all_reduce)

    sender = _engine(last_stage=True)
    receiver = _engine(last_stage=False)
    sender.is_data_parallel = True
    receiver.is_data_parallel = True

    assert sender._aggregate_total_loss(torch.tensor(7.0)).item() == 7
    assert receiver._aggregate_total_loss(torch.tensor(99.0)).item() == 7
    assert calls == [None]


def test_aggregate_loss_averages_all_last_stage_contributors(monkeypatch):
    calls = []
    _set_mpu_groups(monkeypatch)
    monkeypatch.setattr(merak_engine.mpu, "get_data_parallel_world_size", lambda: 2)
    monkeypatch.setattr(merak_engine.dist, "broadcast", lambda tensor, src, group: None)

    def all_reduce(tensor, group):
        calls.append(group)
        tensor.mul_(2)

    monkeypatch.setattr(merak_engine.dist, "all_reduce", all_reduce)

    result = _engine(last_stage=True, data_parallel=True)._aggregate_total_loss(
        torch.tensor(5.0)
    )

    assert result.item() == 5
    assert calls == [None]


def test_loss_broadcast_trace_is_opt_in_and_reports_rank_mapping(monkeypatch, capsys):
    monkeypatch.setenv("CKPT_MOE_TRACE_LOSS_BROADCAST", "1")
    monkeypatch.setattr(merak_engine.dist, "get_rank", lambda: 12)
    monkeypatch.setattr(merak_engine.dist, "get_process_group_ranks", lambda group: [0, 4, 8, 12])
    monkeypatch.setattr(merak_engine.dist, "get_group_rank", lambda group, rank: 3)
    monkeypatch.setattr(merak_engine.dist, "broadcast", lambda tensor, src, group: None)
    monkeypatch.setattr(merak_engine.mpu, "get_pipe_parallel_group", lambda: "pipe")
    engine = _engine(last_stage=False)
    engine.global_rank = 8
    engine.stage_id = 2
    engine.grid.pp_group = [0, 4, 8, 12]

    engine._aggregate_total_loss(torch.tensor(0.0))

    output = capsys.readouterr().out
    assert "default_rank=12 engine_rank=8 stage=2 last=False" in output
    assert "src_global=12 src_group=3" in output
    assert "group_members=[0, 4, 8, 12]" in output
