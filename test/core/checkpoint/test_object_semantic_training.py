import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from Merak.core.amp.amp_optimizer import HalfPrecisionOptimizer
from Merak.core.checkpoint import object_semantic_training as training


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_detector(tmp_path):
    requirements = [
        {
            "rank": rank,
            "path": "/str:optimizer/str:fp16_groups/int:0/int:0",
            "object_id": f"object-{rank}",
            "alias_id": f"alias-{rank}",
        }
        for rank in range(16)
    ]
    catalog = {
        "value_id": "1" * 64,
        "role": "optimizer",
        "dtype": "torch.float16",
        "shape": [2],
        "logical_bytes": 4,
        "checksum": "2" * 64,
        "source_ranks": list(range(16)),
        "source_hosts": ["host0", "host1"],
        "requirements": requirements,
    }
    catalog_hash = _digest(
        {
            "full_logical_bytes": 64,
            "template_ranks": list(range(16)),
            "objects": [
                {
                    key: catalog[key]
                    for key in (
                        "value_id",
                        "role",
                        "dtype",
                        "shape",
                        "logical_bytes",
                        "checksum",
                        "source_ranks",
                        "requirements",
                    )
                }
            ],
        }
    )
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "object_catalog.jsonl").write_text(json.dumps(catalog) + "\n")
    summary = {
        "catalog_hash": catalog_hash,
        "detector_stability_steps": 2,
        "world_size": 16,
        "ranks_per_host": 8,
    }
    summary_path = frozen / "summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True))
    config = {
        "run_kind": "matched_detector_capture",
        "status_at_creation": "capture_and_freeze_complete",
        "model_argument": "cpu-model",
        "detector_comparison_scope": "world",
        "precision": "bf16",
        "parallelism": {
            "world_size": 16,
            "hosts": 2,
            "ranks_per_host": 8,
            "pipeline": 1,
            "tensor": 1,
            "data": 16,
        },
        "workload": {
            "rank_topology": [
                {"rank": rank, "data": rank, "pipeline": 0, "tensor": 0}
                for rank in range(16)
            ]
        },
        "frozen_catalog": {
            "summary_path": str(summary_path),
            "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "catalog_hash": catalog_hash,
            "detector_stability_steps": 2,
        },
    }
    path = tmp_path / "detector.json"
    path.write_text(json.dumps(config, sort_keys=True))
    return path


def _half(module):
    parameter = next(module.parameters())
    half = object.__new__(HalfPrecisionOptimizer)
    half.optimizer = torch.optim.AdamW(
        [parameter], lr=0.01, foreach=False, fused=False
    )
    half.dynamic_loss_scale = False
    half.cur_scale = 1.0
    half.cur_iter = 0
    half.global_step = 0
    half.args = SimpleNamespace(ada_checkpoint=False)
    half.fp16_groups = [[parameter]]
    half.fp32_groups = [[parameter]]
    half._pre_optimizer_step_callback = None
    half._capture_raw_gradients = False
    return half


def _worker(rank, rendezvous, config, root):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=16
    )
    training.mpu.get_data_parallel_rank = lambda: rank
    training.mpu.get_pipe_parallel_rank = lambda: 0
    training.mpu.get_model_parallel_rank = lambda: 0
    training.mpu.get_model_parallel_world_size = lambda: 1
    training.ReplicaServiceClient.from_endpoint = staticmethod(
        lambda _, timeout_seconds: object()
    )
    base_calls = []

    def commit_base(_clients, **kwargs):
        assert kwargs["max_parallel_uploads"] == 2
        state = kwargs["local_base"]
        assert state["kind"] == "merak_incremental_base"
        assert state["runtime_state"]["iteration"] == 0
        assert all(
            tensor.device.type == "cpu"
            for tensor in state["model"].values()
            if isinstance(tensor, torch.Tensor)
        )
        base_calls.append(kwargs["epoch"])
        return {"epoch": kwargs["epoch"]}

    training.commit_distributed_object_semantic_base = commit_base
    training.capture_runtime_state = lambda epoch, _sampler: {
        "iteration": epoch,
        "random_rng_state": (3, (epoch,), None),
        "np_rng_state": ("MT19937", [epoch], 0, 0, 0.0),
        "torch_rng_state": torch.tensor([epoch], dtype=torch.uint8),
        "cuda_rng_state": torch.tensor([epoch], dtype=torch.uint8),
        "rng_tracker_states": {"model": torch.tensor([epoch], dtype=torch.uint8)},
        "sampler": {},
    }
    try:
        output = []
        for arm in ("adacheck_fixed", "host_global_fixed", "m1"):
            os_environ = {
                "CKPT_OBJECT_SEMANTIC_ENABLE": "1",
                "CKPT_EXPERIMENT_ARM": arm,
                "CKPT_OBJECT_DETECTOR_CONFIG": config,
                "CKPT_OBJECT_RUN_ID": "cpu-builder",
                "CKPT_OBJECT_ROOT": root,
                "CKPT_REPLICA_ENDPOINTS": json.dumps(["fake-a", "fake-b"]),
                "CKPT_RANKS_PER_NODE": "8",
                "CKPT_OBJECT_MAX_INFLIGHT": "2",
                "CKPT_OBJECT_BASE_UPLOAD_PARALLELISM": "2",
            }
            for name, value in os_environ.items():
                __import__("os").environ[name] = value
            module = torch.nn.Linear(2, 1, bias=False).to(torch.bfloat16)
            controller = training.build_semantic_fixed_training_controller(
                module,
                _half(module),
                torch.optim.lr_scheduler.LambdaLR(
                    torch.optim.AdamW(module.parameters(), foreach=False), lambda _: 1.0
                ),
                SimpleNamespace(
                    bf16=True,
                    zero_stage=None,
                    dataloader_num_workers=0,
                ),
                None,
                lambda: 0,
            )
            assert controller.runtime is None
            controller.start(0)
            with pytest.raises(RuntimeError, match="already started"):
                controller.start(0)
            assert controller.runtime._locator_path(1).name == "epoch-00000001.json"
            assert controller.runtime._locator_path(2).name == "epoch-00000002.json"
            summary = controller.finalize(0)
            output.append(
                {
                    "arm": arm,
                    "plan_hash": controller.runtime.plan.plan_hash,
                    "pairs": [list(pair) for pair in controller.semantic_plan.pairs],
                    "base_id": controller.base_id,
                    "summary": summary["valid"],
                }
            )
        Path(root, f"rank-{rank}.json").write_text(json.dumps({"base_calls": base_calls, "output": output}))
    finally:
        dist.destroy_process_group()


def test_cpu_multiprocess_builder_builds_three_static_plans_and_one_base(tmp_path):
    config = _write_detector(tmp_path)
    root = tmp_path / "runtime"
    mp.spawn(
        _worker,
        args=(str(tmp_path / "rendezvous"), str(config), str(root)),
        nprocs=16,
    )
    reports = [json.loads((root / f"rank-{rank}.json").read_text()) for rank in range(16)]
    assert all(report["base_calls"] == [0, 0, 0] for report in reports)
    for index, arm in enumerate(("adacheck_fixed", "host_global_fixed", "m1")):
        values = [report["output"][index] for report in reports]
        assert all(value["arm"] == arm and value["summary"] for value in values)
        assert len({value["plan_hash"] for value in values}) == 1
        assert values[0]["pairs"] == [[rank, rank + 8] for rank in range(8)]
    assert len(list((root / "metrics").glob("rank_*.json"))) == 16


def test_builder_environment_fails_closed(monkeypatch, tmp_path):
    module = torch.nn.Linear(2, 1, bias=False).to(torch.bfloat16)
    args = SimpleNamespace(bf16=True, zero_stage=None, dataloader_num_workers=0)
    with pytest.raises(ValueError, match="CKPT_OBJECT_SEMANTIC_ENABLE"):
        training.build_semantic_fixed_training_controller(
            module, _half(module), object(), args, None, lambda: 0
        )
    monkeypatch.setenv("CKPT_OBJECT_SEMANTIC_ENABLE", "1")
    monkeypatch.setenv("CKPT_OBJECT_MAX_INFLIGHT", "2")
    monkeypatch.setenv("CKPT_EXPERIMENT_ARM", "invalid")
    with pytest.raises(ValueError, match="CKPT_EXPERIMENT_ARM"):
        training.build_semantic_fixed_training_controller(
            module, _half(module), object(), args, None, lambda: 0
        )
    monkeypatch.setenv("CKPT_EXPERIMENT_ARM", "m1")
    monkeypatch.setenv("CKPT_RANKS_PER_NODE", "4")
    with pytest.raises(ValueError, match="eight ranks"):
        training.build_semantic_fixed_training_controller(
            module, _half(module), object(), args, None, lambda: 0
        )
    monkeypatch.setenv("CKPT_RANKS_PER_NODE", "8")
    monkeypatch.setenv("CKPT_OBJECT_BASE_UPLOAD_PARALLELISM", "0")
    with pytest.raises(ValueError, match="BASE_UPLOAD_PARALLELISM"):
        training.build_semantic_fixed_training_controller(
            module, _half(module), object(), args, None, lambda: 0
        )
