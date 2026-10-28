import threading

import pytest
import torch

from Merak.core.amp.amp_optimizer import HalfPrecisionOptimizer
from Merak.core.checkpoint.catalog import tensor_checksum
from Merak.core.checkpoint.incremental_checkpoint import RecoveryBinding
from Merak.core.checkpoint.object_optimizer_update import externalize_local_optimizer_update
from Merak.core.checkpoint.object_semantic_plan import (
    GradientShard,
    RankBinding,
    build_active_gradient_plan,
    confirm_gradient_checksums,
)
from Merak.core.checkpoint.object_semantic_runtime import (
    SEMANTIC_ARMS,
    SemanticFixedRuntimeController,
)


class _Scheduler:
    def __init__(self):
        self.epoch = 0

    def state_dict(self):
        return {"last_epoch": self.epoch, "base_lrs": [0.01]}


def _half(parameter):
    half = object.__new__(HalfPrecisionOptimizer)
    half.optimizer = torch.optim.AdamW(
        [parameter], lr=0.01, foreach=False, fused=False
    )
    half.global_step = 0
    half.fp16_groups = [[parameter]]
    half.fp32_groups = [[parameter]]
    half._pre_optimizer_step_callback = None
    half._capture_raw_gradients = False
    return half


def _plan(arm):
    bindings = (RankBinding(0, 0, 0, 0, 0), RankBinding(1, 0, 0, 1, 1))
    shards = tuple(
        GradientShard(
            rank, "weight", "detector", (2,), (0,), (2,), "torch.bfloat16"
        )
        for rank in range(2)
    )
    return build_active_gradient_plan(
        shards,
        bindings,
        arm=arm,
        detector_catalog_hash="a" * 64,
        semantic_plan_hash="b" * 64,
        fixed_pairs=((0, 1),),
    )


@pytest.mark.parametrize("arm", SEMANTIC_ARMS)
def test_fifo_worker_keeps_poll_free_of_commit_and_uses_epoch_locators(arm, tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.bfloat16))
    parameter.grad = torch.tensor([0.25, -0.5], dtype=torch.bfloat16)
    optimizer = _half(parameter)
    scheduler = _Scheduler()
    plan = _plan(arm)
    epoch = [1]
    trace, commits, profiles, worker_threads = [], [], [], []
    entered, release = threading.Event(), threading.Event()
    main_thread = threading.get_ident()

    def checksum(plan, tensors, *, epoch, group):
        trace.append(("checksum", epoch))
        digest = tensor_checksum(tensors[plan.objects[0].object_id])
        return confirm_gradient_checksums(
            plan,
            {(rank, plan.objects[0].object_id): digest for rank in (0, 1)},
            epoch=epoch,
        )

    def externalize(plan, encoding, rank, record):
        trace.append(("externalize", encoding.epoch))
        return externalize_local_optimizer_update(plan, encoding, rank, record)

    def commit(_clients, *, encoding, update, locator_path, **_):
        worker_threads.append(threading.get_ident())
        trace.append(("commit", encoding.epoch))
        commits.append((encoding, update, locator_path))
        if encoding.epoch == 1:
            entered.set()
            assert release.wait(timeout=5)
        return {"epoch": encoding.epoch}

    controller = SemanticFixedRuntimeController(
        optimizer,
        arm=arm,
        plan=plan,
        binding=RecoveryBinding(0, 2, 0, 0, 0),
        parameter_ids=(("weight",),),
        base_id="sha256:" + "c" * 64,
        epoch_provider=lambda: epoch[0],
        runtime_capture=lambda value: {"iteration": value, "rng": torch.tensor([value])},
        scheduler=scheduler,
        control_group=object(),
        clients={0: object(), 1: object()},
        run_id="cpu-test",
        locator_root=tmp_path / "locators",
        group_validator=lambda _: (0, 2),
        checksum=checksum,
        externalize=externalize,
        commit=commit,
        profile=lambda name, start_ns, end_ns, **fields: profiles.append(
            (name, start_ns, end_ns, fields)
        ),
    )
    controller.start(0)
    optimizer._pre_optimizer_step_callback(1, optimizer.optimizer, 0.125)
    parameter.grad.fill_(9.0)
    assert entered.wait(timeout=5)

    assert controller.poll(1) == ()
    assert worker_threads == [thread for thread in worker_threads if thread != main_thread]

    epoch[0], scheduler.epoch = 2, 2
    parameter.grad = torch.tensor([-0.75, 0.125], dtype=torch.bfloat16)
    optimizer._pre_optimizer_step_callback(2, optimizer.optimizer, 0.25)
    release.set()
    summary = controller.finalize(2)

    assert trace == [
        ("checksum", 1),
        ("externalize", 1),
        ("commit", 1),
        ("checksum", 2),
        ("externalize", 2),
        ("commit", 2),
    ]
    assert [path.name for _, _, path in commits] == [
        "epoch-00000001.json",
        "epoch-00000002.json",
    ]
    first, second = commits
    assert torch.equal(
        first[1].payloads[0].tensor,
        torch.tensor([0.25, -0.5], dtype=torch.bfloat16),
    )
    assert first[1].records[0][1]["gradient_multiplier"] == 0.125
    assert second[1].records[0][1]["gradient_multiplier"] == 0.25
    assert first[1].records[0][1]["runtime_state"]["iteration"] == 1
    assert second[1].records[0][1]["runtime_state"]["iteration"] == 2
    assert first[1].records[0][1]["metadata"] == {
        "scheduler_state_before_step": {"base_lrs": [0.01], "last_epoch": 0},
        "trainer_global_step": 1,
    }
    assert second[1].records[0][1]["metadata"]["scheduler_state_before_step"]["last_epoch"] == 2
    assert summary["boundary_lag"] == [1]
    assert [item["epoch"] for item in summary["commit_metrics"]] == [1, 2]
    assert summary["final_drain_ms"] >= 0
    assert {item[0] for item in profiles} == {
        "checkpoint_object_semantic_snapshot_preallocation",
        "checkpoint_object_semantic_checksum_confirmation",
        "checkpoint_object_semantic_externalize",
        "checkpoint_object_semantic_distributed_commit",
        "checkpoint_object_semantic_final_drain",
    }
    preallocation = next(
        fields
        for name, _, _, fields in profiles
        if name == "checkpoint_object_semantic_snapshot_preallocation"
    )
    assert preallocation["max_inflight"] == 2
    assert preallocation["allocated_bytes"] == 8
    assert all(start <= end for _, start, end, _ in profiles)
    for item in summary["commit_metrics"]:
        assert all(
            item[name] >= 0
            for name in (
                "checksum_confirmation_ms",
                "externalize_ms",
                "distributed_commit_ms",
            )
        )
    assert summary["recovery_valid"] and summary["cadence_valid"]
    assert summary["all_epochs_committed"] and summary["strict_1c1s_valid"]
    assert summary["valid"]


def test_runtime_uses_raw_bfloat16_gradient_instead_of_master_roundtrip(tmp_path):
    raw = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.bfloat16))
    master = torch.nn.Parameter(raw.detach().float())
    raw.grad = torch.tensor([0.25, -0.5], dtype=torch.bfloat16)
    master.grad = torch.tensor([9.0, 9.0])
    optimizer = _half(master)
    optimizer.fp16_groups = [[raw]]
    published = []

    controller = SemanticFixedRuntimeController(
        optimizer,
        arm="m1",
        plan=_plan("m1"),
        binding=RecoveryBinding(0, 2, 0, 0, 0),
        parameter_ids=(("weight",),),
        base_id="sha256:" + "c" * 64,
        epoch_provider=lambda: 1,
        runtime_capture=lambda value: {"iteration": value},
        scheduler=_Scheduler(),
        control_group=object(),
        clients={0: object(), 1: object()},
        run_id="cpu-test",
        locator_root=tmp_path,
        group_validator=lambda _: (0, 2),
        checksum=lambda plan, tensors, *, epoch, group: confirm_gradient_checksums(
            plan,
            {
                (rank, plan.objects[0].object_id): tensor_checksum(
                    tensors[plan.objects[0].object_id]
                )
                for rank in (0, 1)
            },
            epoch=epoch,
        ),
        externalize=lambda plan, encoding, rank, record: (
            published.append(record)
            or externalize_local_optimizer_update(plan, encoding, rank, record)
        ),
        commit=lambda *_args, encoding, **_kwargs: {"epoch": encoding.epoch},
    )
    controller.start(0)
    optimizer._pre_optimizer_step_callback(1, optimizer.optimizer, 0.125)
    controller.finalize(1)

    assert published[0]["gradient_dtype"] == "torch.bfloat16"
    assert torch.equal(published[0]["gradients"][0][0], raw.grad)


def test_runtime_rejects_unsupported_optimizer_and_zero_grad_semantics(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    common = {
        "arm": "m1",
        "plan": _plan("m1"),
        "binding": RecoveryBinding(0, 2, 0, 0, 0),
        "parameter_ids": (("weight",),),
        "base_id": "sha256:" + "c" * 64,
        "epoch_provider": lambda: 1,
        "runtime_capture": lambda _: {},
        "scheduler": _Scheduler(),
        "control_group": object(),
        "clients": {0: object(), 1: object()},
        "run_id": "cpu-test",
        "locator_root": tmp_path,
    }
    with pytest.raises(TypeError, match="HalfPrecisionOptimizer"):
        SemanticFixedRuntimeController(
            torch.optim.AdamW([parameter], foreach=False), **common
        )

    optimizer = _half(parameter)
    optimizer.optimizer = torch.optim.SGD([parameter], lr=0.01)
    with pytest.raises(TypeError, match="torch.optim.AdamW"):
        SemanticFixedRuntimeController(optimizer, **common)

    optimizer = _half(parameter)
    controller = SemanticFixedRuntimeController(
        optimizer, group_validator=lambda _: (0, 2), **common
    )
    controller.start(0)
    with pytest.raises(ValueError, match="set_grads_to_none=True"):
        controller.zero_grad(set_grads_to_none=False)
    controller.finalize(0)
