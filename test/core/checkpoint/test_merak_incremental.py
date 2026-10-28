import copy
from contextlib import contextmanager
from types import SimpleNamespace
import threading
import time

import pytest
import torch

from Merak.core.amp.amp_optimizer import HalfPrecisionOptimizer
from Merak.core.checkpoint.incremental_checkpoint import RecoveryBinding
from Merak.core.checkpoint.global_commit import (
    GlobalCommitRepository,
    create_global_epoch,
)
from Merak.core.checkpoint.incremental_runtime import (
    IncrementalCheckpointSession,
    PeerIncrementalCheckpointSession,
)
from Merak.core.checkpoint.merak_incremental import (
    MerakIncrementalCheckpointController,
    half_precision_parameter_ids,
)
from Merak.core.checkpoint.recovery import ServiceRemoteCheckpointSession
from Merak.core.checkpoint.replica_service import ReplicaTCPServer, ServiceReplicaStore


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@contextmanager
def _service(*, materialize=False, sealed=False):
    store = ServiceReplicaStore(
        max_bytes=64 * 1024 * 1024,
        retained_epochs_per_source=8,
        materialize_incremental=materialize,
        sealed_incremental=sealed,
    )
    server = ReplicaTCPServer(("127.0.0.1", 0), store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _objects(initial=(1.0, -2.0)):
    module = torch.nn.Linear(2, 1, bias=False, device="cuda", dtype=torch.bfloat16)
    module.weight.data.copy_(torch.tensor([initial], device="cuda"))
    inner = torch.optim.AdamW(
        module.parameters(), lr=1e-2, foreach=False, fused=False
    )
    optimizer = HalfPrecisionOptimizer(
        inner,
        SimpleNamespace(ada_checkpoint=False),
        static_loss_scale=1.0,
        dynamic_loss_scale=False,
        clip_grad=0.0,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(inner, step_size=2, gamma=0.9)
    return module, optimizer, scheduler


def _session(endpoint, root, *, rank, target):
    return ServiceRemoteCheckpointSession(
        rank=rank,
        group_ranks=(0, 1),
        target_rank=target,
        service_endpoint=endpoint,
        manifest_root=root,
        max_local_store_bytes=1,
        collect_store_stats=False,
    )


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def _first_difference(left, right, path="root"):
    if isinstance(left, torch.Tensor):
        return None if _equal(left, right) else path
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            return f"{path}.keys"
        for key in left:
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            return f"{path}.length"
        for index, (first, second) in enumerate(zip(left, right)):
            difference = _first_difference(first, second, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    return None if left == right else path


def _close(left, right):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor):
            return False
        if left.is_floating_point():
            return torch.allclose(left.cpu(), right.cpu(), rtol=1e-6, atol=1e-7)
        return torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_close(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(_close(a, b) for a, b in zip(left, right))
        )
    return left == right


def _wait_for_commit(controller, epoch):
    deadline = time.monotonic() + 5
    while epoch not in controller.committed and time.monotonic() < deadline:
        controller.poll(epoch)
        time.sleep(0.001)
    assert epoch in controller.committed


def _step(optimizer, scheduler, value):
    optimizer.fp16_groups[0][0].grad = torch.tensor(
        [value], device="cuda", dtype=torch.bfloat16
    )
    optimizer.step()
    scheduler.step()


def test_full_log_restores_bf16_model_and_fp32_master_state(tmp_path):
    with _service() as endpoint:
        module, optimizer, scheduler = _objects()
        ids = half_precision_parameter_ids(module, optimizer)
        current = {"epoch": 0}
        runtime = {}
        session = IncrementalCheckpointSession(
            _session(endpoint, tmp_path / "source", rank=0, target=1),
            optimizer.optimizer,
            parameter_ids=ids,
            binding=RecoveryBinding(0, 2, 0, 0, 0),
            max_inflight=4,
        )
        controller = MerakIncrementalCheckpointController(
            module,
            optimizer,
            scheduler,
            mode="full_remote",
            session=session,
            binding=RecoveryBinding(0, 2, 0, 0, 0),
            peer_binding=RecoveryBinding(1, 2, 1, 0, 0),
            parameter_ids=ids,
            epoch_provider=lambda: current["epoch"],
            runtime_capture=lambda epoch: {"iteration": epoch, "rank": 0},
            runtime_restore=lambda state: runtime.update(state) or state["iteration"],
        )
        controller.start(0)
        for epoch in range(1, 4):
            current["epoch"] = epoch
            _step(optimizer, scheduler, [epoch / 10, -epoch / 20])
            _wait_for_commit(controller, epoch)
        assert controller.finalize(3)["valid"]
        expected_model = copy.deepcopy(module.state_dict())
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        expected_scheduler = copy.deepcopy(scheduler.state_dict())

        restored_module, restored_optimizer, restored_scheduler = _objects((0.0, 0.0))
        restored = MerakIncrementalCheckpointController(
            restored_module,
            restored_optimizer,
            restored_scheduler,
            mode="full_remote",
            session=IncrementalCheckpointSession(
                _session(endpoint, tmp_path / "restore", rank=0, target=1),
                restored_optimizer.optimizer,
                parameter_ids=ids,
                binding=RecoveryBinding(0, 2, 0, 0, 0),
            ),
            binding=RecoveryBinding(0, 2, 0, 0, 0),
            peer_binding=RecoveryBinding(1, 2, 1, 0, 0),
            parameter_ids=ids,
            epoch_provider=lambda: 3,
            runtime_capture=lambda epoch: {},
            runtime_restore=lambda state: runtime.update(state) or state["iteration"],
        )
        assert restored.restore(base_epoch=0, final_epoch=3) == 3
        assert _equal(restored_module.state_dict(), expected_model)
        assert _equal(restored_optimizer.state_dict(), expected_optimizer), _first_difference(
            restored_optimizer.state_dict(), expected_optimizer
        )
        assert restored_scheduler.state_dict() == expected_scheduler
        assert runtime == {"iteration": 3, "rank": 0}


def test_full_remote_materialized_state_restores_with_global_evidence(tmp_path):
    with _service(materialize=True, sealed=True) as host0, _service(
        materialize=True, sealed=True
    ) as host1:
        objects = [_objects(), _objects()]
        current = {"epoch": 0}
        controllers = []
        for rank, (module, optimizer, scheduler) in enumerate(objects):
            ids = half_precision_parameter_ids(module, optimizer)
            remote_endpoint = host1 if rank == 0 else host0
            controllers.append(
                MerakIncrementalCheckpointController(
                    module,
                    optimizer,
                    scheduler,
                    mode="full_remote",
                    session=IncrementalCheckpointSession(
                        _session(
                            remote_endpoint,
                            tmp_path / f"remote-full-{rank}",
                            rank=rank,
                            target=1 - rank,
                        ),
                        optimizer.optimizer,
                        parameter_ids=ids,
                        binding=RecoveryBinding(rank, 2, rank, 0, 0),
                        max_inflight=4,
                        run_id="full-remote-materialized",
                    ),
                    binding=RecoveryBinding(rank, 2, rank, 0, 0),
                    peer_binding=RecoveryBinding(1 - rank, 2, 1 - rank, 0, 0),
                    parameter_ids=ids,
                    epoch_provider=lambda: current["epoch"],
                    runtime_capture=lambda epoch, rank=rank: {
                        "iteration": epoch,
                        "rank": rank,
                    },
                    runtime_restore=lambda state: state["iteration"],
                    materialized=True,
                    run_id="full-remote-materialized",
                )
            )
        for controller in controllers:
            controller.start(0)
        for epoch in range(1, 4):
            current["epoch"] = epoch
            for _, optimizer, scheduler in objects:
                _step(optimizer, scheduler, [epoch / 10, -epoch / 20])
            for controller in controllers:
                _wait_for_commit(controller, epoch)
        for controller in controllers:
            assert controller.finalize(3)["valid"]

        global_root = tmp_path / "full-remote-global"
        control = create_global_epoch(
            run_id="full-remote-materialized",
            epoch=3,
            world_size=2,
            ranks=[
                controller._rank_commit_evidence(3)
                for controller in controllers
            ],
            publication_protocol="pin-before-publish-v1",
        )
        for controller in controllers:
            controller.session.pin_epoch(control)
        GlobalCommitRepository(global_root).publish(control)
        for controller in controllers:
            controller.session.seal_epoch(control)
            controller.session.wait_sealed(3)
        module, optimizer, scheduler = _objects((0.0, 0.0))
        ids = half_precision_parameter_ids(module, optimizer)
        restored_runtime = {}
        restored = MerakIncrementalCheckpointController(
            module,
            optimizer,
            scheduler,
            mode="full_remote",
            session=IncrementalCheckpointSession(
                _session(
                    host1,
                    tmp_path / "full-remote-restore",
                    rank=0,
                    target=1,
                ),
                optimizer.optimizer,
                parameter_ids=ids,
                binding=RecoveryBinding(0, 2, 0, 0, 0),
                run_id="full-remote-materialized",
            ),
            binding=RecoveryBinding(0, 2, 0, 0, 0),
            peer_binding=RecoveryBinding(1, 2, 1, 0, 0),
            parameter_ids=ids,
            epoch_provider=lambda: 3,
            runtime_capture=lambda epoch: {},
            runtime_restore=lambda state: restored_runtime.update(state)
            or state["iteration"],
            materialized=True,
            sealed=True,
            run_id="full-remote-materialized",
            global_commit_root=global_root,
        )

        assert restored.restore(base_epoch=0, final_epoch=3) == 3
        assert _equal(module.state_dict(), objects[0][0].state_dict())
        assert _close(optimizer.state_dict(), objects[0][1].state_dict())
        assert scheduler.state_dict() == objects[0][2].state_dict()
        assert restored_runtime == {"iteration": 3, "rank": 0}


@pytest.mark.parametrize("materialized", [False, True])
def test_peer_restore_uses_peer_full_chain_and_failed_rank_runtime(
    tmp_path, materialized
):
    with _service(materialize=materialized) as host0, _service(
        materialize=materialized
    ) as host1:
        objects = [_objects(), _objects()]
        assert _equal(objects[0][1].state_dict(), objects[1][1].state_dict())
        current = {"epoch": 0}
        controllers = []
        for rank, (module, optimizer, scheduler) in enumerate(objects):
            ids = half_precision_parameter_ids(module, optimizer)
            local_endpoint, remote_endpoint = (host0, host1) if rank == 0 else (host1, host0)
            session = PeerIncrementalCheckpointSession(
                _session(
                    local_endpoint,
                    tmp_path / f"local-{rank}",
                    rank=rank,
                    target=rank,
                ),
                _session(
                    remote_endpoint,
                    tmp_path / f"remote-{rank}",
                    rank=rank,
                    target=1 - rank,
                ),
                optimizer.optimizer,
                parameter_ids=ids,
                binding=RecoveryBinding(rank, 2, rank, 0, 0),
                peer_rank=1 - rank,
                max_inflight=4,
                run_id="integration-run",
            )
            controllers.append(
                MerakIncrementalCheckpointController(
                    module,
                    optimizer,
                    scheduler,
                    mode="peer_delta",
                    session=session,
                    binding=RecoveryBinding(rank, 2, rank, 0, 0),
                    peer_binding=RecoveryBinding(1 - rank, 2, 1 - rank, 0, 0),
                    parameter_ids=ids,
                    epoch_provider=lambda: current["epoch"],
                    runtime_capture=lambda epoch, rank=rank: {
                        "iteration": epoch,
                        "rank": rank,
                    },
                    runtime_restore=lambda state: state["iteration"],
                    materialized=materialized,
                    run_id="integration-run",
                )
            )
        for controller in controllers:
            controller.start(0)
        for epoch in range(1, 4):
            current["epoch"] = epoch
            for _, optimizer, scheduler in objects:
                _step(optimizer, scheduler, [epoch / 10, -epoch / 20])
            for controller in controllers:
                controller.poll(epoch)
        for controller in controllers:
            assert controller.finalize(3)["all_epochs_contiguous"]
        global_root = tmp_path / "global"
        global_evidence = [
            controller._rank_commit_evidence(3) for controller in controllers
        ]
        assert _equal(objects[0][1].state_dict(), objects[1][1].state_dict()), (
            _first_difference(objects[0][1].state_dict(), objects[1][1].state_dict()),
            objects[0][1].fp32_groups[0][0],
            objects[1][1].fp32_groups[0][0],
        )

        restored_runtime = {}
        module, optimizer, scheduler = _objects((0.0, 0.0))
        ids = half_precision_parameter_ids(module, optimizer)
        restored = MerakIncrementalCheckpointController(
            module,
            optimizer,
            scheduler,
            mode="peer_delta",
            session=PeerIncrementalCheckpointSession(
                _session(host0, tmp_path / "unused-local", rank=0, target=0),
                _session(host1, tmp_path / "failed-rank-delta", rank=0, target=1),
                optimizer.optimizer,
                parameter_ids=ids,
                binding=RecoveryBinding(0, 2, 0, 0, 0),
                peer_rank=1,
                run_id="integration-run",
            ),
            binding=RecoveryBinding(0, 2, 0, 0, 0),
            peer_binding=RecoveryBinding(1, 2, 1, 0, 0),
            parameter_ids=ids,
            epoch_provider=lambda: 3,
            runtime_capture=lambda epoch: {},
            runtime_restore=lambda state: restored_runtime.update(state)
            or state["iteration"],
            materialized=materialized,
            run_id="integration-run",
            global_commit_root=global_root if materialized else None,
        )
        if materialized:
            repository = restored.global_repository
            restored.global_repository = None
            with pytest.raises(RuntimeError, match="requires global commit evidence"):
                restored.restore(base_epoch=0, final_epoch=3)
            restored.global_repository = repository

            wrong_full = copy.deepcopy(global_evidence)
            wrong_full[1]["paths"]["local_full"]["manifest_id"] = "wrong-full"
            GlobalCommitRepository(global_root).publish(
                create_global_epoch(
                    run_id="integration-run",
                    epoch=3,
                    world_size=2,
                    ranks=wrong_full,
                )
            )
            with pytest.raises(ValueError, match="invalid materialized checkpoint"):
                restored.restore(base_epoch=0, final_epoch=3)

            wrong_delta = copy.deepcopy(global_evidence)
            wrong_delta[0]["paths"]["remote_delta"]["payload_checksum"] = (
                "wrong-delta"
            )
            GlobalCommitRepository(global_root).publish(
                create_global_epoch(
                    run_id="integration-run",
                    epoch=3,
                    world_size=2,
                    ranks=wrong_delta,
                )
            )
            with pytest.raises(ValueError, match="does not match materialized recovery"):
                restored.restore(base_epoch=0, final_epoch=3)

            GlobalCommitRepository(global_root).publish(
                create_global_epoch(
                    run_id="integration-run",
                    epoch=3,
                    world_size=2,
                    ranks=global_evidence,
                )
            )
        assert restored.restore(base_epoch=0, final_epoch=3) == 3
        assert _equal(module.state_dict(), objects[0][0].state_dict())
        expected_optimizer = objects[0][1].state_dict()
        if materialized:
            assert _close(optimizer.state_dict(), expected_optimizer)
        else:
            assert _equal(optimizer.state_dict(), expected_optimizer), _first_difference(
                optimizer.state_dict(), expected_optimizer
            )
        assert scheduler.state_dict() == objects[0][2].state_dict()
        assert restored_runtime == {"iteration": 3, "rank": 0}
        if materialized:
            next_gradient = [0.4, -0.2]
            _step(objects[0][1], objects[0][2], next_gradient)
            _step(optimizer, scheduler, next_gradient)
            assert _equal(module.state_dict(), objects[0][0].state_dict())
            assert _close(optimizer.state_dict(), objects[0][1].state_dict())
            assert scheduler.state_dict() == objects[0][2].state_dict()
