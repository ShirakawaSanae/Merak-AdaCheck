import copy
from pathlib import Path
import threading
import time

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from Merak.core.checkpoint.global_commit import (
    FilesystemGlobalCommitCoordinator,
    GlobalCommitCoordinator,
    GlobalCommitRepository,
    create_global_epoch,
    rank_evidence,
    validate_global_epoch,
)


def _rank(rank, epoch):
    return _rank_with_peer(rank, 1 - rank, epoch)


def _rank_with_peer(rank, peer, epoch):
    group = [min(rank, peer), max(rank, peer)]
    return {
        "rank": rank,
        "epoch": epoch,
        "mode": "peer_delta",
        "peer_equivalence_checksum": f"{min(rank, peer):064x}",
        "paths": {
            "local_full": {
                "manifest_id": f"manifest-{rank}",
                "payload_checksum": f"payload-{rank}",
                "epoch": epoch,
                "source_rank": rank,
                "target_rank": rank,
                "group_ranks": group,
            },
            "remote_delta": {
                "manifest_id": f"delta-{rank}",
                "payload_checksum": f"delta-payload-{rank}",
                "epoch": epoch,
                "source_rank": rank,
                "target_rank": peer,
                "group_ranks": group,
            },
        },
    }


def _distributed_worker(rank, world_size, rendezvous, root):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        coordinator = GlobalCommitCoordinator(
            rank=rank,
            world_size=world_size,
            run_id="distributed-run",
            root=root,
        )
        coordinator.advance(0, lambda epoch: _rank(rank, epoch))
        assert coordinator.close() == 0
    finally:
        dist.destroy_process_group()


def _failing_distributed_worker(rank, world_size, rendezvous, root, failure):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        coordinator = GlobalCommitCoordinator(
            rank=rank,
            world_size=world_size,
            run_id="failed-run",
            root=root,
        )
        if failure == "writer" and rank == 0:
            def fail_publish(_control):
                raise OSError("injected writer failure")

            coordinator.repository.publish = fail_publish
        local = _rank(rank, 0)
        if failure == "evidence" and rank == 1:
            local["paths"].pop("remote_delta")
        with pytest.raises(RuntimeError, match="global checkpoint commit failed"):
            coordinator.advance(0, lambda _epoch: local)
            coordinator.close()
        (Path(root) / f"caught-{failure}-{rank}").write_text("caught\n")
    finally:
        dist.destroy_process_group()


def _sealed_distributed_worker(rank, world_size, rendezvous, root, fail_pin):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        def pin(control):
            assert not (Path(root) / "epoch_00000000.json").exists()
            assert control["publication_protocol"] == "pin-before-publish-v1"
            if fail_pin and rank == 1:
                raise OSError("injected pin failure")
            (Path(root) / f"pinned-{rank}").write_text("pinned\n")

        def seal(_control):
            assert (Path(root) / "epoch_00000000.json").exists()
            (Path(root) / f"sealed-{rank}").write_text("sealed\n")

        coordinator = GlobalCommitCoordinator(
            rank=rank,
            world_size=world_size,
            run_id="sealed-run",
            root=root,
            before_publish=pin,
            after_publish=seal,
        )
        if fail_pin:
            with pytest.raises(RuntimeError, match="checkpoint pin failed"):
                coordinator.advance(0, lambda epoch: _rank(rank, epoch))
        else:
            assert coordinator.advance(0, lambda epoch: _rank(rank, epoch)) == 0
    finally:
        dist.destroy_process_group()


def test_global_epoch_is_canonical_and_detects_tampering(tmp_path):
    control = create_global_epoch(
        run_id="run-a", epoch=3, world_size=2, ranks=[_rank(1, 3), _rank(0, 3)]
    )
    assert [item["rank"] for item in control["ranks"]] == [0, 1]
    repository = GlobalCommitRepository(tmp_path)
    repository.publish(control)
    loaded = repository.load(run_id="run-a", epoch=3, world_size=2)
    assert loaded == control
    assert rank_evidence(loaded, 1)["epoch"] == 3

    tampered = copy.deepcopy(control)
    tampered["ranks"][0]["paths"]["local_full"]["payload_checksum"] = "changed"
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_global_epoch(tampered, run_id="run-a", epoch=3, world_size=2)


def test_global_epoch_rejects_missing_or_misbound_rank():
    with pytest.raises(ValueError, match="every rank"):
        create_global_epoch(
            run_id="run-a", epoch=1, world_size=2, ranks=[_rank(0, 1), _rank(0, 1)]
        )
    wrong = _rank(1, 1)
    wrong["paths"]["local_full"]["source_rank"] = 0
    with pytest.raises(ValueError, match="inconsistent"):
        create_global_epoch(
            run_id="run-a", epoch=1, world_size=2, ranks=[_rank(0, 1), wrong]
        )

    cycle = [
        _rank_with_peer(rank, (rank + 1) % 4, 1) for rank in range(4)
    ]
    with pytest.raises(ValueError, match="reciprocal peer pairs"):
        create_global_epoch(
            run_id="run-a", epoch=1, world_size=4, ranks=cycle
        )


def test_sealed_global_epoch_rejects_divergent_peer_state():
    ranks = [_rank(0, 1), _rank(1, 1)]
    ranks[1]["peer_equivalence_checksum"] = "f" * 64

    with pytest.raises(ValueError, match="not equivalent"):
        create_global_epoch(
            run_id="run-a",
            epoch=1,
            world_size=2,
            ranks=ranks,
            publication_protocol="pin-before-publish-v1",
        )


def test_global_epoch_requires_one_complete_semantic_encoder_plan():
    ranks = [_rank(0, 1), _rank(1, 1)]
    common = {
        "schema_version": 1,
        "kind": "adacheck_semantic_fixed",
        "plan_hash": "a" * 64,
        "trainable_action": "gradient_bf16",
        "gradient_dtype": "torch.bfloat16",
        "metadata_action": "metadata_exact",
        "global_action_count": 3,
    }
    ranks[0]["adacheck_semantic_encoder"] = {
        **common,
        "rank_action_count": 1,
    }
    ranks[1]["adacheck_semantic_encoder"] = {
        **common,
        "rank_action_count": 2,
    }
    control = create_global_epoch(
        run_id="semantic", epoch=1, world_size=2, ranks=ranks
    )
    assert control["adacheck_semantic_encoder"] == common

    incomplete = copy.deepcopy(ranks)
    incomplete[1].pop("adacheck_semantic_encoder")
    with pytest.raises(ValueError, match="partial SemanticFixed"):
        create_global_epoch(
            run_id="semantic", epoch=1, world_size=2, ranks=incomplete
        )

    wrong_count = copy.deepcopy(ranks)
    wrong_count[1]["adacheck_semantic_encoder"]["rank_action_count"] = 1
    with pytest.raises(ValueError, match="action counts"):
        create_global_epoch(
            run_id="semantic", epoch=1, world_size=2, ranks=wrong_count
        )


def test_global_coordinator_publishes_one_shared_watermark(tmp_path):
    rendezvous = tmp_path / "rendezvous"
    root = tmp_path / "global"
    mp.spawn(
        _distributed_worker,
        args=(2, str(rendezvous), str(root)),
        nprocs=2,
        join=True,
    )
    control = GlobalCommitRepository(Path(root)).load(
        run_id="distributed-run", epoch=0, world_size=2
    )
    assert [item["rank"] for item in control["ranks"]] == [0, 1]


def test_filesystem_coordinator_advances_without_waiting_for_slowest_rank(tmp_path):
    root = tmp_path / "filesystem"
    phases = []
    phase_lock = threading.Lock()

    def callbacks(rank):
        def pin(control):
            assert not (root / "epoch_00000000.json").exists()
            with phase_lock:
                phases.append(("pin", rank, control["epoch"]))

        def seal(control):
            assert (root / "epoch_00000000.json").exists()
            with phase_lock:
                phases.append(("seal", rank, control["epoch"]))

        return pin, seal

    pin0, seal0 = callbacks(0)
    pin1, seal1 = callbacks(1)
    coordinators = [
        FilesystemGlobalCommitCoordinator(
            rank=0,
            world_size=2,
            run_id="filesystem-run",
            root=root,
            before_publish=pin0,
            after_publish=seal0,
            timeout_seconds=2,
            poll_interval_seconds=0.005,
        ),
        FilesystemGlobalCommitCoordinator(
            rank=1,
            world_size=2,
            run_id="filesystem-run",
            root=root,
            before_publish=pin1,
            after_publish=seal1,
            timeout_seconds=2,
            poll_interval_seconds=0.005,
        ),
    ]
    started = time.perf_counter()
    assert coordinators[0].advance(0, lambda epoch: _rank(0, epoch)) == -1
    assert time.perf_counter() - started < 0.1
    assert not (root / "epoch_00000000.json").exists()

    coordinators[1].advance(0, lambda epoch: _rank(1, epoch))
    assert coordinators[0].wait_through(0) == 0
    assert coordinators[1].wait_through(0) == 0
    assert [coordinator.close() for coordinator in coordinators] == [0, 0]
    assert sorted(phases) == [
        ("pin", 0, 0),
        ("pin", 1, 0),
        ("seal", 0, 0),
        ("seal", 1, 0),
    ]


def test_filesystem_coordinator_propagates_callback_failure(tmp_path):
    root = tmp_path / "filesystem-failure"

    def fail_pin(_control):
        raise OSError("injected filesystem pin failure")

    coordinators = [
        FilesystemGlobalCommitCoordinator(
            rank=rank,
            world_size=2,
            run_id="filesystem-failed-run",
            root=root,
            before_publish=(fail_pin if rank == 1 else lambda _control: None),
            after_publish=lambda _control: None,
            timeout_seconds=2,
            poll_interval_seconds=0.005,
        )
        for rank in range(2)
    ]
    for rank, coordinator in enumerate(coordinators):
        coordinator.advance(0, lambda epoch, rank=rank: _rank(rank, epoch))
    for coordinator in coordinators:
        with pytest.raises(RuntimeError, match="filesystem global checkpoint"):
            coordinator.wait_through(0)
    assert not (root / "epoch_00000000.json").exists()


@pytest.mark.parametrize("failure", ["evidence", "writer"])
def test_global_coordinator_propagates_rank_zero_failure(tmp_path, failure):
    root = tmp_path / failure
    root.mkdir()
    mp.spawn(
        _failing_distributed_worker,
        args=(2, str(tmp_path / f"rendezvous-{failure}"), str(root), failure),
        nprocs=2,
        join=True,
    )
    assert sorted(path.name for path in root.glob(f"caught-{failure}-*")) == [
        f"caught-{failure}-0",
        f"caught-{failure}-1",
    ]


@pytest.mark.parametrize("fail_pin", [False, True])
def test_global_coordinator_pins_before_sealed_publication(tmp_path, fail_pin):
    root = tmp_path / f"sealed-{fail_pin}"
    root.mkdir()
    mp.spawn(
        _sealed_distributed_worker,
        args=(2, str(tmp_path / f"sealed-rendezvous-{fail_pin}"), str(root), fail_pin),
        nprocs=2,
        join=True,
    )
    control_path = root / "epoch_00000000.json"
    assert control_path.exists() is (not fail_pin)
    if not fail_pin:
        control = GlobalCommitRepository(root).load(
            run_id="sealed-run", epoch=0, world_size=2
        )
        assert control["publication_protocol"] == "pin-before-publish-v1"
        assert sorted(path.name for path in root.glob("sealed-*")) == [
            "sealed-0",
            "sealed-1",
        ]
