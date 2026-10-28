from contextlib import ExitStack, contextmanager
import hashlib
import json
import threading
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from Merak.core.checkpoint.incremental_checkpoint import (
    RecoveryBinding,
    capture_optimizer_update,
)
from Merak.core.checkpoint.object_optimizer_update import (
    externalize_local_optimizer_update,
    externalize_owned_snapshot_optimizer_update,
)
from Merak.core.checkpoint.object_replica_service import ObjectReplicaServiceStore
from Merak.core.checkpoint.object_semantic_distributed import (
    gather_active_gradient_plan,
    gather_confirmed_gradient_encoding,
)
from Merak.core.checkpoint.object_semantic_plan import (
    GradientShard,
    RankBinding,
    build_active_gradient_plan,
    confirm_gradient_equivalence,
)
from Merak.core.checkpoint import object_update_distributed
from Merak.core.checkpoint.object_update_distributed import (
    commit_distributed_object_update_epoch,
    commit_owned_snapshot_object_update_epoch,
)
from Merak.core.checkpoint.object_update_transport import (
    load_object_update_locator,
    restore_object_update_epoch,
)
from Merak.core.checkpoint.replica_service import ReplicaServiceClient, ReplicaTCPServer


BASE_ID = "sha256:" + hashlib.sha256(b"distributed-base").hexdigest()


@contextmanager
def _services():
    with ExitStack() as stack:
        services = []
        for _ in range(2):
            server = ReplicaTCPServer(
                ("127.0.0.1", 0),
                ObjectReplicaServiceStore(64 << 20, retained_epochs_per_source=4),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            stack.callback(thread.join, timeout=5)
            stack.callback(server.server_close)
            stack.callback(server.shutdown)
            services.append(server)
        yield tuple(server.server_address for server in services)


def _contains_tensor(value):
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


def _gradients(epoch):
    return {
        "left": torch.tensor([0.25 * epoch, -0.5]),
        "right": torch.tensor([-0.75, 0.125 * epoch]),
    }


def _record(rank, world_size, epoch, gradients):
    parameters = [
        torch.nn.Parameter(torch.tensor([1.0, -2.0])),
        torch.nn.Parameter(torch.tensor([3.0, -4.0])),
    ]
    optimizer = torch.optim.AdamW(parameters, lr=0.003, foreach=False, fused=False)
    for parameter, name in zip(parameters, ("left", "right")):
        parameter.grad = gradients[name].clone()
    return capture_optimizer_update(
        optimizer,
        parameter_ids=(("left", "right"),),
        binding=RecoveryBinding(rank, world_size, rank, 0, 0),
        base_id=BASE_ID,
        base_epoch=0,
        epoch=epoch,
    )


def _worker(rank, world_size, rendezvous, endpoints, output_dir, cases, owned_snapshot=False):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size
    )
    clients = {host: ReplicaServiceClient(*endpoint) for host, endpoint in enumerate(endpoints)}
    original_gather, original_broadcast = dist.all_gather_object, dist.broadcast_object_list
    collective_values = []

    def checked_gather(output, value, *args, **kwargs):
        assert not _contains_tensor(value)
        json.dumps(value, allow_nan=False, sort_keys=True)
        collective_values.append(("gather", value))
        return original_gather(output, value, *args, **kwargs)

    def checked_broadcast(values, *args, **kwargs):
        assert not _contains_tensor(values)
        json.dumps(values, allow_nan=False, sort_keys=True)
        collective_values.append(("broadcast", values))
        return original_broadcast(values, *args, **kwargs)

    dist.all_gather_object, dist.broadcast_object_list = checked_gather, checked_broadcast
    try:
        for arm, epoch in cases:
            binding = RankBinding(rank, 0, 0, rank, rank // 2)
            plan = gather_active_gradient_plan(
                binding,
                tuple(
                    GradientShard(
                        rank, name, f"detector-{name}", (2,), (0,), (2,), "torch.float32"
                    )
                    for name in ("left", "right")
                ),
                arm=arm,
                detector_catalog_hash="a" * 64,
                semantic_plan_hash="b" * 64,
                fixed_pairs=((0, 2), (1, 3)),
                group=dist.group.WORLD,
            )
            gradients = _gradients(epoch)
            encoding = gather_confirmed_gradient_encoding(
                plan,
                {item.object_id: gradients[item.fqn] for item in plan.objects if rank in item.holders},
                epoch=epoch,
                group=dist.group.WORLD,
            )
            externalize = (
                externalize_owned_snapshot_optimizer_update
                if owned_snapshot else externalize_local_optimizer_update
            )
            commit = (
                commit_owned_snapshot_object_update_epoch
                if owned_snapshot else commit_distributed_object_update_epoch
            )
            update = externalize(
                plan, encoding, rank, _record(rank, world_size, epoch, gradients)
            )
            published = commit(
                clients,
                run_id=f"distributed-{arm}",
                plan=plan,
                encoding=encoding,
                update=update,
                locator_path=output_dir / f"{arm}-{epoch}.json",
                group=dist.group.WORLD,
            )
            assert collective_values[-1][0] == "broadcast"
            assert collective_values[-2][0] == "gather"
            assert set(collective_values[-2][1]) == {"ok", "evidence"}
            (output_dir / f"{arm}-{epoch}-rank-{rank}.json").write_text(
                json.dumps(
                    {
                        "plan_hash": plan.plan_hash,
                        "encoding_hash": encoding.encoding_hash,
                        "locator_hash": published.locator.locator_hash,
                        "metrics": published.metrics.__dict__,
                        "timing": published.timing,
                    },
                    sort_keys=True,
                )
            )
    finally:
        dist.all_gather_object, dist.broadcast_object_list = original_gather, original_broadcast
        dist.destroy_process_group()


def _oracle(arm, epoch):
    bindings = tuple(RankBinding(rank, 0, 0, rank, rank // 2) for rank in range(4))
    shards = tuple(
        GradientShard(rank, name, f"detector-{name}", (2,), (0,), (2,), "torch.float32")
        for rank in range(4)
        for name in ("left", "right")
    )
    plan = build_active_gradient_plan(
        shards,
        bindings,
        arm=arm,
        detector_catalog_hash="a" * 64,
        semantic_plan_hash="b" * 64,
        fixed_pairs=((0, 2), (1, 3)),
    )
    values = _gradients(epoch)
    objects = {item.fqn: item for item in plan.objects}
    encoding = confirm_gradient_equivalence(
        plan,
        {(rank, objects[name].object_id): value for rank in range(4) for name, value in values.items()},
        epoch=epoch,
    )
    records = {rank: _record(rank, 4, epoch, values) for rank in range(4)}
    return plan, encoding, records


def _nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _nested_equal(a, b)
    else:
        assert left == right


def _reports(root, arm, epoch):
    return [
        json.loads((root / f"{arm}-{epoch}-rank-{rank}.json").read_text())
        for rank in range(4)
    ]


def test_local_bundle_puts_overlap_and_preserve_host_evidence_order(monkeypatch):
    plan, encoding, records = _oracle("m1", 1)
    update = externalize_local_optimizer_update(plan, encoding, 0, records[0])
    both_started = threading.Event()
    release = threading.Event()
    host_one_finished = threading.Event()
    lock = threading.Lock()
    active = {"count": 0, "maximum": 0}
    entered, completed = set(), []

    def commit(_client, bundle, **kwargs):
        host = kwargs["target_host"]
        with lock:
            active["count"] += 1
            active["maximum"] = max(active["maximum"], active["count"])
            entered.add(host)
            if entered == {0, 1}:
                both_started.set()
        assert release.wait(timeout=5)
        if host == 0:
            assert host_one_finished.wait(timeout=5)
        else:
            host_one_finished.set()
        with lock:
            completed.append(host)
            active["count"] -= 1
        return SimpleNamespace(payload_bytes=host + 10, target_host=host), host + 20, host + 0.5

    monkeypatch.setattr(object_update_distributed, "commit_object_update_state", commit)
    monkeypatch.setattr(
        object_update_distributed,
        "manifest_evidence",
        lambda manifest: {"target_host": manifest.target_host},
    )
    result = {}
    worker = threading.Thread(
        target=lambda: result.setdefault(
            "value",
            object_update_distributed._local_evidence(
                rank=0,
                plan=plan,
                encoding=encoding,
                update=update,
                clients={0: object(), 1: object()},
            ),
        )
    )
    worker.start()
    assert both_started.wait(timeout=5)
    assert active["maximum"] == 2
    release.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert completed == [1, 0]
    assert result["value"]["bundles_by_host"] == [
        [0, {"target_host": 0}],
        [1, {"target_host": 1}],
    ]


def test_distributed_m1_two_epochs_publish_locator_and_restore_from_each_host(tmp_path):
    with _services() as endpoints:
        mp.spawn(
            _worker,
            args=(4, str(tmp_path / "m1-rendezvous"), endpoints, tmp_path, (("m1", 1), ("m1", 2)), True),
            nprocs=4,
        )
        clients = {host: ReplicaServiceClient(*endpoint) for host, endpoint in enumerate(endpoints)}
        for epoch in (1, 2):
            locator_path = tmp_path / f"m1-{epoch}.json"
            assert locator_path.is_file()
            locator = load_object_update_locator(locator_path)
            plan, encoding, records = _oracle("m1", epoch)
            for survivor in (0, 1):
                restored = restore_object_update_epoch(
                    clients[survivor],
                    surviving_host=survivor,
                    locator=locator,
                    plan=plan,
                    encoding=encoding,
                )
                for rank, record in restored.records:
                    _nested_equal(record, records[rank])
            reports = _reports(tmp_path, "m1", epoch)
            assert len({report["plan_hash"] for report in reports}) == 1
            assert len({report["encoding_hash"] for report in reports}) == 1
            assert len({report["locator_hash"] for report in reports}) == 1
            assert all(report["metrics"]["remote_wire_bytes"] > 0 for report in reports)
            assert all(report["timing"]["local_bundle_commit_ms"] >= 0 for report in reports)
            assert all(report["timing"]["manifest_gather_ms"] >= 0 for report in reports)
            assert all(report["timing"]["result_broadcast_ms"] >= 0 for report in reports)
            assert reports[0]["timing"]["coordinator_control_commit_ms"] >= 0
            assert all(
                report["timing"]["coordinator_control_commit_ms"] is None
                for report in reports[1:]
            )


def test_distributed_three_arms_have_one_epoch_hash_agreement(tmp_path):
    cases = (("adacheck_fixed", 1), ("host_global_fixed", 2), ("m1", 3))
    with _services() as endpoints:
        mp.spawn(
            _worker,
            args=(4, str(tmp_path / "arms-rendezvous"), endpoints, tmp_path, cases),
            nprocs=4,
        )
    for arm, epoch in cases:
        reports = _reports(tmp_path, arm, epoch)
        for name in ("plan_hash", "encoding_hash", "locator_hash"):
            assert len({report[name] for report in reports}) == 1
        assert all(report["timing"]["local_bundle_commit_ms"] >= 0 for report in reports)
        assert reports[0]["timing"]["coordinator_seal_locator_ms"] >= 0
        assert all(
            report["timing"]["coordinator_seal_locator_ms"] is None
            for report in reports[1:]
        )
        assert (tmp_path / f"{arm}-{epoch}.json").is_file()
