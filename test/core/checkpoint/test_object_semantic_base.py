from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import copy
import json
import threading
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from Merak.core.checkpoint.object_replica_service import ObjectReplicaServiceStore
from Merak.core.checkpoint.object_semantic_base import (
    commit_distributed_object_semantic_base,
    load_object_semantic_base_locator,
    restore_object_semantic_base,
)
from Merak.core.checkpoint.object_semantic_plan import RankBinding
from Merak.core.checkpoint.incremental_checkpoint import RecoveryBinding
from Merak.core.checkpoint.replica_service import ReplicaServiceClient, ReplicaTCPServer


@contextmanager
def _services():
    with ExitStack() as stack:
        servers = []
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
            servers.append(server)
        yield tuple(server.server_address for server in servers), servers


def _contains_tensor(value):
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(item) for item in value)
    return False


def _layout():
    return tuple(RankBinding(rank, 0, 0, rank, rank // 2) for rank in range(4))


class _FailingClient:
    endpoint = "injected-failure"

    def put_prepared(self, _):
        raise RuntimeError("injected base submission failure")


def _base(rank, epoch):
    module = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(rank + 1)
    optimizer = torch.optim.AdamW(
        ({"params": tuple(module.parameters())}, {"params": ()}),
        lr=0.01,
        foreach=False,
        fused=False,
    )
    module.weight.grad = torch.full_like(module.weight, rank + 0.25)
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return {
        "schema_version": 1,
        "kind": "merak_incremental_base",
        "iteration": epoch,
        "model": copy.deepcopy(module.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "lr_scheduler": copy.deepcopy(scheduler.state_dict()),
        "parameter_ids": [["weight"], []],
        "binding": asdict(RecoveryBinding(rank, 4, rank, 0, 0)),
        "runtime_state": {
            "iteration": epoch,
            "random_rng_state": (3, (rank, epoch), None),
            "np_rng_state": ("MT19937", [rank, epoch], 0, 0, 0.0),
            "torch_rng_state": torch.tensor([rank, epoch], dtype=torch.uint8),
            "cuda_rng_state": torch.tensor([epoch], dtype=torch.uint8),
            "rng_tracker_states": {"model": torch.tensor([rank], dtype=torch.uint8)},
            "sampler": {"consumed_samples": rank * 10, "epoch": epoch},
        },
    }


def _worker(rank, rendezvous, endpoints, locator_path, epoch, fail):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4)
    clients = {host: ReplicaServiceClient(*endpoint) for host, endpoint in enumerate(endpoints)}
    if fail and rank == 2:
        clients[1] = _FailingClient()
    original_gather, original_broadcast = dist.all_gather_object, dist.broadcast_object_list

    def checked_gather(output, value, *args, **kwargs):
        assert not _contains_tensor(value)
        json.dumps(value, allow_nan=False, sort_keys=True)
        return original_gather(output, value, *args, **kwargs)

    def checked_broadcast(values, *args, **kwargs):
        assert not _contains_tensor(values)
        json.dumps(values, allow_nan=False, sort_keys=True)
        return original_broadcast(values, *args, **kwargs)

    dist.all_gather_object, dist.broadcast_object_list = checked_gather, checked_broadcast
    try:
        state = _base(rank, epoch)
        try:
            published = commit_distributed_object_semantic_base(
                clients,
                run_id="sealed-base-run",
                epoch=epoch,
                local_base=state,
                rank_bindings=_layout(),
                locator_path=locator_path,
                group=dist.group.WORLD,
            )
            result = {"ok": True, "locator_hash": published.locator.locator_hash}
        except RuntimeError as error:
            result = {"ok": False, "error": str(error)}
        Path(locator_path).with_name(f"rank-{rank}.json").write_text(json.dumps(result))
    finally:
        dist.all_gather_object, dist.broadcast_object_list = original_gather, original_broadcast
        dist.destroy_process_group()


def _run(tmp_path, endpoints, *, epoch, fail=False):
    locator = tmp_path / f"base-{epoch}.json"
    mp.spawn(
        _worker,
        args=(str(tmp_path / f"rendezvous-{epoch}"), endpoints, str(locator), epoch, fail),
        nprocs=4,
    )
    reports = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(4)]
    return locator, reports


def test_four_rank_two_host_base_restores_target_rank_from_surviving_host(tmp_path):
    with _services() as (endpoints, servers):
        locator_path, reports = _run(tmp_path, endpoints, epoch=7)
        assert locator_path.is_file()
        assert all(report["ok"] for report in reports)
        assert len({report["locator_hash"] for report in reports}) == 1
        raw_locator = json.loads(locator_path.read_text())
        assert "arm" not in raw_locator
        locator = load_object_semantic_base_locator(locator_path)
        for host in (0, 1):
            restored = restore_object_semantic_base(
                ReplicaServiceClient(*endpoints[host]),
                surviving_host=host,
                locator=locator,
                logical_rank=2,
                expected_rank_bindings=_layout(),
                expected_parameter_ids=(("weight",), ()),
            )
            assert restored.state["binding"]["rank"] == 2
            assert restored.state["runtime_state"]["iteration"] == 7
            assert restored.base_id.startswith("sha256:")

        servers[0].shutdown()
        restored = restore_object_semantic_base(
            ReplicaServiceClient(*endpoints[1]),
            surviving_host=1,
            locator=locator,
            logical_rank=3,
            expected_rank_bindings=_layout(),
            expected_parameter_ids=(("weight",), ()),
        )
        assert restored.state["binding"]["rank"] == 3

        raw_locator["run_id"] = "tampered"
        locator_path.write_text(json.dumps(raw_locator))
        with pytest.raises(ValueError, match="checksum"):
            load_object_semantic_base_locator(locator_path)


def test_rank_commit_failure_does_not_publish_locator(tmp_path):
    with _services() as (endpoints, _):
        locator_path, reports = _run(tmp_path, endpoints, epoch=8, fail=True)
    assert not locator_path.exists()
    assert not any(report["ok"] for report in reports)
