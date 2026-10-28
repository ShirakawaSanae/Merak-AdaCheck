from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import copy
import json
import threading

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from Merak.core.checkpoint.incremental_checkpoint import RecoveryBinding
from Merak.core.checkpoint.object_replica_service import ObjectReplicaServiceStore
from Merak.core.checkpoint.object_semantic_base import load_object_semantic_base_locator
from Merak.core.checkpoint.object_semantic_deduplicated_base import (
    commit_distributed_deduplicated_object_semantic_base,
    restore_deduplicated_object_semantic_base,
)
from Merak.core.checkpoint.object_semantic_plan import RankBinding
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


def _layout():
    return tuple(
        RankBinding(rank, rank // 2, 0, rank % 2, rank // 2)
        for rank in range(4)
    )


def _base(rank, epoch, *, divergent=False):
    pipeline = rank // 2
    module = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(pipeline + 1 + (1 if divergent and rank == 1 else 0))
    optimizer = torch.optim.AdamW(
        ({"params": tuple(module.parameters())}, {"params": ()}),
        lr=0.01,
        foreach=False,
        fused=False,
    )
    module.weight.grad = torch.full_like(module.weight, pipeline + 0.25)
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
        "binding": asdict(RecoveryBinding(rank, 4, rank % 2, pipeline, 0)),
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


def _worker(rank, rendezvous, endpoints, locator_path, epoch, divergent):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=4)
    clients = {
        host: ReplicaServiceClient(*endpoint) for host, endpoint in enumerate(endpoints)
    }
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return _base(rank, epoch, divergent=divergent)

    try:
        try:
            published = commit_distributed_deduplicated_object_semantic_base(
                clients,
                run_id="deduplicated-base-run",
                epoch=epoch,
                local_base_factory=factory,
                rank_bindings=_layout(),
                locator_path=locator_path,
                group=dist.group.WORLD,
                max_parallel_uploads=1,
            )
            report = {
                "ok": True,
                "calls": calls,
                "base_id": published.local_base_id,
                "locator_hash": published.locator.locator_hash,
            }
        except RuntimeError as error:
            report = {"ok": False, "calls": calls, "error": str(error)}
        with open(f"{locator_path}.rank-{rank}", "w") as stream:
            json.dump(report, stream)
    finally:
        dist.destroy_process_group()


def _run(tmp_path, endpoints, *, epoch, divergent=False):
    locator = tmp_path / f"base-{epoch}.json"
    mp.spawn(
        _worker,
        args=(str(tmp_path / f"rendezvous-{epoch}"), endpoints, str(locator), epoch, divergent),
        nprocs=4,
    )
    reports = [
        json.loads((tmp_path / f"base-{epoch}.json.rank-{rank}").read_text())
        for rank in range(4)
    ]
    return locator, reports


def test_deduplicated_base_restores_every_rank_from_either_host(tmp_path):
    with _services() as (endpoints, servers):
        locator_path, reports = _run(tmp_path, endpoints, epoch=3)
        assert all(report["ok"] and report["calls"] == 1 for report in reports)
        assert len({report["locator_hash"] for report in reports}) == 1
        locator = load_object_semantic_base_locator(locator_path)
        for rank in range(4):
            restored = restore_deduplicated_object_semantic_base(
                ReplicaServiceClient(*endpoints[rank % 2]),
                surviving_host=rank % 2,
                locator=locator,
                logical_rank=rank,
                expected_rank_bindings=_layout(),
                expected_parameter_ids=(("weight",), ()),
                expected_base_id=reports[rank]["base_id"],
            )
            assert restored.state["binding"]["rank"] == rank
            assert restored.state["runtime_state"]["sampler"]["consumed_samples"] == rank * 10
            assert torch.equal(
                restored.state["model"]["weight"],
                _base(rank, 3)["model"]["weight"],
            )
        sealed = servers[0].store.stats()["object_sealed_epochs"]
        assert {source for _, source, epoch in sealed if epoch == 3} == {-1, 0, 1, 2, 3, 4, 6}


def test_deduplicated_base_rejects_divergent_dp_replica(tmp_path):
    with _services() as (endpoints, _):
        locator_path, reports = _run(tmp_path, endpoints, epoch=4, divergent=True)
    assert not locator_path.exists()
    assert not any(report["ok"] for report in reports)
    assert all("data-parallel base replicas diverged" in report["error"] for report in reports)
