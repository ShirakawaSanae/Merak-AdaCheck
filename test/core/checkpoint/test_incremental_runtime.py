import copy
from contextlib import contextmanager
import json
import threading

import torch

from Merak.core.checkpoint.incremental_checkpoint import (
    RecoveryBinding,
    replay_optimizer_updates,
)
from Merak.core.checkpoint.incremental_runtime import (
    IncrementalCheckpointSession,
    PeerIncrementalCheckpointSession,
    restore_incremental_chain,
)
from Merak.core.checkpoint.recovery import ServiceRemoteCheckpointSession
from Merak.core.checkpoint.replica_service import ReplicaTCPServer, ServiceReplicaStore


@contextmanager
def _service():
    store = ServiceReplicaStore(
        max_bytes=64 * 1024 * 1024, retained_epochs_per_source=8
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


def _adam(parameters):
    return torch.optim.AdamW(parameters, lr=1e-2, foreach=False, fused=False)


def test_base_and_async_updates_restore_as_one_chain(tmp_path):
    with _service() as endpoint:
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        optimizer = _adam([parameter])
        initial_parameter = parameter.detach().clone()
        initial_optimizer = copy.deepcopy(optimizer.state_dict())
        binding = RecoveryBinding(0, 2, 0, 0, 0)
        service_session = ServiceRemoteCheckpointSession(
            rank=0,
            group_ranks=(0, 1),
            service_endpoint=endpoint,
            manifest_root=tmp_path,
            max_local_store_bytes=1,
        )
        incremental = IncrementalCheckpointSession(
            service_session,
            optimizer,
            parameter_ids=(("weight",),),
            binding=binding,
            max_inflight=4,
            update_catalog_dir=tmp_path / "update-catalog",
            run_id="catalog-test",
        )
        incremental.save_base(
            0,
            {"parameter": initial_parameter, "optimizer": initial_optimizer},
        )
        for epoch in range(1, 4):
            parameter.grad = torch.tensor([epoch / 10, -epoch / 20])
            incremental.enqueue_update(epoch, metadata={"step": epoch})
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        committed = incremental.close()
        assert [item.epoch for item in committed] == [1, 2, 3]
        catalog = json.loads(
            (
                tmp_path
                / "update-catalog"
                / "epoch_00000001"
                / "rank_00000.json"
            ).read_text()
        )
        assert catalog["snapshot_point"] == "post_unscale_pre_optimizer_step"
        assert catalog["run_id"] == "catalog-test"
        assert len(catalog["catalog_hash"]) == 64
        assert catalog["gradient_tensor_count"] == 1
        assert catalog["gradient_logical_bytes"] == 8
        assert not catalog["adacheck_payload_faithful"]
        assert not catalog["eligible_for_runtime_gate"]

        expected_parameter = parameter.detach().clone()
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        chain = incremental.restore_chain(3)
        replay_parameter = torch.nn.Parameter(chain.base_state["parameter"].clone())
        replay_optimizer = _adam([replay_parameter])
        replay_optimizer.load_state_dict(chain.base_state["optimizer"])
        final_epoch = replay_optimizer_updates(
            replay_optimizer,
            chain.updates,
            parameter_ids=(("weight",),),
            binding=binding,
            base_id=chain.base_id,
            base_epoch=chain.base_epoch,
        )

        assert final_epoch == 3
        assert torch.equal(replay_parameter, expected_parameter)
        assert replay_optimizer.state_dict().keys() == expected_optimizer.keys()
        assert chain.restore_wire_bytes > 0


def test_peer_local_stream_uses_the_semantic_gradient_dtype(tmp_path):
    with _service() as local_endpoint, _service() as remote_endpoint:
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        optimizer = _adam([parameter])
        binding = RecoveryBinding(0, 2, 0, 0, 0)
        local = ServiceRemoteCheckpointSession(
            rank=0,
            target_rank=0,
            group_ranks=(0, 1),
            service_endpoint=local_endpoint,
            manifest_root=tmp_path / "local",
            max_local_store_bytes=1,
        )
        remote = ServiceRemoteCheckpointSession(
            rank=0,
            target_rank=1,
            group_ranks=(0, 1),
            service_endpoint=remote_endpoint,
            manifest_root=tmp_path / "remote",
            max_local_store_bytes=1,
        )
        incremental = PeerIncrementalCheckpointSession(
            local,
            remote,
            optimizer,
            parameter_ids=(("weight",),),
            binding=binding,
            peer_rank=1,
            max_inflight=2,
            gradient_dtype=torch.bfloat16,
        )
        incremental.save_base(
            0,
            {"optimizer": copy.deepcopy(optimizer.state_dict())},
            runtime_state={"iteration": 0},
        )
        parameter.grad = torch.tensor([0.25, -0.125])
        incremental.enqueue_update(
            1,
            metadata={"adacheck_semantic_encoder": {"plan_hash": "test"}},
            runtime_state={"iteration": 1},
        )
        assert [item.epoch for item in incremental.close()] == [1]

        chain = restore_incremental_chain(local, base_epoch=0, final_epoch=1)
        assert chain.updates[0]["gradient_dtype"] == "torch.bfloat16"
        assert chain.updates[0]["gradients"][0][0].dtype == torch.bfloat16
