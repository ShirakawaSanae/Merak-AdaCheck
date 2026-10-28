import copy
from contextlib import contextmanager
from dataclasses import asdict, replace
from multiprocessing import Process, current_process
from multiprocessing.connection import Client, Listener
from pathlib import Path
from types import SimpleNamespace
import socket
import threading
import time

import pytest
import torch
import Merak.core.checkpoint.replica_service as replica_service_module

from Merak.core.amp.amp_optimizer import HalfPrecisionOptimizer
from Merak.core.checkpoint.cpu_materializer import AdamWCheckpointMaterializer
from Merak.core.checkpoint.catalog import deserialize_checkpoint, serialize_checkpoint
from Merak.core.checkpoint.incremental_checkpoint import (
    RecoveryBinding,
    capture_optimizer_update,
)
from Merak.core.checkpoint.merak_incremental import half_precision_parameter_ids
from Merak.core.checkpoint.global_commit import manifest_evidence
from Merak.core.checkpoint.manifest import create_prepared_manifest
from Merak.core.checkpoint.replica_service import (
    ReplicaServiceClient,
    ReplicaTCPServer,
    ServiceReplicaStore,
)
from Merak.core.checkpoint.shared_tensor import (
    SharedTensorCheckpointSession,
    SharedTensorServer,
    _authkey,
    _configure_tensor_sharing,
    describe_shared_update,
)
from Merak.core.checkpoint.transport import ReplicaEnvelope


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _receive_shared_tensor(socket_path, ready_path):
    current_process().authkey = b"server-default"
    listener = Listener(socket_path, family="AF_UNIX", authkey=_authkey(socket_path))
    Path(ready_path).touch()
    connection = listener.accept()
    tensor = connection.recv()
    connection.send((tuple(tensor.shape), tensor.sum().item()))
    connection.close()
    listener.close()


@contextmanager
def _materializing_service():
    store = ServiceReplicaStore(
        max_bytes=64 * 1024 * 1024,
        materialize_incremental=True,
    )
    server = ReplicaTCPServer(("127.0.0.1", 0), store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield ReplicaServiceClient("127.0.0.1", server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _publish(client, state, *, epoch):
    serialized = serialize_checkpoint(state)
    prepared = create_prepared_manifest(
        epoch=epoch,
        source_rank=0,
        source_host="source",
        target_rank=0,
        target_host=client.endpoint,
        group_ranks=(0, 1),
        serialized=serialized,
    )
    client.put_prepared(ReplicaEnvelope(prepared, serialized.payload))
    committed = replace(
        prepared, state="committed", committed_time_ns=time.time_ns()
    )
    client.commit(committed)
    client.put_prepared(ReplicaEnvelope(prepared, serialized.payload))
    client.commit(committed)
    return committed


def _objects(device="cuda"):
    module = torch.nn.Linear(2, 1, bias=False, device=device, dtype=torch.bfloat16)
    module.weight.data.copy_(torch.tensor([[1.0, -2.0]], device=device))
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


def _base_state(module, optimizer, scheduler, identifiers, binding):
    return {
        "schema_version": 1,
        "kind": "merak_incremental_base",
        "iteration": 0,
        "model": copy.deepcopy(module.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "lr_scheduler": copy.deepcopy(scheduler.state_dict()),
        "lr_scheduler_type": "torch.optim.lr_scheduler.StepLR",
        "parameter_ids": [list(group) for group in identifiers],
        "binding": asdict(binding),
        "stream_role": "peer_local_full",
        "run_id": "test",
    }


def _update_state(optimizer, scheduler, identifiers, binding, base_id, epoch):
    optimizer.fp16_groups[0][0].grad = torch.tensor(
        [[epoch / 10, -epoch / 20]], device="cuda", dtype=torch.bfloat16
    )
    for master, model_parameter in zip(
        optimizer.fp32_groups[0], optimizer.fp16_groups[0]
    ):
        master.grad = model_parameter.grad.float()
    return capture_optimizer_update(
        optimizer.optimizer,
        parameter_ids=identifiers,
        binding=binding,
        base_id=base_id,
        base_epoch=0,
        epoch=epoch,
        metadata={
            "trainer_global_step": epoch,
            "scheduler_state_before_step": scheduler.state_dict(),
            "stream_role": "peer_local_full",
            "run_id": "test",
        },
        runtime_state={"iteration": epoch, "rank": 0},
    )


def test_cpu_materializer_tracks_gpu_adamw_and_produces_restorable_state():
    module, optimizer, scheduler = _objects()
    identifiers = half_precision_parameter_ids(module, optimizer)
    binding = RecoveryBinding(0, 1, 0, 0, 0)
    base_id = "sha256:" + "a" * 64
    base = _base_state(module, optimizer, scheduler, identifiers, binding)
    materializer = AdamWCheckpointMaterializer(
        base,
        base_id=base_id,
        binding=binding,
        parameter_ids=identifiers,
    )

    for epoch in range(1, 4):
        optimizer.fp16_groups[0][0].grad = torch.tensor(
            [[epoch / 10, -epoch / 20]], device="cuda", dtype=torch.bfloat16
        )
        for master, model_parameter in zip(
            optimizer.fp32_groups[0], optimizer.fp16_groups[0]
        ):
            master.grad = model_parameter.grad.float()
        record = capture_optimizer_update(
            optimizer.optimizer,
            parameter_ids=identifiers,
            binding=binding,
            base_id=base_id,
            base_epoch=0,
            epoch=epoch,
            metadata={
                "trainer_global_step": epoch,
                "scheduler_state_before_step": scheduler.state_dict(),
            },
        )
        materializer.apply(record)
        optimizer.optimizer.step()
        optimizer.refresh_model_params()
        scheduler.step()

    state = materializer.snapshot()
    assert state["iteration"] == 3
    assert state["materialization"]["applied_updates"] == 3
    assert torch.equal(state["model"]["weight"], module.weight.detach().cpu())
    assert torch.allclose(
        state["optimizer"]["fp32_groups"][0][0],
        optimizer.fp32_groups[0][0].detach().cpu(),
        rtol=1e-6,
        atol=1e-7,
    )

    restored_module, restored_optimizer, restored_scheduler = _objects()
    restored_module.load_state_dict(state["model"])
    restored_optimizer.load_state_dict(state["optimizer"])
    restored_scheduler.load_state_dict(state["lr_scheduler"])
    restored_scheduler.step()
    restored_optimizer.refresh_model_params()
    assert restored_optimizer.global_step == 3
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert torch.equal(restored_module.weight, module.weight)


def test_materializer_owns_scheduler_snapshot_after_shared_update_reuse():
    module, optimizer, scheduler = _objects()
    identifiers = half_precision_parameter_ids(module, optimizer)
    binding = RecoveryBinding(0, 1, 0, 0, 0)
    base_id = "sha256:" + "a" * 64
    materializer = AdamWCheckpointMaterializer(
        _base_state(module, optimizer, scheduler, identifiers, binding),
        base_id=base_id,
        binding=binding,
        parameter_ids=identifiers,
    )
    record = _update_state(
        optimizer, scheduler, identifiers, binding, base_id, epoch=1
    )
    record["metadata"]["scheduler_state_before_step"]["history"] = [1]

    materializer.apply(record)
    record["metadata"]["scheduler_state_before_step"]["history"].append(2)

    assert materializer.snapshot()["lr_scheduler"]["history"] == [1]


def test_replica_service_materializes_and_discards_update_log():
    module, optimizer, scheduler = _objects()
    identifiers = half_precision_parameter_ids(module, optimizer)
    binding = RecoveryBinding(0, 2, 0, 0, 0)
    base = _base_state(module, optimizer, scheduler, identifiers, binding)

    with _materializing_service() as client:
        committed_base = _publish(client, base, epoch=0)
        base_id = f"sha256:{committed_base.payload_checksum}"
        for epoch in range(1, 4):
            optimizer.fp16_groups[0][0].grad = torch.tensor(
                [[epoch / 10, -epoch / 20]],
                device="cuda",
                dtype=torch.bfloat16,
            )
            for master, model_parameter in zip(
                optimizer.fp32_groups[0], optimizer.fp16_groups[0]
            ):
                master.grad = model_parameter.grad.float()
            record = capture_optimizer_update(
                optimizer.optimizer,
                parameter_ids=identifiers,
                binding=binding,
                base_id=base_id,
                base_epoch=0,
                epoch=epoch,
                metadata={
                    "trainer_global_step": epoch,
                    "scheduler_state_before_step": scheduler.state_dict(),
                    "stream_role": "peer_local_full",
                    "run_id": "test",
                },
            )
            _publish(client, record, epoch=epoch)
            optimizer.optimizer.step()
            optimizer.refresh_model_params()
            scheduler.step()

        stats = client.stats()
        assert stats["materialized_epochs"] == {"0": 3}
        assert stats["record_count"] == 0
        envelope, _ = client.get_materialized(0, 3)
        state = deserialize_checkpoint(
            envelope.payload, envelope.manifest.payload_checksum
        )
        assert state["iteration"] == 3
        assert torch.equal(state["model"]["weight"], module.weight.detach().cpu())


def test_materializer_rejects_catalog_mismatch_before_state_is_visible():
    module, optimizer, scheduler = _objects()
    identifiers = half_precision_parameter_ids(module, optimizer)
    binding = RecoveryBinding(0, 2, 0, 0, 0)
    serialized = serialize_checkpoint(
        _base_state(module, optimizer, scheduler, identifiers, binding)
    )
    prepared = create_prepared_manifest(
        epoch=0,
        source_rank=0,
        source_host="source",
        target_rank=0,
        target_host="target",
        group_ranks=(0, 1),
        serialized=serialized,
    )
    invalid = replace(prepared, objects=())
    store = ServiceReplicaStore(
        max_bytes=64 * 1024 * 1024, materialize_incremental=True
    )
    store.put_prepared(ReplicaEnvelope(invalid, serialized.payload))
    with pytest.raises(ValueError, match="catalog validation"):
        store.commit(
            replace(invalid, state="committed", committed_time_ns=time.time_ns())
        )
    assert store.stats()["materialized_epochs"] == {}


def test_materializer_poisons_wrong_source_binding():
    module, optimizer, scheduler = _objects()
    identifiers = half_precision_parameter_ids(module, optimizer)
    wrong_binding = RecoveryBinding(1, 2, 1, 0, 0)
    serialized = serialize_checkpoint(
        _base_state(module, optimizer, scheduler, identifiers, wrong_binding)
    )
    prepared = create_prepared_manifest(
        epoch=0,
        source_rank=0,
        source_host="source",
        target_rank=0,
        target_host="target",
        group_ranks=(0, 1),
        serialized=serialized,
    )
    store = ServiceReplicaStore(
        max_bytes=64 * 1024 * 1024, materialize_incremental=True
    )
    store.put_prepared(ReplicaEnvelope(prepared, serialized.payload))
    with pytest.raises(ValueError, match="binding does not match"):
        store.commit(
            replace(prepared, state="committed", committed_time_ns=time.time_ns())
        )
    assert 0 in store.stats()["poisoned_sources"]
    with pytest.raises(RuntimeError, match="poisoned"):
        store.get_materialized(0, 0)


def test_materializer_poisons_out_of_order_update():
    module, optimizer, scheduler = _objects()
    identifiers = half_precision_parameter_ids(module, optimizer)
    binding = RecoveryBinding(0, 2, 0, 0, 0)
    with _materializing_service() as client:
        committed_base = _publish(
            client,
            _base_state(module, optimizer, scheduler, identifiers, binding),
            epoch=0,
        )
        record = capture_optimizer_update(
            optimizer.optimizer,
            parameter_ids=identifiers,
            binding=binding,
            base_id=f"sha256:{committed_base.payload_checksum}",
            base_epoch=0,
            epoch=2,
            metadata={
                "trainer_global_step": 2,
                "scheduler_state_before_step": scheduler.state_dict(),
                "stream_role": "peer_local_full",
                "run_id": "test",
            },
        )
        with pytest.raises(RuntimeError, match="not contiguous"):
            _publish(client, record, epoch=2)
        assert "0" in client.stats()["poisoned_sources"]
        with pytest.raises(RuntimeError, match="poisoned"):
            client.get_materialized(0, 0)


def test_put_cannot_replace_a_record_during_materialized_commit(monkeypatch):
    module, optimizer, scheduler = _objects()
    identifiers = half_precision_parameter_ids(module, optimizer)
    binding = RecoveryBinding(0, 2, 0, 0, 0)
    first_state = _base_state(module, optimizer, scheduler, identifiers, binding)
    first = serialize_checkpoint(first_state)
    first_manifest = create_prepared_manifest(
        epoch=0,
        source_rank=0,
        source_host="source",
        target_rank=0,
        target_host="target",
        group_ranks=(0, 1),
        serialized=first,
    )
    store = ServiceReplicaStore(
        max_bytes=64 * 1024 * 1024, materialize_incremental=True
    )
    store.put_prepared(ReplicaEnvelope(first_manifest, first.payload))

    entered = threading.Event()
    release = threading.Event()
    original_deserialize = replica_service_module.deserialize_checkpoint

    def blocked_deserialize(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return original_deserialize(*args, **kwargs)

    monkeypatch.setattr(
        replica_service_module, "deserialize_checkpoint", blocked_deserialize
    )
    commit_errors = []
    commit_thread = threading.Thread(
        target=lambda: _capture_error(
            commit_errors,
            store.commit,
            replace(
                first_manifest,
                state="committed",
                committed_time_ns=time.time_ns(),
            ),
        )
    )
    commit_thread.start()
    assert entered.wait(timeout=2)

    second_state = dict(first_state, run_id="replacement")
    second = serialize_checkpoint(second_state)
    second_manifest = create_prepared_manifest(
        epoch=0,
        source_rank=0,
        source_host="source",
        target_rank=0,
        target_host="target",
        group_ranks=(0, 1),
        serialized=second,
    )
    put_errors = []
    put_thread = threading.Thread(
        target=lambda: _capture_error(
            put_errors,
            store.put_prepared,
            ReplicaEnvelope(second_manifest, second.payload),
        )
    )
    put_thread.start()
    time.sleep(0.05)
    assert put_thread.is_alive()
    release.set()
    commit_thread.join(timeout=2)
    put_thread.join(timeout=2)

    assert not commit_errors
    assert len(put_errors) == 1
    assert isinstance(put_errors[0], ValueError)
    assert "applied replica" in str(put_errors[0])


def test_capacity_eviction_cannot_remove_another_source_commit(monkeypatch):
    module, optimizer, scheduler = _objects()
    identifiers = half_precision_parameter_ids(module, optimizer)
    binding = RecoveryBinding(0, 2, 0, 0, 0)
    serialized = serialize_checkpoint(
        _base_state(module, optimizer, scheduler, identifiers, binding)
    )
    first_manifest = create_prepared_manifest(
        epoch=0,
        source_rank=0,
        source_host="source",
        target_rank=0,
        target_host="target",
        group_ranks=(0, 1),
        serialized=serialized,
    )
    store = ServiceReplicaStore(
        max_bytes=max(len(serialized.payload), 2 * serialized.logical_bytes) + 1,
        materialize_incremental=True,
    )
    store.put_prepared(ReplicaEnvelope(first_manifest, serialized.payload))

    entered = threading.Event()
    release = threading.Event()
    original_deserialize = replica_service_module.deserialize_checkpoint

    def blocked_deserialize(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return original_deserialize(*args, **kwargs)

    monkeypatch.setattr(
        replica_service_module, "deserialize_checkpoint", blocked_deserialize
    )
    commit_errors = []
    commit_thread = threading.Thread(
        target=lambda: _capture_error(
            commit_errors,
            store.commit,
            replace(
                first_manifest,
                state="committed",
                committed_time_ns=time.time_ns(),
            ),
        )
    )
    commit_thread.start()
    assert entered.wait(timeout=2)

    second_manifest = create_prepared_manifest(
        epoch=0,
        source_rank=1,
        source_host="other-source",
        target_rank=1,
        target_host="target",
        group_ranks=(0, 1),
        serialized=serialized,
    )
    with pytest.raises(MemoryError):
        store.put_prepared(ReplicaEnvelope(second_manifest, serialized.payload))
    release.set()
    commit_thread.join(timeout=2)

    assert not commit_errors
    assert store.stats()["materialized_epochs"] == {"0": 0}


def test_shared_tensor_update_reaches_the_materializer(tmp_path):
    module, optimizer, scheduler = _objects()
    identifiers = half_precision_parameter_ids(module, optimizer)
    binding = RecoveryBinding(0, 2, 0, 0, 0)
    store = ServiceReplicaStore(
        max_bytes=64 * 1024 * 1024, materialize_incremental=True
    )
    base_state = _base_state(module, optimizer, scheduler, identifiers, binding)
    base_state["run_id"] = "shared-test"
    base = serialize_checkpoint(base_state)
    prepared = create_prepared_manifest(
        epoch=0,
        source_rank=0,
        source_host=socket.gethostname(),
        target_rank=0,
        target_host="target",
        group_ranks=(0, 1),
        serialized=base,
    )
    store.put_prepared(ReplicaEnvelope(prepared, base.payload))
    store.commit(
        replace(prepared, state="committed", committed_time_ns=time.time_ns())
    )

    optimizer.fp16_groups[0][0].grad = torch.tensor(
        [[0.1, -0.05]], device="cuda", dtype=torch.bfloat16
    )
    for master, model_parameter in zip(
        optimizer.fp32_groups[0], optimizer.fp16_groups[0]
    ):
        master.grad = model_parameter.grad.float()
    update = capture_optimizer_update(
        optimizer.optimizer,
        parameter_ids=identifiers,
        binding=binding,
        base_id=f"sha256:{prepared.payload_checksum}",
        base_epoch=0,
        epoch=1,
        metadata={
            "trainer_global_step": 1,
            "scheduler_state_before_step": scheduler.state_dict(),
            "stream_role": "peer_local_full",
            "run_id": "shared-test",
        },
    )
    socket_path = tmp_path / "shared.sock"
    server = SharedTensorServer(socket_path, store)
    server.start()
    try:
        session = SharedTensorCheckpointSession(
            rank=0,
            group_ranks=(0, 1),
            target_rank=0,
            socket_path=socket_path,
            manifest_root=tmp_path / "manifests",
        )
        result = session.save(
            1, update, state_is_snapshot=True, object_checksums=False
        )
    finally:
        server.stop()

    assert result.transport_kind == "shared_tensor"
    assert result.manifest.target_host == socket.gethostname()
    assert result.transport.payload_bytes == 0
    assert result.shared_memory_bytes == result.manifest.logical_bytes
    assert result.shared_slot_wait_ms >= 0
    assert result.shared_copy_ms >= 0
    assert result.serialize_ms == pytest.approx(
        result.shared_slot_wait_ms + result.shared_copy_ms
    )
    assert store.stats()["materialized_epochs"] == {"0": 1}
    changed = copy.deepcopy(update)
    changed["metadata"]["trainer_global_step"] = 2
    assert describe_shared_update(changed).payload_checksum != (
        describe_shared_update(update).payload_checksum
    )


def test_sealed_materializer_keeps_current_epoch_while_future_updates_are_staged(
    monkeypatch,
):
    module, optimizer, scheduler = _objects()
    identifiers = half_precision_parameter_ids(module, optimizer)
    binding = RecoveryBinding(0, 2, 0, 0, 0)
    store = ServiceReplicaStore(
        max_bytes=64 * 1024 * 1024,
        retained_epochs_per_source=3,
        materialize_incremental=True,
        sealed_incremental=True,
    )
    server = ReplicaTCPServer(("127.0.0.1", 0), store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = ReplicaServiceClient("127.0.0.1", server.server_address[1])
    try:
        base = _base_state(module, optimizer, scheduler, identifiers, binding)
        base["runtime_state"] = {"iteration": 0, "rank": 0}
        committed_base = _publish(client, base, epoch=0)
        base_evidence = manifest_evidence(committed_base)
        base_control = "0" * 64
        client.pin(
            kind="materialized",
            source_rank=0,
            epoch=0,
            control_checksum=base_control,
            expected_manifest=base_evidence,
        )
        client.seal(
            kind="materialized",
            source_rank=0,
            epoch=0,
            control_checksum=base_control,
        )
        client.wait_materialized(0, 0)

        base_id = f"sha256:{committed_base.payload_checksum}"
        updates = [
            store.commit_shared_update(
                epoch=epoch,
                source_rank=0,
                source_host=committed_base.source_host,
                target_rank=0,
                target_host=committed_base.target_host,
                group_ranks=committed_base.group_ranks,
                state=_update_state(
                    optimizer, scheduler, identifiers, binding, base_id, epoch
                ),
            )[0]
            for epoch in (1, 2)
        ]
        assert store.stats()["materialized_epochs"] == {"0": 0}
        assert store.stats()["staged_materialized_epochs"] == [[0, 1], [0, 2]]

        first_control = "1" * 64
        first_evidence = manifest_evidence(updates[0])
        second_evidence = manifest_evidence(updates[1])
        client.pin(
            kind="materialized",
            source_rank=0,
            epoch=1,
            control_checksum=first_control,
            expected_manifest=first_evidence,
        )
        old, _ = client.get_materialized(
            0,
            0,
            control_checksum=base_control,
            expected_manifest=base_evidence,
        )
        assert old.manifest.epoch == 0
        entered = threading.Event()
        release = threading.Event()
        applied = threading.Event()
        publish = threading.Event()
        original_apply = store._materializers[0].apply

        def blocked_apply(state):
            entered.set()
            assert release.wait(timeout=5)
            result = original_apply(state)
            applied.set()
            assert publish.wait(timeout=5)
            return result

        monkeypatch.setattr(store._materializers[0], "apply", blocked_apply)
        client.seal(
            kind="materialized",
            source_rank=0,
            epoch=1,
            control_checksum=first_control,
        )
        assert entered.wait(timeout=2)
        pin_errors = []
        pin_thread = threading.Thread(
            target=lambda: _capture_error(
                pin_errors,
                client.pin,
                kind="materialized",
                source_rank=0,
                epoch=2,
                control_checksum="2" * 64,
                expected_manifest=second_evidence,
            )
        )
        pin_thread.start()
        pin_thread.join(timeout=1)
        pin_completed_while_apply_blocked = not pin_thread.is_alive()
        release.set()
        assert applied.wait(timeout=2)
        get_errors = []
        get_thread = threading.Thread(
            target=lambda: _capture_error(
                get_errors,
                client.get_materialized,
                0,
                1,
                control_checksum=first_control,
                expected_manifest=first_evidence,
            )
        )
        get_thread.start()
        get_thread.join(timeout=1)
        get_waited_for_manifest_publish = get_thread.is_alive()
        publish.set()
        pin_thread.join(timeout=5)
        get_thread.join(timeout=5)
        assert pin_completed_while_apply_blocked
        assert not pin_errors
        assert get_waited_for_manifest_publish
        assert not get_errors
        client.wait_materialized(0, 1)

        first, _ = client.get_materialized(
            0,
            1,
            control_checksum=first_control,
            expected_manifest=first_evidence,
        )
        first_state = deserialize_checkpoint(
            first.payload, first.manifest.payload_checksum
        )
        assert first_state["runtime_state"] == {"iteration": 1, "rank": 0}

        tampered = dict(second_evidence, payload_checksum="wrong")
        with pytest.raises(RuntimeError, match="pin manifest evidence mismatch"):
            client.pin(
                kind="materialized",
                source_rank=0,
                epoch=2,
                control_checksum="2" * 64,
                expected_manifest=tampered,
            )
        client.pin(
            kind="materialized",
            source_rank=0,
            epoch=2,
            control_checksum="2" * 64,
            expected_manifest=second_evidence,
        )
        client.seal(
            kind="materialized",
            source_rank=0,
            epoch=2,
            control_checksum="2" * 64,
        )
        assert client.wait_materialized(0, 2) == 2
        assert store.stats()["record_count"] == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_sealed_shared_tensor_holds_epoch_lease_until_materialized(tmp_path):
    module, optimizer, scheduler = _objects()
    identifiers = half_precision_parameter_ids(module, optimizer)
    binding = RecoveryBinding(0, 2, 0, 0, 0)
    store = ServiceReplicaStore(
        max_bytes=64 * 1024 * 1024,
        materialize_incremental=True,
        sealed_incremental=True,
    )
    tcp_server = ReplicaTCPServer(("127.0.0.1", 0), store)
    tcp_thread = threading.Thread(target=tcp_server.serve_forever, daemon=True)
    tcp_thread.start()
    client = ReplicaServiceClient("127.0.0.1", tcp_server.server_address[1])
    socket_path = tmp_path / "sealed-shared.sock"
    shared_server = SharedTensorServer(socket_path, store)
    shared_server.start()
    try:
        base_state = _base_state(module, optimizer, scheduler, identifiers, binding)
        base_state["runtime_state"] = {"iteration": 0, "rank": 0}
        serialized = serialize_checkpoint(base_state)
        prepared = create_prepared_manifest(
            epoch=0,
            source_rank=0,
            source_host=socket.gethostname(),
            target_rank=0,
            target_host=client.endpoint,
            group_ranks=(0, 1),
            serialized=serialized,
        )
        client.put_prepared(ReplicaEnvelope(prepared, serialized.payload))
        committed_base = replace(
            prepared, state="committed", committed_time_ns=time.time_ns()
        )
        client.commit(committed_base)
        client.pin(
            kind="materialized",
            source_rank=0,
            epoch=0,
            control_checksum="0" * 64,
            expected_manifest=manifest_evidence(committed_base),
        )
        client.seal(
            kind="materialized",
            source_rank=0,
            epoch=0,
            control_checksum="0" * 64,
        )

        session = SharedTensorCheckpointSession(
            rank=0,
            group_ranks=(0, 1),
            target_rank=0,
            socket_path=socket_path,
            manifest_root=tmp_path / "sealed-manifests",
            max_shared_slots=2,
        )
        update = _update_state(
            optimizer,
            scheduler,
            identifiers,
            binding,
            f"sha256:{committed_base.payload_checksum}",
            1,
        )
        result = session.save(
            1, update, state_is_snapshot=True, object_checksums=False
        )
        assert store.stats()["materialized_epochs"] == {"0": 0}
        assert store.stats()["staged_materialized_epochs"] == [[0, 1]]
        assert list(session._leased_shared_states) == [1]

        evidence = manifest_evidence(result.manifest)
        client.pin(
            kind="materialized",
            source_rank=0,
            epoch=1,
            control_checksum="1" * 64,
            expected_manifest=evidence,
        )
        client.seal(
            kind="materialized",
            source_rank=0,
            epoch=1,
            control_checksum="1" * 64,
        )
        session.wait_applied(1)
        assert not session._leased_shared_states
        assert store.stats()["materialized_epochs"] == {"0": 1}

        optimizer.optimizer.step()
        optimizer.refresh_model_params()
        scheduler.step()
        second = session.save(
            2,
            _update_state(
                optimizer,
                scheduler,
                identifiers,
                binding,
                f"sha256:{committed_base.payload_checksum}",
                2,
            ),
            state_is_snapshot=True,
            object_checksums=False,
        )
        second_evidence = manifest_evidence(second.manifest)
        client.pin(
            kind="materialized",
            source_rank=0,
            epoch=2,
            control_checksum="2" * 64,
            expected_manifest=second_evidence,
        )
        client.seal(
            kind="materialized",
            source_rank=0,
            epoch=2,
            control_checksum="2" * 64,
        )
        session.wait_applied(2)
        apply = store.stats()["materializer_apply"]["0"]
        assert [item["epoch"] for item in apply["events"]] == [1, 2]
        assert apply["count"] == 2
        assert all(
            item["started_monotonic_ns"] <= item["finished_monotonic_ns"]
            and item["duration_ns"] >= 0
            for item in apply["events"]
        )
        assert apply["last_finished_monotonic_ns"] >= (
            apply["first_started_monotonic_ns"]
        )
        assert store.stats()["materialized_epochs"] == {"0": 2}
    finally:
        shared_server.stop()
        tcp_server.shutdown()
        tcp_server.server_close()
        tcp_thread.join(timeout=5)


def test_shared_tensor_crosses_process_auth_boundary(tmp_path):
    socket_path = str(tmp_path / "cross-process.sock")
    ready_path = tmp_path / "cross-process.ready"
    process = Process(target=_receive_shared_tensor, args=(socket_path, ready_path))
    process.start()
    original_authkey = current_process().authkey
    try:
        for _ in range(100):
            if ready_path.exists():
                break
            time.sleep(0.01)
        assert ready_path.exists()
        current_process().authkey = b"client-default"
        _configure_tensor_sharing()
        tensor = torch.arange(64, dtype=torch.float32).share_memory_()
        connection = Client(
            socket_path, family="AF_UNIX", authkey=_authkey(socket_path)
        )
        connection.send(tensor)
        assert connection.recv() == ((64,), 2016.0)
        connection.close()
    finally:
        current_process().authkey = original_authkey
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert process.exitcode == 0


def _capture_error(errors, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except Exception as error:
        errors.append(error)
