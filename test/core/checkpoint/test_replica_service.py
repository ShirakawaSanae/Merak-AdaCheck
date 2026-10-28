from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading

import pytest
import torch
import Merak.core.checkpoint.replica_service as replica_service_module

from Merak.core.checkpoint.catalog import serialize_checkpoint
from Merak.core.checkpoint.manifest import create_prepared_manifest
from Merak.core.checkpoint.global_commit import manifest_evidence
from Merak.core.checkpoint.recovery import ServiceRemoteCheckpointSession
from Merak.core.checkpoint.reduced import (
    ReducedCheckpoint,
    build_reduced_checkpoint,
)
from Merak.core.checkpoint.replica_service import (
    ReplicaServiceClient,
    ReplicaTCPServer,
    ServiceReplicaStore,
)
from Merak.core.checkpoint.transport import ReplicaEnvelope


def _prepared(epoch: int = 3, source_rank: int = 0) -> ReplicaEnvelope:
    serialized = serialize_checkpoint(
        {
            "model": torch.arange(16, dtype=torch.float32),
            "optimizer": {"step": torch.tensor(epoch, dtype=torch.int64)},
        }
    )
    manifest = create_prepared_manifest(
        epoch=epoch,
        source_rank=source_rank,
        source_host="source",
        target_rank=1,
        target_host="holder",
        group_ranks=(0, 1),
        serialized=serialized,
    )
    return ReplicaEnvelope(manifest, serialized.payload)


@contextmanager
def _running_service():
    store = ServiceReplicaStore(max_bytes=16 * 1024 * 1024)
    server = ReplicaTCPServer(("127.0.0.1", 0), store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield ReplicaServiceClient(host, port)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def service():
    with _running_service() as client:
        yield client


@pytest.fixture
def mirror_service():
    with _running_service() as client:
        yield client


def test_service_requires_commit_before_get_and_reports_protocol_bytes(service):
    prepared = _prepared()
    service.health()
    put_metrics = service.put_prepared(prepared)

    with pytest.raises(RuntimeError, match="is not committed"):
        service.get(0, 3)

    committed = replace(
        prepared.manifest, state="committed", committed_time_ns=123
    )
    commit_metrics = service.commit(committed)
    restored, get_metrics = service.get(0, 3)
    stats = service.stats()

    assert restored.manifest == committed
    assert restored.payload == prepared.payload
    assert put_metrics.wire_bytes > len(prepared.payload)
    assert commit_metrics.wire_bytes > 0
    assert get_metrics.wire_bytes > len(prepared.payload)
    assert stats["bytes_used"] == len(prepared.payload)
    assert stats["record_count"] == 1
    assert stats["protocol_rx_bytes"] > len(prepared.payload)
    assert stats["protocol_tx_bytes"] > len(prepared.payload)


def test_stats_exposes_request_metrics_for_counter_reconciliation(service):
    before, before_metrics = service.stats_with_metrics()
    after, after_metrics = service.stats_with_metrics()

    assert (
        after["protocol_rx_bytes"] - before["protocol_rx_bytes"]
        == after_metrics.request_bytes
    )
    assert (
        after["protocol_tx_bytes"] - before["protocol_tx_bytes"]
        == before_metrics.response_bytes
    )


def test_service_accepts_rank_burst_health_checks(service):
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [executor.submit(service.health) for _ in range(32)]

    for future in futures:
        assert future.result() is None


def test_service_serializes_materialized_responses():
    store = ServiceReplicaStore(max_bytes=16 * 1024 * 1024)
    server = ReplicaTCPServer(("127.0.0.1", 0), store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = ReplicaServiceClient(*server.server_address)
    envelope = _prepared()
    entered = threading.Event()
    release = threading.Event()
    active = 0
    maximum_active = 0
    counter_lock = threading.Lock()

    def get_materialized(*args, **kwargs):
        nonlocal active, maximum_active
        with counter_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        entered.set()
        assert release.wait(timeout=2)
        with counter_lock:
            active -= 1
        return envelope

    store.get_materialized = get_materialized
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(client.get_materialized, 0, 3)
            assert entered.wait(timeout=2)
            second = executor.submit(client.get_materialized, 0, 3)
            release.set()
            first.result(timeout=2)
            second.result(timeout=2)
        assert maximum_active == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_service_materialized_response_worker_limit_is_configurable():
    store = ServiceReplicaStore(max_bytes=16 * 1024 * 1024)
    server = ReplicaTCPServer(
        ("127.0.0.1", 0), store, materialized_response_workers=2
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = ReplicaServiceClient(*server.server_address)
    envelope = _prepared()
    entered = threading.Event()
    release = threading.Event()
    active = 0
    maximum_active = 0
    counter_lock = threading.Lock()

    def get_materialized(*args, **kwargs):
        nonlocal active, maximum_active
        with counter_lock:
            active += 1
            maximum_active = max(maximum_active, active)
            if active == 2:
                entered.set()
        assert release.wait(timeout=2)
        with counter_lock:
            active -= 1
        return envelope

    store.get_materialized = get_materialized
    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(client.get_materialized, rank, 3)
                for rank in range(3)
            ]
            assert entered.wait(timeout=2)
            release.set()
            for future in futures:
                future.result(timeout=2)
        assert maximum_active == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_service_rejects_corrupt_payload(service):
    prepared = _prepared()
    corrupt = ReplicaEnvelope(
        prepared.manifest,
        prepared.payload[:-1] + bytes([prepared.payload[-1] ^ 0xFF]),
    )

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        service.put_prepared(corrupt)


def test_service_session_commits_manifest_and_restores_without_local_cache(
    service, tmp_path
):
    state = {
        "model": {"weight": torch.arange(8, dtype=torch.float32)},
        "optimizer": {"step": torch.tensor(7), "momentum": torch.ones(8)},
        "rng": torch.arange(4, dtype=torch.uint8),
    }
    session = ServiceRemoteCheckpointSession(
        rank=0,
        group_ranks=(0, 1),
        service_endpoint=service.endpoint,
        manifest_root=tmp_path,
        max_local_store_bytes=1,
    )

    result = session.save(7, state)
    assert result.manifest.committed
    assert result.manifest.replica_count == 1
    assert result.manifest.tolerated_failures == 0
    assert result.transport.payload_bytes == result.manifest.payload_bytes
    assert result.transport.wire_bytes > result.transport.payload_bytes
    assert result.store_bytes == result.manifest.payload_bytes
    assert session.store.bytes_used == 0
    assert session.repository.committed_path(result.manifest).is_file()
    assert session.repository.replica_path(1, result.manifest).is_file()

    restored = session.restore_from_peer(7)
    assert torch.equal(restored["model"]["weight"], state["model"]["weight"])
    assert torch.equal(restored["optimizer"]["momentum"], state["optimizer"]["momentum"])
    assert torch.equal(restored["rng"], state["rng"])
    assert session.last_restore_wire_bytes > result.manifest.payload_bytes
    assert session.store.bytes_used == 0


def test_service_session_can_skip_local_observability_writes(service, tmp_path):
    session = ServiceRemoteCheckpointSession(
        rank=0,
        group_ranks=(0, 1),
        service_endpoint=service.endpoint,
        manifest_root=tmp_path,
        max_local_store_bytes=1,
        record_manifests=False,
        collect_store_stats=False,
    )

    result = session.save(1, {"model": torch.arange(4)})

    assert result.manifest.committed
    assert result.store_bytes == -1
    assert not any(tmp_path.iterdir())
    assert torch.equal(session.restore_from_peer(1)["model"], torch.arange(4))


def test_service_session_can_target_its_local_rank(service, tmp_path):
    session = ServiceRemoteCheckpointSession(
        rank=0,
        group_ranks=(0, 1),
        target_rank=0,
        service_endpoint=service.endpoint,
        manifest_root=tmp_path,
        max_local_store_bytes=1,
    )

    result = session.save(2, {"model": torch.arange(2)})

    assert result.manifest.target_rank == 0
    assert session.repository.replica_path(0, result.manifest).is_file()


def test_service_session_requires_remote_and_local_service_commit(
    service, mirror_service, tmp_path
):
    state = {
        "model": {"weight": torch.arange(8, dtype=torch.float32)},
        "optimizer": {"step": torch.tensor(7)},
    }
    session = ServiceRemoteCheckpointSession(
        rank=0,
        group_ranks=(0, 1),
        service_endpoint=service.endpoint,
        mirror_service_endpoint=mirror_service.endpoint,
        manifest_root=tmp_path,
        max_local_store_bytes=1,
    )

    result = session.save(7, state)
    remote, _ = service.get(0, 7)
    local, _ = mirror_service.get(0, 7)

    assert remote.manifest.committed
    assert result.manifest.replica_count == 2
    assert result.manifest.tolerated_failures == 1
    assert local.manifest == remote.manifest
    assert local.payload == remote.payload
    assert result.transport.payload_bytes == result.manifest.payload_bytes
    assert result.transport.wire_bytes > 2 * result.transport.payload_bytes
    assert service.stats()["record_count"] == 1
    assert mirror_service.stats()["record_count"] == 1


def test_committed_replica_is_idempotent_but_cannot_be_replaced(service):
    prepared = _prepared()
    service.put_prepared(prepared)
    committed = replace(
        prepared.manifest, state="committed", committed_time_ns=123
    )
    service.commit(committed)

    service.put_prepared(prepared)
    service.commit(committed)
    with pytest.raises(RuntimeError, match="commit decision does not match"):
        service.commit(replace(committed, target_rank=7))

    replacement = ReplicaEnvelope(
        replace(prepared.manifest, manifest_id="replacement"), prepared.payload
    )
    with pytest.raises(RuntimeError, match="refusing to replace committed replica"):
        service.put_prepared(replacement)


def test_commit_retry_converges_when_phase_timestamps_change(service):
    first = _prepared()
    service.put_prepared(first)
    service.commit(
        replace(first.manifest, state="committed", committed_time_ns=123)
    )

    retried_manifest = replace(
        first.manifest, prepared_time_ns=first.manifest.prepared_time_ns + 1
    )
    service.put_prepared(ReplicaEnvelope(retried_manifest, first.payload))
    retried_commit = replace(
        retried_manifest, state="committed", committed_time_ns=456
    )
    service.commit(retried_commit)

    restored, _ = service.get(0, 3)
    assert restored.manifest == retried_commit


def test_sealed_record_retention_keeps_unpublished_intermediate_epoch():
    store = ServiceReplicaStore(
        max_bytes=16 * 1024 * 1024,
        retained_epochs_per_source=2,
        materialize_incremental=True,
        sealed_incremental=True,
    )
    controls = [str(epoch) * 64 for epoch in range(3)]
    commits = []
    for epoch in range(3):
        prepared = _prepared(epoch=epoch)
        store.put_prepared(prepared)
        committed = replace(
            prepared.manifest, state="committed", committed_time_ns=epoch + 1
        )
        store.commit(committed)
        commits.append(committed)
        if epoch == 0:
            store.pin(
                kind="record",
                source_rank=0,
                epoch=0,
                control_checksum=controls[0],
                expected_manifest=manifest_evidence(committed),
            )
            store.seal(
                kind="record",
                source_rank=0,
                epoch=0,
                control_checksum=controls[0],
            )

    assert [item["epoch"] for item in store.stats()["records"]] == [0, 1, 2]
    store.pin(
        kind="record",
        source_rank=0,
        epoch=1,
        control_checksum=controls[1],
        expected_manifest=manifest_evidence(commits[1]),
    )
    store.seal(
        kind="record",
        source_rank=0,
        epoch=1,
        control_checksum=controls[1],
    )
    assert [item["epoch"] for item in store.stats()["records"]] == [1, 2]


def test_capacity_cannot_evict_another_source_during_commit(monkeypatch):
    first = _prepared(source_rank=0)
    second = _prepared(source_rank=1)
    store = ServiceReplicaStore(
        max_bytes=len(first.payload), materialize_incremental=True
    )
    store.put_prepared(first)
    entered = threading.Event()
    release = threading.Event()
    errors = []
    original_deserialize = replica_service_module.deserialize_checkpoint

    def blocked_deserialize(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return original_deserialize(*args, **kwargs)

    def commit_first():
        try:
            store.commit(
                replace(first.manifest, state="committed", committed_time_ns=123)
            )
        except Exception as error:
            errors.append(error)

    monkeypatch.setattr(
        replica_service_module, "deserialize_checkpoint", blocked_deserialize
    )
    commit_thread = threading.Thread(target=commit_first)
    commit_thread.start()
    assert entered.wait(timeout=2)

    with pytest.raises(MemoryError):
        store.put_prepared(second)
    release.set()
    commit_thread.join(timeout=2)

    assert not commit_thread.is_alive()
    assert not errors
    assert store.get_committed(0, 3) == ReplicaEnvelope(
        replace(first.manifest, state="committed", committed_time_ns=123),
        first.payload,
    )


def test_sealed_ingest_does_not_wait_for_materializer_source_lock():
    store = ServiceReplicaStore(
        max_bytes=16 * 1024 * 1024,
        materialize_incremental=True,
        sealed_incremental=True,
    )
    prepared = _prepared()
    source_lock = store._source_lock(prepared.manifest.source_rank)
    source_lock.acquire()
    errors = []

    def put():
        try:
            store.put_prepared(prepared)
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=put)
    try:
        thread.start()
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert not errors
    finally:
        source_lock.release()
        thread.join(timeout=1)


def test_reduced_wire_state_commits_and_reconstructs(service, tmp_path):
    states = {
        0: {"model": {"weight": torch.arange(8)}, "rank": torch.tensor([0])},
        1: {"model": {"weight": torch.arange(8)}, "rank": torch.tensor([1])},
    }
    reduced = build_reduced_checkpoint(states)
    session = ServiceRemoteCheckpointSession(
        rank=0,
        group_ranks=(0, 1),
        service_endpoint=service.endpoint,
        manifest_root=tmp_path,
        max_local_store_bytes=1,
    )

    commit = session.save(5, reduced.to_wire_state())
    restored = ReducedCheckpoint.from_wire_state(session.restore_from_peer(5))

    assert commit.manifest.logical_bytes == reduced.reduced_logical_bytes
    assert len(commit.manifest.objects) == len(reduced.objects)
    for rank, expected in states.items():
        actual = restored.reconstruct_rank(rank)
        assert serialize_checkpoint(actual).objects == serialize_checkpoint(expected).objects
