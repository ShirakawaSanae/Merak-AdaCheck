from contextlib import ExitStack, contextmanager
from dataclasses import replace
import threading

import pytest
import torch

from Merak.core.checkpoint.catalog import deserialize_checkpoint, serialize_checkpoint
from Merak.core.checkpoint.global_commit import manifest_evidence
from Merak.core.checkpoint.manifest import create_prepared_manifest
from Merak.core.checkpoint.object_replica_service import ObjectReplicaServiceStore
from Merak.core.checkpoint.replica_service import ReplicaServiceClient, ReplicaTCPServer
from Merak.core.checkpoint.transport import ReplicaEnvelope


@contextmanager
def _service(max_bytes=8 << 20):
    store = ObjectReplicaServiceStore(
        max_bytes=max_bytes,
        retained_epochs_per_source=1,
        sealed_incremental=True,
    )
    server = ReplicaTCPServer(("127.0.0.1", 0), store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield ReplicaServiceClient(*server.server_address), server, thread
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _commit(client, *, epoch, source_rank, state):
    serialized = serialize_checkpoint(state)
    prepared = create_prepared_manifest(
        epoch=epoch,
        source_rank=source_rank,
        source_host=f"rank-{source_rank}",
        target_rank=source_rank,
        target_host=client.endpoint,
        group_ranks=(source_rank, source_rank),
        serialized=serialized,
        replica_count=2,
        tolerated_failures=1,
    )
    client.put_prepared(ReplicaEnvelope(prepared, serialized.payload))
    committed = replace(prepared, state="committed", committed_time_ns=epoch)
    client.commit(committed)
    return committed, serialized.payload_checksum


def _seal_epoch(client, epoch, state=None):
    if state is None:
        state = {
            "kind": "semantic_gradient_bundle",
            "epoch": epoch,
            "gradient": torch.tensor([float(epoch)]),
        }
    data, _ = _commit(
        client,
        epoch=epoch,
        source_rank=0,
        state=state,
    )
    control, checksum = _commit(
        client,
        epoch=epoch,
        source_rank=-1,
        state={"kind": "object_recovery_control", "epoch": epoch,
               "data_manifest": manifest_evidence(data)},
    )
    client.pin(
        kind="record",
        source_rank=control.source_rank,
        epoch=epoch,
        control_checksum=checksum,
        expected_manifest=manifest_evidence(control),
    )
    client.seal(
        kind="record",
        source_rank=control.source_rank,
        epoch=epoch,
        control_checksum=checksum,
    )
    client.pin(
        kind="record",
        source_rank=data.source_rank,
        epoch=epoch,
        control_checksum=checksum,
        expected_manifest=manifest_evidence(data),
    )
    client.seal(
        kind="record",
        source_rank=data.source_rank,
        epoch=epoch,
        control_checksum=checksum,
    )
    return data, control, checksum


def _seal_mirrored_epoch(services, epoch):
    data = [
        _commit(
            client,
            epoch=epoch,
            source_rank=0,
            state={"kind": "semantic_gradient_bundle", "epoch": epoch,
                   "gradient": torch.tensor([float(epoch)])},
        )[0]
        for client, _, _ in services
    ]
    control_state = {
        "kind": "object_recovery_control",
        "epoch": epoch,
        "data_manifests": [manifest_evidence(value) for value in data],
    }
    controls = [
        _commit(client, epoch=epoch, source_rank=-1, state=control_state)
        for client, _, _ in services
    ]
    checksums = {checksum for _, checksum in controls}
    assert len(checksums) == 1
    checksum = checksums.pop()
    for host, (client, _, _) in enumerate(services):
        control = controls[host][0]
        client.pin(
            kind="record",
            source_rank=control.source_rank,
            epoch=epoch,
            control_checksum=checksum,
            expected_manifest=manifest_evidence(control),
        )
        client.seal(
            kind="record",
            source_rank=control.source_rank,
            epoch=epoch,
            control_checksum=checksum,
        )
        manifest = data[host]
        client.pin(
            kind="record",
            source_rank=manifest.source_rank,
            epoch=epoch,
            control_checksum=checksum,
            expected_manifest=manifest_evidence(manifest),
        )
        client.seal(
            kind="record",
            source_rank=manifest.source_rank,
            epoch=epoch,
            control_checksum=checksum,
        )
    return data, [value[0] for value in controls], checksum


def test_two_services_keep_sealed_history_and_one_survivor_recovers():
    for failed_host in (0, 1):
        with ExitStack() as stack:
            services = [stack.enter_context(_service()) for _ in range(2)]
            locators = []
            for epoch in (1, 2):
                data, control, checksum = _seal_mirrored_epoch(services, epoch)
                locators.append(
                    {
                        "epoch": epoch,
                        "control_checksum": checksum,
                        "control_evidence": [manifest_evidence(value) for value in control],
                        "data_evidence": [manifest_evidence(value) for value in data],
                    }
                )

            _, failed_server, failed_thread = services[failed_host]
            failed_server.shutdown()
            failed_server.server_close()
            failed_thread.join(timeout=5)
            survivor_host = 1 - failed_host
            surviving = services[survivor_host][0]

            for locator in locators:
                control_envelope, _ = surviving.get(
                    -1,
                    locator["epoch"],
                    control_checksum=locator["control_checksum"],
                    expected_manifest=locator["control_evidence"][survivor_host],
                )
                restored_control = deserialize_checkpoint(
                    control_envelope.payload, control_envelope.manifest.payload_checksum
                )
                assert restored_control["data_manifests"] == locator["data_evidence"]
                data_envelope, _ = surviving.get(
                    0,
                    locator["epoch"],
                    control_checksum=locator["control_checksum"],
                    expected_manifest=locator["data_evidence"][survivor_host],
                )
                restored = deserialize_checkpoint(
                    data_envelope.payload, data_envelope.manifest.payload_checksum
                )
                assert restored["epoch"] == locator["epoch"]

            assert surviving.stats()["object_sealed_epochs"] == [
                ["record", -1, 1], ["record", -1, 2],
                ["record", 0, 1], ["record", 0, 2],
            ]


def test_object_service_rejects_unsealed_or_wrong_recovery_evidence():
    with _service() as (client, _, _):
        unsealed, _ = _commit(
            client, epoch=1, source_rank=0, state={"value": torch.ones(1)}
        )
        with pytest.raises(RuntimeError, match="not sealed"):
            client.get(0, 1, expected_manifest=manifest_evidence(unsealed))

    with _service() as (client, _, _):
        data, _, checksum = _seal_epoch(client, 1)
        with pytest.raises(RuntimeError, match="not sealed"):
            client.get(0, 1, expected_manifest=manifest_evidence(data))
        with pytest.raises(RuntimeError, match="not sealed"):
            client.get(
                0, 1, control_checksum="f" * 64,
                expected_manifest=manifest_evidence(data),
            )
        wrong_manifest = {**manifest_evidence(data), "payload_bytes": 0}
        with pytest.raises(RuntimeError, match="evidence mismatch"):
            client.get(
                0, 1, control_checksum=checksum, expected_manifest=wrong_manifest
            )
        with pytest.raises(RuntimeError, match="requires manifest evidence"):
            client.get(0, 1, control_checksum=checksum)

    with _service() as (client, _, _):
        data, _ = _commit(client, epoch=1, source_rank=0, state={"value": 1})
        control, checksum = _commit(
            client,
            epoch=1,
            source_rank=-1,
            state={"kind": "object_recovery_control", "epoch": 1},
        )
        with pytest.raises(RuntimeError, match="payload checksum"):
            client.pin(
                kind="record",
                source_rank=-1,
                epoch=1,
                control_checksum="f" * 64,
                expected_manifest=manifest_evidence(control),
            )
        client.pin(
            kind="record",
            source_rank=0,
            epoch=1,
            control_checksum=checksum,
            expected_manifest=manifest_evidence(data),
        )
        with pytest.raises(RuntimeError, match="requires its sealed control"):
            client.seal(
                kind="record",
                source_rank=0,
                epoch=1,
                control_checksum=checksum,
            )


def test_sealed_history_is_not_evicted_when_capacity_is_exhausted():
    first = serialize_checkpoint(
        {"payload": torch.zeros(16 << 10, dtype=torch.uint8)}
    )
    state = {"payload": torch.zeros(16 << 10, dtype=torch.uint8)}
    with _service(max_bytes=2 * len(first.payload)) as (client, _, _):
        manifest, _, checksum = _seal_epoch(client, 1, state)

        with pytest.raises(RuntimeError, match="cannot admit"):
            _commit(
                client,
                epoch=2,
                source_rank=0,
                state={"payload": torch.ones(16 << 10, dtype=torch.uint8)},
            )
        restored, _ = client.get(
            0, 1, control_checksum=checksum,
            expected_manifest=manifest_evidence(manifest),
        )
        assert restored.payload == first.payload
