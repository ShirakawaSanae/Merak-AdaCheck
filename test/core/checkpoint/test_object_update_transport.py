from contextlib import ExitStack, contextmanager
import hashlib
import threading

import pytest
import torch

from Merak.core.checkpoint.incremental_checkpoint import (
    RecoveryBinding,
    apply_optimizer_update,
    capture_optimizer_update,
)
from Merak.core.checkpoint.object_optimizer_update import (
    externalize_local_optimizer_update,
    externalize_owned_snapshot_optimizer_update,
)
from Merak.core.checkpoint.matched import CONTROL_SOURCE_RANK
from Merak.core.checkpoint.object_replica_service import ObjectReplicaServiceStore
from Merak.core.checkpoint.object_semantic_plan import (
    GradientShard,
    RankBinding,
    build_active_gradient_plan,
    confirm_gradient_equivalence,
)
from Merak.core.checkpoint.object_semantic_schema import recover_gradient_encoding
from Merak.core.checkpoint.object_update_transport import (
    ObjectUpdateLocator,
    build_local_object_update_bundles,
    fetch_object_update_control,
    publish_object_update_epoch,
    restore_object_update_epoch,
)
from Merak.core.checkpoint.catalog import serialize_checkpoint
from Merak.core.checkpoint.replica_service import ReplicaServiceClient, ReplicaTCPServer


BASE_ID = "sha256:" + hashlib.sha256(b"base").hexdigest()


class _SpyClient:
    def __init__(self, client):
        self.client = client
        self.get_source_ranks = []

    def get(self, source_rank, *args, **kwargs):
        self.get_source_ranks.append(source_rank)
        return self.client.get(source_rank, *args, **kwargs)


@contextmanager
def _services():
    with ExitStack() as stack:
        services = []
        for _ in range(2):
            store = ObjectReplicaServiceStore(32 << 20, retained_epochs_per_source=1)
            server = ReplicaTCPServer(("127.0.0.1", 0), store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            stack.callback(thread.join, timeout=5)
            stack.callback(server.server_close)
            stack.callback(server.shutdown)
            services.append((ReplicaServiceClient(*server.server_address), server))
        yield services


def _optimizer(parameters):
    return torch.optim.AdamW(
        parameters,
        lr=0.003,
        betas=(0.8, 0.95),
        weight_decay=0.1,
        foreach=False,
        fused=False,
    )


def _epoch(arm, epoch, *, mismatch=False):
    bindings = tuple(
        RankBinding(rank, 0, 0, rank, rank // 2) for rank in range(4)
    )
    shards = tuple(
        GradientShard(
            rank,
            name,
            f"detector-{name}",
            (2,),
            (0,),
            (2,),
            "torch.float32",
        )
        for name in ("left", "right")
        for rank in (0, 1)
    )
    plan = build_active_gradient_plan(
        shards,
        bindings,
        arm=arm,
        detector_catalog_hash="a" * 64,
        semantic_plan_hash="b" * 64,
        fixed_pairs=((0, 2), (1, 3)),
    )
    values = {
        (0, "left"): torch.tensor([0.25 * epoch, -0.5]),
        (1, "left"): torch.tensor([0.25 * epoch, -0.5]),
        (0, "right"): torch.tensor([-0.75, 0.125 * epoch]),
        (1, "right"): torch.tensor([-0.75, 0.125 * epoch]),
    }
    if mismatch:
        values[(1, "left")].add_(1.0)
        values[(1, "right")].sub_(1.0)
    object_by_name = {item.fqn: item for item in plan.objects}
    encoding = confirm_gradient_equivalence(
        plan,
        {
            (rank, object_by_name[name].object_id): gradient
            for (rank, name), gradient in values.items()
        },
        epoch=epoch,
    )
    records = {}
    local_updates = {}
    for rank in range(4):
        parameters = [
            torch.nn.Parameter(torch.tensor([1.0, -2.0])),
            torch.nn.Parameter(torch.tensor([3.0, -4.0])),
        ]
        optimizer = _optimizer(parameters)
        for name, parameter in zip(("left", "right"), parameters):
            gradient = values.get((rank, name))
            parameter.grad = None if gradient is None else gradient.clone()
        record = capture_optimizer_update(
            optimizer,
            parameter_ids=(("left", "right"),),
            binding=RecoveryBinding(rank, 4, rank, 0, 0),
            base_id=BASE_ID,
            base_epoch=0,
            epoch=epoch,
        )
        records[rank] = record
        local_updates[rank] = externalize_local_optimizer_update(
            plan, encoding, rank, record
        )
    return plan, encoding, records, local_updates


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


def _target_sources(plan, encoding, logical_rank, surviving_host):
    actions = {}
    for action in encoding.actions:
        actions.setdefault(
            (
                action.object_id,
                action.source_rank if action.action == "direct_fallback" else None,
            ),
            action,
        )
    publishers = set()
    for item in plan.objects:
        if logical_rank not in item.holders:
            continue
        referenced = actions[
            (
                item.object_id,
                logical_rank if item.object_id in encoding.mismatched_object_ids else None,
            )
        ]
        publishers.add(
            next(
                action.source_rank
                for action in encoding.actions
                if action.object_id == item.object_id
                and action.target_host == surviving_host
                and action.version_id == referenced.version_id
                and action.action == referenced.action
                and (
                    referenced.action != "direct_fallback"
                    or action.source_rank == referenced.source_rank
                )
            )
        )
    return {logical_rank} | publishers


@pytest.mark.parametrize(
    "arm", ("adacheck_fixed", "host_global_fixed", "m1")
)
@pytest.mark.parametrize("failed_host", (0, 1))
def test_two_epoch_sealed_host_loss_restores_and_replays_adamw(arm, failed_host):
    with _services() as services:
        clients = {host: service[0] for host, service in enumerate(services)}
        epochs = []
        for epoch in (1, 2):
            plan, encoding, records, updates = _epoch(arm, epoch)
            published = publish_object_update_epoch(
                clients,
                run_id="cpu-gate",
                plan=plan,
                encoding=encoding,
                local_updates=updates,
            )
            epochs.append((plan, encoding, records, published.locator))

        services[failed_host][1].shutdown()
        survivor = 1 - failed_host
        restored_epochs = []
        for plan, encoding, records, locator in epochs:
            control = fetch_object_update_control(
                clients[survivor], surviving_host=survivor, locator=locator
            )
            assert recover_gradient_encoding(
                plan, control.value["gradient_encoding"], epoch=encoding.epoch
            ) == encoding
            restored = restore_object_update_epoch(
                clients[survivor],
                surviving_host=survivor,
                locator=locator,
                plan=plan,
                encoding=encoding,
            )
            restored_records = dict(restored.records)
            for rank in range(4):
                _nested_equal(restored_records[rank], records[rank])
            restored_epochs.append(restored_records)

        for rank in range(4):
            direct_parameters = [
                torch.nn.Parameter(torch.tensor([1.0, -2.0])),
                torch.nn.Parameter(torch.tensor([3.0, -4.0])),
            ]
            restored_parameters = [
                torch.nn.Parameter(torch.tensor([1.0, -2.0])),
                torch.nn.Parameter(torch.tensor([3.0, -4.0])),
            ]
            direct_optimizer = _optimizer(direct_parameters)
            restored_optimizer = _optimizer(restored_parameters)
            for epoch, (_, _, records, _) in enumerate(epochs, start=1):
                arguments = dict(
                    expected_parameter_ids=(("left", "right"),),
                    expected_binding=RecoveryBinding(rank, 4, rank, 0, 0),
                    expected_base_id=BASE_ID,
                    expected_base_epoch=0,
                    expected_epoch=epoch,
                )
                apply_optimizer_update(direct_optimizer, records[rank], **arguments)
                apply_optimizer_update(
                    restored_optimizer, restored_epochs[epoch - 1][rank], **arguments
                )
            for direct, recovered in zip(direct_parameters, restored_parameters):
                assert torch.equal(direct, recovered)
            _nested_equal(
                direct_optimizer.state_dict(), restored_optimizer.state_dict()
            )


def test_three_arms_preserve_baseline_occurrences_and_m1_balances_publishers():
    results = {}
    for arm in ("adacheck_fixed", "host_global_fixed", "m1"):
        with _services() as services:
            clients = {host: service[0] for host, service in enumerate(services)}
            plan, encoding, _, updates = _epoch(arm, 1)
            published = publish_object_update_epoch(
                clients,
                run_id=f"cpu-gate-{arm}",
                plan=plan,
                encoding=encoding,
                local_updates=updates,
            )
            results[arm] = (plan, encoding, published.metrics)

    ada_plan, ada_encoding, ada_metrics = results["adacheck_fixed"]
    fixed_plan, fixed_encoding, fixed_metrics = results["host_global_fixed"]
    m1_plan, m1_encoding, m1_metrics = results["m1"]
    assert ada_encoding.total_remote_logical_bytes == 32
    assert fixed_encoding.total_remote_logical_bytes == 16
    assert m1_encoding.total_remote_logical_bytes == 16
    assert ada_metrics.remote_data_payload_bytes > m1_metrics.remote_data_payload_bytes
    assert fixed_metrics.remote_gradient_logical_bytes == m1_metrics.remote_gradient_logical_bytes
    assert fixed_plan.maximum_remote_publisher_bytes == 16
    assert m1_plan.maximum_remote_publisher_bytes < fixed_plan.maximum_remote_publisher_bytes
    assert (
        m1_plan.maximum_all_publication_publisher_bytes
        < fixed_plan.maximum_all_publication_publisher_bytes
    )
    assert ada_plan.canonical_hash == fixed_plan.canonical_hash == m1_plan.canonical_hash


@pytest.mark.parametrize("arm", ("adacheck_fixed", "host_global_fixed", "m1"))
def test_owned_payload_bundles_match_default_hashes_bytes_and_recovery(arm):
    plan, encoding, records, default_updates = _epoch(arm, 1)
    owned_updates = {
        rank: externalize_owned_snapshot_optimizer_update(plan, encoding, rank, record)
        for rank, record in records.items()
    }
    for rank in records:
        default_bundles = build_local_object_update_bundles(
            plan, encoding, default_updates[rank]
        )
        owned_bundles = build_local_object_update_bundles(
            plan, encoding, owned_updates[rank], owned_payloads=True
        )
        for default, owned in zip(default_bundles, owned_bundles):
            default_serialized = serialize_checkpoint(default, state_is_snapshot=True)
            owned_serialized = serialize_checkpoint(owned, state_is_snapshot=True)
            assert serialize_checkpoint(
                owned, state_is_snapshot=True
            ).payload_checksum == owned_serialized.payload_checksum
            assert owned_serialized.objects == default_serialized.objects
            assert owned_serialized.logical_bytes == default_serialized.logical_bytes
            assert len(owned_serialized.payload) == len(default_serialized.payload)

    with _services() as services:
        clients = {host: service[0] for host, service in enumerate(services)}
        published = publish_object_update_epoch(
            clients,
            run_id=f"owned-{arm}",
            plan=plan,
            encoding=encoding,
            local_updates=owned_updates,
            owned_payloads=True,
        )
        for surviving_host in (0, 1):
            restored = restore_object_update_epoch(
                clients[surviving_host],
                surviving_host=surviving_host,
                locator=published.locator,
                plan=plan,
                encoding=encoding,
            )
            for rank, restored_record in restored.records:
                _nested_equal(restored_record, records[rank])


def test_default_bundles_isolate_payload_mutation_while_owned_bundles_share_it():
    plan, encoding, records, default_updates = _epoch("m1", 1)
    default_update = default_updates[0]
    owned_update = externalize_owned_snapshot_optimizer_update(
        plan, encoding, 0, records[0]
    )
    default_bundle = build_local_object_update_bundles(
        plan, encoding, default_update
    )[0]
    owned_bundle = build_local_object_update_bundles(
        plan, encoding, owned_update, owned_payloads=True
    )[0]
    default_tensor = default_bundle["occurrences"][0]["tensor"]
    owned_tensor = owned_bundle["occurrences"][0]["tensor"]
    default_payload = default_update.payloads[0].tensor
    owned_payload = owned_update.payloads[0].tensor

    assert default_tensor.data_ptr() != default_payload.data_ptr()
    assert owned_tensor.data_ptr() == owned_payload.data_ptr()
    before_default, before_owned = default_tensor.clone(), owned_tensor.clone()
    default_payload.add_(1)
    owned_payload.add_(1)
    assert torch.equal(default_tensor, before_default)
    assert not torch.equal(owned_tensor, before_owned)


def test_locator_tampering_fails_before_network_restore():
    plan, encoding, _, updates = _epoch("m1", 1)
    with _services() as services:
        clients = {host: service[0] for host, service in enumerate(services)}
        published = publish_object_update_epoch(
            clients,
            run_id="cpu-gate",
            plan=plan,
            encoding=encoding,
            local_updates=updates,
        )
        value = published.locator.to_dict()
        value["plan_hash"] = "f" * 64
        with pytest.raises(ValueError, match="checksum"):
            ObjectUpdateLocator.from_dict(value)


def test_transport_preserves_runtime_mismatch_direct_fallback():
    plan, encoding, records, updates = _epoch("m1", 1, mismatch=True)
    assert encoding.mismatched_object_ids == tuple(
        sorted(item.object_id for item in plan.objects)
    )
    with _services() as services:
        clients = {host: service[0] for host, service in enumerate(services)}
        locator = publish_object_update_epoch(
            clients,
            run_id="cpu-gate-mismatch",
            plan=plan,
            encoding=encoding,
            local_updates=updates,
        ).locator
        restored = restore_object_update_epoch(
            clients[1],
            surviving_host=1,
            locator=locator,
            plan=plan,
            encoding=encoding,
        )
    for rank, record in restored.records:
        _nested_equal(record, records[rank])


@pytest.mark.parametrize("arm", ("adacheck_fixed", "host_global_fixed", "m1"))
@pytest.mark.parametrize("surviving_host", (0, 1))
@pytest.mark.parametrize("mismatch", (False, True))
def test_selective_restore_reads_only_target_and_required_publishers(
    arm, surviving_host, mismatch
):
    logical_rank = 1
    with _services() as services:
        clients = {host: service[0] for host, service in enumerate(services)}
        plan, encoding, _, updates = _epoch(arm, 1, mismatch=mismatch)
        locator = publish_object_update_epoch(
            clients,
            run_id=f"selective-{arm}-{mismatch}",
            plan=plan,
            encoding=encoding,
            local_updates=updates,
        ).locator
        services[1 - surviving_host][1].shutdown()
        full = restore_object_update_epoch(
            clients[surviving_host],
            surviving_host=surviving_host,
            locator=locator,
            plan=plan,
            encoding=encoding,
        )
        spy = _SpyClient(clients[surviving_host])
        control = fetch_object_update_control(
            spy, surviving_host=surviving_host, locator=locator
        )
        selective = restore_object_update_epoch(
            spy,
            surviving_host=surviving_host,
            locator=locator,
            plan=plan,
            encoding=encoding,
            logical_rank=logical_rank,
            fetched_control=control,
        )

    assert spy.get_source_ranks == [
        CONTROL_SOURCE_RANK,
        *sorted(_target_sources(plan, encoding, logical_rank, surviving_host)),
    ]
    assert spy.get_source_ranks.count(CONTROL_SOURCE_RANK) == 1
    assert [rank for rank, _ in selective.records] == [logical_rank]
    _nested_equal(
        dict(selective.records)[logical_rank], dict(full.records)[logical_rank]
    )


def test_selective_restore_rejects_tampered_or_missing_required_publisher():
    with _services() as services:
        clients = {host: service[0] for host, service in enumerate(services)}
        plan, encoding, _, updates = _epoch("m1", 1)
        logical_rank, surviving_host, external_publishers = next(
            (rank, host, sources - {rank})
            for rank in range(4)
            for host in (0, 1)
            if (sources := _target_sources(plan, encoding, rank, host)) - {rank}
        )
        locator = publish_object_update_epoch(
            clients,
            run_id="selective-fail-closed",
            plan=plan,
            encoding=encoding,
            local_updates=updates,
        ).locator

        control = fetch_object_update_control(
            clients[surviving_host], surviving_host=surviving_host, locator=locator
        )
        publisher = min(external_publishers)
        evidence = next(
            item for item in control.value["bundle_manifests_by_host"][surviving_host][1]
            if item["source_rank"] == publisher
        )
        evidence["payload_checksum"] = "f" * 64
        with pytest.raises(RuntimeError, match="evidence mismatch"):
            restore_object_update_epoch(
                clients[surviving_host],
                surviving_host=surviving_host,
                locator=locator,
                plan=plan,
                encoding=encoding,
                logical_rank=logical_rank,
                fetched_control=control,
            )

        control = fetch_object_update_control(
            clients[surviving_host], surviving_host=surviving_host, locator=locator
        )
        host_evidence = control.value["bundle_manifests_by_host"][surviving_host][1]
        publisher_evidence = next(
            item for item in host_evidence if item["source_rank"] == publisher
        )
        target_evidence = next(
            item for item in host_evidence if item["source_rank"] == logical_rank
        )
        host_evidence.remove(publisher_evidence)
        host_evidence.append(dict(target_evidence))
        with pytest.raises(ValueError, match="invalid rank bundles"):
            restore_object_update_epoch(
                clients[surviving_host],
                surviving_host=surviving_host,
                locator=locator,
                plan=plan,
                encoding=encoding,
                logical_rank=logical_rank,
                fetched_control=control,
            )
