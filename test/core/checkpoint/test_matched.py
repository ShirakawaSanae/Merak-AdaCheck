from contextlib import contextmanager
import threading

import pytest
import torch

from Merak.core.checkpoint.matched import (
    build_control_state,
    build_data_shard,
    build_two_host_protection_plan,
    commit_control,
    commit_data_shard,
    restore_reduced_from_service,
)
from Merak.core.checkpoint.opportunity import (
    asymmetric_plan_from_reduced,
    fixed_grouping,
    grouping_for_policy,
)
from Merak.core.checkpoint.reduced import build_reduced_checkpoint
from Merak.core.checkpoint.replica_service import (
    ReplicaServiceClient,
    ReplicaTCPServer,
    ServiceReplicaStore,
)


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


def _states():
    return {
        rank: {
            "model": {
                "host_local": torch.tensor([rank // 2], dtype=torch.float32),
                "rank_local": torch.tensor([rank], dtype=torch.int64),
            },
            "iteration": 7,
        }
        for rank in range(4)
    }


def _plan(reduced):
    opportunity = asymmetric_plan_from_reduced(
        reduced, name="test", ranks_per_host=2
    )
    return build_two_host_protection_plan(
        reduced, fixed_grouping(opportunity), ranks_per_host=2
    )


def test_workload_catalog_adapter_and_policies_are_deterministic():
    reduced = build_reduced_checkpoint(_states())
    opportunity = asymmetric_plan_from_reduced(
        reduced, name="test", ranks_per_host=2
    )

    assert opportunity.world_size == 4
    assert opportunity.host_count == 2
    for policy in ("fixed", "random", "exact"):
        first = grouping_for_policy(opportunity, policy, seed=17)
        second = grouping_for_policy(opportunity, policy, seed=17)
        assert first == second


def test_two_host_plan_places_every_object_once_per_service():
    reduced = build_reduced_checkpoint(_states())
    plan = _plan(reduced)
    expected = set(reduced.tensors)

    for host in range(2):
        actual = [
            value_id
            for assignment in plan.assignments
            if assignment.host_index == host
            for value_id in assignment.value_ids
        ]
        assert len(actual) == len(set(actual))
        assert set(actual) == expected
    assert plan.plan_hash == _plan(reduced).plan_hash


def test_committed_control_restores_from_either_service():
    states = _states()
    reduced = build_reduced_checkpoint(states)
    plan = _plan(reduced)
    peers = {
        rank: other
        for first, second in plan.grouping.groups
        for rank, other in ((first, second), (second, first))
    }

    with _running_service() as first, _running_service() as second:
        clients = (first, second)
        records = []
        for rank in range(4):
            shard = build_data_shard(reduced, plan, rank)
            record, _ = commit_data_shard(
                clients[rank // 2],
                epoch=7,
                shard=shard,
                group_ranks=tuple(sorted((rank, peers[rank]))),
            )
            assert record is not None
            records.append(record)
        control = build_control_state(reduced, plan, records)
        committed, _ = commit_control(clients, epoch=7, control_state=control)
        assert committed.committed

        for host, client in enumerate(clients):
            restored, wire_bytes = restore_reduced_from_service(
                client, epoch=7, host_index=host
            )
            assert wire_bytes > 0
            for rank, expected in states.items():
                actual = restored.reconstruct_rank(rank)
                assert torch.equal(
                    actual["model"]["rank_local"],
                    expected["model"]["rank_local"],
                )


def test_incomplete_shards_cannot_publish_a_control_record():
    reduced = build_reduced_checkpoint(_states())
    plan = _plan(reduced)

    with pytest.raises(ValueError, match="incomplete host"):
        build_control_state(reduced, plan, ())


def test_prepared_data_without_control_is_not_recoverable():
    reduced = build_reduced_checkpoint(_states())
    plan = _plan(reduced)
    shard = build_data_shard(reduced, plan, 0)

    with _running_service() as service:
        record, _ = commit_data_shard(
            service,
            epoch=7,
            shard=shard,
            group_ranks=(0, 2),
            commit=False,
        )
        assert record is None
        with pytest.raises(RuntimeError, match="replica not found"):
            restore_reduced_from_service(service, epoch=7, host_index=0)
