from contextlib import ExitStack, contextmanager
import threading

import torch

from Merak.core.checkpoint.catalog import checkpoint_state_checksum
from Merak.core.checkpoint.matched import commit_control, commit_data_shard, reduced_catalog_hash
from Merak.core.checkpoint.object_protection_runtime import (
    assert_fair_protected_ab,
    build_protected_object_runtime,
    restore_protected_from_services,
    restore_protected_rank_states,
)
from Merak.core.checkpoint.peer_delta import build_peer_delta_matrix, peer_grouping_for_policy
from Merak.core.checkpoint.peer_delta_runtime import (
    build_peer_delta_control_state,
    build_peer_delta_protection_plan,
    build_peer_delta_shard,
)
from Merak.core.checkpoint.reduced import build_reduced_checkpoint
from Merak.core.checkpoint.replica_service import ReplicaServiceClient, ReplicaTCPServer, ServiceReplicaStore


@contextmanager
def _service():
    server = ReplicaTCPServer(("127.0.0.1", 0), ServiceReplicaStore(max_bytes=8 << 20))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield ReplicaServiceClient(*server.server_address)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _states():
    shared_a = torch.ones(32)
    shared_b = torch.full((16,), 2.0)
    return {
        rank: {
            "model": {
                "weight": shared_a if rank in (0, 3) else shared_b,
                "rank": torch.tensor([rank]),
            },
            "optimizer": {"step": torch.tensor(7.0)},
            "iteration": 7,
        }
        for rank in range(4)
    }


def _protection(reduced, policy):
    matrix = build_peer_delta_matrix(
        reduced, ranks_per_host=2, catalog_hash=reduced_catalog_hash(reduced)
    )
    return build_peer_delta_protection_plan(
        reduced,
        peer_grouping_for_policy(matrix, policy),
        ranks_per_host=2,
    )


def _runtime(reduced, protection, arm, payload_hosts=None):
    return build_protected_object_runtime(
        reduced,
        protection,
        arm=arm,
        epoch=7,
        semantic_plan_hash="a" * 64,
        encoder_name="adacheck_object_materialized_v1",
        source_layout_hash="pp2-tp1-dp2",
        payload_hosts=payload_hosts,
    )


def _commit(reduced, protection, clients):
    records = []
    holders = {
        rank: {
            item.value_id for item in reduced.objects if rank in item.source_ranks
        }
        for rank in reduced.templates
    }
    for assignment in protection.assignments:
        shard = build_peer_delta_shard(
            reduced,
            protection,
            source_rank=assignment.source_rank,
            target_host=assignment.target_host,
            tensors={key: reduced.tensors[key] for key in holders[assignment.source_rank]},
        )
        record, _ = commit_data_shard(
            clients[assignment.target_host],
            epoch=7,
            shard={
                **shard,
                "publisher_rank": assignment.source_rank,
                "host_index": assignment.target_host,
            },
            group_ranks=tuple(sorted((assignment.source_rank, assignment.peer_rank))),
        )
        assert record is not None
        records.append(record)
    control = build_peer_delta_control_state(reduced, protection, records)
    commit_control(clients, epoch=7, control_state=control)


def test_m1_reduces_transfer_and_restores_every_rank_after_either_host_loss():
    states = _states()
    reduced = build_reduced_checkpoint(states)
    fixed = _runtime(reduced, _protection(reduced, "fixed"), "adacheck_object_fixed")
    m1 = _runtime(reduced, _protection(reduced, "redundancy_max"), "m1_object")
    assert_fair_protected_ab(fixed, m1)
    assert m1.checkpoint_transfer_bytes < fixed.checkpoint_transfer_bytes
    for failed_host in (0, 1):
        survivor_only = _runtime(
            reduced, m1.protection, "m1_object", payload_hosts=(1 - failed_host,)
        )
        actual = restore_protected_rank_states(survivor_only, failed_hosts=(failed_host,))
        assert {
            rank: checkpoint_state_checksum(value) for rank, value in actual.items()
        } == {
            rank: checkpoint_state_checksum(value) for rank, value in states.items()
        }


def test_service_restore_uses_only_surviving_payload_and_expected_sealed_plan():
    states = _states()
    reduced = build_reduced_checkpoint(states)
    runtime = _runtime(reduced, _protection(reduced, "redundancy_max"), "m1_object")
    with ExitStack() as stack:
        clients = tuple(stack.enter_context(_service()) for _ in range(2))
        _commit(reduced, runtime.protection, clients)
        restored = restore_protected_from_services(
            runtime, clients, epoch=7, failed_hosts=(0,)
        )
        assert restored.wire_bytes > 0
        assert restored.plan_hash == runtime.recovery.plan_hash
        assert {
            rank: checkpoint_state_checksum(value)
            for rank, value in restored.rank_states.items()
        } == {
            rank: checkpoint_state_checksum(value) for rank, value in states.items()
        }
        assert all(
            value["optimizer"]["step"].shape == ()
            for value in restored.rank_states.values()
        )
