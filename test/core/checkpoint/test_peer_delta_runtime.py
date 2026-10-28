from contextlib import ExitStack, contextmanager
import threading

import pytest
import torch

from Merak.core.checkpoint.catalog import checkpoint_state_checksum
from Merak.core.checkpoint.matched import commit_control, commit_data_shard
from Merak.core.checkpoint.hierarchical_peer import build_hierarchical_study
from Merak.core.checkpoint.peer_delta import (
    build_peer_delta_matrix,
    peer_grouping_for_policy,
)
from Merak.core.checkpoint.peer_delta_runtime import (
    build_peer_delta_control_state,
    build_peer_delta_protection_plan,
    build_peer_delta_shard,
    restore_peer_delta_from_service,
    restore_peer_delta_from_services,
)
from Merak.core.checkpoint.reduced import build_reduced_checkpoint
from Merak.core.checkpoint.replica_service import (
    ReplicaServiceClient,
    ReplicaTCPServer,
    ServiceReplicaStore,
)
from Merak.core.checkpoint.matched import reduced_catalog_hash
from Merak.core.checkpoint.opportunity import RankGrouping


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
        0: {"model": {"weight": torch.ones(32), "rank": torch.tensor([0])}},
        1: {"model": {"weight": torch.full((16,), 2.0), "rank": torch.tensor([1])}},
        2: {"model": {"weight": torch.full((16,), 2.0), "rank": torch.tensor([2])}},
        3: {"model": {"weight": torch.ones(32), "rank": torch.tensor([3])}},
    }


def _plan(reduced, policy="redundancy_max", availability_scope="peer"):
    matrix = build_peer_delta_matrix(
        reduced,
        ranks_per_host=2,
        catalog_hash=reduced_catalog_hash(reduced),
    )
    grouping = peer_grouping_for_policy(matrix, policy)
    return build_peer_delta_protection_plan(
        reduced,
        grouping,
        ranks_per_host=2,
        availability_scope=availability_scope,
    )


@pytest.mark.parametrize("availability_scope", ("peer", "host"))
def test_pair_delta_services_restore_from_either_host(availability_scope):
    states = _states()
    reduced = build_reduced_checkpoint(states)
    plan = _plan(reduced, "redundancy_max", availability_scope)
    records = []
    with _running_service() as first, _running_service() as second:
        clients = (first, second)
        for assignment in plan.assignments:
            rank_tensors = {
                item.value_id: reduced.tensors[item.value_id]
                for item in reduced.objects
                if any(
                    requirement.rank == assignment.source_rank
                    for requirement in item.requirements
                )
            }
            shard = build_peer_delta_shard(
                reduced,
                plan,
                source_rank=assignment.source_rank,
                target_host=assignment.target_host,
                tensors=rank_tensors,
            )
            record, _ = commit_data_shard(
                clients[assignment.target_host],
                epoch=7,
                shard={
                    **shard,
                    "publisher_rank": assignment.source_rank,
                    "host_index": assignment.target_host,
                },
                group_ranks=tuple(
                    sorted((assignment.source_rank, assignment.peer_rank))
                ),
            )
            assert record is not None
            records.append(record)
        control = build_peer_delta_control_state(reduced, plan, records)
        commit_control(clients, epoch=7, control_state=control)

        for host, client in enumerate(clients):
            restored, wire_bytes = restore_peer_delta_from_service(
                client, epoch=7, host_index=host
            )
            assert wire_bytes > 0
            for rank, expected in states.items():
                actual = restored.reconstruct_rank(rank)
                assert torch.equal(actual["model"]["weight"], expected["model"]["weight"])
                assert torch.equal(actual["model"]["rank"], expected["model"]["rank"])


def test_peer_control_rejects_missing_delta_record():
    reduced = build_reduced_checkpoint(_states())
    plan = _plan(reduced)
    with pytest.raises(ValueError, match="missing or inconsistent"):
        build_peer_delta_control_state(reduced, plan, ())


def test_peer_shard_rejects_corrupt_source_tensor():
    reduced = build_reduced_checkpoint(_states())
    plan = _plan(reduced)
    assignment = plan.assignments[0]
    corrupt = dict(reduced.tensors)
    value_id = assignment.value_ids[0]
    corrupt[value_id] = corrupt[value_id] + 1
    with pytest.raises(ValueError, match="missing or corrupt"):
        build_peer_delta_shard(
            reduced,
            plan,
            source_rank=assignment.source_rank,
            target_host=assignment.target_host,
            tensors=corrupt,
        )


def test_prepared_peer_delta_without_control_is_not_recoverable():
    reduced = build_reduced_checkpoint(_states())
    plan = _plan(reduced)
    assignment = plan.assignments[0]
    rank_tensors = {
        item.value_id: reduced.tensors[item.value_id]
        for item in reduced.objects
        if any(
            requirement.rank == assignment.source_rank
            for requirement in item.requirements
        )
    }
    shard = build_peer_delta_shard(
        reduced,
        plan,
        source_rank=assignment.source_rank,
        target_host=assignment.target_host,
        tensors=rank_tensors,
    )

    with _running_service() as service:
        record, _ = commit_data_shard(
            service,
            epoch=7,
            shard={
                **shard,
                "publisher_rank": assignment.source_rank,
                "host_index": assignment.target_host,
            },
            group_ranks=tuple(
                sorted((assignment.source_rank, assignment.peer_rank))
            ),
            commit=False,
        )
        assert record is None
        with pytest.raises(RuntimeError, match="replica not found"):
            restore_peer_delta_from_service(
                service, epoch=7, host_index=assignment.target_host
            )


def test_peer_delta_restore_rejects_stale_epoch():
    states = _states()
    reduced = build_reduced_checkpoint(states)
    plan = _plan(reduced)
    records = []
    with _running_service() as first, _running_service() as second:
        clients = (first, second)
        for assignment in plan.assignments:
            rank_tensors = {
                item.value_id: reduced.tensors[item.value_id]
                for item in reduced.objects
                if any(
                    requirement.rank == assignment.source_rank
                    for requirement in item.requirements
                )
            }
            shard = build_peer_delta_shard(
                reduced,
                plan,
                source_rank=assignment.source_rank,
                target_host=assignment.target_host,
                tensors=rank_tensors,
            )
            record, _ = commit_data_shard(
                clients[assignment.target_host],
                epoch=7,
                shard={
                    **shard,
                    "publisher_rank": assignment.source_rank,
                    "host_index": assignment.target_host,
                },
                group_ranks=tuple(
                    sorted((assignment.source_rank, assignment.peer_rank))
                ),
            )
            assert record is not None
            records.append(record)
        control = build_peer_delta_control_state(reduced, plan, records)
        commit_control(clients, epoch=7, control_state=control)

        with pytest.raises(RuntimeError, match="replica not found"):
            restore_peer_delta_from_service(first, epoch=6, host_index=0)


def test_redundancy_aware_plan_sends_less_remote_delta_than_fixed():
    reduced = build_reduced_checkpoint(_states())
    fixed = _plan(reduced, "fixed")
    optimal = _plan(reduced, "redundancy_max")
    assert optimal.pair_local_transfer_bytes < fixed.pair_local_transfer_bytes


def test_runtime_plan_identity_ignores_policy_display_alias():
    reduced = build_reduced_checkpoint(_states())
    plan = _plan(reduced, "fixed")
    alias = build_peer_delta_protection_plan(
        reduced,
        RankGrouping("display_alias", plan.grouping.groups),
        ranks_per_host=2,
    )
    assert alias.plan_hash == plan.plan_hash


def test_host_global_plan_places_each_object_once_per_host():
    reduced = build_reduced_checkpoint(_states())
    matrix = build_peer_delta_matrix(
        reduced,
        ranks_per_host=2,
        catalog_hash=reduced_catalog_hash(reduced),
    )
    grouping = RankGrouping("host_global_control", peer_grouping_for_policy(matrix, "fixed").groups)
    plan = build_peer_delta_protection_plan(
        reduced, grouping, ranks_per_host=2, availability_scope="host"
    )
    expected = set(reduced.tensors)
    assert plan.pair_local_transfer_bytes == plan.host_global_control_bytes
    for host in (0, 1):
        placed = [
            value_id
            for assignment in plan.assignments
            if assignment.target_host == host
            for value_id in assignment.value_ids
        ]
        assert set(placed) == expected
        assert len(placed) == len(expected)
        for assignment in plan.assignments:
            held = {
                item.value_id
                for item in reduced.objects
                if any(
                    requirement.rank == assignment.source_rank
                    for requirement in item.requirements
                )
            }
            assert set(assignment.value_ids) <= held


@pytest.mark.parametrize("availability_scope", ("peer", "host"))
def test_four_host_plan_recovers_after_each_failure_domain_is_excluded(
    availability_scope,
):
    states = {
        rank: {
            "model": {
                "shared": torch.full((16,), float((rank // 2) % 2)),
                "rank": torch.tensor([rank]),
            },
            "optimizer": {
                "state": {
                    "exp_avg": torch.full((8,), float(rank % 2)),
                    "exp_avg_sq": torch.arange(8, dtype=torch.float32) + rank,
                    "step": torch.tensor(11),
                },
                "param_groups": [{"lr": 1e-4, "betas": (0.9, 0.95)}],
            },
            "lr_scheduler": {"last_epoch": 11, "step_count": 12},
            "rng_state": torch.arange(16, dtype=torch.uint8) + rank,
            "sampler": {"epoch": 2, "cursor": 128 + rank},
            "iteration": 11,
        }
        for rank in range(8)
    }
    reduced = build_reduced_checkpoint(states)
    grouping = build_hierarchical_study(reduced, ranks_per_host=2)[
        "hierarchical_joint"
    ].grouping
    plan = build_peer_delta_protection_plan(
        reduced,
        grouping,
        ranks_per_host=2,
        availability_scope=availability_scope,
    )
    assert plan.planned_failure_sets == ((), (0,), (1,), (2,), (3,))
    records = []
    with ExitStack() as stack:
        clients = tuple(stack.enter_context(_running_service()) for _ in range(4))
        for assignment in plan.assignments:
            rank_tensors = {
                item.value_id: reduced.tensors[item.value_id]
                for item in reduced.objects
                if any(
                    requirement.rank == assignment.source_rank
                    for requirement in item.requirements
                )
            }
            shard = build_peer_delta_shard(
                reduced,
                plan,
                source_rank=assignment.source_rank,
                target_host=assignment.target_host,
                tensors=rank_tensors,
            )
            record, _ = commit_data_shard(
                clients[assignment.target_host],
                epoch=11,
                shard={
                    **shard,
                    "publisher_rank": assignment.source_rank,
                    "host_index": assignment.target_host,
                },
                group_ranks=tuple(
                    sorted((assignment.source_rank, assignment.peer_rank))
                ),
            )
            assert record is not None
            records.append(record)
        control = build_peer_delta_control_state(reduced, plan, records)
        commit_control(clients, epoch=11, control_state=control)

        with pytest.raises(ValueError, match="outside the planned topology"):
            restore_peer_delta_from_services(
                clients, epoch=11, failed_host_indices=(4,)
            )
        for failed_host in range(4):
            restored, wire_bytes = restore_peer_delta_from_services(
                clients, epoch=11, failed_host_indices=(failed_host,)
            )
            assert wire_bytes > 0
            for rank, expected in states.items():
                actual = restored.reconstruct_rank(rank)
                assert checkpoint_state_checksum(actual) == (
                    checkpoint_state_checksum(expected)
                )
