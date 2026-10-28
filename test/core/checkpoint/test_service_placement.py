import pytest

from Merak.core.checkpoint.service_placement import (
    plan_replica_service_placement,
)


def test_service_placement_uses_peer_and_local_failure_domains():
    endpoints = ("node0:29590", "node1:29590")
    rank0 = plan_replica_service_placement(
        rank=0,
        group_ranks=(0, 8),
        ranks_per_node=8,
        service_endpoints=endpoints,
        mirror_local=True,
    )
    rank8 = plan_replica_service_placement(
        rank=8,
        group_ranks=(0, 8),
        ranks_per_node=8,
        service_endpoints=endpoints,
        mirror_local=True,
    )

    assert rank0.primary_endpoint == "node1:29590"
    assert rank0.mirror_endpoint == "node0:29590"
    assert rank8.primary_endpoint == "node0:29590"
    assert rank8.mirror_endpoint == "node1:29590"


def test_service_placement_can_restore_every_rank_from_surviving_host():
    placement = plan_replica_service_placement(
        rank=8,
        group_ranks=(0, 8),
        ranks_per_node=8,
        service_endpoints=("node0:29590", "node1:29590"),
        mirror_local=True,
        restore_host_index=1,
    )

    assert placement.primary_endpoint == "node1:29590"
    assert placement.mirror_endpoint is None


def test_service_placement_rejects_missing_host_endpoint():
    with pytest.raises(ValueError, match="cover all rank hosts"):
        plan_replica_service_placement(
            rank=8,
            group_ranks=(0, 8),
            ranks_per_node=8,
            service_endpoints=("node0:29590",),
            mirror_local=True,
        )
