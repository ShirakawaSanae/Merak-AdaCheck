import pytest

from Merak.core.checkpoint.grouping import (
    CheckpointGroupingConfig,
    grouping_config_from_env,
    plan_checkpoint_groups,
)
from Merak.core.checkpoint.topology import (
    bidirectional_pairing_cost,
    plan_local_rank_pairing,
)


PAYLOAD_BYTES = (8, 8, 8, 8, 32, 32, 32, 32)
NIC_COSTS = (1, 1, 1, 1, 4, 4, 4, 4)


def test_cross_numa_pairing_is_bijective():
    pairing = plan_local_rank_pairing(8, "cross_numa")

    assert pairing.permutation == (4, 5, 6, 7, 0, 1, 2, 3)
    assert pairing.peer_local_rank(2, first_host=True) == 6
    assert pairing.peer_local_rank(6, first_host=False) == 2


def test_cross_numa_rejects_odd_local_world_size():
    with pytest.raises(ValueError, match="even"):
        plan_local_rank_pairing(7, "cross_numa")


def test_profile_aware_pairing_minimizes_measured_cost_proxy():
    identity = plan_local_rank_pairing(
        8,
        "identity",
        first_payload_bytes=PAYLOAD_BYTES,
        second_payload_bytes=PAYLOAD_BYTES,
        first_nic_costs=NIC_COSTS,
        second_nic_costs=NIC_COSTS,
    )
    optimized = plan_local_rank_pairing(
        8,
        "profile_aware",
        first_payload_bytes=PAYLOAD_BYTES,
        second_payload_bytes=PAYLOAD_BYTES,
        first_nic_costs=NIC_COSTS,
        second_nic_costs=NIC_COSTS,
    )

    assert sorted(optimized.permutation) == list(range(8))
    assert optimized.estimated_cost < identity.estimated_cost
    assert optimized.estimated_cost == bidirectional_pairing_cost(
        optimized.permutation,
        PAYLOAD_BYTES,
        PAYLOAD_BYTES,
        NIC_COSTS,
        NIC_COSTS,
    )


def test_host_grouping_applies_local_rank_permutation():
    plan = plan_checkpoint_groups(
        16,
        CheckpointGroupingConfig(
            ranks_per_node=8,
            group_size=2,
            scope="host",
            policy="sequential",
            rank_pairing_policy="cross_numa",
        ),
    )

    assert plan.groups == (
        (0, 12),
        (1, 13),
        (2, 14),
        (3, 15),
        (4, 8),
        (5, 9),
        (6, 10),
        (7, 11),
    )


@pytest.mark.parametrize(
    "override",
    (
        {"rank_pairing_policy": "cross_numa"},
        {"local_payload_bytes": PAYLOAD_BYTES},
        {"local_nic_costs": NIC_COSTS},
    ),
)
def test_rank_scope_rejects_host_local_pairing_options(override):
    config = CheckpointGroupingConfig(
        ranks_per_node=8,
        group_size=2,
        scope="rank",
        policy="sequential",
        **override,
    )

    with pytest.raises(ValueError, match="rank scope"):
        plan_checkpoint_groups(8, config)


def test_grouping_environment_parses_profile(monkeypatch):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("CKPT_RANK_PAIRING_POLICY", "profile_aware")
    monkeypatch.setenv("CKPT_LOCAL_PAYLOAD_BYTES", "8,8,8,8,32,32,32,32")
    monkeypatch.setenv("CKPT_LOCAL_NIC_COSTS", "1,1,1,1,4,4,4,4")

    config = grouping_config_from_env()

    assert config.rank_pairing_policy == "profile_aware"
    assert config.local_payload_bytes == PAYLOAD_BYTES
    assert config.local_nic_costs == NIC_COSTS
