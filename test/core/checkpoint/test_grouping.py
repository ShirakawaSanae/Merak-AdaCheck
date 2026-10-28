from Merak.core.checkpoint.grouping import CheckpointGroupingConfig, plan_checkpoint_groups


def test_host_scope_uses_actual_ranks_per_node():
    plan = plan_checkpoint_groups(16, CheckpointGroupingConfig(ranks_per_node=8))
    assert plan.groups == tuple((rank, rank + 8) for rank in range(8))


def test_single_host_is_not_split_into_virtual_hosts():
    plan = plan_checkpoint_groups(8, CheckpointGroupingConfig(ranks_per_node=8))
    assert plan.groups == tuple((rank,) for rank in range(8))


def test_rank_scope_exposes_two_intra_host_baselines():
    sequential = plan_checkpoint_groups(
        8, CheckpointGroupingConfig(ranks_per_node=8, scope="rank", policy="sequential")
    )
    strided = plan_checkpoint_groups(
        8, CheckpointGroupingConfig(ranks_per_node=8, scope="rank", policy="strided")
    )
    assert sequential.groups == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert strided.groups == ((0, 4), (1, 5), (2, 6), (3, 7))
