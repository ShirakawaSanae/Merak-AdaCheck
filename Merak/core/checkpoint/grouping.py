"""Deterministic checkpoint-replica group planning."""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from typing import Literal, Sequence

import torch.distributed as dist

from .topology import RankPairingPolicy, plan_local_rank_pairing

GroupingScope = Literal["host", "rank"]
GroupingPolicy = Literal["sequential", "strided"]


@dataclass(frozen=True)
class CheckpointGroupingConfig:
    """Uniform topology and replica placement configuration.

    ``host`` scope pairs corresponding local ranks across physical hosts.
    ``rank`` scope is only for intra-host functional/transport experiments; it
    does not provide host-failure protection.
    """

    ranks_per_node: int
    group_size: int = 2
    scope: GroupingScope = "host"
    policy: GroupingPolicy = "sequential"
    rank_pairing_policy: RankPairingPolicy = "identity"
    local_payload_bytes: tuple[int, ...] | None = None
    local_nic_costs: tuple[float, ...] | None = None


@dataclass(frozen=True)
class CheckpointGroupingPlan:
    config: CheckpointGroupingConfig
    groups: tuple[tuple[int, ...], ...]

    def group_for_rank(self, rank: int) -> tuple[int, ...]:
        for group in self.groups:
            if rank in group:
                return group
        raise ValueError(f"rank {rank} is absent from grouping plan")


def _partition(
    units: Sequence[int], group_size: int, policy: GroupingPolicy
) -> list[tuple[int, ...]]:
    if group_size < 1:
        raise ValueError("group_size must be at least one")
    if policy == "sequential":
        return [tuple(units[i : i + group_size]) for i in range(0, len(units), group_size)]
    if policy == "strided":
        group_count = (len(units) + group_size - 1) // group_size
        return [tuple(units[i::group_count]) for i in range(group_count)]
    raise ValueError(f"unsupported grouping policy: {policy}")


def plan_checkpoint_groups(
    world_size: int, config: CheckpointGroupingConfig
) -> CheckpointGroupingPlan:
    """Return all groups in one deterministic order shared by every rank."""
    if world_size < 1:
        raise ValueError("world_size must be at least one")
    if config.ranks_per_node < 1:
        raise ValueError("ranks_per_node must be at least one")
    if world_size % config.ranks_per_node:
        raise ValueError(
            f"world_size={world_size} is not divisible by "
            f"ranks_per_node={config.ranks_per_node}"
        )

    if config.scope == "rank":
        if (
            config.rank_pairing_policy != "identity"
            or config.local_payload_bytes is not None
            or config.local_nic_costs is not None
        ):
            raise ValueError(
                "rank scope does not support host-local rank pairing or profiles"
            )
        groups = _partition(tuple(range(world_size)), config.group_size, config.policy)
    elif config.scope == "host":
        host_count = world_size // config.ranks_per_node
        host_groups = _partition(tuple(range(host_count)), config.group_size, config.policy)
        if config.rank_pairing_policy != "identity" and config.group_size != 2:
            raise ValueError("non-identity rank pairing currently requires group_size=2")
        local_pairing = plan_local_rank_pairing(
            config.ranks_per_node,
            config.rank_pairing_policy,
            first_payload_bytes=config.local_payload_bytes,
            second_payload_bytes=config.local_payload_bytes,
            first_nic_costs=config.local_nic_costs,
            second_nic_costs=config.local_nic_costs,
        )
        groups = []
        for local_rank in range(config.ranks_per_node):
            for hosts in host_groups:
                if len(hosts) == 2:
                    peer_local_rank = local_pairing.permutation[local_rank]
                    groups.append(
                        (
                            hosts[0] * config.ranks_per_node + local_rank,
                            hosts[1] * config.ranks_per_node + peer_local_rank,
                        )
                    )
                else:
                    groups.append(
                        tuple(host * config.ranks_per_node + local_rank for host in hosts)
                    )
    else:
        raise ValueError(f"unsupported grouping scope: {config.scope}")

    flattened = [rank for group in groups for rank in group]
    if sorted(flattened) != list(range(world_size)):
        raise ValueError(f"invalid grouping plan: groups={groups}")
    return CheckpointGroupingPlan(config=config, groups=tuple(groups))


def grouping_config_from_env(default_group_size: int = 2) -> CheckpointGroupingConfig:
    """Read experiment-scoped settings without changing machine configuration."""
    ranks_per_node = int(
        os.environ.get("CKPT_RANKS_PER_NODE", os.environ.get("LOCAL_WORLD_SIZE", "1"))
    )
    group_size = int(os.environ.get("CKPT_GROUP_SIZE", str(default_group_size)))
    scope = os.environ.get("CKPT_GROUPING_SCOPE", "host")
    policy = os.environ.get("CKPT_GROUPING_POLICY", "sequential")
    rank_pairing_policy = os.environ.get("CKPT_RANK_PAIRING_POLICY", "identity")
    payload_text = os.environ.get("CKPT_LOCAL_PAYLOAD_BYTES")
    nic_cost_text = os.environ.get("CKPT_LOCAL_NIC_COSTS")
    local_payload_bytes = (
        tuple(int(value) for value in payload_text.split(",")) if payload_text else None
    )
    local_nic_costs = (
        tuple(float(value) for value in nic_cost_text.split(",")) if nic_cost_text else None
    )
    return CheckpointGroupingConfig(
        ranks_per_node=ranks_per_node,
        group_size=group_size,
        scope=scope,  # type: ignore[arg-type]
        policy=policy,  # type: ignore[arg-type]
        rank_pairing_policy=rank_pairing_policy,  # type: ignore[arg-type]
        local_payload_bytes=local_payload_bytes,
        local_nic_costs=local_nic_costs,
    )


def create_checkpoint_process_groups(
    plan: CheckpointGroupingPlan,
    backend: str = "gloo",
    timeout_seconds: int = 60,
) -> tuple[object, tuple[int, ...]]:
    """Create all groups in plan order and return the caller's group."""
    rank = dist.get_rank()
    mine = None
    for group_ranks in plan.groups:
        group = dist.new_group(
            ranks=list(group_ranks),
            backend=backend,
            timeout=datetime.timedelta(seconds=timeout_seconds),
        )
        if rank in group_ranks:
            mine = (group, group_ranks)
    if mine is None:
        raise RuntimeError(f"rank {rank} has no checkpoint group")
    return mine
