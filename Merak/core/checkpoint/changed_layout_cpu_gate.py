"""Small CPU-only binding for changed-layout object recovery tests.

This deliberately accepts ordinary ``torch.optim.AdamW`` state.  It is not a
SemanticFixed runtime hook and does not interact with AdaCheck.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .object_recovery import ObjectAddress, ObjectFragment


@dataclass(frozen=True)
class LiveAdamWObject:
    """One full TP=1 object backed by a live model or AdamW tensor."""

    address: ObjectAddress
    tensor: torch.Tensor
    scalar: bool = False


def live_adamw_objects(
    module: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    masters: Mapping[str, torch.nn.Parameter],
) -> tuple[LiveAdamWObject, ...]:
    """Expose model, master, and standard AdamW slots by stable FQN and role."""
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("changed-layout gate requires torch.optim.AdamW")
    named = dict(module.named_parameters())
    if set(named) != set(masters):
        raise ValueError("masters must bind exactly the module parameter FQNs")
    result = []
    for fqn, model in sorted(named.items()):
        master = masters[fqn]
        state = optimizer.state.get(master, {})
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("AdamW state must contain exactly step, exp_avg, and exp_avg_sq")
        for role, tensor, scalar in (
            ("model", model, False),
            ("optimizer.master", master, False),
            ("optimizer.exp_avg", state["exp_avg"], False),
            ("optimizer.exp_avg_sq", state["exp_avg_sq"], False),
            ("optimizer.step", state["step"], True),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("AdamW object state must be tensor-valued")
            stored = tensor.reshape(1) if scalar else tensor
            result.append(
                LiveAdamWObject(
                    ObjectAddress(f"{fqn}::{role}", role, tuple(stored.shape), str(stored.dtype)),
                    tensor,
                    scalar,
                )
            )
    return tuple(result)


def capture_materialized_adamw_fragments(
    module: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    masters: Mapping[str, torch.nn.Parameter],
    *,
    epoch: int,
    source_rank: int,
    storage_host: int,
) -> tuple[ObjectFragment, ...]:
    """Capture one materialized TP=1 source replica for the object core."""
    return tuple(
        ObjectFragment.from_tensor(
            item.address,
            epoch=epoch,
            offsets=(0,) * len(item.address.global_shape),
            tensor=item.tensor.reshape(1) if item.scalar else item.tensor,
            source_rank=source_rank,
            storage_host=storage_host,
            base_epoch=epoch,
        )
        for item in live_adamw_objects(module, optimizer, masters)
    )


def apply_sealed_adamw_objects(
    module: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    masters: Mapping[str, torch.nn.Parameter],
    tensors: Mapping[str, torch.Tensor],
) -> None:
    """Preflight every object, then mutate the compatible live tensors together."""
    objects = live_adamw_objects(module, optimizer, masters)
    expected = {item.address.semantic_id: item for item in objects}
    if set(tensors) != set(expected):
        raise ValueError("restored FQN/role objects do not match live AdamW bindings")
    for semantic_id, item in expected.items():
        value = tensors[semantic_id]
        shape = (1,) if item.scalar else tuple(item.tensor.shape)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != item.tensor.dtype:
            raise ValueError("restored AdamW object shape or dtype is incompatible")
    with torch.no_grad():
        for semantic_id, item in expected.items():
            value = tensors[semantic_id]
            item.tensor.copy_(value.reshape_as(item.tensor) if item.scalar else value)
