"""Expert-parallel topology, ownership, and token dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.nn.functional import all_to_all_single


_AXES = ("pp", "ep", "dp", "tp")


@dataclass(frozen=True)
class ExpertParallelLayout:
    """Rank layout for ``PP x EP x DP x TP`` jobs.

    ``order`` has the same row-major meaning as Merak's existing mesh order.
    Keeping EP here, instead of overloading DP, is important: ZeRO-1 and
    gradient reduction must use the DP axis while token dispatch uses EP.
    """

    pp: int
    ep: int
    dp: int
    tp: int
    order: tuple[str, ...] = _AXES

    def __post_init__(self) -> None:
        if any(type(value) is not int or value < 1 for value in (self.pp, self.ep, self.dp, self.tp)):
            raise ValueError("PP, EP, DP, and TP must be positive integers")
        if tuple(self.order) != self.order or set(self.order) != set(_AXES) or len(self.order) != 4:
            raise ValueError("order must be a permutation of pp,ep,dp,tp")

    @property
    def world_size(self) -> int:
        return self.pp * self.ep * self.dp * self.tp

    def _dims(self) -> Mapping[str, int]:
        return {"pp": self.pp, "ep": self.ep, "dp": self.dp, "tp": self.tp}

    def coordinates(self, rank: int) -> dict[str, int]:
        if type(rank) is not int or not 0 <= rank < self.world_size:
            raise ValueError("rank is outside the EP layout")
        values = [0] * len(self.order)
        remainder = rank
        dims = self._dims()
        for index in range(len(self.order) - 1, -1, -1):
            axis = self.order[index]
            values[index] = remainder % dims[axis]
            remainder //= dims[axis]
        return dict(zip(self.order, values))

    def rank(self, **coordinates: int) -> int:
        if set(coordinates) != set(_AXES):
            raise ValueError("rank() requires pp, ep, dp, and tp coordinates")
        dims = self._dims()
        result = 0
        for axis in self.order:
            value = coordinates[axis]
            if type(value) is not int or not 0 <= value < dims[axis]:
                raise ValueError(f"{axis} coordinate is outside the EP layout")
            result = result * dims[axis] + value
        return result

    def group_ranks(self, varying_axis: str, **fixed: int) -> tuple[int, ...]:
        if varying_axis not in _AXES:
            raise ValueError("unknown EP layout axis")
        if set(fixed) != set(_AXES) - {varying_axis}:
            raise ValueError("group_ranks() needs every non-varying coordinate")
        values = range(self._dims()[varying_axis])
        return tuple(
            self.rank(**{**fixed, varying_axis: value}) for value in values
        )

    def all_groups(self, varying_axis: str) -> tuple[tuple[int, ...], ...]:
        if varying_axis not in _AXES:
            raise ValueError("unknown EP layout axis")
        other = [axis for axis in _AXES if axis != varying_axis]
        ranges = [range(self._dims()[axis]) for axis in other]
        return tuple(
            tuple(self.rank(**dict(zip(other, values)), **{varying_axis: axis_value})
                  for axis_value in range(self._dims()[varying_axis]))
            for values in product(*ranges)
        )

    def expert_owner(self, expert_id: int, num_experts: int) -> int:
        if type(expert_id) is not int or type(num_experts) is not int:
            raise ValueError("expert IDs and counts must be integers")
        if not 0 <= expert_id < num_experts or num_experts % self.ep:
            raise ValueError("experts must be evenly partitionable across EP")
        return expert_id // (num_experts // self.ep)


@dataclass(frozen=True)
class ExpertParallelGroups:
    layout: ExpertParallelLayout
    rank: int
    ep_group: object
    dp_group: object
    pp_group: object
    tp_group: object

    @property
    def coordinates(self) -> dict[str, int]:
        return self.layout.coordinates(self.rank)

    @property
    def ep_rank(self) -> int:
        return self.coordinates["ep"]

    @property
    def dp_rank(self) -> int:
        return self.coordinates["dp"]


def create_shared_parameter_group(
    layout: ExpertParallelLayout, *, backend: str | None = None
) -> object:
    """Create the shared/router group across EP and replica-DP peers."""
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("shared EP groups require an initialized distributed job")
    if dist.get_world_size() != layout.world_size:
        raise ValueError(
            f"EP layout world size {layout.world_size} != process world {dist.get_world_size()}"
        )

    rank = dist.get_rank()
    selected_group = None
    other_axes = ("pp", "tp")
    for pp in range(layout.pp):
        for tp in range(layout.tp):
            ranks = tuple(
                layout.rank(pp=pp, ep=ep, dp=dp, tp=tp)
                for dp in range(layout.dp)
                for ep in range(layout.ep)
            )
            group = dist.new_group(ranks=list(ranks), backend=backend)
            if rank in ranks:
                selected_group = group
    if selected_group is None:
        raise RuntimeError("rank is absent from all shared-parameter groups")
    return selected_group


@dataclass(frozen=True)
class ExpertRoutingStats:
    send_counts: tuple[int, ...]
    receive_counts: tuple[int, ...]

    @property
    def sent_tokens(self) -> int:
        return sum(self.send_counts)

    @property
    def received_tokens(self) -> int:
        return sum(self.receive_counts)


class _AddAuxiliaryLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return values

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        aux_gradient = None
        if ctx.required_aux_loss:
            aux_gradient = torch.ones(1, dtype=ctx.dtype, device=gradient.device)
        return gradient, aux_gradient


def create_expert_parallel_groups(
    layout: ExpertParallelLayout, *, backend: str | None = None
) -> ExpertParallelGroups:
    """Create all four axis groups in deterministic order and select this rank."""
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("EP groups require an initialized torch.distributed job")
    if dist.get_world_size() != layout.world_size:
        raise ValueError(
            f"EP layout world size {layout.world_size} != process world {dist.get_world_size()}"
        )
    rank = dist.get_rank()
    groups: dict[str, object] = {}
    for axis in _AXES:
        for ranks in layout.all_groups(axis):
            kwargs = {} if backend is None else {"backend": backend}
            group = dist.new_group(ranks=list(ranks), **kwargs)
            if rank in ranks:
                groups[axis] = group
    return ExpertParallelGroups(
        layout,
        rank,
        groups["ep"],
        groups["dp"],
        groups["pp"],
        groups["tp"],
    )


def _variable_all_to_all(
    tensor: torch.Tensor,
    *,
    output_counts: list[int],
    input_counts: list[int],
    group: object,
) -> torch.Tensor:
    shape = (sum(output_counts),) + tensor.shape[1:]
    output = tensor.new_empty(shape)
    return all_to_all_single(
        output,
        tensor.contiguous(),
        output_split_sizes=output_counts,
        input_split_sizes=input_counts,
        group=group,
    )


def dispatch_to_experts(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    experts: Mapping[int, nn.Module],
    num_experts: int,
    ep_rank: int,
    ep_group: object | None,
    ep_size: int,
) -> tuple[torch.Tensor, ExpertRoutingStats]:
    """Dispatch selected tokens to owner ranks and return combined outputs."""
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [tokens, hidden]")
    if selected_experts.shape != routing_weights.shape:
        raise ValueError("selected experts and routing weights must have the same shape")
    if selected_experts.ndim != 2 or selected_experts.shape[0] != hidden_states.shape[0]:
        raise ValueError("routing tensors must have shape [tokens, top_k]")
    if num_experts < 1 or ep_size < 1 or num_experts % ep_size:
        raise ValueError("experts must be evenly partitionable across EP")
    if not 0 <= ep_rank < ep_size:
        raise ValueError("ep_rank is outside the EP group")

    experts_per_rank = num_experts // ep_size
    expected = set(range(ep_rank * experts_per_rank, (ep_rank + 1) * experts_per_rank))
    if set(experts) != expected:
        raise ValueError("local experts do not match the contiguous EP partition")

    top_k = selected_experts.shape[1]
    flat_experts = selected_experts.reshape(-1)
    flat_weights = routing_weights.reshape(-1)
    destinations = torch.div(flat_experts, experts_per_rank, rounding_mode="floor")
    order = torch.argsort(destinations, stable=True)
    # Give each assignment a unique index; expand backward reduces top-k in
    # a fixed order instead of atomically accumulating repeated token indices.
    assignments = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(
        -1, hidden_states.shape[1]
    )
    send_states = assignments.index_select(0, order)
    send_experts = flat_experts.index_select(0, order).contiguous()
    send_counts_tensor = torch.bincount(destinations, minlength=ep_size).to(torch.int64)

    if ep_size == 1:
        receive_counts_tensor = send_counts_tensor
        received_states = send_states
        received_experts = send_experts
    else:
        if ep_group is None or not dist.is_initialized():
            raise RuntimeError("EP dispatch requires an initialized process group")
        if dist.get_world_size(group=ep_group) != ep_size:
            raise ValueError("EP process group size does not match the layout")
        receive_counts_tensor = torch.empty_like(send_counts_tensor)
        dist.all_to_all_single(
            receive_counts_tensor, send_counts_tensor, group=ep_group
        )
        send_counts = send_counts_tensor.tolist()
        receive_counts = receive_counts_tensor.tolist()
        received_states = _variable_all_to_all(
            send_states,
            output_counts=receive_counts,
            input_counts=send_counts,
            group=ep_group,
        )
        received_experts = torch.empty(
            sum(receive_counts), dtype=send_experts.dtype, device=send_experts.device
        )
        dist.all_to_all_single(
            received_experts,
            send_experts,
            output_split_sizes=receive_counts,
            input_split_sizes=send_counts,
            group=ep_group,
        )

    # Keep the dispatch collective in every rank's autograd graph even when a
    # skewed route gives this rank zero received tokens.
    local_output = received_states * 0
    for expert_id, expert in experts.items():
        indices = torch.nonzero(received_experts == expert_id, as_tuple=False).flatten()
        if indices.numel():
            values = expert(received_states.index_select(0, indices))
            local_output = local_output.index_copy(0, indices, values)

    send_counts = send_counts_tensor.tolist()
    receive_counts = receive_counts_tensor.tolist()
    if ep_size == 1:
        returned = local_output
    else:
        returned = _variable_all_to_all(
            local_output,
            output_counts=send_counts,
            input_counts=receive_counts,
            group=ep_group,
        )

    assignment_output = torch.zeros_like(returned).index_copy(0, order, returned)
    weighted = assignment_output * flat_weights.unsqueeze(-1)
    combined = weighted.reshape(hidden_states.shape[0], top_k, -1).sum(dim=1)
    return combined.to(hidden_states.dtype), ExpertRoutingStats(
        tuple(send_counts), tuple(receive_counts)
    )


class Qwen2MoeExpertParallelBlock(nn.Module):
    """Drop-in Qwen2-MoE sparse block holding only locally owned experts."""

    def __init__(self, block: nn.Module, groups: ExpertParallelGroups):
        super().__init__()
        self.num_experts = int(block.num_experts)
        self.top_k = int(block.top_k)
        self.norm_topk_prob = bool(block.norm_topk_prob)
        if self.num_experts % groups.layout.ep:
            raise ValueError("Qwen experts must be divisible by EP size")

        self.gate = block.gate
        self.shared_expert = block.shared_expert
        self.shared_expert_gate = block.shared_expert_gate
        per_rank = self.num_experts // groups.layout.ep
        start = groups.ep_rank * per_rank
        stop = start + per_rank
        self.experts = nn.ModuleDict(
            {str(expert_id): block.experts[expert_id] for expert_id in range(start, stop)}
        )
        self.ep_group = groups.ep_group
        self.ep_rank = groups.ep_rank
        self.ep_size = groups.layout.ep
        self.last_routing: ExpertRoutingStats | None = None
        for expert_id, expert in self.local_experts.items():
            for parameter in expert.parameters():
                parameter._merak_expert_parallel = True
                parameter._merak_expert_id = expert_id

    @property
    def local_experts(self) -> dict[int, nn.Module]:
        return {int(expert_id): expert for expert_id, expert in self.experts.items()}

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        flat_states = hidden_states.reshape(-1, hidden_dim)
        router_logits = self.gate(flat_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True
            )
        routing_weights = routing_weights.to(flat_states.dtype)
        routed, self.last_routing = dispatch_to_experts(
            flat_states,
            selected_experts,
            routing_weights,
            experts=self.local_experts,
            num_experts=self.num_experts,
            ep_rank=self.ep_rank,
            ep_group=self.ep_group,
            ep_size=self.ep_size,
        )
        shared = torch.sigmoid(self.shared_expert_gate(flat_states))
        shared = shared * self.shared_expert(flat_states)
        return (routed + shared).reshape(batch_size, sequence_length, hidden_dim), router_logits


def shard_qwen2_moe_model(module: nn.Module, groups: ExpertParallelGroups) -> nn.Module:
    """Replace every Qwen sparse block with its expert-parallel equivalent."""
    if module.__class__.__name__ == "Qwen2MoeSparseMoeBlock":
        return Qwen2MoeExpertParallelBlock(module, groups)
    for name, child in tuple(module.named_children()):
        replacement = shard_qwen2_moe_model(child, groups)
        if replacement is not child:
            setattr(module, name, replacement)
    return module


class MoEPipelineDecoderLayer(nn.Module):
    """Run one complete MoE decoder layer with pipeline-local attention inputs."""

    def __init__(self, layer: nn.Module, family: str):
        super().__init__()
        if family not in {"qwen2_moe", "deepseek_v2"}:
            raise ValueError(f"unsupported MoE pipeline family: {family}")
        if family == "qwen2_moe" and getattr(
            layer.self_attn.config, "output_router_logits", False
        ):
            raise ValueError("Qwen pipeline training requires output_router_logits=False")
        self.layer = layer
        self.family = family

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = hidden_states.shape[:2]
        positions = torch.arange(sequence_length, device=hidden_states.device)
        position_ids = positions.unsqueeze(0)
        config = self.layer.self_attn.config
        layer_index = getattr(self.layer.self_attn, "layer_idx", 0)
        from transformers.modeling_attn_mask_utils import (
            _prepare_4d_causal_attention_mask,
        )

        use_window = (
            self.family == "qwen2_moe"
            and config.use_sliding_window
            and layer_index < config.max_window_layers
            and sequence_length > config.sliding_window
        )
        attention_mask = _prepare_4d_causal_attention_mask(
            None,
            (batch_size, sequence_length),
            hidden_states,
            0,
            sliding_window=config.sliding_window if use_window else None,
        )
        outputs = self.layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            **(
                {"output_router_logits": False, "cache_position": positions}
                if self.family == "qwen2_moe"
                else {}
            ),
        )
        return outputs[0]


def _causal_mask(hidden_states: torch.Tensor) -> torch.Tensor:
    batch_size, sequence_length = hidden_states.shape[:2]
    mask = torch.full(
        (sequence_length, sequence_length),
        torch.finfo(hidden_states.dtype).min,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    mask = torch.triu(mask, diagonal=1)
    return mask[None, None].expand(batch_size, 1, -1, -1)


class Qwen2MoePipelineDecoderLayer(MoEPipelineDecoderLayer):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = hidden_states.shape[:2]
        positions = torch.arange(sequence_length, device=hidden_states.device)
        position_ids = positions.unsqueeze(0)
        causal_mask = _causal_mask(hidden_states)
        outputs = self.layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            output_router_logits=False,
            use_cache=False,
            cache_position=positions,
        )
        return outputs[0]


class MoEFloatLogits(nn.Module):
    def __init__(self, lm_head: nn.Module):
        super().__init__()
        self.lm_head = lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states).float()


def build_moe_pipeline_layers(module: nn.Module) -> list[nn.Module]:
    """Expose embeddings, complete decoder blocks, and output head to Merak PP."""
    family = getattr(getattr(module, "config", None), "model_type", None)
    if family not in {"qwen2_moe", "deepseek_v2"}:
        raise ValueError("model is not a supported MoE causal LM")
    decoder = getattr(module, "model", None)
    if decoder is None or not all(
        hasattr(decoder, name) for name in ("embed_tokens", "layers", "norm")
    ) or not hasattr(module, "lm_head"):
        raise ValueError("MoE causal LM does not expose the expected decoder modules")
    layer_type = Qwen2MoePipelineDecoderLayer if family == "qwen2_moe" else MoEPipelineDecoderLayer
    decoder_layers = tuple(layer_type(layer, family) for layer in decoder.layers)
    return [
        decoder.embed_tokens,
        *decoder_layers,
        decoder.norm,
        MoEFloatLogits(module.lm_head),
    ]


class DeepseekV2ExpertParallelBlock(nn.Module):
    """DeepSeek-V2 MoE block with global expert IDs and subgroup EP."""

    def __init__(self, block: nn.Module, groups: ExpertParallelGroups):
        super().__init__()
        self.config = block.config
        self.num_experts = int(block.config.n_routed_experts)
        self.num_experts_per_tok = int(block.num_experts_per_tok)
        if self.num_experts % groups.layout.ep:
            raise ValueError("DeepSeek experts must be divisible by EP size")

        self.gate = block.gate
        if hasattr(block, "shared_experts"):
            self.shared_experts = block.shared_experts
        self.ep_group = groups.ep_group
        self.ep_rank = groups.ep_rank
        self.ep_size = groups.layout.ep
        self.experts_per_rank = self.num_experts // self.ep_size
        start = self.ep_rank * self.experts_per_rank
        stop = start + self.experts_per_rank
        owned = {expert_id: block.experts[expert_id] for expert_id in range(start, stop)}
        if any(expert is None for expert in owned.values()):
            raise ValueError("convert DeepSeek from ep_size=1 before subgroup EP sharding")
        self.experts = nn.ModuleDict(
            {str(expert_id): expert for expert_id, expert in owned.items()}
        )
        self.last_routing: ExpertRoutingStats | None = None
        for expert_id, expert in self.local_experts.items():
            for parameter in expert.parameters():
                parameter._merak_expert_parallel = True
                parameter._merak_expert_id = expert_id

    @property
    def local_experts(self) -> dict[int, nn.Module]:
        return {int(expert_id): expert for expert_id, expert in self.experts.items()}

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        shape = hidden_states.shape
        selected_experts, routing_weights, aux_loss = self.gate(hidden_states)
        routed, self.last_routing = dispatch_to_experts(
            hidden_states.reshape(-1, shape[-1]),
            selected_experts,
            routing_weights,
            experts=self.local_experts,
            num_experts=self.num_experts,
            ep_rank=self.ep_rank,
            ep_group=self.ep_group,
            ep_size=self.ep_size,
        )
        routed = routed.reshape(shape)
        if aux_loss is not None:
            routed = _AddAuxiliaryLoss.apply(routed, aux_loss)
        if hasattr(self, "shared_experts"):
            routed = routed + self.shared_experts(identity)
        return routed


def shard_deepseek_v2_moe_model(
    module: nn.Module, groups: ExpertParallelGroups
) -> nn.Module:
    """Replace every DeepSeek-V2 MoE block with subgroup EP dispatch."""
    if module.__class__.__name__ == "DeepseekV2MoE":
        return DeepseekV2ExpertParallelBlock(module, groups)
    for name, child in tuple(module.named_children()):
        replacement = shard_deepseek_v2_moe_model(child, groups)
        if replacement is not child:
            setattr(module, name, replacement)
    return module


def _expert_modules(module: Any) -> tuple[tuple[str, Any, bool], ...]:
    found = []
    if not hasattr(module, "named_modules"):
        return ()
    for name, child in module.named_modules():
        experts = getattr(child, "experts", None)
        if experts is None:
            continue
        values = tuple(experts.values()) if hasattr(experts, "values") else tuple(experts)
        if values:
            found.append(
                (
                    name or "<root>",
                    experts,
                    isinstance(
                        child,
                        (Qwen2MoeExpertParallelBlock, DeepseekV2ExpertParallelBlock),
                    ),
                )
            )
    return tuple(found)


def inspect_expert_partition(
    module: Any,
    *,
    num_experts: int,
    groups: ExpertParallelGroups,
) -> dict[str, Any]:
    """Return a fail-closed EP readiness report for a live model.

    The report is intentionally JSON-like so launchers can persist it in a
    preflight manifest.  ``ready`` means only ownership is plausible; it does
    not claim that token dispatch is differentiable or numerically validated.
    """
    if type(num_experts) is not int or num_experts < 1:
        raise ValueError("num_experts must be positive")
    expected = num_experts // groups.layout.ep if num_experts % groups.layout.ep == 0 else None
    modules = _expert_modules(module)
    blockers: list[str] = []
    details = []
    if expected is None:
        blockers.append("num_experts is not divisible by EP size")
    if not modules:
        blockers.append("no module with an experts collection was found")
    for name, experts, dispatch_validated in modules:
        collection = (
            dict(experts.items())
            if hasattr(experts, "items")
            else dict(enumerate(experts))
        )
        present = tuple(int(index) for index, expert in collection.items() if expert is not None)
        if not hasattr(experts, "items") and len(experts) != num_experts:
            blockers.append(f"{name} exposes {len(experts)} experts, expected {num_experts}")
        if groups.layout.ep > 1 and expected is not None and len(present) == num_experts:
            blockers.append(f"{name} replicates all experts on every rank")
        elif expected is not None:
            owned = tuple(
                index for index in present
                if groups.layout.expert_owner(index, num_experts) == groups.ep_rank
            )
            if len(present) != expected or present != owned:
                blockers.append(f"{name} expert ownership does not match contiguous EP partition")
        if groups.layout.ep > 1 and not dispatch_validated:
            blockers.append(f"{name} does not use validated EP token dispatch")
        details.append({"module": name, "present_expert_ids": list(present)})
    return {
        "ready": not blockers,
        "dispatch_validated": bool(modules)
        and all(item[2] for item in modules),
        "ep_size": groups.layout.ep,
        "ep_rank": groups.ep_rank,
        "num_experts": num_experts,
        "expected_local_experts": expected,
        "modules": details,
        "blockers": blockers,
    }
