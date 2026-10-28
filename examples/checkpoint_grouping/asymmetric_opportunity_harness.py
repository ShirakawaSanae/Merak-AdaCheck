#!/usr/bin/env python3
"""Export real asymmetric checkpoint plans and compare pairwise policies."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from torch import nn

from Merak.core.checkpoint.catalog import tensor_checksum
from Merak.core.checkpoint.opportunity import (
    AsymmetricCheckpointPlan,
    PlanRequirement,
    PlannedCheckpointObject,
    evaluate_grouping,
    exact_oracle_grouping,
    fixed_grouping,
    random_grouping,
)


@dataclass(frozen=True)
class TensorPlacement:
    role: str
    name: str
    rank: int
    index_ranges: tuple[tuple[int, int], ...]
    tensor: torch.Tensor


class UnevenPipelineModel(nn.Module):
    """Sequential model whose equally counted stages have unequal state sizes."""

    WIDTHS = (
        64,
        96,
        128,
        192,
        256,
        384,
        512,
        160,
        80,
        320,
        128,
        448,
        224,
        96,
        384,
        192,
        64,
        512,
        256,
        128,
        320,
        160,
        448,
        96,
        224,
        384,
        80,
        256,
        128,
        512,
        192,
        64,
        128,
    )

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(input_width, output_width)
            for input_width, output_width in zip(self.WIDTHS, self.WIDTHS[1:])
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = inputs
        for layer in self.layers:
            value = torch.relu(layer(value))
        return value


class Expert(nn.Module):
    def __init__(self, width: int, expert_width: int) -> None:
        super().__init__()
        self.up = nn.Linear(width, expert_width)
        self.down = nn.Linear(expert_width, width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.gelu(self.up(inputs)))


class HeterogeneousExpertModel(nn.Module):
    EXPERT_WIDTHS = (
        32,
        48,
        64,
        80,
        96,
        128,
        192,
        256,
        384,
        512,
        320,
        224,
        160,
        112,
        72,
        40,
        512,
        384,
        256,
        192,
        128,
        96,
        80,
        64,
        48,
        32,
        160,
        320,
        224,
        112,
        72,
        40,
    )

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.input = nn.Linear(width, width)
        self.router = nn.Linear(width, len(self.EXPERT_WIDTHS))
        self.experts = nn.ModuleList(
            Expert(width, expert_width) for expert_width in self.EXPERT_WIDTHS
        )
        self.output_norm = nn.LayerNorm(width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.input(inputs))
        weights = torch.softmax(self.router(hidden), dim=-1)
        experts = torch.stack([expert(hidden) for expert in self.experts], dim=-2)
        return self.output_norm((experts * weights.unsqueeze(-1)).sum(dim=-2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--world-size", type=int, default=32)
    parser.add_argument("--ranks-per-host", type=int, default=8)
    parser.add_argument("--random-policy-seed", type=int, default=20260830)
    return parser.parse_args()


def atomic_write(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(contents)
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_write(
        path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    )


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def train_one_step(
    model: nn.Module, input_width: int, output_width: int, seed: int
) -> tuple[torch.optim.AdamW, float]:
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(2, 4, input_width)
    targets = torch.randn(2, 4, output_width)
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()
    return optimizer, float(loss.detach())


def _slice_tensor(
    tensor: torch.Tensor, index_ranges: Sequence[tuple[int, int]]
) -> torch.Tensor:
    if not index_ranges:
        if tensor.ndim != 0:
            raise ValueError("only scalar tensors may omit index ranges")
        return tensor.detach().cpu().clone()
    if len(index_ranges) != tensor.ndim:
        raise ValueError("slice rank does not match tensor rank")
    slices = tuple(slice(start, end) for start, end in index_ranges)
    return tensor.detach().cpu()[slices].contiguous().clone()


def _optimizer_tensor(
    optimizer: torch.optim.Optimizer,
    parameter: nn.Parameter,
    state_name: str,
) -> torch.Tensor:
    state = optimizer.state.get(parameter)
    if not isinstance(state, Mapping) or state_name not in state:
        raise ValueError(f"optimizer state {state_name} is absent")
    value = state[state_name]
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"optimizer state {state_name} is not a tensor")
    return value


def placements_for_parameter(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    parameter_name: str,
    rank: int,
    index_ranges: tuple[tuple[int, int], ...],
) -> tuple[TensorPlacement, ...]:
    parameters = dict(model.named_parameters())
    if parameter_name not in parameters:
        raise ValueError(f"unknown model parameter {parameter_name}")
    parameter = parameters[parameter_name]
    placements = [
        TensorPlacement(
            role="model",
            name=f"model.{parameter_name}",
            rank=rank,
            index_ranges=index_ranges,
            tensor=_slice_tensor(parameter, index_ranges),
        )
    ]
    for state_name in ("exp_avg", "exp_avg_sq"):
        state_tensor = _optimizer_tensor(optimizer, parameter, state_name)
        placements.append(
            TensorPlacement(
                role="optimizer",
                name=f"optimizer.{state_name}.{parameter_name}",
                rank=rank,
                index_ranges=index_ranges,
                tensor=_slice_tensor(state_tensor, index_ranges),
            )
        )
    step = _optimizer_tensor(optimizer, parameter, "step")
    placements.append(
        TensorPlacement(
            role="optimizer",
            name=f"optimizer.step.{parameter_name}",
            rank=rank,
            index_ranges=(),
            tensor=_slice_tensor(step, ()),
        )
    )
    return tuple(placements)


def _object_identity(item: TensorPlacement) -> tuple[str, str, tuple[int, ...], str]:
    return (
        item.role,
        str(item.tensor.dtype),
        tuple(item.tensor.shape),
        tensor_checksum(item.tensor),
    )


def build_checkpoint_objects(
    placements: Iterable[TensorPlacement],
) -> tuple[tuple[PlannedCheckpointObject, ...], int]:
    grouped: dict[
        tuple[str, str, tuple[int, ...], str], list[TensorPlacement]
    ] = defaultdict(list)
    full_logical_bytes = 0
    for placement in placements:
        logical_bytes = placement.tensor.numel() * placement.tensor.element_size()
        full_logical_bytes += logical_bytes
        grouped[_object_identity(placement)].append(placement)

    objects = []
    for identity, requirements in sorted(grouped.items()):
        role, dtype, shape, checksum = identity
        logical_bytes = (
            requirements[0].tensor.numel() * requirements[0].tensor.element_size()
        )
        if any(
            item.tensor.numel() * item.tensor.element_size() != logical_bytes
            for item in requirements
        ):
            raise RuntimeError("equal tensor identities have unequal byte sizes")
        object_payload = json.dumps(
            {
                "role": role,
                "dtype": dtype,
                "shape": shape,
                "checksum": checksum,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        object_id = hashlib.sha256(object_payload).hexdigest()
        plan_requirements = tuple(
            PlanRequirement(item.name, item.rank, item.index_ranges)
            for item in sorted(
                requirements,
                key=lambda value: (value.rank, value.name, value.index_ranges),
            )
        )
        objects.append(
            PlannedCheckpointObject(
                object_id=object_id,
                role=role,
                dtype=dtype,
                shape=shape,
                logical_bytes=logical_bytes,
                checksum=checksum,
                holders=tuple(sorted({item.rank for item in requirements})),
                requirements=plan_requirements,
            )
        )
    return tuple(sorted(objects, key=lambda item: item.object_id)), full_logical_bytes


def _literal(node: ast.AST) -> Any:
    return ast.literal_eval(node)


def _slice_ranges(node: ast.AST) -> tuple[tuple[int, int], ...]:
    if not isinstance(node, ast.Tuple):
        raise ValueError("generated full-map slicers must be a tuple")
    ranges = []
    for item in node.elts:
        if not (
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name)
            and item.func.id == "slice"
            and len(item.args) == 3
        ):
            raise ValueError("generated full-map contains a non-slice range")
        start, stop, step = (_literal(argument) for argument in item.args)
        if not isinstance(start, int) or not isinstance(stop, int) or step is not None:
            raise ValueError("generated full-map contains an unsupported slice")
        ranges.append((start, stop))
    return tuple(ranges)


def load_nnscaler_full_maps(
    generated_root: Path,
) -> tuple[tuple[str, int, tuple[tuple[int, int], ...]], ...]:
    records = []
    files = sorted(generated_root.rglob("gencode*.py"))
    if not files:
        raise RuntimeError("nnScaler did not emit rank-specific modules")
    for path in files:
        suffix = path.stem.removeprefix("gencode")
        if not suffix.isdigit():
            raise ValueError(f"cannot infer generated rank from {path}")
        rank = int(suffix)
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "add_full_map" or len(node.args) != 7:
                continue
            original_name = _literal(node.args[3])
            if not isinstance(original_name, str):
                raise ValueError("generated full-map parameter name is not a string")
            records.append((original_name, rank, _slice_ranges(node.args[5])))
    if not records:
        raise RuntimeError("nnScaler generated no parameter full-map records")
    return tuple(sorted(records))


def build_nnscaler_plan(
    output_dir: Path, world_size: int, ranks_per_host: int, seed: int
) -> AsymmetricCheckpointPlan:
    if world_size != len(UnevenPipelineModel.WIDTHS) - 1:
        raise ValueError("the nnScaler model has exactly one layer per planned rank")
    import nnscaler
    from nnscaler.graph.function.anchor import IRGraphAnchor
    from nnscaler.ir.operator import IRDataOperation, IRFwOperation

    torch.manual_seed(seed)
    model = UnevenPipelineModel()
    optimizer, training_loss = train_one_step(
        model,
        UnevenPipelineModel.WIDTHS[0],
        UnevenPipelineModel.WIDTHS[-1],
        seed + 1,
    )

    def uneven_pipeline_policy(graph: Any, config: Any) -> Any:
        graph._mesh_size = (world_size // ranks_per_host, ranks_per_host)
        current_stage = 0
        used_stages = set()
        for node in graph.select(ntype=IRFwOperation):
            if isinstance(node, IRGraphAnchor) or node.name == "multiref":
                continue
            module_paths = tuple(node.module_stack)
            if module_paths:
                layer_path = module_paths[0]
                if not layer_path.startswith("layers."):
                    raise ValueError(f"unexpected nnScaler module path {layer_path}")
                current_stage = int(layer_path.split(".")[1])
            replica = graph.replicate(node, times=1)[0]
            graph.assign(replica, current_stage)
            used_stages.add(current_stage)
        for node in graph.select(ntype=IRDataOperation):
            replica = graph.replicate(node, times=1)[0]
            graph.assign(replica, 0)
        if used_stages != set(range(world_size)):
            raise RuntimeError("nnScaler policy did not assign every planned rank")
        return graph

    work_dir = output_dir / "nnscaler_work"
    generated_dir = output_dir / "nnscaler_generated"
    (work_dir / "logs").mkdir(parents=True, exist_ok=True)
    with working_directory(work_dir):
        nnscaler.parallelize(
            model,
            {"inputs": torch.randn(2, 4, UnevenPipelineModel.WIDTHS[0])},
            uneven_pipeline_policy,
            nnscaler.ComputeConfig(
                plan_ngpus=world_size,
                runtime_ngpus=world_size,
                trace_strategy="cpu",
                inference_only=True,
            ),
            gen_savedir=generated_dir,
            reuse="override",
            load_module=False,
        )
    full_maps = load_nnscaler_full_maps(generated_dir)
    placements = []
    for parameter_name, rank, ranges in full_maps:
        placements.extend(
            placements_for_parameter(
                model=model,
                optimizer=optimizer,
                parameter_name=parameter_name,
                rank=rank,
                index_ranges=ranges,
            )
        )
    expected_parameters = set(dict(model.named_parameters()))
    mapped_parameters = {name for name, _, _ in full_maps}
    if expected_parameters != mapped_parameters:
        raise RuntimeError("nnScaler output did not map every model parameter exactly once")
    objects, full_logical_bytes = build_checkpoint_objects(placements)
    generated_files = sorted(generated_dir.rglob("gencode*.py"))
    return AsymmetricCheckpointPlan(
        name="nnscaler_uneven_pipeline",
        producer="nnscaler.parallelize custom 32-stage equal-layer policy",
        world_size=world_size,
        ranks_per_host=ranks_per_host,
        objects=objects,
        metadata={
            "model_class": type(model).__name__,
            "training_steps_before_snapshot": 1,
            "training_loss": training_loss,
            "full_logical_bytes_before_value_dedup": full_logical_bytes,
            "nnscaler_generated_module_count": len(generated_files),
            "nnscaler_full_map_record_count": len(full_maps),
            "nnscaler_trace_strategy": "cpu",
            "nnscaler_inference_codegen_only": True,
            "optimizer": "AdamW",
        },
    )


def build_expert_plan(
    world_size: int, ranks_per_host: int, seed: int
) -> AsymmetricCheckpointPlan:
    if world_size != len(HeterogeneousExpertModel.EXPERT_WIDTHS):
        raise ValueError("the expert model has exactly one expert per planned rank")
    torch.manual_seed(seed)
    model = HeterogeneousExpertModel()
    optimizer, training_loss = train_one_step(model, 64, 64, seed + 1)
    placements = []
    shared_prefixes = ("input.", "router.", "output_norm.")
    for parameter_name, parameter in model.named_parameters():
        ranges = tuple((0, size) for size in parameter.shape)
        if parameter_name.startswith(shared_prefixes):
            holders = range(world_size)
        elif parameter_name.startswith("experts."):
            expert_index = int(parameter_name.split(".")[1])
            holders = (expert_index,)
        else:
            raise ValueError(f"unmapped expert parameter {parameter_name}")
        for rank in holders:
            placements.extend(
                placements_for_parameter(
                    model=model,
                    optimizer=optimizer,
                    parameter_name=parameter_name,
                    rank=rank,
                    index_ranges=ranges,
                )
            )
    objects, full_logical_bytes = build_checkpoint_objects(placements)
    return AsymmetricCheckpointPlan(
        name="handwritten_heterogeneous_expert_parallel",
        producer="instantiated heterogeneous PyTorch expert placement",
        world_size=world_size,
        ranks_per_host=ranks_per_host,
        objects=objects,
        metadata={
            "model_class": type(model).__name__,
            "expert_widths": list(HeterogeneousExpertModel.EXPERT_WIDTHS),
            "expert_count": len(model.experts),
            "shared_parameters_replicated_on_all_ranks": True,
            "training_steps_before_snapshot": 1,
            "training_loss": training_loss,
            "full_logical_bytes_before_value_dedup": full_logical_bytes,
            "optimizer": "AdamW",
        },
    )


def evaluate_plan(
    plan: AsymmetricCheckpointPlan, random_seed: int
) -> dict[str, Any]:
    groupings = (
        fixed_grouping(plan),
        random_grouping(plan, random_seed),
        exact_oracle_grouping(plan),
    )
    evaluations = {
        grouping.policy: evaluate_grouping(plan, grouping) for grouping in groupings
    }
    exact = evaluations["exact_oracle"]
    fixed = evaluations["fixed"]

    def reduction_percent(baseline: int, target: int) -> float:
        return 0.0 if baseline == 0 else 100.0 * (baseline - target) / baseline

    return {
        "plan_name": plan.name,
        "plan_hash": plan.plan_hash,
        "replica_count": 2,
        "tolerated_host_failures": 1,
        "objective_scope": (
            "host-disjoint rank pairs; unit-bandwidth full-duplex lower bound; "
            "no shared-link, queue, rail, chunk, or start-time model"
        ),
        "policy_results": {
            name: evaluation.to_dict() for name, evaluation in evaluations.items()
        },
        "fixed_vs_exact": {
            "cross_domain_transfer_bytes_reduction_percent": reduction_percent(
                fixed.cross_domain_transfer_bytes,
                exact.cross_domain_transfer_bytes,
            ),
            "global_full_duplex_lower_bound_reduction_percent": reduction_percent(
                fixed.global_full_duplex_lower_bound_bytes,
                exact.global_full_duplex_lower_bound_bytes,
            ),
            "aggregate_pair_peak_proxy_reduction_percent": reduction_percent(
                fixed.aggregate_pair_peak_bytes,
                exact.aggregate_pair_peak_bytes,
            ),
        },
    }


def write_plan_artifacts(
    output_dir: Path, plan: AsymmetricCheckpointPlan, evaluation: Mapping[str, Any]
) -> None:
    plan_dir = output_dir / "plans" / plan.name
    atomic_json(plan_dir / "model_and_parallel_plan.json", plan.to_dict())
    lines = "\n".join(
        json.dumps(item.to_dict(), sort_keys=True) for item in plan.objects
    )
    atomic_write(plan_dir / "object_catalog.jsonl", (lines + "\n").encode())
    atomic_json(plan_dir / "policy_results.json", evaluation)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.world_size % args.ranks_per_host:
        raise ValueError("world size must be divisible by ranks per host")
    if args.world_size // args.ranks_per_host != 4:
        raise ValueError("this offline study is fixed to four logical hosts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)

    nnscaler_plan = build_nnscaler_plan(
        args.output_dir, args.world_size, args.ranks_per_host, args.seed
    )
    expert_plan = build_expert_plan(
        args.world_size, args.ranks_per_host, args.seed + 1000
    )
    plans = (nnscaler_plan, expert_plan)
    evaluations = tuple(
        evaluate_plan(plan, args.random_policy_seed) for plan in plans
    )
    for plan, evaluation in zip(plans, evaluations):
        write_plan_artifacts(args.output_dir, plan, evaluation)

    primary_gaps = [
        max(
            item["fixed_vs_exact"][
                "cross_domain_transfer_bytes_reduction_percent"
            ],
            item["fixed_vs_exact"][
                "global_full_duplex_lower_bound_reduction_percent"
            ],
        )
        for item in evaluations
    ]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    tracked_dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
    )
    summary = {
        "schema_version": 1,
        "run_kind": "offline_asymmetric_checkpoint_grouping_opportunity",
        "git_commit": commit,
        "tracked_worktree_dirty": tracked_dirty,
        "seed": args.seed,
        "random_policy_seed": args.random_policy_seed,
        "world_size": args.world_size,
        "ranks_per_host": args.ranks_per_host,
        "logical_host_count": args.world_size // args.ranks_per_host,
        "physical_four_host_job_launched": False,
        "plan_count": len(plans),
        "plans": list(evaluations),
        "go_no_go": {
            "decision": "no_go_for_complex_runtime_grouping_planner",
            "max_primary_fixed_vs_exact_gap_percent": max(primary_gaps),
            "required_asymmetric_plans_with_double_digit_primary_gap": 2,
            "observed_asymmetric_plans_with_double_digit_primary_gap": sum(
                gap >= 10.0 for gap in primary_gaps
            ),
            "reason": (
                "Pairwise grouping cannot change required replica bytes or the "
                "uniform-bandwidth global lower bound for these plans. Four-host "
                "traffic and commit latency are unavailable, so the M3 runtime "
                "planner gate is not satisfied."
            ),
        },
        "environment": {
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nnscaler_source": str(Path(os.environ.get("NNSCALER_ROOT", ""))),
        },
        "limitations": [
            "This is an offline logical four-host study, not a four-node run.",
            "No actual wire bytes, commit latency, training overlap, or NIC queueing is measured.",
            "The exact oracle is limited to host-disjoint rank pairs and at most 16 ranks per host.",
            "The primary lower bound assumes equal independent full-duplex paths.",
            "The aggregate pair-peak sum is a secondary proxy and not a latency prediction.",
            "Optimizer parameter ranges follow model ranges; ZeRO ranges are not modeled.",
        ],
    }
    atomic_json(args.output_dir / "summary.json", summary)
    atomic_write(
        args.output_dir / "command.txt", (" ".join(os.sys.argv) + "\n").encode()
    )
    atomic_write(args.output_dir / "git_commit.txt", (commit + "\n").encode())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
