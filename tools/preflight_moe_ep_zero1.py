#!/usr/bin/env python3
"""CPU-only admission check for the registered 16-rank MoE layouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from examples.checkpoint_grouping.workloads import get_workload
from Merak.core.moe.expert_parallel import ExpertParallelLayout, inspect_expert_partition


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pp", type=int, default=4)
    parser.add_argument("--ep", type=int, default=2)
    parser.add_argument("--dp", type=int, default=2)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument(
        "--models",
        nargs="+",
        default=("qwen2_moe_14b", "deepseek_v2_lite"),
    )
    args = parser.parse_args()
    layout = ExpertParallelLayout(args.pp, args.ep, args.dp, args.tp)
    result = {
        "schema_version": 1,
        "kind": "moe_ep_zero1_cpu_preflight",
        "layout": {
            "pp": args.pp,
            "ep": args.ep,
            "dp": args.dp,
            "tp": args.tp,
            "world_size": layout.world_size,
        },
        "zero1": {
            "zero1": True,
            "semantic_fixed_compatible": False,
            "blockers": [
                "current SemanticFixed source and restore paths reject zero_stage=1",
                "current runtime expects full fp32_groups, not ZeRO-1 local ranges",
            ],
        },
        "models": [],
    }
    for name in args.models:
        workload = get_workload(name)
        config = workload.config
        num_experts = getattr(config, "num_experts", None)
        if num_experts is None:
            num_experts = getattr(config, "n_routed_experts", None)
        groups = SimpleNamespace(layout=layout, ep_rank=0, ep_group=None)
        model = workload.build_model(
            empty=True,
            expert_parallel_groups=groups,
        )
        report = inspect_expert_partition(
            model,
            num_experts=int(num_experts),
            groups=groups,
        )
        result["models"].append(
            {
                "name": name,
                "display_name": workload.display_name,
                "num_experts": int(num_experts),
                "parameter_count": workload.parameter_count,
                "expert_partition": report,
                "ep_ownership_ready": report["ready"],
                "checkpoint_ready": False,
            }
        )
    result["ready"] = bool(
        result["models"]
        and all(item["expert_partition"]["ready"] for item in result["models"])
        and result["zero1"]["semantic_fixed_compatible"]
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        temporary.write_text(encoded)
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
