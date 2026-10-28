#!/usr/bin/env python3
"""Run one local-config Merak model for detector capture."""

from __future__ import annotations

import argparse
import datetime
import functools
import json
import os
import socket
import sys
from pathlib import Path

import torch.distributed as dist
from transformers import HfArgumentParser, set_seed

import Merak
from Merak import MerakArguments, MerakTrainer
from Merak.core.checkpoint.catalog import checkpoint_state_checksum
from Merak.core.checkpoint.merak_incremental import capture_runtime_state
from Merak.core.moe import (
    ExpertParallelLayout,
    create_expert_parallel_groups,
    create_shared_parameter_group,
    inspect_expert_partition,
)
from Merak.core.tensor_parallel.mp_attrs import set_tp_layer_lists
from examples.checkpoint_grouping.workloads import (
    SUPPORTED_WORKLOADS,
    build_tiny_workload,
    get_workload,
    validate_parallelism,
)


def _moe_phase(name: str) -> None:
    if os.environ.get("CKPT_MOE_TRACE_PHASES") == "1":
        rank = (
            dist.get_rank()
            if dist.is_available() and dist.is_initialized()
            else os.environ.get("RANK", "?")
        )
        print(f"MOE_PHASE rank={rank} phase={name}", flush=True)


def _configure_process_group_timeout() -> None:
    timeout_seconds = os.environ.get("CKPT_MOE_PROCESS_GROUP_TIMEOUT_SECONDS")
    if timeout_seconds is None:
        return
    timeout = float(timeout_seconds)
    if timeout <= 0:
        raise ValueError("CKPT_MOE_PROCESS_GROUP_TIMEOUT_SECONDS must be positive")
    original = dist.init_process_group

    @functools.wraps(original)
    def init_process_group(*args, **kwargs):
        kwargs.setdefault("timeout", datetime.timedelta(seconds=timeout))
        return original(*args, **kwargs)

    dist.init_process_group = init_process_group


def parse_experiment_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", choices=SUPPORTED_WORKLOADS, required=True)
    parser.add_argument("--pp", type=int, default=4)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--dp", type=int, default=2)
    parser.add_argument(
        "--mesh-order",
        choices=("pp-dp", "dp-pp"),
        default="pp-dp",
        help="Place PP or DP as the outer rank-mesh dimension.",
    )
    parser.add_argument("--expert-parallel-size", type=int, default=1)
    parser.add_argument("--role-aware-zero1", action="store_true")
    parser.add_argument("--tiny-qwen-layers", type=int, default=1)
    parser.add_argument("--tiny-deepseek-layers", type=int, default=1)
    parser.add_argument("--dataset-size", type=int, default=1_000_000)
    parser.add_argument(
        "--tiny-qwen", action="store_true",
        help="Use a small Qwen2-MoE config for distributed correctness runs.",
    )
    parser.add_argument(
        "--tiny-deepseek", action="store_true",
        help="Use a small DeepSeek-V2 config for distributed correctness runs.",
    )
    parser.add_argument("--experiment-metadata", type=Path)
    parser.add_argument("--pretrained-model-dir", type=Path)
    return parser.parse_known_args()


def validate_expert_parallel_options(experiment: argparse.Namespace) -> None:
    ep = experiment.expert_parallel_size
    if ep < 1 or experiment.dp % ep:
        raise ValueError("expert parallel size must divide Merak's composite DP size")
    if ep == 1 and experiment.role_aware_zero1:
        raise ValueError("role-aware ZeRO-1 requires expert parallel size greater than one")
    if ep > 1 and not experiment.role_aware_zero1:
        raise ValueError("genuine EP requires --role-aware-zero1")
    if ep > 1 and experiment.model not in {"qwen2_moe_14b", "deepseek_v2_lite"}:
        raise ValueError("expert parallel conversion requires Qwen2-MoE or DeepSeek-V2-Lite")
    if ep > 1 and experiment.tp != 1:
        raise ValueError("the admitted MoE EP layout requires TP1")
    if getattr(experiment, "tiny_qwen_layers", 1) < 1:
        raise ValueError("tiny Qwen layer count must be positive")
    if getattr(experiment, "tiny_deepseek_layers", 1) < 1:
        raise ValueError("tiny DeepSeek layer count must be positive")
    if getattr(experiment, "tiny_qwen", False) and getattr(
        experiment, "tiny_deepseek", False
    ):
        raise ValueError("select only one tiny MoE workload")
    pretrained_model_dir = getattr(experiment, "pretrained_model_dir", None)
    if pretrained_model_dir is not None and (
        experiment.model != "deepseek_v2_lite" or experiment.tiny_deepseek
    ):
        raise ValueError("--pretrained-model-dir requires full deepseek_v2_lite workload")
    if pretrained_model_dir is not None and experiment.pp != 1:
        raise ValueError("pretrained DeepSeek loading currently requires PP=1")
    if getattr(experiment, "dataset_size", 1_000_000) < 1:
        raise ValueError("dataset size must be positive")


def expert_layout_for_experiment(experiment: argparse.Namespace) -> ExpertParallelLayout:
    return ExpertParallelLayout(
        pp=experiment.pp,
        ep=experiment.expert_parallel_size,
        dp=experiment.dp // experiment.expert_parallel_size,
        tp=experiment.tp,
        order=("pp", "dp", "ep", "tp")
        if experiment.mesh_order == "pp-dp"
        else ("dp", "ep", "pp", "tp"),
    )


def expert_parallel_group_report(layout, groups, shared_group) -> dict[str, object]:
    coordinates = layout.coordinates(dist.get_rank())
    expected_shared = tuple(
        sorted(
            layout.rank(
                pp=coordinates["pp"],
                ep=ep,
                dp=dp,
                tp=coordinates["tp"],
            )
            for dp in range(layout.dp)
            for ep in range(layout.ep)
        )
    )
    expected_expert = tuple(
        sorted(
            layout.group_ranks(
                "dp",
                pp=coordinates["pp"],
                ep=coordinates["ep"],
                tp=coordinates["tp"],
            )
        )
    )
    expected_dispatch = tuple(
        sorted(
            layout.group_ranks(
                "ep",
                pp=coordinates["pp"],
                dp=coordinates["dp"],
                tp=coordinates["tp"],
            )
        )
    )
    actual = {
        "shared_zero1_ranks": tuple(sorted(dist.get_process_group_ranks(shared_group))),
        "expert_zero1_ranks": tuple(sorted(dist.get_process_group_ranks(groups.dp_group))),
        "expert_dispatch_ranks": tuple(sorted(dist.get_process_group_ranks(groups.ep_group))),
    }
    expected = {
        "shared_zero1_ranks": expected_shared,
        "expert_zero1_ranks": expected_expert,
        "expert_dispatch_ranks": expected_dispatch,
    }
    if actual != expected:
        raise ValueError("Merak and logical expert-parallel process groups disagree")
    return {
        "logical_coordinates": coordinates,
        **{name: list(ranks) for name, ranks in actual.items()},
    }


def configure_tensor_parallel(model_name: str) -> None:
    if model_name == "opt":
        set_tp_layer_lists(
            col_para_list=["q_proj", "k_proj", "v_proj", "fc1"],
            row_para_list=["out_proj", "fc2"],
            tp_attr_list=["num_heads", "embed_dim"],
        )
    elif model_name.startswith("llama_"):
        set_tp_layer_lists(
            col_para_list=[
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "gate_proj",
                "up_proj",
            ],
            row_para_list=["self_attn.o_proj", "down_proj"],
            tp_attr_list=["num_heads", "num_key_value_heads"],
        )


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    experiment, remaining = parse_experiment_args()
    validate_expert_parallel_options(experiment)
    _configure_process_group_timeout()
    if experiment.expert_parallel_size > 1:
        _moe_phase("merak_init_pre")
    os.environ["MERAK_MOE_EP"] = str(experiment.expert_parallel_size)
    os.environ["MERAK_MOE_REPLICA_DP"] = str(
        experiment.dp // experiment.expert_parallel_size
    )
    os.environ["MERAK_MESH_ORDER"] = (
        "dp,pp,sp,tp" if experiment.mesh_order == "dp-pp" else "pp,dp,sp,tp"
    )
    Merak.init(experiment.pp, experiment.tp, experiment.dp)
    if experiment.expert_parallel_size > 1:
        _moe_phase("merak_init_complete")
    validate_parallelism(
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        pipeline_parallel=experiment.pp,
        tensor_parallel=experiment.tp,
        data_parallel=experiment.dp,
    )
    parser = HfArgumentParser(MerakArguments)
    (training_args,) = parser.parse_args_into_dataclasses(args=remaining)
    if experiment.role_aware_zero1:
        if training_args.zero_stage != 1:
            raise ValueError("role-aware EP requires --zero_stage 1")
        if training_args.ada_checkpoint:
            raise ValueError("legacy AdaCheck does not support role-aware EP ZeRO-1")
        if os.path.isfile(training_args.zero_reshard):
            raise ValueError("role-aware EP ZeRO-1 reshard loading is not implemented")
    set_seed(training_args.seed)
    configure_tensor_parallel(experiment.model)
    workload = get_workload(experiment.model)
    if experiment.tiny_qwen:
        if experiment.model != "qwen2_moe_14b":
            raise ValueError("--tiny-qwen is only supported for qwen2_moe_14b")
        workload = build_tiny_workload(experiment.model, experiment.tiny_qwen_layers)
    if experiment.tiny_deepseek:
        if experiment.model != "deepseek_v2_lite":
            raise ValueError("--tiny-deepseek is only supported for deepseek_v2_lite")
        workload = build_tiny_workload(
            experiment.model, experiment.tiny_deepseek_layers
        )
    if workload.split_method is not None:
        training_args.split_method = workload.split_method
    expert_groups = None
    shared_expert_group = None
    expert_layout = None
    ep_reports = None
    if experiment.expert_parallel_size > 1:
        timeout = os.environ.get("CKPT_MOE_CONTROL_TIMEOUT_SECONDS")
        group_options = {}
        if timeout is not None:
            group_options["timeout"] = datetime.timedelta(seconds=float(timeout))
        _moe_phase("topology_control_group_create_pre")
        control_group = dist.new_group(
            ranks=list(range(dist.get_world_size())), backend="gloo", **group_options
        )
        _moe_phase("topology_control_group_create_post")
        expert_layout = expert_layout_for_experiment(experiment)
        expert_groups = create_expert_parallel_groups(expert_layout, backend="gloo")
        shared_expert_group = create_shared_parameter_group(expert_layout, backend="gloo")
        _moe_phase("expert_groups_create_post")
        group_report = expert_parallel_group_report(
            expert_layout,
            expert_groups,
            shared_expert_group,
        )
    model = workload.build_model(
        empty=True,
        expert_parallel_groups=expert_groups,
    )
    if expert_groups is not None:
        model._merak_ep_layout = expert_layout
        model._merak_role_aware_zero1 = True
        model._merak_sampler_data_parallel_rank = expert_groups.dp_rank
        model._merak_sampler_data_parallel_size = expert_layout.dp
        model._merak_replica_data_parallel_group = expert_groups.dp_group
        model._merak_zero1_shared_process_group = shared_expert_group
        model._merak_zero1_expert_process_group = expert_groups.dp_group
        model._merak_zero1_control_group = control_group
        model._merak_moe_phase_marker = _moe_phase
    if expert_groups is not None:
        num_experts = getattr(workload.config, "num_experts", None)
        if num_experts is None:
            num_experts = workload.config.n_routed_experts
        local_report = inspect_expert_partition(
            model, num_experts=int(num_experts), groups=expert_groups,
        )
        local_report.update(group_report)
        ep_reports = [None] * dist.get_world_size()
        # This metadata is small; gathering on NCCL allocates the object buffer
        # on every rank's current GPU and can OOM on otherwise tiny workloads.
        _moe_phase("expert_report_gather_pre")
        dist.all_gather_object(ep_reports, local_report, group=control_group)
        _moe_phase("expert_report_gather_post")
        if not all(report["ready"] for report in ep_reports):
            raise RuntimeError("MoE expert-parallel ownership preflight failed")
    dataset = workload.build_dataset(
        size=experiment.dataset_size,
        seq_length=training_args.seq_length,
        seed=training_args.seed if experiment.expert_parallel_size > 1 else None,
    )
    if expert_layout is not None:
        training_args._merak_data_parallel_size = expert_layout.dp
    pretrained_load_report = {}

    def load_pretrained(pipeline):
        if experiment.pretrained_model_dir is not None:
            pretrained_load_report.update(
                workload.load_deepseek_pipeline_weights(
                    pipeline, experiment.pretrained_model_dir
                )
            )

    trainer = MerakTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        leaf_modules=workload.leaf_modules,
        post_pipeline_init=load_pretrained,
    )
    pretrained_report = None
    if experiment.pretrained_model_dir is not None:
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, pretrained_load_report)
        pretrained_report = {"rank_reports": reports}
    _moe_phase("trainer_construct_post")
    topology = [None] * dist.get_world_size()
    dist.all_gather_object(
        topology,
        {
            "rank": dist.get_rank(),
            "host": socket.gethostname(),
            "local_rank": int(os.environ["LOCAL_RANK"]),
            "pipeline": Merak.core.mpu.get_pipe_parallel_rank(),
            "data": Merak.core.mpu.get_data_parallel_rank(),
            "tensor": Merak.core.mpu.get_model_parallel_rank(),
            "expert": None if expert_groups is None else expert_groups.ep_rank,
            "expert_data": None if expert_groups is None else expert_groups.dp_rank,
        },
        group=control_group if expert_groups is not None else None,
    )
    _moe_phase("topology_report_gather_post")
    if dist.get_rank() == 0 and experiment.experiment_metadata is not None:
        atomic_json(
            experiment.experiment_metadata,
            {
                "schema_version": 1,
                "workload": workload.metadata(
                    initialization=(
                        "pretrained_safetensors"
                        if experiment.pretrained_model_dir is not None
                        else "random"
                    )
                ),
                "pretrained_checkpoint": (
                    {
                        "directory": str(experiment.pretrained_model_dir),
                        **pretrained_report,
                    }
                    if pretrained_report is not None and dist.get_rank() == 0
                    else None
                ),
                "parallelism": {
                    "pipeline": experiment.pp,
                    "tensor": experiment.tp,
                    "data": experiment.dp,
                    "expert": experiment.expert_parallel_size,
                    "expert_data": experiment.dp // experiment.expert_parallel_size,
                    "world_size": dist.get_world_size(),
                    "split_method": training_args.split_method,
                    "mesh_order": os.environ["MERAK_MESH_ORDER"],
                    "role_aware_zero1": experiment.role_aware_zero1,
                },
                "rank_topology": topology,
                "expert_parallel_preflight": ep_reports,
                "sequence_length": training_args.seq_length,
                "seed": training_args.seed,
            },
        )
    trainer.train()
    digest_root = os.environ.get("CKPT_FINAL_DIGEST_DIR")
    if digest_root:
        optimizer = trainer.engine.optimizer
        iteration = int(trainer.tuning_params.global_steps)
        optimizer_iteration = int(
            getattr(optimizer, "step_count", optimizer.global_step)
        )
        if optimizer_iteration != iteration:
            raise RuntimeError(
                f"trainer iteration {iteration} differs from optimizer step "
                f"{optimizer_iteration}"
            )
        sampler = getattr(
            getattr(trainer, "train_dataloader", None), "batch_sampler", None
        )
        runtime = capture_runtime_state(iteration, sampler)
        components = {
            "model": trainer.engine.module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": trainer.engine.lr_scheduler.state_dict(),
            "runtime": runtime,
        }
        atomic_json(
            Path(digest_root) / f"rank_{dist.get_rank():05d}.json",
            {
                "schema_version": 2,
                "rank": dist.get_rank(),
                "iteration": iteration,
                "optimizer_iteration": optimizer_iteration,
                "sampler": runtime.get("sampler", {}),
                "state_checksum": checkpoint_state_checksum(components),
                "component_checksums": {
                    name: checkpoint_state_checksum(value)
                    for name, value in components.items()
                },
            },
        )


if __name__ == "__main__":
    main()
