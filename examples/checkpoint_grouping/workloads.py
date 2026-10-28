"""Shared local model definitions for checkpoint-grouping experiments."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import types
from dataclasses import dataclass, replace as dataclass_replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors import safe_open
from transformers import (
    BertConfig,
    BertForMaskedLM,
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    OPTConfig,
    OPTForCausalLM,
    Qwen2MoeConfig,
    Qwen2MoeForCausalLM,
)
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.opt.modeling_opt import OPTAttention
from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeAttention

from Merak import init_empty_weights
from Merak.core.moe import (
    ExpertParallelGroups,
    Qwen2MoeExpertParallelBlock,
    build_moe_pipeline_layers,
    shard_qwen2_moe_model,
)
from Merak.core.moe.expert_parallel import (
    DeepseekV2ExpertParallelBlock,
    shard_deepseek_v2_moe_model,
)
from Merak.utils.datasets import DynamicGenDataset
from examples.models.bert.config import load_config as load_bert_config
from examples.models.gpt2.config import load_config as load_gpt2_config
from examples.models.opt.config import load_config as load_opt_config


SUPPORTED_WORKLOADS = (
    "bert",
    "gpt2",
    "opt",
    "llama_1b",
    "llama_3b",
    "llama_3_8b",
    "gpt_6b",
    "qwen2_moe_14b",
    "deepseek_v2_lite",
)

DEEPSEEK_MODEL_DIR = Path(
    os.environ.get(
        "CKPT_DEEPSEEK_MODEL_DIR",
        str(Path(__file__).resolve().parents[2] / "third_party/deepseek_v2_lite"),
    )
)


@dataclass(frozen=True)
class Workload:
    name: str
    display_name: str
    config: Any
    model_class: type
    leaf_modules: tuple[type, ...] = ()
    split_method: str | None = None
    parameter_count: int | None = None
    config_source: str = "repository"

    def load_deepseek_pipeline_weights(
        self, pipeline: torch.nn.Module, checkpoint_dir: str | Path
    ) -> dict[str, int]:
        if self.name != "deepseek_v2_lite":
            raise ValueError("pretrained loading is only implemented for DeepSeek-V2-Lite")
        checkpoint_dir = Path(checkpoint_dir)
        index_path = checkpoint_dir / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"missing sharded safetensors index: {index_path}")
        weight_map = json.loads(index_path.read_text())["weight_map"]
        expected_keys = set(weight_map)
        parameters = dict(pipeline.named_parameters())
        by_shard = {}
        for name, parameter in parameters.items():
            if parameter.device.type == "meta":
                raise ValueError(f"cannot load unmaterialized pipeline parameter: {name}")
            layer_name, local_name = name.split(".", 1)
            layer_index = int(layer_name)
            if layer_index == 0:
                if local_name != "weight":
                    raise ValueError(f"unexpected embedding key: {name}")
                checkpoint_key = "model.embed_tokens.weight"
            elif 1 <= layer_index <= self.config.num_hidden_layers:
                if not local_name.startswith("layer."):
                    raise ValueError(f"unexpected DeepSeek pipeline key: {name}")
                checkpoint_key = f"model.layers.{layer_index - 1}.{local_name[6:]}"
            elif layer_index == self.config.num_hidden_layers + 1:
                if local_name != "weight":
                    raise ValueError(f"unexpected final norm key: {name}")
                checkpoint_key = f"model.norm.{local_name}"
            elif layer_index == self.config.num_hidden_layers + 2:
                if not local_name.startswith("lm_head."):
                    raise ValueError(f"unexpected DeepSeek output key: {name}")
                checkpoint_key = local_name
            else:
                raise ValueError(f"pipeline layer index out of range: {name}")
            try:
                shard = weight_map[checkpoint_key]
            except KeyError as error:
                raise ValueError(f"checkpoint is missing {checkpoint_key} for {name}") from error
            by_shard.setdefault(shard, []).append((name, checkpoint_key, parameter))

        loaded = set()
        source_digest = hashlib.sha256()
        with torch.no_grad():
            for shard, entries in by_shard.items():
                shard_path = checkpoint_dir / shard
                if not shard_path.is_file():
                    raise FileNotFoundError(f"missing safetensors shard: {shard_path}")
                with safe_open(str(shard_path), framework="pt", device="cpu") as tensors:
                    for name, key, parameter in entries:
                        source = tensors.get_tensor(key)
                        if source.shape != parameter.shape:
                            raise ValueError(
                                f"shape mismatch for {name} <- {key}: "
                                f"{tuple(source.shape)} != {tuple(parameter.shape)}"
                            )
                        if parameter.device.type == "meta":
                            raise ValueError(f"cannot load unmaterialized parameter: {name}")
                        parameter.copy_(source.to(parameter.device, parameter.dtype))
                        loaded.add(name)
                        source_digest.update(key.encode())
                        source_digest.update(source.contiguous().view(torch.uint8).numpy().tobytes())
        if loaded != set(parameters):
            raise RuntimeError("DeepSeek checkpoint did not initialize every local parameter")
        role_aware = getattr(pipeline, "_merak_role_aware_zero1", False)
        if role_aware and not all(
            hasattr(pipeline, name)
            for name in (
                "_merak_zero1_shared_process_group",
                "_merak_zero1_expert_process_group",
            )
        ):
            raise ValueError("role-aware pretrained loading requires shared and expert groups")
        for name, parameter in parameters.items():
            if role_aware:
                group = (
                    pipeline._merak_zero1_expert_process_group
                    if getattr(parameter, "_merak_expert_parallel", False)
                    else pipeline._merak_zero1_shared_process_group
                )
                dist.broadcast(parameter.detach(), dist.get_global_rank(group, 0), group=group)
        return {
            "loaded_parameters": len(loaded),
            "loaded_shards": len(by_shard),
            "checkpoint_keys": len(expected_keys),
            "loaded_tensor_sha256": source_digest.hexdigest(),
        }

    @property
    def config_hash(self) -> str:
        encoded = json.dumps(
            self.config.to_dict(), sort_keys=True, separators=(",", ":"), default=str
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def build_model(
        self,
        *,
        empty: bool = True,
        expert_parallel_groups: ExpertParallelGroups | None = None,
    ):
        if empty:
            with init_empty_weights():
                model = self.model_class(self.config)
        else:
            model = self.model_class(self.config)
        if expert_parallel_groups is not None and self.name == "qwen2_moe_14b":
            model = shard_qwen2_moe_model(model, expert_parallel_groups)
        elif expert_parallel_groups is not None and self.name == "deepseek_v2_lite":
            model = shard_deepseek_v2_moe_model(model, expert_parallel_groups)
        elif expert_parallel_groups is not None:
            raise NotImplementedError(
                f"expert parallel conversion is not implemented for {self.name}"
            )
        if (
            self.name in {"qwen2_moe_14b", "deepseek_v2_lite"}
            and hasattr(getattr(model, "model", None), "embed_tokens")
            and hasattr(model, "lm_head")
        ):
            object.__setattr__(
                model, "_merak_pipeline_layers", build_moe_pipeline_layers(model)
            )
            object.__setattr__(
                model, "_merak_pipeline_input_to_shard", {"input_ids": 0}
            )
            object.__setattr__(model, "_merak_init_weights", model._init_weights)
        if expert_parallel_groups is not None:
            object.__setattr__(model, "_merak_ep_layout", expert_parallel_groups.layout)
            object.__setattr__(model, "_merak_role_aware_zero1", True)
            if hasattr(expert_parallel_groups, "dp_group"):
                object.__setattr__(
                    model,
                    "_merak_zero1_expert_process_group",
                    expert_parallel_groups.dp_group,
                )
        return model

    def build_dataset(
        self,
        *,
        size: int = 1_000_000,
        seq_length: int | None = None,
        seed: int | None = None,
    ):
        return DynamicGenDataset(
            self.config,
            mode="text_only",
            dataset_size=size,
            seq_length=seq_length,
            seed=seed,
        )

    def metadata(self, *, initialization: str = "random") -> dict[str, object]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "model_type": self.config.model_type,
            "config_hash": self.config_hash,
            "hidden_size": int(self.config.hidden_size),
            "num_hidden_layers": int(self.config.num_hidden_layers),
            "num_attention_heads": int(self.config.num_attention_heads),
            "vocab_size": int(self.config.vocab_size),
            "parameter_count": self.parameter_count,
            "initialization": initialization,
            "config_source": self.config_source,
        }


def _load_deepseek_classes(model_dir: Path) -> tuple[type, type]:
    """Load the pinned local DeepSeek implementation without requiring flash-attn."""
    required = ("config.json", "configuration_deepseek.py", "modeling_deepseek.py")
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"DeepSeek model directory {model_dir} is missing {', '.join(missing)}"
        )
    package = "ckpt_grouping_deepseek_v2"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(model_dir)]
        sys.modules[package] = module
    for name in ("configuration_deepseek", "modeling_deepseek"):
        qualified = f"{package}.{name}"
        if qualified in sys.modules:
            continue
        path = model_dir / f"{name}.py"
        spec = importlib.util.spec_from_file_location(qualified, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load DeepSeek module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    config_class = sys.modules[f"{package}.configuration_deepseek"].DeepseekV2Config
    model_class = sys.modules[f"{package}.modeling_deepseek"].DeepseekV2ForCausalLM
    return config_class, model_class


def build_tiny_workload(name: str, layers: int = 1) -> Workload:
    """Return a small random-initialized MoE workload for correctness tests."""
    if layers < 1:
        raise ValueError("tiny MoE layer count must be positive")
    workload = get_workload(name)
    if name == "qwen2_moe_14b":
        config = Qwen2MoeConfig(
            vocab_size=31,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=16,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=16,
            decoder_sparse_step=1,
            use_cache=False,
            output_router_logits=False,
            tie_word_embeddings=False,
            return_dict=False,
        )
        config._attn_implementation = "eager"
        attention = Qwen2MoeAttention
        experts = Qwen2MoeExpertParallelBlock
        family = "Qwen2-MoE"
    elif name == "deepseek_v2_lite":
        config_class, _ = _load_deepseek_classes(DEEPSEEK_MODEL_DIR)
        config = config_class(
            vocab_size=31,
            hidden_size=16,
            intermediate_size=24,
            moe_intermediate_size=8,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=4,
            n_shared_experts=1,
            n_routed_experts=4,
            ep_size=1,
            routed_scaling_factor=1.0,
            kv_lora_rank=4,
            q_lora_rank=None,
            qk_rope_head_dim=4,
            v_head_dim=4,
            qk_nope_head_dim=4,
            topk_method="greedy",
            n_group=1,
            topk_group=1,
            num_experts_per_tok=2,
            moe_layer_freq=1,
            first_k_dense_replace=0,
            max_position_embeddings=16,
            use_cache=False,
            return_dict=False,
            tie_word_embeddings=False,
        )
        config._attn_implementation = "eager"
        attention = sys.modules[
            "ckpt_grouping_deepseek_v2.modeling_deepseek"
        ].DeepseekV2Attention
        experts = DeepseekV2ExpertParallelBlock
        family = "DeepSeek-V2"
    else:
        raise ValueError("tiny workloads support Qwen2-MoE and DeepSeek-V2-Lite")
    return dataclass_replace(
        workload,
        display_name=f"tiny {family} correctness workload",
        config=config,
        leaf_modules=(attention, experts),
        parameter_count=None,
        config_source=f"build_tiny_workload({name!r})",
    )


def get_workload(name: str) -> Workload:
    if name == "bert":
        config = BertConfig(**load_bert_config("bert-large-uncased"))
        return Workload(name, "BERT-4L-1024", config, BertForMaskedLM)
    if name == "gpt2":
        config = GPT2Config(**load_gpt2_config("gpt2"))
        return Workload(name, "GPT-2", config, GPT2LMHeadModel)
    if name == "opt":
        config = OPTConfig(**load_opt_config("opt-350m"))
        return Workload(
            name,
            "OPT-350M",
            config,
            OPTForCausalLM,
            leaf_modules=(OPTAttention,),
            split_method="layer_split",
        )
    if name in {"llama_1b", "llama_3b"}:
        sizes = {
            "llama_1b": {
                "display_name": "LLaMA-like-1B",
                "hidden_size": 2048,
                "intermediate_size": 5504,
                "num_hidden_layers": 16,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
            },
            "llama_3b": {
                "display_name": "LLaMA-like-3B",
                "hidden_size": 3072,
                "intermediate_size": 8192,
                "num_hidden_layers": 28,
                "num_attention_heads": 24,
                "num_key_value_heads": 8,
            },
        }
        size = sizes[name]
        config = LlamaConfig(
            architectures=["LlamaForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=size["hidden_size"],
            intermediate_size=size["intermediate_size"],
            max_position_embeddings=2048,
            model_type="llama",
            num_attention_heads=size["num_attention_heads"],
            num_hidden_layers=size["num_hidden_layers"],
            num_key_value_heads=size["num_key_value_heads"],
            pad_token_id=0,
            rms_norm_eps=1e-6,
            tie_word_embeddings=False,
            use_cache=False,
            vocab_size=32000,
            return_dict=False,
        )
        config._attn_implementation = "eager"
        return Workload(
            name,
            size["display_name"],
            config,
            LlamaForCausalLM,
            leaf_modules=(LlamaAttention,),
            split_method="layer_split",
        )
    if name == "llama_3_8b":
        config = LlamaConfig(
            architectures=["LlamaForCausalLM"],
            attention_bias=False,
            attention_dropout=0.0,
            bos_token_id=128000,
            eos_token_id=128001,
            hidden_act="silu",
            hidden_size=4096,
            intermediate_size=14336,
            max_position_embeddings=8192,
            mlp_bias=False,
            num_attention_heads=32,
            num_hidden_layers=32,
            num_key_value_heads=8,
            pad_token_id=128002,
            rms_norm_eps=1e-5,
            rope_theta=500000.0,
            tie_word_embeddings=False,
            use_cache=False,
            vocab_size=128256,
            return_dict=False,
        )
        config._attn_implementation = "eager"
        return Workload(
            name,
            "LLaMA-3-8B architecture",
            config,
            LlamaForCausalLM,
            leaf_modules=(LlamaAttention,),
            split_method="layer_split",
            parameter_count=8_030_261_248,
            config_source="Meta-Llama-3-8B config",
        )
    if name == "gpt_6b":
        config = GPT2Config(
            architectures=["GPT2LMHeadModel"],
            n_embd=4096,
            n_inner=14336,
            n_layer=32,
            n_head=32,
            n_positions=2048,
            n_ctx=2048,
            vocab_size=50304,
            tie_word_embeddings=False,
            use_cache=False,
            return_dict=False,
        )
        config._attn_implementation = "eager"
        return Workload(
            name,
            "GPT dense 6.3B architecture",
            config,
            GPT2LMHeadModel,
            leaf_modules=(GPT2Attention,),
            split_method="layer_split",
            parameter_count=6_327_705_600,
            config_source="AdaCheck GPT-7B-scale dimensions",
        )
    if name == "qwen2_moe_14b":
        config = Qwen2MoeConfig(
            architectures=["Qwen2MoeForCausalLM"],
            hidden_size=2048,
            intermediate_size=5632,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=16,
            vocab_size=151936,
            num_experts=60,
            num_experts_per_tok=4,
            moe_intermediate_size=1408,
            shared_expert_intermediate_size=5632,
            decoder_sparse_step=1,
            tie_word_embeddings=False,
            use_cache=False,
            return_dict=False,
        )
        config._attn_implementation = "eager"
        return Workload(
            name,
            "Qwen2-MoE 14.3B architecture",
            config,
            Qwen2MoeForCausalLM,
            leaf_modules=(Qwen2MoeAttention, Qwen2MoeExpertParallelBlock),
            split_method="layer_split",
            parameter_count=14_315_784_192,
            config_source="transformers Qwen2-MoE public config",
        )
    if name == "deepseek_v2_lite":
        config_class, model_class = _load_deepseek_classes(DEEPSEEK_MODEL_DIR)
        config = config_class.from_json_file(str(DEEPSEEK_MODEL_DIR / "config.json"))
        config.use_cache = False
        config.return_dict = False
        config._attn_implementation = "eager"
        attention_class = sys.modules[
            "ckpt_grouping_deepseek_v2.modeling_deepseek"
        ].DeepseekV2Attention
        return Workload(
            name,
            "DeepSeek-V2-Lite architecture",
            config,
            model_class,
            leaf_modules=(attention_class, DeepseekV2ExpertParallelBlock),
            split_method="layer_split",
            parameter_count=15_706_484_224,
            config_source=str(DEEPSEEK_MODEL_DIR),
        )
    raise ValueError(
        f"unsupported workload {name!r}; expected one of {SUPPORTED_WORKLOADS}"
    )


def validate_parallelism(
    *, world_size: int, pipeline_parallel: int, tensor_parallel: int, data_parallel: int
) -> None:
    dimensions = (pipeline_parallel, tensor_parallel, data_parallel)
    if any(value < 1 for value in dimensions):
        raise ValueError("parallel dimensions must be positive")
    product = pipeline_parallel * tensor_parallel * data_parallel
    if product != world_size:
        raise ValueError(
            f"pp*tp*dp={product} does not match world_size={world_size}"
        )
