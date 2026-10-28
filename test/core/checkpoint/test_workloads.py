import pytest
from types import SimpleNamespace
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.models.opt.modeling_opt import OPTAttention

from Merak.core.moe import ExpertParallelLayout, Qwen2MoeExpertParallelBlock
from Merak.core.tensor_parallel.mp_mapping import MP_MODEL_MAPPING
from Merak.core.tensor_parallel.mp_attrs import get_tp_attr_list, set_mp_attr
from examples.checkpoint_grouping.run_workload import configure_tensor_parallel
from examples.checkpoint_grouping.workloads import (
    SUPPORTED_WORKLOADS,
    build_tiny_workload,
    get_workload,
    validate_parallelism,
)


def test_all_research_workloads_have_honest_local_configs():
    metadata = {name: get_workload(name).metadata() for name in SUPPORTED_WORKLOADS}

    assert metadata["bert"]["num_hidden_layers"] == 4
    assert metadata["gpt2"]["num_hidden_layers"] == 12
    assert metadata["opt"]["num_hidden_layers"] == 24
    assert metadata["llama_1b"]["num_hidden_layers"] == 16
    assert metadata["llama_3b"]["num_hidden_layers"] == 28
    assert metadata["llama_3_8b"]["parameter_count"] == 8_030_261_248
    assert metadata["gpt_6b"]["parameter_count"] == 6_327_705_600
    assert metadata["qwen2_moe_14b"]["parameter_count"] == 14_315_784_192
    assert metadata["deepseek_v2_lite"]["parameter_count"] == 15_706_484_224
    assert all(item["initialization"] == "random" for item in metadata.values())
    assert len({item["config_hash"] for item in metadata.values()}) == len(
        SUPPORTED_WORKLOADS
    )


def test_llama_3b_config_has_the_declared_parameter_scale():
    model = get_workload("llama_3b").build_model(empty=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert parameter_count == 3_015_355_392


@pytest.mark.parametrize(
    ("name", "expected"),
    (
        ("llama_3_8b", 8_030_261_248),
        ("gpt_6b", 6_327_705_600),
        ("qwen2_moe_14b", 14_315_784_192),
        ("deepseek_v2_lite", 15_706_484_224),
    ),
)
def test_paper_scale_configs_have_declared_parameter_counts(name, expected):
    workload = get_workload(name)
    model = workload.build_model(empty=True)

    assert sum(parameter.numel() for parameter in model.parameters()) == expected


def test_opt_attention_is_an_fx_leaf_module():
    workload = get_workload("opt")
    assert workload.leaf_modules == (OPTAttention,)
    assert workload.split_method == "layer_split"


def test_gpt_6b_attention_is_an_fx_leaf_module():
    workload = get_workload("gpt_6b")
    assert workload.leaf_modules == (GPT2Attention,)
    assert workload.split_method == "layer_split"
    assert workload.config._attn_implementation == "eager"
    assert type(workload.build_model(empty=True).transformer.h[0].attn) is GPT2Attention


def test_gpt_6b_mlp_ratio_has_a_tensor_parallel_mapping():
    mapping = MP_MODEL_MAPPING["gpt2"]["input_output_mapping"]
    assert (2, 7, "col") in mapping
    assert (7, 2, "row") in mapping


def test_qwen_expert_parallel_build_keeps_only_owned_experts():
    groups = SimpleNamespace(
        layout=ExpertParallelLayout(1, 2, 1, 1), ep_rank=1, ep_group=None
    )
    workload = get_workload("qwen2_moe_14b")
    model = workload.build_model(empty=True, expert_parallel_groups=groups)
    blocks = [
        module
        for module in model.modules()
        if isinstance(module, Qwen2MoeExpertParallelBlock)
    ]

    assert len(blocks) == workload.config.num_hidden_layers
    assert all(tuple(block.local_experts) == tuple(range(30, 60)) for block in blocks)
    assert Qwen2MoeExpertParallelBlock in workload.leaf_modules


def test_dataset_uses_the_requested_experiment_sequence_length():
    assert get_workload("opt").build_dataset(seq_length=128).sequence_length == 128


@pytest.mark.parametrize("name", ("qwen2_moe_14b", "deepseek_v2_lite"))
def test_tiny_moe_workload_has_small_random_ep_architecture(name):
    workload = build_tiny_workload(name, layers=2)
    model = workload.build_model(empty=False)

    assert workload.config.num_hidden_layers == 2
    assert workload.config.vocab_size == 31
    assert workload.config_source.startswith("build_tiny_workload")
    assert len(model._merak_pipeline_layers) == 5
    assert workload.parameter_count is None
    experts = getattr(workload.config, "num_experts", None)
    if experts is None:
        experts = workload.config.n_routed_experts
    assert experts == 4
    assert workload.config.num_experts_per_tok == 2


def test_tiny_moe_workload_rejects_invalid_model_or_layer_count():
    with pytest.raises(ValueError, match="positive"):
        build_tiny_workload("qwen2_moe_14b", layers=0)
    with pytest.raises(ValueError, match="support Qwen2-MoE"):
        build_tiny_workload("gpt2")


def test_opt_tensor_parallel_updates_attention_width():
    configure_tensor_parallel("opt")
    assert get_tp_attr_list() == ["num_heads", "embed_dim"]
    model = get_workload("opt").build_model(empty=True)
    set_mp_attr(model, 2)
    attention = model.model.decoder.layers[0].self_attn
    assert (attention.num_heads, attention.embed_dim) == (8, 512)
    assert attention.q_proj.out_features == 512
    assert attention.out_proj.in_features == 512


def test_llama_3b_tensor_parallel_updates_attention_heads():
    configure_tensor_parallel("llama_3b")
    assert get_tp_attr_list() == ["num_heads", "num_key_value_heads"]
    model = get_workload("llama_3b").build_model(empty=True)
    set_mp_attr(model, 2)
    attention = model.model.layers[0].self_attn
    assert (attention.num_heads, attention.num_key_value_heads) == (12, 4)


def test_parallel_dimensions_must_match_the_launched_world():
    validate_parallelism(
        world_size=16, pipeline_parallel=4, tensor_parallel=2, data_parallel=2
    )

    with pytest.raises(ValueError, match="does not match"):
        validate_parallelism(
            world_size=8, pipeline_parallel=4, tensor_parallel=2, data_parallel=2
        )
