import json

import torch
from safetensors.torch import save_file

from examples.checkpoint_grouping.workloads import build_tiny_workload
from Merak.core.moe.expert_parallel import ExpertParallelLayout
from types import SimpleNamespace


def _export_hf_checkpoint(model, path):
    state = {}
    for key, value in model.state_dict().items():
        if key.startswith("model.layers."):
            parts = key.split(".")
            if parts[3] == "mlp" and parts[4] == "experts":
                parts[5] = f"experts.{parts[5]}"
                key = ".".join(parts[:4] + [parts[5]] + parts[6:])
        state[key] = value.detach().cpu().contiguous()
    save_file(state, path / "model.safetensors")
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "model.safetensors" for key in state}})
    )
    return state


def test_deepseek_pretrained_loader_roundtrips_pipeline_and_ep_local_experts(tmp_path):
    workload = build_tiny_workload("deepseek_v2_lite", layers=2)
    source = workload.build_model(empty=False)
    expected = _export_hf_checkpoint(source, tmp_path)

    groups = SimpleNamespace(
        layout=ExpertParallelLayout(1, 2, 1, 1), ep_rank=1, ep_group=None
    )
    target = workload.build_model(empty=False, expert_parallel_groups=groups)
    pipeline = torch.nn.Module()
    for index, layer in enumerate(target._merak_pipeline_layers):
        pipeline.add_module(str(index), layer)
    pipeline._layer_specs = target._merak_pipeline_layers
    pipeline._merak_role_aware_zero1 = False

    loaded = workload.load_deepseek_pipeline_weights(pipeline, tmp_path)
    assert loaded["loaded_parameters"] == len(dict(pipeline.named_parameters()))
    assert loaded["loaded_shards"] == 1
    for name, value in pipeline.state_dict().items():
        layer_index, local_name = name.split(".", 1)
        layer_index = int(layer_index)
        if layer_index == 0:
            key = "model.embed_tokens.weight"
        elif 1 <= layer_index <= 2:
            key = f"model.layers.{layer_index - 1}.{local_name.removeprefix('layer.') }"
        elif layer_index == 3:
            key = f"model.norm.{local_name}"
        else:
            key = "lm_head.weight"
        assert torch.equal(value, expected[key])
    experts = [name for name in pipeline.state_dict() if ".experts." in name]
    assert experts
    assert all(".experts.2." in name or ".experts.3." in name for name in experts)
