import copy
import tempfile
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM

from Merak.core.checkpoint.zero1_adapter import (
    iter_zero1_shards,
    zero1_semantic_compatibility,
)
from Merak.core.moe.expert_parallel import (
    DeepseekV2ExpertParallelBlock,
    ExpertParallelLayout,
    Qwen2MoeExpertParallelBlock,
    create_expert_parallel_groups,
    create_shared_parameter_group,
    build_moe_pipeline_layers,
    inspect_expert_partition,
    dispatch_to_experts,
    shard_deepseek_v2_moe_model,
)
from examples.checkpoint_grouping.workloads import Workload


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_top6_dispatch_repeats_bitwise_forward_and_backward():
    torch.manual_seed(20261028)
    hidden = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    selected = torch.rand(256, 8, device="cuda").topk(6, dim=-1).indices
    weights = torch.softmax(torch.randn(256, 6, device="cuda"), dim=-1)
    experts = {i: nn.Linear(128, 128, bias=False).cuda().bfloat16() for i in range(8)}
    upstream = torch.randn_like(hidden)
    reference = None
    for _ in range(10):
        values = hidden.detach().clone().requires_grad_()
        routing = weights.detach().clone().requires_grad_()
        for expert in experts.values():
            expert.zero_grad(set_to_none=True)
        output, _ = dispatch_to_experts(
            values, selected, routing, experts=experts, num_experts=8,
            ep_rank=0, ep_group=None, ep_size=1,
        )
        output.backward(upstream)
        result = [output.detach(), values.grad, routing.grad]
        result += [expert.weight.grad for expert in experts.values()]
        if reference is None:
            reference = [value.clone() for value in result]
        else:
            for index, (before, after) in enumerate(zip(reference, result)):
                assert torch.equal(before, after), f"repeated dispatch differs: component {index}"


class _TinyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2, bias=False)

    def forward(self, values):
        return torch.tanh(self.proj(values))


class _TinyQwenBlock(nn.Module):
    def __init__(self, *, skewed=False):
        super().__init__()
        self.num_experts = 4
        self.top_k = 2
        self.norm_topk_prob = True
        self.gate = nn.Linear(2, 4, bias=False)
        self.experts = nn.ModuleList([_TinyExpert() for _ in range(4)])
        self.shared_expert = _TinyExpert()
        self.shared_expert_gate = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            if skewed:
                self.gate.weight.copy_(
                    torch.tensor([[8.0, 8.0], [6.0, 7.0], [0.0, -8.0], [-8.0, -8.0]])
                )
            else:
                self.gate.weight.copy_(
                    torch.tensor([[4.0, 0.0], [0.0, 4.0], [-4.0, 0.0], [0.0, -4.0]])
                )

    def forward(self, hidden_states):
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        logits = self.gate(flat)
        weights, selected = torch.softmax(logits, dim=-1).topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        routed = torch.zeros_like(flat)
        token_ids = torch.arange(flat.shape[0], device=flat.device).repeat_interleave(self.top_k)
        selected = selected.reshape(-1)
        weights = weights.reshape(-1)
        for expert_id, expert in enumerate(self.experts):
            assignments = torch.nonzero(selected == expert_id, as_tuple=False).flatten()
            indices = token_ids.index_select(0, assignments)
            if indices.numel():
                values = expert(flat.index_select(0, indices)) * weights.index_select(
                    0, assignments
                ).unsqueeze(-1)
                routed = routed.index_add(0, indices, values)
        shared = torch.sigmoid(self.shared_expert_gate(flat)) * self.shared_expert(flat)
        return (routed + shared).reshape(shape), logits


class _TestAuxLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, loss):
        ctx.required_aux_loss = loss.requires_grad
        return values

    @staticmethod
    def backward(ctx, gradient):
        return gradient, torch.ones(1, device=gradient.device) if ctx.required_aux_loss else None


class _TinyDeepseekGate(nn.Module):
    def __init__(self, *, skewed=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(4, 2))
        with torch.no_grad():
            if skewed:
                self.weight.copy_(
                    torch.tensor([[8.0, 8.0], [6.0, 7.0], [0.0, -8.0], [-8.0, -8.0]])
                )
            else:
                self.weight.copy_(
                    torch.tensor([[4.0, 0.0], [0.0, 4.0], [-4.0, 0.0], [0.0, -4.0]])
                )

    def forward(self, hidden_states):
        scores = torch.softmax(hidden_states.reshape(-1, 2) @ self.weight.t(), dim=-1)
        weights, selected = scores.topk(2, dim=-1, sorted=False)
        return selected, weights, scores.mean() * 0.03


class DeepseekV2MoE(nn.Module):
    def __init__(self, config=None, *, skewed=False):
        super().__init__()
        self.config = config or SimpleNamespace(
            n_routed_experts=4,
            n_shared_experts=1,
        )
        self.num_experts_per_tok = 2
        self.gate = _TinyDeepseekGate(skewed=skewed)
        self.experts = nn.ModuleList([_TinyExpert() for _ in range(4)])
        self.shared_experts = _TinyExpert()

    def forward(self, hidden_states):
        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        selected, weights, aux_loss = self.gate(hidden_states)
        routed = torch.zeros_like(flat)
        token_ids = torch.arange(flat.shape[0], device=flat.device).repeat_interleave(2)
        flat_selected = selected.reshape(-1)
        flat_weights = weights.reshape(-1)
        for expert_id, expert in enumerate(self.experts):
            assignments = torch.nonzero(
                flat_selected == expert_id, as_tuple=False
            ).flatten()
            indices = token_ids.index_select(0, assignments)
            if indices.numel():
                values = expert(flat.index_select(0, indices)) * flat_weights.index_select(
                    0, assignments
                ).unsqueeze(-1)
                routed = routed.index_add(0, indices, values)
        routed = _TestAuxLoss.apply(routed.reshape(shape), aux_loss)
        return routed + self.shared_experts(hidden_states)


def _assert_close(left, right):
    torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)


def test_qwen_pipeline_layers_match_transformers_logits_loss_and_gradients():
    config = Qwen2MoeConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
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
        return_dict=False,
    )
    config._attn_implementation = "eager"
    model = Qwen2MoeForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 4, 7, 2], [3, 5, 8, 6]])
    labels = torch.tensor([[1, 4, 7, 2], [3, 5, 8, 6]])

    expected_logits = model(input_ids)[0]
    expected_loss = nn.functional.cross_entropy(
        expected_logits[..., :-1, :].reshape(-1, config.vocab_size),
        labels[..., 1:].reshape(-1),
    )
    expected_loss.backward()
    expected_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)

    layers = build_moe_pipeline_layers(model)
    assert len({id(parameter) for layer in layers for parameter in layer.parameters()}) == len(
        {id(parameter) for parameter in model.parameters()}
    )
    hidden = input_ids
    for layer in layers:
        hidden = layer(hidden)
    torch.testing.assert_close(hidden, expected_logits, atol=1e-6, rtol=1e-5)
    actual_loss = nn.functional.cross_entropy(
        hidden[..., :-1, :].reshape(-1, config.vocab_size),
        labels[..., 1:].reshape(-1),
    )
    actual_loss.backward()
    for name, parameter in model.named_parameters():
        if name in expected_gradients:
            torch.testing.assert_close(
                parameter.grad, expected_gradients[name], atol=1e-6, rtol=1e-5
            )


def test_tiny_deepseek_pipeline_matches_model_logits_loss_and_gradients():
    from examples.checkpoint_grouping.workloads import build_tiny_workload

    workload = build_tiny_workload("deepseek_v2_lite", layers=2)
    model = workload.build_model(empty=False).eval()
    input_ids = torch.tensor([[1, 4, 7, 2], [3, 5, 8, 6]])
    labels = input_ids.clone()

    expected_logits = model(input_ids)[0]
    expected_loss = nn.functional.cross_entropy(
        expected_logits[..., :-1, :].reshape(-1, workload.config.vocab_size),
        labels[..., 1:].reshape(-1),
    )
    expected_loss.backward()
    expected_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)

    hidden = input_ids
    for layer in model._merak_pipeline_layers:
        hidden = layer(hidden)
    torch.testing.assert_close(hidden, expected_logits, atol=1e-6, rtol=1e-5)
    actual_loss = nn.functional.cross_entropy(
        hidden[..., :-1, :].reshape(-1, workload.config.vocab_size),
        labels[..., 1:].reshape(-1),
    )
    actual_loss.backward()
    for name, parameter in model.named_parameters():
        if name in expected_gradients:
            torch.testing.assert_close(
                parameter.grad, expected_gradients[name], atol=1e-6, rtol=1e-5
            )


def test_qwen_pipeline_layer_builds_without_fx_input_wrapping(monkeypatch):
    from Merak.core import mpu
    from Merak.core.pipeline.module import PipelineModule
    from Merak.core.recompute import checkpointing

    config = Qwen2MoeConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
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
    )
    config._attn_implementation = "eager"
    model = Qwen2MoeForCausalLM(config).eval()
    object.__setattr__(model, "_merak_pipeline_layers", build_moe_pipeline_layers(model))
    object.__setattr__(model, "_merak_pipeline_input_to_shard", {"input_ids": 0})
    object.__setattr__(model, "_merak_init_weights", model._init_weights)

    monkeypatch.setattr(mpu, "get_pipe_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_pipe_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "get_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_sequence_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mpu, "get_sequence_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "get_data_parallel_rank", lambda: 0)
    monkeypatch.setattr(mpu, "get_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(checkpointing.torch.distributed, "get_rank", lambda **kwargs: 0)

    class Grid:
        def stage_to_global(self, stage_id):
            return stage_id

    args = SimpleNamespace(
        partition_method="uniform_floor",
        split_method="layer_split",
        custom_partition=None,
        trace_method="fx",
        fp16=False,
        bf16=False,
        half_precision_backend="none",
        device=torch.device("cpu"),
        num_layers=1,
        seed=1234,
        init_method_std=0.02,
        tp_overlapping_level=0,
        pytorch_tp=False,
        cache_sharding=False,
        activation_checkpointing=False,
        activation_checkpoint_ratio=None,
        checkpoint_num_layers=0,
        no_tie_modules=True,
        parallel_vocab=False,
        get_communication_grid=lambda: Grid(),
    )

    def initialize_parallel_properties(self):
        self._grid = Grid()
        self.global_rank = self.world_size = self.local_rank = 0
        self.model_parallel = self.sequence_parallel = False
        self.stage_id = 0
        self.num_stages = 1
        self.micro_offset = 0

    monkeypatch.setattr(
        PipelineModule,
        "_initialize_parallel_properties",
        initialize_parallel_properties,
    )
    monkeypatch.setattr(PipelineModule, "configure_distributed_model", lambda self: None)
    monkeypatch.setattr(PipelineModule, "_synchronize_tied_weights", lambda self: None)
    monkeypatch.setattr("Merak.core.pipeline.module.dist.new_group", lambda **kwargs: None)
    monkeypatch.setattr("Merak.core.pipeline.module.dist.get_world_size", lambda **kwargs: 1)

    pipeline = PipelineModule(model, args, loss_fn=lambda outputs, labels: outputs.mean())
    input_ids = torch.tensor([[1, 4, 7, 2]])
    expected = model(input_ids)[0]
    actual = pipeline(input_ids)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_expert_parallel_pipeline_rejects_fx_fallback(monkeypatch):
    from Merak.core.moe.expert_parallel import ExpertParallelLayout
    from Merak.core.pipeline import module as pipeline_module
    from Merak.core.pipeline.module import PipelineModule

    config = Qwen2MoeConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
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
    )
    config._attn_implementation = "eager"
    model = Qwen2MoeForCausalLM(config).eval()
    object.__setattr__(
        model, "_merak_ep_layout", ExpertParallelLayout(1, 2, 1, 1)
    )

    monkeypatch.setattr(pipeline_module, "ModuleRebuild", lambda *args: object())
    pipeline = PipelineModule.__new__(PipelineModule)
    pipeline.args = SimpleNamespace()
    pipeline.model_parallel = False
    pipeline._traced_module = lambda *args: pytest.fail("FX tracing must not run")
    with pytest.raises(ValueError, match="expert-parallel models require explicit"):
        pipeline._initialize_layers(model, (), lambda *args: None)


def test_qwen_merak_trainer_single_rank_optimizer_step(tmp_path):
    from Merak import MerakArguments, MerakTrainer
    from Merak.core import mpu
    from Merak.core.mpu import initialize as mpu_initialize
    from Merak.core.mpu.topology import (
        PipeDataSequenceModelParallelTopology,
        PipelineParallelGrid,
    )

    os.environ.update(DP="1", PP="1", TP="1", SP="1", LOCAL_RANK="0")
    if not dist.is_initialized():
        rendezvous = tmp_path / "gloo-rendezvous"
        dist.init_process_group(
            "gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1
        )

    config = Qwen2MoeConfig(
        vocab_size=31,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
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
    )
    config._attn_implementation = "eager"
    model = Qwen2MoeForCausalLM(config)
    object.__setattr__(model, "_merak_pipeline_layers", build_moe_pipeline_layers(model))
    object.__setattr__(model, "_merak_pipeline_input_to_shard", {"input_ids": 0})
    object.__setattr__(model, "_merak_init_weights", model._init_weights)

    grid = PipelineParallelGrid(PipeDataSequenceModelParallelTopology(1, 1, 1, 1))
    mpu_initialize.set_data_parallel_group(grid.get_data_parallel_group())
    mpu_initialize.set_model_parallel_group(grid.get_slice_parallel_group())
    mpu_initialize.set_pipe_parallel_group(grid.get_pipe_parallel_group())
    mpu_initialize.set_sequence_parallel_group(grid.get_sequence_parallel_group())

    class TinyQwenDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, index):
            ids = torch.tensor([1, 4, 7, 2]) if index % 2 == 0 else torch.tensor([3, 5, 8, 6])
            return {"input_ids": ids, "labels": ids.clone()}

    args = MerakArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        max_steps=1,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        lr_scheduler_type="constant",
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
        data_seed=7,
        seed=7,
        seq_length=4,
        num_layers=1,
        hidden_size=16,
        num_heads=4,
        zero_stage=None,
        save=False,
        ada_checkpoint=False,
        activation_checkpointing=False,
        checkpoint_num_layers=0,
        no_tie_modules=True,
        parallel_vocab=False,
        split_method="layer_split",
        partition_method="uniform_floor",
        trace_method="fx",
        half_precision_backend=None,
        input_names=["input_ids"],
    )
    args.get_communication_grid = lambda: grid
    # Exercise pipeline ownership, batching, loss, backward, and the real AdamW update.
    trainer = MerakTrainer(model, args, train_dataset=TinyQwenDataset())
    before = [parameter.detach().clone() for parameter in trainer.model.parameters()]
    trainer.train()
    assert trainer.tuning_params.global_steps == 1
    assert any(
        not torch.equal(old, current)
        for old, current in zip(before, trainer.engine.module.parameters())
    )
    dist.destroy_process_group()


class _TinyDistributedQwenDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, index):
        ids = torch.tensor([1, 4, 7, 2]) if index % 2 == 0 else torch.tensor([3, 5, 8, 6])
        return {"input_ids": ids, "labels": ids.clone()}


def _qwen_role_aware_trainer_worker(rank, init_method, output_dir):
    from datetime import timedelta

    from Merak import MerakArguments, MerakTrainer
    from Merak.core import mpu
    from Merak.core.mpu import initialize as mpu_initialize
    from Merak.core.mpu.topology import (
        PipeDataSequenceModelParallelTopology,
        PipelineParallelGrid,
    )
    from Merak.core.moe.zero1_optimizer import RoleAwareZeroOptimizerStage1

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    os.environ.update(
        DP="4",
        PP="1",
        TP="1",
        SP="1",
        LOCAL_RANK=str(rank),
        MERAK_MESH_ORDER="dp,pp,sp,tp",
    )
    dist.init_process_group(
        "nccl",
        init_method=init_method,
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=180),
    )
    try:
        layout = ExpertParallelLayout(
            1, 2, 2, 1, order=("dp", "ep", "pp", "tp")
        )
        groups = create_expert_parallel_groups(layout)
        shared_parameter_group = create_shared_parameter_group(layout)
        grid = PipelineParallelGrid(
            PipeDataSequenceModelParallelTopology(1, 4, 1, 1)
        )
        mpu_initialize.set_data_parallel_group(grid.get_data_parallel_group())
        mpu_initialize.set_model_parallel_group(grid.get_slice_parallel_group())
        mpu_initialize.set_pipe_parallel_group(grid.get_pipe_parallel_group())
        mpu_initialize.set_sequence_parallel_group(grid.get_sequence_parallel_group())
        torch.manual_seed(73)
        config = Qwen2MoeConfig(
            vocab_size=31,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=1,
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
        )
        config._attn_implementation = "eager"
        workload = Workload(
            "qwen2_moe_14b", "tiny Qwen2-MoE", config, Qwen2MoeForCausalLM
        )
        model = workload.build_model(empty=False, expert_parallel_groups=groups)
        model._merak_sampler_data_parallel_rank = groups.dp_rank
        model._merak_sampler_data_parallel_size = layout.dp
        model._merak_replica_data_parallel_group = groups.dp_group
        model._merak_zero1_shared_process_group = shared_parameter_group
        model._merak_zero1_expert_process_group = groups.dp_group

        args = MerakArguments(
            output_dir=output_dir,
            use_cpu=False,
            local_rank=rank,
            max_steps=1,
            num_train_epochs=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            lr_scheduler_type="constant",
            logging_strategy="no",
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            dataloader_num_workers=0,
            seed=73,
            data_seed=73,
            seq_length=4,
            num_layers=1,
            hidden_size=16,
            num_heads=4,
            zero_stage=1,
            zero_reduce_bucket_size=256,
            zero_allgather_bucket_size=256,
            save=False,
            ada_checkpoint=False,
            activation_checkpointing=True,
            checkpoint_num_layers=1,
            no_tie_modules=True,
            parallel_vocab=False,
            split_method="layer_split",
            partition_method="uniform_floor",
            trace_method="fx",
            half_precision_backend=None,
            input_names=["input_ids"],
        )
        args.get_communication_grid = lambda: grid
        trainer = MerakTrainer(
            model, args, train_dataset=_TinyDistributedQwenDataset()
        )
        assert trainer.model._merak_pipeline_model
        assert len(trainer.model._layer_specs) == 4
        assert all(
            not isinstance(layer, torch.fx.GraphModule)
            for layer in trainer.model._layer_specs
        )
        before = trainer.model.forward_funcs[1].layer.self_attn.q_proj.weight.detach().clone()
        trainer.train()

        optimizer = trainer.engine.optimizer
        assert isinstance(optimizer, RoleAwareZeroOptimizerStage1)
        assert set(optimizer.group_roles) == {"shared", "expert"}
        assert set(optimizer.partition_counts) == {2, 4}
        assert optimizer.global_step == trainer.tuning_params.global_steps == 1
        assert torch.isfinite(trainer.engine.agg_train_loss)
        after = trainer.model.forward_funcs[1].layer.self_attn.q_proj.weight.detach()
        assert not torch.equal(before, after)

        expert_block = next(
            module
            for module in trainer.engine.module.modules()
            if isinstance(module, Qwen2MoeExpertParallelBlock)
        )
        assert expert_block.last_routing is not None
        assert expert_block.last_routing.sent_tokens == 8

        shared_copies = [torch.empty_like(after) for _ in range(4)]
        dist.all_gather(shared_copies, after)
        assert all(torch.equal(shared_copies[0], value) for value in shared_copies[1:])

        expert_parameter = next(
            expert_block.local_experts[min(expert_block.local_experts)].parameters()
        ).detach()
        expert_copies = [torch.empty_like(expert_parameter) for _ in range(2)]
        dist.all_gather(expert_copies, expert_parameter, group=groups.dp_group)
        assert torch.equal(expert_copies[0], expert_copies[1])

        if os.environ.get("CKPT_TEST_REAL_MOE_DETECTOR") == "1":
            from Merak.core.moe.semantic_adapter import (
                build_moe_optimizer_manifest,
                capture_moe_adacheck_groups,
                moe_detector_audit_report,
            )
            module = trainer.engine.module
            topology = layout.coordinates(rank)
            host_by_rank = {item: item // 2 for item in range(dist.get_world_size())}
            filtered = capture_moe_adacheck_groups(module, host_by_rank=host_by_rank)
            aliases = filtered.model_aliases().get(rank, {})
            report = moe_detector_audit_report(filtered)
            assert report["placement_applied"] is False
            assert filtered.evidence
            if aliases:
                from Merak.core.checkpoint.zero1_recovery_controller import capture_zero1_rank

                captured, _ = capture_zero1_rank(
                    module,
                    optimizer,
                    trainer.engine.lr_scheduler,
                    rank=rank,
                    host=host_by_rank[rank],
                    epoch=optimizer.global_step,
                    sampler=None,
                    storage_hosts=(0, 1),
                    model_aliases=aliases,
                )
                for name, canonical in aliases.items():
                    matches = [
                        item for item in captured
                        if item.address.semantic_id == f"{canonical}::model"
                        and item.address.role == "model"
                        and item.source_rank == rank
                    ]
                    assert len(matches) == 1, name
            manifest = build_moe_optimizer_manifest(
                module,
                optimizer,
                rank=rank,
                shared_group=dist.get_process_group_ranks(
                    module._merak_zero1_shared_process_group
                ),
                expert_group=dist.get_process_group_ranks(
                    module._merak_zero1_expert_process_group
                ),
            )
            assert manifest
            assert all(item.rank == rank for item in manifest)
            assert any(item.expert_id is not None for item in manifest)
            assert topology["ep"] >= 0
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not dist.is_nccl_available(),
    reason="CUDA/NCCL is unavailable",
)
def test_qwen_merak_trainer_ep2_dp2_activation_checkpoint_zero1(tmp_path):
    if torch.cuda.device_count() < 4:
        pytest.skip("four CUDA devices are required")
    mp.spawn(
        _qwen_role_aware_trainer_worker,
        args=(f"file://{tmp_path / 'qwen-ep-rendezvous'}", str(tmp_path)),
        nprocs=4,
        join=True,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not dist.is_nccl_available(),
    reason="CUDA/NCCL is unavailable",
)
def test_qwen_real_adacheck_moe_detector_and_zero1_manifest(tmp_path, monkeypatch):
    if torch.cuda.device_count() < 4:
        pytest.skip("four CUDA devices are required")
    monkeypatch.setenv("CKPT_TEST_REAL_MOE_DETECTOR", "1")
    mp.spawn(
        _qwen_role_aware_trainer_worker,
        args=(f"file://{tmp_path / 'qwen-detector-rendezvous'}", str(tmp_path)),
        nprocs=4,
        join=True,
    )


def _run_ep_case(rank, groups, *, skewed):
    torch.manual_seed(91)
    reference = _TinyQwenBlock(skewed=skewed)
    parallel = Qwen2MoeExpertParallelBlock(copy.deepcopy(reference), groups)
    if skewed:
        values = torch.tensor([[[1.0 + rank, 1.0], [2.0 + rank, 0.5]]])
    else:
        values = (
            torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
            if rank == 0
            else torch.tensor([[[-1.0, 0.0], [0.0, -1.0]]])
        )
    reference_input = values.clone().requires_grad_(True)
    parallel_input = values.clone().requires_grad_(True)
    reference_output, reference_logits = reference(reference_input)
    parallel_output, parallel_logits = parallel(parallel_input)
    _assert_close(parallel_output, reference_output)
    _assert_close(parallel_logits, reference_logits)

    reference_output.square().sum().backward()
    parallel_output.square().sum().backward()
    _assert_close(parallel_input.grad, reference_input.grad)
    for name in ("gate.weight", "shared_expert.proj.weight", "shared_expert_gate.weight"):
        _assert_close(
            dict(parallel.named_parameters())[name].grad,
            dict(reference.named_parameters())[name].grad,
        )

    expected_expert_grads = []
    for reference_expert in reference.experts:
        expected = reference_expert.proj.weight.grad
        if expected is None:
            expected = torch.zeros_like(reference_expert.proj.weight)
        else:
            expected = expected.clone()
        dist.all_reduce(expected, group=groups.ep_group)
        expected_expert_grads.append(expected)
    for expert_id, expert in parallel.local_experts.items():
        expected = expected_expert_grads[expert_id]
        actual = expert.proj.weight.grad
        if actual is None:
            actual = torch.zeros_like(expected)
        _assert_close(actual, expected)
        assert expert.proj.weight._merak_expert_id == expert_id
        assert expert.proj.weight._merak_expert_parallel is True

    assert parallel.last_routing.sent_tokens == values.shape[1] * parallel.top_k
    if skewed and rank == 1:
        assert parallel.last_routing.received_tokens == 0


def _run_deepseek_ep_case(rank, groups, *, skewed):
    torch.manual_seed(191)
    reference = DeepseekV2MoE(skewed=skewed)
    parallel = shard_deepseek_v2_moe_model(copy.deepcopy(reference), groups)
    assert isinstance(parallel, DeepseekV2ExpertParallelBlock)
    values = (
        torch.tensor([[[1.0 + rank, 1.0], [2.0 + rank, 0.5]]])
        if skewed
        else torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]]) * (rank + 1)
    )
    reference_input = values.clone().requires_grad_(True)
    parallel_input = values.clone().requires_grad_(True)
    reference_output = reference(reference_input)
    parallel_output = parallel(parallel_input)
    _assert_close(parallel_output, reference_output)

    reference_output.square().sum().backward()
    parallel_output.square().sum().backward()
    _assert_close(parallel_input.grad, reference_input.grad)
    for name in ("gate.weight", "shared_experts.proj.weight"):
        _assert_close(
            dict(parallel.named_parameters())[name].grad,
            dict(reference.named_parameters())[name].grad,
        )

    expected_expert_grads = []
    for reference_expert in reference.experts:
        expected = reference_expert.proj.weight.grad
        expected = (
            torch.zeros_like(reference_expert.proj.weight)
            if expected is None
            else expected.clone()
        )
        dist.all_reduce(expected, group=groups.ep_group)
        expected_expert_grads.append(expected)
    parameter_names = dict(parallel.named_parameters())
    for expert_id, expert in parallel.local_experts.items():
        actual = expert.proj.weight.grad
        actual = torch.zeros_like(expected_expert_grads[expert_id]) if actual is None else actual
        _assert_close(actual, expected_expert_grads[expert_id])
        assert f"experts.{expert_id}.proj.weight" in parameter_names
        assert expert.proj.weight._merak_expert_id == expert_id
        assert expert.proj.weight._merak_expert_parallel is True
    assert not hasattr(parallel.gate.weight, "_merak_expert_parallel")
    assert not hasattr(parallel.shared_experts.proj.weight, "_merak_expert_parallel")
    if skewed and rank == 1:
        assert parallel.last_routing.received_tokens == 0


def _run_real_deepseek_ep_case(rank, groups):
    from examples.checkpoint_grouping.workloads import build_tiny_workload

    torch.manual_seed(811)
    workload = build_tiny_workload("deepseek_v2_lite")
    reference = workload.model_class(workload.config).model.layers[0].mlp
    parallel = shard_deepseek_v2_moe_model(copy.deepcopy(reference), groups)
    values = torch.randn(1, 3, workload.config.hidden_size) + rank * 0.2
    reference_input = values.clone().requires_grad_(True)
    parallel_input = values.clone().requires_grad_(True)

    reference_output = reference(reference_input)
    parallel_output = parallel(parallel_input)
    _assert_close(parallel_output, reference_output)
    reference_output.square().mean().backward()
    parallel_output.square().mean().backward()
    _assert_close(parallel_input.grad, reference_input.grad)

    reference_parameters = dict(reference.named_parameters())
    parallel_parameters = dict(parallel.named_parameters())
    for name in ("gate.weight", "shared_experts.gate_proj.weight"):
        _assert_close(parallel_parameters[name].grad, reference_parameters[name].grad)

    for expert_id, expert in parallel.local_experts.items():
        prefix = f"experts.{expert_id}."
        for suffix, parameter in expert.named_parameters():
            name = prefix + suffix
            expected = reference_parameters[name].grad
            if expected is None:
                expected = torch.zeros_like(parameter)
            else:
                expected = expected.clone()
            dist.all_reduce(expected, group=groups.ep_group)
            actual = parameter.grad
            if actual is None:
                actual = torch.zeros_like(parameter)
            _assert_close(actual, expected)


def _distributed_ep_worker(rank, init_method):
    dist.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=init_method
    )
    try:
        groups = create_expert_parallel_groups(ExpertParallelLayout(1, 2, 1, 1))
        _run_ep_case(rank, groups, skewed=False)
        _run_ep_case(rank, groups, skewed=True)
        _run_deepseek_ep_case(rank, groups, skewed=False)
        _run_deepseek_ep_case(rank, groups, skewed=True)
        _run_real_deepseek_ep_case(rank, groups)
        converted = Workload(
            "deepseek_v2_lite",
            "tiny",
            SimpleNamespace(n_routed_experts=4, n_shared_experts=1),
            DeepseekV2MoE,
        ).build_model(empty=False, expert_parallel_groups=groups)
        assert isinstance(converted, DeepseekV2ExpertParallelBlock)
    finally:
        dist.destroy_process_group()


def _distributed_nccl_ep_worker(rank, init_method):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=2, init_method=init_method
    )
    try:
        groups = create_expert_parallel_groups(ExpertParallelLayout(1, 2, 1, 1))
        torch.manual_seed(271)
        reference = _TinyQwenBlock(skewed=True).to(device=f"cuda:{rank}")
        parallel = Qwen2MoeExpertParallelBlock(copy.deepcopy(reference), groups).cuda()
        values = torch.tensor(
            [[[1.0 + rank, 1.0], [2.0 + rank, 0.5]]], device=f"cuda:{rank}"
        )
        reference_input = values.clone().requires_grad_(True)
        parallel_input = values.clone().requires_grad_(True)

        reference_output, _ = reference(reference_input)
        parallel_output, _ = parallel(parallel_input)
        torch.testing.assert_close(parallel_output, reference_output)
        reference_output.square().sum().backward()
        parallel_output.square().sum().backward()
        torch.testing.assert_close(parallel_input.grad, reference_input.grad)

        for name in ("gate.weight", "shared_expert.proj.weight", "shared_expert_gate.weight"):
            torch.testing.assert_close(
                dict(parallel.named_parameters())[name].grad,
                dict(reference.named_parameters())[name].grad,
                atol=1e-5,
                rtol=1e-5,
            )

        for expert_id, expert in enumerate(reference.experts):
            expected = expert.proj.weight.grad
            if expected is None:
                expected = torch.zeros_like(expert.proj.weight)
            else:
                expected = expected.clone()
            dist.all_reduce(expected, group=groups.ep_group)
            if expert_id in parallel.local_experts:
                actual = parallel.local_experts[expert_id].proj.weight.grad
                if actual is None:
                    actual = torch.zeros_like(expected)
                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=1e-5,
                    rtol=1e-5,
                )
        if rank == 1:
            assert parallel.last_routing.received_tokens == 0
    finally:
        dist.destroy_process_group()


def test_ep_layout_keeps_dp_and_ep_axes_distinct():
    layout = ExpertParallelLayout(pp=4, ep=2, dp=2, tp=1)
    assert layout.world_size == 16
    assert layout.coordinates(layout.rank(pp=3, ep=1, dp=0, tp=0)) == {
        "pp": 3,
        "ep": 1,
        "dp": 0,
        "tp": 0,
    }
    assert layout.all_groups("ep")[0] == (0, 2)
    assert layout.all_groups("dp")[0] == (0, 1)
    assert layout.group_ranks("ep", pp=0, dp=0, tp=0) == (0, 2)
    assert layout.expert_owner(0, 60) == 0
    assert layout.expert_owner(59, 60) == 1


def test_ep_partition_report_rejects_replicated_experts():
    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])

    groups = SimpleNamespace(layout=ExpertParallelLayout(1, 2, 1, 1), ep_rank=0)
    report = inspect_expert_partition(Experts(), num_experts=4, groups=groups)
    assert report["ready"] is False
    assert "replicates all experts" in report["blockers"][0]


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_ep2_matches_ep1_forward_backward_and_empty_routes():
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            _distributed_ep_worker,
            args=(f"file://{directory}/init",),
            nprocs=2,
            join=True,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not dist.is_nccl_available(),
    reason="CUDA/NCCL is unavailable",
)
def test_ep2_nccl_cuda_forward_backward_and_empty_routes():
    if torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are required")
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            _distributed_nccl_ep_worker,
            args=(f"file://{directory}/init",),
            nprocs=2,
            join=True,
        )


def test_zero1_manifest_uses_parameter_local_offsets():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.ones(3))
            self.b = torch.nn.Parameter(torch.ones(4))

    model = Model()
    optimizer = SimpleNamespace(
        fp16_groups=[[model.a, model.b]],
        params_in_rank_sub_partitions=[[[[model.a], [model.b]]]],
        params_in_rank_sub_partitions_offsets=[[[[0], [1]]]],
        local_sub_partitions_of_fp32_groups=[[
            torch.ones(3),
            torch.ones(3),
        ]],
        local_rank=0,
    )
    shards = iter_zero1_shards(model, optimizer)
    assert [(item.parameter_id, item.parameter_offset, item.length) for item in shards] == [
        ("a", 0, 3),
        ("b", 1, 3),
    ]
    assert zero1_semantic_compatibility(optimizer)["semantic_fixed_compatible"] is False
