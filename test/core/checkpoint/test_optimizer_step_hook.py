from types import SimpleNamespace

import pytest
import torch

from Merak.core.amp.amp_optimizer import HalfPrecisionOptimizer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_half_precision_hook_observes_applied_gradient_before_step():
    parameter = torch.nn.Parameter(
        torch.tensor([1.0, -2.0], device="cuda", dtype=torch.bfloat16)
    )
    base = torch.optim.AdamW([parameter], lr=1e-2, foreach=False)
    optimizer = HalfPrecisionOptimizer(
        base,
        SimpleNamespace(ada_checkpoint=False),
        static_loss_scale=1.0,
        dynamic_loss_scale=False,
        clip_grad=0.0,
    )
    observed = {}

    def capture(epoch, applied_optimizer, gradient_multiplier):
        master = applied_optimizer.param_groups[0]["params"][0]
        observed["epoch"] = epoch
        observed["gradient"] = master.grad.detach().cpu().clone()
        observed["parameter_before"] = master.detach().cpu().clone()
        observed["gradient_multiplier"] = gradient_multiplier

    optimizer.set_pre_optimizer_step_callback(capture)
    parameter.grad = torch.tensor(
        [0.25, -0.5], device="cuda", dtype=torch.bfloat16
    )
    optimizer.step()
    master_after = base.param_groups[0]["params"][0].detach().cpu()

    assert observed["epoch"] == 1
    assert torch.equal(observed["gradient"], torch.tensor([0.25, -0.5]))
    assert observed["gradient_multiplier"] == 1.0
    assert not torch.equal(observed["parameter_before"], master_after)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_half_precision_hook_sees_clipped_gradient():
    parameter = torch.nn.Parameter(
        torch.tensor([1.0, -2.0], device="cuda", dtype=torch.bfloat16)
    )
    optimizer = HalfPrecisionOptimizer(
        torch.optim.AdamW([parameter], lr=1e-2, foreach=False),
        SimpleNamespace(ada_checkpoint=False),
        static_loss_scale=1.0,
        dynamic_loss_scale=False,
        clip_grad=1.0,
    )
    observed = []
    optimizer.set_pre_optimizer_step_callback(
        lambda _epoch, applied, _multiplier: observed.append(
            applied.param_groups[0]["params"][0].grad.detach().cpu().clone()
        ),
    )
    parameter.grad = torch.tensor(
        [3.0, 4.0], device="cuda", dtype=torch.bfloat16
    )
    optimizer.step()

    assert torch.allclose(observed[0], torch.tensor([0.6, 0.8]), atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_half_precision_raw_hook_observes_bfloat16_gradient_and_exact_scale():
    parameter = torch.nn.Parameter(
        torch.tensor([1.0, -2.0], device="cuda", dtype=torch.bfloat16)
    )
    optimizer = HalfPrecisionOptimizer(
        torch.optim.AdamW([parameter], lr=1e-2, foreach=False),
        SimpleNamespace(ada_checkpoint=False),
        static_loss_scale=8.0,
        dynamic_loss_scale=False,
        clip_grad=0.0,
    )
    observed = {}

    def capture(_epoch, applied, multiplier):
        observed["gradient"] = (
            applied.param_groups[0]["params"][0].grad.detach().cpu().clone()
        )
        observed["multiplier"] = multiplier

    optimizer.set_pre_optimizer_step_callback(capture, capture_raw_gradients=True)
    parameter.grad = torch.tensor([2.0, -4.0], device="cuda", dtype=torch.bfloat16)
    optimizer.step()

    assert torch.equal(observed["gradient"], torch.tensor([2.0, -4.0]))
    assert observed["multiplier"] == 0.125
