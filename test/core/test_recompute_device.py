import pytest
import torch
import torch.nn.functional as F
import torch.distributed as dist
from datetime import timedelta

from Merak.core.recompute.checkpointing import checkpoint, pre_checkpoint


def _recompute_worker(rank, init_method, checkpoint_name):
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=1,
        timeout=timedelta(seconds=30),
    )
    try:
        from Merak.core.mpu import initialize

        initialize.set_model_parallel_group(dist.group.WORLD)
        initialize.set_data_parallel_group(dist.group.WORLD)
        initialize.set_pipe_parallel_group(dist.group.WORLD)
        initialize.set_sequence_parallel_group(dist.group.WORLD)
        checkpoint_fn = checkpoint if checkpoint_name == "checkpoint" else pre_checkpoint
        _assert_cpu_recompute(checkpoint_fn)
    finally:
        dist.destroy_process_group()


def _assert_cpu_recompute(checkpoint_fn):
    torch.manual_seed(41)
    cuda_rng_before = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    x = torch.ones(16, requires_grad=True)
    weight = torch.nn.Parameter(torch.ones(16))

    def run(value):
        assert value.device.type == "cpu"
        return F.dropout(value, p=0.5, training=True) * weight

    output = checkpoint_fn(run, x)
    assert output.device.type == "cpu"
    output.sum().backward()
    assert x.grad is not None and weight.grad is not None
    if cuda_rng_before is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng_before)


@pytest.mark.parametrize("checkpoint_fn", [checkpoint, pre_checkpoint])
def test_cpu_recompute_stays_on_cpu_and_preserves_rng(tmp_path, checkpoint_fn):
    torch.multiprocessing.spawn(
        _recompute_worker,
        args=(f"file://{tmp_path / 'recompute-rendezvous'}", checkpoint_fn.__name__),
        nprocs=1,
        join=True,
    )
