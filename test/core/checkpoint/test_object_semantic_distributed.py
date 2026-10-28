import json

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from Merak.core.checkpoint.incremental_checkpoint import (
    RecoveryBinding,
    capture_optimizer_update,
)
from Merak.core.checkpoint.object_optimizer_update import (
    externalize_local_optimizer_update,
)
from Merak.core.checkpoint.object_semantic_distributed import (
    gather_active_gradient_plan,
    gather_confirmed_gradient_encoding,
)
from Merak.core.checkpoint.object_semantic_plan import GradientShard, RankBinding
from Merak.core.checkpoint.object_update_transport import (
    build_local_object_update_bundles,
)


def _contains_tensor(value):
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


def _worker(rank, world_size, rendezvous, result_dir, arm, mismatch):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size
    )
    original_gather = dist.all_gather_object
    wire_payloads = []

    def checked_gather(output, value, *args, **kwargs):
        assert not _contains_tensor(value)
        json.dumps(value, sort_keys=True)
        wire_payloads.append(value)
        return original_gather(output, value, *args, **kwargs)

    dist.all_gather_object = checked_gather
    try:
        binding = RankBinding(rank, 0, 0, rank, rank // 2)
        shards = tuple(
            GradientShard(
                rank, fqn, f"detector-{fqn}", (2,), (0,), (2,), "torch.float32"
            )
            for fqn in ("left", "right")
        )
        plan = gather_active_gradient_plan(
            binding,
            shards,
            arm=arm,
            detector_catalog_hash="a" * 64,
            semantic_plan_hash="b" * 64,
            fixed_pairs=((0, 2), (1, 3)),
            group=dist.group.WORLD,
        )
        objects = {item.fqn: item for item in plan.objects}
        gradients = {
            "left": torch.tensor([0.25, -0.5]),
            "right": torch.tensor([-0.75, 0.125]),
        }
        if mismatch and rank == 3:
            gradients["left"].add_(1.0)
        encoding = gather_confirmed_gradient_encoding(
            plan,
            {objects[fqn].object_id: gradient for fqn, gradient in gradients.items()},
            epoch=7,
            group=dist.group.WORLD,
        )

        parameters = [
            torch.nn.Parameter(torch.tensor([1.0, -2.0])),
            torch.nn.Parameter(torch.tensor([3.0, -4.0])),
        ]
        optimizer = torch.optim.AdamW(parameters, lr=0.003, foreach=False, fused=False)
        for parameter, fqn in zip(parameters, ("left", "right")):
            parameter.grad = gradients[fqn].clone()
        record = capture_optimizer_update(
            optimizer,
            parameter_ids=(("left", "right"),),
            binding=RecoveryBinding(rank, world_size, rank, 0, 0),
            base_id="sha256:" + "c" * 64,
            base_epoch=0,
            epoch=7,
        )
        update = externalize_local_optimizer_update(plan, encoding, rank, record)
        bundles = build_local_object_update_bundles(plan, encoding, update)
        planned_versions = {
            action.version_id for action in encoding.actions if action.source_rank == rank
        }
        assert {payload.version_id for payload in update.payloads} == planned_versions
        assert all(not _contains_tensor(bundle) for bundle in bundles) == (
            not planned_versions
        )
        for bundle in bundles:
            if not planned_versions:
                assert not bundle["occurrences"]
            else:
                assert {
                    occurrence["version_id"] for occurrence in bundle["occurrences"]
                } <= planned_versions
        assert len(wire_payloads) == 2
        assert set(wire_payloads[0]) == {"binding", "shards"}
        assert all(set(item) == {"rank", "object_id", "checksum"} for item in wire_payloads[1])
        (result_dir / f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "plan_hash": plan.plan_hash,
                    "canonical_hash": plan.canonical_hash,
                    "encoding_hash": encoding.encoding_hash,
                    "mismatched": list(encoding.mismatched_object_ids),
                },
                sort_keys=True,
            )
        )
    finally:
        dist.all_gather_object = original_gather
        dist.destroy_process_group()


@pytest.mark.parametrize("arm", ("adacheck_fixed", "host_global_fixed", "m1"))
@pytest.mark.parametrize("mismatch", (False, True))
def test_gloo_semantic_checksum_gate(tmp_path, arm, mismatch):
    rendezvous = tmp_path / "rendezvous"
    mp.spawn(_worker, args=(4, str(rendezvous), tmp_path, arm, mismatch), nprocs=4)
    reports = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(4)]
    for name in ("plan_hash", "canonical_hash", "encoding_hash"):
        assert len({report[name] for report in reports}) == 1
    mismatched = reports[0]["mismatched"]
    assert all(report["mismatched"] == mismatched for report in reports)
    assert bool(mismatched) is mismatch
