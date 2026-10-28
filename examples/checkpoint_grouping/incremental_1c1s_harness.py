#!/usr/bin/env python3
"""Two-host 1C1S training and fresh-job recovery validation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from Merak.core.checkpoint.grouping import CheckpointGroupingConfig, plan_checkpoint_groups
from Merak.core.checkpoint.incremental_checkpoint import (
    RecoveryBinding,
    apply_optimizer_update,
)
from Merak.core.checkpoint.incremental_runtime import (
    IncrementalCheckpointSession,
    PeerIncrementalCheckpointSession,
    restore_incremental_chain,
    restore_peer_incremental_chain,
)
from Merak.core.checkpoint.recovery import ServiceRemoteCheckpointSession
from Merak.core.checkpoint.replica_service import ReplicaServiceClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("train", "restore"), required=True)
    parser.add_argument(
        "--mode", choices=("off", "full_remote", "peer_delta"), required=True
    )
    parser.add_argument("--service-endpoints", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--continue-steps", type=int, default=5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--compute-repeats", type=int, default=4)
    parser.add_argument("--max-inflight", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--recovery-rtol", type=float, default=1e-6)
    parser.add_argument("--recovery-atol", type=float, default=1e-7)
    return parser.parse_args()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def build_objects(args: argparse.Namespace, local_rank: int):
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = nn.Sequential(
        nn.Linear(args.width, args.width),
        nn.GELU(),
        nn.Linear(args.width, args.width),
    ).to(device)
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        static_graph=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, foreach=False, fused=False
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.9)
    return device, model, optimizer, scheduler


def parameter_ids(model: DistributedDataParallel, optimizer) -> tuple[tuple[str, ...], ...]:
    names = {id(parameter): name for name, parameter in model.module.named_parameters()}
    return tuple(
        tuple(names[id(parameter)] for parameter in group["params"])
        for group in optimizer.param_groups
    )


def batch_for_step(args, step: int, rank: int, device: torch.device):
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + step * 1009 + rank * 9176)
    inputs = torch.randn(
        args.batch_size, args.width, generator=generator, device=device
    )
    targets = torch.randn(
        args.batch_size, args.width, generator=generator, device=device
    )
    return inputs, targets


def runtime_state(step: int, device: torch.device) -> dict[str, Any]:
    return {
        "iteration": step,
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device),
    }


def restore_runtime_state(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["random_rng_state"])
    np.random.set_state(state["np_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state(state["cuda_rng_state"], device)


def model_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def model_difference(model: nn.Module, expected: dict[str, torch.Tensor]) -> dict:
    mismatched = []
    max_absolute_error = 0.0
    for name, value in model.state_dict().items():
        actual = value.detach().cpu()
        reference = expected[name]
        if tensor_bitwise_equal(actual, reference):
            continue
        mismatched.append(name)
        if actual.is_floating_point() or actual.is_complex():
            difference = (actual - reference).abs()
            error = (
                float(difference.max())
                if bool(torch.isfinite(difference).all())
                else float("inf")
            )
        else:
            error = float((actual != reference).max())
        max_absolute_error = max(max_absolute_error, error)
    return {
        "mismatched_tensor_count": len(mismatched),
        "first_mismatched_tensor": mismatched[0] if mismatched else None,
        "max_absolute_error": max_absolute_error,
    }


def states_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and tensor_bitwise_equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(states_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(states_equal(a, b) for a, b in zip(left, right))
        )
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    return left == right


def tensor_bitwise_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    left_bytes = left.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    right_bytes = right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(left_bytes, right_bytes)


def model_allclose(
    model: nn.Module,
    expected: dict[str, torch.Tensor],
    *,
    rtol: float,
    atol: float,
) -> bool:
    for name, value in model.state_dict().items():
        actual = value.detach().cpu()
        reference = expected[name]
        if actual.is_floating_point() or actual.is_complex():
            if not torch.allclose(
                actual, reference, rtol=rtol, atol=atol, equal_nan=False
            ):
                return False
        elif not torch.equal(actual, reference):
            return False
    return True


def model_is_finite(model: nn.Module) -> bool:
    return all(
        not tensor.is_floating_point() or bool(torch.isfinite(tensor).all())
        for tensor in model.state_dict().values()
    )


def train_step(args, step, rank, device, model, optimizer) -> float:
    inputs, targets = batch_for_step(args, step, rank, device)
    optimizer.zero_grad(set_to_none=True)
    outputs = torch.zeros_like(targets)
    for _ in range(args.compute_repeats):
        outputs = outputs + model(inputs)
    outputs = outputs / args.compute_repeats
    loss = torch.nn.functional.mse_loss(outputs, targets)
    loss.backward()
    return float(loss.detach())


def checkpoint_base(model, optimizer, scheduler, device, iteration) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "runtime_state": runtime_state(iteration, device),
    }


def peer_context(rank: int, world_size: int, ranks_per_node: int):
    plan = plan_checkpoint_groups(
        world_size,
        CheckpointGroupingConfig(
            ranks_per_node=ranks_per_node,
            group_size=2,
            scope="host",
            policy="sequential",
        ),
    )
    group = plan.group_for_rank(rank)
    peer = group[1] if group[0] == rank else group[0]
    return plan, group, peer


def service_session(args, rank, group, target_rank, endpoint):
    return ServiceRemoteCheckpointSession(
        rank=rank,
        group_ranks=group,
        target_rank=target_rank,
        service_endpoint=endpoint,
        manifest_root=args.output_dir / "manifests",
        max_local_store_bytes=1,
        record_manifests=False,
        collect_store_stats=False,
    )


def service_sessions(args, rank, group, peer, ranks_per_node):
    endpoints = json.loads(args.service_endpoints)
    local_endpoint = endpoints[rank // ranks_per_node]
    remote_endpoint = endpoints[peer // ranks_per_node]
    if args.mode == "full_remote":
        return service_session(args, rank, group, peer, remote_endpoint)
    if args.mode == "peer_delta":
        return (
            service_session(args, rank, group, rank, local_endpoint),
            service_session(args, rank, group, peer, remote_endpoint),
        )
    return None


def save_oracle(path: Path, model, optimizer, scheduler, losses, runtime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {key: value.detach().cpu() for key, value in model.module.state_dict().items()},
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()),
            "losses": losses,
            "runtime_state": runtime,
            "model_digest": model_digest(model.module),
        },
        path,
    )


def train_phase(args, rank, peer, binding, device, model, optimizer, scheduler, sessions):
    for step in range(1, args.warmup_steps + 1):
        train_step(args, step, rank, device, model, optimizer)
        optimizer.step()
        scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    incremental = None
    base_full = None
    base_delta = None
    ids = parameter_ids(model, optimizer)
    if args.mode == "full_remote":
        incremental = IncrementalCheckpointSession(
            sessions,
            optimizer,
            parameter_ids=ids,
            binding=binding,
            max_inflight=args.max_inflight,
        )
        base_full = incremental.save_base(
            0,
            checkpoint_base(
                model, optimizer, scheduler, device, args.warmup_steps
            ),
        )
    elif args.mode == "peer_delta":
        local_full, remote_delta = sessions
        incremental = PeerIncrementalCheckpointSession(
            local_full,
            remote_delta,
            optimizer,
            parameter_ids=ids,
            binding=binding,
            peer_rank=peer,
            max_inflight=args.max_inflight,
        )
        state = runtime_state(args.warmup_steps, device)
        base_full, base_delta = incremental.save_base(
            0,
            checkpoint_base(
                model, optimizer, scheduler, device, args.warmup_steps
            ),
            runtime_state=state,
        )
    dist.barrier()
    torch.cuda.synchronize(device)

    capture_ms = []
    step_ms = []
    losses = []
    completed = []
    latest_commit = 0
    boundary_lag = []
    measured_start = time.perf_counter()
    for step in range(1, args.steps + 1):
        started = time.perf_counter()
        losses.append(
            train_step(
                args, args.warmup_steps + step, rank, device, model, optimizer
            )
        )
        if incremental is not None:
            launched = incremental.enqueue_update(
                step,
                metadata={"step": step},
                runtime_state=runtime_state(args.warmup_steps + step, device),
            )
            capture_ms.append(
                sum(launched) if isinstance(launched, tuple) else launched
            )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if incremental is not None:
            newly_completed = incremental.poll()
            completed.extend(newly_completed)
            if newly_completed:
                latest_commit = max(latest_commit, *(item.epoch for item in newly_completed))
            boundary_lag.append(step - latest_commit)
        step_ms.append((time.perf_counter() - started) * 1000)
    torch.cuda.synchronize(device)
    loop_ms = (time.perf_counter() - measured_start) * 1000
    drain_started = time.perf_counter()
    if incremental is not None:
        completed.extend(incremental.close())
    drain_ms = (time.perf_counter() - drain_started) * 1000

    save_oracle(
        args.output_dir / "oracle" / f"rank-{rank:04d}-step-{args.steps}.pt",
        model,
        optimizer,
        scheduler,
        [],
        runtime_state(args.warmup_steps + args.steps, device),
    )
    continuation_losses = []
    for step in range(args.steps + 1, args.steps + args.continue_steps + 1):
        continuation_losses.append(
            train_step(
                args, args.warmup_steps + step, rank, device, model, optimizer
            )
        )
        optimizer.step()
        scheduler.step()
    torch.cuda.synchronize(device)
    save_oracle(
        args.output_dir / "oracle" / f"rank-{rank:04d}-continued.pt",
        model,
        optimizer,
        scheduler,
        continuation_losses,
        runtime_state(args.warmup_steps + args.steps + args.continue_steps, device),
    )
    completed.sort(key=lambda item: item.epoch)
    return {
        "rank": rank,
        "host": socket.gethostname(),
        "mode": args.mode,
        "loop_ms": loop_ms,
        "final_drain_ms": drain_ms,
        "step_ms": summary(step_ms),
        "loss": summary(losses),
        "training_finite": all(math.isfinite(value) for value in losses)
        and model_is_finite(model.module),
        "capture_ms": summary(capture_ms),
        "per_rank_remote_commit_lag_steps": summary(
            [float(value) for value in boundary_lag]
        ),
        "base_full_payload_bytes": (
            0 if base_full is None else base_full.manifest.payload_bytes
        ),
        "base_remote_delta_payload_bytes": (
            0 if base_delta is None else base_delta.manifest.payload_bytes
        ),
        "update_epochs": [item.epoch for item in completed],
        "local_full_update_payload_bytes": sum(
            item.local_full.result.manifest.payload_bytes for item in completed
        )
        if args.mode == "peer_delta"
        else 0,
        "remote_update_payload_bytes": sum(
            (
                item.remote_delta.result.manifest.payload_bytes
                if args.mode == "peer_delta"
                else item.result.manifest.payload_bytes
            )
            for item in completed
        ),
        "remote_update_wire_bytes": sum(
            (
                item.remote_delta.result.transport.wire_bytes
                if args.mode == "peer_delta"
                else item.result.transport.wire_bytes
            )
            for item in completed
        ),
        "update_commit_ms": summary(
            [
                max(item.local_full.result.commit_ms, item.remote_delta.result.commit_ms)
                if args.mode == "peer_delta"
                else item.result.commit_ms
                for item in completed
            ]
        ),
        "update_serialize_ms": summary(
            [
                max(
                    item.local_full.result.serialize_ms,
                    item.remote_delta.result.serialize_ms,
                )
                if args.mode == "peer_delta"
                else item.result.serialize_ms
                for item in completed
            ]
        ),
        "update_transport_ms": summary(
            [
                max(
                    item.local_full.result.transport.duration_ms,
                    item.remote_delta.result.transport.duration_ms,
                )
                if args.mode == "peer_delta"
                else item.result.transport.duration_ms
                for item in completed
            ]
        ),
        "update_queue_ms": summary(
            [
                max(item.local_full.queue_ms, item.remote_delta.queue_ms)
                if args.mode == "peer_delta"
                else item.queue_ms
                for item in completed
            ]
        ),
    }


def restore_phase(
    args,
    rank,
    peer,
    binding,
    peer_binding,
    device,
    model,
    optimizer,
    scheduler,
    sessions,
    ranks_per_node,
):
    ids = parameter_ids(model, optimizer)
    started = time.perf_counter()
    delta_restore_wire_bytes = 0
    if args.mode == "full_remote":
        chain = restore_incremental_chain(
            sessions, base_epoch=0, final_epoch=args.steps
        )
        expected_binding = binding
        final_runtime_state = chain.updates[-1]["runtime_state"]
    else:
        endpoints = json.loads(args.service_endpoints)
        recovered = restore_peer_incremental_chain(
            ReplicaServiceClient.from_endpoint(endpoints[peer // ranks_per_node]),
            failed_rank=rank,
            peer_rank=peer,
            base_epoch=0,
            final_epoch=args.steps,
        )
        chain = recovered.full
        expected_binding = peer_binding
        final_runtime_state = recovered.runtime_states[-1]
        delta_restore_wire_bytes = recovered.delta_restore_wire_bytes
    model.module.load_state_dict(chain.base_state["model"], strict=True)
    optimizer.load_state_dict(chain.base_state["optimizer"])
    scheduler.load_state_dict(chain.base_state["scheduler"])
    for epoch, update in enumerate(chain.updates, start=1):
        apply_optimizer_update(
            optimizer,
            update,
            expected_parameter_ids=ids,
            expected_binding=expected_binding,
            expected_base_id=chain.base_id,
            expected_base_epoch=0,
            expected_epoch=epoch,
        )
        scheduler.step()
    restore_runtime_state(dict(final_runtime_state), device)
    torch.cuda.synchronize(device)
    restore_ms = (time.perf_counter() - started) * 1000

    step_oracle = torch.load(
        args.output_dir / "oracle" / f"rank-{rank:04d}-step-{args.steps}.pt",
        map_location="cpu",
        weights_only=False,
    )
    step_exact = all(
        tensor_bitwise_equal(value, step_oracle["model"][name])
        for name, value in model.module.state_dict().items()
    )
    optimizer_exact = states_equal(optimizer.state_dict(), step_oracle["optimizer"])
    scheduler_exact = states_equal(scheduler.state_dict(), step_oracle["scheduler"])
    runtime_exact = states_equal(
        runtime_state(args.warmup_steps + args.steps, device),
        step_oracle["runtime_state"],
    )
    step_model_difference = model_difference(model.module, step_oracle["model"])
    recovered_step_digest = model_digest(model.module)
    continuation_losses = []
    for step in range(args.steps + 1, args.steps + args.continue_steps + 1):
        continuation_losses.append(
            train_step(
                args, args.warmup_steps + step, rank, device, model, optimizer
            )
        )
        optimizer.step()
        scheduler.step()
    torch.cuda.synchronize(device)
    final_oracle = torch.load(
        args.output_dir / "oracle" / f"rank-{rank:04d}-continued.pt",
        map_location="cpu",
        weights_only=False,
    )
    loss_exact = continuation_losses == final_oracle["losses"]
    loss_close = len(continuation_losses) == len(final_oracle["losses"]) and all(
        math.isfinite(actual)
        and math.isfinite(expected)
        and math.isclose(
            actual,
            expected,
            rel_tol=args.recovery_rtol,
            abs_tol=args.recovery_atol,
        )
        for actual, expected in zip(continuation_losses, final_oracle["losses"])
    )
    final_exact = all(
        tensor_bitwise_equal(value, final_oracle["model"][name])
        for name, value in model.module.state_dict().items()
    )
    final_close = model_allclose(
        model.module,
        final_oracle["model"],
        rtol=args.recovery_rtol,
        atol=args.recovery_atol,
    )
    final_model_difference = model_difference(model.module, final_oracle["model"])
    return {
        "rank": rank,
        "host": socket.gethostname(),
        "restore_ms": restore_ms,
        "restore_wire_bytes": chain.restore_wire_bytes,
        "delta_restore_wire_bytes": delta_restore_wire_bytes,
        "replayed_updates": len(chain.updates),
        "step_parameter_bitwise_equal": step_exact,
        "step_optimizer_bitwise_equal": optimizer_exact,
        "step_scheduler_equal": scheduler_exact,
        "step_runtime_state_equal": runtime_exact,
        "step_model_difference": step_model_difference,
        "recovered_step_model_digest": recovered_step_digest,
        "oracle_step_model_digest": step_oracle["model_digest"],
        "continued_loss_bitwise_equal": loss_exact,
        "continued_parameter_bitwise_equal": final_exact,
        "continued_loss_numerically_equal": loss_close,
        "continued_parameter_numerically_equal": final_close,
        "continued_model_difference": final_model_difference,
        "recovery_valid": (
            step_exact
            and optimizer_exact
            and scheduler_exact
            and runtime_exact
            and loss_close
            and final_close
            and model_is_finite(model.module)
        ),
    }


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.continue_steps < 1:
        raise ValueError("steps and continue-steps must both be positive")
    if any(
        not math.isfinite(value) or value < 0 or value > 1e-3
        for value in (args.recovery_rtol, args.recovery_atol)
    ):
        raise ValueError("recovery tolerances must be finite values in [0, 1e-3]")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    ranks_per_node = int(
        os.environ.get("CKPT_RANKS_PER_NODE", os.environ["LOCAL_WORLD_SIZE"])
    )
    torch.cuda.set_device(local_rank)
    if world_size != 2 * ranks_per_node:
        raise ValueError("the harness requires exactly two equal-size hosts")
    if args.phase == "restore" and args.mode == "off":
        raise ValueError("restore phase requires a checkpoint mode")
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    plan, group, peer = peer_context(rank, world_size, ranks_per_node)
    device, model, optimizer, scheduler = build_objects(args, local_rank)
    random.seed(args.seed + rank)
    np.random.seed((args.seed + rank) % (2**32))
    torch.set_rng_state(torch.Generator().manual_seed(args.seed + rank).get_state())
    torch.cuda.manual_seed(args.seed + rank)
    binding = RecoveryBinding(rank, world_size, rank, 0, 0)
    peer_binding = RecoveryBinding(peer, world_size, peer, 0, 0)
    sessions = service_sessions(args, rank, group, peer, ranks_per_node)
    if args.phase == "train":
        local = train_phase(
            args,
            rank,
            peer,
            binding,
            device,
            model,
            optimizer,
            scheduler,
            sessions,
        )
    else:
        local = restore_phase(
            args,
            rank,
            peer,
            binding,
            peer_binding,
            device,
            model,
            optimizer,
            scheduler,
            sessions,
            ranks_per_node,
        )

    gathered = [None] * world_size
    dist.all_gather_object(gathered, local)
    atomic_json(args.output_dir / f"{args.phase}-rank-{rank:04d}.json", local)
    dist.barrier()
    if rank == 0:
        contiguous = (
            all(
                item["update_epochs"] == list(range(1, args.steps + 1))
                for item in gathered
            )
            if args.mode != "off" and args.phase == "train"
            else True
        )
        global_max_commit_lag = (
            max(
                float(item["per_rank_remote_commit_lag_steps"]["max"])
                for item in gathered
            )
            if args.mode != "off" and args.phase == "train"
            else None
        )
        valid = (
            all(item["recovery_valid"] for item in gathered)
            if args.phase == "restore"
            else contiguous
            and (global_max_commit_lag is None or global_max_commit_lag <= 1)
            and all(item["training_finite"] for item in gathered)
        )
        result = {
            "schema_version": 1,
            "run_kind": "two_host_incremental_1c1s",
            "phase": args.phase,
            "mode": args.mode,
            "world_size": world_size,
            "ranks_per_node": ranks_per_node,
            "groups": [list(value) for value in plan.groups],
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "compute_repeats": args.compute_repeats,
            "continue_steps": args.continue_steps,
            "seed": args.seed,
            "all_update_epochs_contiguous": contiguous,
            "global_max_remote_commit_lag_steps": global_max_commit_lag,
            "valid": valid,
            "ranks": gathered,
        }
        atomic_json(args.output_dir / f"{args.phase}-summary.json", result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
