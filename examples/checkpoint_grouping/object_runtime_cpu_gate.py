#!/usr/bin/env python3
"""Run the non-GPU ObjectRecovery/SemanticFixed correctness gate."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

import torch

from Merak.core.checkpoint.incremental_checkpoint import (
    RecoveryBinding,
    apply_optimizer_update,
    capture_optimizer_update,
)
from Merak.core.checkpoint.object_optimizer_update import (
    externalize_local_optimizer_update,
)
from Merak.core.checkpoint.object_replica_service import ObjectReplicaServiceStore
from Merak.core.checkpoint.object_semantic_plan import (
    GradientShard,
    RankBinding,
    build_active_gradient_plan,
    confirm_gradient_equivalence,
)
from Merak.core.checkpoint.object_update_transport import (
    publish_object_update_epoch,
    restore_object_update_epoch,
)
from Merak.core.checkpoint.replica_service import ReplicaServiceClient, ReplicaTCPServer


BASE_ID = "sha256:" + hashlib.sha256(b"object-runtime-cpu-gate").hexdigest()
ARMS = ("adacheck_fixed", "host_global_fixed", "m1")


@contextmanager
def services():
    with ExitStack() as stack:
        result = []
        for _ in range(2):
            store = ObjectReplicaServiceStore(64 << 20, retained_epochs_per_source=1)
            server = ReplicaTCPServer(("127.0.0.1", 0), store)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            stack.callback(thread.join, timeout=5)
            stack.callback(server.server_close)
            stack.callback(server.shutdown)
            result.append((ReplicaServiceClient(*server.server_address), server))
        yield result


def optimizer(parameters):
    return torch.optim.AdamW(
        parameters,
        lr=0.003,
        betas=(0.8, 0.95),
        weight_decay=0.1,
        foreach=False,
        fused=False,
    )


def scenario(arm, epoch, *, mismatch=False):
    bindings = tuple(RankBinding(rank, 0, 0, rank, rank // 2) for rank in range(4))
    shards = tuple(
        GradientShard(
            rank, name, f"detector-{name}", (2,), (0,), (2,), "torch.float32"
        )
        for name in ("left", "right")
        for rank in (0, 1)
    )
    plan = build_active_gradient_plan(
        shards,
        bindings,
        arm=arm,
        detector_catalog_hash="a" * 64,
        semantic_plan_hash="b" * 64,
        fixed_pairs=((0, 2), (1, 3)),
    )
    gradients = {
        (0, "left"): torch.tensor([0.25 * epoch, -0.5]),
        (1, "left"): torch.tensor([0.25 * epoch, -0.5]),
        (0, "right"): torch.tensor([-0.75, 0.125 * epoch]),
        (1, "right"): torch.tensor([-0.75, 0.125 * epoch]),
    }
    if mismatch:
        gradients[(1, "left")].add_(1.0)
        gradients[(1, "right")].sub_(1.0)
    objects = {item.fqn: item for item in plan.objects}
    encoding = confirm_gradient_equivalence(
        plan,
        {
            (rank, objects[name].object_id): value
            for (rank, name), value in gradients.items()
        },
        epoch=epoch,
    )
    records = {}
    updates = {}
    for rank in range(4):
        parameters = [
            torch.nn.Parameter(torch.tensor([1.0, -2.0])),
            torch.nn.Parameter(torch.tensor([3.0, -4.0])),
        ]
        opt = optimizer(parameters)
        for name, parameter in zip(("left", "right"), parameters):
            value = gradients.get((rank, name))
            parameter.grad = None if value is None else value.clone()
        record = capture_optimizer_update(
            opt,
            parameter_ids=(("left", "right"),),
            binding=RecoveryBinding(rank, 4, rank, 0, 0),
            base_id=BASE_ID,
            base_epoch=0,
            epoch=epoch,
        )
        records[rank] = record
        updates[rank] = externalize_local_optimizer_update(plan, encoding, rank, record)
    return plan, encoding, records, updates


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def replay_matches(direct, restored):
    for rank in range(4):
        direct_parameters = [
            torch.nn.Parameter(torch.tensor([1.0, -2.0])),
            torch.nn.Parameter(torch.tensor([3.0, -4.0])),
        ]
        restored_parameters = [
            torch.nn.Parameter(torch.tensor([1.0, -2.0])),
            torch.nn.Parameter(torch.tensor([3.0, -4.0])),
        ]
        direct_optimizer = optimizer(direct_parameters)
        restored_optimizer = optimizer(restored_parameters)
        for epoch in (1, 2):
            arguments = dict(
                expected_parameter_ids=(("left", "right"),),
                expected_binding=RecoveryBinding(rank, 4, rank, 0, 0),
                expected_base_id=BASE_ID,
                expected_base_epoch=0,
                expected_epoch=epoch,
            )
            apply_optimizer_update(direct_optimizer, direct[epoch][rank], **arguments)
            apply_optimizer_update(
                restored_optimizer, restored[epoch][rank], **arguments
            )
        if not all(
            torch.equal(left, right)
            for left, right in zip(direct_parameters, restored_parameters)
        ) or not equal(direct_optimizer.state_dict(), restored_optimizer.state_dict()):
            return False
    return True


def arm_metrics(arm):
    plan, encoding, _, updates = scenario(arm, 1)
    with services() as active:
        clients = {host: value[0] for host, value in enumerate(active)}
        metrics = publish_object_update_epoch(
            clients,
            run_id=f"cpu-gate-{arm}",
            plan=plan,
            encoding=encoding,
            local_updates=updates,
        ).metrics
    return {
        "arm": arm,
        "action_count": len(encoding.actions),
        "mismatch_count": len(encoding.mismatched_object_ids),
        "remote_gradient_logical_bytes": metrics.remote_gradient_logical_bytes,
        "remote_data_payload_bytes": metrics.remote_data_payload_bytes,
        "remote_control_payload_bytes": metrics.remote_control_payload_bytes,
        "total_remote_payload_bytes": metrics.total_remote_payload_bytes,
        "remote_wire_bytes": metrics.remote_wire_bytes,
        "maximum_remote_put_ms": metrics.maximum_remote_put_ms,
        "maximum_remote_publisher_bytes": plan.maximum_remote_publisher_bytes,
    }


def recovery_gate(arm, failed_host):
    direct = {}
    locators = {}
    with services() as active:
        clients = {host: value[0] for host, value in enumerate(active)}
        for epoch in (1, 2):
            plan, encoding, records, updates = scenario(arm, epoch)
            direct[epoch] = records
            published = publish_object_update_epoch(
                clients,
                run_id=f"cpu-gate-{arm}-fail-{failed_host}",
                plan=plan,
                encoding=encoding,
                local_updates=updates,
            )
            locators[epoch] = (plan, encoding, published.locator)
        active[failed_host][1].shutdown()
        survivor = 1 - failed_host
        restored = {
            epoch: dict(
                restore_object_update_epoch(
                    clients[survivor],
                    surviving_host=survivor,
                    locator=locator,
                    plan=plan,
                    encoding=encoding,
                ).records
            )
            for epoch, (plan, encoding, locator) in locators.items()
        }
    records_equal = all(
        equal(direct[epoch][rank], restored[epoch][rank])
        for epoch in (1, 2)
        for rank in range(4)
    )
    return {
        "arm": arm,
        "failed_host": failed_host,
        "surviving_host": 1 - failed_host,
        "sealed_epochs_restored": [1, 2],
        "records_equal": records_equal,
        "adamw_replay_equal": records_equal and replay_matches(direct, restored),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    summary = {
        "schema_version": 1,
        "run_kind": "object_runtime_cpu_tcp_correctness_gate",
        "evidence_level": "cpu_tcp_validation_not_end_to_end_gpu",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "scenario": {
            "world_size": 4,
            "hosts": 2,
            "objects": 2,
            "object_holders": [0, 1],
            "fixed_pairs": [[0, 2], [1, 3]],
            "epochs": [1, 2],
        },
        "arms": [arm_metrics(arm) for arm in ARMS],
        "recovery": [
            recovery_gate(arm, failed_host)
            for arm in ARMS
            for failed_host in (0, 1)
        ],
        "runtime_mismatch_fallback": {
            "covered_by": "test_object_update_transport.py",
            "policy": "rank-specific direct_fallback replicated to both hosts",
        },
        "elapsed_seconds": time.time() - started,
    }
    if not all(
        item["records_equal"] and item["adamw_replay_equal"]
        for item in summary["recovery"]
    ):
        raise RuntimeError("ObjectRecovery CPU/TCP gate failed")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (args.output_dir / "COMPLETE").write_text("cpu/tcp correctness gate passed\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
