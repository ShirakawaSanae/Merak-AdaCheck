import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
ANALYZER = ROOT / "tools" / "analyze_object_semantic_gate.py"


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _locator(path, kind, epoch, run_id, **extra):
    value = {
        "schema_version": 1,
        "kind": kind,
        "run_id": run_id,
        "epoch": epoch,
        "world_size": 16,
        **extra,
    }
    value["locator_hash"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write(path, value)


def _digests(root, iteration):
    for rank in range(16):
        _write(
            root / f"rank_{rank:05d}.json",
            {"rank": rank, "iteration": iteration, "state_checksum": f"digest-{rank}"},
        )


@pytest.mark.parametrize(
    ("parallelism", "source_steps", "continue_steps", "failed_service_node"),
    [
        ({"pp": 4, "tp": 1, "dp": 4, "mesh_order": "pp-dp"}, 10, 5, 8),
        ({"pp": 2, "tp": 1, "dp": 8, "mesh_order": "pp-dp"}, 3, 5, 17),
    ],
)
def test_object_semantic_gate_analyzer_accepts_complete_fixture(
    tmp_path, parallelism, source_steps, continue_steps, failed_service_node
):
    root = tmp_path / "gate"
    arm, run_id = "m1", "gate"
    reference_steps = source_steps + continue_steps
    surviving_host = 0 if failed_service_node == 8 else 1
    _write(
        root / "run-contract.json",
        {
            "arm": arm,
            "run_id": run_id,
            "restore_enabled": True,
            "failed_service_node": failed_service_node,
            "surviving_host": surviving_host,
            "scheduler": "constant",
            "optimizer": {
                "name": "scalar_adamw",
                "learning_rate": 5e-5,
                "weight_decay": 0.0,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
            },
            "nodes": [17, 8],
            "world_size": 16,
            "ranks_per_host": 8,
            "model": "gpt2",
            "parallelism": parallelism,
            "workload": {
                "sequence_length": 128,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "seed": 20260913,
                "precision": "bf16",
            },
            "checkpoint_enabled": True,
            "source_steps": source_steps,
            "restore_epoch": source_steps,
            "restore_continue_steps": continue_steps,
            "reference_steps": reference_steps,
        },
    )
    _write(
        root / "launcher-status.json",
        {"source": {"local": 0, "remote": 0}, "restore": {"local": 0, "remote": 0}},
    )
    _write(
        root / "phase-times.json",
        {
            phase: {
                "started_monotonic_ns": 0,
                "finished_monotonic_ns": 1,
                "duration_ns": 1,
                "scope": phase,
            }
            for phase in ("source", "restore")
        },
    )
    for phase, start, count in (("source", 1, source_steps), ("restore", source_steps + 1, continue_steps)):
        path = root / "logs" / phase / "node-0.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                f"step wall: {step} train-plus-checkpoint-poll time (s): 1.0\n"
                f"steps: {step} loss: {10 - step / 10:.4f} iter time (s): 1.0"
                for step in range(start, start + count)
            )
            + "\n"
        )
    for rank in range(16):
        _write(
            root / "source" / "metrics" / f"rank_{rank:05d}.json",
            {
                "rank": rank,
                "arm": arm,
                "base_epoch": 0,
                "committed_epochs": list(range(1, source_steps + 1)),
                "valid": True,
                "all_epochs_committed": True,
                "maximum_boundary_lag": 3,
                "strict_1c1s_valid": False,
                "final_drain_ms": rank + 0.5,
                "plan_load_metrics": {
                    "confirmation_mode": "post_allreduce_dp_invariant",
                    "local_staged_logical_bytes": rank + 100,
                    "local_verifier_logical_bytes": rank + 10,
                    "publisher_policy": "test_all_publication_least_load",
                    "all_publication_publisher_bytes": [
                        [publisher, publisher + 1] for publisher in range(16)
                    ],
                    "total_all_publication_logical_bytes": 136,
                    "maximum_all_publication_publisher_bytes": 16,
                    "total_remote_logical_bytes": 100,
                    "maximum_remote_publisher_bytes": 10,
                },
                "base_result": {"metrics": {}},
                "commit_metrics": [
                    {
                        "epoch": epoch,
                        "capture_ms": rank + epoch / 10,
                        "publisher_queue_ms": rank,
                        "snapshot_ready_wait_ms": 0,
                        "enqueue_to_commit_ms": rank + epoch,
                        "checksum_confirmation_ms": 1.0,
                        "externalize_ms": 2.0,
                        "distributed_commit_ms": 3.0,
                        "local_bundle_commit_ms": 1.5,
                        "manifest_gather_ms": 0.5,
                        "coordinator_evidence_ms": 0.1 if rank == 0 else None,
                        "coordinator_control_commit_ms": 0.2 if rank == 0 else None,
                        "coordinator_seal_locator_ms": 0.3 if rank == 0 else None,
                        "result_broadcast_ms": 0.4,
                        "remote_gradient_logical_bytes": 100,
                        "remote_data_payload_bytes": 110,
                        "remote_control_payload_bytes": 10,
                        "remote_wire_bytes": 140,
                        "maximum_remote_put_ms": 1.5,
                        "total_remote_payload_bytes": 120,
                    }
                    for epoch in range(1, source_steps + 1)
                ],
            },
        )
    _digests(root / "source" / "digests", source_steps)
    _digests(root / "restore" / "digests", reference_steps)
    for rank in range(16):
        _write(
            root / "source" / "recovery_metrics" / f"rank_{rank:05d}.json",
            {
                "rank": rank,
                "requested_epoch": source_steps,
                "final_epoch": source_steps,
                "base_epoch": 0,
                "arm": arm,
                "plan_hash": "plan",
                "mapping_hash": "mapping",
                "surviving_host": surviving_host,
                "total_restore_wall_ms": 1.0,
                "boundary_checksum": "boundary",
                "component_checksums": {},
                "base_payload_bytes": 1,
                "base_wire_bytes": 1,
                "base_get_ms": 1.0,
                "suffix_get_metrics": "unavailable",
                "checksum_diagnostic_only": True,
            },
        )
    _locator(
        root / "source" / "base" / "epoch-00000000.json",
        "object_semantic_base_locator",
        0,
        run_id,
    )
    for epoch in range(1, source_steps + 1):
        _locator(
            root / "source" / "updates" / f"epoch-{epoch:08d}.json",
            "object_optimizer_update_locator",
            epoch,
            run_id,
            arm=arm,
            plan_hash="plan",
            detector_catalog_hash="catalog",
        )
    for name in ("restore-service-before.json", "restore-service-after.json"):
        _write(root / name, {
            "node17": {"status": "unavailable" if failed_service_node == 17 else "ok"},
            "node8": {"status": "unavailable" if failed_service_node == 8 else "ok"},
        })

    subprocess.run([sys.executable, str(ANALYZER), str(root)], check=True)
    result = json.loads((root / "analysis.json").read_text())
    assert result["source"]["plan_load_metrics"]["publisher_policy"] == (
        "test_all_publication_least_load"
    )
    assert result["source"]["plan_load_metrics"][
        "maximum_all_publication_publisher_bytes"
    ] == 16
    assert result["source"]["plan_load_metrics"][
        "staged_logical_bytes_by_rank"
    ][-1] == [15, 115]
    assert result["source"]["rank_metrics"] == 16
    assert len(result["source"]["suffix_transfer"]) == source_steps
    assert len(result["source"]["per_rank_local_commit_timing"]) == 16
    assert result["source"]["per_rank_final_drain_ms"][-1] == 15.5
    assert result["source"]["per_rank_local_commit_timing"][0]["epochs"][0]["distributed_commit_ms"] == 3.0
    assert result["source"]["maximum_boundary_lag"] == 3
    assert result["source"]["strict_1c1s_valid"] is False
    assert len(result["loss_trace"]["restore"]) == continue_steps
    assert result["restore"]["rank_metrics"] == 16
    assert result["restore"]["continued_to_epoch"] == reference_steps

    source_metric_path = root / "source" / "metrics" / "rank_00000.json"
    source_metric = json.loads(source_metric_path.read_text())
    source_metric["plan_load_metrics"]["confirmation_mode"] = "unconfirmed"
    _write(source_metric_path, source_metric)
    assert subprocess.run([sys.executable, str(ANALYZER), str(root)]).returncode != 0
    source_metric["plan_load_metrics"]["confirmation_mode"] = (
        "post_allreduce_dp_invariant"
    )
    _write(source_metric_path, source_metric)

    metric_path = root / "source" / "recovery_metrics" / "rank_00000.json"
    metric = json.loads(metric_path.read_text())
    metric["surviving_host"] = 1 - surviving_host
    _write(metric_path, metric)
    assert subprocess.run([sys.executable, str(ANALYZER), str(root)]).returncode != 0


def test_object_semantic_gate_analyzer_accepts_uninterrupted_reference(tmp_path):
    root = tmp_path / "reference"
    _write(
        root / "run-contract.json",
        {
            "arm": "nockpt",
            "run_id": "reference",
            "restore_enabled": False,
            "checkpoint_enabled": False,
            "source_steps": 15,
            "restore_continue_steps": 0,
            "reference_steps": 15,
            "nodes": [17, 8],
            "world_size": 16,
            "ranks_per_host": 8,
            "model": "gpt2",
            "parallelism": {"pp": 4, "tp": 1, "dp": 4, "mesh_order": "pp-dp"},
            "workload": {
                "sequence_length": 128,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "seed": 20260913,
                "precision": "bf16",
            },
            "scheduler": "constant",
            "optimizer": {
                "name": "scalar_adamw",
                "learning_rate": 5e-5,
                "weight_decay": 0.0,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-8,
            },
        },
    )
    _write(root / "launcher-status.json", {"reference": {"local": 0, "remote": 0}})
    _write(root / "phase-times.json", {"reference": {"duration_ns": 1}})
    _digests(root / "reference" / "digests", 15)
    log = root / "logs" / "reference" / "node-0.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "\n".join(
            f"step wall: {step} train-plus-checkpoint-poll time (s): 1.0\n"
            f"steps: {step} loss: {10 - step / 10:.4f} iter time (s): 1.0"
            for step in range(1, 16)
        )
        + "\n"
    )

    subprocess.run([sys.executable, str(ANALYZER), str(root)], check=True)
    result = json.loads((root / "analysis.json").read_text())
    assert result["reference"]["rank_digests"] == 16
    assert len(result["reference"]["loss_trace"]) == 15


def test_dense_training_contract_is_admitted_without_relaxing_topology():
    from tools.analyze_object_semantic_gate import validate_topology

    contract = {
        "nodes": [17, 8],
        "world_size": 16,
        "ranks_per_host": 8,
        "model": "llama_3_8b",
        "parallelism": {"pp": 4, "tp": 1, "dp": 4, "mesh_order": "pp-dp"},
    }
    assert validate_topology(contract) == 16
    contract["parallelism"] = {"pp": 4, "tp": 2, "dp": 2, "mesh_order": "dp-pp"}
    with pytest.raises(ValueError, match="not frozen"):
        validate_topology(contract)
