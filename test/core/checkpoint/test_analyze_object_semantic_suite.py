import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ANALYZER = ROOT / "tools" / "analyze_object_semantic_suite.py"

COMMON = {
    "nodes": [17, 8], "world_size": 16, "ranks_per_host": 8, "model": "gpt2",
    "parallelism": {"pp": 4, "tp": 1, "dp": 4, "mesh_order": "pp-dp"},
    "workload": {"sequence_length": 128, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 1, "seed": 20260913, "precision": "bf16"},
    "scheduler": "constant",
    "optimizer": {"name": "scalar_adamw", "learning_rate": 5e-5, "weight_decay": 0.0, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
}
SCOPES = {
    "reference": "uninterrupted fifteen-step no-checkpoint reference",
    "source": "synchronized base, ten training steps, and final drain",
    "restore": "fresh construction, restore to step 10, and five continued steps",
}


def scopes(source_steps, continue_steps):
    reference_steps = source_steps + continue_steps
    return {
        "reference": (
            "uninterrupted fifteen-step no-checkpoint reference"
            if reference_steps == 15 else f"uninterrupted {reference_steps}-step no-checkpoint reference"
        ),
        "source": (
            "synchronized base, ten training steps, and final drain"
            if source_steps == 10 else f"synchronized base, {source_steps} training steps, and final drain"
        ),
        "restore": (
            "fresh construction, restore to step 10, and five continued steps"
            if (source_steps, continue_steps) == (10, 5)
            else f"fresh construction, restore to step {source_steps}, and {continue_steps} continued steps"
        ),
    }


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def locator(path, kind, epoch, run_id, **extra):
    value = {"schema_version": 1, "kind": kind, "run_id": run_id, "epoch": epoch, "world_size": 16, **extra}
    value["locator_hash"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    write(path, value)


def digests(root, iteration):
    for rank in range(16):
        write(root / f"rank_{rank:05d}.json", {"rank": rank, "iteration": iteration, "state_checksum": f"digest-{rank}"})


def log(root, phase, steps):
    (root / "logs" / phase).mkdir(parents=True, exist_ok=True)
    (root / "logs" / phase / "node-0.log").write_text("\n".join(
        f"step wall: {step} train-plus-checkpoint-poll time (s): 1.0\nsteps: {step} loss: {10 - step / 10:.4f} iter time (s): 1.0"
        for step in steps
    ) + "\n")


def phase_times(*names, values=SCOPES):
    return {
        name: {"started_monotonic_ns": 10, "finished_monotonic_ns": 20, "duration_ns": 10, "scope": values[name]}
        for name in names
    }


def checkpoint_contract(arm, run_id, *, common=COMMON, source_steps=10, continue_steps=5, failed_service_node=8):
    return {
        **common, "arm": arm, "run_id": run_id, "checkpoint_enabled": True, "restore_enabled": True,
        "source_steps": source_steps, "restore_epoch": source_steps,
        "restore_continue_steps": continue_steps, "reference_steps": source_steps + continue_steps,
        "failed_service_node": failed_service_node,
        "surviving_host": 0 if failed_service_node == 8 else 1, "max_inflight": source_steps,
        "base_upload_parallelism": 4,
        "service_capacity": {"configured_mib": 163840}, "cadence_claim": "final-drain",
    }


def checkpoint_artifacts(root, arm, **kwargs):
    run_id, plan = f"fixture-{arm}", f"plan-{arm}"
    contract = checkpoint_contract(arm, run_id, **kwargs)
    source_steps = contract["source_steps"]
    continue_steps = contract["restore_continue_steps"]
    reference_steps = contract["reference_steps"]
    failed_service_node = contract["failed_service_node"]
    write(root / "run-contract.json", contract)
    write(root / "launcher-status.json", {"source": {"local": 0, "remote": 0}, "restore": {"local": 0, "remote": 0}})
    write(root / "phase-times.json", phase_times("source", "restore", values=scopes(source_steps, continue_steps)))
    log(root, "source", range(1, source_steps + 1))
    log(root, "restore", range(source_steps + 1, reference_steps + 1))
    base_metrics = {"remote_data_payload_bytes": 10, "remote_control_payload_bytes": 2, "remote_wire_bytes": 15, "maximum_remote_put_ms": 3.0}
    for rank in range(16):
        write(root / "source" / "metrics" / f"rank_{rank:05d}.json", {
            "rank": rank, "arm": arm, "base_epoch": 0, "committed_epochs": list(range(1, source_steps + 1)), "valid": True,
            "all_epochs_committed": True, "maximum_boundary_lag": 2, "strict_1c1s_valid": False,
            "base_result": {"metrics": base_metrics},
            "commit_metrics": [{"epoch": epoch, "capture_ms": 1.0, "publisher_queue_ms": 1.0, "snapshot_ready_wait_ms": 0.0,
                                "enqueue_to_commit_ms": 1.0, "remote_gradient_logical_bytes": 15,
                                "remote_data_payload_bytes": 18, "remote_control_payload_bytes": 2,
                                "total_remote_payload_bytes": 20, "remote_wire_bytes": 30, "maximum_remote_put_ms": 2.0}
                               for epoch in range(1, source_steps + 1)],
        })
        write(root / "source" / "recovery_metrics" / f"rank_{rank:05d}.json", {
            "rank": rank, "requested_epoch": source_steps, "final_epoch": source_steps, "base_epoch": 0, "arm": arm,
            "plan_hash": plan, "mapping_hash": "mapping", "surviving_host": contract["surviving_host"], "total_restore_wall_ms": 1.0,
            "boundary_checksum": "boundary", "component_checksums": {}, "base_payload_bytes": 1,
            "base_wire_bytes": 1, "base_get_ms": 1.0, "suffix_get_metrics": "unavailable", "checksum_diagnostic_only": True,
        })
    digests(root / "source" / "digests", source_steps)
    digests(root / "restore" / "digests", reference_steps)
    locator(root / "source" / "base" / "epoch-00000000.json", "object_semantic_base_locator", 0, run_id)
    for epoch in range(1, source_steps + 1):
        locator(root / "source" / "updates" / f"epoch-{epoch:08d}.json", "object_optimizer_update_locator", epoch, run_id, arm=arm, plan_hash=plan, detector_catalog_hash="catalog")
    for name in ("restore-service-before.json", "restore-service-after.json"):
        write(root / name, {
            "node17": {"status": "unavailable" if failed_service_node == 17 else "ok"},
            "node8": {"status": "unavailable" if failed_service_node == 8 else "ok"},
        })


def make_suite(tmp_path, *, parallelism=None, source_steps=10, continue_steps=5, failed_service_node=8):
    common = {**COMMON, "parallelism": parallelism or COMMON["parallelism"]}
    reference_steps = source_steps + continue_steps
    roots = {arm: tmp_path / arm for arm in ("nockpt", "adacheck_fixed", "host_global_fixed", "m1")}
    for root in roots.values():
        root.mkdir(parents=True)
        (root / "detector-experiment-config.json").write_text("same-detector\n")
        (root / "COMPLETE").write_text("complete\n")
    reference = roots["nockpt"]
    write(reference / "run-contract.json", {**common, "arm": "nockpt", "run_id": "fixture-nockpt", "checkpoint_enabled": False, "restore_enabled": False, "source_steps": reference_steps, "restore_continue_steps": 0, "reference_steps": reference_steps})
    write(reference / "launcher-status.json", {"reference": {"local": 0, "remote": 0}})
    write(reference / "phase-times.json", phase_times("reference", values=scopes(source_steps, continue_steps)))
    log(reference, "reference", range(1, reference_steps + 1))
    digests(reference / "reference" / "digests", reference_steps)
    for arm in ("adacheck_fixed", "host_global_fixed", "m1"):
        checkpoint_artifacts(roots[arm], arm, common=common, source_steps=source_steps, continue_steps=continue_steps, failed_service_node=failed_service_node)
    return roots


def run(roots, output):
    return subprocess.run(
        [sys.executable, str(ANALYZER), *[str(roots[arm]) for arm in ("nockpt", "adacheck_fixed", "host_global_fixed", "m1")], "--output", str(output)],
        check=False,
    )


def test_object_semantic_suite_recomputes_gate_evidence_and_rejects_raw_tamper(tmp_path):
    roots = make_suite(tmp_path)
    output = tmp_path / "suite.json"
    write(roots["m1"] / "analysis.json", {"arm": "m1", "untrusted": True})
    assert run(roots, output).returncode == 0
    result = json.loads(output.read_text())
    assert result["loss_continuity"]["m1"]["valid"] is True
    assert result["summary"]["m1"]["transfer"]["total_remote_payload_bytes"] == 212
    assert result["summary"]["m1"]["transfer"]["suffix_remote_gradient_logical_bytes"] == 150
    assert result["summary"]["m1"]["maximum_remote_put_ms"] == {"base": 3.0, "suffix": 2.0, "overall": 3.0}
    assert result["summary"]["m1"]["rto_ms"]["count"] == 16
    assert result["final_digest_match"]["m1"] is True

    metric_path = roots["m1"] / "source" / "recovery_metrics" / "rank_00000.json"
    metric = json.loads(metric_path.read_text())
    metric["surviving_host"] = 1
    write(metric_path, metric)
    assert run(roots, output).returncode != 0


def test_object_semantic_suite_rejects_mixed_base_upload_parallelism(tmp_path):
    roots = make_suite(tmp_path)
    output = tmp_path / "suite.json"
    contract_path = roots["m1"] / "run-contract.json"
    contract = json.loads(contract_path.read_text())
    contract["base_upload_parallelism"] = 2
    write(contract_path, contract)
    assert run(roots, output).returncode != 0


def test_object_semantic_suite_maps_legacy_missing_base_parallelism_to_four(tmp_path):
    roots = make_suite(tmp_path)
    output = tmp_path / "suite.json"
    for arm in ("adacheck_fixed", "host_global_fixed", "m1"):
        contract_path = roots[arm] / "run-contract.json"
        contract = json.loads(contract_path.read_text())
        del contract["base_upload_parallelism"]
        write(contract_path, contract)
    assert run(roots, output).returncode == 0
    result = json.loads(output.read_text())
    assert result["checkpoint_fault_contract"]["base_upload_parallelism"] == 4


def test_object_semantic_suite_accepts_pp2_three_plus_five_and_node17_loss(tmp_path):
    roots = make_suite(
        tmp_path,
        parallelism={"pp": 2, "tp": 1, "dp": 8, "mesh_order": "pp-dp"},
        source_steps=3,
        continue_steps=5,
        failed_service_node=17,
    )
    output = tmp_path / "suite-pp2.json"
    assert run(roots, output).returncode == 0
    result = json.loads(output.read_text())
    assert result["checkpoint_fault_contract"]["failed_service_node"] == 17
    assert result["summary"]["m1"]["step_wall"]["source"]["count"] == 3
    assert result["summary"]["m1"]["step_wall"]["restore"]["count"] == 5
