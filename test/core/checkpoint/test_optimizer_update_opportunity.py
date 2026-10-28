import hashlib
import json

from tools.analyze_optimizer_update_opportunity import (
    analyze_transition,
    load_epoch_catalogs,
)


def _write_epoch(root, epoch, checksums):
    for rank, checksum in enumerate(checksums):
        record = {
            "schema_version": 1,
            "run_kind": "merak_pre_optimizer_update_catalog",
            "snapshot_point": "post_unscale_pre_optimizer_step",
            "run_id": "run-a",
            "rank": rank,
            "world_size": len(checksums),
            "epoch": epoch,
            "base_epoch": 0,
            "base_id": f"base-{rank}",
            "binding": {
                "rank": rank,
                "world_size": len(checksums),
                "pipeline_parallel_rank": rank,
                "tensor_parallel_rank": 0,
                "data_parallel_rank": 0 if rank < len(checksums) // 2 else 1,
            },
            "topology_hash": f"topology-{rank}",
            "parameter_ids": [[f"weight-{rank}"]],
            "gradient_tensor_count": 1,
            "missing_gradient_count": 0,
            "gradient_logical_bytes": 4,
            "tensor_logical_bytes": 4,
            "objects": [
                {
                    "object_id": "path-id",
                    "path": "/str:gradients/int:0/int:0",
                    "role": "gradients",
                    "dtype": "torch.float32",
                    "shape": [1],
                    "logical_bytes": 4,
                    "checksum": checksum,
                }
            ],
            "adacheck_detector_filtered": False,
            "adacheck_payload_faithful": False,
            "eligible_for_runtime_gate": False,
            "limitations": [],
        }
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        record["catalog_hash"] = hashlib.sha256(encoded).hexdigest()
        destination = root / f"rank_{rank:05d}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(record))


def test_previous_epoch_plan_is_evaluated_without_hindsight(tmp_path):
    plan_root = tmp_path / "epoch_1"
    evaluation_root = tmp_path / "epoch_2"
    _write_epoch(plan_root, 1, ("a", "b", "b", "a"))
    _write_epoch(evaluation_root, 2, ("c", "d", "c", "e"))

    result = analyze_transition(
        load_epoch_catalogs(plan_root),
        load_epoch_catalogs(evaluation_root),
        ranks_per_host=2,
    )

    assert result["policies"]["fixed"]["pair_local_transfer_bytes"] == 8
    frozen = result["policies"]["frozen_previous_epoch_exact"]
    assert frozen["grouping"]["groups"] == [[0, 3], [1, 2]]
    assert frozen["pair_local_transfer_bytes"] == 16
    hindsight = result["policies"]["current_epoch_hindsight_exact"]
    assert hindsight["grouping"]["groups"] == [[0, 2], [1, 3]]
    assert result["frozen_plan_delta_reduction_percent"] == -100.0
    assert result["runtime_causal_plan_executed"] is False
    assert not result["eligible_for_runtime_gate"]
