import argparse
import json
import subprocess
import sys
from pathlib import Path

from Merak.core.checkpoint.catalog import CheckpointObject, TensorRedundancyEvidence
from examples.checkpoint_grouping.build_reduced_capture import build_metadata_capture


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _write_epoch(root, epoch, model_checksum):
    for rank in range(2):
        objects = (
            CheckpointObject(
                object_id="model-shared",
                path="/str:model/str:shared",
                role="model",
                dtype="torch.float32",
                shape=(4,),
                logical_bytes=16,
                checksum=model_checksum,
            ),
            CheckpointObject(
                object_id="zero-local",
                path="/str:zero_optimizer/str:local",
                role="zero_optimizer",
                dtype="torch.float32",
                shape=(2,),
                logical_bytes=8,
                checksum="same-but-unverified",
            ),
        )
        source = root / f"epoch_{epoch:08d}" / "source"
        _write_json(source / f"rank_{rank:05d}.objects.json", [x.to_dict() for x in objects])
        _write_json(
            source / f"rank_{rank:05d}.json",
            {
                "capture_format": "metadata_only",
                "rank": rank,
                "world_size": 2,
                "local_world_size": 1,
                "host": f"node{rank}",
                "epoch": epoch,
                "logical_bytes": 24,
                "object_count": 2,
                "detector_comparison_scope": "world",
                "detector_evidence_roles": ["model"],
                "conservative_unique_roles": ["zero_optimizer"],
            },
        )
        evidence = TensorRedundancyEvidence(
            rank=rank,
            role="model",
            path="/str:model/str:shared",
            local_index=0,
            equivalent_locations=((0, 0), (1, 0)),
        )
        _write_json(
            root / f"epoch_{epoch:08d}" / "detector" / f"rank_{rank:05d}.json",
            [evidence.to_dict()],
        )


def test_metadata_capture_keeps_unverified_zero_state_rank_local(tmp_path):
    capture = tmp_path / "capture"
    _write_epoch(capture, 1, "model-v1")
    _write_epoch(capture, 2, "model-v2")
    output = tmp_path / "frozen"

    build_metadata_capture(
        argparse.Namespace(
            capture_dir=capture / "epoch_00000001",
            stability_capture_dir=capture / "epoch_00000002",
            output_dir=output,
            ranks_per_host=1,
        )
    )

    summary = json.loads((output / "summary.json").read_text())
    objects = [
        json.loads(line)
        for line in (output / "object_catalog.jsonl").read_text().splitlines()
    ]
    assert summary["full_logical_bytes"] == 48
    assert summary["reduced_logical_bytes"] == 32
    assert summary["recoverable_payload_present"] is False
    assert summary["detector_stability"]["stable_records"] == 2
    shared = [item for item in objects if item["role"] == "model"]
    zero = [item for item in objects if item["role"] == "zero_optimizer"]
    assert len(shared) == 1 and shared[0]["source_ranks"] == [0, 1]
    assert len(zero) == 2

    analysis = tmp_path / "analysis"
    subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).parents[3]
                / "tools"
                / "analyze_peer_delta_opportunity.py"
            ),
            "--frozen-dir",
            str(output),
            "--output-dir",
            str(analysis),
        ],
        check=True,
    )
    result = json.loads((analysis / "summary.json").read_text())
    assert result["evidence_source"] == "adacheck_detector_confirmed_metadata_only"
    assert result["recoverable_payload_present"] is False
