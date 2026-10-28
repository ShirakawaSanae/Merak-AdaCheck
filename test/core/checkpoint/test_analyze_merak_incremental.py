import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).parents[3] / "tools" / "analyze_merak_incremental.py"
SPEC = importlib.util.spec_from_file_location("analyze_merak_incremental", MODULE_PATH)
analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analysis)


def test_remote_data_plane_max_time_uses_longest_edge_not_rank_sum():
    summary = analysis.summarize_remote_data_plane(
        [
            {"epoch": 1, "path": "remote_delta", "transport_ms": 4.0,
             "payload_bytes": 10, "wire_bytes": 12, "logical_bytes": 9},
            {"epoch": 1, "path": "remote_delta", "transport_ms": 7.0,
             "payload_bytes": 20, "wire_bytes": 24, "logical_bytes": 18},
            {"epoch": 1, "path": "local_full", "transport_ms": 99.0,
             "payload_bytes": 30, "wire_bytes": 36, "logical_bytes": 27},
            {"epoch": 2, "path": "remote_delta", "transport_ms": 6.0,
             "payload_bytes": 40, "wire_bytes": 48, "logical_bytes": 36},
        ]
    )

    assert summary["max_time_ms"] == 7.0
    assert summary["mean_epoch_max_time_ms"] == 6.5
    assert summary["epoch_maxima"] == [
        {"epoch": 1, "edge_count": 2, "max_time_ms": 7.0,
         "application_payload_bytes": 30, "application_wire_bytes": 36,
         "logical_bytes": 27},
        {"epoch": 2, "edge_count": 1, "max_time_ms": 6.0,
         "application_payload_bytes": 40, "application_wire_bytes": 48,
         "logical_bytes": 36},
    ]
    assert summary["application_payload_bytes"] == 70
    assert summary["application_wire_bytes"] == 84
    assert summary["logical_bytes"] == 63
    assert summary["nic_bytes"] is None
    assert summary["training_blocking_ms"] is None


def test_remote_data_plane_marks_legacy_logical_bytes_unknown():
    summary = analysis.summarize_remote_data_plane(
        [{"epoch": 1, "path": "remote_full", "transport_ms": 2.0,
          "payload_bytes": 10, "wire_bytes": 12}]
    )

    assert summary["logical_bytes"] is None
    assert "legacy" in summary["logical_bytes_reason"]


def test_training_step_summary_excludes_first_warmup_step(tmp_path):
    log = tmp_path / "node-0.log"
    log.write_text(
        "step wall: 1 train-plus-checkpoint-poll time (s): 9.000\n"
        "step wall: 2 train-plus-checkpoint-poll time (s): 2.000\n"
        "step wall: 3 train-plus-checkpoint-poll time (s): 4.000\n"
    )

    summary = analysis.summarize_training_steps(log)

    assert summary["sample_count"] == 3
    assert summary["steady_sample_count"] == 2
    assert summary["steady_mean_seconds"] == 3.0
    assert summary["steady_p50_seconds"] == 3.0
    assert summary["steady_max_seconds"] == 4.0
    assert summary["training_window_wall_seconds"] == 15.0


def test_phase_wall_time_keeps_train_and_restore_separate(tmp_path):
    (tmp_path / "train-phase-timing.txt").write_text(
        "started_utc=a\nfinished_utc=b\nduration_ns=2500000000\n"
    )
    (tmp_path / "restore-phase-timing.txt").write_text(
        "started_utc=c\nfinished_utc=d\nduration_ns=1250000000\n"
    )

    summary = analysis.phase_wall_time(tmp_path)

    assert summary["phases"]["train"]["duration_seconds"] == 2.5
    assert summary["phases"]["restore"]["duration_seconds"] == 1.25
    assert summary["total_recorded_phase_wall_seconds"] == 3.75


def test_checkpoint_rate_uses_boundary_progress_and_excludes_final_drain():
    boundaries = [
        {"step": 1, "max_global_lag_steps": 1},
        {"step": 2, "max_global_lag_steps": 2},
        {"step": 3, "max_global_lag_steps": 2},
        {"step": 4, "max_global_lag_steps": 2},
        {"step": 5, "max_global_lag_steps": 3},
    ]

    summary = analysis.summarize_training_checkpoint_rate(boundaries)

    expected_fields = {
        "optimizer_steps_observed": 5,
        "globally_visible_checkpoints_during_training": 2,
        "optimizer_steps_per_visible_checkpoint": 2.5,
        "longest_steps_without_global_progress": 2,
    }
    for key, expected in expected_fields.items():
        assert summary[key] == expected
    assert [item["globally_visible_epoch"] for item in boundaries] == [0, 0, 1, 2, 2]


def test_communication_profile_reports_cross_rank_epoch_tail(tmp_path):
    root = tmp_path / "checkpoint" / "communication-profile"
    root.mkdir(parents=True)
    for rank, durations in enumerate(((1_000_000, 3_000_000), (2_000_000, 4_000_000))):
        records = [
            {
                "schema_version": 1,
                "kind": "interval",
                "name": "checkpoint_poll",
                "duration_ns": duration,
                "fields": {"rank": rank, "epoch": epoch},
            }
            for epoch, duration in enumerate(durations, start=1)
        ]
        (root / f"rank_{rank:05d}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )

    summary = analysis.summarize_communication_profiles(tmp_path)

    assert summary["rank_file_count"] == 2
    poll = summary["intervals"]["checkpoint_poll"]
    assert poll["all_rank_samples"]["max_ms"] == 4.0
    assert poll["per_epoch_rank_maxima"]["count"] == 2
    assert poll["per_epoch_rank_maxima"]["mean_ms"] == 3.0


def test_update_manifest_role_bytes_excludes_base(tmp_path):
    root = tmp_path / "checkpoint" / "manifests" / "full-remote" / "rank_00000"
    root.mkdir(parents=True)
    for epoch, objects in (
        (0, [{"role": "model", "logical_bytes": 100}]),
        (1, [
            {"role": "gradients", "logical_bytes": 20},
            {"role": "runtime_state", "logical_bytes": 3},
        ]),
    ):
        (root / f"epoch_{epoch:08d}.manifest.json").write_text(
            json.dumps({"epoch": epoch, "objects": objects})
        )

    summary = analysis.manifest_update_role_bytes(
        tmp_path, "full-remote", base_epoch=0, final_epoch=1
    )

    assert summary["manifest_count"] == 1
    assert summary["total_by_role"] == {"gradients": 20, "runtime_state": 3}
