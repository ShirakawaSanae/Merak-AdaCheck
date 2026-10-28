import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).parents[3] / "tools" / "analyze_matched_grouping.py"
SPEC = importlib.util.spec_from_file_location("analyze_matched_grouping", MODULE_PATH)
analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analysis)


def _write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def test_load_and_aggregate_matched_trial_excludes_warmup(tmp_path):
    run = tmp_path / "trial-01-fixed"
    run.mkdir()
    _write_json(
        run / "condition.json",
        {"model": "bert", "policy": "fixed", "trial": 1, "seed": 17},
    )
    _write_json(
        run / "summary.json",
        {
            "run_kind": "matched_detector_grouping_save",
            "fault": "none",
            "last_epoch_committed": True,
            "world_size": 2,
            "epochs": 3,
            "catalog_hash": "catalog",
            "plan_hash": "plan",
            "grouping": {"groups": [[0, 1]]},
            "cross_host_logical_bytes": 100,
            "nic_records": [
                {
                    "status": "available",
                    "workload_delta": {
                        "total_xmit_bytes": 300,
                        "total_rcv_bytes": 600,
                    },
                    "idle_delta": {
                        "total_xmit_bytes": 3,
                        "total_rcv_bytes": 6,
                    },
                }
            ],
        },
    )
    for rank, measured_commit in enumerate((30.0, 40.0)):
        events = []
        for commit_ms in (999.0, 998.0, measured_commit):
            events.append(
                {
                    "committed": True,
                    "commit_ms": commit_ms,
                    "exchange_ms": commit_ms / 2,
                    "exchange_wire_bytes": 10 + rank,
                    "service_wire_bytes": 20 + rank,
                    "control_wire_bytes": 7 if rank == 0 else 0,
                    "exchange_payload_bytes": 5 + rank,
                }
            )
        _write_json(run / f"rank-{rank:05d}.json", {"events": events})

    trial = analysis.load_trial(run / "condition.json", warmup_epochs=2)
    aggregated = analysis.aggregate([trial])

    assert trial["commit_mean_ms"] == 35.0
    assert trial["commit_p99_ms"] == 40.0
    assert trial["application_wire_bytes_per_epoch"] == 69
    assert trial["aggregate_pair_peak_bytes_per_epoch"] == 6
    assert trial["max_service_wire_bytes_per_epoch"] == 21
    assert trial["nic_xmit_bytes_per_epoch"] == 100
    assert trial["nic_rcv_bytes_per_epoch"] == 200
    assert trial["measured_epoch_count"] == 1
    assert aggregated[0]["commit_p99_ms_mean"] == 40.0
    assert aggregated[0]["commit_p99_ms_ci95"] == 0.0


def test_mean_ci95_uses_small_sample_interval():
    mean, interval = analysis.mean_ci95([10.0, 20.0, 30.0])

    assert mean == 20.0
    assert round(interval, 3) == 24.843


def test_capture_summary_and_reduction_plot(tmp_path):
    root = tmp_path / "capture"
    frozen = root / "frozen"
    frozen.mkdir(parents=True)
    _write_json(
        root / "workload.json",
        {"workload": {"name": "bert", "display_name": "BERT"}},
    )
    summary = frozen / "summary.json"
    _write_json(
        summary,
        {
            "run_kind": "matched_detector_capture_build",
            "all_rank_catalogs_reconstructed": True,
            "catalog_hash": "catalog",
            "full_logical_bytes": 1024,
            "reduced_logical_bytes": 256,
            "object_count": 3,
            "detector_confirmation": {"confirmed_pairs": 4, "rejected_pairs": 1},
        },
    )

    catalog = analysis.load_capture_summary(summary)
    analysis.plot_capture_reduction([catalog], tmp_path / "figure")

    assert catalog["model"] == "bert"
    assert catalog["reduced_logical_bytes"] == 256
    assert (tmp_path / "figure.png").is_file()
    assert (tmp_path / "figure.svg").is_file()
