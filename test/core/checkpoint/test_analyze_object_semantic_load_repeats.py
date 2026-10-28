import json
import runpy
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ANALYZER = ROOT / "tools" / "analyze_object_semantic_load_repeats.py"
FIXTURES = runpy.run_path(str(Path(__file__).with_name("test_analyze_object_semantic_suite.py")))


def write_plan_load(root, arm):
    if arm == "host_global_fixed":
        loads = [[0, 30], *[[rank, 0] for rank in range(1, 16)]]
    else:
        loads = [[0, 15], [1, 15], *[[rank, 0] for rank in range(2, 16)]]
    value = {
        "publisher_policy": f"{arm}_policy",
        "all_publication_publisher_bytes": loads,
        "total_all_publication_logical_bytes": 30,
        "maximum_all_publication_publisher_bytes": max(item[1] for item in loads),
        "total_remote_logical_bytes": 15,
        "maximum_remote_publisher_bytes": 15,
    }
    for rank in range(16):
        path = root / "source" / "metrics" / f"rank_{rank:05d}.json"
        metric = json.loads(path.read_text())
        metric["final_drain_ms"] = rank + 1.0
        metric["plan_load_metrics"] = value
        for item in metric["commit_metrics"]:
            item.update({
                "distributed_commit_ms": 1.0,
                "local_bundle_commit_ms": 1.0,
                "manifest_gather_ms": 1.0,
            })
        FIXTURES["write"](path, metric)


def build_runs(tmp_path):
    roots = []
    for index, arm in enumerate(("host_global_fixed", "m1") * 3):
        root = tmp_path / f"repeat-{index}"
        root.mkdir()
        (root / "COMPLETE").write_text("complete\n")
        (root / "detector-experiment-config.json").write_text("same detector\n")
        FIXTURES["checkpoint_artifacts"](root, arm)
        if index != 0:
            write_plan_load(root, arm)
        else:
            for rank in range(16):
                path = root / "source" / "metrics" / f"rank_{rank:05d}.json"
                metric = json.loads(path.read_text())
                metric["final_drain_ms"] = rank + 1.0
                for item in metric["commit_metrics"]:
                    item.update({
                        "distributed_commit_ms": 1.0,
                        "local_bundle_commit_ms": 1.0,
                        "manifest_gather_ms": 1.0,
                    })
                FIXTURES["write"](path, metric)
        roots.append(root)
    return roots


def run(roots, output):
    return subprocess.run(
        [sys.executable, str(ANALYZER), *map(str, roots), "--output", str(output)],
        check=False,
    )


def test_load_repeat_analyzer_recomputes_gate_and_rejects_raw_tamper(tmp_path):
    roots = build_runs(tmp_path)
    output = tmp_path / "repeats.json"
    assert run(roots, output).returncode == 0
    result = json.loads(output.read_text())
    assert result["runs"]["m1_1"]["base_maximum_remote_put_ms"] == 3.0
    assert result["runs"]["m1_1"]["suffix_maximum_remote_put_ms"] == 2.0
    assert result["runs"]["m1_1"]["overall_maximum_remote_put_ms"] == 3.0
    assert result["three_sample_statistics"]["m1"]["suffix_maximum_remote_put_ms"]["count"] == 3
    assert "suffix_maximum_remote_put_ms" in result["m1_vs_host_reduction_percent"]
    assert result["runs"]["host1"]["plan_load_metrics"] is None
    assert result["runs"]["m1_1"]["timing_ms"]["distributed_commit_ms"]["count"] == 160

    path = roots[1] / "source" / "metrics" / "rank_00000.json"
    metric = json.loads(path.read_text())
    metric["commit_metrics"][0]["remote_gradient_logical_bytes"] = 999
    FIXTURES["write"](path, metric)
    assert run(roots, output).returncode != 0
