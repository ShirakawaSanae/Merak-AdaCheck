#!/usr/bin/env python3
"""Run the auditable 16-rank CPU/Gloo SemanticFixed service-scale gate."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_RUN_DIR = Path("runs/two-node-v2/object-semantic-service-scale-gate-r120-20260913")
FALLBACK_PYTHON = Path("/opt/conda/envs/sjw_ckpt_grouping/bin/python")


def _python() -> str:
    probe = [sys.executable, "-c", "import torch"]
    if subprocess.run(probe, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        return sys.executable
    if FALLBACK_PYTHON.is_file():
        return str(FALLBACK_PYTHON)
    raise RuntimeError("the CPU gate requires a Python environment with torch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    complete = args.run_dir / "COMPLETE"
    complete.unlink(missing_ok=True)
    summary = args.run_dir / "summary.json"
    summary.unlink(missing_ok=True)
    junit = args.run_dir / "junit.xml"
    command = [
        _python(),
        "-m",
        "pytest",
        "-q",
        "test/core/checkpoint/test_object_semantic_service_scale.py",
        f"--junitxml={junit}",
    ]
    environment = {**os.environ, "OBJECT_SEMANTIC_SCALE_SUMMARY": str(summary)}
    result = subprocess.run(command, check=False, env=environment)
    if result.returncode or not summary.is_file():
        return result.returncode or 1
    prior_failure = args.run_dir / "junit-topology-development-failure.xml"
    prior_note = (
        "\nThe retained `junit-topology-development-failure.xml` is a development-only "
        "failure from the initial all-ranks-share-one-object topology. It produced zero "
        "remote bytes because every target host already had a local copy; it is not a "
        "negative system result.\n"
        if prior_failure.is_file() else ""
    )
    (args.run_dir / "README.md").write_text(
        "# Object semantic service-scale CPU gate r120\n\n"
        "Evidence level: CPU/loopback-TCP/Gloo correctness and service accounting only; "
        "this is not GPU, remote-host, trainer-throughput, or performance evidence.\n\n"
        "The gate uses 16 logical ranks, ten sealed suffix epochs, all three arms, and "
        "each single-service-host loss. Every fresh logical rank restores its sealed base "
        "and selective suffix, checking model, AdamW, scheduler, runtime state, exact GET "
        "sources, publication accounting, and retained service entries.\n\n"
        "Command:\n\n```bash\n" + " ".join(command) + "\n```\n" + prior_note
    )
    complete.write_text("16-rank, ten-suffix, three-arm, both-single-service-loss CPU gate passed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
