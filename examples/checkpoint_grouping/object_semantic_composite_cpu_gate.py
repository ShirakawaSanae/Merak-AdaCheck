#!/usr/bin/env python3
"""Run the CPU/TCP/Gloo composite SemanticFixed recovery gate and retain evidence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_RUN_DIR = Path("runs/two-node-v2/object-semantic-composite-gate-r119-20260913")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    junit = args.run_dir / "junit.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "test/core/checkpoint/test_object_semantic_composite.py",
        f"--junitxml={junit}",
    ]
    result = subprocess.run(command, check=False)
    if result.returncode:
        return result.returncode
    (args.run_dir / "README.md").write_text(
        "# Object semantic composite CPU gate r119\n\n"
        "Evidence level: CPU/loopback-TCP/Gloo correctness only; this is not GPU "
        "or end-to-end trainer performance evidence.\n\n"
        "The gate commits real sealed bases and two distributed SemanticFixed "
        "suffixes for all three arms, loses one local service host, and recovers "
        "a fresh AdamW model/optimizer/scheduler/runtime state from the survivor.\n\n"
        "Command:\n\n```bash\n"
        + " ".join(command)
        + "\n```\n"
    )
    (args.run_dir / "COMPLETE").write_text(
        "six CPU/TCP/Gloo composite base-plus-suffix recovery cases passed\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
