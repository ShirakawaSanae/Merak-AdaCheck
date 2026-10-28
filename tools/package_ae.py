"""Build the curated source-only AE archive from the working checkout."""

import hashlib
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    trees = (
        "Merak", "examples/checkpoint_grouping", "process_monitor/procguard",
        "test/core/checkpoint", "test/core/moe",
    )
    files = set()
    allowed = {".py", ".sh", ".md"}
    for tree in trees:
        for path in (root / tree).rglob("*"):
            if path.is_file() and not path.is_symlink() and path.suffix in allowed:
                if not any(p.startswith(".") or p == "__pycache__"
                           for p in path.relative_to(root).parts):
                    files.add(path.relative_to(root).as_posix())
    for pattern in ("analyze_*.py", "aggregate_*.py", "compare_*.py",
                    "build_exact_peer_oracle.py", "package_ae.py"):
        files.update(p.relative_to(root).as_posix()
                     for p in (root / "tools").glob(pattern))
    files.update((
        "AE_README.md", "README.md", "setup_experiment.sh",
        "process_monitor/README.md",
        "tools/run_in_tmux.sh", "tools/launch_ckpt_grouping.sh",
        "tools/run_two_node_moe_zero1_smoke.sh",
        "tools/run_two_node_moe_capacity.sh", "tools/run_two_node_moe_restore.sh",
        "tools/run_moe_full_policy_queue.sh",
        "tools/run_two_node_object_semantic.sh",
        "tools/preflight_moe_ep_zero1.py",
        "docs/moe_snapshot_ae.md", "docs/09-24.md",
        "docs/adacheck_semantic_baseline.md", "docs/communication_profile.md",
        "docs/two_node_v2_experiment_matrix.md",
    ))
    mapping = {name: name for name in files}
    mapping["LICENSE.md"] = "LICENSE"
    output = root / "code1028.zip"
    manifest = ["# AE File Manifest", "",
                "SHA-256 covers the exact archived bytes. Paths are relative to the checkout.",
                "", "| Archive path | Source path | SHA-256 |",
                "| --- | --- | --- |"]
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, source in sorted(mapping.items()):
            path = root / source
            data = path.read_bytes()
            info = zipfile.ZipInfo.from_file(path, arcname=name)
            archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)
            digest = hashlib.sha256(data).hexdigest()
            manifest.append(f"| `{name}` | `{source}` | `{digest}` |")
        archive.writestr("AE_MANIFEST.md", "\n".join(manifest) + "\n")
    with zipfile.ZipFile(output) as archive:
        assert all(Path(n).suffix in allowed and not n.startswith("/")
                   and ".." not in Path(n).parts for n in archive.namelist())
        assert archive.testzip() is None
        print(f"{output}: {len(archive.namelist())} files, {output.stat().st_size} bytes; CRC OK")
        for suffix in sorted(allowed):
            print(f"{suffix}: {sum(Path(n).suffix == suffix for n in archive.namelist())}")


if __name__ == "__main__":
    main()
