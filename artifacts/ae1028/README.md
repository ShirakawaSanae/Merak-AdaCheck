# Retained 16-Rank Experiment Evidence

The cluster was reclaimed before the complete planned matrix could finish.
`docs/10-27.md` is the final interpretation; unfinished experiments must not
be inferred from launch scripts or planned run names.

This directory retains selected raw metadata, command lines, configurations,
state digests, metrics, primary logs and contemporaneous source archives.
`manifest.json` records the retained files and SHA-256 hashes. The original
large `runs/` and `experiments/` trees are removed after the final M1 run.
Historical absolute paths inside raw evidence remain unchanged as provenance.
For example, original `runs/moe-capacity/...` evidence now resides at
`artifacts/ae1028/runs/moe-capacity/...`. Service endpoints in these files are
historical; retained locators are not live checkpoints.

Model weights, CPU checkpoint payloads, build outputs, tensorboard files,
caches, repeated torchrun logs and unrelated preliminary trials are not kept.
Raw evidence distinguishes failed attempts, deliberate interruptions and
successful recovery. In particular, the Host source's FAILED marker refers
to the interrupted old reader; the r2 retry carries the valid final verdict.

## Offline Analysis

From the repository root:

```bash
python tools/analyze_moe_snapshot.py \
  artifacts/ae1028/runs/moe-capacity/20261028-p2-ds-fixed \
  --output /tmp/deepseek-fixed-analysis.json
python tools/analyze_moe_snapshot.py \
  artifacts/ae1028/runs/moe-capacity/20261028-p2-ds-host \
  --restore-run artifacts/ae1028/runs/moe-restore/20261028-p2-ds-host-r2 \
  --output /tmp/deepseek-host-analysis.json
```

Tiny model gates use the same analyzer with their retained run directories.
Dense primary suite analyses and all four corresponding arm directories are
retained under `runs/two-node-v2/`, including detector configurations and
reference/source/restore digests. `runs/research-summary-20261027/` contains
the checked dense contracts and condensed intermediate results.

## Rerun Inputs

Use `tools/run_two_node_moe_capacity.sh` and
`tools/run_two_node_moe_zero1_smoke.sh` with fresh run IDs on available hardware.
The existing launchers target node17/node8 and the prepared Python environment;
adapt site-specific addresses and environment paths before using another cluster.
All new GPU experiments should run under `tools/run_in_tmux.sh`.

DeepSeek architecture/configuration files and their upstream license are kept
in `third_party/deepseek_v2_lite/`. For dense experiments, set
`CKPT_OBJECT_DETECTOR_CONFIG` to the retained arm's
`detector-experiment-config.json`, and match its model/layout contract and
recorded command. Do not substitute another model's detector configuration.

The source-only `code1028.zip` deliberately contains only `.py`, `.sh`, `.md`;
the checkout also retains necessary JSON configurations, licenses, and evidence.
Thus the ZIP alone does not include all reproduction inputs.
