# AE Source Snapshot: 2026-10-28

Refreshed after the top-k dispatch determinism fix. The initial archive is
retained in the original checkout under
`runs/research-summary-20261027/code1028-initial.zip` (not bundled here).

This archive contains only Python, shell, and Markdown files from the working
tree. Start here rather than the upstream README. It is a source snapshot,
not a completed experimental-results bundle or an offline environment image.

## Contents

- `Merak/`: training, checkpoint grouping, publication, and recovery code.
- `examples/checkpoint_grouping/`: workload definitions, services, CPU gates.
- `process_monitor/procguard/`: process-monitor runtime.
- `tools/`: selected two-node launchers and checkpoint analysis helpers.
- `test/core/checkpoint/`, `test/core/moe/`: focused correctness tests.
- `docs/moe_snapshot_ae.md`: tiny MoE policy reproduction and accounting.
- `docs/09-24.md`: metric definitions, particularly section 9.
- `docs/adacheck_semantic_baseline.md` and
  `docs/two_node_v2_experiment_matrix.md`: dense experimental contracts.
- `AE_MANIFEST.md`: exact file hashes and source-path mapping for this archive.

Logs, checkpoints, weights, raw results, caches, historical source copies,
third-party trees, native extension sources, and non-approved file types are
excluded. The upstream LICENSE is preserved verbatim as `LICENSE.md`.

## Environment And External Inputs

The launchers target node17 (172.20.17.2) and node8 (172.20.8.2), eight GPUs
each, with passwordless SSH and the same shared checkout path. They currently
use `/opt/conda/envs/sjw_ckpt_grouping/bin/python`. Adapt these site-specific
paths and addresses before running elsewhere. Do not run two GPU experiments
concurrently on these nodes.

Use the prepared Python 3.10 environment: PyTorch 2.3.1/CUDA 12.1,
transformers 4.42.4, datasets 2.21.0, numpy 1.26.4, safetensors 0.4.5.
Other runtime dependencies include scipy, networkx, psutil, pynvml,
tensorboard, tensorboardX, dill, more-itertools, pulp, pyyaml, tqdm,
pyzmq, requests, flask, and flask-socketio; tests use pytest.
These are environment requirements, not a freshly validated installer.

From the extracted root:

```bash
export CKPT_RUNTIME_ONLY=1
source setup_experiment.sh
export PYTHONPATH="$PWD:$PWD/process_monitor${PYTHONPATH:+:$PYTHONPATH}"
```

The selected uniform pipeline partition does not require the optional native
autopipe extension. An upstream build installer and nnScaler setup are not
included because their required native/vendor inputs are excluded.

DeepSeek uses external `config.json` and model-definition Python files under
`CKPT_DEEPSEEK_MODEL_DIR`; inspect `examples/checkpoint_grouping/workloads.py`
for the loading contract. Full architecture experiments use random weights.
Dense runs require a model/layout-matched detector configuration supplied as
`CKPT_OBJECT_DETECTOR_CONFIG`; the runner's historical default under `runs/`
is NOT included. JSON configurations and raw result files are intentionally
outside this source-only archive. Existing documentation may link to those
excluded artifacts; these links are provenance, not bundled evidence.

## Run Entry Points

Use the commands in `docs/moe_snapshot_ae.md` for tiny Qwen/DeepSeek validation
of `adacheck_fixed`, `host_global_fixed`, and `m1`, always explicitly setting
`CKPT_MOE_MESH_ORDER=pp-dp`. Tiny correctness results are not full-model timing.

Full architecture source, snapshot, and fresh-process recovery chain:

```bash
tools/run_in_tmux.sh ae-ds-fixed runs/ae-ds-fixed.log -- \
  env CKPT_CAPACITY_PUBLISH=1 CKPT_CAPACITY_CHAIN_RESTORE=1 \
  CKPT_CAPACITY_STEPS=2 CKPT_CAPACITY_ARM=adacheck_fixed \
  CKPT_ZERO1_SERIAL_PUBLISH=host \
  bash tools/run_two_node_moe_capacity.sh ae-ds-fixed deepseek_v2_lite
```

Use unique run IDs. The full chain requires hundreds of GiB of host RAM per
node, checks the source/restored step-2 state digests, and retains the survivor
service on recovery failure. The source runner writes `COMPLETE` only after
the chain passes. Substitute `qwen2_moe_14b` for Qwen and the policy name for
other arms. Support in the runner is not evidence that every full-model arm
has already passed.

To run all six full model/policy combinations sequentially, with source
archives and stop-on-failure behavior:

```bash
tools/run_in_tmux.sh ae-policy-queue runs/ae-policy-queue.log -- \
  bash tools/run_moe_full_policy_queue.sh ae1028
```

Use a fresh campaign ID, at most 16 safe alphanumeric/dot/underscore/hyphen
characters. Run this queue only when both nodes are free; it starts with
DeepSeek fixed, then DeepSeek Host/M1 and Qwen fixed/Host/M1. Each arm must
pass the full-state continuation oracle before the next starts.

Dense entry point: `tools/run_two_node_object_semantic.sh ARM OUTPUT_ROOT`,
with model, layout, detector and cadence environment variables documented in
that script and the dense contract documents. Wrap it in `run_in_tmux.sh`.
Analysis entry points include `analyze_moe_snapshot.py` and
`analyze_object_semantic_suite.py`.

## Evidence Boundary

At packaging, both tiny MoE families have three-policy, 16-rank bitwise
recovery evidence. Both full architectures have training-capacity evidence.
Full Qwen fixed snapshot and survivor recovery continued training successfully,
but that run lacks an uninterrupted step-2 digest reference. Full DeepSeek p1
publication and fresh continuation completed, but model/optimizer step-2
digests differed on all ranks; this is a failed oracle, not a valid result.
A separate single-GPU top-6 repeated-forward test reproduced nondeterminism
without any checkpoint. The included dispatch fix uses unique assignments and
fixed-order reduction; repeated forward/backward plus CPU/Gloo and two-GPU
NCCL EP tests passed. A fresh full-model six-arm p2 queue is running to check
whether this resolves the full-state mismatch. Earlier tiny evidence predates
this arithmetic change and is not mislabeled as a new six-arm rerun.

A four-rank CPU test also reconstructs BF16 parameters exactly from serialized
FP32 optimizer shards after clearing live state. This supports the need for
a stronger encoder control omitting reconstructible model parameters; the
three-policy service/recovery version of that control is not yet implemented.
No completed full-model three-policy matrix, 32-rank result, sustained 1C1S,
or stable throughput advantage is claimed by this archive.

Rebuild from the original checkout with `python tools/package_ae.py`.
