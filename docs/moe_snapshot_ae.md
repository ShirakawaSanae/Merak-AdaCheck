# MoE Snapshot Artifact

## Full-Model Admission Update (2026-10-28)

`runs/moe-capacity/20261028-p2-ds-fixed` passed full DeepSeek-V2-Lite
fixed-policy source/fresh-restore step-2 bitwise comparison on all16 ranks,
including model, optimizer, scheduler and runtime. Node8's checkpoint service
was stopped before recovery from node17. Source/restore COMPLETE and
recovery-oracle.json establish the result; analysis.json is generated with
`tools/analyze_moe_snapshot.py`.

Remote total payload:267682992393 bytes; restore data reads:267681521756 bytes
in16 bundles. Maximum source step walls:1613.204s and23.815s; checkpoint
exposed stall:1584.701s. Restore controller:1407.467s excluding process startup;
restored next-step wall:30.727s. This single two-step admission run does not
establish sustained cadence, repeated timing gains, or an end-to-end RTO.
Matched Host/M1 and Qwen runs are still pending. The earlier p1 full DeepSeek
run failed the oracle and must not enter successful results.

The runtime now uses fixed-order top-k dispatch reduction after a CUDA
nondeterminism reproduction. Older tiny results below predate this change.

## Tiny Gate And Prior Evidence

Current hardware scope: node17 (`172.20.17.2`) and node8 (`172.20.8.2`),
8 A40 GPUs each, 16 ranks. No 32-rank result is claimed.

The current executable gate uses tiny Qwen2-MoE or DeepSeek-V2-Lite with
PP4/TP1/EP2/replica-DP2, BF16, role-aware ZeRO-1, random initialization,
three optimizer steps, and a complete snapshot at step 2. Each policy runs
source training, uninterrupted reference, and fresh-process restore through
step 3. This gate establishes correctness and byte accounting; full-model
performance evaluation remains outstanding.

## Reproduce

From the repository root on node17, with passwordless SSH to node8 and the
shared checkout available at the same path:

```bash
tools/run_in_tmux.sh moe-ae-qwen-fixed runs/moe-ae-qwen-fixed.log -- \
  env CKPT_MOE_MESH_ORDER=pp-dp CKPT_MOE_ADACHECK_GROUPING=1 \
  CKPT_MOE_STOP_REMOTE_SERVICE_BEFORE_RESTORE=1 \
  tools/run_two_node_moe_zero1_smoke.sh ae-qwen-fixed adacheck_fixed qwen2_moe_14b
```

Use a fresh run/session name for every invocation. Run policies sequentially:
`adacheck_fixed`, `host_global_fixed`, `m1`. Models are `qwen2_moe_14b`
and `deepseek_v2_lite`; these names select the model family, while this smoke
runner explicitly constructs its tiny variant. The Python environment is
`/opt/conda/envs/sjw_ckpt_grouping`.

`CKPT_MOE_MESH_ORDER=pp-dp` selects the primary candidate. `dp-pp` is the
negative control and the runner's default; always pass the layout explicitly.
The service-loss switch stops node8's checkpoint service before restoring all
16 training ranks from node17's service.

## Inspect

Each run lives under `runs/two-node-moe-zero1/<run-id>/`. `COMPLETE` is written
after phase exit checks, 16-rank full-state bitwise comparison, alias checks,
and inventory/byte reconciliation. Important evidence includes:

- `source/metadata.json`: actual model dimensions, initialization and rank map.
- `locator.json`: sealed epoch, aliases and committed manifest evidence.
- `source/trainer-output/zero1-object-metrics.jsonl`: transfer counters and
  committed per-host/per-publisher fragment and runtime inventory.
- `recovery-oracle.json`: model, optimizer, scheduler and runtime comparison.
- `logs/`, `phase-status.txt`: raw process outputs and exit status.

To independently recompute remote-only role totals and publisher loads:

```bash
/opt/conda/envs/sjw_ckpt_grouping/bin/python tools/analyze_moe_snapshot.py \
  runs/two-node-moe-zero1/ae-qwen-fixed --output runs/ae-qwen-fixed-summary.json
```

The analyzer checks bundle identity against the sealed locator and reconciles
inventory logical bytes and serialized payload against publication counters.
It includes actual topology and model config in the output so distinct
layouts and model sizes cannot silently share a result label.

## Measurement Boundaries

Inventory export reads committed bundles back during publication. Its time
is included in these smoke timings; do not use them for performance claims.
Logical bytes count tensor contents; serialized payload also includes opaque
non-tensor runtime values and serialization metadata. Wire counters describe
the existing PUT/commit metric scope, not total NIC traffic or inventory GETs.
The historical inventory field `role_logical_bytes_per_host` sums both stored
copies; use the analyzer's `remote_role_logical_bytes` for cross-host claims.

DP-first Qwen runs `20261027-inv-fix2`, `20261027-inv-host`, and
`20261027-inv-m1` all restored bitwise. Fixed/Host remote logical bytes were
249,152 each; M1 was 262,336. This is a negative control: the current M1
publisher selection can increase remote traffic. PP-first results must be
reported separately. All results here use synchronous full snapshots, not
the dense-model gradient-suffix encoder.

The local-publisher correction was subsequently validated in
`20261027-dp-m1-local-i1`: M1 now prefers holders on the destination host,
then balances load among eligible holders. Its remote logical bytes returned
to 249,152 B (model 1,152; optimizer 166,592; runtime 81,408), with bitwise
survivor recovery. The earlier DP-first result above documents the pre-fix
behavior, not the current policy.

The explicit PP-first Qwen tiny runs `20261027-pp-qwen-{fixed,host,m1}-i1`
all passed the committed inventory and 16-rank bitwise survivor-restore gates:

| Policy | Remote model B | Remote optimizer B | Remote runtime B | Total payload B | Max publisher payload B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed | 86,400 | 166,592 | 81,408 | 806,318 | 73,141 |
| Host | 28,608 | 166,592 | 81,408 | 637,230 | 73,125 |
| M1 | 28,608 | 166,592 | 81,408 | 637,313 | 63,098 |

Total payload includes control; maximum publisher payload here covers remote
data bundles only. These single tiny runs show a 17.28% logical-byte reduction
from fixed to Host/M1, without a timing claim.

DeepSeek tiny runs `20261027-pp-ds-{fixed,host,m1}-i1` passed the same gates:

| Policy | Remote model B | Remote optimizer B | Remote runtime B | Total payload B | Max publisher payload B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed | 80,896 | 158,304 | 81,408 | 739,640 | 65,776 |
| Host | 27,328 | 158,304 | 81,408 | 601,554 | 65,762 |
| M1 | 27,328 | 158,304 | 81,408 | 601,613 | 54,345 |

DeepSeek complete-state logical bytes fell 16.71%. The six-run campaign
finished with exit code 0; its log is
`runs/two-node-moe-zero1/20261027-pp-inventory-campaign.log`.
The independently reconciled six-run summary is
`runs/two-node-moe-zero1/20261027-pp-inventory-summary.json`.

## Full-Model Training Capacity

The complete Qwen14.3B and DeepSeek15.7B architectures also completed one
training step on16 A40s with PP4/EP2/DP2, BF16, seq128 and GA8. All16 ranks
recorded step1 and both nodes exited0 for each run. Artifacts are
`runs/moe-capacity/20261027-full-{qwen,ds}-c1`; summary is
`runs/moe-capacity/20261027-capacity-summary.json`.

| Full Architecture | Maximum Rank-Step Wall (s) | Maximum Sampled GPU Memory (MiB) |
| --- | ---: | ---: |
| Qwen2-MoE | 25.564 | 37,628 |
| DeepSeek-V2-Lite | 28.473 | 41,838 |

These cold-step capacity measurements do not include checkpoint publication
or recovery and are not repeated performance results. Reproduce sequentially:

```bash
tools/run_in_tmux.sh full-qwen-cap runs/full-qwen-cap.log -- \
  bash tools/run_two_node_moe_capacity.sh full-qwen-cap qwen2_moe_14b
```

Use a fresh ID and `deepseek_v2_lite` for the second model. The launcher
records commands, model/topology metadata, per-rank step times and periodic
GPU-memory samples, and rejects occupied GPUs before starting.
