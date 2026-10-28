# Checkpoint grouping experiments

`train_toy.py` is a single-host 8-GPU functional and transport harness for the
AdaCheck checkpoint grouping path. It runs a real DDP training loop and peer
copies a versioned, rank-local reduced-state shard after each checkpoint. The
script exercises `RcharaCheckpointSaver`'s group-construction function but
uses an NCCL peer-copy data plane because the artifact constructs groups without
calling its unused CPU/Gloo transfer helper.

Run with `torchrun --standalone --nproc-per-node=8`. Use `sequential` and
`strided` as the two rank-scope baselines. Results apply only to intra-host
process/GPU placement and cannot establish host-failure recovery.

## M1 remote checkpoint correctness

`remote_checkpoint_harness.py` validates a complete fixed-pair checkpoint state:
model, optimizer, scheduler, Python/NumPy/Torch/CUDA RNG, and rank-local
metadata. It publishes prepared and committed source manifests plus a committed
holder manifest, deletes the local CPU replica, restores from the peer, and
requires an exact continued-training loss trace and parameter checksum.

```bash
PYTHONPATH=.:process_monitor torchrun --standalone --nproc-per-node=8 \
  examples/checkpoint_grouping/remote_checkpoint_harness.py \
  --policy sequential \
  --output-dir runs/m1-remote-correctness/sequential-8gpu
```

The real Rchara path is opt-in so ordinary and legacy AdaCheck baselines remain
unchanged. Set `CKPT_REMOTE_ENABLE=1`, `CKPT_REMOTE_MANIFEST_DIR`, and
`CKPT_REMOTE_STORE_MIB`; use the grouping environment described in
`Merak/core/checkpoint/grouping.py`. The default `gloo` backend uses a bounded in-process CPU replica store and is
limited to same-job validation. The `service` backend survives training-worker
termination; it does not by itself survive failure of the host running the service.
See AGENT.md sections 18 and 19.

## M1 worker-restart recovery

Launch one replica service per physical holder host, outside `torchrun`:

```bash
PYTHONPATH=.:process_monitor python \
  examples/checkpoint_grouping/run_replica_service.py \
  --bind 0.0.0.0 --port 29590 --max-store-mib 4096
```

Select it with `CKPT_REMOTE_BACKEND=service`. For a single-host worker-failure
experiment, set `CKPT_REPLICA_ENDPOINT=127.0.0.1:29590`. For multiple hosts, set
`CKPT_REPLICA_ENDPOINTS` to a JSON list indexed by physical host, for example
`["node0:29590", "node1:29590"]`; every source rank writes to the service on
its paired rank's host. `CKPT_REMOTE_RESTORE_EPOCH=N` makes a fresh
`MerakTrainer.train()` restore the committed full state before resuming its
step and sampler counters.

`service_recovery_harness.py` is the destructive worker-failure correctness
test. Its `save-and-fail` phase deliberately sends `SIGKILL` to rank 0 after
all rank references are published. Run the `restore` phase with the same
service and output directory. The final-state oracle is used only for
post-training comparison, never as a recovery source.

## Matched AdaCheck grouping experiment

This path fixes one real `RedundancyChecker` result, then changes only the
two-host protection grouping. The supported synthetic-data workloads are
`bert`, `gpt2`, `opt`, and `llama_1b`. Their local model configs and hashes are
recorded by `run_workload.py`; no pretrained weights or datasets are fetched.

On node17, capture one 16-rank snapshot from node17 and node1 and build its
immutable reduced catalog:

```bash
tmux new-session -d -s matched-capture \
  'tools/run_two_node_matched_capture.sh bert'
```

The builder treats AdaCheck's packed hash locations as candidate buckets.
Role, dtype, shape, and full SHA-256 split those buckets into confirmed tensor
components. It refuses missing paths or reconstruction catalog changes and
records candidate/confirmed/rejected pair counts in `frozen/summary.json`.

Replay the same frozen bundle through fixed, random, and restricted exact
grouping. The default is three repetitions and eight checkpoint epochs
per condition; the first two epochs are analysis warmup. Two worker-independent
services retain the committed shards, and the driver also runs bounded protocol
faults plus recovery from node1 after the node17 service stops.

```bash
tmux new-session -d -s matched-matrix \
  'tools/run_two_node_matched_matrix.sh bert \
   runs/matched-two-node-capture/bert-seed-20260831/frozen/reduced_bundle.pt'
```

After all selected model matrices have a `COMPLETE` marker, aggregate them and
write a local analysis artifact. The legacy figure stems emitted by this command
are reproducible exploration outputs and are intentionally not retained under
`docs/figures`; do not revive them as evidence for the SemanticFixed dense path.

```bash
/opt/conda/envs/sjw_ckpt_grouping/bin/python \
  tools/analyze_matched_grouping.py \
  runs/matched-two-node-matrix/bert-v7-formal-clean \
  --output-json \
    runs/matched-two-node-matrix/bert-v7-formal-clean/analysis.json \
  --docs-dir docs/figures \
  --capture-summary \
    runs/matched-two-node-capture/bert-seed-20260831-v7/frozen/summary.json \
  --capture-summary \
    runs/matched-two-node-capture/gpt2-seed-20260831-v1/frozen/summary.json \
  --capture-summary \
    runs/matched-two-node-capture/opt-seed-20260831-v4/frozen/summary.json \
  --capture-summary \
    runs/matched-two-node-capture/llama_1b-seed-20260831-v1/frozen/summary.json
```

Do not use a wildcard that also selects pilot or diagnostic roots. `analysis.json`
retains every plotted value and its input path.

The LLaMA-like 1B config is a practical pipeline and opportunity-screening
workload, not sufficient scale evidence for a final systems paper.
`llama_3b` is a local 3,015,355,392-parameter config prepared for the next
capture; merely supporting it in the runner is not experimental coverage. A
positive grouping result should be repeated at 3B and, when memory/time allow,
8B. This two-node stage deliberately does not claim physical three-host
failure, automatic failover, chunk backpressure, or 3B/8B coverage.

## Peer-redundancy delta experiment

A frozen catalog is one immutable `RedundancyChecker` result: object IDs,
holders, logical bytes, checksums, reconstruction templates, detector scope,
and catalog hash. Every policy in a matched comparison reads the same frozen
catalog, so the experiment changes checkpoint peer assignment without changing
what AdaCheck detected.

For peer ranks `i` and `j`, the remote delta is the recovery state required by
`i` but absent from `j`. `redundancy_max` solves the two-host perfect matching
that minimizes these deltas. `host_global` lets every rank reuse all objects
already present anywhere on the target host; it is an availability-scope
control, not another rank-pairing heuristic.

The reportable clean BERT formal command was:

```bash
tmux new-window -t ckpt-peer-redundancy-20260901 -n bert-formal \
  'env EPOCHS=8 REPETITIONS=5 \
   POLICIES_CSV=fixed,random,redundancy_max,host_global \
   PROFILE_JSON=runs/peer-delta-profile/bert-world-v2-fixed.json \
   MASTER_PORT_BASE=31300 SERVICE_PORT=31390 \
   MAX_IDLE_NIC_BYTES=10485760 FAILED_SERVICE_HOST_INDEX=0 \
   bash tools/run_two_node_peer_delta_pilot.sh bert \
   runs/matched-two-node-capture/bert-world-seed-20260901-v1/frozen/reduced_bundle.pt \
   runs/peer-delta-formal/bert-world-v2'
```

Each condition writes its expanded command, rank events, service counters, NIC
snapshots, and summary. The root `experiment-config.json` records the model,
parallelism, seeds, cyclic policy order, source/profile hashes, ports, hardware,
software, failure scope, and metric definitions. A result is formal only after
the root has `COMPLETE` and this analyzer accepts all 20 unique cells:

```bash
/opt/conda/envs/sjw_ckpt_grouping/bin/python \
  tools/analyze_peer_delta_formal.py \
  --formal bert=runs/peer-delta-formal/bert-world-v2 \
  --formal gpt2=runs/peer-delta-formal/gpt2-world-v1 \
  --opportunity bert=runs/peer-delta-opportunity/bert-world-v1/summary.json \
  --opportunity gpt2=runs/peer-delta-opportunity/gpt2-world-v1/summary.json \
  --opportunity opt=runs/peer-delta-opportunity/opt-world-v1/summary.json \
  --opportunity llama-3b=runs/peer-delta-opportunity/llama-3b-world-v1/summary.json \
  --warmup-epochs 2 \
  --output-json runs/peer-delta-formal/two-model-analysis.json \
  --docs-dir docs/figures
```

`commit_ms` is recorded per rank. For publication-facing latency, an epoch's
checkpoint completion is the maximum `commit_ms` across all 16 ranks. P99 is
then computed from those epoch completion samples within each trial; means and
95% t intervals use the five repeated trials, not rank events as independent
samples. With six measured epochs per trial, P99 is effectively a near-maximum
tail statistic and must be described with that finite-sample limitation.

The peer analyzer emits legacy exploratory figures with distinct meanings:

- `adacheck_peer_redundancy_heatmap`: reusable frozen state for every
  node17/node1 rank pair; brighter cells offer larger delta savings.
- `adacheck_peer_offline_bytes`: catalog-predicted fixed,
  redundancy-max, and host-global logical bytes before network execution.
- `adacheck_peer_runtime_bytes`: application protocol bytes versus independent
  two-node PMA transmit counters from pilot runs.
- `adacheck_peer_pilot_commit_p99`: descriptive rank-level pilot tail only;
  it is not the formal checkpoint-completion statistic.
- `adacheck_peer_survivor_restore`: time to reconstruct the catalog from the
  surviving service after the other service process is stopped.
- `adacheck_peer_formal_*`: five-trial BERT formal means and 95% t intervals;
  these files are generated only after the formal analyzer accepts the matrix.
- `adacheck_peer_formal_pareto`: formal protocol-byte/P99 tradeoff with 95%
  intervals for both axes.
- `adacheck_peer_multimodel_offline_bytes`: four-model frozen-catalog logical
  byte comparison; OPT/3B bars are opportunity evidence, not runtime results.

An idle PMA probe exceeding 10 MiB in either direction marks that trial's NIC
metric contaminated and excludes only the NIC value. Protocol bytes remain
available and must reconcile exactly with both services' RX/TX counter deltas.
Restore results establish catalog reconstruction after a service-process
failure; they do not establish physical-host failure or continued training.

The paired recovery noninferiority matrix uses one committed checkpoint per
condition and restores it independently from both services. Its exact command
is also retained in `driver-command.txt`:

```bash
env EPOCHS=1 REPETITIONS=5 POLICIES_CSV=fixed,redundancy_max \
  PROFILE_JSON=runs/peer-delta-profile/bert-world-v2-fixed.json \
  MASTER_PORT_BASE=31600 SERVICE_PORT=31690 SERVICE_MAX_MIB=65536 \
  MAX_IDLE_NIC_BYTES=10485760 FAILED_SERVICE_HOST_INDEX=0 \
  RESTORE_EACH_CONDITION=1 RECOVERY_SLOWDOWN_LIMIT_PERCENT=5 \
  bash tools/run_two_node_peer_delta_pilot.sh bert \
  runs/matched-two-node-capture/bert-world-seed-20260901-v1/frozen/reduced_bundle.pt \
  runs/peer-delta-recovery/bert-world-v1

/opt/conda/envs/sjw_ckpt_grouping/bin/python \
  tools/analyze_peer_delta_recovery.py \
  --recovery-root runs/peer-delta-recovery/bert-world-v1 \
  --output-json runs/peer-delta-recovery/bert-world-v1/analysis.json \
  --docs-dir docs/figures
```

The trial metric is the slower of the two service restore times. The paired
test passes only when the upper endpoint of the five-trial 95% t interval for
`redundancy_max / fixed - 1` is at most the pre-registered 5% slowdown margin.
The recovery plot from this analyzer remains catalog fetch, validation, and
template reconstruction, not a continued-training recovery-time objective.

New pilot roots archive the four runtime source files in `source-snapshot/` in
addition to recording hashes. The older BERT/GPT-2 formal roots predate that
change: their core Python hashes are recorded, but their then-untracked launcher
exists only as a hash. They are valid execution artifacts but are not fully
source-reconstructible from the repository commit alone.

## Four-host hierarchical experiment

The four-host contract is `PLAN_FOUR_NODES.md`. Available nodes are
`{1,8,17,20}` and the fixed host-index order is `[17,1,8,20]`. Do not launch a
32-rank job until all four environment and SSH checks pass. Nodes 8 and 20 use
the existing environment archive; the bootstrap refuses to overwrite an
existing non-matching prefix:

```bash
env RUN_ROOT=runs/four-node-env-bootstrap/<immutable-node8-run> \
  bash tools/bootstrap_ckpt_grouping_env.sh 8
env RUN_ROOT=runs/four-node-env-bootstrap/<immutable-node20-run> \
  bash tools/bootstrap_ckpt_grouping_env.sh 20
```

Capture two detector steps and build one immutable 32-rank catalog:

```bash
tmux new-window -t ckpt-four-node-20260902 -n gpt2-4x2x4-capture \
  'env PP=4 TP=2 DP=4 MASTER_PORT=31810 \
   bash tools/run_four_node_matched_capture.sh gpt2 \
   runs/four-node-capture/gpt2-pp4tp2dp4-seed-20260902'
```

The capture driver first requires a separate 32-rank NCCL all-reduce. The
offline analyzer emits executable treatment names and refuses a formal gate
unless detector stability covers two steps. Start with a one-trial smoke; it
also excludes each service artifact in turn and reconstructs every template:

```bash
tmux new-window -t ckpt-four-node-20260902 -n gpt2-smoke \
  'env STAGE=smoke MASTER_PORT_BASE=32300 SERVICE_PORT=32490 \
   bash tools/run_four_node_peer_delta.sh gpt2 \
   runs/four-node-capture/gpt2-pp4tp2dp4-seed-20260902/frozen/reduced_bundle.pt \
   runs/four-node-capture/gpt2-pp4tp2dp4-seed-20260902/hierarchical-opportunity/summary.json \
   runs/four-node-peer/gpt2-pp4tp2dp4-smoke-v1'
```

Use `STAGE=pilot` with the smoke `analysis.json` in `SMOKE_ANALYSIS_JSON` for
exactly five paired trials. Its `analysis.json` lists `confirmatory_policies`;
pilot samples are never merged into the formal set.

```bash
env STAGE=pilot \
  SMOKE_ANALYSIS_JSON=runs/four-node-peer/gpt2-pp4tp2dp4-smoke-v1/analysis.json \
  bash tools/run_four_node_peer_delta.sh gpt2 \
  runs/four-node-capture/gpt2-pp4tp2dp4-seed-20260902/frozen/reduced_bundle.pt \
  runs/four-node-capture/gpt2-pp4tp2dp4-seed-20260902/hierarchical-opportunity/summary.json \
  runs/four-node-peer/gpt2-pp4tp2dp4-pilot-v1
```

Run exactly ten new trials only with that decision artifact:

```bash
env STAGE=formal \
  SMOKE_ANALYSIS_JSON=runs/four-node-peer/gpt2-pp4tp2dp4-smoke-v1/analysis.json \
  PILOT_DECISION_JSON=runs/four-node-peer/gpt2-pp4tp2dp4-pilot-v1/analysis.json \
  bash tools/run_four_node_peer_delta.sh gpt2 \
  runs/four-node-capture/gpt2-pp4tp2dp4-seed-20260902/frozen/reduced_bundle.pt \
  runs/four-node-capture/gpt2-pp4tp2dp4-seed-20260902/hierarchical-opportunity/summary.json \
  runs/four-node-peer/gpt2-pp4tp2dp4-formal-v1
```

Formal completion can generate local traffic and checkpoint-completion figures.
This runtime is a frozen-catalog service replay. A paper-level training-overhead
claim additionally requires integration of the winning plan into the save lifecycle
and a fresh-job checked optimizer step after restore.
