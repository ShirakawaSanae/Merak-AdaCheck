# 09-12 SemanticFixed experiment report

## Executive conclusion

September 12 completed the two requested 16-rank dense-model common-runtime
recovery gates on node17/node8. GPT-6.33B and LLaMA-3-8B can both reconstruct a
committed epoch in a fresh 16-rank job and continue for five numerically valid
steps. This is necessary baseline infrastructure, but it does not yet prove the
M1 innovation end to end. Both treatments used fixed identity DP peers in a
symmetric `PP4/TP2/DP2` layout, so no optimized grouping policy was executed and
each rank had only one whole-rank recovery-compatible peer.

The direct research baseline is AdaCheck-style fixed grouping on the same
recoverable runtime, not NoCheckpoint. NoCheckpoint is retained only to measure
absolute training overhead. Future claims must compare M1 directly with
AdaCheckFixed for total checkpoint transfer, checkpoint frequency, bottleneck
transfer/commit time, total training wall, exposed training stall and the same
recovery oracle.

## Comparison contract

Every formal model/layout uses three minimum arms:

| Arm | Purpose | Valid comparisons |
| --- | --- | --- |
| `NoCheckpoint` | Training-only cost floor | Absolute checkpoint overhead only |
| `AdaCheckStyleFixed` | Direct recoverable checkpoint baseline | Detector/encoder benefit and fixed checkpoint cost |
| `M1Object` | Proposed recovery-constrained object grouping | Direct fixed-to-M1 benefit |

All checkpoint arms must share model/input/seed, effective batch, parallel and
physical placement, detector output, active encoder and object representation,
requested checkpoint epochs, failure budget, service capacity, transport,
atomic commit and recovery oracle. Report:

1. `AdaCheckFixed - NoCheckpoint` and `M1 - NoCheckpoint` for absolute training
   overhead;
2. `M1 - AdaCheckFixed` for total 16/32-rank transfer bytes, application/wire/NIC
   bytes, observed checkpoint frequency, checkpoint critical-path `max_time`,
   total training wall and exposed stall;
3. identical recovery/failure outcomes for fixed and M1, rather than buying
   performance by weakening recovery or committing fewer checkpoints.

Detector reduction and M1 reduction are separate stages:

```text
full logical state
  -> AdaCheck detector/active encoder -> AdaCheckStyleFixed bytes
  -> M1 destination-aware placement  -> M1 residual bytes
```

Only the second difference is credited to M1.

## Current whole-rank peer constraint

The online `peer_delta` controller currently restores failed rank `r` by taking
one peer's complete model/optimizer operation stream and combining it with
`r`'s exact runtime state. For that algorithm, the peer must have:

- the same PP coordinate, hence the same pipeline-stage parameters;
- the same TP coordinate, hence compatible tensor shards;
- a different DP coordinate, hence a replica rather than the same logical
  training role;
- the same optimizer parameter IDs/group ordering;
- a different physical host for the declared one-host failure contract.

These are hard constraints of the current whole-rank implementation. The
controller checks them before checkpointing. Removing the checks without
changing recovery would substitute another PP stage or tensor shard for the
failed rank, leave missing/wrong parameter and Adam moment entries, or lose both
copies in one host failure. A rank-only fault contract would not intrinsically
require different hosts, but the current experiment contract is host-F1.

They are not hard constraints of the intended object-level M1. Cross-PP/TP
group membership is valid only when an object plan explicitly transfers every
missing versioned object, proves same-epoch dependency closure for every
failure set, reconstructs the failed rank from multiple sources, and passes the
full continuation oracle. Until that multi-source path exists, cross-stage
frozen pairs are motivation evidence only.

## Experiments

### R104: GPT fixed-peer negative cadence gate

- Model/config: random-init GPT dense 6.33B, 16 A40 ranks, BF16
  `PP4/TP2/DP2`, `dp-pp`, seq1024, GA80, scalar AdamW, constant LR, seed
  20260908; frozen detector, SemanticFixed BF16-gradient encoder, fixed
  `(0,8)..(7,15)` DP peers.
- Training reached 10/10 optimizer steps and global controls 0--10; all 160
  update manifests and 16/16 materialized numerical checks exist.
- Only 6/10 checkpoints were visible during training. Rank/global maximum lag
  was 4/4 against configured lag two. The coupled gate stopped before restore,
  so this is not a recovery result.
- Cross-host runtime-only logical/application/wire totals were
  `814,080/2,372,072/2,750,010 B`; epoch bottleneck edge mean/max was
  `113.523/373.408 ms`. Local p/o recovery-operation logical work was
  `269,959,982,080 B`.
- Final drain was about `2726.014 s`. Sequential full-state validation and the
  final barrier dominated the tail.

### R105/R106: corrected GPT reference and 10+5 recovery

- R105 is the current-source 15-step NoCheckpoint reference. It completed with
  16 digests. Steps 2--15 mean/P95/max were
  `12.755/12.782/12.788 s`; training-window/full-launcher wall was
  `192.472/255.034 s`.
- R106 used the identical source and workload with fixed SemanticFixed peers,
  two materialized readers, two response workers and four CPU threads. Train
  and fresh restore launchers exited 0/0; global epochs 0--10, all 160 updates
  and all required 16-rank digest sets are complete.
- Materialized model/optimizer state and fresh steps 11--15 pass the
  preregistered numerical oracle; continuation losses equal R105 bit for bit.
- Recovery passes but cadence fails: 6/10 visible checkpoints, 1.667
  steps/checkpoint and rank/global maximum lag 4/4.
- Steady mean/P95/P99/max was `13.294/13.967/13.983/13.987 s`, a single ordered
  `+4.22%` mean difference from R105. This is descriptive, not a replicated
  non-blocking claim.
- Cross-host runtime-only logical/application/wire was
  `814,080/2,370,792/2,748,739 B`; edge mean/max was
  `340.429/1133.870 ms`. The separate local p/o operation stream was
  `269,959,982,080 B`.
- Train/restore launcher wall was `2172.851/780.096 s`; finalization max was
  `1494.986 s` and restore total was `651.650 s`. Reader concurrency reduced
  R104's tail but did not make it small.

### R107/R108: LLaMA detector preparation

- The first malformed detector command was stopped on both hosts and its small
  diagnostic moved to `/tmp/merak-aborted-detector-r107-20260912/`. R107b then
  correctly failed preflight while stale remote processes were removed. Neither
  is an experiment result.
- R108 completed two real world-scope detector observations for LLaMA-3-8B,
  BF16 `PP4/TP2/DP2`, `dp-pp`. It froze 2,653 stable components and 5,348 stable
  records under catalog hash
  `b33b02b4b7be9086a8a88489735c07861eb61bb700b66e2b41bd06af35bb9aa8`.
- Full/reduced logical metadata was
  `290,598,506,544/145,297,178,116 B`. No recoverable payload exists in this
  detector artifact, so those values are not transfer measurements.

### R109/R110: LLaMA reference and 10+5 recovery

- R109 is the fresh 15-step NoCheckpoint reference: LLaMA-3-8B, 16 ranks, BF16
  `PP4/TP2/DP2`, `dp-pp`, seq1024, GA96, scalar AdamW, constant LR and seed
  20260908. Steps 2--15 mean/P95/P99/max were
  `21.512/21.540/21.560/21.565 s`; training-window/full-launcher wall was
  `323.792/409.769 s`.
- R110 used the same source/workload, R108 detector and fixed SemanticFixed DP
  peers. Train and restore launchers exited 0/0; controls 0--10, all updates and
  all required rank sets are complete. Materialized state and steps 11--15 pass
  numerical recovery at `rtol=atol=1e-4`; maximum loss absolute/relative error
  was `0.0002/2.6692e-5`. State/losses are not bitwise equal.
- It passes the configured bounded-lag-two contract but not strict 1S1C: 8/10
  visible checkpoints, 1.25 steps/checkpoint and rank/global maximum lag 1/2.
- Steady mean/P95/P99/max was `22.100/22.763/22.783/22.788 s`, a single ordered
  `+2.735%` mean difference from R109.
- Cross-host runtime-only logical/application/wire was
  `814,080/2,371,432/2,749,370 B`; edge mean/max was `23.456/49.621 ms`.
  The separate local p/o operation stream was `363,248,839,680 B`, with
  `20.728 s` maximum shared-path transport.
- Train/restore launcher wall was `2946.320/1000.154 s`; finalization and
  restore-total maxima were `2105.938/808.645 s`. Materialized-reader wait,
  fetch/decode, numerical comparison and barriers dominate this tail.

## Engineering completed on 09-12

- Recovery validity and cadence validity are separate. Recovery always fails
  closed; a relaxed cadence gate can proceed to restore while retaining an
  honest cadence failure.
- The recovery oracle requires all rank sets 0--15, exactly the expected five
  continuation steps, finite losses and explicit BF16 tolerances.
- Materialized readers use a bounded one/two-slot per-node lock. Communication
  profiling separates publisher drain, global seal, reader wait, fetch/decode,
  numerical comparison, validation barrier, restore/load and restore barrier.
- Training-window wall and complete train/restore launcher phase wall are
  separately reported.
- The composite figure and machine-readable data are under
  `docs/figures/semanticfixed_dense_16r_composite.*`.

## Unresolved problems

1. No 09-12 run executes random-feasible, redundancy-max, profile-exact or an
   object-level M1 treatment. The runner forces sequential/identity pairing.
2. Symmetric DP2 offers one recovery-compatible whole-rank peer, so fixed is a
   structural zero-opportunity control.
3. The early positive frozen policies use destination-aware snapshot set
   difference, often across PP stages. The live materializer still consumes one
   mirror DP peer's full p/o stream plus failed-rank runtime state.
4. Remote runtime-state bytes and their edge time are not total checkpoint
   transfer or the complete checkpoint critical path. Per-epoch NIC counters
   are absent.
5. GPT cadence fails; LLaMA only reaches bounded lag two. Both have very large
   post-training recovery-readiness tails.
6. Performance evidence is one ordered trial per model. There are no confidence
   intervals or execution-order rotation.
7. Recovery is a healthy-service worker restart, not the four-case rank/host by
   unchanged/changed-plan fault matrix.
8. No common-runtime AdaCheckFixed-versus-M1 result exists; therefore no M1
   transfer, max-time, checkpoint-frequency, total-training-time or
   non-blocking benefit has been established.

## Immediate priority: end-to-end M1

1. Freeze one common object runtime for both `AdaCheckStyleFixed` and M1. Pair
   reordering alone is insufficient because destination-independent rank
   records sum to the same total.
2. Promote snapshot object assignments to versioned online updates: canonical
   value ID, source, publisher, destination, epoch/version, base dependency,
   direct fallback and planned failure sets must be committed together.
3. Extend the materializer from one-peer full-state copying to multi-source
   object assembly. Reconstruct complete model, optimizer, scheduler, RNG,
   sampler and progress for every failed rank.
4. Run CPU closure for all four policies and cross-stage groups. Missing,
   stale, mixed-epoch or under-covered plans must fail closed.
5. Use at least one held-out detector/activity step to build costs and another
   to validate a nonzero fixed-to-M1 opportunity. Do not select the layout from
   its treatment result.
6. Only then launch a 16-rank three-arm 10+5 gate and compare all required
   metrics against AdaCheckFixed. A positive result authorizes replicated
   20-step work; otherwise revise the object/layout design before spending on
   50/200 steps or 32 ranks.

This ordering makes the next GPU experiment test the innovation rather than
repeat a zero-choice baseline.
