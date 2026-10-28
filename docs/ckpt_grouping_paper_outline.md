# Paper Outline: From Logical Redundancy to Physical Protection

This file is the preregistered decision tree. The concise Chinese narrative is
in [`ckpt_grouping_paper_draft.md`](ckpt_grouping_paper_draft.md), the AdaCheck
audit is in [`adacheck_breakdown.md`](adacheck_breakdown.md), and the work still
required for a systems claim is in
[`ckpt_grouping_research_gaps.md`](ckpt_grouping_research_gaps.md).

## Working thesis

AdaCheck answers which training states are logically redundant. Our work asks
two subsequent and separable questions. Under a fixed parallel placement, M1
groups and places canonical checkpoint objects across physical failure domains
while proving recovery closure. M2 then chooses among valid parallel placements
and physical mappings by calling the unchanged M1 as its inner protection
policy. The intended result is not merely fewer logical bytes, but lower
checkpoint critical-path cost under the same recovery contract and a bounded
training-only slowdown.

The corrected r225/r229 suites satisfy the M1 condition with nonzero,
recovery-valid runtime treatments on two dense models. M2 remains conditional:
it needs a four-host comparison against the strongest sequential baseline,
`training-only-best placement + best post-hoc M1`. More hosts or a more complex
optimizer are not contributions by themselves.

## Module boundary

```text
AdaCheck-style detector and active encoder
                 |
                 v
M1: object-aware checkpoint grouping for fixed placement P
                 |
                 | protection cost Phi(P) + recovery proof
                 v
M2: parallel-placement and physical-failure-domain co-design
                 |
                 v
shared transport, atomic commit and recovery substrate
```

M1 owns checkpoint groups, publishers, target domains, object dependencies and
their schedule. M2 owns only selection and physical mapping of a validated
parallel candidate; it regenerates the catalog/profile and invokes the same M1.
Detector reduction is preprocessing and is not counted again as either module's
gain.

### Current grouping, recovery and optimization contract

- The opening motivation compares four rank-grouping policies on frozen object
  catalogs: AdaCheckStyle, GEMINI adjacent pairs, RandomFeasible and
  RedundancyMax. It only motivates that peer choice changes usable existing
  redundancy; it is not recovery-valid end-to-end evidence.
- Formal end-to-end attribution uses AdaCheckStyleFixed, HostGlobalFixed and
  M1Object on one public runtime. Ada is the primary baseline. Host has the same
  object-byte contract as M1 but deterministic first-eligible publishers, so
  Host-to-M1 isolates publisher scheduling. Literal GEMINI adjacency may be
  same-host under node-major placement, and Random is a variance control; they
  are not silently substituted for a host-loss-valid primary baseline.
- ObjectRecovery identifies state by FQN, global slice, PP coordinate and TP
  coordinate, never by transient global rank. DP equivalence is admitted only
  after the training allreduce invariant is confirmed by one checksum verifier
  per complete DP set. A deduplicated base plus versioned suffixes is sealed to
  two independent service domains. After one service loss, a fresh process
  assembles every required object and restores model, optimizer, scheduler,
  RNG, sampler and iteration state before continuing training.
- M1 holds parallel placement `P` fixed. It selects one valid publisher per
  canonical object, publishes the same staged snapshot to both service domains
  and minimizes bytes and bottleneck publisher load subject to recovery closure.
  It exploits existing redundancy and never creates redundancy. M2 alone may
  change the parallel plan or logical-rank-to-host mapping and invokes the same
  M1 as its inner protection optimizer.

## Evidence already established

- The original artifact implements hash/ring redundancy detection and local
  checkpoint writes, but does not connect its remote-buffer helper to an
  end-to-end remote commit/recovery path; its AdaCheck loader is empty.
- A clean eight-A40 GPT-1.4B artifact run produced a `3.6716x` physical-file
  reduction, but the smaller file is not independently recoverable and the
  reachable saver is synchronous. It is an artifact-reachability result, not a
  recoverable performance baseline.
- The current extension captures and fully confirms AdaCheck detector evidence,
  commits peer-local base/delta objects to independent services and validates
  survivor reconstruction.
- The existing host-scope snapshot plan already pools canonical objects per
  host and reconstructs all ranks after exclusion of each of four logical
  loopback services. Its online incremental materializer remains bound to a
  whole-rank mirror-DP peer.
- Frozen-catalog peer selection reduced measured BERT protocol bytes by about
  23% and GPT-2 by about 71%; completion maximum/P99 estimates improved by about
  11% and 31%, respectively.
- All four two-host world-scope catalogs passed the 5% offline opportunity gate.
- BERT paired recovery passed a +5% restore-time non-inferiority margin.
- Historical r199/r211 dense suites first established a recovery-valid transfer
  effect, but predate the final TP-local object boundary and single-publisher
  runtime. They remain audit evidence, not the headline.
- Final corrected r225/r229 suites use one detector, encoder, object runtime,
  transport, capacity and node8-service-loss recovery contract per model. On
  16-rank LLaMA-8B and GPT-6.33B, M1 reduces suffix application bytes
  `49.999/74.999%`, common-base-inclusive application bytes `18.182/40.000%`,
  suffix maximum PUT `44.953/68.673%`, maximum distributed commit
  `45.549/68.769%`, final drain `43.588/68.468%` and observed source-step mean
  `12.021/17.304%` versus AdaCheckStyleFixed. Every checkpoint arm has zero
  loss error and exact 16-rank final full-state digests after fresh recovery.
- M1 reduces planned maximum publisher load `49.965/74.754%` versus
  HostGlobalFixed. Host-to-M1 timings remain mixed in both one-run suites, so
  this proves the load objective but not stable load-balancing speedup. M1
  source-step mean remains `4.9/11.8%` above matching NoCheckpoint; lag two and
  final drain remain. The evidence supports a recovery-valid Ada-to-M1
  transfer/tail and observed foreground claim, not sustained non-blocking,
  strict 1C1S or statistical significance.
- The common-runtime calibration replaces per-holder duplicate gradient
  hashing with fail-closed post-allreduce DP-invariant confirmation and sparse
  pinned staging. On 16-rank GPT-2 `PP2/TP1/DP8`, clean M1/Ada runs retain exact
  recovery while M1 lowers suffix application `87.475%`, max PUT `85.458%`,
  max commit `79.440%`, drain `80.252%` and source-step mean `26.270%` versus
  Ada. M1 remains `12.968%` slower than the first-two-step NoCheckpoint mean,
  so this remains a mechanism calibration rather than a non-blocking claim.
- The first corrected LLaMA attempt rejected a live gradient mismatch between
  different TP coordinates. The final object identity is therefore
  `(FQN, global slice, PP coordinate, TP coordinate)`, with DP equality derived
  only inside that coordinate after allreduce. M1 may reuse one selected
  publisher snapshot for both failure-domain services; the two durable copies
  remain distinct. This failed run is correctness evidence for the object
  boundary, not a performance sample; r225/r229 are the admitted reruns.

These results establish an online, recovery-valid two-host mechanism, not a
physical compute-host power-off result, changed-layout GPU recovery, sustained
non-blocking or a production P99 claim. Each corrected model/arm has one
ordered run, so timings are descriptive.

## Preregistered research questions

1. Under a fixed parallel placement, do at least two recovery-feasible object
   groupings have different held-out active or service costs?
2. Can M1 convert that difference into lower application/NIC traffic or exposed
   checkpoint stall while preserving atomic full-state recovery?
3. Which changes in PP/TP/DP/EP placement alter canonical object holders,
   usable compute slack and physical-domain protection cost?
4. Does joint M2 selection improve over the fastest training-only placement
   followed by its best post-hoc M1 plan?
5. Do all improvements survive full accounting, independent repetitions,
   declared host/service faults and fresh-process numerical continuation?

## Candidate idea decision tree

- If M1 improves a recovery-valid runtime endpoint under fixed `P`, retain it as
  an independent contribution: **recovery-constrained object-aware checkpoint
  grouping**.
- If M2 additionally dominates `training-only-best + M1`, retain the second
  contribution: **parallel-protection and physical-failure-domain co-design**.
- If M1 has zero feasible opportunity but M2 creates one, present M1 as the
  protection substrate and M2 as the main idea; do not manufacture an M1
  speedup.
- If placement candidates have indistinguishable protection cost, stop M2 and
  report that sequential placement is sufficient in the tested regime.
- If only a frozen/offline byte proxy improves, report a negative systems result
  and do not promote either planner as an achieved contribution.

## System model

- Shared input: confirmed canonical-object catalog, active encoded bytes,
  object requirements and dependency chains, a validated parallel-plan
  snapshot, rank-to-device/host mapping, physical failure topology, service and
  link capacities, and held-out compute/network slack.
- M1 decision: host/service groups plus one valid publisher, target domain and
  schedule for every protected object. Whole-rank peer matching is a baseline
  and special case, not the recovery authority.
- M2 decision: one candidate parallel plan and its logical-rank-to-physical-
  domain mapping. Every candidate calls the same M1 and binds the resulting
  catalog, placement and protection hashes.
- Hard constraints: complete same-epoch dependency closure after every declared
  fault, atomic visibility, checksum-valid object identity, memory/link/service
  capacity, bounded lag and a fixed recovery-point/recovery-time contract.
- Objective: lexicographically minimize exposed checkpoint tail, bottleneck
  resource load, active cross-domain bytes, recovery tail and plan churn after
  feasibility. M2 additionally applies a preregistered training-only slowdown
  cap. Logical bytes alone are explanatory, not the primary endpoint.
- Observability plane: an independent monotonic-interval module records compute,
  capture, poll, drain and completed transport edges without synchronizing CUDA
  or participating in detector/encoder/grouping decisions. It is evaluation
  infrastructure unless a separately ablated slack-aware scheduler consumes
  the profile and improves the exposed checkpoint tail.

## Expected paper structure

1. Motivation: logical redundancy does not determine recoverable physical
   protection or the training placement that exposes it.
2. AdaCheck background, clean artifact breakdown and common baseline substrate.
3. Canonical-object, dependency, service and physical failure-domain model.
4. M1: fixed-placement recovery-constrained object grouping.
5. M1 correctness, snapshot implementation and online versioned-object path.
6. M2: finite-candidate parallel-placement and failure-domain co-design.
7. Common transport, atomic commit, recovery and full cost accounting.
8. Two-host correctness/rate evaluation and four-host M1/M2 methodology.
9. End-to-end results, sequential/joint baselines and module ablations.
10. Negative results, artifact boundary and limitations.

## Required figures and tables

- Paper-to-code-to-extension mechanism table.
- Host supergraph with rank-level bipartite matchings.
- Peer-local byte and host-pooled byte/load opportunity decomposition by model/layout.
- Predicted logical versus application/service/PMA bytes.
- Paired completion distributions with trial-level uncertainty.
- Four controlled failure-domain-loss simulations and recovery time.
- Model/layout generalization and unrestricted-oracle gap.
- Fault matrix and explicit claim-boundary table.

Figures are generated as both PNG and SVG under `docs/figures/`, with
machine-readable source JSON and estimator/sample-count annotations. Their
metric meanings and permitted uses are indexed in `docs/figures/README.md`.

## Claim boundaries

- Call the current system an AdaCheck-semantic detector capture plus an
  independent protection-placement extension.
- Do not call service exclusion a physical power-off experiment.
- Do not infer P99 from a static makespan proxy or six measured checkpoints.
- Do not claim `F_host=2` with two replicas.
- Do not claim AdaCheck online-gradient performance without reproducing that
  path.
- Do not describe the existing full reduced-snapshot object path as online
  incremental materialization.
- Do not count detector reduction again as an M1 or M2 benefit.
- Do not claim M2 improves placement unless it beats the strongest sequential
  `training-only-best + best post-hoc M1` baseline under the same contract.
- Do not choose policies, models or endpoints after seeing formal results.
- Do not use NoCheckpoint as the direct baseline for a checkpoint-system
  improvement. It is the training-only floor. Every M1 claim must compare with
  `AdaCheckStyleFixed` on the same detector, encoder, object runtime, transport,
  service budget, cadence, commit and recovery contract. For total training
  time and exposed stall, show all three deltas: AdaCheck versus NoCheckpoint,
  M1 versus NoCheckpoint, and M1 versus AdaCheck.

## Update protocol

This version is the breakdown-stage preregistration. After four-host results,
replace the conditional thesis with exactly one evidence-supported idea, add
effect sizes and confidence intervals, and retain failed hypotheses as negative
results rather than deleting them.
