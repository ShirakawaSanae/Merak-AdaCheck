# AdaCheck checkpoint grouping: current research charter

Updated: 2026-09-13

This file is the active research charter. Historical run-by-run narration was
removed on 2026-09-10 because the authoritative details already live in
`TWO_NODE_RUN_LOG.md`, experiment manifests under `runs/`, and the paper notes
under `docs/`. Sections 70 and 71 retain the detailed design and code audit that
led to the current direction; their original numbering is preserved so existing
references remain valid.

When facts conflict, use this order:

1. current worktree code and focused tests;
2. immutable experiment manifests and machine-readable analyses;
3. Sections 1--9 of this charter;
4. Sections 70--71;
5. `PLAN_TWO_NODES_V2.md`, `docs/TWO_NODE_V2_ACTIVE_GOAL.md`, and
   `docs/two_node_v2_experiment_matrix.md`;
6. narrative run logs and older plans.

Do not overwrite or revert the large uncommitted research worktree. The current
highest priority is to convert the recovery-valid SemanticFixed substrate into
a genuine end-to-end M1 comparison. On a frozen model/layout/workload, run an
AdaCheck-style fixed arm and M1 on the same detector, active encoder, object
runtime, transport, service limits, cadence, commit and recovery contract.
`NoCheckpoint` remains a training-overhead control and may not replace the
AdaCheck fixed baseline for checkpoint transfer size, frequency, max time,
total training time, exposed stall or recovery. Before another large GPU run,
the online object plan must pass CPU recovery closure and a held-out active
opportunity gate with at least two legal plans of different cost. The current
GPU scope remains node17/node8 and 16 ranks; 20/50/200-step, 32-rank/four-host,
EP and M2 remain frozen until that gate is reviewed.

## 1. Research objective

Complete a paper-quality checkpointing system that starts from AdaCheck's
confirmed redundancy information and answers the next physical-placement
question:

> Given a parallel placement, active canonical checkpoint objects, a declared
> failure model, and shared service/network resources, how should checkpoint
> objects be grouped, published, and placed so that the checkpoint is fully
> recoverable with minimum critical-path protection cost? Can choosing the
> parallel placement and physical failure-domain mapping jointly expose a
> better solution than optimizing them sequentially?

The immediate engineering goal remains the two-host, 16-GPU V2.1 closure:
stable AdaCheck evidence, complete-state recovery constraints, an online
non-blocking checkpoint path, representative large-model runs, 200-step
sustainability at the fastest honestly supported checkpoint cadence, and
fresh-process recovery validation. Recovery-correct one-step-one-checkpoint
(`1S1C`) is the final cadence target, not a prerequisite for early baseline or
grouping experiments. Two hosts establish the common substrate and negative
boundaries; nontrivial host grouping and the M2 headline require at least four
physical hosts.

The intended paper contribution is not another redundancy detector and not a
generic automatic-parallelization system. AdaCheck supplies the logical
redundancy evidence. This work determines how much of that evidence can be
safely exploited by a recoverable physical protection plan, and whether the
training placement should be selected with that protection cost in view. The
target scope is every validated symmetric or asymmetric 16- or 32-rank
parallel placement for which Merak can export a complete object/holder graph;
the current mirror-DP online implementation is only an intermediate compatible
subset of that target.

## 2. Three innovation points

The paper has three innovation points but only two optimizer modules. M1 and M2
are the module boundary; physical-resource load balancing is a cross-cutting
objective and design contribution used by both.

### 2.1 Recovery-constrained object-level checkpoint grouping

For a fixed parallel placement, replace fixed whole-rank groups with a
proof-carrying object-level protection plan. For each canonical checkpoint
object, M1 chooses a valid publisher, destination service/failure domain, and
dependency-complete recovery route.

This policy does **not** create new redundancy. Model structure, optimizer
semantics, and the parallel plan determine which values are redundant. Grouping
only determines whether existing redundancy can be reused without violating
recovery. The accurate claim is:

> M1 improves utilization of AdaCheck-confirmed redundancy and avoids duplicate
> protection caused by coarse fixed groups, thereby reducing the residual data
> that must be transferred or materialized under the same recovery contract.

The byte reduction is not produced by pair reordering alone. M1 adds a
destination-aware suppression rule: for target service `j`, a publisher sends
only the active objects that are not already valid and available at `j` for the
same checkpoint epoch. Grouping changes which naturally redundant objects are
therefore usable. AdaCheck's detector/reducer is the common upstream input;
`full`/`partial` redundancy and paper `p`/`o` cases are provenance/reporting
labels, not hard constraints on M1's object solver.

The protected catalog includes model parameters, optimizer tensors and scalars,
scheduler, scaler when present, random-number-generator state, sampler state,
and training progress. A plan is invalid if any declared failure set loses a
required object or a complete base/delta dependency chain.

### 2.2 Joint parallel placement and physical failure-domain optimization

M2 searches a finite, preregistered set of validated parallel plans and
logical-rank-to-device/host mappings. For every candidate `P`, it regenerates
the catalog and active profile and calls the unchanged M1:

```text
Phi(P, F) = M1.solve(catalog(P), profile(P), topology, F)
P* = argmin JointCost(training(P), Phi(P, F))
```

M2 owns parallel-plan selection and physical mapping. It never reimplements
grouping, publisher selection, transport, commit, or recovery. The decisive
baseline is the fastest valid training-only placement followed by its best
post-hoc M1 plan, not a weak default placement.

Changing ranks, hosts, or a parallel configuration invalidates the old catalog
and protection-plan hash and triggers replanning. Until change detection,
bounded replan time, state migration, and atomic switchover are implemented and
measured, describe this as offline reconfiguration support, not seamless online
adaptation to sudden topology changes.

### 2.3 Recovery-preserving physical-resource load balancing

Total logical bytes are insufficient. The optimizer and runtime must also
control the busiest physical resources:

- per-rank publisher serialization and D2H work;
- per-host NIC ingress and egress;
- shared links or rails;
- destination CPU deserialize/materialize/apply work;
- queue depth, memory pressure, checkpoint completion tail, and exposed stall.

The primary objective is lexicographic: satisfy recovery, atomic visibility,
capacity, and RPO/RTO constraints first; then minimize checkpoint critical path
or exposed tail; then minimize bottleneck resource load, active cross-domain
bytes, recovery tail, and plan churn. A byte reduction that overloads a shared
NIC or CPU materializer is a failed end-to-end optimization.

## 3. System and correctness boundary

The shared pipeline is:

```text
AdaCheck-style stable detector + active encoder
                       |
                       v
M1(P): object-level recovery-constrained protection plan
                       |
                       v
M2: select/map P by repeatedly calling the same M1
                       |
                       v
bounded transport -> same-epoch atomic commit -> recovery -> continuation
```

Core contracts:

- `RedundancyCatalog`: semantic object identity, role, encoded bytes, natural
  holders, rank requirements, stability evidence, and dependencies.
- `ParallelPlanSnapshot`: PP/TP/DP/EP role, shard ownership, rank/device/host
  mapping, memory, collectives, and training profile.
- `FailureTopology`: real GPU/host/service containment and only those rack/ToR/
  power domains that can be honestly represented.
- `ActiveProfile`: held-out active bytes, wire expansion, usable slack, and
  serialization/materialization service distributions.
- `ProtectionPlan`: group, object publisher/destination, dependency closure,
  schedule, predicted loads, and a recovery proof.

An epoch is recoverable only if every required object remains available as an
exact same-epoch value or complete dependency chain after every planned failure
set. It is visible only when catalog, parallel-plan, protection-plan, encoding,
range, version, and checksum evidence agree. Prepared or partial state fails
closed.

The live Merak incremental path is still more restrictive than this model: it
uses whole-rank mirror-DP peers and one contiguous AdamW chain per source rank.
Controller construction currently rejects a peer unless PP and TP coordinates
match, DP coordinates differ, and optimizer parameter IDs have identical
topology; base validation also rejects a different tensor-manifest digest.
Cross-stage or otherwise heterogeneous pairs therefore fail before useful
checkpointing, and bypassing those checks would make the current peer-base plus
source-delta reconstruction incorrect.

This restriction is an implementation boundary, not the final grouping
definition. `peer_delta_runtime.py` can already describe cross-host object
assignments in a snapshot plan, but online arbitrary-placement support requires
canonical object IDs, per-object publishers and destinations, versioned
base/update dependencies, multi-source target reconstruction, and commit keys
that bind catalog, parallel-plan, grouping-plan, encoding, and epoch hashes. The
snapshot object plane must not be described as online arbitrary-object
materialization until those dependencies are connected to the materializer and
global commit protocol.

## 4. Current evidence and interpretation

Only the following compact evidence summary belongs in this charter. Exact
commands, failures, timings, and per-run boundaries remain in the run log and
manifests.

- The public AdaCheck artifact implements hash/ring redundancy detection and
  local saving, but its remote update/recovery path is incomplete. The clean
  eight-A40 GPT-1.4B run is an artifact-reachability baseline, not a recoverable
  performance baseline. Use `AdaCheckStyleFixed` on the common substrate for
  controlled M1 comparisons.
- Frozen-catalog rank matching reduced BERT and GPT-2 protocol bytes by about
  23% and 71%, with descriptive completion-tail improvements of about 11% and
  31%. OPT and LLaMA-like-3B also showed offline byte opportunities. These
  results validate the frozen pair-local objective only.
- Those early gains disappeared in the current end-to-end layout because the
  legal whole-rank recovery graph has only mirror-DP peers, the active next
  epoch can have zero grouping gap, and local CPU materialization dominates
  after remote bytes become small. This is evidence for the new recovery- and
  service-aware formulation, not evidence that grouping is universally useful.
- In the two-host DP-first `PP4/TP2/DP2` layout, fixed and exact whole-rank
  pairing are structurally identical. With one host-level service on each side
  and `F_host=1`, every under-protected object must cross to the other host, so
  host-deduplicated cross-domain bytes are often invariant. Two hosts can still
  test object correctness, publisher balance, NIC/CPU load, service loss, and
  recovery; they cannot prove nontrivial host grouping.
- The 200-step GA64 checkpoint treatment failed at epoch 16 from bounded-queue
  backpressure, while the matching NoCheckpoint workload completed 200 steps.
  GA80 completed a 20-step strict-lag-one cadence run, showing that compute
  window versus CPU service rate is a governing constraint rather than queue
  size alone. That single short diagnostic lacks the complete recovery and
  paired sustained contract and is not counted as achieved general 1S1C.
- The snapshot code now carries explicit `required_value_ids_by_host` and
  `planned_failure_sets`. Complete model/optimizer/scheduler/RNG/sampler/
  iteration recovery tests and the peer-delta, hierarchical-peer, and matched
  CPU regression set were last reverified as 30 passing tests. This is CPU and
  logical-service evidence, not physical-host runtime evidence.

### 4.1 Four-case fault matrix and reconfiguration gap

Formal fault experiments use only two axes: failed compute scope `{rank, host}`
and recovery plan `{unchanged, changed}`. Protocol corruption, stale epochs,
service-process loss, and logical service exclusion remain prerequisite
fail-closed tests, not additional headline fault categories.

| Case | Injection and available capacity | Required recovery behavior |
| --- | --- | --- |
| rank / unchanged | transiently `SIGKILL` one rank, then restore the same logical rank on the same GPU/role | preserve world size, PP/TP/DP coordinates, catalog hash, and grouping-plan hash; restore the latest committed epoch and continue |
| host / unchanged | terminate all workers and checkpoint service on one host, then restart that same-capacity host stack | preserve the logical parallel plan and group memberships; restore all affected ranks from surviving domains and continue |
| rank / changed | make one GPU/rank placement permanently unavailable while a reserved spare is available | construct a new rank-to-device/failure-domain mapping `P'`, migrate affected state, rerun the unchanged M1 to produce `G'`, reject `P/G`-stale records, and continue |
| host / changed | remove one host and its service for the rest of the run | shrink from 32 to 24 ranks using a validated `P'` such as dense `PP4/TP2/DP3`, migrate/repartition the committed state, compute `G'=M1(P')`, atomically switch hashes, and continue on three hosts |

The changed-rank case can first use 16 active ranks on four hosts so spare GPUs
exist; a full 32-rank same-world replacement would require a fifth/spare host.
The unchanged-host case is a host-stack restart, not proof of uninterrupted
operation while a physical machine remains absent. Both limitations must be
stated rather than simulated away.

Every case injects the fault at a fixed optimizer step while checkpoint traffic
is in flight, fences the old run/epoch, restores the latest globally committed
checkpoint in a fresh distributed job, and compares at least five continued
steps with an uninterrupted oracle. Report success, RPO/lost steps, RTO,
restore/migration bytes, replan time, old-plan rejection, new plan hashes,
maximum transfer-edge time, and post-restore numerical error.

No case currently passes this full contract. Existing evidence covers pieces:
toy rank `SIGKILL` plus fresh restart, healthy-service worker restart, one
stopped checkpoint-service process, static four-domain exclusion, and protocol
fail-closed checks. None changes the training parallel plan or replans grouping.
The incremental path and legacy `CheckpointSync` are also mutually exclusive;
rank pairs, world size, process groups, optimizer topology, catalog hash, and
grouping hash are fixed at controller construction.

A real rank/host loss normally invalidates the NCCL job, so the target is
bounded failure-triggered restart and reconfiguration rather than seamless
in-process adaptation. The recovery manager must fence the old epoch, select
and instantiate `P'`, rebuild canonical requirements/holders, restore and
migrate state, call the unchanged M1, atomically publish `P'/G'` hashes, and
only then resume training.

## 5. Final experimental goals and gates

### 5.1 E0: stable and complete common input

- Confirm AdaCheck candidates at two detector steps using role, dtype, shape,
  range, and full checksum; unstable candidates fall back to unique objects.
- Export complete holders, requirements, active encoded sizes, parallel
  coordinates, physical domains, and dependency chains.
- Separate full logical state, AdaCheck-reduced state, active update state, and
  M1 residual protection bytes. Detector reduction is preprocessing and is not
  counted again as an M1 or M2 gain.

Gate: every rank's complete checkpoint can be reconstructed from the catalog,
and the catalog/plan/profile hashes are reproducible.

### 5.2 E1: M1 snapshot and online recovery correctness

- Finish arbitrary allowed target domains, per-domain required object sets, and
  planned failure-set validation in the snapshot path.
- Connect the same object plan to versioned online updates, the CPU
  materializer, bounded transport, global commit, and fresh-process restore.
- Validate model, optimizer, scheduler, RNG, sampler, and progress at the
  checkpoint boundary and for at least one continued optimizer step.
- Inject missing, stale, corrupt, partial-commit, service-loss, and each declared
  domain-exclusion case. Recovery must fail closed when coverage is incomplete.

Gate: all planned failure sets recover the complete same-epoch state and match
the uninterrupted numerical continuation contract. No grouping result is
eligible before this gate.

### 5.3 E2: M1 opportunity and end-to-end benefit

- Under each fixed `P`, compare `AdaCheckStyleFixed`, whole-rank constrained
  exact, object-static, held-out object-active, recovery-constrained object
  placement, service-aware placement, random feasible plans, and a small exact
  oracle where tractable.
- Use held-out activity and service profiles. The treatment must not choose a
  plan using current-epoch hindsight.
- Report publisher, host NIC, shared-link, and CPU materializer maxima as well
  as application/wire bytes and checkpoint completion/exposed stall.
- Use the same AdaCheck-style detector/reducer and active encoder for fixed and
  optimized arms. Report `full logical -> detector-reduced/encoded fixed`
  separately from `fixed -> M1 residual transfer`; only the latter is a grouping
  gain. M1 must beat the fixed AdaCheck-style baseline in 16- and 32-rank total
  transfer or its measured critical path, not merely produce a smaller unique
  canonical catalog.
- For every checkpoint endpoint, the direct baseline is `AdaCheckStyleFixed`,
  not `NoCheckpoint`. Compare fixed versus M1 for total 16/32-rank checkpoint
  transfer bytes, application/wire/NIC bytes, observed checkpoint frequency,
  checkpoint critical-path `max_time`, total training wall, exposed stall and
  the identical recovery oracle. Also report both checkpoint arms against
  `NoCheckpoint` for absolute training overhead. A claim is incomplete if it
  reports only `M1 versus NoCheckpoint` or only remote runtime-state bytes.

Gate: at least two representative workloads/layouts must have a preregistered
nontrivial recovery-feasible active/service gap, and M1 must reproduce that gap
in actual wire or critical-path measurements without weakening recovery.
Otherwise M1 remains a correctness substrate rather than a performance
headline.

### 5.4 E3: M2 parallel-placement/failure-domain co-design

- Use at least three valid parallel/mapping candidates within a preregistered
  training-only slowdown band.
- Regenerate the catalog/profile and run exactly the same final M1 for every
  candidate.
- Compare joint selection against `training-only-best + best post-hoc M1`,
  `checkpoint-only-best + M1`, default placement, and ablations without the
  correlated-domain or service-rate term.
- Report training collective cost and step time separately so checkpoint gains
  cannot hide a slower training plan.

Gate: joint M2 must dominate the strongest sequential baseline on a declared
Pareto endpoint or useful-job completion model. If both select the same point,
the co-design hypothesis is rejected for that regime.

### 5.5 E4: sustained cadence, non-blocking, and recovery validation

- For each paper-level model, first find the smallest sustainable checkpoint
  interval `N` optimizer steps, starting conservatively and tightening through
  a preregistered sweep such as `N=16,8,4,2,1`. Use bounded host memory/queue,
  no silently dropped update, and a recovery-complete checkpoint at every
  declared cadence boundary.
- Report both the configured `N` and the observed training-window global commit
  rate. Final drain, queued per-step updates, or a larger gradient-accumulation
  window must not be relabeled as a smaller checkpoint interval.
- Report capture rate, commit rate, boundary lag, checkpoints per step and per
  token, exposed stall, throughput, and all resource tails. Call a run strict
  1S1C only when `N=1`, every boundary has `lag <= 1`, and fresh-process
  recovery passes. Optimize toward this final target after a larger `N` closes
  the baseline correctly.
- Inject the declared service/host-stack failure at a fixed step, restore from a
  surviving committed epoch, and compare at least five continued steps with an
  uninterrupted reference.
- After the common restore path passes, execute the four formal cases in
  Section 4.1: rank/unchanged, host/unchanged, rank/changed, and host/changed.
  Protocol faults remain prerequisite checks rather than extra headline cases.
- Use at least three independent paired trials for primary endpoints; use five
  where practical. P99 requires enough independent checkpoint samples and must
  not be inferred from a static proxy or a handful of epochs.

Candidate model coverage is BERT/GPT-2 for baseline continuity, GPT dense 6.33B
and LLaMA-3-8B for large dense experiments, and a sparse/MoE architecture only
when its actual EP/placement semantics are executable. Random initialization
and synthetic tokens are valid for systems behavior but support no model-quality
or convergence claim.

The current worktree contains 16-rank smoke parameter chains for
`llama_3_8b`, `gpt_6b`, `qwen2_moe_14b`, and `deepseek_v2_lite` through
`run_workload.py` and `run_two_node_merak_1c1s.sh`. This is not yet a complete
formal matrix: all three entry files are uncommitted research files, the shell
launcher is hidden by the repository-wide `*.sh` ignore rule, Qwen currently
has no real EP path, and DeepSeek has no distributed preflight artifact. Treat
Qwen only as no-EP compatibility until Section 10.3 passes, and do not advance
DeepSeek beyond preflight until its local implementation is validated on both
hosts.

## 6. Baselines and fair-comparison invariants

All performance arms share model, input, seed, effective global batch, parallel
semantics, detector output, active encoder, fault budget, replica/protection
budget, service capacity, transport, chunking, cadence, warm-up, and fault time.
Only the variable named by the ablation may change.

Required identities:

- `NoCheckpoint`: training-only upper bound, not a reliability baseline.
- clean public AdaCheck artifact: reachability and paper-to-code audit only.
- `AdaCheckStyleFixed`: fixed grouping on the common recoverable substrate.
- `RandomFeasible`: multiple preregistered seeds under identical constraints.
- `WholeRankExact`: special-case/negative-control exact matching.
- `M1Object`: active, recovery-constrained object plan.
- `M1ObjectServiceAware`: M1 plus physical-resource load objective.
- `Oracle`: small-instance optimum or lower bound.
- M2 sequential and joint arms defined in Section 5.4.

Do not compare a host-global availability control directly with peer-local
grouping as if they had the same recovery semantics. Do not compare GA64 and
GA80 as a grouping treatment; that sweep is only a service-rate sensitivity.

The required comparison ladder for every model/layout is:

```text
NoCheckpoint                 -> absolute training-cost floor only
AdaCheckStyleFixed           -> direct checkpoint baseline
M1Object / M1ServiceAware    -> proposed treatments
```

For bytes, checkpoint frequency, checkpoint `max_time` and recovery, report
`M1 - AdaCheckStyleFixed`. For total training time and non-blocking behavior,
report both `AdaCheckStyleFixed - NoCheckpoint` and `M1 - NoCheckpoint`, then
the direct `M1 - AdaCheckStyleFixed` improvement. All arms must use the same
number of requested and committed recovery points; a faster arm that commits
fewer checkpoints is not a valid improvement.

## 7. Metrics and artifacts

Every formal run must retain enough small evidence to reconstruct its analysis:

```text
command and source/commit provenance
environment and physical topology
model and parallel-plan snapshot
detector catalog and active profile
fault model and planned failure sets
protection plan and all binding hashes
events, queue/memory, NIC/link and CPU service metrics
commit and recovery manifests
analysis summary and status marker
```

Large tensor payloads may be pruned after validation when commands, catalogs,
checksums, plans, summaries, and provenance remain; record every deletion under
`runs/`. The 2026-09-10 cleanup manifest is
`runs/PAYLOAD_CLEANUP_20260910.md`.

Report at least:

- full, reduced, active, residual, application, wire, cross-domain, committed,
  retry, and durable-fallback bytes;
- per-publisher, per-host NIC, shared-link, and CPU apply/serialization load;
- enqueue/commit/exposed-stall distributions, queue depth, peak host/pinned
  memory, iteration time, tokens/s, and planner/replan time;
- per epoch maximum completed remote data-plane edge `transport_ms`, measured
  from serialized publication readiness through remote commit ACK; never sum
  rank times or substitute aggregate bytes/current `commit_ms` for this value;
- recovery success, RPO, RTO, restore bytes, lost/recomputed steps, checksum and
  post-restore numerical continuation;
- absolute values, paired relative changes, trial count, estimator, and 95%
  confidence intervals for primary endpoints.

## 8. Claim boundaries and stop rules

- Say "improves utilization of existing redundancy," not "creates" or
  "expands" redundancy.
- Do not count AdaCheck detector compression as a grouping gain.
- Do not call the clean artifact or the current extension a complete AdaCheck
  reproduction.
- Do not call frozen-catalog bytes online traffic, a static makespan P99, or a
  service-process exclusion a physical host power-off experiment.
- Do not claim rack/ToR/power tolerance without real topology and a matching
  fault experiment; logical domains are simulations.
- Do not claim arbitrary-object online grouping while the materializer remains
  whole-rank and mirror-DP constrained.
- Do not claim globally optimal grouping outside the explicitly solved finite
  special case.
- Do not claim strict 1S1C from final drain, relaxed lag, or a larger queue.
- Do not claim M2 unless it beats the strongest sequential baseline under the
  same training and recovery contract.

Stop or narrow the paper if recovery constraints erase all held-out active
opportunity, if byte gains do not survive physical bottleneck accounting, if
fresh-process continuation fails, or if joint placement is canceled by slower
training. Negative results remain part of the evidence but not in this charter
as run-by-run narration.

## 9. Current implementation priority

1. Freeze the common comparator contract: `AdaCheckStyleFixed` and M1 must use
   the same active detector/encoder, object representation, transport, service
   budget, cadence, global commit, failure sets and recovery oracle. Keep
   `NoCheckpoint` only as the training-cost floor.
2. Connect the existing snapshot `required_value_ids_by_host` and
   `planned_failure_sets` plan to versioned online objects, per-object
   publisher/destination assignments, multi-source materialization and global
   commit. Do not relax the whole-rank DP-peer checks while still restoring by
   copying one peer's full state.
3. Prove CPU recovery closure for AdaCheckStyle, GeminiStyleAdjacent,
   RandomFeasible and RedundancyMax plans, including cross-PP object groups, missing/stale objects
   and every declared failure set. Then use held-out active/service profiles to
   reject zero-opportunity layouts before GPU use.
4. Run the smallest 16-rank end-to-end three-arm gate:
   `NoCheckpoint`, common-runtime `AdaCheckStyleFixed`, and common-runtime M1.
   Require a nonzero fixed-to-M1 active-byte or critical-path prediction before
   launch, and measure total transfer, frequency, max time, total training wall,
   exposed stall and fresh-process recovery. Add random/profile arms only after
   this correctness gate; do not rerun the current zero-choice DP2 treatment.
5. After a recovery-valid positive 10+5/20-step result, add a recovery-correct
   explicit cadence sweep and independent trials. Only then resume 50/200-step
   or request the four-host/32-rank M1 and M2 matrix.

Primary detailed references:

- `PLAN_TWO_NODES_V2.md`
- `docs/TWO_NODE_V2_ACTIVE_GOAL.md`
- `docs/two_node_v2_experiment_matrix.md`
- `experiments/adacheck_clean_baseline/RESULTS.md`
- `docs/ckpt_grouping_paper_outline.md`
- `TWO_NODE_RUN_LOG.md`
- `runs/PAYLOAD_CLEANUP_20260910.md`
- `docs/report-09-12.md`

## 10. Locked baseline-to-M2 execution plan (2026-09-10)

This section records the user-approved plan after correcting the interpretation
of the early experiments. The early runs changed grouping alone. They did not
jointly optimize parallel placement or failure domains, and the 16-rank
`PP4/TP2/DP2` layout made the legal peer graph nearly fixed. They are prior
evidence only, not final performance evidence.

### 10.1 Research sequence and shared baseline

Preserve `../Merak@cf910be` as an immutable public-code artifact. Implement a
separately named `AdaCheckSemanticFixed` reconstruction on the common Merak
checkpoint substrate. It must provide stable two-pass redundancy evidence,
configurable-cadence gradient/direct-value capture with per-step capture as the
final target, ordered remote CPU AdamW
materialization, same-epoch global commit, complete training-state recovery,
and a bounded asynchronous full checkpoint every 50 steps. The 50-step period
is this project's common experiment setting, not a frequency specified by the
AdaCheck paper.

Keep two explicit recovery contracts:

- `AdaCheckSemanticFixed-rank-k2` reproduces the paper's worker-failure and
  group-size-two semantics;
- `AdaCheckSemanticFixed-host1` is the formal common contract for the fixed
  baseline, M1, and M2: excluding any one host and all data local to it must
  still leave a complete same-epoch recoverable job state.

The system modes are `NoCheckpoint`, `AsyncFullCheckpoint`, and
`AdaCheckSemanticFixed`. GeminiStyleAdjacent is a grouping policy rather than a
claim to reproduce the full GEMINI system. Model, input, batch, parallel
placement, fault schedule, service capacity, transport, and measurement windows
remain identical across comparable arms.

### 10.2 Two-host motivation matrix

Use `node17` and `node8` with BERT, GPT-2, and LLaMA-like-3B. Four-rank results
are out of scope because too few workers make grouping artificially fixed.

| World size | GPUs per host | Parallel layout and placement | Purpose |
| ---: | ---: | --- | --- |
| 16 | 8 | `PP4/TP2/DP2`, PP-first, world scope | original positive candidate and DP2 boundary |
| 16 | 8 | `PP4/TP1/DP4`, PP-first, world scope | multiple cross-stage grouping choices |
| 16 | 8 | `PP2/TP2/DP4`, DP-first, data scope | symmetric zero-opportunity control |

For each new frozen catalog compare `AdaCheckStyle`, `GeminiStyleAdjacent`,
`RandomFeasible`, and `RedundancyMax`. Report detector reduction separately from grouping
residual bytes. The exploratory matrix may deliberately include a placement
that exposes a nonzero grouping gap, but its rank order, detector scope, seed,
and rationale must be recorded before inspecting policy results. Keep the
symmetric zero-opportunity layout in the evidence so a favorable placement is
not presented as universal behavior. The motivation claim is that usable
redundancy depends on the joint parallel placement and physical failure-domain
mapping; the policy should find the best recovery-valid grouping for either a
symmetric or asymmetric exported object graph.

### 10.3 Real expert parallelism gate

Before any formal sparse-model result, Merak must have a real `EP` topology
axis and process groups, sharded expert ownership, token dispatch/all-to-all/
combine, and checkpoint holder metadata for shared, router, and expert state.
EP1-versus-EP2 numerical agreement, token conservation, empty/skewed routing,
expert optimizer state, `F_host=1` recovery, and five continued steps are
mandatory gates. Merely instantiating every expert on every rank is not EP
evidence.

### 10.4 Four-host formal matrix

Use exactly 32 ranks in physical order `[node17,node8,node1,node16]`, eight GPUs
per host. Dense models are GPT-6.3B and LLaMA-3-8B with `PP4/TP2/DP4`. Sparse
models are Qwen2-MoE-14.3B and DeepSeek-V2-Lite with
`PP4/TP2/EP2/DP2`. The sparse runs start only after Section 10.3 passes.

Every formal model proceeds through distributed preflight, one step, two
detector passes, a 20-step smoke at the first sustainable explicit cadence,
exclusion of each physical host, a 200-step sustained run, fresh-process
restore, and five continued steps. Then tighten the cadence toward `N=1` rather
than requiring strict 1S1C before the baseline exists. A NoCheckpoint preflight
selects a runnable batch, which is then frozen across all comparable modes and
policies.

### 10.5 M1 and M2 acceptance

The formal grouping-policy set for new experiments is exactly `AdaCheckStyle`,
`GeminiStyleAdjacent`, `RandomFeasible`, and `RedundancyMax`; the retired greedy
and historical standalone `profile_exact` paths are not additional policies.
Communication-profile makespan remains a secondary objective/tie-break inside
M1 rather than a fifth grouping arm. M1 fixes the parallel placement and chooses object publisher,
target failure domain, dependency-complete recovery route, and schedule. It may
group ranks with different PP/TP coordinates when object-level coverage proves
that recovery is complete; same-PP/TP mirror-DP pairing is not an M1
eligibility rule. Its lexicographic objective is: recovery and atomic commit;
actual cross-domain bytes; commit makespan; then publisher/NIC/CPU/materializer/
queue imbalance. Compare fixed, random feasible, whole-rank optimized,
object-static, object-active, M1, service-aware M1, and tractable diagnostic
oracles. Count a gain only when the common recovery contract passes and actual
wire bytes or checkpoint critical path improves.

M2 evaluates validated parallel/rank-to-host candidates by regenerating their
catalogs and calling the unchanged M1. Compare default+fixed, default+M1,
training-only-best+M1, checkpoint-only+M1, the strongest sequential result,
and joint parallelism/failure-domain+M1. Also validate plan-hash invalidation,
same-world remapping, and 32-to-24-rank recovery after one host disappears. M2
is a contribution only if it beats the strongest sequential baseline without
losing the gain to slower training.

### 10.6 Current authorized execution slice and stop point

The BERT, GPT-2, and LLaMA-like-3B two-host motivation slice completed on
2026-09-10. The current authorization supersedes its old stop point and is
limited to **two hosts, 16 ranks on `node17` and `node8`, and the two dense
paper-scale candidates GPT-6.33B and LLaMA-3-8B**. It permits CPU/code work,
short GPU preflights, 10-step checkpoint-and-restart gates, and, only after the
corresponding gate passes, 200-step runs for these two models.

Do not start Qwen/DeepSeek/other EP runs, a third or fourth host, any 32-rank
run, or an M2 four-host treatment without a new explicit user instruction. All
remote experiments must run in named `tmux` sessions through
`tools/run_in_tmux.sh` (or an equivalently durable wrapper), with an append-only
driver log and an exit-status file. SSH or VS Code connection lifetime must not
own an experiment process.

### 10.7 Corrected motivation evidence and metric boundary (2026-09-10)

`RedundancyChecker` now consumes the initialized Merak DP process group instead
of reconstructing DP ranks with PP-major arithmetic. The corrected BERT
`PP2/TP2/DP4`, DP-first, data-scope run has 4,428,588,960 B fixed cross-host
overlap and only 81,408 B residual pair-local transfer. Fixed remains exact, so
this is the symmetric 0% negative control.

The primary byte metric is **checkpoint transfer size across all 16 ranks per
checkpoint epoch**. For a pairing `M`, the current frozen-snapshot study computes
the two directional set differences for every pair and sums them over all eight
pairs. `reduced_logical_bytes` is instead the size of the global unique canonical
object union. One canonical object can be sent by several rank pairs, so this
catalog diagnostic is neither the transfer-size denominator nor an AdaCheck
baseline measurement.

A fresh node17/node8 GPT-2 `PP4/TP2/DP2`, PP-first, world-scope payload capture
completed two training/detector steps. Full rank-local logical state is
7,773,374,288 B and the unique canonical union is 1,097,425,412 B. On the
unchanged frozen catalog, the 16-rank transfer-size comparison is fixed
4,825,431,552 B, random-feasible 4,109,827,584 B, and both redundancy-max and
profile-exact 1,396,784,640 B per epoch. Thus exact grouping reduces this
snapshot-transfer proxy by 71.054% versus fixed. The corresponding two-epoch
service replay measured 71.025% application-wire and 71.021% uncontaminated
aggregate IB-TX reductions. The capture seed is 20260910; the service replay's
random grouping seed is independently frozen at 20260901.

The second node17/node8 configuration uses `PP4/TP1/DP4`, PP-first and
world-scope detector comparison. BERT, GPT-2, and LLaMA-like-3B all completed
fresh two-step metadata captures under the same seed/precision/sequence-length
contract. Their fixed/random/exact 16-rank snapshot-transfer bytes and exact
reductions are respectively: BERT
`5,066,040,704/4,366,214,528/4,132,939,136 B` (`18.419%`); GPT-2
`6,467,501,568/5,539,560,960/5,230,247,424 B` (`19.130%`); and
LLaMA-like-3B `54,534,422,016/52,175,126,016/51,388,694,016 B`
(`5.768%`). `redundancy_max` and `profile_exact` select the same optimum in
each instance. These captures are metadata-only and have neither payload
bundles nor runtime transfer evidence. The selected cross-stage pairs are not
legal peers for the current online incremental path, which requires matching
PP/TP coordinates and distinct DP coordinates. They are placement-conditioned
motivation for the arbitrary-object backend and joint optimization, not
deployable DP4 runtime results.

These comparisons isolate grouping only inside a full-state frozen-object replay.
They are not yet the required AdaCheck baseline because that baseline must use
AdaCheck's per-step gradient/direct-value encoder. The fair baseline gate remains
open: use the same encoder, checkpoint epoch, transport, recovery contract and
16-rank sum for fixed and all proposed policies, changing only protection
grouping. Report encoded logical bytes, application wire bytes and NIC bytes
separately. Do not combine canonical-union reduction with grouping reduction.

With paper AdaCheck `k=2`, each rank sends one destination-independent encoded
record `E_r` to its peer, so baseline payload is `sum_r bytes(E_r)` and rank-pair
reordering alone cannot reduce it. Any M1 byte reduction must be credited to the
new destination-aware reuse rule `E_r - A_peer`, not to AdaCheck or to pairing
alone, and it must carry same-epoch recovery evidence. Role decomposition of the
GPT-2 snapshot result confirms that 1,440,399,360 of the 1,714,323,456 additional
exact overlap bytes are optimizer snapshot objects; this cannot be projected
onto per-step gradients without new update-equivalence evidence.

Retain the paper's redundancy threshold only in the explicitly named
paper-semantics reproduction arm: more than `k` node holders means fully
redundant, while `k` or fewer requires storage. The public/current checker's
strict `< k` boundary remains an artifact-semantic ablation. M1 itself consumes
confirmed canonical equality, holder, requirement, dependency, and byte data;
it does not branch on AdaCheck's `full`/`partial` class or require the paper's
`p`/`o` encoding cases. Those labels remain useful for provenance and byte
attribution. A fixed arm on this common generalized catalog must be called
`AdaCheckDetectorFixed`, while `AdaCheckSemanticFixed` remains reserved for the
paper-semantics reconstruction.

The 16-rank goal is to show that the common detector plus object solver accepts
every exported valid symmetric or asymmetric layout and either returns the
minimum recovery-valid transfer plan for the solved finite instance or an
explicit `zero_opportunity`/`infeasible` result. The 32-rank goal adds enough
physical domains to compare nontrivial groupings and to test M2. “Arbitrary”
does not waive memory, training-validity, failure-coverage, or exact-solver
scope; it means the grouping API is driven by the exported object and topology
contracts rather than hard-coded PP/TP equality.

The common incremental analyzer now reports `remote_data_plane.max_time_ms` as
the maximum completed `remote_*` data-plane edge `transport_ms` per epoch and
the maximum across measured epochs; the two-node 1C1S launcher invokes that
analyzer automatically. It does not sum rank times or use `commit_ms` or a
bytes/bandwidth proxy. An existing LLaMA-3-8B r14 diagnostic re-analysis gives
epoch maxima `17.325/81.907/19.797 ms` and overall `81.907 ms`, but it is not a
grouping comparison. The new motivation captures are metadata-only and contain
no transfer-time samples. Non-blocking evidence still requires a paired
NoCheckpoint comparison of training step-time distributions and foreground
enqueue/stall time.

The analyzer also reports `training_checkpoint_rate` from global progress seen
at optimizer-step boundaries: completed checkpoints during the training window,
optimizer steps per visible checkpoint, and the longest run of steps without
new global progress. It excludes final drain and is an observed service rate,
not a configured `N`. A recovery-correct `N>1` implementation remains open
because simply dropping intermediate optimizer callbacks would lose Adam update
history.

The four GPT-2 replay save conditions are valid within this boundary, but their
shared restore phase hit an occupied port after all saves. A separate
profile-exact run completed two-service catalog restore and survivor restore
after stopping one checkpoint service. Neither result is physical-host failure
or fresh-job continued training. See
`runs/motivation-two-node-16r/corrected-20260910-node17-node8/RESULTS.md`.

### 10.8 Two-host dense end-to-end phase (authorized 2026-09-10)

The frozen motivation figure establishes a placement-conditioned opportunity:
after canonical equality detection, destination-aware set difference and better
grouping can reduce the bytes required to complete a frozen full-state catalog.
It does **not** yet establish that the live AdaCheck per-step encoder transmits
fewer bytes. The current live `peer_delta` path stores a complete optimizer
update on the source host and sends only runtime state to its peer; merely
changing `CKPT_INCREMENTAL_GROUPS` changes the peer identity, not the active
tensor encoder. Under two-host `PP4/TP2/DP2` DP-first placement, the additional
same-PP/same-TP/different-DP peer constraint also leaves only the mirror pairing.
Therefore a policy-labeled A/B is forbidden until its actual remote payload
differs and the restore contract validates that difference.

Use the following attribution ladder. `NoCheckpoint` is the training control.
`AdaCheckSemanticFixed` is the target baseline: its active two-observation
detector must compile the paper's `p`/`o` cases into direct-parameter,
half-precision-gradient, peer-derived, and metadata actions before it may use
that name. It is not implemented by the generic `full_remote` stream.
`M1PeerReuseFixed` is the first executable M1 special
case: the same captured optimizer update is retained/materialized on the local
host, exact peer equivalence is checked for every epoch, and only rank-specific
runtime state crosses to the peer. This comparison can establish the benefit of
destination-aware reuse on top of AdaCheck's detected redundancy, but it is an
encoder/availability ablation, not evidence that an exact rank matching beats a
fixed matching. `M1GroupingOptimized` enters the main grouping result only when
the arbitrary-object recovery plane exposes at least two legal groupings and
changes actual transmitted tensor objects. All arms must bind the same catalog,
parallel plan, protection plan, checkpoint boundary, global commit rule, service
limits, and recovery oracle; treatment-specific payload placement is the intended
difference.

Configuration screening may favor the proposed method, but it must be causal
and pre-registered. First enumerate runnable two-host layouts, reject OOM or
invalid recovery layouts, and use two early detector/update epochs to select a
layout with nonzero held-out active opportunity. Freeze that layout, grouping
seed, optimizer, batch, sequence length, service resources, and trial order
before fixed/M1 measurements. Do not select a configuration from its final
`max_time` or 200-step result. A zero-opportunity model/configuration is reported
as a negative control instead of silently replaced.

For each model, execute in this order:

1. a same-source 15-step `NoCheckpoint` reference;
2. a 10-step checkpoint run, fresh-job restore at committed step 10, and five
   continuation steps for `AdaCheckSemanticFixed`;
3. the identical 10+5 gate for `M1PeerReuseFixed`, followed by
   `M1GroupingOptimized` only after the object plane is active;
4. independent recovery-oracle comparison of each continued step-15 state with
   the uninterrupted reference;
5. only for arms whose 10+5 gate passes, a 200-step sustained run. Start at the
   fastest already demonstrated safe checkpoint service rate and tighten toward
   1S1C; never relabel final draining, a larger queue, or lag-only relaxation as
   a higher checkpoint frequency.

GPT-6.33B starts from the validated BF16 `PP4/TP2/DP2`, `dp-pp`, seq1024,
scalar AdamW, constant-LR, GA80 configuration of r83/r84. LLaMA-3-8B has no
strict large-model checkpoint pass yet: its GA64/GA96 history reached lag two or
worse. It therefore starts with a 10-step `NoCheckpoint` resource preflight and
must find a sustainable checkpoint gate before any 200-step treatment. Training
configuration remains identical across comparable arms; increasing GA is a new
workload and must not be used only in the proposed arm.

The primary per-checkpoint metrics are: (a) total logical encoded bytes summed
over all 16 ranks, (b) total cross-host application and wire bytes over all 16
ranks, and (c) `max_time`, the maximum completed cross-host data-edge transport
time for an epoch, followed by the maximum/quantiles across measured epochs.
Report local shared-memory handoff separately. Non-blocking evidence additionally
requires checkpoint enqueue/stall time and a paired distribution of training
step wall times against `NoCheckpoint`. Checkpoint frequency is the number of
globally committed and recoverable epochs visible during the training window,
reported as checkpoints per optimizer step and steps per checkpoint; final drain
is excluded.

The two-host 10+5 failure gate is a fresh worker/job restart with the same
world size and unchanged parallel/grouping plan. A stopped checkpoint service
may be reported separately as service-process loss. Losing an entire eight-GPU
host leaves only eight GPUs and cannot sustain the 16-rank job without a spare;
it is not a valid two-host host-loss claim. Changed-plan rank/host recovery and
M2 remain deferred to the explicitly authorized four-host phase.

### 10.9 Dense calibration and active-baseline boundary (2026-09-10)

The paper/code audit fixes the naming boundary. Current `full_remote` captures
every AdamW master-parameter gradient and sends the complete record to the
fixed peer service. It does not execute the AdaCheck detector or use detector
results to select its payload; its optional semantic hashes are identity guards,
not an active encoder. It is therefore named `full_remote_proxy` or
`FixedPairGradientIncremental` and is only a protocol/recovery control. The
`adacheck_fixed` experiment label remains reserved until all of the following
are active on the live path:

1. two stable detector observations confirmed by object path, role, dtype,
   shape, and full checksum;
2. a frozen per-object `p`/`o` encoder plan that actually changes the bytes sent;
3. mixed recovery from direct parameters, gradients, and peer-canonical objects
   with complete same-epoch coverage.

The public artifact cannot supply the missing third part: its remote helper has
no caller and `RcharaCheckpointLoader` is empty. Filtering `full_remote` without
adding mixed recovery would merely create a smaller but stale checkpoint.

The first same-workload GPT-6.33B calibration uses BF16 `PP4/TP2/DP2`,
`dp-pp`, seq1024, GA80, constant LR, and seed 20260908 on node17/node8:

- r85 completed all 15 training steps but is excluded because an active-shell
  launcher edit caused a post-training unmatched-quote failure.
- r86 is the clean 15-step `NoCheckpoint` recovery reference: 16/16 terminal
  digests, `COMPLETE`, steady step-wall mean/P99/max
  `12.736/12.771/12.772 s`.
- r87 is a two-step sealed, materialized `full_remote_proxy` calibration. Each
  update sends `53,991,915,008 B` logical and about `53,992,909,740 B` wire over
  all 16 ranks. The per-epoch bottleneck edge is `66.587/51.544 s`; no update
  becomes globally visible during the two-step training window, so GA80 does
  not sustain 1S1C for this proxy. Its step-wall samples are
  `17.601/14.973 s`. The run is valid at lag two and reaches materialized epoch
  two, but does not run a recovery oracle.

r87 predicted a large cross-host `max_time` reduction for
`M1PeerReuseFixed`. The current-source follow-up resolves the GPT gate as
follows:

- r88 used the same GPT workload with a preregistered strict lag-one gate. All
  ten requested epochs became globally committed and all 16 materialized-state
  oracles were numerically close, but every rank observed maximum global lag
  two. The runtime correctly rejected the condition before terminal digests or
  restore. Its ten epochs sent `814,080 B` logical and `2,586,503 B` wire across
  hosts; maximum remote edge time was `0.463 s`. This is a cadence negative
  control, not a recovery pass.
- r89 is excluded. The tmux server did not inherit environment variables placed
  outside `run_in_tmux.sh`, so the frozen command was the launcher's default
  seq128/GA8 diagnostic rather than the named seq1024/GA80 condition. It was
  stopped during setup and has a `FAILED` marker. All later tmux commands place
  `env ...` after the wrapper's `--` and verify `driver-command.txt`.
- r90 reran the identical GPT seq1024/GA80 condition with the acceptance bound
  set to the previously observed lag two. It is `COMPLETE`: both train and
  fresh-restore launchers exited `0/0`; 16/16 rank metrics are valid and
  contiguous; train, restored-boundary and final-restore digest sets each have
  16 ranks; all 16 materialized-state oracles are numerically close; and the
  recovery oracle is valid. Continued steps 11--15 exactly match r86's logged
  losses. Bitwise model/optimizer digests differ for all ranks, as expected
  under the explicit BF16 numerical contract; the LR scheduler is bitwise
  equal.
- Although r90 was accepted against lag two, this trial actually observed
  rank/global maximum lag one. Nine checkpoints became globally visible during
  the ten-step training window, excluding final drain: `0.9 checkpoint/step`,
  or `1.111 steps/checkpoint`. This is encouraging but is not relabeled a
  sustained strict-1S1C result because r88 failed the lag-one gate under the
  same workload.
- r90's ten cross-host epochs total `814,080 B` logical, `2,208,552 B`
  application payload and `2,586,499 B` wire. Per-epoch bottleneck-edge
  `max_time` has mean/maximum `17.50/24.30 ms`. Its steady training step-wall
  mean/P99/max is `13.824/14.769/14.770 s`, versus r86
  `12.736/12.771/12.772 s`; the mean overhead is about `8.5%`.

The r87-to-r90 maximum-edge reduction is large, but remains an explainable
full-gradient-protocol versus exact-DP-peer reuse ablation, not an AdaCheck
fixed-versus-optimized-grouping result. GPT has now passed the authorized 10+5
substrate gate. LLaMA-3-8B may proceed to its same-configuration NoCheckpoint
reference and bounded-lag 10+5 gate; no 200-step treatment advances until that
second model also has a valid recovery oracle, and no four-host run is
authorized.

### 10.10 SemanticFixed-first short-term gate (authorized 2026-09-11)

Do not start a 50-step or 200-step dense treatment until a live
`AdaCheckSemanticFixed` baseline passes its short gate. Reconstruct the missing
public-artifact behavior on the common Merak substrate, preserving the paper's
essential semantics: two stable detector observations, fixed `k=2` checkpoint
groups, per-step half-precision gradient or direct-value encoding, remote CPU
updating, and recoverable training metadata. The implementation may add the
full checksums, dependency closure, canonical publishers, mixed recovery, and
atomic seal needed to make those semantics executable. Record those additions
as this project's reconstruction, not as released AdaCheck code.

Use one frozen per-object action plan with `gradient_bf16`,
`direct_poststep`, `peer_derived`, and `metadata_exact` actions. Unstable or
uncovered objects fail closed or use an explicitly recorded direct fallback;
they are never silently omitted. `AdaCheckSemanticFixed` keeps the paper-style
fixed pairs. Later M1 arms must reuse its detector, encoder, transport, commit,
and recovery implementation and change only destination-aware availability and
grouping decisions.

The current GPU scope remains node17/node8 and 16 ranks. For both GPT-6.33B and
LLaMA-3-8B, evaluate at least two parallel configurations. Start from the
validated `PP4/TP2/DP2`, `dp-pp` configuration. Screen
`PP4/TP1/DP4`, PP-first/world-scope with a one-step resource and collective
preflight before freezing its batch and accumulation settings. Keep
`PP2/TP2/DP4`, DP-first/data-scope as an optional zero-opportunity control, not
as a replacement for a nontrivial treatment.

The execution order is CPU semantic/recovery tests, a same-source
NoCheckpoint reference, a ten-step SemanticFixed save, fresh restore and five
continued steps, then a paired 20-step run only after the 10+5 gate passes.
The gate requires 16/16 rank evidence, contiguous sealed global commits,
complete action coverage, a valid numerical recovery oracle, no poisoned or
queue-full source, and separately reported encoded logical, application, wire,
maximum-edge transport, lag, checkpoint rate, foreground poll/stall, and
training step-time metrics. This authorization does not include 50/200-step,
32-rank, four-host, EP, M2, or the 2x2 fault matrix.

Communication profiling is a separate optional module. It records monotonic
events and computes interval-union idle/slack and tail statistics without
owning grouping, transport, commit, or recovery behavior and without inserting
barriers or CUDA synchronization. The nnScaler EIMC profiler is a design
reference only because its hard-coded layouts and synchronization would perturb
this experiment. Treat profiling as measurement infrastructure until an active
slack-aware schedule beats the same policy without it; only then may it become
a separate system-design contribution.

Every remote command runs in a uniquely named `tmux` session through
`tools/run_in_tmux.sh`, with append-only driver output and exit status. After a
run is classified, retain commands, environment, source hash, manifests,
metrics, analysis, oracle, and status. Delete only explicitly listed,
reconstructable payloads and append the deletion to the cleanup record.

Implementation checkpoint on 2026-09-11: `adacheck_semantic.py` now validates a
same-model/same-rank-topology world-scope two-observation detector artifact,
derives the parallel/protection hashes, and emits a frozen per-rank action plan.
The evaluated two-host host-f1 specialization conservatively emits BF16
gradients for every trainable p/o recovery action and exact runtime metadata;
the generic direct/peer/fallback mixed record remains CPU-tested but is not an
online measured branch. Encoder identity and action counts are checked in the
base, updates, global control, CPU materializer, restore, and metrics. The
standalone `communication_profile.py` records caller-timestamped compute,
capture, poll, and final-drain intervals without CUDA synchronization or policy
dependencies; the analyzer separately reports these tails, manifest role-byte
attribution, and remote-edge `max_time`.

Focused regressions passed 49 CPU tests and 24 single-GPU runtime/materializer
tests. The valid GPT-6.33B detector artifact is
`runs/two-node-v2/detector-gpt6b-bf16-pp4tp2dp2-world-semanticfixed-r2-20260911T151808Z`:
BF16, `PP4/TP2/DP2`, DP-first, world scope, metadata-only, 16 ranks, two stable
observations, catalog hash `43a09457bc0eb36a9d8af23e62b1a39b75d0f74cbbe9aef83e2c78ce5aa8df31`.
The earlier same-name run ending `151523Z` is explicitly `ABORTED`: custom env
was outside the persistent tmux command and defaults were used; it produced no
frozen artifact.

The corrected r99 SemanticFixed 10+5 attempt is a negative gate, not a baseline
result. It uses node17/node8, BF16 GPT-6.33B, `PP4/TP2/DP2`, DP-first, seq1024,
GA80, constant LR, seed 20260908, fixed cross-host pairs, the frozen detector
above, materialized/sealed `full_remote`, four inflight slots and two CPU
threads. Epochs 0--2 reached global control; each update is
`26,995,998,208 B` logical and about `26,996,425,592 B` payload across all 16
ranks. Capture was `0.012--2.116 s`, but foreground poll reached `35.140 s` and
the per-rank CPU AdamW materializer reached `51.371 s`. The producer exhausted
its four bounded slots while capturing epoch 7, so train exited nonzero, no
restore ran, and r99 proves neither a successful baseline nor recovery.

The diagnosed causes are separate: the NCCL-collective global coordinator made
every step wait for the slowest publisher, while replica ingestion shared a
per-source lock with long CPU materialization. The follow-up implementation
adds a shared-filesystem background pin/publish/seal coordinator and a distinct
ingest lock; the standalone communication profiler remains observation-only.
Focused regressions after this change pass 47 tests.

R100 repeats exactly the r99 workload with those two changes. All ten training
steps, all 160 rank/epoch update manifests, and global controls 0--10 complete;
there is no queue-full or poisoned source. Foreground poll falls from r99's
`35.140 s` maximum to `0.771 s`, while the standalone profile reports steady
step-wall mean/P99/max `19.128/22.670/22.765 s`. Five checkpoints become
globally visible during ten training steps (`2.0 steps/visible checkpoint`),
and observed rank/global lag reaches 4/5. Each epoch remains
`26,995,998,208 B` logical; ten epochs total `269,959,982,080 B` logical,
`269,964,256,072 B` payload and `269,970,084,334 B` wire. The bottleneck remote
edge is `45.592 s`; final validation/drain is `3,019.427 s`, and the slowest CPU
AdamW apply is `94.411 s`. This proves foreground control-lock removal, not
non-blocking or 1C1S throughput.

R100 is still a negative recovery gate. All model components are numerically
close, but all optimizer oracles fail the unchanged FP32 tolerance (maximum
absolute error `1.9923318177461624e-6` across ranks), so the strict launcher
correctly stops before fresh restore. The cause is a second lossy conversion:
the live hook first upcast raw BF16 gradients to FP32 and applied unscale/clip,
then the encoder cast that FP32 result back to BF16. The replacement records
the original BF16 gradient plus the exact finite positive per-step multiplier;
CPU replay performs the same BF16-to-FP32 conversion and multiplication as the
live optimizer. This adds one scalar, does not increase gradient tensor bytes,
and does not relax the oracle. Focused semantic/materializer/service/profile
regressions after the repair pass 74 tests. A new 10+5 gate must establish
fresh-process recovery before any 20/50/200-step treatment advances.

R101 (`merak-gpt6b-bf16-pp4tp2dp2-semanticfixed-10plus5-r101-20260911T172451Z`)
is that repaired 10+5 gate and is `COMPLETE`. It keeps r100's node17/node8,
16-rank BF16 GPT-6.33B `PP4/TP2/DP2`, DP-first, seq1024, GA80, scalar AdamW,
fixed cross-host pairs, frozen detector, materialized/sealed `full_remote`, four
inflight slots and two CPU threads. The only semantic data change is raw BF16
gradient capture plus the exact step multiplier. Both train and fresh-restore
launchers exit `0/0`; global controls 0--10 and all 160 update manifests are
contiguous, and no source is poisoned or queue-full.

All 16 materialized-state oracles pass the unchanged contract. The maximum
model absolute error is `0.000244140625`, maximum BF16 tensor-relative Linf is
`0.004651162790697674 < 1/128`, and maximum optimizer absolute error is
`1.1920928955078125e-7`; this fixes r100's optimizer rejection without relaxing
the tolerance. A new 16-rank worker job restores epoch 10, emits all 16 boundary
and final digests, and continues steps 11--15 with losses
`7.9062/7.8438/7.6562/7.6562/7.5625`, exactly equal to the uninterrupted r86
reference. `recovery-oracle.valid=true`. Model/optimizer checksums remain
non-bitwise on all ranks under the explicit BF16 numerical contract; scheduler
checksums match.

Each epoch transfers `26,995,998,208 B` logical across 16 ranks. Ten epochs
total `269,959,982,080 B` logical, `269,964,283,752 B` application payload and
`269,970,112,020 B` wire. Mean/maximum epoch bottleneck edge time is
`26.640/34.815 s`. The independent profiler records rank-max capture
`2.944 s`, poll `0.726 s`, and steady rank-zero step-wall mean/P99/max
`19.075/21.261/21.312 s`, versus r86 NoCheckpoint mean `12.736 s`. Six
checkpoints become globally visible within ten training steps, or
`1.667 steps/visible checkpoint`; rank/global lag is 3/4. This is a bounded-lag
pass, not strict 1C1S or a non-blocking claim. Final drain is `3,040.828 s`, the
slowest CPU apply is `102.732 s`, and fresh restore plus five steps takes
`1,290.592 s`; validation and RTO remain major system limitations.

R101 is the first complete live SemanticFixed baseline gate, but it is still a
fixed arm only. It contains no optimized grouping, service/host fault, changed
parallel plan, NIC counter, or physical-failure-domain comparison. The same-
configuration 20-step paired gate may now run; 50/200-step, 32-rank, four-host,
EP, M2 and the 2x2 fault matrix remain frozen. Its communication profile is a
standalone observability-plane artifact: 16 JSONL files contain 160 compute,
capture and poll intervals plus 16 final-drain intervals, while the module owns
no detector, encoder, grouping, transport, commit or recovery decision.

### 10.11 Paper-story correction: frozen cross-PP opportunity is not live gain (2026-09-12)

The corrected motivation matrix and the current SemanticFixed experiment use
different state and recovery semantics. The former is a frozen full-state
object replay. Its positive exact matchings may pair ranks from different
pipeline stages and count model-parameter and optimizer-snapshot set
differences. For GPT-2, 15.979% of the 3,428,646,912 B fixed-to-exact reduction
is model state and 84.021% is optimizer snapshot state. The latter is an online
incremental reconstruction that conservatively encodes every trainable p/o
recovery action as a destination-independent raw BF16 gradient plus its exact
step multiplier. Its current whole-rank materializer requires equal PP/TP,
different DP and equal optimizer-parameter topology.

Consequently, the frozen result must be named an **offline object-placement
opportunity study** or **frozen snapshot traffic proxy**. It may motivate the
hypothesis that fixed protection groups leave existing redundancy unavailable,
but it is not an AdaCheck online transfer reduction, a deployable cross-PP rank
group, or evidence that M1 improves end-to-end traffic. The paper must not put
the frozen 71.054% GPT-2 result and live SemanticFixed bytes on one axis without
an explicit semantic break. The symmetric DP-first zero-gap result remains a
necessary counterexample: grouping does not create redundancy and may correctly
return `zero_opportunity`.

For one cross-host pair with frozen requirement sets `E_i` and `E_j`, the old
proxy assumes bidirectional difference publication:

```text
i -> host(j): E_i - E_j
j -> host(i): E_j - E_i
```

If each host retains its local objects, both sides then hold `E_i union E_j`.
A perfect cross-host matching can therefore give each failure domain the union
of all rank requirements at the static set level. This does not mean one rank
contains the full model; the services on one failure domain collectively hold
enough objects to instantiate all distributed roles. It also does not prove the
old experiment placed, committed or recovered those objects. That experiment
did not establish independent per-domain same-epoch coverage, a complete
model/optimizer/runtime restore, host-failure survival or continued training.

Cross-PP reuse can become executable only through M1's object-level
holder/requirement graph. Equal bytes at one epoch permit a canonical blob to
populate multiple declared logical requirements, but do not make different
logical parameters permanently identical. Every reusable object needs typed
identity, range, role, version and dependency evidence; each optimizer update
must target the correct logical parameter; the global commit must prove full
same-epoch coverage after every planned failure set. Until that path passes a
fresh-process continuation oracle, PP0-to-PP3 pairing remains an offline oracle,
and the current online runtime must continue to fail closed.

The strongest truthful paper sequence is:

1. frozen traces expose placement-conditioned, model-dependent opportunity;
2. the DP2 symmetric layout provides a zero-opportunity boundary;
3. SemanticFixed reconstructs an executable conservative AdaCheck-style
   detector/gradient/CPU-update/complete-recovery baseline;
4. one common destination-aware object runtime runs both corresponding-rank
   Fixed and optimized M1 plans, with grouping as the only treatment variable;
5. recovery-valid live bytes, critical path and continued training establish
   the causal grouping result; and
6. only then does M2 vary parallel placement and physical failure domains while
   calling the unchanged M1.

The final comparison therefore contains at least `NoCheckpoint`, the existing
`AdaCheckSemanticFixed` reconstruction, common-object-runtime Fixed, and the
same common-object-runtime M1 treatment. SemanticFixed validates the semantic
substrate but is not by itself the grouping control if M1 changes the encoder or
materializer. Fixed versus M1 on the common runtime isolates grouping; any
difference between SemanticFixed and common-runtime Fixed is an encoder/runtime
ablation and must not be credited to grouping.

### 10.12 Dense experiment metrics and 2026-09-12 execution gate

This subsection supersedes the older authorization limits in Sections 10.6 and
10.8 while retaining their correctness gates.

The active execution session is `ckpt-semantic-dense-20260912`, with tmux
windows `guard`, `gpt`, `llama` and `analysis`. Work proceeds on node17/node8,
16 ranks, for GPT-6.33B (the GPT-7B-scale paper shape) and LLaMA-3-8B. Each
model first closes SemanticFixed and common-runtime treatment correctness at
10+5, then a paired 20-step service-rate gate, then the longest sustainable
preregistered run up to 200 steps. At least two training-valid configurations
are screened, but a favorable configuration is retained only when its rationale
is recorded before the policy result, its NoCheckpoint training behavior is
valid, and a symmetric/zero-opportunity control is not hidden.

Do not infer success from a configured save callback. Every treatment must
commit the same requested number of globally visible checkpoints; no drop,
coalescing, queue overflow or final-drain substitution is allowed. Total bytes
are comparable only after this gate. Report absolute and relative values at
three distinct levels: encoded cross-failure-domain logical bytes, application
wire bytes and uncontaminated NIC bytes. Normalize bytes per committed global
checkpoint and report the total over the common committed window. Full logical
state size and canonical-union size remain diagnostics, not transfer metrics.

`remote_data_plane.max_time` remains the slowest completed remote transport
edge, but a single maximum cannot prove non-blocking behavior. For each trial,
compute the per-epoch bottleneck edge and report its P50/P95/P99/max across
epochs, plus capture-to-global-seal makespan. Separately report foreground
capture/poll/exposed-stall tails, publisher queue depth/backpressure, service
apply tails, globally visible checkpoints, steps per visible checkpoint and
maximum rank/global lag. Load balancing requires maximum and mean publisher
bytes, host/NIC/CPU work, imbalance ratio and the bottleneck resource; total
bytes alone cannot validate innovation point 2.3.

Add two noninterchangeable training-time metrics. `training_window_wall` spans
the common post-warmup optimizer-step window and is the primary throughput
comparison against NoCheckpoint. `job_wall` includes initialization/base
publication, training, final seal/drain and mandatory validation. Report both:
excluding the drain can hide an unusable checkpoint service, while including a
large setup or validation phase can obscure whether checkpointing blocked the
training critical path. Use independent paired trials with rotated order and
confidence intervals; ranks and steps inside one trial are not replicates.

Recovery reliability is a vector, not one Boolean: declared failure scenario,
recovery success rate, RPO/last sealed epoch, RTO to the first resumed optimizer
step, model/optimizer numerical error, exact scheduler/RNG/sampler/iteration
coverage and continued-loss/state trajectory. Healthy-service worker restart,
service loss, physical host loss and durable fallback remain separate claims.
Also report peak GPU, pinned/shared host and service-resident memory so a byte or
time gain cannot be purchased by unbounded buffering.

R101 is the current GPT SemanticFixed 10+5 reference, not a non-blocking pass:
only 6/10 checkpoints were visible during training, max global lag was four,
steady step mean was 19.075 s versus the r86 NoCheckpoint 12.736 s, and final
drain was 3,040.828 s. The immediate engineering target is to preserve its
numerical recovery contract while raising sustained service rate and then
measure a common-runtime Fixed/M1 byte difference. Do not launch 32 ranks merely
because a 16-rank job completes. First require both dense models to show a
recovery-valid positive treatment under identical semantics and close the
long-tail/backpressure problem; then stop, summarize the evidence and evaluate
whether the four-host gate is scientifically justified.

### 10.13 Detector-active bytes and the faithful fixed-peer boundary (2026-09-12)

The public AdaCheck implementation makes a narrower decision than r101. For
each parameter, it intersects the detector result for `p` with the result for
the corresponding optimizer state `o`; it adds that parameter to gradient
checkpointing only when the distinct holder-node count is below `k`. Its public
fixed grouping then pairs corresponding local ranks on consecutive groups of
`k` nodes. The public remote updater has no reachable caller and its loader is
empty, so our recovery implementation is necessarily a reconstruction, but the
detector/active-set rule must not be silently replaced by "send every gradient."

Applying only the node-count part of this rule to the frozen GPT-6.33B detector
catalog gives a useful diagnostic: 112 one-host objects and `605,696 B` total,
split into `262,144 B` model, `262,144 B` optimizer and `81,408 B` RNG state.
The other 3,601 objects (`107,981,291,524 B`) already have confirmed holders on
both hosts. On this diagnostic active catalog, fixed, random-feasible,
redundancy-max and profile-exact all produce `605,696 B` transfer and the same
`87,008 B` maximum pair peak. This is not yet a final AdaCheck action plan:
mapping `p` to its corresponding `o` and selecting one recovery action per
parameter remains required. The immutable source is the r2 detector's
`frozen/object_catalog.jsonl`; do not substitute r101's encoder-plan counts.

R101's `26,995,998,208 B/epoch` all-gradient stream is therefore retained as a
conservative executable recovery control, not the faithful detector-driven
AdaCheck byte baseline. The r103 20-step extension confirms that it is also
unsustainable at 1S1C: rank 6 reaches epoch 13 with all four publisher slots
occupied and raises `CheckpointBackpressureError`. Increasing inflight capacity
would move the failure and enlarge the final drain; it is not a service-rate
fix. `MATERIALIZED_RESPONSE_WORKERS=2` may parallelize final GET responses, but
does not improve training-time publisher or CPU replay throughput.

For a DP-exact layout, the next fixed baseline may safely exploit a stronger
live condition. Each fixed pair must have equal PP/TP, distinct DP, identical
optimizer topology, equal complete base tensor evidence and equal per-epoch
BF16 update evidence. Each host locally materializes its own full p/o update;
only the source rank's exact runtime state crosses to its peer. A global epoch
is visible only after the reciprocal records and the per-epoch SHA-256
equivalence proof agree. Recovery reads the surviving DP peer's full p/o state
and the failed rank's runtime record. This is an AdaCheck-style fixed
redundancy/encoder result with a runtime equality extension, not M1 grouping.
If equality fails, the epoch fails closed because the current path has no sparse
direct fallback.

This correction sharpens the 16-rank expected result. `PP4/TP2/DP2` has exactly
one legal cross-host DP peer per rank, so grouping has no choice. In
DP-first `PP4/TP1/DP4`, there are 16 legal whole-rank perfect matchings, but
synchronized dense DP replicas make their active holder sets and encoded costs
equivalent; more combinatorial choices do not imply byte opportunity. PP-first
`PP4/TP1/DP4` puts PP0/1 on one host and PP2/3 on the other and has no legal
cross-host same-PP peer. It becomes recovery-valid only after the object-level
online materializer exists. Consequently, do not force a positive dense
grouping result by rotating ranks or weakening recovery. Uniform dense layouts
are valid common-runtime correctness and zero-opportunity controls; a positive
M1 byte result requires a preregistered heterogeneous partial-replica holder
graph and the sparse object runtime described in Section 71.3.

### 10.14 r104 negative gate and preregistered rerun contract (2026-09-12)

The fixed-peer SemanticFixed run
`merak-gpt6b-bf16-pp4tp2dp2-semanticfixed-peer-10plus5-r104-20260912T045100Z`
is a useful negative service-rate result, not a completed 10+5 recovery run.
It used node17/node8, 16 A40 ranks, random-init GPT dense 6.33B (the repository's
AdaCheck GPT-7B-scale dimensions), BF16 `PP4/TP2/DP2`, `dp-pp`, seq1024, GA80,
scalar AdamW, constant LR, seed 20260908, the frozen r2 detector, the
SemanticFixed BF16 gradient encoder, fixed cross-host DP peers, shared-tensor
local publication, two materialized response workers, four inflight slots and
a configured lag-two strict gate.

All 10 requested epochs and all 160 rank updates eventually committed; global
controls 0--10 are contiguous and 16/16 materialized model/optimizer oracles
are numerically close. Nevertheless only six checkpoints became globally
visible in the ten-step training window, and maximum rank/global lag was 4/4,
so the strict lag-two gate correctly failed. The train launcher exited 1/143,
the runner did not launch fresh restore, and no continuation or recovery oracle
exists. It must not be reported as recovery success, 1C1S, non-blocking, or M1.

The remote runtime-only stream is `81,408 B` logical per epoch; ten epochs total
`2,372,072 B` application payload and `2,750,010 B` wire, with epoch bottleneck
edge mean/max `113.523/373.408 ms`. The local p/o recovery-operation stream is
`269,959,982,080 B` over ten epochs. These are disjoint accounting layers: the
small cross-host runtime stream is enabled by fixed equivalent DP-peer reuse,
not a measurement of the complete checkpoint catalog and not an optimized-
grouping gain over the same SemanticFixed runtime.

Steady rank-zero step time is `13.287 s` versus r86 NoCheckpoint `12.736 s`
(`+4.33%` as a single ordered development observation). The old combined final
drain is about `2726.014 s` per rank. The last materializer apply finishes about
64 seconds into that interval; most of the remainder is the per-host serialized
94.49 GB full-state validation responses and rank barrier. The old profile
cannot split those terms exactly, so the source now records publisher drain,
global publish/seal, reader wait, fetch/deserialization, oracle comparison,
barrier, restore load and total finalize/restore separately.

The following acceptance contract is frozen before the next GPU run:

1. `recovery_valid` requires contiguous rank epochs, final global commit,
   unchanged model buffers and the enabled materialized numerical oracle.
   `cadence_valid` independently reports whether rank/global lag meets the
   configured target. `STRICT_GATE=0` permits a recovery-valid bounded-lag run
   to continue to restore; it does not relabel cadence failure as 1C1S.
2. Full-state reads use one or two node-local lock slots. The runner enforces
   `MATERIALIZED_RESPONSE_WORKERS >= MATERIALIZED_READERS_PER_NODE` and records
   the choice. Two readers are an engineering tail experiment, not a grouping
   treatment, and require host-memory headroom.
3. Fresh continuation must cover exactly steps 11--15. Train,
   restored-boundary, restore-final and uninterrupted-reference digests must
   each cover ranks 0--15. Losses must be finite and satisfy the preregistered
   BF16 contract `rtol=1e-4, atol=1e-4`; exact equality remains a stronger
   diagnostic but is not the numerical acceptance rule.
4. Timing reports three non-substitutable quantities: rank-zero training-window
   step wall, complete train/restore launcher phase wall, and background remote
   bottleneck-edge time. Only exposed capture/poll or paired step/phase overhead
   supports a `not-block-training` statement. Final drain is recovery-readiness
   tail and is never hidden inside checkpoint frequency.

The next execution order is a fresh current-source 15-step NoCheckpoint GPT
reference, then a 10+5 fixed-peer SemanticFixed recovery run with the same
training workload and relaxed cadence enforcement while retaining the lag-two
measurement target. LLaMA-3-8B gets its own two-observation detector and fresh
NoCheckpoint/SemanticFixed pair only after the GPT framework passes. The
four-host/32-rank gate remains closed.

Paper writing must preserve the following causal distinction. The early BERT,
GPT-2 and LLaMA-like-3B motivation figure used a frozen full-state catalog and
allowed cross-PP rank pairing; it demonstrates that placement-conditioned
grouping opportunity exists, and GPT-2's offline reduction contains about
15.979% parameter bytes and 84.021% optimizer-state bytes. Cross-PP unions can
contain all tensor names across a group but cannot restore either whole PP rank:
each failed rank requires its own shard, optimizer ownership and runtime state,
not the union of unrelated stages. Therefore the motivation result is an
observation that motivates recovery-constrained object placement, not an
end-to-end AdaCheckFixed-versus-M1 result. A headline M1 gain requires the same
detector, encoder, active object stream, transport, commit and recovery
contract, and a layout with at least two heterogeneous recovery-feasible
choices. Symmetric DP2 is intentionally a zero-opportunity control.

### 10.15 r105/r106 corrected common-runtime gate (2026-09-12)

R105 is the fresh current-source 15-step NoCheckpoint reference for random-init
GPT dense 6.33B: 16 ranks, BF16 `PP4/TP2/DP2`, `dp-pp`, seq1024, GA80, scalar
AdamW, constant LR and seed 20260908. It is `COMPLETE`, both launchers exit 0,
and all 16 final digests exist. Steps 2--15 have mean/P95/max
`12.755/12.782/12.788 s`; the 15-step training window is `192.472 s` and full
train launcher phase is `255.034 s`.

R106 uses the identical source snapshot SHA-256
`4f53ed6ce3ccd7a10cb42416230029fc644a668c48693c782c2ac3616780b67c`,
training configuration and r2 SemanticFixed detector, with fixed equivalent DP
peers, two node-local materialized readers, two response workers, four service
CPU threads, four inflight slots, lag-two measurement and relaxed cadence
enforcement. Train and fresh restore both exit 0/0 and the run is `COMPLETE`.
All 160 updates and global controls 0--10 are contiguous; all 16 train,
restored-boundary and restore-final digests exist; all 16 materialized model and
optimizer states are numerically close; no source is poisoned. Exactly steps
11--15 are present and their losses equal r105 bit for bit. The preregistered
finite `rtol=atol=1e-4` continuation oracle passes with zero observed loss error.

This is `recovery_valid=true` but `cadence_valid=false`: six checkpoints are
visible during ten optimizer steps (`1.667 steps/checkpoint`), longest no-
progress interval is two steps, and maximum rank/global lag is 4/4. It is not
1C1S. Steady step mean/P95/P99/max is
`13.294/13.967/13.983/13.987 s`, a single ordered mean difference of `+4.22%`
from r105. Capture max is `1.443 s`, poll max `77.529 ms`; thus foreground
polling is small, but the full condition is not yet a zero-overhead or formal
non-blocking claim.

Ten remote runtime-only updates contain `814,080 B` logical,
`2,370,792 B` application payload and `2,748,739 B` wire. Epoch bottleneck-edge
mean/max is `340.429/1133.870 ms`, worse than r104's single-run mean/max, so no
max-time improvement is claimed. The local recovery-operation stream remains
`269,959,982,080 B` logical: `269,959,168,000 B` gradient actions plus
`814,080 B` runtime state. It is not cross-host total transfer and cannot be compared
as an M1 byte treatment.

The train phase is `2172.851 s`, including `1494.986 s` finalization; restore
plus five steps is `780.096 s`, including `651.650 s` restore total. Relative to
r104, two-reader validation reduces finalization from about `2726.014 s` to
`1494.936 s` mean (`45.2%`) and the train phase from about `3381.387 s` to
`2172.851 s` (`35.7%`). The split tail shows publisher drain max `14.214 s`,
global publish/seal max `63.656 s`, fetch/decode max `143.607 s`, oracle compare
max `261.378 s`, reader wait max `1081.390 s`, and validation-barrier max
`1079.587 s`. Concurrency removes serialization batches, but full-state
validation remains a large post-training recovery-readiness tail.

R106 closes the corrected GPT 10+5 common-runtime gate only. Symmetric DP2 has
one legal peer per rank, so it remains a zero-opportunity grouping control. Do
not promote it to M1, max-time improvement, strict 1C1S, physical host fault or
paper-level replicated performance. The independent LLaMA-3-8B detector,
reference and 10+5 gate were subsequently completed as R108--R110 below;
32-rank work remains closed.

### 10.16 r108--r110 LLaMA-3-8B common-runtime gate and stop point (2026-09-12)

R108 is the independent LLaMA-3-8B detector run on node17/node8: 16 ranks,
BF16 `PP4/TP2/DP2`, `dp-pp`, two real training observations and world-scope
model/optimizer metadata comparison. It is `COMPLETE` with launcher `0/0`.
The two observations produce 2,653 stable components and 5,348 stable records;
the frozen catalog hash is
`b33b02b4b7be9086a8a88489735c07861eb61bb700b66e2b41bd06af35bb9aa8`.
Its `290,598,506,544 B` full and `145,297,178,116 B` reduced values are detector
metadata sizes, not a recoverable bundle or measured transfer.

R109 is the fresh current-source NoCheckpoint reference: the same 16-rank
layout, BF16, seq1024, GA96, scalar AdamW, constant LR and seed 20260908. It is
`COMPLETE` with 16 final digests. Steps 2--15 have mean/P95/P99/max
`21.512/21.540/21.560/21.565 s`; the 15-step training window is `323.792 s`
and the complete train launcher phase is `409.769 s`.

R110 uses that workload and source snapshot, R108's frozen detector,
SemanticFixed BF16-gradient encoding, fixed equivalent DP peers, two readers,
two response workers and four service CPU threads. Train and fresh restore both
exit `0/0`; global controls 0--10 and all 160 rank updates are contiguous, all
required digest rank sets cover ranks 0--15, all materialized model/optimizer
oracles are numerically close, and no source is poisoned. The fresh job restores
epoch 10 and produces exactly steps 11--15. Continuation losses are numerically
close to R109 under the preregistered `rtol=atol=1e-4` contract: maximum absolute
and relative errors are `0.0002` and `2.6692e-5`. State and losses are not
bitwise equal, so only the numerical recovery claim passes.

R110 is `recovery_valid=true` and `cadence_valid=true` for the configured
bounded-lag-two contract. Eight checkpoints become visible in ten steps
(`1.25 steps/checkpoint`); maximum rank lag is one but maximum global lag is two,
so this is not strict 1C1S. Its steady mean/P95/P99/max is
`22.100/22.763/22.783/22.788 s`, a single ordered mean difference of `+2.735%`
from R109. Capture and foreground poll maxima are `1.701 s` and `54.173 ms`.
This is useful bounded-lag evidence, not a replicated non-blocking result.

The ten cross-host runtime-only updates contain `814,080 B` logical,
`2,371,432 B` application payload and `2,749,370 B` wire. Their epoch
bottleneck-edge mean/max is `23.456/49.621 ms`. The local full recovery-operation
stream contains `363,248,839,680 B` logical, of which `363,248,025,600 B` are
gradient actions; its maximum shared-path transport time is `20.728 s`. These
measure different paths and must not be summed or relabeled as total NIC bytes.
The gradient actions represent operations that reconstruct paper `p/o` state,
not an M1 transfer saving.

The train/restore launcher phases are `2946.320/1000.154 s`; finalization and
restore-total maxima are `2105.938/808.645 s`. Finalization is dominated by
materialized-reader wait (`1500.130 s` max), validation barrier (`1737.204 s`),
full-state fetch/decode (`194.506 s`) and numerical comparison (`414.121 s`).
Thus the background update path does not create a large foreground poll, but
recovery-readiness still has a severe post-training tail.

R106 and R110 complete the two requested dense 16-rank common-runtime recovery
gates. Neither is a positive M1 grouping experiment: in symmetric DP2 every
rank has exactly one recovery-legal same-PP/TP peer. The prerequisite “both
models show positive grouping gain and the tail is solved” is false. Therefore
no 20/50/200-step treatment, 32-rank/four-host run, EP run or 2x2 fault matrix
is authorized from these results. The next research action is an offline and
CPU-tested recovery-feasible layout/object-plan opportunity gate; only a layout
with at least two heterogeneous legal choices should consume another large GPU
run.

The auditable composite is
`docs/figures/semanticfixed_dense_16r_composite.{png,svg}` with parsed source
data in `semanticfixed_dense_16r_composite_data.{json,csv}` and a colocated
regeneration script. It visibly excludes both controls from strict 1C1S/M1
benefit aggregation and keeps remote runtime bytes separate from local
recovery-operation logical bytes.

## 70. Clean AdaCheck baseline and the two-module research architecture (2026-09-09)

This section records a three-way design review by the main agent, the
`gpt-5.6-sol-high` clean-baseline agent, and the `gpt-5.6-terra-high` research
auditor. Only the clean-baseline agent ran a new experiment. The design review
did not launch experiments or change execution code.

### 70.1 What the clean public artifact can and cannot provide

The canonical baseline archive is `experiments/adacheck_clean_baseline/`; its
only formal run is
`runs/artifact-cf910be-gpt14b-8a40-20260909-adacheck-clean` below that directory.
It uses eight A40 GPUs, the paper's Table 2 GPT-1.4B shape (1.516B constructed
parameters), `PP1/TP1/DP8`, ZeRO-1, sequence length 128, and four steps. The
compatibility supplement changes three environment/framework compatibility
points and supplies one harness; the AdaCheck saver and detector files are
byte-identical to public commit `cf910be`.

The run completes and invokes the detector at steps 1 and 2. Its two full saves
are `37,238,490,360 B` each and average `65.4349 s`; its two later artifact
`Saving grad` files are `10,142,400,529 B` each and average `11.9440 s`. The
physical-file ratio is `3.6716x`; the loaded logical-tensor ratio is only
`1.9392x`, partly because serialized tensor views retain larger backing storage.

This is not a recoverable one-step-one-checkpoint baseline. The smaller model
file contains only two flat parameter tensors, not gradient records, and lacks
iteration, complete model/optimizer, random-number-generator, scheduler, or
base-chain metadata. `RcharaCheckpointLoader` is empty; the trainer selects the
base loader, which rejects the partial state. The public call graph has no
reachable remote sender, CPU updater, commit, or AdaCheck recovery path. The
reachable saver performs synchronous `torch.save` plus barriers, with save time
far above step time. Hence written directories are 4/4, remotely committed and
recoverable epochs are 0/4, and strict 1S1C is **FAIL** for this artifact path.

The result is an artifact-reachability fact, not a direct refutation of the
paper's `6--896x` result: hardware, parallel strategy, denominator, and recovery
semantics differ. It is also not the performance baseline for our modules.
Future comparisons use one common recoverable substrate; the fixed control is
called `AdaCheckStyleFixed`, while the clean native/compatibility artifact is
reported separately. See `experiments/adacheck_clean_baseline/RESULTS.md` and
`PAPER_TO_CODE.md` for the full evidence contract.

### 70.2 Separation principle for the two innovations

The project is split into two modules with a strict ownership boundary:

```text
AdaCheck-style detector + active encoder
                 |
                 v
M1: grouping under a fixed parallel placement P
                 |
                 | exposes value function Phi(P) and a proof-carrying plan
                 v
M2: choose P and failure-domain mapping using the same M1 as an inner policy
                 |
                 v
unchanged transport -> atomic commit -> recovery
```

M1 never changes the training parallel placement. M2 never reimplements object
grouping, publisher selection, or transport; it calls M1 for every candidate
placement. Detector compression is preprocessing and is not counted as a gain
of either module. This ownership rule makes the ablations interpretable and
prevents a monolithic optimizer from claiming the same reduction twice.

Shared data contracts should be small and explicit:

- `RedundancyCatalog`: semantic object identity, role, encoded bytes, natural
  holders, requirements, stability/provenance, and base/delta dependencies;
- `ParallelPlanSnapshot`: PP/TP/DP/EP role, shard ownership, rank/device/host
  mapping, memory, collective schedule, and training-time profile;
- `FailureTopology`: GPU/host/service and, only when real, rack/ToR/power
  containment plus link and service capacities;
- `ActiveProfile`: held-out encoded bytes, wire expansion, usable compute/network
  slack, CPU apply/serialization distributions, and confidence bounds;
- `ProtectionPlan`: checkpoint groups, per-object publisher/destination,
  dependency closure, deadlines, predicted resource loads, and plan proof;
- `JointPlan`: one `ParallelPlanSnapshot`, its M1 `ProtectionPlan`, and hashes
  binding both to the detector catalog and failure contract.

The transport, commit, and recovery layers consume `ProtectionPlan`; they do
not import or understand either optimizer.

### 70.3 Module 1: recovery-constrained object-aware checkpoint grouping

**Problem boundary.** M1 receives a fixed `P`, fixed physical services, and a
fixed minimum failure-eligibility mask. It chooses disjoint checkpoint groups
of host services and, inside each group, assigns every protected object to a
valid publisher and destination. Thus it remains a checkpoint grouping policy,
but objects are the protection/recovery atoms, hosts/services are the group
vertices and resource/failure units, and ranks are logical requirements and
publisher identities. Whole-rank rank pairing remains a baseline/special case,
not the recovery authority.

For source host `i`, let `Q_i(P)` be the dependency-closed set needed to restore
all logical ranks placed on `i`, `A_j(P,e)` the exact same-epoch objects already
available on target host `j`, and `a_oi(e)` the post-AdaCheck active encoded
bytes for object `o`. The directional residual is

```text
c_ij(P,e) = sum(a_oi(e), o in Q_i(P) - A_j(P,e)),
```

where every canonical object is charged once per destination service, not once
per rank that consumes it. A conservative planner uses held-out upper residuals
or lower confidence bounds on reuse; runtime exact schema/version/checksum
validation either proves reuse or sends the missing object. Prediction may miss
a performance target but cannot weaken correctness.

For pairwise groups, define a profiled edge cost

```text
tau_ij = max(network_ij(c_ij), network_ji(c_ji),
             cpu_j(apply_i), cpu_i(apply_j), shared_service_load_ij)
         minus only slack that profiling proves is simultaneously usable.
```

The primary objective is lexicographic: first satisfy recovery, atomicity,
capacity and a declared recovery-point/recovery-time contract; then minimize
checkpoint critical-path/CVaR or exposed stall; then bottleneck NIC/link/CPU
load, cross-domain active bytes, recovery tail, and plan churn. Logical bytes
alone are not the primary objective because r81--r84 show a CPU-service-rate
bottleneck after remote payload becomes small.

M1 correctness has two necessary invariants:

1. **Recovery closure.** For every required object and every fault allowed by
   M1's fixed eligibility contract, at least one surviving service holds an
   exact same-epoch object or a complete base/delta dependency chain. The set
   includes model, optimizer, scheduler, RNG, sampler, and progress state.
2. **Atomic visibility.** An epoch is visible only when every required coverage
   record is bound by the same catalog, parallel-plan, group-plan, encoding and
   control hashes. Partial groups fail closed.

The reconstruction proof is by induction over the dependency directed acyclic
graph: checksum-valid leaves are available after each registered fault; if all
predecessors of an object and its same-epoch update are available, that object
is reconstructible. Coverage for every required root plus the frozen templates
reconstructs every logical rank. This is a static proof obligation; fresh-job
numerical continuation remains the runtime validation.

For pairwise host groups with one independent backup, additive edge costs and
no shared-capacity coupling, minimizing the largest `tau_ij` is a bottleneck
perfect-matching problem: binary search an edge threshold and use a Blossom
perfect-matching feasibility test. A minimum-weight matching breaks ties by
total work. With only four hosts, enumerating the three pairings is simpler and
provides an exact oracle. Per-object independent placement with linear capacity
can use min-cost flow. Dependency chains, shared NIC/CPU capacity, indivisible
objects, multi-fault coverage, and group size above two require a small mixed-
integer model or a heuristic checked against small exact instances; no general
global-optimum claim is allowed. The current 8x8 bitmask dynamic program is only
an exact oracle for its old bipartite rank-matching special case.

**Structural stop cases.** M1 must return `zero_opportunity` instead of forcing
an optimized label when any of the following holds:

- current DP2 whole-rank recovery gives every rank one legal remote peer;
- two physical hosts and one pooled service per host leave only one destination
  failure domain and invariant host-deduplicated bytes;
- only frozen full-state cost changes while held-out active/application/NIC
  bytes and service critical path do not;
- detector evidence, optimizer coupling, metadata, or dependency closure is
  incomplete; or
- candidate plans cannot pass atomic commit and fresh-job recovery on the common
  substrate.

M1 has an independent research contribution only if, under a fixed `P`, at
least two recovery-feasible groups have heterogeneous post-reduction active or
service costs, and its proof-carrying object plan converts that opportunity into
repeatable wire/critical-path improvement without weakening recovery. Otherwise
M1 is a necessary mechanism for M2, not a standalone headline.

### 70.4 Module 2: parallel-placement and failure-domain co-design

**Problem boundary.** M2 starts only after the detector, active encoder, and M1
are fixed. An offline planner such as nnScaler supplies a small preregistered
candidate set `Pi`; M2 does not call it online and does not claim to solve the
general automatic-parallelization problem. Each candidate `P` fixes training
semantics but changes shard/object holders, collective traffic, compute slack,
memory use, and their physical failure domains. M2 regenerates the detector
catalog/profile for that `P`, derives domain eligibility from the real topology,
and calls exactly the same M1:

```text
Phi(P,F) = M1.solve(catalog(P), P, profile(P), eligibility(F)).cost
P_star = argmin over P in Pi of JointCost(P, Phi(P,F)).
```

M2 owns only selection/mapping of a valid parallel candidate to physical
devices and the hierarchical domain contract. M1 continues to own checkpoint
groups, publishers, destinations, and object schedule. Physical domains are not
movable labels; M2 changes which logical shards/replicas occupy them.

Every candidate must satisfy model/collective correctness, GPU/CPU memory and
the same recovery-point objective (RPO), recovery-time objective (RTO), commit
and numerical-recovery contract. It must also satisfy a preregistered training-
only slowdown cap relative to the fastest valid training plan. The initial
functional gate is not necessarily strict lag one: all requested epochs must be
committed with bounded queue/memory, final drain and reported lag; frequency is
then swept toward lag one. A result is called strict 1S1C only when every
boundary has `lag<=1` and the corresponding fresh-job recovery contract passes.

After hard feasibility, M2 minimizes expected useful-job completion time or its
Pareto components, not `training time + checkpoint bytes`. One auditable form is

```text
J(P) = training critical path
     + exposed checkpoint stall under M1(P)
     + sum_f failure_rate_f * (RTO_f + lost_steps_f * step_time(P)),
```

while separately reporting training collective bytes, AdaCheck active payload,
M1 protection traffic, local CPU work, and durable-fallback traffic. Shared
links are capacity-constrained once using the joint schedule; their time is not
added a second time in separate modules. If fault rates are not defensible, the
paper reports a Pareto frontier and scenario-specific results instead of a
fabricated scalar.

On a frozen candidate set, enumeration plus one M1 solve per candidate is
`O(|Pi| * T_M1)` and is exact only relative to that finite set. The composition
proof is likewise modular: the parallel-plan validator proves training/shard
semantics, M1 proves recovery closure under the candidate's domain mask, and
catalog/plan hashes bind them. Any placement change invalidates the old catalog
and group plan.

M2's decisive comparison is not against a weak fixed layout:

```text
joint M2(P, M1(P))
    versus
fastest training-only P followed by the best post-hoc M1(P).
```

If both select the same Pareto point or joint checkpoint savings are canceled
by slower training, the co-design hypothesis is rejected. The defensible
novelty is that parallel placement changes AdaCheck-reduced object-holder graphs
and protection/service cost, while explicit physical-domain coverage constrains
which reuse remains usable. It is not first checkpoint placement (GEMINI), first
failure-domain backup (ByteRobust), first erasure-coded in-memory checkpoint
(ECCheck), or a new generic auto-parallel planner (nnScaler).

### 70.5 Baselines and ablations that keep the modules separable

All arms share detector, active encoder, transport, atomic commit, recovery,
model/input/seed, checkpoint request stream, service limits, and fault contract.
The clean public artifact remains a reachability audit, not a performance arm.

M1 arms under one fixed `P` are:

```text
AdaCheckStyleFixed
whole-rank constrained exact       # DP2 zero-opportunity control
object-static                      # frozen-catalog upper-bound ablation
object-active                      # held-out active residual
object-active + recovery closure
object-active + recovery closure + service-rate objective
small exact oracle lower-bound diagnostic
```

M2 arms, all using the same final M1 implementation, are:

```text
default P + M1
training-only-best P + M1          # sequential baseline
checkpoint-only-best P + M1
joint P/domain + M1
joint without correlated-domain constraint
joint without service-rate term
joint reduced to whole-rank grouping
```

For M1, report full logical state -> AdaCheck-reduced active state -> grouping
residual -> application/NIC bytes as four non-overlapping stages, then commit
critical path/lag/RPO, training overhead, RTO and numerical continuation. For
M2, additionally report training-only step time and collective traffic per
candidate so that checkpoint savings cannot hide a worse parallel plan.

### 70.6 What two hosts and four hosts can establish

Two hosts and 16 GPUs remain useful for the common substrate, detector stability,
held-out active-cost prediction, object manifest/dependency recovery, publisher
balancing, service-loss injection, numerical next-step continuation, and the
DP2 unique-matching negative theorem. They cannot show a nontrivial host
grouping, physical correlated-domain diversity, or continuation of the same
16-GPU job after losing one eight-GPU host without spare/replacement capacity.

Four hosts and 32 GPUs first provide three pairwise host matchings, multiple
object destination domains, shared-link/load alternatives, and a meaningful
sequential-versus-joint M2 comparison. Even then, four physical hosts establish
only host-level diversity unless rack/ToR/power placement and telemetry are real
and the corresponding fault is actually injected; logical labels must be called
simulation.

The staged go/no-go policy is:

1. Use two hosts to finish the recoverable object substrate and show that held-
   out active cost predicts actual bytes/service load. Do not spend 200-step
   treatments on a catalog whose recovery-feasible opportunity is zero.
2. Run M1's grouping headline only when at least four physical hosts provide two
   or more feasible groupings and at least two workloads show a preregistered
   nontrivial active/service gap. First complete bounded-lag 200-step runs, then
   tighten frequency and report the fastest sustainable checkpoint interval.
3. Start M2 only when at least three valid parallel candidates lie within a
   preregistered training-only slowdown band and their M1 costs materially
   differ. Require joint optimization to dominate the sequential baseline on a
   Pareto metric, not merely reduce checkpoint bytes.
4. Stop or narrow the paper if gains exist only in `C_static`, disappear after
   dependency/failure constraints, fail fresh-job continuation, or rely on
   simulated domains presented as physical reliability.

The resulting paper story has two non-overlapping proposed contributions:

1. **M1:** for a fixed parallel placement, a recovery-constrained, object-aware
   checkpoint grouping policy that converts AdaCheck redundancy into a
   proof-carrying and service-aware protection plan;
2. **M2:** on top of the unchanged M1, an offline parallel-placement and
   physical-failure-domain co-design that exposes redundancy/protection choices
   unavailable to sequential training-only placement.

Both remain proposed contributions. M1 needs a nonzero recovery-valid runtime
treatment, and M2 needs a four-host sequential-versus-joint result before either
can be written as an achieved system contribution.

## 71. Code-grounded implementation boundary for M1 and M2 (2026-09-09)

This section records the follow-up code audit after the clean AdaCheck baseline
and the three-agent design review. It is deliberately narrower than Section 70:
it states what the current implementation already proves and the smallest safe
path from that implementation to the two proposed modules. No experiment or
execution-code change was made during this audit.

### 71.1 Reuse the existing snapshot protection plane

The snapshot form of M1 already has an executable object-level core:

- `ReducedCheckpoint` and `ReducedObject` form a canonical catalog after the
  AdaCheck-style detector. Detector candidates are still confirmed using role,
  dtype, shape and checksum, and `reconstruct_rank()` rebuilds the nested state
  required by each logical rank.
- `PeerDeltaProtectionPlan(availability_scope="host")` assigns each canonical
  object to one publisher per target host, pools equal objects across ranks and
  balances remote publisher bytes. It is therefore already object-aware and
  host-pooled, although host topology is currently restricted to disjoint host
  pairs.
- `commit_data_shard -> build_peer_delta_control_state -> commit_control ->
  restore_peer_delta_from_services -> reconstruct_rank` is a complete snapshot
  record path. Catalog and plan hashes are checked, incomplete controls are
  rejected, and a canonical object may not be duplicated within one target
  host.
- `test_peer_delta_runtime.py` already tests restore from either host and, with
  four loopback services, reconstruction after excluding each one of four
  logical hosts. The focused file currently passes `11/11` tests.

This is a useful recovery-closure mechanism, but its four-host test uses only
small model tensors and logical loopback services. It does not yet establish a
complete training-state checkpoint, a physical host failure, continued
training, or runtime performance.

### 71.2 Do not conflate the snapshot and online incremental paths

The live Merak path is still a whole-rank mirror-DP protocol:

- a service session requires a two-rank group;
- an incremental peer must have the same PP/TP coordinates, a different DP
  coordinate and the same optimizer-parameter topology;
- global commit requires reciprocal `{local_full, remote_delta}` rank evidence;
- the CPU materializer maintains one contiguous AdamW epoch chain per source
  rank and cannot consume arbitrary `value_id` placement.

Consequently, changing the rank-pair validator is not an implementation of M1.
An online object plan needs explicit object version, base/delta dependency,
publisher and same-epoch coverage semantics. Until those exist, the current
host-scope plan must be described as a full reduced-snapshot protection plane,
not AdaCheck-style one-checkpoint-per-step object materialization.

### 71.3 Minimal staged implementation

The next implementation should extend existing interfaces rather than introduce
a second planner hierarchy:

1. **M1-Snapshot.** Generalize `PeerDeltaProtectionPlan` from one implicit host
   partner to explicit allowed target domains, required object sets per domain
   and declared failure sets. Include these fields in `plan_hash`. Make control
   validation prove that every declared survivor set covers the complete
   required catalog. Retain the present service record format and final
   `ReducedCheckpoint.validate()` check.
2. **Complete-state closure test.** Extend the four-logical-host unit test with
   model, optimizer tensor/scalar state, scheduler, random-number-generator,
   sampler and progress state. Excluding each declared host must reproduce each
   rank's full checkpoint checksum; missing records and changed plan/catalog
   hashes must fail closed.
3. **M1-Online.** Only after the snapshot invariant is stable, define versioned
   canonical update objects and their dependency chains, then adapt the
   materializer and global commit protocol to consume an object plan. Validate
   fresh-process next-step equivalence before measuring performance.
4. **M2.** Keep a finite offline set of validated parallel plans. Regenerate the
   catalog and active profile for each plan, call the unchanged M1 solver, and
   compare joint selection with `training-only-best + best post-hoc M1`. M2 must
   not own checkpoint grouping or bypass its recovery proof.

Two-host work should concentrate on Steps 1--3, prediction accuracy, service
rate and recovery correctness. It cannot establish nontrivial host grouping.
Four physical hosts are required for the M1 host-matching headline and for a
meaningful M2 physical-domain comparison. This sequence keeps the research
claim aligned with executable recovery semantics rather than an offline byte
oracle.

## 72. Rank-independent object recovery decision (2026-09-13)

This section records the recovery-paper audit and the implementation decision
that follows Sections 70 and 71. It does not revise the three innovations in
Section 70. It replaces the assumption that one live peer rank must itself be a
complete recovery authority.

### 72.1 Literature result and the chosen boundary

AdaCheck supplies the redundancy detector and active encoder, but its public
artifact does not supply a complete changed-parallel-plan loader. GEMINI keeps
full local/remote rank replicas and normally restores the same rank identity;
it is useful for placement and traffic scheduling, not arbitrary resharding.
UCP and ByteCheckpoint provide the relevant separation: name checkpoint state
independently of rank, describe global tensor slices, and construct the target
rank's load plan at restore time. ByteRobust reinforces that a backup target is
a failure-domain storage destination rather than necessarily a semantic peer.
Oobleck and SWIFT require a substantially different training/replay design,
while ECCheck adds erasure coding that is not required for the first M1 gate.

Primary sources used for this decision:

- AdaCheck, FAST 2026: `https://www.usenix.org/system/files/fast26-liu-weijie.pdf`;
- GEMINI, SOSP 2023: `https://www.cs.rice.edu/~eugeneng/papers/SOSP23.pdf`;
- UCP, ATC 2025: `https://www.usenix.org/system/files/atc25-lian.pdf`;
- ByteCheckpoint, NSDI 2025: `https://www.usenix.org/system/files/nsdi25-wan-borui.pdf`;
- ByteRobust, SOSP 2025: `https://i.cs.hku.hk/~cwu/papers/brwan-sosp25.pdf`;
- Oobleck, SOSP 2023: `https://symbioticlab.org/publications/files/oobleck:sosp23/oobleck-sosp23.pdf`;
- SWIFT: `https://arxiv.org/abs/2302.06173`;
- ECCheck, ICDCS 2025: `https://i.cs.hku.hk/~cwu/papers/gcqi-icdcs25.pdf`.

The adopted system boundary is therefore:

```text
unchanged AdaCheck detector + unchanged SemanticFixed encoder
                              |
                              v
              versioned rank-independent object catalog
                              |
                  +-----------+-----------+
                  |                       |
       AdaCheckStyleFixed           M1Object placement
       fixed publisher/target       optimized publisher/target/schedule
                  |                       |
                  +-----------+-----------+
                              |
             common transport + sealed object commit
                              |
            failure-set validation + load-time reshard
                              |
        p/o/scheduler/RNG/sampler/progress restore oracle
```

`M1Object` decides who publishes each already encoded object, which physical
failure-domain service receives it, and how transfers are scheduled and load
balanced. It does not create redundancy, change AdaCheck's classification, or
change the encoded action set. A parallel-layout change invalidates the old
load plan but not the semantic object address; recovery constructs new target
slices from surviving object versions and then regenerates M1 for the new
layout.

### 72.2 Independent-module rule

The existing AdaCheck and peer implementations are frozen compatibility paths.
Do not implement the new recovery semantics by weakening checks in
`adacheck_semantic.py`, `merak_incremental.py`, `peer_delta_runtime.py`, the
current materializer, or the current global peer commit. Add a separately named
module, initially `Merak/core/checkpoint/object_recovery.py`, and a separate
runner/adapter. It may consume `ReducedCheckpoint`, frozen SemanticFixed plan
evidence, replica-service clients and materialized states through public
interfaces. Existing code remains a negative control and regression oracle.

The minimum address separates stable semantics from epoch content:

```text
semantic address = role + parameter FQN/slot + global shape + dtype
shard address    = semantic address + global offsets + lengths
object version   = shard address + epoch + checksum + base/delta dependencies
placement        = object version + publisher rank + target failure domain
```

Model state uses parameter FQN. Optimizer state uses parameter FQN plus slot,
not a process-local parameter number. Scheduler/progress may use global singleton
addresses. RNG and sampler state require an explicit logical stream/consumer
identity and transformation rule; if none exists for a target layout, recovery
must reject that layout rather than silently bind them to a new rank.

The first implementation exposes four conceptual operations: adapt an existing
materialized/reduced checkpoint plus an explicit layout descriptor into a
versioned object catalog; build either the fixed or M1 protection plan; commit
one sealed object epoch; and restore a requested target layout from surviving
services. Slice-overlap planning may fetch one target tensor from multiple
sources. Missing coverage, overlap conflict, stale epoch, changed catalog/plan,
checksum mismatch, undeclared failure set, or unsupported state transformation
must fail closed.

### 72.3 Fair comparison and metric attribution

`AdaCheckStyleFixed` and `M1Object` must share the detector artifact, semantic
encoder plan/hash, encoded object versions, transport implementation, service
limits, commit request stream, failure set, target layout, reconstruction code
and numerical oracle. Their only difference is the proof-carrying placement and
schedule. `NoCheckpoint` remains a training lower bound, not the baseline for an
M1 claim. The public AdaCheck artifact remains a reachability audit.

Every result reports non-overlapping stages:

```text
full logical p/o state
  -> AdaCheck encoded active objects       # detector/encoder attribution
  -> fixed or M1 cross-domain placement    # M1 attribution
  -> application payload and wire bytes    # transport attribution
```

The required experiment metrics are total committed cross-domain application
and wire bytes over all 16 ranks, per-epoch bottleneck edge time (`max_time`),
publisher/service load and tail, visible checkpoint frequency/RPO lag, exposed
training stall, total training time, and fresh-job numerical continuation.
Capture, enqueue, transfer, materialization, sealed visibility and final drain
remain separate intervals. A lower remote runtime-only stream is not a lower
total p/o checkpoint stream.

Keep historical and current policy vocabularies separate. Existing frozen
catalog artifacts retain their recorded `fixed`, `random-feasible`,
`redundancy_max` and `profile_exact` labels. New experiments use exactly
`AdaCheckStyle`, `GeminiStyleAdjacent`, `RandomFeasible` and `RedundancyMax`;
greedy remains retired and `profile_exact` is only a legacy diagnostic. The
current end-to-end experiment instead has `AdaCheckStyleFixed`,
`HostGlobalFixed` and `M1Object`. The first is the primary fixed-worker
baseline, the second holds host-global coalescing constant, and the third adds
M1 publisher/placement/load scheduling. A frozen policy label is not evidence
that its common-runtime arm ran.

### 72.4 Two-host theorem and honest opportunity gate

With two physical hosts, one pooled service on each host, and a contract that
either whole host may disappear, every unique object needed after failure must
be represented on both hosts. Once both arms use the same rank-independent
object representation and deduplicate once per host, the minimum cross-host
total bytes are invariant to host pairing. M1 can still choose among natural
publishers, balance NIC/CPU/service work, and reduce the maximum publisher or
critical-path time, but it must return `zero_opportunity` for total bytes when
the target closure is identical.

This theorem does not remove the system-level comparison with AdaCheck's fixed
worker groups. The reconstructed faithful baseline protects active objects
inside separate `k=2` groups; the proposed object/failure-domain grouping may
coalesce equal semantic objects at a host service and avoid duplicate copies
across those groups while preserving the same detector, active encoder and
host-failure recovery guarantee. A lower total transfer or physical footprint
against that baseline is attributable to object-level grouping plus
destination-aware deduplication, not to publisher load balancing and not to a
smaller AdaCheck logical checkpoint. It becomes an end-to-end M1 result only
when both paths use the same encoded request stream, transport accounting,
sealed commit and recovery oracle.

Accordingly every 16-rank result needs two controls:

1. `AdaCheckStyleFixed`: faithful fixed worker-group placement, used for the
   primary system comparison and allowed to expose duplicate protection;
2. `HostGlobalFixed`: the same rank-independent host object pool as M1 with a
   deterministic publisher, used to isolate the incremental benefit of M1's
   publisher and schedule optimization.

M1 may reduce total bytes relative to `AdaCheckStyleFixed`; under the two-host
lower bound it should have the same total bytes as `HostGlobalFixed` and improve
publisher/service load or `max_time`. A further total-byte difference versus
`HostGlobalFixed` requires a preregistered target-shard closure in which two
recovery-feasible plans genuinely protect different subsets of the same encoded
object set. Pairing unrelated PP stages is legal only after the new loader proves
all required global slices can be assembled; it is never accepted merely because
an offline byte oracle found overlap.

Thus the 16-rank gate can establish, in order:

1. same-layout and changed-layout object recovery independent of rank identity;
2. complete p/o/scheduler/RNG/sampler/progress closure and fresh continuation;
3. fixed/M1 equality of detector, encoder and recovery semantics;
4. total transfer/footprint improvement over faithful AdaCheck fixed groups by
   eliminating cross-group duplicate protection;
5. publisher-load and `max_time` improvement over HostGlobalFixed at the same
   two-host total-byte lower bound;
6. any further total transfer-size improvement over HostGlobalFixed only when a
   held-out target closure proves a real nonzero opportunity under the identical
   representation.

This narrows, rather than weakens, the paper story. Early PP-stage-crossing
motivation remains an observation that grouping changes exploitable overlap.
It is not end-to-end recovery evidence. The final M1 claim must come from this
common object runtime. M2 remains frozen until M1 is recovery-valid and a later
authorized multi-host run exposes multiple physical destination/group choices.

### 72.5 Execution gates

The current scope remains node17 and node8, 16 ranks only. The guarded session
is `ckpt-object-recovery-20260913` with `control`, `tests` and `runs` windows.
No 32-rank, four-node, EP, M2, 2x2 fault matrix, 50-step or 200-step treatment is
authorized at this stage.

The remaining implementation sequence is:

1. connect the checksum-confirmed active-gradient plan to the existing
   SemanticFixed trainer hook, transport, sealed commit and service payloads;
2. prove exact and multi-source reshard for model/optimizer slices plus strict
   handling of scheduler, RNG, sampler and progress metadata;
3. retain detector/semantic hashes, epoch/version identity, direct fallback and
   failure-closure checks through the persisted transport schema;
4. construct a held-out 16-rank opportunity report from a frozen real catalog;
5. only when the relevant opportunity is nonzero, run paired 10+5
   `AdaCheckStyleFixed`, `HostGlobalFixed` and `M1Object`; advance to 20+5 only
   when all three recover,
   all requested epochs seal, and measured direction agrees with the held-out
   prediction.

The currently indispensable evidence is r105/r106 and r108/r109/r110, their
detector/source artifacts, summaries and recovery oracles, plus the current
motivation figure inputs. Cleanup may remove large payloads from superseded
failed/debug runs only after preserving command, source hash, catalog/plan,
summary/analysis, relevant logs and an auditable deletion manifest.

### 72.6 ObjectRecovery implementation status and exact claim boundary

The independent recovery implementation now consists of six deliberately
small layers:

1. `object_recovery.py` defines stable semantic tensor addresses, versioned
   global-slice fragments, exact opaque records, target-layout requests,
   failure-set closure, sealed fixed/M1 load plans and CPU slice assembly.
2. `object_recovery_adapter.py` binds a live Merak module and its
   `HalfPrecisionOptimizer` to model tensors, FP32 optimizer masters and AdamW
   tensor slots. It also captures scheduler/RNG/sampler/progress as exact opaque
   records. The current live identity is parameter FQN plus role, global shape
   and dtype; source rank and storage host are placement metadata, not identity.
3. `object_protection_runtime.py` bridges an AdaCheck detector
   `ReducedCheckpoint` and a `PeerDeltaProtectionPlan` into the common object
   catalog. It verifies the catalog/plan control stored by the existing
   `ReplicaService`, rebuilds the object runtime using survivor payloads only,
   and reconstructs every rank through the unchanged reduced-checkpoint
   templates.
4. `object_recovery_controller.py` is an opt-in restore-only live controller.
   It validates the complete restored root before mutation, installs model,
   FP32 master and AdamW slot tensors, then restores scheduler, optimizer
   progress, RNG, sampler and iteration state. `MerakTrainer` only dispatches
   this path when both `CKPT_OBJECT_RECOVERY_FACTORY=module:function` and
   `CKPT_OBJECT_RESTORE_EPOCH` are explicit; it rejects simultaneous disk,
   legacy remote, incremental or CheckpointSync restore modes and synchronizes
   ranks after restore. A fresh Adam/AdamW optimizer may start without materialized
   slots: the adapter creates exactly `step`, `exp_avg` and `exp_avg_sq` after
   validating the complete restored tensor set. AMSGrad remains unsupported and
   fails closed.
5. `object_recovery_service.py` is the concrete service-backed factory for the
   same-logical-rank/same-layout path. It validates checksummed metadata and the
   exact protection plan, selects only surviving services and reconstructs the
   full reduced checkpoint before live apply. It supports deterministic Fixed
   and least-loaded M1 publisher policies, but not load-time resharding, and it
   currently fetches a full reduced checkpoint per fresh rank.
6. `object_semantic_plan.py` is a pure-CPU active BF16 gradient plan. Reuse
   requires the same detector class, FQN/global slice and PP/TP coordinate
   across distinct DP ranks. Its hashes bind detector provenance, semantic
   plan, epoch, object version and failure sets. Runtime checksum disagreement
   emits predeclared per-rank direct publications to both hosts. This proves
   fail-closed host-loss closure in tests, but is not connected to distributed
   checksum exchange, the trainer hook, transport or sealed service commit.

The executable data flow is:

```text
live rank-local p/o shards or detector ReducedCheckpoint
  -> CPU object fragments + exact opaque state
  -> fixed or M1 publisher/target-host placement
  -> sealed plans for no failure and each declared host loss
  -> survivor service fetch and checksum/plan verification
  -> CPU global-slice assembly for the requested target layout
  -> reconstructed rank dictionaries or copy into live model/optimizer tensors
```

Object payloads are currently cloned to CPU memory; replica services store
serialized CPU payloads. Restore allocates and assembles CPU tensors, then the
live adapter copies them into the destination tensors, which may be on GPU.
There is no GPU checkpoint cache or durable disk backend in this path.

The current two-host failure-domain mapping is explicit host indexing with
`ranks_per_host=8`: node17 owns global ranks 0--7 (`host=0`) and node8 owns
ranks 8--15 (`host=1`). The sealed contract covers `()`, `(0,)` and `(1,)`.
`required_value_ids_by_host` proves that the surviving host has every required
content class; exact opaque templates are logically replicated to both hosts.
This is host-failure closure, not a separate per-rank, GPU, NIC, rack or storage
failure model.

The M1 planning design connects at two points without changing AdaCheck
redundancy semantics; the active trainer path is not connected yet.
The protection plan changes publisher, target host, group and schedule for the
same detected/encoded objects; the M1 recovery selector then chooses among
equivalent surviving fragments using current publisher bytes, whereas Fixed
uses deterministic lowest-rank selection. `assert_fair_protected_ab` rejects an
A/B that changes detector catalog, availability scope, topology, target layout
or declared failure sets. A synthetic four-rank test has a strict M1 transfer
reduction and exact recovery after either host loss. A real TCP service test
verifies survivor-only restore and the sealed plan hash. The live-controller
tests additionally cover a fresh AdamW restore, complete runtime state and the
explicit trainer dispatch. The focused object/controller/peer regressions must
be rerun after each factory integration change; no GPU claim follows from these
CPU tests.

These tests do **not** yet prove real Merak end-to-end recovery. The live CPU
adapter has proved host loss plus TP2-to-TP1 reconstruction for model, FP32
master, Adam moments and scalar step, but the following remain open:

- the detector bridge still addresses epoch content classes by `value_id`,
  while the live adapter uses FQN/role; the two identities are not yet one
  persisted service schema;
- the service bridge currently reconstructs the original rank templates, so a
  changed PP layout is not yet executable;
- the live adapter accepts only contiguous TP slices and rejects tied
  parameters; its fresh-state bootstrap is restricted to non-AMSGrad Adam or
  AdamW with exactly the three standard tensor slots;
- the same-layout service factory exists and has survivor-only CPU coverage,
  but no active SemanticFixed run has produced its exact metadata/payload schema
  or invoked it from a fresh 16-rank job;
- the active-gradient plan remains coordinator-side CPU logic; distributed
  checksum exchange must retain payload ownership on publishers rather than
  gathering full tensors centrally;
- opaque state remains bound to its logical owner because no valid cross-rank
  transformation has been defined.

Therefore the correct current statement is: rank-independent tensor recovery,
CPU reshard, host-failure closure, M1 placement interfaces, a same-layout
service factory and a restore-only live apply controller are implemented;
active SemanticFixed transport integration, changed-layout service recovery,
fresh 16-rank ObjectRecovery and numerical continuation through this path are
not yet measured.
Existing `merak_incremental.py` fresh recovery remains a regression oracle, not
evidence for ObjectRecovery.

### 72.7 Held-out BF16 GPT-2 object-opportunity audit (r111--r115)

Three two-host, 16-rank, two-observation detector captures and one 10-step
stability capture were run in the guarded `ckpt-object-recovery-20260913` tmux
session. All completed with launcher status zero. They are metadata-only,
single-epoch full-state cost proxies: no
recoverable payload, service transfer, runtime failure, `max_time`, or fresh-job
continuation occurred.

| run | parallel placement | seed | full logical | reduced catalog | Fixed pair-local | profile-exact | result |
|---|---|---:|---:|---:|---:|---:|---|
| r111 | PP4/TP1/DP4, PP-first | 20260913 | 10,493,347,664 B | 2,611,789,316 B | 10,442,706,432 B | 10,442,706,432 B | 0% |
| r112 | PP4/TP1/DP4, PP-first | 20260910 | 10,493,347,664 B | 2,611,789,316 B | 10,442,706,432 B | 10,442,706,432 B | 0% |
| r113 | PP4/TP2/DP2, PP-first | 20260910 | 7,773,374,288 B | 3,862,563,332 B | 7,722,733,056 B | 7,722,733,056 B | 0% |
| r115 | PP4/TP1/DP4, PP-first | 20260913 | 10,493,347,664 B | 2,611,789,316 B | 10,442,706,432 B | 10,442,706,432 B | 0% |

For r111/r112 every formal policy has the same provisional makespan proxy
`1.883247872 s`; for r113 every policy has `1.354364160 s`. These values use
assumed 7 GB/s directional links and 3 GB/s services, so they are opportunity
proxies, not measured `max_time`. Repeating PP4/TP1/DP4 with a different seed did
not change the zero result. In r113, Fixed, redundancy-max and profile-exact all
select corresponding cross-host ranks `(0,8)..(7,15)`; random changes pairs but
not total pair-local bytes or the provisional critical path.

The host-global lower-bound diagnostics are 2,610,737,664 B for r111/r112/r115
and 3,861,511,680 B for r113. The gap from pair-local Fixed is a candidate
system-level benefit of replacing AdaCheck-style fixed worker groups with
object/failure-domain grouping and host-destination deduplication. It is not a
pure publisher-selection benefit, and the current metadata-only run cannot make
the end-to-end claim. The analyzer therefore also adds a stricter host-global
Fixed/M1 diagnostic:
both arms use the same catalog, host scope, required objects, failure sets and
total bytes. Fixed selects the first eligible publisher; M1 selects the
least-loaded eligible publisher. On r111/r112/r115, both transfer
`2,610,737,664 B`, while maximum publisher load is predicted to fall from
`1,034,412,000 B` to `258,607,584 B` (`74.9996%`). On r113 both transfer
`3,861,511,680 B`, while the maximum falls from `826,640,352 B` to
`413,332,704 B` (`49.9985%`). The equality of total bytes is expected from the
two-host theorem; the nonzero result is a load/makespan opportunity, not a byte
reduction.

r115 trained for ten steps and repeated detector observations at steps 9 and
10. It freezes 747 objects, 699 stable components and 2,844 stable records,
with 780 singleton fallbacks; the two confirmations are identical. This makes
the 75% publisher-load opportunity stable near the end of the short training
window, but `eligible_for_runtime_gate=false` and
`adacheck_payload_faithful=false` remain explicit because no active encoded
payload was captured. The next valid objective is to connect the active
semantic plan to the common trainer/transport/commit path, then perform a
same-layout service-backed fresh ObjectRecovery continuation, followed by an
actual same-runtime host-global Fixed/M1 measurement of bottleneck edge time
and exposed training stall at equal total bytes. In the eventual three-arm common-runtime experiment,
`AdaCheckStyleFixed -> M1Object` measures the whole object-grouping system benefit,
while `HostGlobalFixed -> M1Object` isolates publisher/load optimization. Any
claim that scheduling alone lowers total bytes remains blocked until a held-out
target-layout closure proves two recovery-feasible plans consume different
subsets of the same encoded object representation.

No r115 value belongs in a formal M1 result figure: its `75%` is predicted
publisher load, not measured transfer time. A stage-result figure is promoted
only after compared arms use the common runtime and report sealed transfer plus
recovery evidence. The first such composite should include all three arms'
logical/application/wire bytes, bottleneck edge time, publisher/service load,
cadence, training wall/stall and fresh-recovery verdict; negative gates remain
visible.

### 72.8 2026-09-13 cleanup and preservation record

Disk cleanup used `tools/cleanup_research_artifacts.py` and is audited at
`runs/cleanup-20260913T081942Z/manifest.json`. It removed the failed r114
launcher-contract diagnostic, superseded by complete r115, and the obsolete
`runs/motivation-two-node-16r/preliminary-20260910-node17-node8` tree,
superseded by the corrected motivation evidence. The manifest preserves paths,
sizes, mtimes and SHA-256 digests for all 774 deleted files; small audit copies
are retained. Total logical content removed was 62,743,076 B. Corrected
motivation inputs, r105/r106/r108/r109/r110, r111--r113/r115 and all current
paper figures remain. Large deleted payloads are recoverable only by rerunning
their recorded commands; do not delete further result trees merely because
`runs/` remains large.

### 72.9 r116 common-runtime CPU/TCP correctness gate

The active SemanticFixed path keeps three system/implementation layers explicit:
detector-bound grouping, ObjectRecovery, and M1's fixed-`P` object publisher,
destination failure-domain and load objective. These are not three optimizer
modules: the paper has M1 and M2, and only M2 selects the parallel plan and
logical-to-physical host mapping. The layers are
implemented by the following independent files without modifying the legacy
AdaCheck implementation:

1. `object_detector_mapping.py` maps only frozen AdaCheck optimizer
   `/str:optimizer/str:fp16_groups/int:G/int:I` requirements to live parameter
   slots. It recomputes the frozen semantic catalog hash exactly, treats
   detector classes only as equality candidates, uses stable rank-local classes
   for missing slots, and rejects non-contiguous TP slices. The r115 production
   catalog validates unchanged: 747 objects, 596 optimizer slots and catalog
   hash `fdf6c704d4f11680cbb920c98207cdcd67f91224dd3751781e31f9f23fa66e60`.
2. `object_semantic_plan.py` builds the three arms
   `adacheck_fixed`, `host_global_fixed` and `m1` from the same active gradient
   objects and failure closure. Per-step equality is accepted only after
   checksum confirmation; mismatch creates rank-specific versions on both
   hosts.
3. `object_optimizer_update.py` replaces standard optimizer-update gradients
   by semantic version references. A rank keeps tensor payloads only for
   versions that the plan assigns it to publish; materialization recreates the
   unchanged `apply_optimizer_update` record and has exact AdamW replay tests.
4. `object_update_transport.py` writes one rank template bundle to each target
   host and includes gradient tensors only at planned publisher occurrences.
   AdaCheck pair-local occurrences remain distinct, so the common transport
   does not silently give the baseline host-global deduplication. Data and a
   self-checksummed control are committed to `ObjectReplicaServiceStore`, the
   control is sealed first, and every data epoch is then sealed against that
   exact checksum and manifest evidence. The external locator contains no
   checkpoint tensor; it is the authority needed for fresh restore.

`ObjectReplicaServiceStore` is a separate subclass. It retains authorization
per `(kind, source_rank, epoch)` rather than only the newest source epoch, so
sealing epoch 2 does not make epoch 1 unreadable. Sealed history is protected
from eviction and capacity exhaustion fails explicitly. Compaction and unpin of
obsolete suffixes remain future work; do not use unlimited append-only runs.

r116 is at
`runs/two-node-v2/object-runtime-cpu-gate-r116-20260913`. It is a controlled
four-rank, two-host, two-object CPU/loopback-TCP gate, not a GPU model result.
All three arms restore epochs 1 and 2 after actual loss of either service, and
the reconstructed rank updates replay AdamW exactly. Runtime checksum mismatch
also passes through rank-specific direct fallback. Focused plus legacy object,
peer and incremental regressions total 98 passes before the expanded transport
matrix; the transport matrix itself is nine passes.

The r116 attribution is deliberately small but honest:

| arm | remote gradient logical | remote data payload | remote control payload | total remote payload | max publisher logical |
|---|---:|---:|---:|---:|---:|
| AdaCheckStyleFixed | 32 B | 8,730 B | 2,364 B | 11,094 B | 16 B |
| HostGlobalFixed | 16 B | 8,300 B | 2,370 B | 10,670 B | 16 B |
| M1Object | 16 B | 8,384 B | 2,340 B | 10,724 B | 8 B |

M1 therefore removes the pair-local duplicate gradient occurrences relative to
AdaCheckStyleFixed and halves the bottleneck publisher load relative to
HostGlobalFixed in this controlled case. It does **not** beat HostGlobalFixed
total serialized payload: splitting two tiny tensors across two publishers adds
54 B of serialization framing. This is expected to become negligible for large
tensors but must be measured, not assumed. The generated
`docs/figures/object_runtime_cpu_tcp_gate_r116.{png,svg}` is explicitly labelled
CPU/TCP correctness and attribution; it is excluded from formal end-to-end M1
performance claims.

The r117 CPU/Gloo gate at
`runs/two-node-v2/object-semantic-gloo-gate-r117-20260913` now closes the first
part of distributed integration. A dedicated world-size Gloo group gathers
only JSON-like rank/shard metadata at setup and rank/object/checksum records per
step; local CPU or CUDA tensors are never collective payloads. Six four-process
cases cover all three arms crossed with equality-confirmed and mismatch-direct-
fallback encodings. Every rank obtains identical plan, canonical and encoding
hashes, and bundle tensors remain only on planned publishers. The API accepts
an explicit Gloo group because Merak's default world group is NCCL and rejects
subgroups that could produce inconsistent PP/TP-local plans.

The remaining gate is no longer detector identity, checksum consensus or sealed
suffix recovery. It is production runtime integration: connect the
pre-optimizer-step raw BF16/FP16 snapshot lifetime to the background publisher,
combine a sealed base with the object suffix for fresh restore, then use the
same three-arm runner for a 16-rank 10+5 continuation on node17/node8. No M1
end-to-end positive claim, `max_time` claim, non-blocking claim, 20/50/200-step
 run, 32-rank run or M2 execution is authorized by r116/r117.

### 72.10 r118 distributed object commit and sealed encoding gate

`object_update_distributed.py` now executes the suffix protocol from every
rank. Each rank writes only its two local rank-template bundles; planned
publisher occurrences are the only tensor payloads. A world-size Gloo group
gathers JSON manifest evidence and metrics, rank zero publishes the identical
control to both hosts, both services seal control before data, and an external
locator is written only after all ranks and seals succeed. The control embeds
tensor-free holder checksum evidence, so a fresh process reconstructs the exact
confirmed/mismatch encoding rather than trusting a caller-supplied in-memory
encoding. No collective carries a checkpoint tensor.

r118 is recorded at
`runs/two-node-v2/object-distributed-commit-gate-r118-20260913`; its generated
JUnit reports 18 passes. It covers all three arms, distributed plan/encoding/
locator hash agreement, two M1 epochs, restore from either surviving host,
sealed encoding reconstruction, mismatch fallback, locator tamper rejection
and exact AdamW replay. The broader detector/object/incremental/peer regression
is 129 passes. The combined evidence figure is
`docs/figures/object_runtime_cpu_distributed_gate_r118.{png,svg}` and remains a
controlled four-rank CPU/TCP/Gloo correctness figure.

The implementation boundary remains three layers, not one module per Python
file: detector-bound grouping, ObjectRecovery, and M1 fixed-`P` publisher,
destination failure-domain and load optimization. M1 does not choose the
parallel plan or logical-to-physical placement; M2 alone owns that search. M1
continues to share the exact detector, encoder, transport,
failure sets and recovery rules with both fixed controls. r118 closes the
distributed suffix commit blocker but does not include a sealed base, Trainer
callback, fresh-process training continuation, GPU timing, or exposed-stall
measurement. Those gates remain mandatory before node17/node8 16-rank 10+5;
32-rank/M2 and 20/50/200-step runs remain frozen.

### 72.11 r119 fresh base-plus-suffix ObjectRecovery gate

The production recovery boundary is now exercised by
`object_semantic_restore.py`, which is separate from the legacy AdaCheck,
incremental and peer restore implementations. A fresh controller rebuilds the
same frozen detector binding and semantic plan, fetches a sealed epoch-zero
base plus a contiguous sequence of object-update suffixes from one surviving
service, and applies model, fresh AdamW, scheduler, RNG, sampler and iteration
state for one logical rank. Suffix restore is rank selective and fetches the
control once, then only the target template and actual publisher bundles needed
by that target rank. This path is currently same-layout only: changed
PP/TP/DP layouts fail closed rather than pretending that load-time resharding
is implemented.

r119 is retained at
`runs/two-node-v2/object-semantic-composite-gate-r119-20260913`. Its JUnit has
six passes: all three common-runtime arms crossed with loss of either one of
the two service hosts. Every case commits a real sealed base and two suffixes,
constructs a fresh model/optimizer/scheduler/runtime, and exactly reproduces
the controlled AdamW continuation. Recovery metrics are written atomically
per logical rank and include base GET bytes/time and total restore wall time.
Boundary hashes remain diagnostic only because an actual CPU/GPU replay need
only satisfy the registered numerical oracle, not bitwise equality.

This closes the fresh composite correctness blocker, not the end-to-end GPU
claim. A 16-rank trainer run must additionally show a complete 10-step source,
fresh 5-step continuation, an independent uninterrupted constant-LR reference,
and a loss/state trajectory oracle before it can be called reliable training
recovery.

### 72.12 r120 16-rank service-scale attribution gate

r120 is retained at
`runs/two-node-v2/object-semantic-service-scale-gate-r120-20260913`. It is a
16-process CPU/Gloo/loopback-TCP scale gate with ten suffix epochs, all three
arms and both single-service-host losses. The synthetic layout mirrors the
r115 `PP4/TP1/DP4`, PP-first holder geometry but uses four tiny stage-local
parameters per PP stage. Every target's observed GET source set equals its
rank template plus the publishers selected by the sealed plan, and AdamW,
scheduler and runtime recovery pass after either host is removed.

Across ten suffixes, measured remote gradient logical bytes are
`5,120/1,280/1,280 B` for AdaCheckStyleFixed, HostGlobalFixed and M1Object.
Application bytes are `3,842,684/3,754,424/3,759,724 B`; wire bytes are
`4,608,056/4,292,292/4,297,586 B`. Thus M1 removes 75% of logical duplicate
gradient occurrences relative to pair-local AdaCheck fixed, while application
and loopback wire reductions are only about 2.16% and 6.74% because tiny-object
metadata and framing dominate. M1 does not lower total logical bytes relative
to HostGlobalFixed. Its extra `5,300 B` application and `5,294 B` wire are the
cost of distributing these tiny payloads among more publisher bundles.

The publisher objective is distinct from total bytes. HostGlobalFixed's
planned maximum publisher load is 32 B per suffix and M1's is 8 B, a 75%
reduction at equal total logical bytes. The corresponding two loopback
replicates report maximum PUT times `57.213/60.775 ms` versus
`39.169/43.312 ms`; this is diagnostic timing, not GPU/NIC `max_time` evidence.
Service retained bytes are approximately 4.20/4.11/4.11 MB per host for the
three arms.

The recovery-side tradeoff is explicit. AdaCheckStyleFixed and
HostGlobalFixed each issue 440 suffix GETs across all target ranks and ten
epochs. M1 issues 620 because reconstructing one rank may need distinct
publisher bundles after load balancing; corresponding suffix application
traffic is about 8.53/8.45/12.50 MB. The current M1 objective optimizes
checkpoint publication, not recovery read amplification. A future objective
may include recovery GET cost, but r120 must not be retroactively tuned.

The active online suffix encodes confirmed BF16 gradients plus runtime state
and reconstructs optimizer state through the shared AdamW transition. These
bytes are therefore recovery-update payload, not a literal static `p`-only or
`o`-only checkpoint. AdaCheck detector reduction is held common and is never
counted again as M1 benefit.

The next authorized gate remains node17/node8 only: GPT-2 r115 geometry,
16 ranks, a 10-step source and fresh continuation under the common detector,
encoder, transport, service and failure contract. Before a formal M1 result,
the run set needs NoCheckpoint, AdaCheckStyleFixed, HostGlobalFixed and
M1Object under constant LR, with total checkpoint transfer, bottleneck PUT,
cadence/lag, total training time, exposed stall and a continuation oracle.
M2, 32 ranks, four hosts and 20/50/200-step treatments remain frozen.

### 72.13 r125--r131 live 16-rank GPT-2 common-runtime result

The first minimal fixed-layout common-runtime M1 transfer-and-continuation gate
is complete under the fixed node17/node8
contract: GPT-2 (163.171M parameters), BF16, `PP4/TP1/DP4`, `pp-dp`, seq128,
batch one, GA1, scalar AdamW, constant LR and seed 20260913. r125 is the
uninterrupted 15-step NoCheckpoint reference. r127, r129 and r130 are the
M1Object, AdaCheckStyleFixed and HostGlobalFixed runs respectively; each trains
steps 1--10, stops the node8 checkpoint service, constructs a fresh 16-rank job
against node17 alone, restores epoch 10 and continues through step 15. r131 is
the fail-closed four-arm summary at
`runs/two-node-v2/object-semantic-gpt2-four-arm-r131-20260913/analysis.json`.

All three checkpoint arms exactly reproduce the reference loss at every step.
Their 16 final JSON digests are also byte-identical to r125 rank by rank,
covering model, scalar AdamW optimizer, scheduler and iteration/runtime state.
This is a real Trainer plus GPU continuation and surviving-service recovery
result. It is not physical compute-host loss: both hosts and all 16 compute
ranks are relaunched for restore while only node8's checkpoint service is
unavailable. Recovery is same-layout only; changed-layout ObjectRecovery still
fails closed.

| arm | base application | 10-suffix logical | 10-suffix application | 10-suffix wire | suffix max PUT | total application |
|---|---:|---:|---:|---:|---:|---:|
| AdaCheckStyleFixed | 9,188,627,804 B | 13,053,665,280 B | 13,060,178,100 B | 13,063,583,937 B | 1,887.923 ms | 22,248,805,904 B |
| HostGlobalFixed | 9,188,627,903 B | 3,263,416,320 B | 3,268,883,820 B | 3,270,083,147 B | 515.256 ms | 12,457,511,723 B |
| M1Object | 9,188,627,804 B | 3,263,416,320 B | 3,268,889,740 B | 3,270,089,454 B | 425.787 ms | 12,457,517,544 B |

Thus M1Object versus AdaCheckStyleFixed reduces suffix logical bytes by exactly
75%, suffix application/wire bytes by 74.971%/74.968%, and base-plus-suffix
application bytes by 44.008%. This is the online counterpart of the r120
attribution: the detector and active-gradient encoder are common, and grouping
avoids four pair-local publications of each equality-confirmed version. It does
not create redundancy. HostGlobalFixed and M1 have equal suffix logical bytes;
M1's 5,920 B application excess is bundle/control framing. Their comparison
isolates publisher placement/load: the observed suffix max PUT falls from
515.256 to 425.787 ms (17.364%), while M1 versus AdaCheck falls 77.447%. These
are one ordered run per arm, so timing changes are promising measurements, not
distributional performance claims.

Cadence and blocking remain negative. Every checkpoint arm has
`maximum_boundary_lag=10` and `strict_1c1s_valid=false`. Source-phase walls are
103.998/93.796/83.801 s and source-step means are
0.494/0.510/0.462 s for AdaCheck/HostGlobal/M1, while r125's differently scoped
15-step reference phase is 29.150 s and step mean is 0.263 s. The phases include
model construction; checkpoint source additionally includes a synchronized
9.189 GB base and final drain, so phase ratios are not a clean training-overhead
claim. Per-rank maximum restore-controller walls are 29.263/22.198/24.339 s;
the full restore phases, including fresh construction and five continued steps,
are 64.464/59.268/54.001 s. M1 therefore has no recovery-RTO advantage over
HostGlobalFixed in this sample, consistent with r120's recovery-read tradeoff.

The lag is caused by one FIFO publisher thread per rank. Ten training steps take
only about 4.7 s, but r127 reaches 24.297 s enqueue-to-commit tail, including a
21.729 s publisher-queue tail, and finalization waits for the queue. Base transfer
is a separate one-time cost (9.189 GB, max PUT
8.706--9.975 s). Suffix max PUT is only 0.426 s for M1; the remaining roughly
2.3--3.6 s per epoch combines CPU encoding/serialization and Gloo object
collectives. Lowering max inflight merely moves this wait into training. Before
changing concurrency, add shared per-stage timing for checksum, externalize,
bundle, collective, control, seal and final drain, then apply any throughput
change identically to all three arms.

r124 (missing runtime-only environment) and r128 (preflight matched the outer
tmux-launch command) are explicitly invalidated pre-training attempts. r126 is a
retained failed development run that exposed Merak's TP1 `model_parallel=True`
marker; `_global_slice` now treats validated `tensor_world_size==1` as an
unsharded full tensor while TP>1 still requires explicit slice metadata. No
32-rank, M2, 20/50/200-step or 2x2 fault experiment was started. The next gate
is shared-runtime profiling and removal of the measured suffix throughput tail,
then a repeated GPT-2 comparison or a second authorized 16-rank dense model.
The post-gate r132 CPU regression records 98 passes and zero failures.

### 72.14 r133 shared-runtime stage profile

r133 is a source-only M1 diagnostic at
`runs/two-node-v2/object-semantic-m1-profile-r133-20260913`; it repeats the
r127 GPT-2, node17/node8, 16-rank `PP4/TP1/DP4` source workload with the new
observation-only timing fields and deliberately skips restore. It is not a new
treatment result. Across 16 ranks and ten suffix epochs, mean capture/checksum/
externalize/distributed-commit times are 73.6/350.9/197.5/2203.0 ms. Within
distributed commit, mean local-bundle/manifest-gather/result-broadcast times
are 402.5/1582.8/203.7 ms. The maximum enqueue-to-commit and publisher-queue
tails are 24.025 and 21.217 s, and final drain is about 23.80 s per rank.

Rank 12 is the local-bundle critical rank in every epoch: it serially commits
about 77.36 MB to node17's service and 129.34 MB to node8's service, taking up
to 2.005 s, while other ranks wait in the Gloo manifest gather. This supports
one minimal common-runtime change: issue the two independent failure-domain
bundle commits concurrently with the same immutable snapshot, then retain the
same gather, control-first seal, locator and recovery validation. The change
must apply identically to AdaCheckStyleFixed, HostGlobalFixed and M1Object. A
source-only post-change run is diagnostic; the common-runtime comparison still
requires repeated three-arm recovery-valid runs before a non-blocking or timing
claim. No M2, 32-rank or longer run is authorized by this profile.

### 72.15 r134--r139 parallel-PUT gate

`object_update_distributed.py` now submits each rank's two immutable suffix
bundles to the two independent checkpoint services with a two-worker standard-
library executor. Bundle content, source publisher, host targets, manifest
gather, control-first seal, locator, failure closure and ObjectRecovery are
unchanged. All three checkpoint arms use this shared runtime. If one request
fails, no control or seal is published; the other service may retain an
unreachable unsealed record, as was already possible after a later failure in
the serial protocol. r134 exercises the focused object-semantic regression:
pytest reports 99 passes. Its outer zsh wrapper then used the reserved variable
`status` and exited 1; the retained status file records pytest exit 0 and
wrapper exit 1.

r135 is an M1-only source diagnostic comparing the new overlap with r133. It
keeps all ten suffix logical bytes exactly equal at 3,263,416,320 B and all 16
source final digests byte-identical. Across 160 rank-epochs, mean distributed
commit falls from 2203.0 to 1591.1 ms; mean local bundle and manifest gather
fall from 402.5/1582.8 to 326.6/1070.9 ms. Maximum enqueue-to-commit,
publisher-queue and final-drain tails fall from 24.025/21.217/23.814 s to
17.249/15.150/17.120 s. Source phase falls from 89.344 to 73.977 s. Maximum
remote PUT rises from 471.384 to 556.007 ms, so the admissible conclusion is
better drain/throughput with a single-PUT contention tradeoff, not better
`max_time`. The diagnostic figure is
`docs/figures/object_semantic_m1_put_parallelism_r135.{png,svg}`.

r136 M1Object, r137 AdaCheckStyleFixed and r138 HostGlobalFixed then repeat the
complete r125 workload with the shared parallel-PUT runtime. Each performs ten
source steps, loses node8's checkpoint service, restores epoch 10 from node17
in a fresh 16-rank same-layout process and continues through step 15. r139
recomputes every arm from raw artifacts and is `valid=true`: every source and
restore loss equals r125, every final model/optimizer/scheduler/runtime digest
matches exactly, and all arms use detector SHA
`26dfb1ee7daa3869eb4c36507f652b216fc2d2f9753f8b9f28026e6ffb2c061d`.

| arm | 10-suffix logical | 10-suffix application | suffix max PUT | source phase | max restore-controller wall | lag |
|---|---:|---:|---:|---:|---:|---:|
| AdaCheckStyleFixed | 13,053,665,280 B | 13,060,178,170 B | 2,466.345 ms | 93.700 s | 31.229 s | 10 |
| HostGlobalFixed | 3,263,416,320 B | 3,268,883,890 B | 448.136 ms | 79.043 s | 22.221 s | 9 |
| M1Object | 3,263,416,320 B | 3,268,889,840 B | 430.046 ms | 73.825 s | 22.192 s | 9 |

M1 versus AdaCheck therefore retains exactly 75% suffix logical reduction,
74.971% suffix application reduction and 44.008% base-plus-suffix application
reduction. Under this common runtime its measured suffix max PUT is 82.563%
lower and source phase is 21.211% shorter. HostGlobal-to-M1 isolates the current
remote publisher objective: suffix logical bytes are equal, serialized framing
differs by 5,950 B, suffix max PUT is only 4.037% lower and source phase 6.601%
shorter. One ordered run cannot establish stable load-balancing performance;
the max recovery-controller wall is effectively equal. Every checkpoint arm
still requires final drain and fails strict 1C1S. The r139 candidate figure is
`docs/figures/object_semantic_gpt2_parallel_put_four_arm_r139.{png,svg}`.

The next M1-only design gate keeps `P` and the physical rank-to-host mapping
fixed but accounts for every object publication to either service when choosing
a publisher. The current plan balances only remote placements and always picks
the minimum local holder, which makes rank 12 the local-bundle critical path for
the heavy final pipeline stage. Extending the publisher-load objective to both
local and remote service publications remains M1; changing the parallel plan or
physical mapping remains M2. Preserve remote byte accounting as a separate
metric and prove unchanged object/failure/recovery contracts before another GPU
run. No 32-rank, M2, physical compute-host fault or longer run has started.

### 72.16 r140--r148 all-publication M1 load gate

M1 now minimizes the maximum accumulated publisher bytes over every planned
publication, including both the local and remote checkpoint services. It still
prefers an object holder local to the destination service whenever one exists,
so the objective cannot introduce extra cross-host transfer. The parallel plan
`P`, logical-to-physical rank mapping, detector, confirmed object versions,
failure sets, encoder, transport, sealed commit and ObjectRecovery are unchanged.
This is a fixed-`P` M1 publisher/schedule decision; choosing `P` or the physical
mapping remains the frozen M2 problem. AdaCheckStyleFixed and HostGlobalFixed
are semantically unchanged. r140 records the complete focused CPU regression:
`99 passed`, zero failures.

r141 is the recovery-valid live M1 run at
`runs/two-node-v2/object-semantic-m1-all-publication-r141-20260913`. It repeats
the node17/node8, 16-rank GPT-2 BF16 `PP4/TP1/DP4, pp-dp` contract, commits a
synchronized base and ten suffixes, removes node8's checkpoint service, restores
epoch 10 from node17 in a fresh same-layout process, and continues steps 11--15.
Losses and all 16 final state digests exactly match r125. r142 recomputes the
four-arm comparison from r125/r137/r138/r141 and is `valid=true`. M1 versus
AdaCheckStyleFixed retains the exact 75% ten-suffix logical reduction and
44.008% base-plus-suffix application-byte reduction. This is grouping/object
publication reuse of detector-confirmed redundancy, not redundancy created by
M1. HostGlobalFixed and M1 retain equal suffix logical bytes and therefore form
the load-only ablation.

For one suffix epoch, HostGlobalFixed and M1 both publish 652,683,264 B in total,
of which 326,341,632 B is remote. HostGlobalFixed concentrates publication on
ranks 0/4/8/12 and reaches 258,607,104 B on rank 12. M1 distributes the same
logical total over all 16 ranks and caps the largest planned publisher at
77,328,384 B, a 70.098% reduction. The single-run r144 figure is retained as a
planning/mechanism diagnostic; stable timing is taken only from r148.

r148 is the fail-closed analysis of three alternating HostGlobal/M1 source runs:
r138/r141, r143/r145 and r146/r147. Each run, not each rank-epoch, is one repeat.
All six runs share the exact detector, workload, ten-step loss trace, 16 final
digests and 3,263,416,320 B ten-suffix remote logical total. r138 predates only
the new plan-load metric fields; its HostGlobal algorithm and shared transport
are unchanged. One HostGlobal and one M1 sample are recovery-valid (r138/r141);
the other four repeats are source-only timing samples.

| three-run mean metric | HostGlobalFixed | M1Object | M1 reduction |
|---|---:|---:|---:|
| planned maximum publisher bytes | 258,607,104 B | 77,328,384 B | 70.098% |
| distributed-commit mean | 1,860.030 ms | 1,184.506 ms | 36.318% |
| distributed-commit p95 | 2,397.993 ms | 1,438.931 ms | 39.994% |
| maximum publisher queue | 17,470.494 ms | 11,947.193 ms | 31.615% |
| maximum final drain | 19,358.612 ms | 13,441.420 ms | 30.566% |
| source phase, base + 10 steps + drain | 80.843 s | 74.171 s | 8.252% |
| suffix maximum remote PUT | 515.698 ms | 512.000 ms | 0.717% |
| base-dominated overall maximum PUT | 9,015.187 ms | 8,512.219 ms | 5.579% |

The suffix max-PUT ranges overlap (`448.136--592.925 ms` HostGlobal versus
`474.960--538.804 ms` M1), so r148 does not establish a stable per-request
`max_time` advantage over HostGlobal. It does establish a deterministic plan
load reduction and a consistent reduction in collective/queue/drain critical
paths in these three runs. Spreading work raises mean local-bundle time from
255.182 to 419.275 ms because more ranks publish, while lowering its p95/max;
this mechanism tradeoff must remain visible. Against the primary AdaCheck fixed
baseline, r142 still shows a much larger suffix max PUT reduction, but it is a
single recovery-valid comparison. All checkpoint runs rely on final drain,
have lag nine or ten, and fail strict 1C1S; neither stable non-blocking nor an
RTO advantage is claimed. The repeat analysis is
`runs/two-node-v2/object-semantic-m1-load-repeats-r148-20260913/analysis.json`.

### 72.17 r150 second fixed-P detector input

r150 creates the missing metadata-only detector input for a second authorized
16-rank layout on node17/node8: GPT-2 BF16 `PP2/TP1/DP8, pp-dp`, two training
steps and world-scope detector comparison. Both distributed launchers exit zero
and both captures contain every rank record. The frozen catalog contains 747
objects, 699 stable components and 5,624 stable records; its hash is
`f9bf738cefc8f9f8f0ac52c72df3facbf9956f3a331cb4a4a2313ad2062be224`.
The full/reduced catalog sizes `20,986,613,920/2,611,789,316 B` are detector
metadata opportunity, not online transfer measurements, and recoverable payload
is deliberately absent. The artifact occupies about 69 MiB.

In `pp-dp` rank order, ranks 0--7 are PP0/DP0--7 on node17 and ranks 8--15 are
PP1/DP0--7 on node8. AdaCheck-style fixed pairs `(i, i+8)` therefore cross PP
stages, but they do not merge stage parameters: semantic object identity includes
FQN plus PP/TP coordinates, each rank publishes only its own held objects, and
same-layout recovery validates the complete topology and target logical rank.
Changed-layout recovery still fails closed. Before any result claim, the common
launcher/analyzers must accept run-contract-driven layout and step counts, then
a 3+5 NoCheckpoint/AdaCheck/HostGlobal/M1 gate must pass for both possible
single checkpoint-service failures. Only then may this layout advance to 10+5.
No 32-rank, M2, physical compute-host failure or longer run is authorized here.

### 72.18 PP2 launcher hardening, same-layout guard and 3+5 dual-service gate

The common launcher and gate/suite analyzers now derive `PP`, `TP`, `DP`, mesh
order, source length, continuation length and failed checkpoint-service side
from the sealed run contract. They retain the original `PP4/TP1/DP4` defaults
and fail closed on mismatched detector topology, capacity declaration, step
range, failure side, loss trace or final digest. `SameLayoutServiceMetadata`
also now requires `source_layout_hash == target_layout_hash` before mutating the
model. This closes an admission bug; it does not implement changed-layout
resharding. r161 records the focused regression: `125 passed`, one dependency
warning and zero failures.

r152 was a recovery-valid development pilot, but its operator retention estimate
`24,915,026,588 B` was below the measured service bytes
`26,212,278,369 B`. It is marked `EXCLUDED` and never enters comparison or
figures. Its replacements use a 30 GB estimate with a 32 GiB declared minimum
for the 3+5 gate; configured service capacity remains 160 GiB.

r151 is the uninterrupted eight-step reference. r153/r154/r155 and r156 form a
valid AdaCheckStyleFixed/HostGlobalFixed/M1 suite with node8's checkpoint service
unavailable. r157/r158/r159 and r160 repeat the same contract with node17's
checkpoint service unavailable. Both use node17/node8, 16-rank GPT-2 BF16,
`PP2/TP1/DP8, pp-dp`, three source steps, final drain, fresh same-layout restore
from epoch 3 and continuation through step 8. All three arms at both failure
sides exactly match r151's source/restore losses and all 16 final full-state
digests. Both suites use the same detector hash
`eb8846bf0702df68835861032edfaa0d8178f3b0e2945113e44757b35a40c980`.

Across three suffixes, Ada/Host/M1 remote gradient logical bytes are
`7,832,199,168/979,024,896/979,024,896 B`; M1 therefore reduces suffix logical
transfer by exactly 87.5% relative to AdaCheckStyleFixed while remaining equal
to HostGlobalFixed. Including the common approximately 18.377 GB base, total
application payload is approximately `26.212/19.358/19.358 GB`, a 26.148% M1
reduction relative to Ada. The active suffix is confirmed BF16 gradients plus
runtime control for AdamW replay, not a literal `p` or `o` snapshot. M1 does not
create redundancy: Ada-to-M1 measures duplicate publications avoided by the
grouping/object-publication policy using the same detector; Host-to-M1 remains
the equal-byte load ablation.

Suffix max PUT for failed service node8 is
`4700.392/761.025/552.783 ms`; for failed service node17 it is
`5743.076/707.275/546.609 ms`. Each cell is one timing sample, so these are
diagnostics rather than statistical max-time claims. All arms have lag three,
require final drain and fail strict 1C1S. Each injected failure removes one
checkpoint service while both compute hosts and all ranks are relaunched and
available; this is not simultaneous two-service failure or physical compute-host
loss. The figure is
`docs/figures/object_semantic_pp2_dual_service_loss_r160.{png,svg}`.

### 72.19 r162--r166 PP2 10+5 recovery-valid M1 gate

After both 3+5 failure sides passed, r162 records the uninterrupted 15-step
reference and r163/r164/r165 run AdaCheckStyleFixed, HostGlobalFixed and M1 with
ten source steps, final drain, node8 checkpoint-service loss, fresh epoch-10
same-layout restore and continuation through step 15. r166 recomputes the
four-arm suite as `valid=true`. All source and continuation losses are exact,
all 16 final digests match r162, all ten suffixes are sealed, and node8 is
observed unavailable during recovery. The 50 GB operator estimate and 48 GiB
minimum exceed measured retained bytes; configured capacity is 160 GiB.

| 10+5 metric | AdaCheckStyleFixed | HostGlobalFixed | M1Object |
|---|---:|---:|---:|
| ten-suffix remote gradient logical | 26,107,330,560 B | 3,263,416,320 B | 3,263,416,320 B |
| ten-suffix application payload | 26,117,647,980 B | 3,271,264,540 B | 3,271,266,210 B |
| base + suffix application payload | 44,494,632,135 B | 21,648,248,563 B | 21,648,250,167 B |
| suffix maximum remote PUT | 4,808.148 ms | 838.024 ms | 474.930 ms |
| source phase, base + 10 + drain | 158.821 s | 113.846 s | 108.739 s |
| fresh recovery maximum RTO | 45.799 s | 32.247 s | 33.235 s |
| maximum boundary lag | 10 | 9 | 8 |

M1 versus Ada reduces ten-suffix logical/application transfer by
`87.5/87.475%`; after the common base, total application transfer falls
51.346%. Host and M1 logical bytes are identical; their 1,670 B suffix
application difference is only control framing. HostGlobal concentrates all
publication on ranks 0 and 8 and reaches 353,117,184 B planned maximum publisher
load. M1 distributes the identical 652,683,264 B all-publication total across
all 16 ranks and caps the maximum at 77,328,384 B, a 78.101% reduction; its
maximum remote publisher load falls from 176,558,592 to 77,328,384 B.

The single r166 ordering also shows lower M1 suffix max PUT and source phase
than HostGlobal, but these are not stable timing claims; r148 remains the
three-repeat PP4 load evidence. All r166 checkpoint arms require final drain and
fail strict 1C1S, so this establishes a second fixed parallel layout with live
transfer reduction and reliable same-layout recovery, not non-blocking or 1C1S.
Recovery is independent of which rank originally published a semantic object,
but target logical rank and layout are still fixed. A sealed FQN/role/global-
slice manifest, target ownership mapping, model/base assembly, suffix projection,
and RNG/sampler progress transform are still required for arbitrary changed-
layout load-time resharding. No M2, 32-rank, physical compute-host fault or
20/50/200-step experiment has started.
The stage figure is
`docs/figures/object_semantic_pp2_10p5_loss8_r166.{png,svg}`.

### 72.20 r167 changed-layout materialized CPU gate

The independent `changed_layout_cpu_gate.py` binds ordinary PyTorch module and
scalar AdamW state to stable FQN/role objects for model parameters, FP32 master
parameters, `exp_avg`, `exp_avg_sq` and scalar `step`. It reuses the existing
`ObjectCatalog`, `TargetLayout`, failure-aware selection and global-slice
assembly code; no AdaCheck module or legacy recovery path is changed.

r167 constructs a materialized `PP4/TP1/DP4` source in which four pipeline
stages own separate layer FQNs and DP replicas span two storage failure domains.
It restores with host 0 unavailable into a `PP2/TP1/DP8` target in which each
pipeline stage owns two source FQNs and each object is replicated across eight
DP ranks. Every target model/master/moment/step tensor equals an independent
canonical global oracle. Missing coverage, overlapping fragments, unknown FQN
or role, epoch mismatch and target-layout-hash mismatch all fail before live
model or optimizer mutation. The saved test artifact is
`runs/two-node-v2/changed-layout-pp4dp4-to-pp2dp8-cpu-gate-r167-20260914.log`
with JUnit XML beside it: `16 passed`, one dependency warning, exit zero.

This is a real changed PP/DP ownership correctness slice, but not a production
SemanticFixed recovery. TP stays one, fragments are in-memory materialized CPU
objects, and no checkpoint service, base locator, suffix gradient replay,
scheduler/iteration, RNG or sampler transform is exercised. The next recovery
gate must derive source/target ownership from sealed production manifests,
fetch fragments from surviving services, apply the base through this preflight,
then project and replay every suffix before restoring global progress state.

### 72.21 Parameterized recovery theorem and validation boundary

Do not enumerate PP/TP/DP configurations to argue recovery generality. State a
parameterized theorem over the layouts admitted by the object contract. A source
or target layout is supported only when its adapter exports or requests every
declared state item with the rank-independent address
`(parameter FQN, role, global shape, dtype)`, an exact global slice, epoch and
checksum. A failure set is admissible only when the surviving sealed fragments
form a same-epoch, non-conflicting exact cover of every requested tensor slice
and retain every opaque singleton and base/update dependency. Scheduler,
iteration, RNG and sampler state additionally require a deterministic global or
logical-stream transformation; an absent transformation rejects the target
layout.

Under those premises, `restore_object_layout` is independent of source rank,
publisher rank and the source PP/TP/DP ownership: intersecting each requested
target slice with its surviving exact cover reconstructs the unique global
value, and atomic preflight applies it only after inventory, shape, dtype,
epoch, checksum, dependency and layout-hash validation. For the active
SemanticFixed stream, an exact base plus the ordered confirmed BF16 gradients
and step multipliers reconstructs the AdamW model and optimizer state by
induction over suffix epochs; a supported progress-state transform then defines
the same next training boundary. This is the recovery correctness argument for
every layout in the declared support domain, not merely the configurations used
in performance experiments.

The validation obligation is therefore orthogonal rather than exhaustive:

1. prove the exact-cover assembler and fail-closed conditions once;
2. test each state-role adapter and each transformation primitive, including PP
   ownership changes, TP split/merge, DP replication changes and progress-state
   mapping;
3. run representative fresh-process end-to-end recovery gates with a service
   failure and compare the continued numerical state with an uninterrupted job;
4. mechanically validate the catalog, target-layout witness and failure closure
   for every newly admitted plan, without rerunning the full layout cross-product.

r167 currently supplies materialized PP/DP ownership change for model and AdamW
tensor roles, while `test_object_recovery.py` supplies a TP-style multi-source
slice assembly gate. The live GPU runtime still restores the same layout, and
the sealed service-to-changed-layout adapter plus progress/RNG/sampler
transformations are not implemented. Consequently the paper may present the
parameterized correctness theorem and representative evidence, but the current
artifact claim remains **arbitrary contract-supported object reconstruction by
design**, not **arbitrary parallel configuration already demonstrated end to
end**. This boundary prevents an unimplemented adapter from becoming a theorem
premise by assertion.

### 72.22 Mainline baselines and r168/r169 diagnostic

Recovery is supporting infrastructure, not the experimental axis. The research
mainline is fixed as follows:

```text
grouping baselines:          AdaCheckStyle, GeminiStyleAdjacent, RandomFeasible
proposed grouping:           RedundancyMax / M1 under the same P/F
M1 causal comparison:       each recovery-feasible baseline -> M1
M1 load ablation:           HostGlobalFixed -> M1Object at equal logical bytes
M2 causal comparison:       training-only-best P + M1(P)
                              -> joint selected P*, mapping* + unchanged M1(P*)
```

GEMINI is used here only for its published fixed adjacent-worker grouping, not
as a claim to reproduce the complete unpublished GEMINI system. The local
nnScaler work under `../../eimc/Enhanced-In-Memory-Checkpoint` supplies separate
communication-gap profiling code; it is not mixed into the grouping algorithm.
`GeminiStyleAdjacent` means literal adjacent logical worker indices
`(0,1),(2,3),...`. Under the current node-major 16-rank mapping, most such pairs
are in one physical host and therefore fail the `F_host=1` recovery constraint.
That arm may be measured under rank-failure semantics, reported infeasible for
host failure, or made cross-domain by an explicit M2 physical mapping. It must
not be silently rewritten as `(0,8),(1,9),...`; those are AdaCheck's
corresponding-local-rank pairs on the current two-host layout.

The historical `profile_exact` solver is not a fifth formal grouping policy.
Its profiled makespan objective is folded into M1 after recovery feasibility and
transfer bytes, as a load/critical-path tie-break. `HostGlobalFixed` is likewise
an attribution control, not a fifth grouping policy. AdaCheckStyle-to-M1, with
detector, encoder, transport, failure set and recovery held constant, remains
the clean grouping benefit on the current host-fault gate.

M1 keeps the parallel plan and logical-to-physical mapping fixed and chooses
object grouping/coalescing, publisher, destination failure domain and schedule.
M2 alone searches the finite validated parallel-plan and physical-placement
set, regenerates its catalog/profile, and calls unchanged M1. The three paper
contributions remain: recovery-constrained object grouping that better exploits
detected natural redundancy; M2 parallel-placement/failure-domain co-design;
and publisher/service/critical-path load balancing. M1 never creates
redundancy, and M2 is not a new general auto-parallel planner.

r168 validates the common runtime's owned-snapshot copy-elision with `91 passed`
and one dependency warning. It avoids redundant CPU gradient/payload clones but
does not alter detector, object set, plan, wire schema, seal or recovery.
r169 repeats the r165 `PP2/TP1/DP8` M1 10+5 node8-service-loss gate and passes
exact source/restore loss plus all 16 final digests. Logical suffix transfer is
unchanged at `3,263,416,320 B`; externalize mean falls
`380.781 -> 285.031 ms`, maximum queue/final drain fall
`13.493 -> 12.463 s` and `15.104 -> 14.005 s`, and source phase falls
`108.739 -> 103.989 s`. Maximum PUT worsens `474.930 -> 591.598 ms`, recovery
phase worsens `63.758 -> 68.437 s`, lag remains eight and final drain is still
required. This is one M1 optimization diagnostic, not a new fair baseline suite
or a stable non-blocking/max-time result. Any post-change comparison must rerun
all baseline arms on this same runtime.

### 72.23 r170--r172 owned-runtime fair M1 suite

r170 and r171 rerun AdaCheckStyleFixed and HostGlobalFixed with the same
owned-snapshot common runtime as r169 M1. r172 combines them with the unchanged
r162 NoCheckpoint reference. It is a valid GPT-2 BF16 `PP2/TP1/DP8, pp-dp`
four-arm suite: ten source suffixes, final drain, node8 checkpoint-service loss,
fresh epoch-10 same-layout restore and five continued steps. Every checkpoint
arm has exact source and restore loss continuity, all 16 final full-state
digests equal r162, and all arms share detector SHA-256
`eb8846bf0702df68835861032edfaa0d8178f3b0e2945113e44757b35a40c980`.

| owned-runtime 10+5 metric | AdaCheckStyleFixed | HostGlobalFixed | M1Object |
|---|---:|---:|---:|
| suffix gradient logical | 26,107,330,560 B | 3,263,416,320 B | 3,263,416,320 B |
| suffix application payload | 26,117,624,200 B | 3,271,261,620 B | 3,271,263,290 B |
| base + suffix application | 44,494,608,553 B | 21,648,245,841 B | 21,648,247,445 B |
| planned max all-publication publisher | 353,117,184 B | 353,117,184 B | 77,328,384 B |
| suffix max remote PUT | 5,221.777 ms | 874.066 ms | 591.598 ms |
| source phase, including base and drain | 158.933 s | 118.896 s | 103.989 s |
| maximum final drain | 61.367 s | 21.304 s | 14.005 s |
| fresh recovery maximum RTO | 44.475 s | 32.371 s | 32.205 s |
| maximum boundary lag | 10 | 9 | 8 |

M1 versus AdaCheck reduces suffix logical/application transfer by
`87.5/87.475%` and base-plus-suffix total application transfer by `51.346%`.
This is the grouping/object-coalescing result under the common detector,
encoder, transport, commit, failure and recovery contract. HostGlobal and M1
have identical logical bytes; their 1,670 B suffix application difference is
opaque framing. Relative to HostGlobal, M1 lowers the planned all-publication
and remote maximum publisher loads by `78.101%` and `56.202%`.

The one ordered r172 timing sample also lowers HostGlobal-to-M1 distributed
commit mean, maximum queue, maximum drain, suffix max PUT and source phase by
`41.291/33.958/34.264/32.317/12.539%`. These are diagnostic values, not a
stable timing claim. Every checkpoint arm still needs final drain and fails
strict 1C1S; the fault is a checkpoint-service outage, not a physical compute-
host failure. The fail-closed analysis is
`runs/two-node-v2/object-semantic-pp2-owned-four-arm-r172-20260914/analysis.json`,
SHA-256 `ed8c12bbb3a25a9abd08cc8e5a3c1112f83f716389fc85657bc15a87f04a17dc`.
The comprehensive figure is
`docs/figures/object_semantic_pp2_owned_10p5_loss8_r172.{png,svg}`.

### 72.24 r173--r181 recovery-valid repeated M1 evidence

Two additional alternating blocks complete nine distinct checkpoint runs under
the unchanged r172 contract. The run orders are Host/M1/Ada and Ada/Host/M1,
complementing the first M1/Ada/Host block. Every arm in every block completes
ten source steps, node8 checkpoint-service loss, fresh epoch-10 restore and five
continued steps. The r181 analyzer reruns the fail-closed four-arm suite for all
three blocks and requires exact source/restore loss plus all 16 final state
digests before computing statistics.

| three-run mean | AdaCheckStyleFixed | HostGlobalFixed | M1Object |
|---|---:|---:|---:|
| suffix gradient logical | 26.107 GB | 3.263 GB | 3.263 GB |
| base + suffix application | 44.495 GB | 21.648 GB | 21.648 GB |
| suffix bottleneck PUT | 4,874.829 ms | 878.693 ms | 574.995 ms |
| source phase, base + ten + drain | 153.926 s | 113.929 s | 102.136 s |
| maximum final drain | 59.587 s | 21.625 s | 13.328 s |
| maximum fresh-recovery RTO | 44.425 s | 32.789 s | 32.981 s |
| maximum boundary lag | 10 | 9 | 8 |
| exposed source step wall | 0.491 s/step | 0.556 s/step | 0.569 s/step |

M1 versus AdaCheck reduces suffix logical/application, total application,
suffix bottleneck PUT, source phase and maximum drain by
`87.5/87.475/51.346/88.205/33.646/77.632%`. At equal logical bytes, M1 versus
HostGlobal reduces suffix bottleneck PUT, source phase and maximum drain by
`34.562/10.352/38.366%`; maximum RTO changes by `+0.584%`. This upgrades the
transfer and tail result from a single timing observation to `n=3` descriptive
run-level evidence, without an inferential-significance claim.

The negative boundary is equally important. A single r162 NoCheckpoint
reference has mean step wall `0.220 s`; Ada/Host/M1 means are
`0.491/0.556/0.569 s`. M1 is `15.970%` slower than Ada and `2.277%` slower than
Host within the observed training steps. All runs accept final drain and the
queue limit equals the ten-step horizon, so r181 does not prove sustained
non-blocking behavior. Strict 1C1S remains an aspirational upper bound, not an
admission gate; later runs report the actual sustainable cadence and RPO lag.

The authoritative aggregate is
`runs/two-node-v2/object-semantic-pp2-owned-recovery-repeats-r181-20260914/analysis.json`,
SHA-256 `297724555da853673d09e79b2b516953e726c9b5bc15b6b8ef54c092cd536dce`.
The paper figure is
`docs/figures/object_semantic_pp2_owned_repeats_r181.{png,svg}`. The next GPU
stage remains node17/node8 only: generate model-matched `PP4/TP1/DP4, pp-dp`
detectors, then run the same three-arm 10+5 gate for GPT-6.33B and LLaMA-3-8B.
After both dense models have favorable, honestly reported transfer, bottleneck
time, cadence, training-wall/stall and recovery results, stop launching new
experiments and prepare the paper discussion. No 32-rank, M2 or 20/50/200-step
run is authorized by this section.

### 72.25 r187/r196--r199 LLaMA-8B recovery-valid M1 stage gate

r199 fail-closed combines a fresh NoCheckpoint r187 with M1 r196,
HostGlobalFixed r197 and AdaCheckStyleFixed r198. All arms use node17/node8,
16 ranks, the same tracked-diff SHA-256 `4ed8df4474ff...`, the same detector
SHA-256 `daace4aafcaa...`, BF16 LLaMA-3-8B (8.03B architecture), scalar AdamW,
sequence length 128 and `PP4/TP2/DP2, pp-dp`. The checkpoint arms each write a
common base plus two live gradient-driven suffixes, remove the node8 checkpoint
service, restore epoch 2 using node17, and continue through step 3.

Every checkpoint arm has exact source and restore loss continuity and all 16
final full-state digests exactly match r187. The loss trace is
`12.7010 -> 12.4718 -> 12.5569`. Ada/Host/M1 two-suffix application bytes are
`72,651,824,931/36,325,673,315/36,325,673,299 B`; including their common
approximately 254.275 GB base, totals are
`326,926,668,182/290,600,516,599/290,600,516,517 B`. Thus M1 versus Ada lowers
live suffix application transfer by `50.000%` and base-plus-suffix total by
`11.111%`. Host and M1 bytes are identical; the M1 load objective halves the
planned maximum all-publication publisher from `5,591,285,760` to
`2,795,577,344 B`.

This single ordered stage sample gives Ada/Host/M1 suffix maximum remote PUT
`48.184/27.121/23.708 s`, maximum distributed commit
`60.688/37.606/33.130 s`, maximum final drain
`130.482/83.348/76.856 s`, source phase
`1400.224/1325.329/1289.968 s`, and maximum recovery RTO
`307.242/255.760/300.888 s`. M1 versus Ada reduces suffix max PUT by `50.797%`
and source phase by `7.874%`; at equal bytes its Host-to-M1 timing differences
remain descriptive, not statistical. The checkpoint-arm source-step means are
`3.097/3.878/3.056 s`, versus the three-step NoCheckpoint reference at
`1.067 s`; all checkpoint arms have maximum lag 2 and require final drain.
Therefore r199 proves recovery-valid large-model transfer reduction, but not
sustained non-blocking, strict 1C1S, compute-host failure or changed-layout GPU
recovery.

The authoritative suite is
`runs/two-node-v2/object-semantic-llama8b-pp4tp2dp2-suite-r199-20260915/analysis.json`,
SHA-256 `f17920d8f43dea876204be62cd9ff4dcf4aa853d9fe5aaba3a9f79be66e91336`.
The fail-closed stage figure is
`docs/figures/object_semantic_llama8b_four_arm_r199.{png,svg}`. GPT-6.33B
`PP4/TP1/DP4` remains the second dense-model gate; do not start 32-rank or M2
work before the user-authorized 16-rank stop condition is reviewed.

### 72.26 GPT-6.33B base-memory rejection r201 and guarded retry r202

r201 attempted M1 on node17/node8 with 16 ranks, BF16 GPT-6.33B,
`PP4/TP1/DP4, pp-dp`, seq128, scalar AdamW, two source steps and one continued
step. Its four-rank base-upload window reached 16 base records and
354,352,883,364 B on node17 but only 12 records and 270,450,155,256 B on
node8. During the final four concurrent serializations node8 had approximately
12 GiB available; the kernel OOM-killed the 289.7 GB-RSS object replica service
at `2026-09-15 01:57:02 UTC`. Ranks then failed the base commit with
`ConnectionRefusedError`; tmux exited 1 and the run has `FAILED`, not
`COMPLETE`. It contains no training, suffix, recovery or performance result.

The minimal correction parameterizes the already-existing
`max_parallel_uploads` input as `CKPT_OBJECT_BASE_UPLOAD_PARALLELISM`, records
it in the command and run contract, and retains the default of four. The
training/base focused suite passes `4 passed`; shell syntax, bytecode and diff
checks pass. r202 reran the identical M1 contract with only base upload
parallelism changed `4 -> 2`. It reached 14 base records and about 312.4 GB on
node8 before the kernel killed the service at about 321.2 GB RSS. The run is
`FAILED` and has no training/suffix/recovery result. The steady problem is one
full rank-local base per rank per service, not upload concurrency; the next
valid correction is an object-deduplicated common base shared by all comparison
arms.

### 72.27 r203/r204 motivation scope correction

r203 freshly captured BERT, GPT-2 and LLaMA-3B on node17/node8, 16 ranks, two
BF16 detector steps, world scope, and both `PP4/TP2/DP2` and `PP4/TP1/DP4`.
Its first analysis constrained every feasible candidate pair to span the two
failure domains. Under the fixed PP-first node-major placement all such pairs
have equal cost, so AdaCheckStyle, RandomFeasible and RedundancyMax show zero
gain. Literal GEMINI adjacency is same-host and therefore excluded from a
one-host-loss claim. This is a valid fixed-placement recovery-constrained
negative control, not the paper's opening motivation.

r204 reuses the exact same six r203 catalogs and changes only the offline
grouping contract: AdaCheckStyle remains fixed `(0,8)..(7,15)`, GEMINI remains
literal adjacency, RandomFeasible samples any rank-level perfect matching, and
RedundancyMax exactly maximizes overlap over any rank-level perfect matching.
No host-loss recovery constraint is imposed. RedundancyMax relative to the
AdaCheck-style fixed peer reduces 16-rank pair-local logical bytes by
`99.99869/99.99896%` for BERT, `99.99720/99.99922%` for GPT-2, and
`99.99973/99.99996%` for LLaMA-3B across the DP2/DP4 layouts. It selects the
same PP/TP slice's DP replicas; only `81,408--278,016 B` runtime metadata
remains. This proves the intended existence statement: peer choice can expose
far more already-existing redundancy. It does not create redundancy and is not
an AdaCheck gradient-wire, host-loss recovery, or end-to-end performance claim.

Paper wording must say “AdaCheck-style detector/equality evidence followed by
destination-aware peer-difference encoding,” not imply that changing the
destination of an unchanged destination-independent AdaCheck record `E_r`
changes `sum_r |E_r|`. r204 is retained only as
`docs/figures/motivation_unrestricted_upper_bound_r204.{png,svg}`. The r203
control is `docs/figures/motivation_failure_constrained_control_r203.{png,svg}`.
This observation creates the system question: recover the opportunity while
satisfying object completeness, failure domains and load balance through M1,
then jointly change physical placement/parallel plan through M2.

### 72.28 Motivation evidence hierarchy correction

The opening motivation does not need the complete end-to-end system contract,
but every number must remain a correct observation under its stated contract.
The current main figure therefore restores the original cross-host candidate
evidence from 2026-09-09/10 rather than presenting r204's trivial exact-DP
upper bound. For BERT, GPT-2 and LLaMA-3B, RedundancyMax reductions are
`23.022/71.054/24.851%` in `PP4/TP2/DP2` and
`18.419/19.130/5.768%` in `PP4/TP1/DP4`. Random is not claimed as an
optimizer: LLaMA-3B DP2 regresses by `159.194%`, while other sampled cells may
improve. This variance is evidence for an informed grouping policy.

`docs/figures/motivation_transfer_size_current_evidence.{png,svg}` now plots
AdaCheckStyleFixed, Random and RedundancyMax on those six cross-host candidate
sets. Literal GEMINI adjacency is omitted because node-major rank adjacency is
same-host and is not a member of the compared cross-host set. This omission is
explicit rather than relabeling another policy as GEMINI. The full four-policy
contract remains required in end-to-end comparisons, where detector, encoder,
transport, recovery constraints and training execution must all be both
correct and complete. r203 remains the strict fixed-placement host-loss
negative control; r204 remains an unrestricted upper-bound diagnostic only.

### 72.29 DP-deduplicated recovery base and GPT-6.33B r205/r206 gate

The rank-local base used by r201/r202 was a capacity bug, not a property of
M1: both services retained all 16 complete rank snapshots even though DP
replicas in one `(PP,TP)` class have the same heavy model/optimizer state. The
new independent `object_semantic_deduplicated_base.py` leaves the old AdaCheck
and rank-local recovery paths unchanged. It stores one checksum-addressed
`{model, optimizer}` bundle for each `(PP,TP)` class plus one small binding,
parameter-topology, scheduler, RNG, sampler and iteration overlay for every
logical rank. Every bundle and the common control is copied to both failure
domains and sealed. Publication fails closed unless all DP replicas in a class
have identical heavy-state checksums and topology; restore fetches one shared
bundle plus the logical-rank overlay, reconstructs the exact full base, and
verifies its rank-specific base id before applying suffixes. This is a common
recovery/storage facility for every comparison arm, not an M1 transfer gain.

Focused tests validate restore of every logical rank from either host and
reject one divergent DP replica. The deduplicated-base tests pass `2 passed`;
the broader SemanticFixed focused set passes `28 passed` (one warning), and
the gate analyzer tests pass `4 passed`.

r205 is a 16-rank GPT-2 `PP4/TP1/DP4` one-source-step/one-continued-step live
smoke. It successfully stored four shared bases and 16 overlays on both
services, removed the node8 service, restored all 16 ranks from node17 and
continued with loss `11.0 -> 10.875`. Its base remote payload is
`2,297,362,859 B` plus `46,692 B` control; the one M1 suffix application is
`326,888,121 B` with `513.401 ms` max PUT. The original driver exited 1 only
because the then-current analyzer incorrectly required the intentionally
rank-specific base ids to be identical. The analyzer now requires a common
locator/metrics and validates each rank's own base id; post-hoc analysis
passes. The original `FAILED` marker is retained, so r205 is implementation
evidence rather than an admitted treatment result.

r206 is the first admitted 6.33B validation of this base. It uses node17/8,
16 ranks, BF16 GPT-6.33B, scalar AdamW, `PP4/TP1/DP4, pp-dp`, seq128, M1,
two source steps, node8 checkpoint-service loss, fresh epoch-2 restore and one
continued step. Both services sealed exactly four heavy bases, 16 overlays and
one base control. Base remote application is `88,588,559,100 B`, a
`74.999905%` reduction from r201's 16-copy `354,352,883,364 B` service state,
but this reduction is common to Ada/Host/M1. The two live M1 suffixes are each
`12,656,433,254 B` application (`12,655,411,200 B` gradient logical); their
max PUT times are `19.021/17.229 s`. Base plus two suffixes totals
`113,901,425,608 B`; maximum distributed commit is `25.147 s`, maximum final
drain is `65.836 s`, and lag is 2, so this is not strict 1C1S or sustained
non-blocking evidence.

r206 is `COMPLETE`, exits zero and has all 16 source/recovery metrics. It
reproduces loss `11.4375 -> 11.5625 -> 11.0625`; every restored epoch-3 full
state digest exactly matches the uninterrupted r200 reference. Its analysis
SHA-256 is `bb4a2b971803772ea587e013149a5c16bfc688831d9cee5685cd922179f24740`.
It proves capacity-safe, service-loss-recoverable M1 execution, but cannot by
itself establish M1 improvement.

r207 used the same 160 GiB service limit for AdaCheckStyleFixed. Its common
base and first suffix sealed correctly, but the second suffix was rejected at
`170,549,654,950 B`: capacity `171,798,691,840 B` could not admit the next
`3,013,343,145 B` immutable record. The run is `FAILED`, has no recovery, and
is not a treatment result. This is an experiment-sizing error caused by the
expected fourfold Ada suffix, not an Ada runtime failure. r208 reruns Ada with
an explicit common suite capacity of 224 GiB (`229,376 MiB`; conservative
retention estimate `200,000,000,000 B`). r208 completes this contract and
fresh recovery. Ada's two suffix applications total `101,245,883,284 B` and
base plus suffixes total `189,834,442,698 B`; maximum suffix PUT, distributed
commit and final drain are `66.003/82.773/180.847 s`. All 16 ranks restore
epoch 2 from node17 and reproduce the r200 loss/final-state oracle. Analysis
SHA-256 is `6326877ac65b866ea84410104c074aacfde4a031a6d5d49514aed6a4af1fe3a9`.

Against the capacity-mismatched but otherwise identical r206 candidate, M1
reduces suffix application `74.998621%`, base-plus-suffix application
`39.999600%`, max suffix PUT `71.182%`, max distributed commit `69.619%`,
max drain `63.596%` and maximum RTO `16.310%`. These become formal only after
the same-capacity M1 rerun. M1's two-step mean wall is `3.756 s` versus Ada
`3.715 s` and the first two r200 NoCheckpoint steps `1.123 s`; therefore the
current result does not prove training non-blocking.

r209 completes the same-capacity HostGlobalFixed arm. Its two suffixes total
`25,312,866,468 B` application and base plus suffixes total
`113,901,422,031 B`; maximum suffix PUT, distributed commit, final drain and
RTO are `21.009/34.653/85.414/325.456 s`. Source-step wall is
`4.052/3.336 s` (mean `3.694 s`), lag is 2 and final drain remains required.
After node8 checkpoint-service loss, all 16 ranks reproduce the reference loss
and final full-state digests. Host and the earlier r206 M1 candidate have equal
suffix bytes by design, so their provisional timing difference isolates only
publisher/load scheduling.

r210 completes M1 under the same 224 GiB service capacity. Its two suffixes
total `25,312,866,522 B` application and base plus suffixes total
`113,901,425,881 B`; maximum suffix PUT, distributed commit, final drain and
RTO are `19.346/25.718/66.854/317.163 s`. Source phase is `2,156.559 s`;
source-step wall is `3.789/4.436 s` (mean `4.113 s`), restored step 3 is
`1.487 s`, lag is 2 and final drain remains required. All 16 ranks reproduce
the exact reference loss and final full-state digests.

r211 recomputes r200/r208/r209/r210 and is `valid=true` under one detector,
training, failure and capacity contract. M1 versus Ada reduces suffix and
base-plus-suffix application transfer by `74.998621/39.999600%`, suffix max
PUT by `70.688%`, distributed commit by `68.930%`, final drain by `63.033%`,
source phase by `3.341%` and maximum RTO by `15.043%`. At equal suffix bytes,
M1 versus Host reduces PUT/commit/drain/source phase by
`7.916/25.784/21.729/2.336%` in this one ordered sample. The suite SHA-256 is
`c1ce14c9dd974cef43149e621d951b2f4c77441e65bb6445cb177e895a2e2950`.

The negative boundary remains explicit: GPT M1 source-step mean is `10.715%`
slower than Ada and `4.36x` NoCheckpoint. Together with LLaMA r199, this closes
the requested two-model recovery-valid transfer and tail stage, but does not
establish sustained non-blocking or strict 1C1S. The composite figure is
`docs/figures/object_semantic_dense_two_model_16r.{png,svg}`. Stop starting new
GPU runs here and discuss whether the paper claim ends at transfer/tail or
retains non-blocking as a further admission gate.

### 72.30 Post-r211 common-runtime correction and r218/r219 calibration

The user retained non-blocking as an active admission target, so the r211 stop
was explicitly lifted while preserving the node17/node8 16-rank limit. The
independent SemanticFixed path now treats post-allreduce DP equality as a
training invariant that must still be confirmed by one CPU SHA-256 verifier per
complete `(PP,TP)` DP set. `object_semantic_dp_invariant.py` compares those
verifiers across TP coordinates and expands the confirmed checksum to all DP
holders; incomplete DP sets are hashed conservatively on every holder and any
disagreement fails the checkpoint. Pinned snapshots contain only the union of
planned publishers and verifiers. The sparse externalizer requires every
planned publisher payload and never substitutes a missing publisher. Suffix
bundles retain their semantic tensor SHA-256 and whole serialized-payload
SHA-256, while skipping the redundant generic per-object bundle hash. Legacy
AdaCheck and recovery modules remain unchanged.

Focused analyzer/runtime/recovery tests pass `35 passed` (one environment
warning). r212 and r215 are pre-correction diagnostics: raw BF16 and preallocated
pinned slots reduce LLaMA capture from about `1.305 s` to `8.027 ms`, but
duplicated hashing still costs about `5 s`. r216 correctly failed fresh restore
because its new encoder provenance did not match the old restore schema. r217
then completed source and restore with exact state, but the original driver
failed only because the analyzer rejected the newly rank-local staged/verifier
fields; its post-hoc analysis is diagnostic and the `FAILED` marker remains.
The analyzer now admits only the exact `post_allreduce_dp_invariant` mode while
still requiring all common plan fields to agree.

r218 M1 and r219 AdaCheckStyleFixed are clean `COMPLETE` runs with an otherwise
identical node17/node8, 16-rank GPT-2 BF16 `PP2/TP1/DP8, pp-dp`, scalar AdamW,
seq128, detector, base, encoder, transport, node8 service-loss and 2+1 fresh
recovery contract. Both reproduce loss `10.9375 -> 10.8125 -> 10.8125`; all
16 restored final full-state digests exactly equal r214 NoCheckpoint. Ada-to-M1
two-suffix logical/application traffic falls
`5,221,466,112 -> 652,683,264 B` (`87.5%`) and
`5,223,524,850 -> 654,252,678 B` (`87.475%`). Including the common base,
application traffic falls `7,520,989,112 -> 2,951,714,271 B` (`60.754%`).
Suffix max PUT, max distributed commit, max final drain, source phase and max
RTO fall `4.834 -> 0.703 s`, `5.711 -> 1.174 s`, `10.947 -> 2.162 s`,
`123.683 -> 114.225 s` and `21.467 -> 16.709 s`, respectively.

M1 source-step mean is `0.588 s`, `26.270%` below Ada's `0.798 s`, but still
`12.968%` above r214's first-two-step NoCheckpoint mean `0.521 s`. Both have
lag two and final drain. This is a successful small-model calibration of the
new common runtime and recovery/M1 interaction, not sustained non-blocking or
a replacement for the two dense-model gate. LLaMA-8B r220 is the next admitted
run; no 32-rank or M2 experiment is authorized.

### 72.31 r220 TP-boundary rejection and r221 corrected recovery gate

r220 attempted the same corrected runtime on LLaMA-8B `PP4/TP2/DP2`. Its first
suffix committed, but epoch two failed closed because one detector candidate
that combined identical FQN/slice metadata across TP coordinates had different
live gradient checksums. No restore ran; r220 is `FAILED` and supplies no
treatment result. The failure establishes the missing semantic boundary:
post-allreduce equality is guaranteed only among DP replicas at one `(PP,TP)`
coordinate, not between different TP coordinates even when their static slice
descriptors match.

The canonical object key now includes the TP coordinate. M1 also chooses one
least-loaded holder per canonical object and reuses its single pinned CPU
snapshot when publishing to both failure-domain services. This keeps the two
required service copies and remote transfer unchanged while removing duplicate
rank-side staging. Encoder/base provenance now declares
`object_partition=fqn_global_slice_pp_tp` and
`dp_equivalence_basis=post_allreduce_training_invariant`; restore requires
those exact fields. Planning, encoding, runtime and recovery tests pass
`50 passed` (one environment warning).

r221 is the repaired GPT-2 `PP2/TP1/DP8` 2+1 node8-service-loss gate. It is
`COMPLETE`; source/restore loss and all 16 final state digests exactly match
r214. Suffix bytes remain `652,683,264 B` logical and `654,253,810 B`
application over two epochs, while staged bytes fall from r218's
`652,683,264 B` to `326,341,632 B`; verifier bytes are also `326,341,632 B`.
Its source steps are `0.795/0.378 s`. This validates the repaired semantics and
single-publisher lifetime before LLaMA r222; it is not a new paper treatment.

### 72.32 LLaMA r222 corrected M1 arm

r222 is the corrected LLaMA-8B BF16 `PP4/TP2/DP2, pp-dp` M1 arm on node17/8.
It uses the r185 world-scope detector, scalar AdamW, seq128, deduplicated base,
two source steps, node8 checkpoint-service loss, fresh epoch-2 restore and one
continued step. It is `COMPLETE`: loss is exactly
`12.7010 -> 12.4718 -> 12.5569`, and all 16 restored final-state digest files
are byte-identical to the uninterrupted r187 reference.

The corrected TP-local plan transfers `18,162,401,280 B` logical gradient per
suffix and `36,326,761,018 B` suffix application bytes over both epochs. The
common base is `127,137,637,040 B`, giving `163,464,398,058 B` total
application transfer. Maximum suffix PUT, distributed commit, final drain and
RTO are `29.748/33.871/69.231/242.896 s`. Source step wall is `1.948/0.716 s`
(mean `1.332 s`), versus r187's first-two-step mean `1.270 s`; the foreground
overhead is about `4.9%`, while lag two and final drain still preclude a
sustained non-blocking or 1C1S claim. The result is not yet an M1 effect size:
r223 AdaCheckStyleFixed and a same-runtime HostGlobalFixed arm must complete
under the identical detector/base/transport/fault contract. Analysis SHA-256:
`ef6074784d1314a6597eb92e291c364e06b9ad58d65f651935eaed09f04b3666`.

### 72.33 LLaMA r223 corrected Ada comparison

r223 is the corrected AdaCheckStyleFixed arm under the exact r222 detector,
model, topology, base, encoder, transport, capacity and node8 service-loss
contract. It is `COMPLETE`; its loss trajectory and all 16 restored final-state
digests exactly match r187. Ada-to-M1 suffix application transfer is
`72,651,842,688 -> 36,326,761,018 B` (`49.999%`), and common-base-inclusive
application transfer is `199,789,483,792 -> 163,464,398,058 B` (`18.182%`).
Maximum suffix PUT, distributed commit, final drain and source phase fall
`54.041 -> 29.748 s`, `62.203 -> 33.871 s`, `122.724 -> 69.231 s` and
`1225.926 -> 1170.703 s`, respectively. Source-step mean falls
`1.514 -> 1.332 s` (`12.0%`); M1 remains about `4.9%` above the matching
NoCheckpoint first-two-step mean. Maximum RTO falls only
`250.825 -> 242.896 s`, so the recovery-time result is modest.

This admits a corrected recovery-valid M1 transfer, bottleneck-edge and
observed foreground improvement over Ada for LLaMA. It does not yet admit the
load-balancing attribution or a sustained non-blocking claim: r224
HostGlobalFixed is running, and all checkpoint arms still use lag two plus
final drain. r223 analysis SHA-256 is
`3b0693d987a3229e26eb27e071f95653735503977b2fd04a7f7e7828cd32bf20`.

### 72.34 LLaMA r224/r225 corrected strict suite

r224 HostGlobalFixed is `COMPLETE` and exactly reproduces r187 after the same
node8 service loss. r225 fail-closed recomputes r187/r223/r224/r222 and is
`valid=true`: all checkpoint arms share the detector, training and fault
contracts, have zero loss error and match all 16 final full-state digests.

M1 versus Ada reduces suffix and base-inclusive application transfer by
`49.999/18.182%`, suffix max PUT by `44.953%`, maximum distributed commit by
`45.549%`, maximum final drain by `43.588%`, source phase by `4.505%`, maximum
RTO by `3.161%`, and observed source-step mean by `12.021%`. M1 remains about
`4.9%` slower than the matching first two NoCheckpoint steps, with lag two and
final drain; this is a favorable two-step foreground observation, not sustained
non-blocking or strict 1C1S.

Host and M1 have equal suffix logical bytes (`36,324,802,560 B`) and nearly
identical application bytes. M1 reduces the planned maximum all-publication
publisher load `5,591,285,760 -> 2,797,600,768 B` (`49.965%`). In this single
ordered sample, M1 improves maximum commit/drain/source/total phase and max RTO
by `2.140/3.344/0.396/0.982/1.972%`, but worsens max PUT and source-step mean by
`6.517/5.047%`. Therefore LLaMA proves the load-balancing objective, not a
stable Host-to-M1 timing win. r224/r225 SHA-256 values are
`844e1d37bc430e9ad4dfcd84265794f137611034761570dd0472ee4492770310` and
`224be2980c9acb39a5fc111f890e3c115139176c29948abd807cff65ef3f67b4`.
GPT-6.33B r226 is the next corrected dense arm; no 32-rank run is authorized.

### 72.35 GPT-6.33B r226 corrected M1 arm

r226 is `COMPLETE` under the corrected TP-local, DP-invariant single-publisher
runtime: BF16 GPT-6.33B, `PP4/TP1/DP4, pp-dp`, scalar AdamW, seq128, two
source steps, node8 checkpoint-service loss, fresh epoch-2 restore and one
continued step. Loss is exactly `11.4375 -> 11.5625 -> 11.0625`; all 16
restored final-state digest files are byte-identical to the uninterrupted r200
reference.

The common deduplicated base transfers `88,588,559,642 B` application bytes.
The two M1 suffixes transfer `25,310,822,400 B` logical gradients and
`25,312,867,676 B` application bytes, for `113,901,427,318 B` including the
base. Maximum suffix PUT, distributed commit and final drain are
`22.094/24.686/49.902 s`; source/restore/combined phase wall times are
`1465.843/524.102/1989.945 s`, and maximum RTO is `336.095 s`. Source-step
wall is `1.900/0.609 s` (mean `1.2545 s`), about `11.8%` above r200's matching
first-two-step mean. Lag is two and a final drain remains, so this is not
sustained non-blocking or strict 1C1S evidence.

r226 is an admitted raw M1 arm, not an effect size. Its analysis SHA-256 is
`6ea97c8fef959399482c4d3bea27e647b48ba2dc682da056e0e05125eefb7e6a`.
The corrected same-capacity AdaCheckStyleFixed run is r227c; r227/r227b are
preflight-only attempts with no GPU training or treatment data. HostGlobalFixed
must follow before the GPT suite can attribute bytes and load scheduling.

### 72.36 GPT-6.33B r227c corrected Ada arm

r227c is `COMPLETE` under the exact r226 model, detector, encoder, transport,
service-capacity, fault and recovery contract. It reproduces loss
`11.4375 -> 11.5625 -> 11.0625`, and all 16 restored final-state digest files
are byte-identical to r200. Its common base, two-suffix and base-inclusive
application transfer are `88,588,562,812`, `101,245,883,304` and
`189,834,446,116 B`. Maximum suffix PUT, distributed commit, final drain and
RTO are `70.527/79.042/158.259/417.325 s`; source/restore/combined phase wall
times are `1510.658/605.539/2116.197 s`. Source-step wall is
`2.209/0.825 s` (mean `1.517 s`), with lag two and final drain.

The provisional Ada-to-M1 deltas are positive for suffix/total application,
max PUT/commit/drain, source phase, max RTO, source-step mean and combined
phase. They remain provisional until HostGlobalFixed r228 and fail-closed suite
r229 complete. r227c analysis SHA-256 is
`fea88708499f812be495eee818ff25d1b71d75333a8af8491cd318a2c3e418c0`.
r227 and r227b were preflight-only rejects caused by a missing explicit service
capacity and two hung read-only SSH diagnostics, respectively; neither started
GPU training or supplies treatment data.

### 72.37 GPT-6.33B r228/r229 corrected strict suite and 16-rank stop

r228 HostGlobalFixed is `COMPLETE` and exactly reproduces r200 after the same
node8 checkpoint-service loss. r229 fail-closed recomputes
r200/r227c/r228/r226 and is `valid=true`: all four arms have the same detector
hash and training contract; all checkpoint arms have zero source/restore loss
error and byte-identical 16-rank final full-state digests.

For GPT-6.33B, M1 versus Ada reduces two-suffix application transfer
`101,245,883,304 -> 25,312,867,676 B` (`74.999%`) and base-inclusive total
application transfer `189,834,446,116 -> 113,901,427,318 B` (`40.000%`). It
reduces maximum suffix PUT, distributed commit, final drain, source phase,
maximum RTO, source-step mean and combined source+restore phase by
`68.673/68.769/68.468/2.967/19.465/17.304/5.966%`. M1 source-step mean is
`1.2545 s`, still `11.8%` above the matching first-two-step NoCheckpoint mean;
lag two and final drain remain. Thus this is a recovery-valid improvement over
Ada, not sustained non-blocking or strict 1C1S.

Host and M1 have identical suffix logical bytes and differ by only `1,190 B`
of suffix framing. M1 lowers planned maximum all-publication publisher load
`6,645,620,736 -> 1,677,721,600 B` (`74.754%`) and observed commit/drain by
`14.880/20.555%`. In this one ordered sample, M1 nevertheless worsens suffix
max PUT, source-step mean, source phase, combined phase and maximum RTO by
`9.951/16.752/0.303/1.708/8.848%` versus Host. The plan objective is proven;
stable load-to-time conversion is not.

r228/r229 analysis SHA-256 values are
`752cb5123e1011d6d49c719c956c69d75ce3421c24debc2731fbe27ba637f0ab` and
`9fdb6370606bfa0c35623803db476b4e9721863e9abaf86a020d85f698e9aa38`.
The corrected two-model figure is
`docs/figures/object_semantic_dense_two_model_16r_corrected.{png,svg}` and
reads r225/r229 directly. Focused planning/runtime/recovery/analyzer regression
is `92 passed`, one environment warning. The requested 16-rank stopping point
is reached: do not start further GPU experiments until paper discussion with
the user; 32-rank and M2 remain closed.

### 72.38 2026-09-28: 16-rank dense/MoE continuation, EP gate, and metric contracts

The user authorized further node17/node8 16-rank experiments, including the
two dense and two MoE workloads. Run GPU jobs in dedicated tmux sessions with
immutable commands, manifests, logs, and exit status. Only edit files under
`/data/haiqwa/zevin_nfs/jwens`; the `sjw_ckpt_grouping` environment may be
updated. 1C1S remains an aspirational endpoint, not a gate. Changes to
`docs/09-24.md` are limited to the experiment data table: document plans,
failed gates, metric semantics, and bandwidth details **here**.

The CPU-only `tools/preflight_moe_ep_zero1.py` rejects the registered
PP4/TP1/EP2/DP2 cell (`ready=false`): the current Qwen2-MoE construction
contains all 60 experts per layer on each process, and DeepSeek-V2-Lite all
64 per layer. The existing Merak forward does not dispatch tokens over EP or
shard expert ownership. `object_semantic_training.py` and
`object_semantic_restore.py` also reject ZeRO; their optimizer-update path
expects full per-parameter FP32 masters while Merak ZeRO-1 owns flattened
local shards. New opt-in `Merak/core/moe/expert_parallel.py` supplies rank
axes, EP group creation and an explicit ownership gate; independent
`Merak/core/checkpoint/zero1_adapter.py` exposes local parameter-range
identities, without claiming a working object publisher/restorer. The
preflight is **structural CPU evidence**, not 16-rank EP training. No MoE
checkpoint, Ada/M1 gain, or EP recovery result exists yet. Next implement
genuine expert partition, token dispatch/return and backward, separate
expert/shared gradient reductions, common ZeRO-1 exact-shard base/update
and restore adapters, then EP1/EP2 numerical/empty-route tests and complete
same-contract Ada/Host/M1 CPU recovery before GPU admission. Fail closed
on missing model/optimizer/router/scheduler/RNG/sampler/iteration objects;
do not relabel prior Qwen no-EP r58–r60 as this experiment.

The primary checkpoint transfer metric is all 16 ranks' **cross-host**
application payload of the common base plus all actually committed suffix
updates under the same detector/encoder/failure/retention contract. Report
suffix/control separately, transport wire bytes, service occupancy, and
per-rail NIC counters; the canonical unique-state catalog is not transfer.
The current `maximum_remote_put_ms` times a remote client's individual
`put_prepared` call, then takes the max over ranks and committed epochs;
the table reports the suffix-only value. This is a **max remote PUT proxy**,
not the duration of a shared physical NIC edge, global commit, or exposed
training stall. To substantiate the intended physical max-edge claim, sample
NIC counters and aligned per-edge intervals; do not equate payload divided
by nominal line rate with observed transfer time.

For GEMINI-style recovery wasted time, preserve separate components:
checkpointing completion/exposed overhead `T_ckpt`, progress lost since
the latest recoverable *committed* epoch `T_lost`, and checkpoint retrieval/
restore `T_retrieve`, along with real job restart and the first successful
continued step. Label any `T_ckpt + T_lost + T_retrieve` as a
GEMINI-style estimate and document the exact checkpoint-exposure definition;
`E[T_lost] ≈ I/2` requires uniformly timed failures and stable committed
interval `I`, not simply one captured update per iteration. Current r225/r229
provide a single same-layout **checkpoint-service** failure and per-rank
restore wall time (max Ada→M1: LLaMA `250.825→242.896 s`, GPT
`417.325→336.095 s`), but no representative failure-time distribution or
steady committed cadence; hence no comparable average recovery wasted time.
Do not conflate service-only failure with physical compute-rank/host loss.

Both hosts expose `mlx5_0` and `mlx5_1` at nominal 100 Gb/s per link. These
rates are not measured training or checkpoint bandwidth. Reuse existing
`PerfqueryReader` and `ib_counter_delta` for non-resetting per-rail PMA
measurements. Record port/NIC mapping and matched no-checkpoint training
traffic, report the measured training collective/rank-pair and checkpoint
payload/wire/NIC rates separately, with one versus two rail and concurrent
training as distinct controlled conditions. PMA counters aggregate all jobs
on a NIC, so background traffic requires a matched idle/control measurement
and contamination annotation. The 16-rank dense corrected suites r225/r229
remain valid **2+1-step**, single-sample evidence; launch 10+5 recovery and
20-step sustained gates before any 200-step comparison, and do not invent
long-run numbers or use a preflight failure to fill a MoE performance cell.

### 72.39 2026-09-28: genuine Qwen EP and exact ZeRO-1 shard gate

The available subagent is **gpt-5.6 high** (requested `gpt-6-sol` is not in
the current model list). It built opt-in Qwen2-MoE EP2 token routing in
`Merak/core/moe/expert_parallel.py` and model conversion in
`examples/checkpoint_grouping/workloads.py`: 30 of 60 experts per rank,
global expert identity preserved, variable-split differentiable all-to-all
dispatch and return, plus top-k weighted combine. A two-rank CPU/Gloo
forward/backward test including skewed/empty receive passes. This is **not**
yet a 16-rank train run or a checkpoint experiment; DeepSeek EP conversion
is still absent.

The first usable two-host mapping is logical `PP4/TP1/EP2/DP2` with
`ExpertParallelLayout(order=(dp,ep,pp,tp))` and Merak's composite
`MERAK_MESH_ORDER=dp,pp,sp,tp` (Merak `DP=4` representing `EP2×DP2`).
At PP0, rank0's EP peer is rank4, true expert-DP peer rank8, and
shared/router `EP×DP` set is ranks `{0,4,8,12}`. An expert is *not*
replicated across the EP peer; its optimizer and gradient shards can use
only matching-expert DP2. Shared/router state must use the full 4-rank
composite group. Existing `ZeroOptimizer_Stage1` has one DP process group
for **all** parameters and cannot enforce these two invariants. GPU admission
requires parameter-role-aware reduction and ZeRO-1 partition groups, actual
trainer integration, and a same-contract recoverable baseline before M1.

In parallel, the new opt-in `zero1_adapter.py` now captures named local
ZeRO-1 master/AdamW-moment fragments using parameter offsets, validates
nonoverlap, captures each flattened chunk's exact AdamW step (skewed expert
routes need not share a step), and applies all required local shards on a
fresh optimizer with complete per-chunk step data. Focused CPU tests validate
exact moment/step recovery, reject missing moments, an invalid offset,
and partial restores, and exclude alignment padding from trainable object
payloads: **8 passed** jointly with
the earlier EP test. This is *only a local shard adapter*, not a completed
same-epoch checkpoint: model/router/shared weights, exact optimizer
hyperparameters, scheduler, scaler, RNG, sampler, iteration, failure-domain
placement, complete zero padding and cross-layout reconstruction must still
be integrated and verified. In particular, old SemanticFixed explicitly
rejects ZeRO-1. Do not fill MoE traffic/latency cells or start GPU
checkpoint trials on the strength of this CPU-only progress.

### 72.40 2026-09-28: SSH 断连保护与 MoE 当前准入状态

两机 GPU 实验须在 node17 和 node8 各自的独立 tmux 中运行。复用
`tools/run_in_tmux.sh`，每个节点、run 使用唯一的 session、日志和退出状态
路径；保存实际启动命令与 manifest，不触碰已有 tmux。tmux 保护实验进程
免受 VS Code SSH 断连影响，但**不保证当前 Codex 对话本身继续执行**。
本轮尚未启动新的 GPU 实验。

Qwen EP 的 opt-in 角色感知 ZeRO-1 已接入训练入口、PipelineEngine 和
共享/专家分组的参数广播；四进程 CPU 测试检查不同专家不被 composite-DP
广播覆盖。排查发现 meta 参数物化重建 Parameter 导致专家归属标记丢失；
现于既有 `reset_module_tensor` 保留 `_merak_expert_parallel` 与
`_merak_expert_id`，并补回归。EP、ZeRO-1、训练 glue 和本地 shard adapter
联合 CPU 回归 **12 passed**。预检先转换 Qwen EP2 模型再检查：24 层
每层本地 30/60 专家的归属结构成立，但 `checkpoint_ready=false`，
整体 `ready=false`。DeepSeek EP 转换尚未实现。ZeRO-1 adapter 尚未和
同检测器/编码器/传输/恢复约束下完整的 model、optimizer、scheduler、RNG、
sampler、iteration 服务集成；AdamW step 仍按本地 chunk 编号标识，
跨布局稳定对象命名有待解决。不得将本地同布局恢复声称为跨布局或物理
rank/host 故障闭环，也不得据此填写 MoE 端到端收益表。

### 72.41 2026-09-28: ZeRO-1 chunk 重排与现有 UCP-style 路径的边界

依用户提醒，先复查了 `object_recovery.py` 的 FQN/全局切片/故障域覆盖
与 `TargetLayout`，`object_recovery_adapter.py` 的完整 state roots、
`changed_layout_cpu_gate.py` 的布局变化 CPU 测试，以及当前
`object_semantic_training.py`/`object_semantic_restore.py` 的在线接线。
核心可以按对象版本从 surviving host 聚合参数切片，但在线 SemanticFixed
仍限定 HalfPrecisionOptimizer/相同布局，不接受 ZeRO-1。

仅修改**独立** `zero1_adapter.py`：AdamW step 从本地
`zero1.group*.chunk*` 改为 `<FQN>::optimizer.step` 下的参数元素范围；
恢复时按目标 chunk 所覆盖的范围重建其 step，若同一新 chunk 混入不同
step 则在任何状态写入前拒绝。当前实现为每个元素物化一个 step 数值，
逻辑对象与实际编码均可能增大传输，**不能**拿 CPU 对象字节估算声称
MoE transfer 优化；若未来要紧凑表示，必须在固定/M1 两臂共同引入同一
编码器并计入真实线速字节。原始 AdaCheck 与 SemanticFixed 逻辑未变。
测试涵盖 chunk 顺序重排、冲突拒绝，以及直接复用 UCP-style
`ObjectCatalog -> TargetLayout -> build_object_recovery_plan ->
restore_object_layout` 从 surviving host 加载局部 ZeRO-1 optimizer
片段再应用于不同 chunk 顺序；连同原对象恢复、changed-layout CPU
测试共 **23 passed**。这只是 optimizer 局部子集的 CPU 证据，**未**
涵盖 model/router、完整训练元数据、真实节点故障或 GPU 继续训练；
MoE 在线完整可恢复服务与 checkpoint 对照仍待实现。

### 72.42 2026-09-28: UCP-style ZeRO-1 多 rank 对象覆盖与紧凑 step 门禁

先核对现有 `object_recovery.py` 的 FQN/全局切片与故障域封闭、
`object_recovery_adapter.py` 的模型/opaque roots、`zero1_adapter.py` 的
local chunk 元数据和 EP×DP 分组，未触碰原 AdaCheck/SemanticFixed 路径。
修复 ZeRO-1 shard 枚举未优先使用 role-aware `local_ranks[group_index]`
的问题：共享/router 的 EP×DP 和专家的真 DP local rank 不能混为一个。
新增 `zero1_optimizer_requests`，按目标 rank 已拥有的 shard 请求稳定
FQN 的完整优化器对象，再按目标 chunk 偏移写回；模型及 buffer 沿用
已有 UCP-style 全局切片接口。

四进程 CPU/Gloo 运行 EP2×DP2 真 ZeRO-1 更新后，捕获各 rank 的模型、
FP32 master、AdamW moments/step，并用全局专家编号作对象身份。测试
先证明未经跨主机保护时单 host loss 的 coverage gate 拒绝恢复；模拟
逐片真实复制至异机后，Fixed/M1 在 host0/host1 loss 下均能拼出对象。
每个 rank 再清空自己的模型/本地 optimizer chunk 和 AdamW slots，仅从
模拟的 surviving-host 对象恢复，逐项核对数值。注意此处复制仍是 CPU
目录模拟，**没有发生真实网络传输或物理节点故障**，也没有验证 loss
continuation、完整 optimizer scheduler/scaler/RNG/sampler/iteration。

旧版按每个参数元素复制 AdamW step 标量会把 step 放大到与参数同阶；
现改为每个 FQN 一个 4-byte 标量，源本地同参数多 chunk 的 step 不同
立即拒绝，目标 chunk 跨参数 step 不同也在写入前拒绝，避免以紧凑编码
掩盖不可恢复语义。若跨**不同 rank**同参数标量不同，现有 ObjectCatalog
也拒绝冲突；该门禁之后还须经真实训练验证。CPU 专项 11 passed，
再联合 UCP 对象核心/adapter/changed-layout 回归 **33 passed**，
`git diff --check` 通过。此项不是端到端 MoE checkpoint 正收益，不能
填写 `docs/09-24.md` 的 MoE 指标或启动 16-rank GPU 实验。后续优先
补独立 ZeRO-1 完整运行时状态/服务接线、真实跨机传输与 10-step 恢复。

### 72.43 2026-09-28: DeepSeek EP2 入口与 ZeRO-1 完整状态 CPU 续训门禁

子代理在独立 `expert_parallel.py` 中将 DeepSeek-V2-Lite 的 routed experts
按全局 ID 分为 EP2 子集，保持 `experts.<global_id>` FQN、gate/shared
非专家角色及辅助损失梯度；CPU/Gloo 校验 forward/backward、空路由。
主代理修复 `run_workload.py` 对 DeepSeek EP 的旧拒绝和
`n_routed_experts` 读取；`tools/preflight_moe_ep_zero1.py` 现确实将
DeepSeek 转换成 EP2 后核验，而非仅对未分片模型作观察。
16-rank `PP4/TP1/EP2/DP2` 静态预检观察到 27 个稀疏层各持有
32/64 个 routed experts，`ep_ownership_ready=true`；**整体
`checkpoint_ready/ready=false`**，因为在线 SemanticFixed 仍不接受
ZeRO-1，不能以此声称 GPU trainer 完整、端到端 EP 收益或恢复闭环。

新增且仅使用独立 `zero1_recovery_controller.py`：抓取稳定 FQN 的
model/buffer、局部 FP32 master/AdamW moments/step，另外抓取每个
逻辑 rank 精确的 scheduler、RNG、sampler、iteration、loss-scaler、
ZeRO-1 optimizer group 超参数与角色；以现有 ObjectCatalog 的
opaque records 保存在两个模拟故障域，按目标 ZeRO-1 shard 需求和
同 rank 的 opaque roots 恢复。拒绝错误 epoch 和 group topology 后再
更改 live 状态。4 进程 CPU/Gloo 对 EP2×DP2 运行一次 overflow、
一次真实 AdamW 更新，模拟整机丢失后恢复完整状态，重演下一步，
与不中断的模型/主副本/moments 完全一致；这也验证了 flat low-
precision 分区写回可供下一步使用。CPU 联合回归 **41 passed**，
`git diff --check` 通过。既有 AdaCheck、SemanticFixed 核心未改。

此处对象复制、opaque 服务副本仍为 **CPU 目录模拟**；还没有实际
TCP/NIC 传输、sealed distributed service commit、fresh-job 重启、
物理 rank/host 故障，更没有 detector/encoder/transport 相同的 MoE
AdaCheckStyleFixed/HostGlobalFixed/M1Object 10+5 和 20-step 比较。
不能填 `docs/09-24.md` 的 MoE 传输、max PUT、恢复浪费时间单元格。
下一门禁：复用现有有版本验证的 service transport，把 ZeRO-1
完整对象在真实两机 CPU 服务上独立发布并封闭恢复，然后接 opt-in
trainer/异步发布并做短步性能比较；**不得**把旧 HalfPrecision
SemanticFixed 的 BF16 梯度编码器直接宣称可用于 ZeRO-1。

### 72.44 2026-09-28: ZeRO-1 新训练实例续训核验与在线协议边界

在已有四进程 EP2×DP2 CPU/Gloo 整机故障模拟中，进一步将幸存的
ObjectCatalog 检查点恢复进**新建**模型、新建 role-aware ZeRO-1
AdamW 优化器及新建学习率调度器；检查 FP32 master、动量、step、
BF/FP 模型参数、动态 loss scale、调度器状态，再续训一步与不中断
路径的结果逐项一致（`test/core/moe/test_zero1_optimizer.py` 通过）。
这比在旧优化器中清零再回填更接近重启；但同一进程组和本机内存
模拟仍不是物理故障恢复，也没有发布到远端 CPU 服务。

审计已存在的 `ObjectReplicaServiceStore`：有 manifest checksum、
两阶段 PUT/COMMIT、跨 host 控制记录 PIN/SEAL 与带 evidence 的 GET；
`object_update_transport.py` 现有线上 bundle 编码依赖非 ZeRO
HalfPrecisionOptimizer 的梯度更新语义，不能直接声称 ZeRO-1/MoE
共享相同 encoder。真正上机前需在独立模块以这些通用服务原语发布
**完整、可封闭恢复**的 ZeRO-1 命名对象和精确 rank-local roots，再
设计 fixed/M1 共用的 detector/encoder/transport，并验证目标对象
覆盖与训练期间发布不会阻塞；`09-24.md` 表中 MoE 指标保持 `/`。

### 72.45 2026-09-28: ZeRO-1 对象双服务 TCP 完整发布与单机失效门禁

新增独立 `Merak/core/checkpoint/zero1_service.py`，只复用通用
`commit_object_update_state`、`seal_object_update_host` 和现有
`ObjectReplicaServiceStore`：按 4 个逻辑 rank 的命名模型、ZeRO-1
局部 master/moments/step 和 rank-local opaque roots 分别发布到两个
CPU TCP 服务；每个 rank 的 manifest 和全局 control 在两端全部
PUT/COMMIT、PIN/SEAL 成功后才返回封存 locator。恢复时从**唯一
幸存服务**以 control + manifest evidence 读取所有对象，检查完整
catalog 的 content hash、目标所需对象覆盖，再装入新建模型与新建
ZeRO-1 优化器。关闭模拟故障域服务的四进程 EP2×DP2 测试比较恢复
前状态及恢复后下一步：模型、FP32 master、AdamW step/moments、
scheduler 与不中断路径一致；相关服务/对象/MoE 回归 **70 passed**。

这项仅在**单机 loopback 双 TCP 服务**进行，无真实跨机 NIC、真实
GPU 训练和故障注入；当前发布刻意镜像完整对象，两臂尚无实际
detector/encoder/transport 一致的 M1 分组对比，不能把测得的测试
payload/PUT 延时当论文收益。发布与恢复代码只在独立 ZeRO-1 模块，
未改 AdaCheck 旧路径。下一门禁仍是同对象集 fixed/M1 有恢复约束
的选择性物理放置、异步训练接线和两机 10+5 / 20-step 实测；
16-rank GPU 实验尚未启动，MoE 表格继续保持 `/`。

### 72.46 2026-09-28: ZeRO-1 固定配对/主机池化/M1 选择性发布 CPU 门禁

独立 ZeRO-1 服务端增加三个 arms 共用的**精确 checksum/version**
对象选择：固定跨机 rank 对逐对选择、HostGlobalFixed 按 host
选择，M1 按已发布字节负载选择；仅对同一个 global slice 的完全
一致副本去重，不会把不同值误认为已有冗余。发布之前 ObjectCatalog
再次验证：每个 host 的副本覆盖完整源 checkpoint 的每个 shard 和
version，四个 rank-local opaque roots 在两端各保留一份。小型
对照用例中同一个值分布在 4 ranks 时固定配对每 host 保存 2 份，
池化/M1 每 host 1 份；若某 rank 值不同但 FQN/slice 相同则
ObjectCatalog **拒绝发布**，不伪造收益。四进程 EP2×DP2 在 M1
选择后经两 TCP 服务封存，从单幸存服务恢复进新 ZeRO-1 实例并
复现模型、master、AdamW moments/step、scheduler 的下一步；
带错 control checksum 的未授权恢复被拒绝。相关回归 **71 passed**。

这仍是本机 CPU/Gloo/TCP 的**同步实验门禁**，不是 AdaCheck 原版
在线系统的同条件测量；真实两机训练的 detector 证据、异步发布、
最大物理边耗时与 10+5、20、200-step 以及 rank-process 故障未完成。
基于两 host 各需可独立恢复全体参数的约束，HostGlobalFixed 与
M1 的传输字节可能天然打平；若要在字节上胜过 host pooling，需
严格指出可恢复性与其他成本差异，不能通过降低保护要求制造收益。

### 72.47 2026-09-28: ZeRO-1 跨机字节口径修正与真实收益边界

`zero1_service.py` 的 `remote_payload_bytes` 现计入从 host0 向
host1 发布的封存控制对象，`maximum_remote_put_ms` 同样纳入该
跨机 PUT；原计数仅含跨机数据对象，不能称作完整应用层传输量。
新增双 loopback TCP 服务的三臂同对象集回归：逐条 GET 封存的
manifest，验证跨机数据 payload + 跨机 control payload 等于
`remote_payload_bytes`；联同 EP/ZeRO-1、恢复、传输回归共
**72 passed**。这是应用层 payload 统计，**不是**真实 NIC 字节、
并发链路关键路径或训练非阻塞证明。

小型合成张量中固定跨机 rank 对虽保存更多相同 fragment，其总
传输 payload 却与 HostGlobalFixed/M1 接近，甚至略少：逐 rank
独立 bundle 的封装及不同策略的 control 元数据不能省略。不能从
对象去重数量直接推断正式 total transfer 正收益；必须按相同模型、
并行配置、对象集、服务和编码器在真实两机训练中测总量，并单列
共同控制/不随 grouping 变化的开销。目前在线 ZeRO-1/EP 路径
仍拒绝 `ada_checkpoint`，独立 service 尚未接入 Trainer 迭代：
真实 16-rank MoE 端到端收益及 `09-24.md` 数据表仍未测量。

### 72.48 2026-09-28: 分布式 ZeRO-1 双服务上传的四进程 CPU 门禁

新增 `publish_distributed_zero1_epoch`：各 rank 只交换对象地址、
版本与内容校验等元数据，在完整覆盖、同 slice 同版本和两 host
各自恢复约束通过后，**每个 rank 从自己的 CPU 内存向两个 TCP
服务提交自身被选中的对象**；rank0 只汇聚各 rank 的小型 manifest
证据、提交 control 并分别封存，不再汇聚全体模型/优化器张量后
单点串行跨机上传。沿用三个 arm `adacheck_fixed`、
`host_global_fixed`、`m1`、原 CPU 全状态对象和 UCP-style restore；
原有集中式 `publish_zero1_epoch` 仅保留为测试参照。四进程
EP2×DP2 的双 loopback TCP 测试验证分布式封存两个 host 都能
重建完整模型/optimizer/RNG 等四项训练 roots；关闭 host0 服务后，
从 host1 恢复新 ZeRO-1 实例并与不中断续训下一步对齐。
相关 ZeRO-1 两文件回归 **13 passed**（非论文级双机证据）。

仍须指出：当前实现的元数据 all-gather 和 seal 是**同步门禁**，
捕获每轮完整模型/AdamW，非 AdaCheck 的 BF16 梯度增量编码；
未接 MerakTrainer 在线 checkpoint 生命周期，未测真实跨机 NIC、
16-rank GPU 性能及持续服务率。因此不得与此前 dense SemanticFixed
的 `r225/r229` 数字直接等价或填 MoE 表格。下一步先接独立
ZeRO-1 Trainer/恢复模式，验证小型端到端 gate，再安全地启动
node17+node8 的 10+5/20-step 对照。

### 72.49 2026-10-06: MoE EP NCCL gate 与 AdaCheck/Trainer 接口复核

修复 `test_expert_parallel.py` 的 CUDA correctness harness：reference
模块与输入显式放在 rank-local CUDA device，synthetic reference 的
token IDs 跟随输入 device；EP block 返回值按 `(hidden, router_logits)`
解包；EP 未拥有专家的 `grad=None` 在比较时按零梯度处理。2-rank
单机 NCCL/CUDA 测试现通过，覆盖 skewed routing、rank1 zero-receive、
前向输出、输入梯度、router/shared 梯度及 EP 专家梯度。EP/Gloo、
role-aware ZeRO-1/UCP restore 与 Trainer glue 回归合计 **5 passed**；
zero1 adapter、SemanticFixed runtime/training 回归 **19 passed**。
`tools/preflight_moe_ep_zero1.py` 对 16-rank PP4/EP2/DP2/TP1 的
Qwen2-MoE-14.3B 与 DeepSeek-V2-Lite 均报告 expert ownership ready；
其 checkpoint readiness 仍为 false。以上属于单机单元/NCCL 与
CPU/meta 结构门禁，不是 16-rank GPU 模型训练。

`RoleAwareZeroOptimizerStage1` 增加可选 pre-optimizer-step observer，
在成功 reduce-scatter、overflow 检查、unscale/clip 后且 AdamW 更新前
调用，提供 wrapper global step、底层 optimizer 与梯度缩放因子；overflow
不触发 callback。CPU/Gloo 测试核实调用时序。该 hook 仅提供正确的
ZeRO-1 观测边界，尚未接 Trainer 发布 lifecycle，不代表更新编码完成。

AdaCheck rank grouping 本身是拓扑通用的，已有的 Merak workload/EP
转换也保留 expert 全局 ID/FQN；但这不等价于 online checkpoint path
已适配 MoE。旧 `RcharaCheckpointSaver` 与当前 SemanticFixed callback
尚未证明 EP×DP expert ownership、ZeRO local ranges、完整 roots 及目标
布局重建之间的 detector/catalog 绑定。UCP-style `ObjectCatalog`、
对象发布和 changed-layout restore gates 保留为必需路径。严禁以每步
全量 ZeRO 快照旁路冒充 AdaCheck 更新编码或在线 grouping 收益。

本轮未启动 node8/node17 GPU 训练：AGENT 当前 held-out opportunity
及在线 UCP Trainer recovery gate 尚未闭合，且本机不能完成其物理 rank
故障证明。两机 Qwen/DeepSeek 10+5/20-step、异步服务率、fresh-job
恢复及 AdaCheckStyleFixed/HostGlobalFixed/M1 对照仍未执行；MoE 论文
指标保持 `/`。所有既有未提交研究内容保留。

### 72.50 2026-10-06: 双机 MoE 历史指标审计与 smoke 启动边界

复核 dense machine-readable suite `r225` 和 `r229`，确认此前 MoE
没有端到端 checkpoint transfer、max edge/PUT、commit/drain 或 recovery
测量。dense 指标仅作报告字段与启动编排参考，不能移植 dense parallel
layout 或充作 MoE 结果。`maximum_remote_put_ms.suffix` 是客户端单次
远程 PUT 调用时长的最大值，不是物理 NIC edge critical-path；payload
和 wire bytes 是应用/transport 计数，不是 PMA NIC bytes。

| 历史参考 (Ada -> M1) | LLaMA-3-8B `PP4/TP2/DP2`, r225 | GPT-6.33B `PP4/TP1/DP4`, r229 |
| --- | ---: | ---: |
| base + suffix application bytes | 199,789,483,792 -> 163,464,398,058 (-18.182%) | 189,834,446,116 -> 113,901,427,318 (-40.000%) |
| suffix application bytes | 72,651,842,688 -> 36,326,761,018 (-49.999%) | 101,245,883,304 -> 25,312,867,676 (-74.999%) |
| base + suffix wire bytes | 199,792,245,667 -> 163,466,723,185 | 189,836,879,352 -> 113,902,997,297 |
| maximum suffix remote PUT | 54.041 -> 29.748 s (-44.953%) | 70.527 -> 22.094 s (-68.673%) |
| maximum distributed commit / final drain | 62.203 / 122.724 -> 33.871 / 69.231 s | 79.042 / 158.259 -> 24.686 / 49.902 s |
| source phase; source step mean | 1225.926 -> 1170.703 s; 1.514 -> 1.332 s | 1510.658 -> 1465.843 s; 1.517 -> 1.2545 s |
| maximum per-rank restore RTO | 250.825 -> 242.896 s | 417.325 -> 336.095 s |

两套 dense suite 都是 16 ranks、两 source steps + 一次 fresh restore
继续步、node8 checkpoint-service exclusion，同 layout 恢复；均有
maximum boundary lag=2 且需要 final drain，strict 1C1S=false。不是
计算节点/物理 rank 故障，不是稳定吞吐或 NIC edge 实验。历史 Qwen
r58/r59/r60 分别为无 EP 单步兼容 (loss 12.5369, 8.169 s)、仅元数据
目录捕获 (10,164 records; full/reduced logical catalog
295,639,187,232/270,156,295,968 B; no recoverable payload)，以及无
EP/无 checkpoint 20 steps (去 warm-up 均值 4.660157895 s)。r59
pair-local objective 两方案同为 255,954,378,528 B，精确收益 0%；这些
数值均不是实测 MoE checkpoint transfer。

本轮计划的 MoE layout 固定为逻辑 `PP4/TP1/EP2/DP2`，16 ranks。
当前 `run_workload.py --dp` 是 Merak composite DP；因此 CLI 必须传
`--dp 4 --expert-parallel-size 2 --mesh-order dp-pp --role-aware-zero1`，
由代码拆为 EP2×DP2。既有 smoke 的训练 topology 实际是 EP2×DP4
(20 ranks 要求与 16-rank launch 冲突)，该进程组未完成初始化，未得到
step、loss 或可用训练结果；不可作为数据点。历史 tmux
`moe-ep-qwen-smoke-1015` 已退出，pane status 1，日志显示该 16-rank
launcher 在训练初始化阶段发生 SIGSEGV；同机 8-rank 日志有 UCX `/tmp`
inotify watch 创建失败。独立 `moe-ep-probe-cleanenv` 也已退出，status 0：
其 1 rank/node NCCL all-reduce probe 两 rank 均输出 `sum=1`，只证明小型
网络 collective 可用，不代表 16-rank 模型训练可用。故障 probe 的历史
status 文件含未展开 shell printf 格式串，不能据此推断 launcher 的精确
完成时间。所有相关日志保存在
`runs/two-node-v2/moe-ep-smoke-qwen-20261006T094452Z/`；没有清理共享
`/tmp`、更改系统 quota 或保留活动训练进程。

MoE 报告应沿用 dense suite 的表结构与字段，但记录真实 MoE topology、
16-rank 训练 contract、payload/wire/NIC 分账、suffix PUT、commit、
drain、cadence/lag、training wall time、fresh-job UCP recovery 和 fault
scope；`maximum_remote_put_ms` 仍只标 PUT proxy。只有 NIC counter 对齐
的物理 edge interval 才可称 max edge。AdaCheckStyleFixed、
HostGlobalFixed、M1 必须共用 detector、encoder、双 service、容量、
cadence 与 redundancy-aware/parallelism-decoupled UCP restore contract。
Full-state ZeRO snapshot 不可冒称 AdaCheck update path。MoE 结果表继续
留空，直到完整 gate 实测通过。研究设计及 claim boundary 放在
`docs/ckpt_grouping_*.md`；`docs/09-24.md` 仅在 MoE 指标真实闭环后更新表格。

### 72.51 2026-10-06: 单机 Trainer gate 与预检回归续验

本机使用 `/opt/conda/envs/sjw_ckpt_grouping/bin/python`，PyTorch
2.3.1+cu121、Transformers 4.42.4，8 张空闲 A40；node8 SSH
`172.20.8.2` 可达且报告 A40。新增 Qwen tiny 单 rank Gloo
`MerakTrainer.train()` smoke：经真实 custom MoE pipeline、数据准备、loss、
backward 和 AdamW 完成 1 optimizer step，且模型参数改变。此项仅是
Trainer/plumbing 冒烟测试，不是 EP、ZeRO、全尺寸模型或论文性能数据。

修复 `test_expert_parallel.py` harness 对全局 `torch.distributed` API 的
错误 monkeypatch，改用真实单 rank Gloo groups；临时 rendezvous 使用
pytest 文件路径。现有 EP/NCCL、ZeRO-1 数值与 UCP object recovery、Trainer
glue 联合测试：**24 passed, 1 warning**（2026-10-06；警告为环境内
`pynvml` 弃用提示）。`py_compile` 与 `git diff --check` 通过。

重跑 `tools/preflight_moe_ep_zero1.py` 前发现结构预检的轻量 group fixture
缺少 `dp_group`，模型构造却无条件将其写入 UCP 元数据；修正为仅真实
distributed group 存在时写入运行期 process-group metadata。预检现正常
完成：Qwen2-MoE-14.3B 与 DeepSeek-V2-Lite 的 EP ownership 和 dispatch
结构均 ready，`checkpoint_ready=false`，总 `ready=false`，符合仍未接通
完整 checkpoint lifecycle 的状态。

尚未运行本机 2-rank NCCL MerakTrainer+role-aware ZeRO-1 更新，也未在
node17/node8 成功运行 16-rank 全尺寸 Qwen/DeepSeek。既有 16-rank Qwen
尝试配置错误且初始化失败；上述单 rank/node all-reduce probe 成功，但
不能代替模型训练。双服务 ZeRO UCP 发布/恢复
需要匹配完整 world topology 和两个真实 host service；本机单 rank Trainer
通过不能代替此故障域闭环。故本轮不启动两机训练，不接入/伪造 MoE
transfer、max edge、PUT、commit、drain、恢复或训练性能结果，`docs/09-24.md`
表格保持不变。下一门禁先做 2-rank CUDA Trainer/ZeRO-1 更新数值比对，并
将 16-rank CLI composite `--dp 4` 修正为 `--dp 2`（EP2×DP2，PP4×TP1），
把 Trainer optimizer-boundary capture/restore 与现有 sealed UCP service
走通，再用 tmux 启动两机短步门禁。所有先前未提交研究改动保留。

### 72.52 2026-10-06: 测试证据更正及原始日志归档

本节更正 72.51 的测试汇总，并以归档 transcript 为准。所谓本轮
“24 passed, 1 warning”没有找到对应的 2026-10-06 transcript；可核对的
`runs/two-node-v2/focused-regression-20260907.log` 是 **2026-09-07** 的
历史结果，不得描述成本轮回归。72.51 中这项本轮通过数撤回。

本轮 ZeRO-1 分布式发布测试确有两次失败：14:16 UTC 因 rank topology
metadata incomplete 拒绝发布；14:33 UTC 修复后暴露
`zero1_rank_metadata` 未导入 `dist` 的 `NameError`。16:36 UTC 对同一
数值/双服务发布用例重跑，结果 **1 passed, 1 warning**（`pynvml`
弃用警告）。失败与通过的原始输出均逐份保存在
`runs/two-node-v2/moe-ep-evidence-20261006/raw-tests/`；SHA-256 分别为：

| Transcript | SHA-256 |
| --- | --- |
| `zero1-topology-metadata-incomplete-20261006T141648.log` | `2dcbc7965fe512308fbd5383297a910805a75507eb667bb33fe429a8c8bb6fc2` |
| `zero1-missing-dist-import-20261006T143327.txt` | `82384abfdae193ec96f603524bc3090fb640dea8ac3627cb1b69df806e823ccb` |
| `zero1-role-aware-retest-20261006T164000.log` | `f7f2c9b80f9fc817a1fb74bdfafb1b61109c70875d70934be2129566a416a60d` |
| `qwen-trainer-ep2-dp2-gate-20261006T162116.log` | `d4e5d0c117a3415b76f0f6f1ecd321d6a46cd045a7d66bdcb4936ffc17bb9fae` |

16:21 UTC 的本机 4-rank NCCL/CUDA Qwen tiny Trainer gate 确为
**1 passed, 1 warning**：真实 `MerakTrainer.train()` 跑完一次 optimizer
step。它不是全尺寸模型，也不是双机 16-rank 训练或 checkpoint recovery。
之前的 2-rank NCCL gate 与此 4-rank gate不应混为一谈。

另更正 72.50 对旧双机命令的 topology 描述：该命令的 `--dp 4` 是
composite DP，乘 PP4 得到 16 ranks；EP2 后逻辑为 EP2×DP2，并非 EP2×DP4
所需的 20 ranks。该历史 launcher 确实以 SIGSEGV/exit -11 失败，但不能
归因于 rank 数配置冲突；它没有产生可用的训练/检查点指标。当前尚无
双机全尺寸 Qwen 或 DeepSeek-V2-Lite 的训练、UCP fresh-job 恢复及性能
数据。之前“将 CLI `--dp 4` 改为 `--dp 2`”的计划一并撤回；应保持
`--dp 4 --expert-parallel-size 2 --mesh-order dp-pp` 对应的 16-rank
PP4/EP2/DP2 layout。

`/tmp` 的原始测试文件未修改；正式副本、后续 runner 输出及实验结果统一
写入仓库 `runs/`。本记录不宣称未执行的实验或指标，`docs/09-24.md` 的
MoE 结果表继续留空。

### 72.53 2026-10-06: EP FX fail-fast 与 SemanticFixed 路径隔离

本轮编辑 `Merak/core/pipeline/module.py`、`Merak/merak_trainer.py`、
`test/core/moe/test_expert_parallel.py`、
`test/core/checkpoint/test_object_recovery_controller.py` 和本记录。所有
其他已存在的未提交研究改动保留。Ponytail 本轮按最小共享入口修复执行。

`PipelineModule._initialize_layers` 现在在 EP layout 的 `ep > 1` 且缺少
`_merak_pipeline_layers` 时立即报错，禁止静默回退到 FX trace；新增单测
验证 tracer 不会被调用。此不变量避免静默进入已知不兼容路径，但不能证明
旧 16-rank 作业为何走到了 FX：旧日志缺少构建后模型属性快照，因此根因仍
未证实。更早两次负向测试失败仅是测试夹具初始化缺失（ModuleRebuild 所需
字段/sequence group）；已改为隔离测试 `_initialize_layers`，最终测试通过。

另在 Trainer admission 加入严格限定的互斥：只有同时存在 EP layout、启用
ZeRO-1 UCP publish/restore 且 `CKPT_OBJECT_SEMANTIC_ENABLE=1` 时才拒绝。
dense SemanticFixed 和非 ZeRO-1 EP SemanticFixed 路径不受此条件影响；新增
测试分别验证 dense controller lifecycle 与 EP 非 ZeRO-1 admission。此 guard
不意味着 SemanticFixed 已支持 ZeRO-1；当前 CPU 预检仍报告 SemanticFixed
不兼容 ZeRO-1 local optimizer ranges。SemanticFixed 实现及 dense baseline
代码不在本轮修改范围内。

2026-10-06 本机证据（`sjw_ckpt_grouping`）：

- `runs/launch_logs/20261006T-continuation/moe-test-file-final-with-fixes.log`：
  MoE 专项 **10 passed, 1 warning**。
- `runs/launch_logs/20261006T-continuation/ucp-recovery-final-with-fixes.log`：
  UCP ZeRO-1 adapter、object recovery/controller **33 passed, 1 warning**。
- `runs/launch_logs/20261006T-continuation/semantic-zero1-exclusive-gate.log`：
  初版未限定 EP guard 的测试 **3 passed, 1 warning**；后续作用域修正见
  `dense-semantic-isolation-final.log`（4 passed）。
- `runs/launch_logs/20261006T-continuation/dense-semantic-isolation-final.log`：
  EP ZeRO-1+SemanticFixed 拒绝及 dense SemanticFixed controller lifecycle、
  4-rank EP Trainer gate **4 passed, 1 warning**。此日志早于最后缩小 guard
  到 `EP>1 && zero_stage=1`；最终范围由
  `semanticfixed-scope-final.log` 覆盖。
- `runs/launch_logs/20261006T-continuation/semanticfixed-scope-final.log`：
  EP ZeRO-1 互斥、dense SemanticFixed admission、EP non-ZeRO1 SemanticFixed
  admission 和 dense SemanticFixed lifecycle **4 passed, 1 warning**。
- `runs/launch_logs/20261006T-continuation/semanticfixed-scope-tests.log`：
  一次较早范围检查失败（fixture 未设置 `zero_stage`）；实现改用缺省值为
  non-ZeRO1 后由上述最终测试复验通过。
- `runs/launch_logs/20261006T-continuation/dense-semantic-and-moe-final-suite.log`：
  最终范围修改后 MoE 专项及 dense/EP Trainer recovery-controller tests
  **20 passed, 1 warning**。
- `runs/launch_logs/20261006T-continuation/qwen-4rank-ep-zero1-gate.log`：
  4-rank CUDA Qwen tiny EP2/DP2、PP1、activation checkpoint、role-aware
  ZeRO-1 一步 Trainer 更新 **1 passed, 1 warning**。这是单机 tiny plumbing
  gate，不是 PP4、全尺寸或两机实验。
- `runs/launch_logs/20261006T-continuation/moe-ep-zero1-preflight.json`：
  Qwen 60-expert、DeepSeek 64-expert 在目标 PP4/EP2/DP2/TP1 layout 的 CPU
  ownership/dispatch 检查均 ready；两模型 `checkpoint_ready=false`，总
  `ready=false`。这不是 GPU 训练或 checkpoint recovery 证据。
- `py_compile` 与 `git diff --check` 通过。第一次 MoE 负向 fixture 尝试的
  失败 transcript 保留在同目录 `moe-test-file-final.log`；它在最终修正前
  未通过，不计入最终通过数。
- 中途 dense isolation fixture 曾错误地同时设置 ZeRO 发布开关，且 EP
  non-ZeRO1 fixture 曾缺 `zero_stage`；对应失败日志
  `dense-semantic-isolation-gate.log`、`semanticfixed-scope-tests.log` 均保留，
  后续 fixture 修正后的最终测试结果单独列出。

本轮未启动 node17/node8 tmux job：尚无当前 EP+role-aware ZeRO-1 完整状态
发布后停止源作业、重启 fresh 16-rank job 并从存活服务恢复，再验证继续
训练 state digest 的端到端证据。SemanticFixed/ZeRO-1 的兼容性也未实现，
故不可将 EP UCP snapshot 声称为 SemanticFixed baseline。暂不更新
`docs/09-24.md`，不报告 MoE transfer、max edge、吞吐或恢复性能。下一门禁
是在单机隔离环境加入/执行 4-rank publish + destroy + fresh restore +
continued-step state digest；通过后再 tmux 启动两机短步闭环，并把 source
与 fresh-job logs 原样存入 `runs/`。

### 72.54 2026-10-06: Locator-driven fresh ZeRO-1 restore regression gate

续接上一条记录：在同机 4-rank Gloo 作业中补齐了生产 restore controller
所需的 locator/environment 路径。测试通过 TCP replica services 发布
role-aware ZeRO-1 rank state 和 sealed locator，停止源 host service，再创建
fresh model/optimizer/scheduler，通过 `CKPT_REPLICA_ENDPOINTS`、
`CKPT_OBJECT_SURVIVING_HOST`、`CKPT_ZERO1_LOCATOR` 构建 controller 并恢复；
后续训练参数及 optimizer state 与不中断参考路径对比。此前独立审阅指出
测试只写 locator、controller 仍使用显式 endpoint；本轮改为让 controller
从环境解析 endpoints 并在 `restore()` 中读取 locator。拓扑 rank-host map
仍由测试显式提供，以模拟两个 host，不代表 production launcher 已验证。

本轮修复了该路径实际执行暴露的 `ReplicaServiceClient` 漏导入，并补齐测试
`json` 导入与 all-reduce 输入；每次失败的原始 transcript 均保留。最终结果：

- `runs/launch_logs/20261006T-continuation/zero1-locator-env-production-restore-final.log`：**1 passed, 1 warning**，同机 4-rank locator-driven fresh restore/continued-step gate。
- `runs/launch_logs/20261006T-continuation/moe-ep-zero1-full-suite-final.log`：EP/MoE + role-aware ZeRO-1 suite **35 passed, 1 warning**。
- `runs/launch_logs/20261006T-continuation/ucp-object-recovery-full-suite-final.log`：UCP object recovery suite **11 passed, 1 warning**。
- 上述 warning 均为环境中 `pynvml` package 弃用 FutureWarning。
- 失败迭代保留于 `zero1-production-controller-fresh-restore.log`（缺少 locator/survivor 环境变量）、`zero1-production-controller-fresh-restore-retest.log`（controller 漏导入 `ReplicaServiceClient`）、`zero1-production-controller-fresh-restore-final.log`（测试 all-reduce 输入类型错误）、`zero1-locator-env-production-restore.log`（测试缺 `json` 导入）。这些结果不计入最终通过数。
- 最终 `py_compile` 和 `git diff --check` 通过；SemanticFixed 与 dense baseline 实现未修改。此前 `semanticfixed-scope-final.log` 的 4 项隔离回归仍是该范围的证据。

该门禁验证的是一台机器上的 4-rank Gloo/TCP 服务、停止源 service 后经存活副本 fresh-state restore 和 continued-step state 对比；它不覆盖在 restore 进行中触发故障，也不等于 MerakTrainer 的 PP4/16-rank fresh-job restart，更不是 node8/node17 双机实验。尚未启动两机 tmux job；尚无本轮 MoE transfer size、max edge、吞吐或恢复时延数据。原 CPU preflight 的 `checkpoint_ready=false` 表示 SemanticFixed 尚不兼容 ZeRO-1 local optimizer ranges，不是本轮独立 UCP object-controller gate 失败。下一步仍需打通真实 16-rank Trainer 的 publish、作业重启、存活副本恢复及 continued-step digest；通过后再启动 node8/node17 tmux smoke，并只按原始日志报告指标。

### 72.55 2026-10-06: 双机启动前拓扑边界复核

续接 MoE 16-rank 恢复工作。只读检查 node17/node8：两台机器各有 8 张
A40，检查时 GPU process 和显存占用均为空；node17 已有多个历史 tmux
session，不能按名称清理或复用。本轮未启动新 tmux、service 或训练作业，
也未修改/终止既有 session。node8 的只读 SSH 查询正常；一次远端端口检查
因环境未安装 `ss` 未能提供端口证据，不将其记作端口空闲证明。

源码复核发现目前不具备直接启动 PP4/EP2/DP2 fresh-recovery gate 的充分
证据：`run_workload.py` 把 composite `--dp 4` 拆为 EP2×replica-DP2，并在
`MERAK_MESH_ORDER=dp,pp,sp,tp` 下建立 EP rank layout；但 Merak legacy
`PipelineParallelGrid.get_model_parallel_world_size/group/rank` 表示 DeepSpeed
composite model/data axis，而不必然表示 tensor-parallel 轴。模型 `tp=1`
不能据此推导 MPU model-parallel size 为 1。`PipelineModule` 的 TP rebuild、
LoRA admission、AMP overflow/norm 和其他共享调用点读取这些 MPU API；全局
重定义会影响 dense 行为，不能作为 MoE 专属修复。本轮没有修改这些共享
调用点或 SemanticFixed/dense baseline。

ZeRO-1 restore controller 的 tensor slice 参数也读取 MPU model-parallel
rank/world；在 PP/DP composite topology 中其语义须与实际 TP slice 坐标核对，
不能用 CPU layout arithmetic 代替实际 fresh pipeline module/optimizer
metadata 对照。已有 `qwen-4rank-ep-zero1-gate.log` 只验证 PP1 的 tiny
Trainer 一步更新；本机 8 张 GPU 无法以单机复现完整 16-rank PP4/EP2/DP2
Trainer，CPU 16-rank topology gate 也不能证明 CUDA pipeline/AMP 执行正确。

一条 pytest collect 命令因误写测试函数名返回 exit 4（目标不存在，非测试
失败）；正确的 PP1 gate 历史 transcript 仍显示 **1 passed, 1 warning**，
但本轮未重跑或升级其证据范围。未改代码，未产生新的训练或检查点指标，
不更新 `docs/09-24.md`。下一步应新增独立 EP/MoE topology + recovery gate，
显式记录 Merak grid 的 slice TP coordinate、PP/DP rank、EP coordinates、
ZeRO group ranks 与 capture/restore layout hash；先在可用的 16-rank GPU
资源上证明真实 PP4/EP2/DP2 Trainer one-step 和 fresh optimizer-state restore
一致，再经 Ponytail 式最小隔离修复处理已证实的错误映射。未通过前不启动
node17/node8 tmux 作业、不报告 MoE transfer/max-edge/性能结果；所有现有
未提交研究改动与历史 tmux session 均保持原样。

### 72.56 2026-10-06: PP4 EP topology gate 失败与 rank 映射更正

本轮重读本地 `ponytail` skill 并复核 §72.55 的 gate 边界。曾临时加入
8-rank PP4/EP2/DP1 Trainer gate；该 gate未执行训练，在 shared/data group
rank-set assertion 处失败，三份原始失败日志保留于
`runs/launch_logs/20261006T-continuation/qwen-8rank-pp4-ep2-dp1-zero1-gate.log`、
`qwen-8rank-pp4-ep2-dp1-zero1-gate-retry1.log`、
`qwen-8rank-pp4-ep2-dp1-zero1-gate-retry2.log`。第一次失败是本机
`torch.multiprocessing.spawn` 不支持 `kwargs`；其后一次失败发生在
group assertion。该临时 gate 已移除，不计通过；没有启动双机任务。

用 `ProcessTopology` 的 row-major rank 定义重新枚举目标 PP4/EP2/DP2/TP1：
Merak `dp,pp,sp,tp` 的 rank formula 是 `rank=8*dp+2*pp+tp`；EP layout
`order=(dp,ep,pp,tp)` 的 formula 是 `rank=8*dp+4*ep+pp`，两者不等价。
Merak `pp,dp,sp,tp` 对应 `rank=4*pp+2*dp+tp`，同样不等价。此前将
`mpu.get_data_parallel_group()` 直接当作 EP×replica-DP shared ZeRO group
也没有 rank-set 依据。PP1 gate 会掩盖 PP 坐标，不能外推到 PP4。

本轮尝试参数化 PP1 worker 时引入了临时测试改动；随后只恢复了已能确认的
worker 结构，未修改 `run_workload.py` 原有 mesh-selection 内容。由于两个
源文件未跟踪且没有可信的初始快照，不把当前文本称为精确恢复原样；dense
SemanticFixed 文件与实现未修改。原始 PP4 失败日志保留，不能视为训练数据。

本轮重跑原 4-rank PP1 CUDA Qwen tiny Trainer/EP2×DP2/role-aware ZeRO-1
一步 gate：**1 passed, 1 warning**，日志
`runs/launch_logs/20261006T-continuation/qwen-4rank-pp1-ep2-dp2-zero1-gate-recheck.log`。
重跑 role-aware ZeRO-1 clipping/overflow/empty-route Gloo 数值测试：
**1 passed, 1 warning**，日志
`runs/launch_logs/20261006T-continuation/zero1-role-aware-focused-recheck-final.log`。
两条 warning 均为环境内 `pynvml` package 弃用 FutureWarning。`py_compile`
和 `git diff --check -- AGENT.md` 通过。这些结果不升级为 PP4 或双机证据。

当前已证实的是目标 rank encoding/group selection 未满足 PP4 admission，
不是“换 mesh order”即可修复：Merak PP stage lookup、data-group 语义、shared
参数同步和 EP all-to-all 必须依据同一个实际 rank map 闭合。下一步先用只读
rank-set 推导和小型 topology test 固化 Merak stage/data、EP dispatch、
replica-DP、shared EP×DP 的坐标合同，再作最小 MoE-local wiring 修复。通过
8-rank PP4 CUDA Trainer gate 与真实 16-rank UCP publish/fresh-job restore/
continued-step digest 后，才启动 node17/node8 tmux 实验。现在没有 MoE
训练、恢复、transfer、max edge 或性能数据；不更新 `docs/09-24.md`。

### 72.57 2026-10-06: PP4 topology 失败证据范围更正与只读 group 矩阵

更正 §72.56 两处表述。其失败的 8-rank PP4/EP2/DP1 gate 在测试 worker
中手工构造 `PipeDataSequenceModelParallelTopology(4,2,1,1)`，并 monkeypatch
MPU groups；它没有调用 `run_workload.py`、`set_topo_grid_communication()`
或 PyTorch DeviceMesh 初始化。故该断言失败只证明该 fixture 的 data group
与 EP layout 预期不一致，**不证明生产 launcher 已运行并失败**。初次
`spawn(kwargs=...)` 是 harness TypeError，后两次是 fixture group assertion；
三份日志均保留，但均未执行训练。§72.56 所说“PP1 gate 会掩盖 PP 坐标”
只对 PP1 证据范围成立；失败 gate 是 PP4 fixture，不应混称 PP1。

直接用当前 `ProcessTopology.get_axis_comm_lists()` 与
`ExpertParallelLayout.all_groups()` 做只读 CPU rank-set 枚举（非分布式
运行、非训练测试）得到：

- 目标 16 ranks、PP4、composite DP4=EP2×replica-DP2、TP/SP1 下，legacy
  fallback 的 `MERAK_MESH_ORDER=dp,pp,sp,tp` 确实使 Merak PP groups 与
  EP layout `order=(dp,ep,pp,tp)` 的 PP groups 一致。此前 §72.56 将两种
  rank formula 直接比较后称为 PP stage 不对齐，结论过度；不同坐标名不能
  仅比较同序表达式，应该比较实际 stage/group rank sets。
- 同一 legacy fallback 下 Merak `data` groups 为
  `(0,4,8,12)`、`(1,5,9,13)`、`(2,6,10,14)`、`(3,7,11,15)`；它们跨
  PP stages。EP/replica-DP layout 的单轴 ZeRO groups 是 `(0,8)` 等
  replica-DP pairs（shared/expert 使用不同角色组），不是这些 composite
  data groups。当前 trainer 将 Merak data group 作为 shared role-aware
  ZeRO group，因此完整 16-rank 实际 gate 仍未具备正确的 group contract。
- 生产 `run_workload.py` 当前明确选择 `dp,pp,sp,tp`。在 torch 2.3.1 的
  DeviceMesh 可用时，`set_topo_grid_communication()` 走
  `MeshParallelGrid`，各命名轴独立取 mesh；其 `pp` group 是固定 DP
  coordinate 下的 ranks。对当前 order，PP stages 变化会跨 EP/replica-DP
  layout 的固定 `(dp,ep)` 坐标，故与 EP layout PP group 不一致。若走
  legacy fallback，则 PP groups 对齐，但 data group 仍不等于角色化 ZeRO
  groups。两个 backend 必须分别做真实 rank 对照，不能把 CPU 枚举视作
  实际 launcher gate。
- 8-rank PP4/EP2/replica-DP1 的 legacy `dp,pp,sp,tp` 中，Merak composite
  data groups 恰等 EP groups；这解释旧 PP4 fixture 为何在 data-group
  assertion 失败，但 fixture 手工指定了 PP-first topology，仍不能替代
  launcher/backend 实测。

所以 §72.56“目标 rank encoding/group selection 已被失败 gate 证实不满足”
改为未证实：目标 gate 尚未走到生产 launcher wiring。已读代码足以指出
16-rank ZeRO shared-group 候选存在不匹配，并且 DeviceMesh/legacy fallback
拓扑语义不同；仍需经当前 `Merak.init` 和 `MerakTrainer` 路径实际 gather
global ranks、stage、EP/DP coordinates、ZeRO role groups 与 optimizer shard
metadata，才能定性运行时问题。不得只修改 mesh order：mesh 同时决定 PP
stage、MoE dispatch、replica-DP 和 recovery coordinates。

本轮只读审查，没有改 EP、UCP recovery、dense SemanticFixed 或 launcher
源码；保留所有既有未提交内容。没有启动 GPU、tmux、服务或远程任务，也无
新训练/恢复/检查点指标。下一步先在独立 EP/MoE topology gate 中直接调用
生产 grid initializer，在 DeviceMesh 与 fallback 可用路径下验证 8-rank
PP4 EP2，一致检查 PP stage groups、expert ownership/dispatch groups、
replica-DP optimizer groups、shared optimizer groups及 shared overflow/norm
collective；该 group contract 通过后再运行 8-rank Qwen Trainer 一步。继而
在 node8+17 实际 16 ranks 验证 fresh UCP recovery 与 continued-step digest，
之后才按 tmux 启动双机实验。当前仍无 MoE transfer size、max edge、性能
数据；不更新 `docs/09-24.md`。

### 72.58 2026-10-06: Production DeviceMesh rank-set gate（8 ranks）

按 §72.57 继续，新建独立测试 `test/core/moe/test_production_topology_groups.py`，
通过 8-rank Gloo 分布式 worker 调用生产 `mpu.set_topo_grid_communication()`，
对其 DeviceMesh 分支将 `init_device_mesh("cuda",...)` 的 device type 替换成
PyTorch 原生 CPU DeviceMesh；rank mesh、命名维度和 process-group 构造仍由
真实 DeviceMesh API 完成。另创建生产 `ExpertParallelLayout` groups，并逐 rank
收集 Merak pipe/data、stage-to-global、EP、replica-DP 与 layout PP group 成员。
该方法验证 PyTorch DeviceMesh/Gloo 的 group 语义，不等价于 CUDA/NCCL/Merak
Trainer 运行，也没有实例化 ZeRO optimizer。

最终 transcript：
`runs/launch_logs/20261006T-continuation/production-topology-groups-device-mesh-final-recheck.log`：
**2 passed, 1 warning**。8-rank PP4/EP2/replica-DP1 下，当前生产顺序
`dp,pp,sp,tp` 的 Merak PP groups/stage lookup 与 EP layout PP groups 相符；
Merak data groups 与 EP groups 相符。对照顺序 `pp,dp,sp,tp` 的两项都不符。
这个 8-rank 配置的 replica-DP=1，ZeRO expert groups 是 singleton；shared
group size=2。因此 data==EP 仅验证本地 EP2 group，不能外推至目标
16-rank PP4/EP2/replica-DP2 下 shared EP×DP group size=4，也不能证明
ZeRO reduce-scatter、overflow/norm collective、optimizer state capture/restore。

测试过程中的失败 transcript 全部保留：`production-topology-groups-gloo.log`
（早期直接 assertion，1 passed/1 failed）、`...-gloo-recheck.log`
（对照断言范围修正后仍失败，1 passed/1 failed）、
`...-gloo-final.log`（rank-local stage 与 EP coordinate 比较不成立，1 failed）、
`...-gloo-final-recheck.log`（8-rank 对照 expected group 修正前，1 failed）、
`...-gloo-backend-separated.log`（错误强制 DP-first legacy fallback，1 passed/
1 failed；暴露 generic fallback 以 `dp/pp` 轴调用 `PipelineParallelGrid`
时访问 `.pipe` 失败）、`...-gloo-backend-separated-retry.log`
（删除该错误 fallback case 后 2 passed, 1 warning）、
`...-device-mesh-cpu.log` 与 `...-device-mesh-final.log`（CPU DeviceMesh
结果预期修正前，均 2 failed）。只以最终 recheck 作为当前 gate 结果；较早失败
不删除、不覆盖、不计通过。所有 warning 是环境内 `pynvml` package 弃用提示。

CPU DeviceMesh 检查过程中修正了 §72.57 的一个拓扑推断：对本轮实际 8-rank
mesh，`dp,pp,sp,tp` DeviceMesh 的 PP group 是 `(0,1,2,3)`/`(4,5,6,7)`，
与 EP layout 相符；`pp,dp,sp,tp` 的 PP group 为 `(0,2,4,6)`/
`(1,3,5,7)`，不符。之前依据“固定 DP coordinate”的文字推导作出的相反
结论属于误判；以 `stage_to_global()` 和实际 process-group rank sets 为准。
legacy fallback 单独路径不等于 DeviceMesh 分支；强行让 generic fallback
处理 `dp,pp,sp,tp` 会暴露轴名不兼容，本轮没有修改共享 MPU topology。

本轮仅新增 MoE 专项 topology test 与本节记录，未改现有 EP/UCP/dense
SemanticFixed/launcher 实现；未启动 CUDA、双机 tmux、service 或模型训练。
下一 gate 是在 16-rank node8+17 上通过实际 CUDA DeviceMesh/NCCL 验证
EP2×replica-DP2 的 role-aware optimizer groups、shared/expert ZeRO shard sizes、
overflow 与 norm collective；然后再完成 real PP4 Trainer step 和 UCP fresh
restore/continued-state digest。此前不启动性能实验、不填写 MoE transfer size、
max edge 或恢复/吞吐结果。

§72.58 transcript 范围补充：
`production-topology-groups-device-mesh-final-recheck.log` 与其后纯签名清理
版本 `production-topology-groups-device-mesh-final-verified.log` 的 **2 passed**
来自 CPU DeviceMesh，不是 legacy fallback。更早
`production-topology-groups-backend-separated-retry.log` 才是仅 PP-first
fallback 对照的 2-pass，不覆盖当前生产 DP-first 配置；强制 fallback 的失败
`production-topology-groups-backend-separated.log` 仍保留。另有早期实际集合
输出 `production-topology-groups-device-mesh-cpu.log`，初次预期不符而失败，
不得引用为 gate 通过。最终 DeviceMesh test 的 `cpu_mesh` 只替换设备类型，
mesh shape/order、原生 DeviceMesh groups 和生产 `MeshParallelGrid` 代码不变。

虽然 §72.58 计划随后验证 role-aware optimizer groups，实际本轮没有实例化
`RoleAwareZeroOptimizerStage1`；ZeRO role group、reduce-scatter、shared overflow/
norm collective 与 state capture/restore 均未测试。由于仅 8 ranks 且 replica-DP=1，
不能以 singleton expert group 的 CPU plumbing 推导目标 16-rank optimizer
正确。该门禁和当前生产 CUDA/NCCL/backend 等价性仍待 16-rank GPU 实验验证。

### 72.59 2026-10-06: 16-rank role-aware ZeRO Gloo gate

续接 §72.58，保留此前所有未提交研究文件。新增 16-rank CPU DeviceMesh/Gloo
用例覆盖 PP4/EP2/replica-DP2/TP1 下的 shared、expert、dispatch、pipeline
process-group rank sets，以及 role-aware ZeRO-1 partition sizes、梯度归约、
overflow/norm 全局 collective 与更新后的 all-gather。新增 shared group helper
沿用 EP group helper 的全局确定性创建约定：所有 rank 以相同顺序创建全部
PP/TP-local shared groups，再选取本 rank 所属 group。

首次测试失败日志
`runs/launch_logs/20261006T-continuation/moe-16rank-role-aware-zero1-gloo.log`
显示 group replica 校验期间 Gloo connection reset；后续检查发现测试 fixture
给 shared 参数按 global rank 写入不同初值，违反副本初始化一致约束。修正该
fixture 后的第二次运行
`moe-16rank-role-aware-zero1-gloo-rerun.log` 又在 `new_group` 时失败：helper
仅由所属 ranks 创建当前 shared group，不满足全局同步创建顺序。修正生产 helper
后，`moe-16rank-role-aware-zero1-gloo-rerun2.log` **1 passed, 2 deselected**。
随后完整 topology 文件
`moe-production-topology-gloo-final.log` **3 passed, 1 warning**；EP、role-aware
ZeRO optimizer、Trainer glue focused suite
`moe-focused-tests.log` **14 passed, 1 warning**。warning 均为环境
`pynvml` package 弃用提示。`moe-ep-zero1-preflight-rerun.json` 中 Qwen2-MoE
60 experts 与 DeepSeek-V2-Lite 64 experts 的目标 ownership/dispatch 均 ready，
但 `checkpoint_ready=false` 且总 `ready=false`，原因仍是该 preflight 要求
SemanticFixed 对 ZeRO-1 local ranges 兼容；不把 ownership gate 写成 recovery
通过。

本轮改动限于 MoE `create_shared_parameter_group`、其 launcher 调用及独立
16-rank topology/ZeRO 测试；未改 SemanticFixed 或 dense baseline。`py_compile`
与 `git diff --check` 通过。node17 与 node8 的 GPU process 查询为空，但本机
存在多个历史 tmux session（包括已退出的历史 MoE smoke 名称）；没有复用、
关闭或清理它们。未启动任何新 GPU/tmux 作业：`PLAN_TWO_NODES_V2.md` 明确要求
双机 GPU 运行另行授权，而本轮“继续”未明确授权。CPU/Gloo 结果不等价于 CUDA/
NCCL；本轮没有 16-rank真实模型训练、UCP fresh-job restore 或模型性能数据，
不更新 `docs/09-24.md` 的 MoE 指标。下一门仍是取得明确 GPU 运行授权后，先用
独立 tmux 跑 16-rank NCCL role-aware optimizer/短步 Trainer gate，再跑 UCP
publication、fresh 16-rank job restore 与 continued-step state digest；任何
失败原样留存于 `/runs`，通过前不启动性能对照。

### 72.60 2026-10-06: 单机 PP4 CUDA group gate 与 Trainer 阻塞定位

用户明确允许单机 GPU 正确性实验。本轮仅使用 node17 的 8 张 A40；node8 未
启动。先新增单机 CUDA DeviceMesh/NCCL PP4/EP2/replica-DP1 group gate，并
用独立 tmux session `moe-single-node-pp4-groups-20261006` 运行。原始输出
`runs/launch_logs/20261006T-single-node-gpu/pp4-ep2-nccl-group-contract-tmux.log`
及 `.status` 记录 `exit_code=0`、**1 passed, 1 warning**；检查确认 Merak
PP groups 与显式 EP layout stage group 对齐、EP dispatch group size=2、
shared ZeRO group size=2、expert replica-DP group size=1。此前非 tmux 同门
输出 `pp4-ep2-nccl-group-contract.log` 也为 **1 passed**。这是实际 CUDA/
NCCL process-group membership 验证，不是模型 forward/backward、Trainer 或
ZeRO optimizer gate。

尝试建立同布局 8-rank tiny Qwen Trainer gate。首次 Gloo transcript
`pp4-qwen-trainer-gloo-gate.log` 在读取固定 `forward_funcs[1]` 时 `IndexError`；
修正为跟踪本地实际参数后，`pp4-qwen-trainer-gloo-rerun.log` 在
`RoleAwareZeroOptimizerStage1` 初始化处失败：PP partition 中至少一个 rank
没有任何 marked expert parameter，触发 `role-aware ZeRO-1 needs marked expert
parameters`。这不是 Gloo 或 GPU 资源错误，也没有进入训练 step。结果揭示
现有 optimizer wrapper 要求每个 pipeline rank 同时具备 shared 与 expert
角色，而合法 PP partition 可将 MoE block/expert 参数放在部分 stages；当前
wrapper 的参数角色拆分、global overflow/norm 参与及 optimizer/scheduler
状态在这种 stage-local 空角色下没有完整 contract。不能向空 stage 注入伪专家
参数，也不能把 PP1 Trainer gate 或 NCCL group gate扩写成 PP4 Trainer 成功。

`ponytail` 约束下本轮没有继续扩写跨 stage 空角色协议，也未启动 NCCL Trainer
作业。测试仍保留 Gloo 预检 worker 及原失败 transcript，用于后续修复验证；
没有修改共享 dense SemanticFixed、AMP/MPU 或 MerakEngine 默认路径。`py_compile`
与 `git diff --check` 通过，实验后本机 8 卡显存均为 0 MiB。下一步实现 MoE
专属 stage-aware ZeRO 参数角色合同：支持每 stage 仅 shared、仅 expert 或两者
皆有，同时所有 PP ranks 对 overflow、norm、step/scheduler 与 checkpoint
metadata 采用一致参与语义；先以 Gloo 建立 PP4 两类空角色测试，再跑本单机
NCCL tiny Qwen Trainer gate。通过前不启动两节点模型训练，不声称 Merak/AdaCheck
已有原生 EP 支持，也不更新 MoE 性能表格。

### 72.61 2026-10-07: ZeRO-1 padding-only chunk recovery 修复

复核 2026-10-06 单机 Qwen 失败 transcript 后，发现 role-aware ZeRO adapter
用完整 rank/group/chunk 请求 optimizer state，但 capture 对无参数 shard
coverage 的 chunk 直接 `continue`。ZeRO-1 flatten 会为 DP/通信区间补齐；某 rank
可持有完全由 padding 组成的 optimizer chunk。即使没有模型参数覆盖，该 chunk 的
AdamW master、exp_avg、exp_avg_sq 和 step 仍被 optimizer 持有，不能遗漏。仅凭参数
名及展平长度反推 chunk 范围也会丢失 ZeRO 通信区间边界。

修复限于 MoE 专用 role-aware ZeRO optimizer 和独立 object adapter：初始化时从
实际 `element_intervals` 保存每个本地 chunk 的 parameter offset、长度及 chunk
offset；adapter 使用该显式 manifest。Role-aware capture 对 padding-only chunk
也发布完整 chunk objects，request/apply 保持同一稳定 rank/group/chunk identity。
原 AdaCheck/SemanticFixed、dense optimizer 和 Merak 默认路径未修改。

CPU regression 覆盖 DP=2 rank-local optimizer chunk，其中第二块没有任何参数覆盖，
经 object catalog/plan/restore 后比较 master 与 AdamW 三个 slot；另覆盖多 interval
参数映射。两用例 **2 passed**。完整 adapter/optimizer/PP4 stage-local UCP Gloo
suite **16 passed, 1 warning**；单独完整 adapter 文件 **14 passed, 1 warning**。
warning 为 pynvml 弃用提示。单机 node17 8×A40 使用独立 tmux 跑
`test_cuda_qwen_tiny_pp4_ep2_dp1_role_aware_zero1_trainer`，**1 passed, 1 warning**；
该测试训练一步、构造 UCP-style mirrored object catalog、模拟 host loss、建立
recovery plan、restore 到 fresh Trainer，并逐参数 exact equality。测试没有在 restore
后执行下一训练 step，因此只证明该同布局 fresh restore gate，不证明 continued-step
等价、真实 service publication、AdaCheck detector 组合路径或双机恢复。
原始 pytest transcript 保存于 `runs/launch_logs/20261007T-single-node-gpu/`；首次
tmux 命令没有生成 exit marker，transcript 保留且显示 1 passed。没有改实验指标。

随后将 CUDA gate 扩展为 continued-step differential oracle。第一次尝试因手工
iterator 只提供一个 batch，而 PP schedule 需要第二个 microbatch，失败于
`StopIteration`；该失败 transcript 保留在
`qwen-pp4-ep2-continued-restore.log`。修正为同一固定 batch 的双 microbatch iterator
后，独立 tmux session `moe-qwen-continued-rerun-20261007` transcript
`qwen-pp4-ep2-continued-restore-rerun.log` **1 passed, 1 warning**：fresh Trainer
恢复 step1 后运行 reference step2，重新应用同一 checkpoint 和 RNG，再以相同 batch
运行恢复分支 step2；逐 rank 比较模型参数、全部本地 ZeRO master/moments/step、
optimizer global_step/overflow/loss-scaler/group metadata 与 scheduler state，均通过。
这比前一版增加了真实 continued-step 数值一致门禁，但仍是同进程手工 mirrored catalog，
不是 Trainer 退出后由两个真实 UCP services 发布并在新 job 恢复；sampler=None 且
验证输入显式固定，因此不声称生产 dataloader 消费位置恢复正确。

期间两次初始 targeted regression 各有 2 个失败，暴露手工构造的 rank metadata fixture
不符合 Merak `get_all_sub_partition_info` 结构；失败 transcript 未覆盖或删除。修正为
显式 chunk-range fixture 后 targeted tests 通过。`py_compile` 与 `git diff --check`
通过。node17 实验后 8 卡显存为 0 MiB。下一 gate 是测试真实 UCP service publication/
recovery、独立 fresh job 和 sampler/data progress；之后再考虑 detector/encoder 在线绑定。未完成
这些 gate 前不声称 MoE checkpoint recovery 完整正确，不启动双机性能实验。

### 72.62 2026-10-07: 双机 tiny-Qwen source 初始化阻塞，未形成恢复结果

按用户授权启动 node17/node8 共 16×A40 的 tmux correctness smoke，实验 ID
`20261007T-dual-vfy`，布局为 Qwen2-MoE tiny 随机初始化配置、PP4/EP2/DP4/TP1、
role-aware ZeRO-1、两端真实 UCP replica service；完整日志保留在
`runs/two-node-moe-zero1/20261007T-dual-vfy/`，driver 和本轮 Gloo gate 日志在
`runs/launch_logs/20261007T-moe-dual-validation/`。两个节点均成功完成 torchrun
分布式启动和模型 pipeline partition；stage 列表显示 4 个 tiny decoder layers。
之后超过 10 分钟没有 training-step、rank digest、locator 或 replica record；两端
service `record_count=0`，训练 worker持续占满 PP stage 所在设备，但无 Python traceback。
因此无法确认首个训练 step 是否完成，更未发布或恢复 checkpoint；本 run 不提供
checkpoint transfer size、max PUT/max-edge、训练吞吐或 recovery latency 指标。

在训练期间对 replica service 的只读 stats 探测误用了 raw `urllib`（service 使用
自定义带长度前缀协议），收到 `header is too large: 5135603447297700961`。该探测
没有写入 record，也不能据此认定训练端 publish 协议本身失败。由于首步持续无进展，
于 2026-10-07 17:25 UTC 中止本次短 smoke；只清理本 run 精确命名的 tmux session，
随后确认 node17/node8 均无本 run 的 torchrun、worker 或 service 进程。原始训练、
torchrun、service 和 driver 日志完整保留；无 `COMPLETE`、locator、digest 或 metrics
文件，结果记为 **中止/未完成**，不是测试通过或恢复失败。

本轮 focused CPU/Gloo suite 初次为 **18 passed, 1 failed**：stage-local fixture
故意把 PP stage 参数角色改成每 stage 单角色，但没有验证 PP optimizer 空角色的
global collectives；临时修改的 Gloo Trainer worker 还把 pipeline `_communicate`
替换成 no-op，不能作为有效 Trainer 正确性测试。已按
`runs/analysis_backups/20261007-moe-ucp/test/core/moe/test_production_topology_groups.py`
保存快照，逐函数撤销这两处本轮临时实验修改；保留此前补齐的 16-rank Gloo worker
局部 `SimpleNamespace` 导入，以及 Trainer fixture 对 tiny layer count 的修正。
当时 16-rank Gloo suite 的 **19 passed, 1 warning** 和单机 CUDA Trainer/UCP
continued-step gate 的 **1 passed, 1 warning** 分别只支持其原有证据范围，不覆盖本次
双机 source 或真实 service fresh-job recovery。`SemanticFixed` 与 dense baseline 未改。

下一步先单独 profile/定位 tiny Qwen 首个 `train_batch` 阻塞位置，并用仓库的
`ReplicaServiceClient.stats()`（而非 raw HTTP）获取 service 指标；经单机门禁确认
训练/发布步骤能在合理时间内完成后，再重启双机 tmux source/restore。不能将本次
未完成 run 的初始化 GPU utilization 或错误协议探测计为实验结果。

### 72.63 2026-10-07: 双机 broadcast 追踪定位到 tiny 数据词表越界；UCP source 尚未完成

新建诊断 run `runs/two-node-moe-zero1/20261007T-broadcast-trace/`，未修改
`PipelineModule` 正式实现。monkeypatch 仅记录原 `_broadcast_model` 里每个参数
broadcast 的 rank/stage/名称/角色/group/src 和 PRE/POST，并把默认 process-group
timeout 限为 120 秒。首轮 16-rank trace 的 shared/expert 参数 broadcast 均出现匹配
POST；失败发生在其后的首次 forward，CUDA 报 `indexSelectSmallIndex` 越界，排除本轮
“初始化 broadcast 卡住”为直接根因。检查 `Merak/utils/datasets.py` 后确认
`DynamicGenDataset(mode="text_only")` 固定从 `[1,1000)` 生成 input/label，而本次 tiny
Qwen `vocab_size=31`。据此最小修复为 token 上界 `min(vocab_size,1000)`，并新增
`test/core/test_datasets.py` 覆盖词表 2、31、1000、32000 和非法词表 1。目标 conda
环境 pytest 命令结果 **15 passed, 1 warning**（新数据集测试和 `test_expert_parallel.py`）；
另外直接从真实 Qwen config 采样 100 次均满足 token ID 小于 31。`conda run` wrapper
因当前登录用户的 conda base 路径解析错误未运行，改用指定环境 Python 绝对路径通过。

相同双机 16-rank 配置修复后，无 UCP 的 Trainer smoke 于 19:01 UTC 两端正常退出，完成
1 个训练 step，日志 loss `3.4146`，step wall `2.204 s`；这只验证 tiny Qwen
PP4/EP2/composite-DP4/TP1 + role-aware ZeRO-1 的一步训练，不是恢复测试或有意义的模型
质量/吞吐指标。逐参数 trace 也表明构造期广播成功。

随后执行既有 UCP smoke harness，run ID `20261007T-vocabfix`、service 端口 29961、
torchrun 端口 29962。两个 service ready，source workers完成分布式初始化和 pipeline
partition，但长时间无 step、locator、digest 或 PUT record；通过正确的
`ReplicaServiceClient.stats()` 查询，两端 `record_count=0`。为取得 Python 栈而向训练
worker 发送 `SIGUSR1` 是本轮操作性错误：进程未安装该信号的 stack-dump handler，遂以
signal 10 退出；不得把这个退出算作训练/recovery 缺陷，也未产生 checkpoint 指标。
随后只终止该 run 精确 node8 torchrun 和两端 `20261007T-vocabfix` service tmux，确认本轮
worker/service 均退出且 16 张卡显存回到 0 MiB。原始日志、状态和失败轨迹保留，无
`COMPLETE`、locator、rank digests 或 publish metrics。

当前证据：双机 MoE 一步训练通过；broadcast 路径通过；真实 UCP source 与 fresh-job
surviving-host restore 仍未通过/未完成，不能报告 UCP recovery 正确性或 transfer size、
max PUT/max-edge、recovery latency。下一步应在新的 tmux run 里对 UCP source 增加
faulthandler 在 SIGUSR1 上的明确 handler（或做可结束的定时 traceback，而不是给进程发
默认致命信号），先定位 UCP-enabled `train_batch`/capture 的停点，再继续 source/restore。
`SemanticFixed` 和 dense baseline 未改。

### 72.64 2026-10-07: 双机 16-rank Qwen2-MoE UCP source 与 surviving-host fresh-job restore

用户已明确授权双机 16 GPU 正确性验证。本轮使用 node17 (`172.20.17.2`) 与
node8 (`172.20.8.2`)，各 8×A40，conda 环境
`/opt/conda/envs/sjw_ckpt_grouping`，并通过独立 tmux 会话执行 torchrun 与服务。
配置为 tiny Qwen2-MoE、PP4/EP2/composite-DP4（shared ZeRO-1 DP2、expert replica-DP2）/
TP1、role-aware ZeRO-1、bf16、每步 batch 1。此为 checkpoint/训练路径 correctness
workload，不是实际 14B Qwen 性能测试。

Source run `runs/two-node-moe-zero1/20261007T-ucpsourcefix/` 两端 torchrun 均
`exit_code=0`，16 ranks 完成一步训练，rank0 loss `3.4146`，step wall `2.260 s`。
step 1 的 synchronous complete-state UCP snapshot 成功 seal，locator epoch=1，arm
`adacheck_fixed`。原始 metrics `source/output/zero1-object-metrics.jsonl`：
`capture_seconds=0.0088822`、`publish_seconds=0.556416`、跨 host
`remote_payload_bytes=787196`、`maximum_remote_put_ms=10.4198`。这些只描述本 tiny
配置的单次 publication；不等价于 dense transfer 指标或正式模型吞吐。node17 source
service 随后停止，node8 replica service 保持运行。

Fresh restore run 复用同 topology 和 seed，从唯一存活的 node8 service 加载 epoch 1；
torchrun 两端均 `exit_code=0`。日志显示全部 16 ranks 均报告
`Resumed from UCP ZeRO-1 object epoch 1`，并执行训练至 global step 2；rank0 step 2
loss `3.3663`、train batch `2.214 s`。16 个 restore rank digest 均记录 iteration 2。
这证明真实双机服务发布、seal、源 host service 下线、fresh job 从另一 host 读取完整
rank state 并继续训练的路径可运行。

严格 uninterrupted equivalence **尚未验证**。现存 source digest 在 step 1，restore
digest 在继续训练后的 step 2，二者不同是预期现象，不能当作 mismatch 证据。也没有
运行完成同配置 uninterrupted step-2 逐 rank model/optimizer/scheduler digest 对照；因此
本轮不声称 bitwise/numerical equivalence 或完整训练语义正确。下一步应先用独立、长度和
scheduler 设置一致的 gate（固定 batch 序列，比较同一 optimizer boundary 的 model、
master/optimizer、scheduler 和 sampler/RNG 状态），再将该检查移入双机门禁。source 的
`max_steps=1`、restore 的 `max_steps=2` 以及 tiny dataset epoch 边界使本次不适合性能
对照。loss/throughput 只留作 smoke 日志，不作质量或性能结论；未统计 recovery latency、
max-time edge 或正式模型 transfer baseline。

本轮另有既存运行前修复记录：`DynamicGenDataset(text_only)` token 上界限制为
`min(vocab_size,1000)`，新增 tiny vocabulary 测试；MoE UCP tensor group 坐标改为验证
pipeline grid slice group 与 EP layout TP ranks 一致。相关 focused suite 为
**25 passed, 1 warning**（包含 dataset、MoE Trainer glue、ZeRO-1 object adapter）；
本次差异之前 dataset/EP suite 为 **15 passed, 1 warning**。没有修改 `SemanticFixed`
或 dense baseline。

所有 source、restore、service 日志、locator、metadata、16-rank digests 和 metrics
保留于上述 run 目录。已停止本轮 `ucpsrcfix` 两 host service/torchrun；node17/node8
GPU 显存均为 0 MiB。历史失败 run `ucpdiag2` 的 node17 service 仍在运行，本轮未将其
误杀或清理。不要将本次结果写入 dense 性能表，也不要将旧的 step1/step2 digest 直接
相互比较。

### 72.65 2026-10-08: 16-rank EP sampler diagnosis 与故障实验准入

本轮继续遵守 Ponytail：先验证生产 rank mapping 和既有恢复证据，再判断是否值得改训练
核心。目标 conda Python 为 `/opt/conda/envs/sjw_ckpt_grouping/bin/python`；当前登录 shell
的 `conda run` 错误解析到 `/root/miniconda3`，该 wrapper 报环境不存在，随后改用目标
环境 Python 绝对路径。没有启动新双机训练/故障注入，也没有触碰 SemanticFixed、dense
MPU/collective 或既有未提交文件。

在 16-rank Gloo 生产 mesh + EP layout + role-aware ZeRO-1 worker 中加入 sampler diagnosis：
Merak composite-DP sampler 的数据组大小为 4（EP2×replica-DP2），而 logical EP layout 的
replica-DP 大小为 2。比较同一 PP stage / replica-DP 下 EP peers 的首批 index，当前
composite-DP sampler **不一致**（例如 rank0/rank4 分别为 44/108）；以 EP layout 的真实
replica-DP rank/size 构造 sampler 时两 EP peers 一致（均为 44），两个 DP replicas 则不同。
16-rank Gloo worker 通过，生成了上述具体 ID 证据；但为保留正确性 gate 可运行，本轮没有
把该有缺陷的现有行为写成预期 pass 断言。新增测试记录诊断样本；下一步需先根据完整 MoE
梯度/输入合同实现修正，再将“EP peers 相同、DP replicas 不同”固化为通过标准。命令：

`/opt/conda/envs/sjw_ckpt_grouping/bin/python -m pytest -q test/core/moe/test_production_topology_groups.py::test_16rank_device_mesh_role_aware_zero1_groups -k gloo`

诊断版结果 **1 passed, 1 deselected, 1 warning**，约 10.2 s；warning 仅为 pynvml 弃用。
测试代码记录两组 sample ID，但不是生产修复，也不应把“诊断 worker 通过”理解为 sampler
语义正确。当前 source/restore 双机
smoke 使用相同的错误 composite-DP sampler，所以虽然服务发布、surviving-host fresh restore
和训练继续执行成功，仍不能证明 MoE 训练目标/数据语义正确。UCP 保存的是各 rank 本地
sampler state；目前没有跨 EP peers 的 sampler-state 一致性检查。此次已纠正早先怀疑
shared/router 梯度必然缩放错误的推断：现有 EP block 的 Gloo/CUDA differential tests 对
相同输入下 router/shared 梯度与单 EP reference 对齐；已证实的 blocker 是输入不同，不是
据此推断 collective 必须更改。

准入决定：在 MoE 专用 replica-DP sampler、dataset cursor/RNG recovery 与 effective batch
合同落地并有单机 8-GPU continued-step oracle 之前，不启动新的双机性能 trial 或 kill-rank
故障实验。仅改 sampler 不充分：`MerakTrainer.get_train_sampler/create_dataloader`、
`BaseParams` 全局 batch、pipeline stage iterator、checkpoint capture/restore cursor 都必须
一致；不得改变 dense 默认 MPU 或 SemanticFixed。下一实现应限制于 model opt-in metadata，
并在 16-rank CPU/Gloo 证明：每个 PP/replica-DP 组内 EP ranks 每个 microbatch 的
`input_ids`/`labels` 一致、不同 replica-DP 有各自 shard、各 rank optimizer-boundary cursor
满足合同、fresh restore 后下一批与 uninterrupted oracle 相同；再运行 node17 8-GPU NCCL
continued-step gate。Gloo Trainer 的 `batch_isend_irecv/startCoalescing` 不支持仍是 backend
限制，不用 no-op P2P mock 代替生产语义。

五项 MoE 指标沿用 AGENT §10.12 的 dense 定义，不可用当前 smoke 字段替代：

1. 每个已提交 global checkpoint 的跨故障域 logical encoded bytes、application wire bytes，
   另列独立 PMA/NIC bytes；按共同窗口和提交次数归一化。
2. 实际全局提交 cadence/frequency、lag、队列深度和 backpressure，不能用配置的 save step
   代替 sealed locator。
3. 每 epoch 完成的跨 host physical edge interval 最大值分布 P50/P95/P99/max，并单列
   capture-to-global-seal makespan；`maximum_remote_put_ms` 只是单 PUT proxy。
4. warmup 后共同 optimizer-step window wall/吞吐及含初始化、训练、drain/validation 的
   job wall；需独立 paired trials，rank/step 不是独立样本。
5. recovery 成功率、last sealed epoch/RPO、RTO 至首个成功 optimizer step、丢失/重算步数，
   以及同边界 model/ZeRO master+moments/scheduler/RNG/sampler/iteration 和后续 loss/state
   的数值 oracle。

当前 `zero1-object-metrics.jsonl` 只有同步完整状态 snapshot 的 capture/publish wall、
cross-host payload 和最大 PUT；没有上述 2、3 的完整定义，也没有可比的 4、5。因此不报告
有利性能结果，不向 `docs/09-24.md` 填正式指标，也不以该 smoke 作为故障实验基线。

后续 rank-failure recovery study 预注册草案（仅在上述正确性门槛及双机 uninterrupted
oracle 通过后执行）：固定代码/config/seed、PP4/EP2/replica-DP2/TP1、dataset sample ID
序列、optimizer/scheduler、checkpoint cadence 与两个独立 host services；每个注入点前先
由所有 16 ranks seal epoch k 并验证 locator + 双服务对象完整。reference 为同配置不注入
故障、运行相同 sample sequence 到 k+n 的 uninterrupted job。分别测试而不混报：

- **Committed-boundary rank death**：提交 k 后、下一 optimizer step 前，对预先选定且记录
  host/stage/EP/DP 坐标的单一 worker PID 发精确 `SIGKILL`。预期旧 gang abort，由外部重新
  启动完整 16-rank job，从最近 sealed k 恢复；不得声称单 rank 原地弹性续跑。
- **During-capture death**：在 capture/upload 已开始且 global seal 尚未完成时杀指定 worker。
  预期本轮 locator 不可见，所有未 seal 数据不算 commit，恢复点应是 k-1；测 gang abort、
  orphan bytes 和回滚/清理行为。
- **During-publish/pre-seal death**：放在独立 cell，记录各服务收到对象与 global commit
  状态；禁止把部分 host 接收误记作 checkpoint。未 seal 的 epoch 不得被 restore selector
  选中。
- **Surviving-host restore**：另列 host-service-loss 场景，与 rank process failure 分开；
  只允许从故障域外幸存服务恢复。不要混称物理整机/GPU 故障。

每种确定性注入位置先做至少 3 次 correctness repetitions（报告每次原始值和成功率，不能
把重复 rank 当样本）；性能分布另做预注册的更多独立 paired trials。统一时钟用
`time.monotonic_ns()` 同机区间 + 明确跨主机同步误差界，RTO 从注入 timestamp 到 restored
job 第一个成功 optimizer step。每次检查 RPO/丢步、所有 16-rank model digest、local ZeRO
master/Adam moments/step、scheduler、loss scaler、RNG、sampler cursor/sample IDs 和至少
后续 2 步的 loss/state tolerance；report timeout/hang as failure，不删失败样本。PID 必须由
run-specific pidfile/launcher manifest 精确定位；不使用宽泛 `pkill`。完整命令、session、
同步日志、service stats、locator、digests、注入时间及失败轨迹写入新的 `runs/` 子目录并
追加本文件；先成功的正常双机数值 oracle 才可进入 fault matrix。

### 72.66 2026-10-08: topology gate 失败复现与双机准入暂停

本轮没有启动 node8/node17 服务、训练、故障注入或 tmux 作业；GPU/双机结果没有新增。
先前 16-rank Gloo role-aware ZeRO gate 失败被完整复现并保存到
`runs/launch_logs/20261008T034756Z-topology-gate/production-16rank-gloo.log`：
**1 failed, 1 deselected, 1 warning**（约 9.7 秒），失败发生在测试对生产
`mpu.get_data_parallel_group()` 与 EP-only group 相等的断言；rank 15 首先报错，spawn
随后终止其余 worker。该失败是测试契约断言，不能说成训练失败，也不能说明后续 role-aware
ZeRO/sampler/cursor 断言已执行。它揭示 gate 的 expected group 仍沿用 EP-only 预期，未按
Merak composite DP=EP×replica-DP 的生产语义核验。

另对现存 `test/core/test_merak_engine.py` 运行 focused regression，日志
`runs/launch_logs/20261008T034756Z-topology-gate/engine-regression-before.log` 为
**2 failed, 1 warning**。两项都在 `_aggregate_total_loss` 读取 `pipeline.module` 时因 mock
fixture 没有 `module` 属性而退出；这是测试 fixture 不完整，不能据此推断生产引擎行为。
日志保留，不计 pass 或训练/recovery 结论。

生产入口审计确认 `run_workload.py` 当前 EP CLI `dp-pp` 映射
`MERAK_MESH_ORDER=dp,pp,sp,tp`，而 MoE layout order 是 `(dp,ep,pp,tp)`；既有
§72.57--72.58 记录指出该组合的 DeviceMesh PP group 与逻辑 EP PP group 不一致，legacy
fallback 下 PP 组虽可能对齐，composite data group 仍不是角色化 shared/expert ZeRO 组。
此前关于改成 `pp-dp` 会修好所有分组的推断没有经过完整 Trainer/ZeRO/UCP 验证，不能当作
修复结论。本轮短暂尝试变更 EP mesh order 与 MoE loss aggregation，因未完成完整 rank-set
和 numerical gate，已撤回这些未验证草稿；SemanticFixed、dense loss/collective、UCP
recovery 与其余历史 dirty changes 均保留。`merak_engine.py` 中先前已有的
effective-batch、RNG-device、role-aware ZeRO 等工作不属于本轮新验证。

本轮最后通过的检查仅是目标 conda 环境下相关 Python 文件 `py_compile` 和所选文件
`git diff --check`，二者 exit 0；这不等于测试或训练通过。当前明确 blocker 是需要一个单一
生产 rank-map 契约同时覆盖 DeviceMesh/legacy fallback、Merak pipeline stage 和 P2P peer、EP
dispatch、replica-DP sampler/effective batch、shared/expert ZeRO-1 组、loss/overflow/norm
collective 与 UCP rank ownership。必须先让 16-rank Gloo 报告并逐项验证真实 rank sets，再让
8-GPU NCCL Qwen tiny Trainer 完成 step 和同边界恢复 oracle；两关未过，不启动双机性能或
fault trial，不填五项指标，不更新 `docs/09-24.md`。

### 72.67 2026-10-08: composite-DP gate 修正与 ZeRO replica 参数顺序检查

本轮仅在单机执行 CPU/Gloo 和 4-GPU NCCL correctness tests；没有启动双机实验、性能测量、
故障注入或 tmux job。所有新增测试日志存于
`runs/launch_logs/20261008T-adacheck-moe-audit/`。

已证实的 ZeRO-1 静默错误条件：同一 replica process group 内参数 shape/dtype 相同但注册
顺序不同，旧初始化校验只比较 shape/dtype，无法发现 flatten/reduce-scatter 将梯度切片对应到
错误参数的问题。新增 2-rank Gloo reversed-registration regression，修改前曾按预期失败
（未抛出异常）；随后 role-aware optimizer 使用调用方传入模型的 canonical parameter names、
expert ID、shape、dtype 和 requires-grad 顺序签名校验副本。修改后
`zero1-reordered-replicas.log` 为 **1 passed**。该验证针对初始化拒绝错位，不替代完整训练
数值验证。

16-rank Gloo topology gate 的旧断言把生产 DeviceMesh data group 错误地与 replica-DP 轴
比较。Merak 生产配置的 DeviceMesh composite DP 轴是 EP×replica-DP；因此 gate 改为验证它
等于 shared/router composite group，同时单独检查角色化 ZeRO shared group 与 expert replica-DP
group。修正后 `topology-16rank-gloo-rerun.log` 为 **1 passed, 7 deselected**，涵盖 rank sets、
角色化分片计数、同步 overflow、sampler cursor round-trip；这不是实际 MoE Trainer 的 16-rank
训练或 UCP 恢复通过。16-rank NCCL 分支因本机只有 8 GPU 未运行。

单机 4×A40 Qwen2-MoE PP1/EP2/replica-DP2、role-aware ZeRO-1 Trainer 单步测试通过：
`qwen-moe-single-host-gpu.log` **1 passed**。它检查一步参数更新、shared 副本一致和 expert
replica 一致，但没有 checkpoint/restore。CPU Zero1/UCP numerical oracle
`zero1-optimizer-ucp-full.log` **2 passed**；stage-local checkpoint recovery 加 16-rank Gloo
topology 组合 `topology-and-ucp-targeted.log` **2 passed, 1 skipped**（16-rank NCCL 因设备不足
跳过）；Trainer glue `trainer-glue.log` **6 passed**。

较宽回归 `role-aware-zero1-ucp-regression.log` 结果为 **12 passed, 1 failed, 3 deselected**。
唯一失败是 PP4 Gloo Trainer fresh-engine 重建进入 `batch_isend_irecv` 后，PyTorch Gloo backend
报 `does not implement startCoalescing`；这不是成功结果，也不是 NCCL 生产运行失败。一次误用
pytest `-k` 的命令选中 0 项并 exit 5，最初日志内容随后被正确 test-id 的 NCCL 结果覆盖；
该误筛选不计测试结果。

因此已修复 replica 参数静默错位检查，并澄清 composite-DP gate 契约；仍不能宣称完整
AdaCheck-style detector + MoE + ZeRO-1/EP + UCP 端到端正确。尤其缺少真实 MoE Trainer 的
checkpoint fresh restore 后继续训练，并与相同数据序列的 uninterrupted optimizer boundary
逐项比较 model、master/moments/step、scheduler、RNG、sampler 和后续 loss。此前 §72.65 的
sampler/effective-batch 语义 blocker 与本轮测试中的简化 sampler fixture 仍在；UCP role-aware
对象身份也仍是 rank-local complete-state snapshot，不是已证明有效的 EP semantic
deduplication。故不报告五项正式实验指标、不写入 `docs/09-24.md`，不启动双机性能或
rank-failure recovery trial。

### 72.68 2026-10-08: MoE correctness follow-up 与双机实验准入边界

本轮未改生产训练、collective、dense SemanticFixed 或 UCP 实现；未启动 node8/node17 新
作业、性能计时或故障注入。没有覆盖/删除已有 run/session。本轮启动 shell 的 `conda run`
仍指向错误 base；测试使用 `/opt/conda/envs/sjw_ckpt_grouping/bin/python`。

重新执行 16-rank Gloo role-aware ZeRO topology gate：
`test_16rank_device_mesh_role_aware_zero1_groups -k gloo` 为 **1 passed, 1 deselected,
1 warning**（9.89 s）。该 gate 检查 16-rank rank sets、composite/shared 与 expert replica-DP
组、分片计数、overflow 同步和独立 sampler cursor round-trip；不是 16-rank MoE Trainer/UCP
运行。

重新执行单机 8×A40 NCCL Qwen tiny PP4/EP2/replica-DP1 role-aware ZeRO Trainer + mirrored
object recovery + fresh engine continued-step test：**1 passed, 1 warning**（16.22 s）。它在
fresh engine 上恢复 UCP rank-local object state，再比较第二步参数、ZeRO master/moments/step、
scheduler、loss scaler 和 optimizer group metadata。延续 §72.61 的测试限制：该测试
`sampler=None`，continuation 输入是固定手工 batch；不是真实 dataloader/sampler cursor 的
uninterrupted 对照，也不是双机 service 发布/恢复。

另执行 PP4 Gloo Trainer 对照，结果 **1 failed, 1 warning**（11.61 s）。所有 rank 在 fresh
engine 初始化的 Merak loss P2P warm-up 中调用 `batch_isend_irecv`，PyTorch Gloo 报
`Backend gloo does not implement startCoalescing`；故障在 UCP state apply 前，不表示对象恢复
数据损坏，也不能算 Gloo pass。8 个 rank traceback 保留于
`runs/launch_logs/20261008T-moe-correctness-followup/`。这是 Gloo backend 能力限制；生产
恢复门仍以本机 NCCL/双机 NCCL 为准，不用 mock P2P 伪造通过。

双机脚本审计发现，历史 runner 的 `source(max_steps=1)` 与 `restore(max_steps=2)` 没有同边界
uninterrupted 参考，因此不能用 step1/step2 digest 证明 continuation 数值等价；其 sampler
和 RNG digest 也未输出。另，`CKPT_ZERO1_ARM=adacheck_fixed` 在
`zero1_service.select_zero1_publications` 中只选择固定跨 host pair placement；它不是 AdaCheck
redundancy detector 调用，也不证明 semantic deduplication。role-aware EP runner 仍拒绝
legacy `--ada_checkpoint`。已有 §72.64 双机 run 只能保留为 service seal、source-service
下线、surviving-host fresh restore 并继续训练的 path smoke，不扩写为 exact oracle、detector
收益或正式性能结果。

本轮没有进行有效的双机端口准入探测：一次远端 Python one-liner 因 shell 换行转义错误退出，
没有 bind 或改动端口。node17 可见 29971 上有既存 `pt_main_t` listener 和多个用户 tmux
session，均保持原样；新试验不得复用/清理这些资源，须采用独立 service/master ports 并在
两端做精确 preflight。此前 `current-loss-contract.log` 只是源码文本 probe，明确未执行
distributed behavior，不计测试或结果。

当前结论：group/ZeRO/sampler Gloo gate 与单机 NCCL fixed-input UCP continued-step test 通过；
真实 sampler/RNG resume oracle、双机 uninterrupted same-boundary 数值对照、AdaCheck detector
接入 role-aware MoE ZeRO-1、五项正式指标和 rank-failure matrix 仍未完成。因此不报告有利
实验指标，不更新 `docs/09-24.md`，不开始双机性能/fault trial。下一步先增强独立 MoE runner：
在 `CKPT_FINAL_DIGEST_DIR` 写出 sampler cursor 与完整 runtime/RNG checksum，并运行同一
16-rank 配置、同一数据/scheduler/步数的 uninterrupted reference 与 UCP fresh restore；逐 rank
比较相同 optimizer boundary 的 model、optimizer、scheduler、sampler/RNG、iteration 和后续
loss。该 correctness gate 通过后再由 tmux 启动 node17/node8 正常双机复测，之后才进入五指标
paired performance 和分类型 fault study。

### 72.69 2026-10-08: 固定输入恢复 oracle 复验与 MoE policy 对齐审计

本轮保留所有既有工作区改动；只修改了独立 MoE 测试
`test/core/moe/test_production_topology_groups.py`，将 fresh-engine continuation
oracle 的输入恢复为两个相同的显式固定 batch。此前一次未留可读日志的试跑将
`fresh_trainer.train_dataloader` 直接交给 `train_batch`，不满足该 oracle 原先“隔离 UCP/ZeRO
状态恢复、固定后续输入”的测试合同；不能根据该次结果归因生产恢复失败。修正后使用
`/opt/conda/envs/sjw_ckpt_grouping/bin/python` 复测，PP4/EP2/replica-DP1、8×A40 NCCL 的
Qwen tiny Trainer + UCP mirrored object recovery + fresh engine continuation 为 **1 passed,
7 deselected, 1 warning (16.72 s)**，日志为
`runs/launch_logs/20261008T-moe-correctness-followup/pp4-nccl-ucp-oracle-rerun.log`。它比较恢复
后的下一训练步与固定输入参照的 model、ZeRO master/moments/step、scheduler、loss scaler、
optimizer group metadata；sampler 仍为 `None`，因此不是 sampler/RNG 真实恢复门。

执行 `test/core/checkpoint/test_zero1_object_adapter.py`：**14 passed, 1 warning (7.73 s)**，
日志为 `runs/launch_logs/20261008T-moe-correctness-followup/zero1-policy-adapter.log`，覆盖
ZeRO-1 fixed-pair、host-global 与 M1 发布选择、等版本对象验证及冲突拒绝。此结果只证明
CPU adapter 行为，不证明物理故障域恢复、在线训练 detector 或完整 MoE 四臂公平性。

MoE 与 dense 四策略对齐仍有实质差距：当前 runner 的 `adacheck_fixed` 是完整
role-aware ZeRO-1 rank-local state 的 fixed-pair object placement 标签，不是 AdaCheck detector
和 SemanticFixed encoder；生产 MoE runner 拒绝 `--ada_checkpoint`，而通用 SemanticFixed
不支持 ZeRO-1 local ranges。`host_global_fixed` 与 `m1` 目前同样是该快照适配器上的 placement
策略。因而不能把这三个标签写作四个 dense checkpoint grouping policy，也不能把无检查点
control 或已有完整状态快照当成 MoE SemanticFixed 实验。必须先定义同一 MoE object catalog、
故障预算、发布/编码合同下可执行且可恢复的四臂策略；否则 byte、max-edge、提交频率、训练
吞吐和恢复指标不具备公平对照意义。dense SemanticFixed 保持不动。

本轮没有启动 node17/node8 作业、tmux session、性能统计或故障注入。双机 runner 仍没有
same-boundary uninterrupted reference，也未证明真实 sampler/RNG cursor digest；固定端口未做
有效双端占用预检，不得启动。先实现独立 MoE 的固定数据序列 same-boundary oracle（覆盖 runtime
RNG 与 sampler），并做端口/作业精准预检；再以 tmux 跑双机 correctness。该 gate 通过后才能
按 `docs/09-24.md` 的同一五指标 schema 运行四臂、重复实验和故障注入，原始结果追加到
`runs/` 并保留失败状态。

### 72.70 2026-10-08: ZeRO-1 opaque RNG/sampler round-trip gate

本轮仅在 MoE 专项测试中扩展 sampler/runtime-state oracle；未改生产 Trainer、ZeRO 实现、
dense SemanticFixed 或 dense 实验文件。所有测试均使用
`/opt/conda/envs/sjw_ckpt_grouping/bin/python`，日志保存在
`runs/launch_logs/20261008T-moe-correctness-followup/`。

在 `test/core/moe/test_zero1_optimizer.py` 的 4-rank Gloo role-aware ZeRO-1/UCP object
recovery 测试中，以 `MegatronPretrainingRandomSampler` 替代 `sampler=None`，在 step-2
checkpoint 边界保存 replica-DP cursor，并经 `restore_runtime_state` 恢复到 fresh sampler；
验证恢复后的 consumed-sample cursor、epoch、下一 sample index，以及 Python/NumPy/Torch RNG
输出。相同 oracle 还覆盖 fresh ZeRO optimizer、TCP service surviving-host restore 与
`build_zero1_object_restore_controller`，并比较 continued-step model、Adam master/moments、
optimizer step 和 scheduler。最终日志
`zero1-ucp-sampler-rng-controller.log`：**1 passed, 1 warning (11.29 s)**。

两次中间失败均保留原始日志且未计为通过：第一次 negative topology fixture 对带 sampler
状态调用 `sampler=None`，触发正确的 `ZeRO-1 sampler state is incompatible`；第二次正式
memory restore fixture 仍传 `sampler=None`，同样被 fail-closed 拒绝。随后给 negative fixture
提供同型 sampler、并给内存恢复和 restore controller 接上 fresh replica-DP sampler，最终
用例通过。日志分别为 `zero1-ucp-sampler-rng-oracle.log` 与
`zero1-ucp-sampler-rng-oracle-rerun.log`。这两项是测试夹具 mismatch，不是生产状态损坏；
同时确认省略 sampler 的调用不能消费带 sampler opaque root 的检查点。

MoE CPU 回归 `test_zero1_optimizer.py test_zero1_trainer_glue.py test_expert_parallel.py` 为
**18 passed, 1 warning (41.36 s)**，见 `moe-cpu-regression-rerun.log`。单机 8×A40 NCCL
Qwen tiny PP4/EP2/replica-DP1 Trainer + UCP mirrored object restore continued-step 用例为
**1 passed, 1 warning (16.17 s)**，见 `pp4-nccl-ucp-after-sampler-gate.log`；该 NCCL Trainer
用例仍显式固定 continuation batch、sampler=None，不应和新增 4-rank Gloo sampler gate 合并
表述成一个端到端真实数据流水线证明。16-rank Gloo topology gate仍是独立已有证据，非 16-rank
Trainer/UCP NCCL 运行。

研究结果尚未进入 dense 对齐阶段：该轮没有 node8/node17 tmux 作业、双机同边界 uninterrupted
oracle、故障注入、长窗口或五指标测量。当前 `adacheck_fixed` 仍是 UCP ZeRO-1 fixed-pair
placement，不是 AdaCheck detector/SemanticFixed encoder；两主机下 host-global 与 M1 对同一
必需对象可能有相同最低保护字节，需按 `docs/09-24.md` 的故障覆盖下界与实测 service load
解释，不能预设 MoE transfer 一定下降。四臂策略要在同一 MoE object/encoder/故障预算合同中
实现后，才测总 application/wire transfer、最慢物理 edge 与 capture-to-seal、globally
committed frequency/lag、训练吞吐/阻塞、故障后恢复及 continuation correctness；本轮没有可
填写到 `docs/09-24.md` 的 MoE 性能数值。下一门是将这个 sampler/runtime oracle 纳入真实
MoE Trainer fresh-job 双机 same-boundary test，并预检独立端口/进程后用 tmux；通过后再做四臂
paired 16-rank runs 和预注册 fault matrix。

### 72.71 2026-10-08: 双机 16-rank真实 Trainer 首步门禁失败

本轮保留所有既有未提交研究改动；仅调整独立 MoE runner
`tools/run_two_node_moe_zero1_smoke.sh` 和本记录。runner 的训练作业进程预检改为匹配
`torchrun` 进程名及 `run_workload.py` 参数；source/reference/restore topology metadata
分别写入各自 phase 目录，避免后续 phase 覆盖 source 记录。Replica service 以
`0.0.0.0` bind 时会把 wildcard 写进 ready file；runner 现将其转换为已核实的 node17
`172.20.17.2`、node8 `172.20.8.2` endpoint，并用两端 `STATS` 请求做训练前健康检查。
shell 语法、`run_workload.py` py_compile 与 `git diff --check` 通过。

首次 tmux run `20261008T-recovery-oracle` 在训练前失败：两 service 的 ready endpoint
为 `0.0.0.0:port`，跨主机连接被拒绝。原始日志保留于
`runs/two-node-moe-zero1/20261008T-recovery-oracle/`，没有启动训练。

修正 endpoint 后第二次双机 run `20261008T-recovery-oracle-b` 两端 service health check
通过，随后真实 16-rank tiny Qwen PP4/EP2/replica-DP2/TP1 role-aware ZeRO-1 Trainer
source 在首个训练 step 未取得进展。两端 rank 在模型分区和初始化日志后没有 step/loss
或 UCP publication；两服务仍为 0 records/0 bytes。约 10 分钟后 node8 rank 8 首先退出，
其他 node8 ranks 的 PG0 `ALLREDUCE` watchdog 报 600 秒 timeout；node17 ranks 随后因
peer exit 报 NCCL remote error。由于操作者在观察停滞后发送了 SIGINT，watchdog 时间点
与中断重合，故目前只确认该 collective 未完成，不能据此断言自然超时或唯一根因。此 run
非零退出，没有进入 uninterrupted reference、fresh restore、digest 比较或 fault trial，
不计为通过且不生成性能指标。原始编排、torchrun、两端 runner/service 日志及 status
保留于 `runs/two-node-moe-zero1/20261008T-recovery-oracle-b/`；本次专属训练及
service tmux sessions 已停止，无本 run 后台 GPU process。

该失败后复跑单机 8×A40 CUDA 用例
`test_cuda_qwen_tiny_pp4_ep2_dp1_role_aware_zero1_trainer`，结果 **1 passed, 1 warning
(17.70 s)**，完整输出为
`runs/launch_logs/20261008T-two-node-followup/one-node-pp4-ucp-rerun.log`。该用例是固定
continuation batch 的单机 Trainer + UCP fresh-engine regression，不含真实 sampler，不能
覆盖或抵消双机 source 首步失败。

本轮唯一双机训练准入结果为失败；健康检查成功不等于训练成功。现有单机 8-rank
fixed-input NCCL recovery test、CPU/Gloo sampler/RNG gate、16-rank Gloo topology gate
与本次双机真实 sampler Trainer run 的证据范围各自独立。下一步先复跑单机 8-rank
EP+PP tiny Qwen Trainer/UCP CUDA gate，再构造逐阶段日志与 collective 观察的 16-rank
门禁，将首步拆为 model forward/EP dispatch、backward 与 optimizer boundary，定位阻塞
在 UCP 之外还是 ZeRO/controller。只有 source、同边界 uninterrupted reference 和 fresh
restore 的逐 rank model/optimizer/scheduler/RNG/sampler digest 全部匹配，才开始四种可恢复
grouping policy 和 `docs/09-24.md` 五指标对齐。dense SemanticFixed 保持不动；本轮无可填入
09-24 的 MoE 性能结果。

### 72.72 2026-10-08: 双机 MoE 首步重试与 UCP 初始化停滞

本轮先复核 dense 对照，未改 `docs/09-24.md`、dense SemanticFixed、训练或 UCP
生产实现。dense GPT-6.33B r229 的记录为 M1 总 application transfer
113.901 GB（Ada 189.834 GB）、suffix transfer 25.313 GB（Ada 101.246 GB）、
max suffix PUT 22.094 s（Ada 70.527 s）；但只有两个 source commit 后 drain，且服务失效后
fresh-job same-layout restore，不是持续 1C1S、compute-host fault 或多次独立试验。MoE
比较仍须按本文件 §72.66 和 `PLAN_TWO_NODES_V2.md` 的四臂合同：NoCheckpoint、
AdaCheckStyleFixed、HostGlobalFixed、M1Object；同一 detector/encoder、对象目录、故障预算及
恢复语义，分别统计 `docs/09-24.md` 定义的五项指标。当前 MoE 三个 placement 标签还不是
前三种可比策略，不得提前填性能表。

先前 `20261008T-recovery-oracle-b` 在首步前卡住。原 runner 的 phase worker 直接启动时没有
来源码里 `tools/launch_ckpt_grouping.sh` 设置的 `GLOO_SOCKET_IFNAME=eth0`；另一个手动无
UCP trace 重试也犯了相同配置遗漏，node8 rank 8--15 明确警告回退 loopback，随后 Gloo
`connectFullMesh` 失败。该诊断由本轮操作者错误造成，保留在
`runs/two-node-moe-zero1/20261008T-noucptrace/`，不当作模型/UCP 缺陷。

随后经正式 launcher、两端 `eth0` 环境，在 node17/node8 tmux 运行相同 16-rank tiny-Qwen
PP4/TP1/EP2/replica-DP2 role-aware ZeRO-1、真实 replica-DP sampler、无 UCP 的 1-step trace：
两端 exit code 均为 0；rank 0 loss `3.4234`、训练 step `2.532 s`（rank 0 outer step wall
`2.533 s`），16 ranks 均有 `engine.train_batch` 和 `trainer.train` POST。日志在
`runs/two-node-moe-zero1/20261008T-noucptrace-fixed/`。这是一个短正确性重试，不是吞吐基准、
恢复验证或五指标数据。

再用同一正式 launcher 和拓扑，启用 ZeRO-1 UCP、同一训练 step 并请求 epoch 1 publication；
run `20261008T-ucp-one-step` 在两节点模型 partition/Trainer 构造日志后无 `train_batch`
开始标记、loss、locator、rank digest 或 publication。现有本次日志没有 Python traceback 或
定时栈，故无法将停点缩小到 UCP 初始化中的具体 collective，也不能把它归因为 model
capture、对象去重或网络上传。两个预检 service endpoint 均已通过前一健康检查；此 run 没有
新的 service record 可证明 epoch 已开始。两端精确命名的本次 tmux 已停止，本机 GPU 显存
回到 0 MiB；node17 上既有 `20261007T-ucpdiag2` service 仍在运行且未触碰，node8 当时无该
run 的残留训练进程。日志保留在
`runs/two-node-moe-zero1/20261008T-ucp-one-step/`，不计为通过。

辅助环境现象：双机 CUDA worker 启动时 UCX 多次报告
`inotify_add_watch(/tmp): No space left on device`；当前交互容器 `df -h/-i /tmp` 正常，
因此不能由容器指标推断远端 mount/namespace 资源状态，也尚无证据证明该 UCX warning 是
UCP 停滞原因。后续先为 UCP one-step 单独加入可控、可关联 rank 的阶段/stack 观测并显式
设置短 default-process-group timeout，区分 Gloo group 建立、Trainer 构造、训练首步和
publication；同时采集 node8/node17 本地 `/tmp` 与 UCX 状态。不得先改 object identity 或
静默放宽 equal-version 检查。本轮尚无 MoE source sealed epoch、same-boundary uninterrupted
oracle、fresh restore、fault trial 或 dense 对齐五指标，`docs/09-24.md` 不新增 MoE 数字。

### 72.73 2026-10-08: loss broadcast root 失败复核与测试失败归档

继续排查 §72.72 的 UCP-enabled r2。完整 traceback 已在
`runs/two-node-moe-zero1/20261008T-ucp-phase-diagnostic-r2/torchrun/` 找回：
16 ranks 均开始 `trainer.train()`，失败位于
`CommunicationEngine._aggregate_total_loss()` 的 PP loss broadcast，NCCL 报
`Broadcast: invalid root 3 (root should be in the 0..3 range)`；sender ranks 3、7、11、15
也在同一 collective 失败/随后观察到 peer exit。两个 node status 均 exit 1，未产生 loss、
UCP locator、digest 或 checkpoint publication。之前把这解释为 UCP control-group 初始化停滞
不准确，现更正为首个训练 step 的 loss collective 失败。PyTorch 2.3.1 本地源码确认
`dist.broadcast(src=...)` 的 `src` 是 global rank，并在调用 ProcessGroup 前转换为 group rank；
当前 Merak 调用传入 stage-to-global rank，不能仅凭 warning 推断 API 使用 global/local 错误。

该错误尚无根因结论。相同参数布局的既有双机无 UCP run
`20261008T-noucptrace-fixed` 及 2026-10-07 的 Qwen smoke/UCP source 均成功执行 loss
broadcast，故不能据当前一次 NCCL 错误断言通用 PP topology 或 UCP 有缺陷。需要从有效
`trainer.engine.comm_engine` 读取并记录 global rank、stage ID、`grid.pp_group`、PP ProcessGroup
成员、最后 stage 的 global source，以及 process-group identity；再与已成功 run 的日志和
实际加载代码版本比较。确认之前不修改 `CommunicationEngine` collective，也不把本轮失败说成
训练或恢复通过。

目标环境 focused regression：
`/opt/conda/envs/sjw_ckpt_grouping/bin/python -m pytest -q test/core/test_merak_engine.py test/core/moe/test_expert_parallel.py test/core/checkpoint/test_zero1_object_adapter.py test/core/moe/test_production_topology_groups.py -k 'not nccl'`
结果 **29 passed, 3 failed, 2 deselected, 1 warning**。两项失败来自
`test/core/test_merak_engine.py` mock 缺少 `grid.pp_group` 和 MPU group 初始化；第三项
`test_gloo_qwen_tiny_pp4_ep2_dp1_role_aware_zero1_trainer` 失败于 Gloo backend 不支持
Merak pipeline P2P 所需的 `startCoalescing`。这些失败没有验证或推翻真实 NCCL root cause；
当前不得笼统引用整个 suite 为通过。PyTorch 版本、traceback、失败 logs 保留在 pytest 输出和
本节引用的 `runs/` 原始文件中。

本轮也确认已成功的 10 月 8 日无 UCP run 使用同样 `--dp 4 --mesh-order dp-pp`、EP2/PP4/TP1，
但本次 phase diagnostic 改动加入了更多 group 初始化/共享 metadata 路径。因此目前既没有证据
表明它们导致 NCCL root 错误，也没有证据排除影响。dense SemanticFixed、`docs/09-24.md`、
既有未提交研究改动均未改；没有新恢复或五指标结果。下一步只做可复现的运行时 rank/group
观测和最小生产路径回归定位；通过之前不启动 UCP publication/fault/performance trial，不填
MoE 指标。

补充本节：新建最小 CUDA/NCCL collective gate
`runs/two-node-moe-zero1/20261008T-loss-broadcast-gate/nccl_broadcast_gate.py`，调用真实
`set_topo_grid_communication()`、DeviceMesh PP group 和 PyTorch 2.3.1
`dist.broadcast(src=global_rank, group=pp_group)`，在 8×A40 上逐 rank 验证 source/value。
结果 **8 ranks 全部 BCAST_GATE_POST，exit_code=0**：groups `(0,1,2,3)` 使用 source 3，
`(4,5,6,7)` 使用 source 7；没有 invalid-root。日志及状态留在同一目录。随后目标环境真实
单机 CUDA Trainer gate
`test_cuda_qwen_tiny_pp4_ep2_dp1_role_aware_zero1_trainer` **1 passed, 1 warning (16.22 s)**，
warning 为 pynvml 弃用；该 gate 完成 PP4/EP2/role-aware ZeRO-1 tiny-Qwen 训练一步。此结果
不是双机证据，也不证明 rank 8–15 的失败 root 问题已解决。

因此当前可排除的是“8-rank NCCL 上 Merak PP group 本身不能从末 stage 广播”；未排除双机时
effective grid、PP ProcessGroup identity/membership、engine global rank 与默认 process-group rank
不一致。失败日志只记录报错的 root 3，未保存调用点 rank/source/group，root cause 仍未知。
下一步双机复测应在 `_aggregate_total_loss` 每次 broadcast 前用诊断 wrapper 记录：
`dist.get_rank()`、`self.global_rank`、`self.stage_id`、last-stage flag、computed source、
`grid.pp_group`、`dist.get_process_group_ranks(mpu.get_pipe_parallel_group())` 及
`dist.get_group_rank(group, src)`；并和成功 run 做同配置逐字段比较。没有该证据前不改 shared
collective、不重启性能/故障实验。本次 NCCL gate session 已退出，所有 8 GPU 显存为 0；
`20261007T-ucpdiag2` 历史 service 不受影响。

### 72.74 2026-10-08: 双机 loss broadcast 复现与初始化差异隔离

本轮在 `CommunicationEngine._aggregate_total_loss()` 加入环境变量门控的只读诊断：仅当
`CKPT_MOE_TRACE_LOSS_BROADCAST=1` 时打印 default/global rank、engine rank、PP stage、source
global/group rank、`grid.pp_group`、实际 PP ProcessGroup 成员与本地 group 对象标识；不更改
collective 参数或训练状态。`test/core/test_merak_engine.py` 更新 mock 并新增诊断分支测试，结果
**3 passed, 1 warning**。目标 conda 环境 `py_compile`、shell syntax 与相关 `git diff --check`
通过。此前 focused suite 的 **29 passed, 3 failed, 2 deselected** 是独立历史结果，不能覆盖
本次单文件门禁。

双机 16-rank run
`runs/two-node-moe-zero1/20261008T-loss-broadcast-probe/` 使用正式 launcher、node17/node8、
PP4/TP1/composite-DP4/EP2/replica-DP2、tiny-Qwen、ZeRO-1、无 UCP，打开上述诊断。两端 status
均 **exit_code=1**；rank 0--15 实际 `grid_pp` 与 PP ProcessGroup members 一致，computed
source 映射为各自 group local rank 3；NCCL 仍在 loss broadcast 报
`invalid root 3 (root should be in the 0..3 range)`。日志没有显示 `_aggregate_total_loss`
之前的 collective 被成功 `POST` 的逐操作 trace，故根因未确定。原始日志、两个节点日志与
status 均保留于该目录；两端进程已退出、GPU 显存归零。

新增双机 NCCL 隔离 gate `runs/two-node-moe-zero1/20261008T-loss-broadcast-gate-16/`：
1. production DeviceMesh PP groups 的 16 ranks/four groups broadcast 全部 `BCAST16_POST`；
2. 再加入 MoE shared/expert/control Gloo groups 及 singleton groups 后仍通过；
3. 再加入真实 tiny-Qwen EP 分片模型和逐参数 shared/expert NCCL 初始化广播后仍通过。
上述各 gate 两节点均 `exit_code=0`。它们说明 PP group 构造、source 映射以及独立运行的
模型参数广播不足以单独复现训练错误；不等价于完整 PipelineModule/Trainer 初始化。

已有 `test_16rank_device_mesh_role_aware_zero1_groups -k gloo` **1 passed, 7 deselected**，
验证 Gloo 下 16-rank topology、role-aware ZeRO-1 参数副本与 sampler contract；不验证 NCCL。
因此本轮证据不支持“EP PP group 与 production DeviceMesh 不匹配”，相关既有测试明确断言两者
一致。也不支持把错误归咎于 UCP：上述失败 probe 未启用 UCP。

更正 §72.73 的比较描述：`20261008T-noucptrace-fixed` 命令确实显式传入
`--tiny-qwen-layers 4`，但 partition 日志仍为 7 个 pipeline layers、stage layer counts
1/2/2/2；`20261007T-ucpsourcefix` 同样显示 1/2/2/2。该参数表面差异不能作为当前 root
cause。成功旧 run 的 ZeRO/UCP source loss broadcast 结果仍是历史成功证据，但没有对应的
loss 调用点 group/source trace；本轮新 probe 的实际 PP group/source 信息因此不能与旧 run
逐字段比较。

下一步：用已有 `runs/two-node-moe-zero1/20261007T-broadcast-trace/trace_broadcast.py` 思路，
在完整 `PipelineModule._broadcast_model()` 和 ZeRO 构造期间记录**所有** NCCL/Gloo
collective 的 PRE/POST（rank、group members、source、op、参数名/role），先与 10 月 7 日成功
UCP source run 的可用 trace 比较，再用唯一 run id 在 tmux 做 16-rank 最小复测。重点检查 PP
group 创建之后、loss broadcast 之前是否某个 process group collective 未对齐，而不是继续
替换 root local/global rank。没有恢复 oracle 和三组语义等价 grouping arms 前不跑性能/故障
矩阵、不填 `docs/09-24.md` MoE 五指标。dense SemanticFixed 及其他已有研究改动未改动。

### 72.75 2026-10-08: PP broadcast root/backend 隔离

独立代码审阅曾提出 `src` global/group rank 转换可能导致 invalid-root。核对本机及 node 8
安装的 PyTorch 2.3.1 `distributed_c10d.py` 后确认 `dist.broadcast(src=...)` 的 `src` 是 global
rank，包装层会经 `get_group_rank()` 转成 `opts.rootRank`；因此不能直接把该解释当作根因。

运行 `runs/two-node-moe-zero1/20261008T-mesh-backend-trace-r3/` 双机 16-rank microgate，顺序
与失败 MoE 初始化相同：Merak DeviceMesh PP groups，EP/DP Gloo groups，shared Gloo groups，
world control Gloo group，singleton groups。每个 PP group 均观测到 PyTorch PG 和 CUDA backend
`size=4`、`rank=0..3`，`group_name` 对四组各异且 members 与 `grid.pp_group` 一致。对 PP
groups `[0..3]`、`[4..7]`、`[8..11]`、`[12..15]` 先后执行 `dist.broadcast(global_src)` 和直接
`ProcessGroup.broadcast(rootRank=3)`；所有 rank、两种 API 均 `BCAST_POST`，两节点 status
均 `exit_code=0`。这排除了单独的 PyTorch global/group 映射问题和这些 Gloo/singleton 组创建
顺序在 microgate 中破坏 PP communicator 的假设；不等价于完整 Trainer collective 序列。

该 run 的第一次启动因未设置 `CKPT_RUNTIME_ONLY=1` 在 launcher 环境检查处退出 3；第二次因
run id 切换时 gate 路径错误在 torchrun 前退出 2。两次均未启动 workers/GPU，原始 wrapper
日志/status 保留于 `20261008T-mesh-backend-trace/` 和 `...-r2/`。修复 run-local 启动脚本后，
`...-r3/` 才是有效 microgate。脚本编译、shell syntax、diff whitespace checks 通过。

根因仍未定位。失败仅在完整 Trainer 的 `_aggregate_total_loss` 报 NCCL invalid root，然而
独立 microgate 及既有 16-rank NCCL gates 均通过；下一步应在完整训练中记录实际传给
`ProcessGroup.broadcast` 的 options root、tensor device/dtype/shape、group backend metadata，
并跟踪同一 PP group 从 pipeline 参数广播、ZeRO 初始化到 loss broadcast 的 NCCL collective
顺序。不要据当前证据修改 loss collective 或宣布训练恢复通过。模型正确性、UCP recovery
oracle、checkpoint grouping 比较实验及五项指标仍未通过/未测；dense SemanticFixed 和
`docs/09-24.md` 未修改。

### 72.76 2026-10-09: MoE replica-DP loss metadata 与双机 UCP restore oracle

本轮只在 MoE metadata 路径添加 `_merak_replica_data_parallel_group`，由既有
`PipelineModule` `_merak_*` 属性传递至 `CommunicationEngine`；loss averaging 对 MoE 使用
真实 replica-DP group/size，普通 dense 路径仍从 MPU 取得原 group。runner 不再改写全局 MPU
data group。loss buffer 改为从 loss tensor reshape/cast 到 pipeline device，避免构造 CPU
`torch.Tensor([agg_loss])` 后再搬运。该改动与旧 invalid-root 失败 trace 不是单变量对照，故不能
宣称已定位或单因解决 §72.73–72.75 的 root cause。

本轮验证：

- `test/core/test_merak_engine.py test/core/moe/test_zero1_trainer_glue.py`：**10 passed**；
  覆盖 dense fallback、MoE replica-DP loss group 选择和原有 glue。
- `test_device_mesh_grid_group_report` 两种 mesh order：**2 passed**；确认没有全局 MPU
  data-group side effect。
- 新增/修正后的单机 CUDA recovery gate
  `test_cuda_qwen_tiny_pp4_ep2_dp1_role_aware_zero1_ucp_restore`：**1 passed, 1 warning
  (16.06 s)**。范围仅 `PP4/EP2/replica-DP1`，测试内部模拟 host failure 并 fresh engine
  restore/continued-step，不是物理主机故障。
- 双机无 UCP NCCL 全步 trace
  `runs/two-node-moe-zero1/20261009T-replica-loss-r1/`：node17/node8 各 8 rank，均
  `exit_code=0`；step loss `3.4234`，step wall `2.282 s`，throughput `0.877 samples/s`，
  loss all-reduce 与四个 PP NCCL broadcast 均 POST。该结果说明当前源码在该配置能完成一步，
  不能据此把旧 invalid-root 归因于本轮某一改动。
- 正式双机 UCP smoke 的有效完整 run 为
  `runs/two-node-moe-zero1/20261009T-formal-ucp-r5/`。PP4/TP1/EP2/replica-DP2、
  role-aware ZeRO-1 tiny Qwen source 3 steps，在 epoch 1 通过两个真实 ReplicaService
  seal UCP object checkpoint；uninterrupted reference 3 steps；停止 node17 service 后，
  从 node8 surviving service 对 fresh 16-rank job 恢复 epoch 1 并继续至 step 3。三个 phase
  两端均 `exit_code=0`，restore 日志有全部 16 ranks `Resumed from UCP ZeRO-1 object epoch 1`。
  离线 oracle 比较每 rank model、optimizer、scheduler、runtime component checksum：
  reference 与 restore **16/16 全匹配**；source 与 reference 也 **16/16 全匹配**。
  sampler cursor 按真实 pipeline input stage 判定：PP stage 0/3 的 rank 为 6，stage 1/2
  的 dataloader 被 Merak 替换为 `range`、opaque sampler state 为 0；三 phase 的同 rank
  sampler state 一致。首次 oracle 脚本错误地要求全部 rank cursor=6，已如实记录为校验失败；
  按 `input_to_stage_dic` 的 stage 0/3 数据所有权修正后离线 16-rank digest oracle 通过。
  该结论限于相同布局的 service failure/surviving-service recovery；没有杀训练 rank 或物理
  host fault injection。

UCP source publication 单次观测：epoch 1 `remote_payload_bytes=787194`、capture
`0.008731 s`、publish `0.644148 s`、`maximum_remote_put_ms=20.9983`。这是一个 tiny-model
单次 run，非五策略比较或稳定性能结论，不填入 `docs/09-24.md`。

runner 失败记录不可覆盖：`20261009T-formal-ucp-r1` source 在训练前因 launcher 未导出
`NODE_RANK` 退出 1，修复只加在 MoE smoke runner 的 `env`。`r2` source 三步和 seal 成功，
但下一 phase 撞到 dead tmux session 名；`r3` source 成功但 orchestrator 因同一 dead-pane
生命周期问题未进入 reference；`r4` 同样 source 成功、phase 间 dead tmux 名冲突未完成。
`r5` 才完整完成 source/reference/restore。无效 Gloo Trainer selector 曾返回
“test not found”，没有启动 worker；正确的 Gloo Trainer 尝试因 Gloo 不支持
`batch_isend_irecv` 的 `startCoalescing` 失败，不能算训练测试通过。前述失败日志均保留，未
覆盖旧目录。

尚未完成：`adacheck_fixed` 以外的三种 grouping policy、同合同的五指标对齐、多次随机顺序
重复、DeepSeek-V2-Lite、checkpoint transfer 五指标完整统计及 rank/host 故障注入。因此当前
只能报告 MoE 一步训练与 AdaCheck-style 固定组 UCP 同布局恢复正确性；不能称研究目标全部
完成、不能宣称积极性能收益。Dense SemanticFixed、`docs/09-24.md` 与 UCP dense 路径未修改。

### 72.77 2026-10-09: 恢复证据固化与 MoE/ SemanticFixed 对齐边界

复核 `docs/09-24.md` 与当前实现后，明确区分 dense 主线合同和 MoE ZeRO-1 实验：正式归因顺序
是 `FullState -> AdaCheckStyleFixed -> HostGlobalFixed -> M1Object`；`NoCheckpoint` 仅为训练开销
下界，不能替代 Ada fixed 的 transfer/recovery baseline。MoE `run_workload.py` 对 `--ada_checkpoint`
的 SemanticFixed 完整状态路径仍不兼容 role-aware ZeRO-1 local ranges；`tools/preflight_moe_ep_zero1.py`
也明确报告 `semantic_fixed_compatible=false`。当前 `Merak/core/checkpoint/zero1_service.py`
的 selector 只含 `adacheck_fixed`、`host_global_fixed`、`m1`，其数据面是 ZeRO-1 rank-local
object snapshot，不是 dense SemanticFixed 的共同 detector、active encoder 与梯度更新链。
所以当前三臂 smoke 不能伪装成 dense 四臂结果；两主机拓扑下 host-global/M1 字节等价的预期
也不能自动外推到 EP 专家对象。

将 r5 离线 oracle 的可复核结果保存至
`runs/two-node-moe-zero1/20261009T-formal-ucp-r5/recovery-oracle.json`。该文件由既有 16-rank
source/reference/restore digest 重算，source-vs-reference 与 reference-vs-restore 的 model、
optimizer、scheduler、runtime checksum 均为 16/16；同时显式记录原 runner 未写出 `COMPLETE`
（最初 sampler oracle 误将所有 PP rank 视为 dataloader owner），不补写 COMPLETE、不更改原始日志。
两节点各 phase status 均 exit 0，恢复作业从 sealed epoch 1 继续到 step 3。发布单次观测仍为
787194 remote bytes、capture 0.008731 s、publish 0.644148 s、max remote PUT 20.9983 ms；不把
这些单次 tiny-model 指标说成四策略收益。节点 8 GPUs 全部空闲，本机与 node8 未发现该 smoke
遗留 torchrun worker。

本轮最终 focused regression：
`pytest -q test/core/test_merak_engine.py test/core/moe/test_zero1_trainer_glue.py
test/core/moe/test_production_topology_groups.py::test_device_mesh_grid_group_report`，**12 passed,
1 warning (20.25 s)**；warning 为环境中的 pynvml 弃用。`git diff --check` 通过。工作区其余
既有 dirty changes 均保留。

后续 MoE 五指标正式比较的前置工作不是扩写 selector，而是给 MoE 定义并验证与 dense 相同语义的
对象 catalog、detector/encoder 和完整恢复状态，再实现 Ada fixed、HostGlobalFixed、M1 在同一
ZeRO-1/EP 合同下的三臂，另把 FullState 作为 detector attribution arm、NoCheckpoint 作为训练
开销 control。当前在线 selector 的 `adacheck_fixed` 只是 placement 标签，故也不能仅凭名称称为
“AdaCheck detector baseline”。在这道兼容性与共同合同 gate 通过前，不启动长时四策略统计或 rank
SIGKILL；避免产出指标不可比的矩阵。dense SemanticFixed、`docs/09-24.md` 与已存在研究改动未修改。

### 72.78 2026-10-09: MoE ZeRO-1 publication metrics instrumentation

本轮修复 `zero1_service.py` 中 distributed observer 的导入/API 错误；legacy serial publisher
恢复原来的 `commit_object_update_state` 路径，没有改其返回契约。分布式 MoE observer 分开记录
跨主机 data bundle 的序列化 payload、逻辑对象字节、client wire 字节与 PUT-to-commit-ACK 最大
间隔；host-1 控制 bundle 的 payload/logical/wire 单列。`maximum_remote_put_ms` 单指客户端观察的
PUT 调用，不能与 data-edge ACK interval 混称。全状态 UCP 快照包含 runtime/opaque 信息，既不等于
逻辑 tensor bytes，也不是 dense 的增量梯度 transfer size；client wire bytes 不等于物理 NIC
计数。训练 step wall 按 rank 分文件，避免 rank0 覆盖全局分布。

focused 验证：`test/core/moe/test_zero1_optimizer.py` **2 passed, 1 warning (16.65 s)**；warning
为环境 pynvml 弃用。之前失败均未隐藏：首轮 traceback 是 serial publisher 遗漏 import；修正后
发现分布式 gathered list 长度误判；再后指出 Gloo fixture 的 data-only logical tensor bytes 可为
零，故改为验证总逻辑 bytes 与 control/data 组成关系。随后本机回归通过。runner 调整为 16 个
样本、replica-DP2、已验证的 `max_steps=3` smoke；源作业在 step 1 发布，restore epoch 1 后继续至
step 3。当前新 runner draft 将边界移至 step 2，使恢复作业必须实际执行 step 3，再与 uninterrupted
step 3 digest 对照，并保留 node17 service 到 step-2 source/restore digest 比较之后再停止；这可
验证训练 continuation，但仍不等于独立逐样本输入 trace。该短 smoke 的
checkpoint frequency 只能记为 1 committed/3 optimizer steps，不能与 dense
长窗口 cadence 实验等同。测试/runner 草稿曾错误假设可按 epoch 发布 steps 1/2/3，后经 Trainer
循环和 pipeline optimizer schedule 审阅撤回，未启动相应 GPU run。测试/runner 修正前曾出现失败：一次
错误要求空 tensor catalog 的 data logical bytes 非零，另一次把 sampler cursor 误估为 24；已按
control/data logical 口径及 `micro_batch × replica-DP-size` 修正，失败仍保留记录且不记为 pass。
shell、py_compile、`git diff --check` 均通过；埋点尚未由双机 Trainer 运行验证，不能补写或替代
r5 既有单次指标。

### 72.79 2026-10-09: MoE metric audit、三 placement pilot 与 survivor-service restore

审计并修复了上一轮 `20261009T-metricgate-fixed` 指标中的逻辑字节缺失：通用
`iter_checkpoint_objects` 不递归 `ObjectFragment`/`OpaqueRecord` dataclass 内部，因而通用
serializer manifest 的 `logical_bytes=0` 不是本 MoE bundle 的真实逻辑 tensor bytes。MoE
ZeRO-1 transport observer 现显式统计已选 fragment tensor 与 opaque runtime tensor 的
`numel × element_size`；一个 8-float fragment 加 4-byte opaque tensor 的 focused probe 得到
36 B，而通用 dataclass manifest 仍为 0 B，runner 现要求跨主机 data logical bytes 非零。控制
metadata 没有 tensor，control logical bytes 为 0 是预期。`remote_payload_bytes` 仍是跨主机
序列化 payload（含完整快照和控制），`remote_wire_bytes` 是客户端协议计数，不是 NIC counter。
此前观察到的 `maximum_remote_edge_ack_ms` 实际覆盖客户端 PUT 到 COMMIT ACK 的整段时间，故
Trainer JSON 改名为 `maximum_remote_data_put_to_commit_ack_ms`，不得解释成纯数据传输时间或物理
资源 max-edge。`checkpoint_exposed_stall_seconds` 现从 capture 前计时至全局发布 barrier 后，逐
rank 写入；NoCheckpoint/reference 可通过显式 `CKPT_RECORD_STEP_METRICS=1` 写同 schema step wall，
默认关闭，不影响 dense 运行。

真实双机 16-rank Qwen2-MoE tiny correctness/measurement pilot 使用同一 PP4/TP1/EP2/
replica-DP2、seed 与 3 optimizer steps，step 2 seal、fresh restore 至 step 3。以下四个
run 的 source/reference/restore 两节点 status 均 exit 0；每个最终 step-3 oracle 都比较 source、
uninterrupted reference、restore 的每 rank model/optimizer/scheduler/runtime 摘要，16/16 全匹配：

- `runs/two-node-moe-zero1/20261009T-metricfix-fixed/`：`adacheck_fixed` placement；
  remote payload 787,188 B，remote data logical tensor 338,560 B，client wire 808,775 B，
  max data PUT-to-COMMIT-ACK 11.883 ms，rank-0 exposed stall 0.6704 s。
- `runs/two-node-moe-zero1/20261009T-metricfix-host/`：`host_global_fixed`；payload
  787,195 B，logical 338,560 B，wire 808,790 B，max interval 8.578 ms，rank-0 stall
  0.6569 s。
- `runs/two-node-moe-zero1/20261009T-metricfix-m1/`：`m1`；payload 787,176 B，logical
  338,560 B，wire 808,769 B，max interval 9.550 ms，rank-0 stall 0.4308 s。
- `runs/two-node-moe-zero1/20261009T-survm1-r2/`：M1 survivor-service failure repeat；停止
  node8 replica service 后仅使用 node17 survivor endpoint，fresh 16-rank job restore epoch 2
  并继续到 step 3，final oracle 16/16 一致；payload 787,178 B，logical 338,560 B，wire
  808,764 B，max interval 24.948 ms，rank-0 stall 0.5155 s。

以上每臂只有单次 tiny-model、单 checkpoint sample、固定顺序，且 max interval 是客户端 PUT+
commit ACK，不是 dense 正式 `max_time` 的纯物理边传输口径；不能据 stall 单次差异宣称 M1 性能
收益。NoCheckpoint 每 rank 3-step wall 文件存在（48 samples），但窗口太短且不构成密集工作负载
开销统计。frequency 为 1 committed checkpoint / 3 optimizer steps。三 placement 在本 2-host
全量对象合同下 payload/logical/wire 基本相同是预期边界结果，不存在 transfer-size 改善证据。

故障实验首轮 `20261009T-metricfix-m1-survivor/` **失败**：source/reference 通过，停 node8
service 后 restore job 在 Trainer eager 创建两端 client 时仍连接故障 endpoint，rank 报
`ConnectionRefused`，runner 未写 COMPLETE。修复为 restore 初始化只连接
`CKPT_OBJECT_SURVIVING_HOST` 指定端点，且 survivor host index 设为存活的 node17=0；保留首轮
日志与失败状态，随后 `20261009T-survm1-r2/` 才通过。此证据仅代表 checkpoint service process
failure，不代表杀训练 rank、整机故障或物理主机永久丢失。

验证：`python -m pytest test/core/moe/test_zero1_optimizer.py` **2 passed, 1 warning**；
`test/core/moe/test_zero1_trainer_glue.py test/core/test_merak_engine.py` **10 passed, 1
warning**；warning 均为环境 pynvml 弃用。系统 pytest（Python 3.12）曾因缺 numpy 在 collection
失败，随后使用 `/opt/conda/envs/sjw_ckpt_grouping/bin/python -m pytest` 正确环境运行。逻辑字节
初版曾因从错误模块访问 iterator 导致 distributed upload test 失败，已修正并通过。shell syntax、
py_compile、diff whitespace 检查通过；所有 run 后 tmux/torchrun 已退出，本机 GPU 空闲。

当前结论是：ZeRO-1/EP fresh same-layout continuation 与单 survivor-service restore 在这份 tiny
Qwen 配置下可复现；尚未达到 dense 四策略同构实验。`adacheck_fixed` 仍不是 AdaCheck detector/
encoder baseline；FullState、语义 detector/encoder、200-step matched trials、五指标正式分布、
DeepSeek-V2-Lite 和训练-rank SIGKILL 均未完成。保持 `docs/09-24.md` dense 结果不变，不把 pilot
数据写入正式结论。

### 72.80 2026-10-09: DeepSeek fresh-job restore 的非持久 RoPE cache

继续排查 DeepSeek-V2-Lite tiny EP/ZeRO-1 restore。r2、r3、r4 的 source/reference
均成功，但 restore 均失败；r4 日志明确显示相同语义的 `rotary_emb.cos_cached`
capture shape 为 `(4, 4)`，fresh target request 为 `(16, 4)`。根因是训练态 RoPE
cache 按实际序列长度变化，而 fresh model 在首次 forward 前仍保留初始化长度。该
cache 是 PyTorch `persistent=False` 派生 buffer：既不在 `state_dict()`，也不在最终
model checksum 中；旧 UCP adapter 却枚举全部 `named_buffers()`，因此要求模型持久化状态
合同之外的 cache 精确匹配。这不是 MoE expert ownership 或 ZeRO optimizer shard 问题。
此前把 TP1 参数地址修正推断为根因并不正确；r4 反证了该假设，现保留该修正但不再将其当作
本故障解决证据。

`zero1_adapter.py` 新增 `persistent_model_buffers()`，按各子模块的
`_non_persistent_buffers_set` 过滤 cache；capture、target request、apply 和 live restore
完整性预检统一使用这一集合。persistent buffers（如 `inv_freq`）仍进入 UCP 对象并严格
恢复。新增回归测试模拟 source cache `(4,4)` 与 fresh cache `(16,4)`，确认两者不作为恢复
对象，而 persistent buffer 仍匹配并应用。focused pytest：
`test_zero1_object_adapter.py test_object_recovery.py test_zero1_optimizer.py
test_zero1_trainer_glue.py` **39 passed, 1 environment pynvml deprecation warning**。

修复后双机 16-rank DeepSeek-V2-Lite tiny correctness run 为
`runs/two-node-moe-zero1/20261009-dsv2ep-r5/`。独立 tmux orchestrator 及内部 phase
均正常结束。布局 PP4/TP1/EP2/replica-DP2、ZeRO-1、三 optimizer steps、step 2 seal；source、
uninterrupted reference、fresh restore 节点 17/8 status 全为 exit 0。所有 16 ranks 均从
epoch 2 恢复并继续到 step 3。`recovery-oracle.json` 验证 source/reference 与
reference/restore 的 model、optimizer、lr_scheduler、runtime 最终 digest 均 **16/16
bitwise match**。该证据证明此 tiny 配置的同布局恢复 continuation，不是规模化 DeepSeek
性能结果或五指标对比。日志仍有 UCX `/tmp` inotify “No space left on device”告警，但本轮
作业 exit 0，需作为环境告警继续跟踪。

r5 source 单次 publication 的 `remote_bytes=700540` 与 `max_remote_put_ms=21.854` 仅留在
原始 run log；它不是 dense `ckpt transfer size`/`max edge time` 的同口径稳定指标，不追加
到 `docs/09-24.md`。本轮未修改该文件。下一步对 Qwen 运行相同 persistent-buffer regression，
随后再推进恢复故障注入和同合同 grouping-policy 设计；`adacheck_fixed` 当前仍只是 placement
arm，不能称作 AdaCheck detector/encoder baseline。

随后使用相同代码/布局运行 Qwen2-MoE tiny 回归：
`runs/two-node-moe-zero1/20261009-qwen-persist-r1/`。source、reference、restore 两节点
status 均 exit 0；step 2 checkpoint 后 fresh 16-rank job 恢复并继续至 step 3；
`recovery-oracle.json` 的 model、optimizer、lr_scheduler、runtime 在 source/reference
及 reference/restore 比较中均 **16/16 bitwise match**。这验证通用 persistent-buffer
过滤未破坏既有 Qwen UCP continuation。DeepSeek 与 Qwen 本轮均为 tiny 模型、单次短 smoke，
仅是 correctness 证据，不是五指标统计或 AdaCheck essay baseline 对比。两个 run 的
orchestrator status 均 exit 0，未留本轮实验 worker/session。

### 72.81 2026-10-09: MoE 副本放置的 AdaCheck 证据边界

本课题的 placement 问题严格限定为：AdaCheck 已识别、并经强校验确认的等价
checkpoint 值给定后，选择副本发布者和目标故障域，最大化复用这些既有副本，最小化
建立同一恢复合同所需的残余传输。placement 不负责再发现等价性，也不得通过额外复制
制造冗余；detector/reducer 的压缩收益单独报告，不能计为 grouping 收益。

核对当前 EP+role-aware ZeRO-1 主路径后，MoE 对比尚未接入上述实验合同：

- `examples/checkpoint_grouping/run_workload.py` 对 role-aware EP ZeRO-1 显式拒绝
  `ada_checkpoint=true`。旧 `RedundancyChecker` 的 ZeRO-1 输入读取
  `params_in_partition/first_offset/partition_size`，而当前 optimizer 保存的是按
  shared/expert process group 分区的 local chunks；直接移除 guard 会调用不匹配的状态路径。
- detector 默认只在 Merak data-parallel process group 比较。EP ZeRO-1 有两种不同等价
  域：shared/router 的 EP×replica-DP group，以及 expert 的同 EP coordinate 下 replica-DP
  group。world-scope 全局 hash 不能替代这些语义约束，因为相同内容不代表对象可互换。
- `tensor_redundancy_evidence()` 能导出常规 model/optimizer path 的 rank/index 候选；ZeRO-1
  optimizer ranges 当前明确不导出为可寻址证据。MoE adapter 对 role-aware optimizer chunk
  使用 rank/group/chunk 局部地址，因此目前既没有跨-rank moment canonical slice 身份，也没有
  将 AdaCheck 候选接到 UCP publisher selection 的步骤。
- UCP smoke 中 `adacheck_fixed` 仅表示固定跨主机 peer placement。训练启动器同时要求
  `ada_checkpoint=false`，且 `zero1-object-metrics.jsonl` 不含 detector catalog/evidence 或
  active encoder 字段。检查现存 13 条 MoE epoch-2 UCP 指标记录，全部缺少这些字段；故这些
  run 只证明完整快照 placement/recovery，不是 AdaCheck semantic baseline，也不能用于回答
  detector 后 placement 的传输收益。历史 run 未留 detector capture，不能事后补算。
- `tools/preflight_moe_ep_zero1.py` 在 PP4/TP1/EP2/replica-DP2/ZeRO-1 下对 Qwen2-MoE 与
  DeepSeek-V2-Lite 的 expert ownership 检查通过，但整体输出 `ready=false`、
  `checkpoint_ready=false`。使用实际环境
  `/opt/conda/envs/sjw_ckpt_grouping/bin/python` 执行的 focused tests：
  `test_zero1_object_adapter.py test_zero1_optimizer.py test_reduced_checkpoint.py
  test_redundancy_checker.py` 为 **34 passed, 1 environment pynvml deprecation warning**。
  这些测试不覆盖 EP detector-to-placement 集成，不能将 CPU preflight 记作该 gate 通过。

正式 MoE placement 对比恢复前，独立 MoE semantic adapter 至少必须：以 shared/expert
实际 process-group membership 作为候选比较域；将 optimizer/master/moment 的 chunk ranges
映射回参数及准确 offset；把 expert ID、PP/TP slice 和 role 纳入 canonical identity；使用
AdaCheck 候选后再对 epoch、dtype、shape、slice 和强 checksum 做二次确认；给 fixed 与 proposed
policy 导出同一不可变 catalog、同一 active encoder/profile 和完整恢复要求；然后在 CPU fault
closure 与 held-out opportunity gate 上显示至少两个成本不同的合法 placement。任何一项失败，
不得启动长时双机矩阵或把传输指标写入正式表。该模块不得改变 dense SemanticFixed 路径。

`docs/09-24.md` 本轮保持未修改：现有 MoE UCP payload/logical/wire/PUT-ACK 数据与 dense 表
的应用层传输、completed remote edge、持续频率及训练开销口径不匹配；两种 MoE 也仅有 tiny
correctness smoke，缺少 AdaCheck detector/reducer/encoder 的 matched arms。当前可报告的结论
仍是 Qwen 与 DeepSeek tiny PP4/TP1/EP2/replica-DP2/ZeRO-1 同布局 fresh restore 后续训，最终
16-rank model/optimizer/scheduler/runtime digest 均 16/16 bitwise match；不能报告优于 baseline
或五指标完整结果。没有为追求正向数值改动 dense 文件、旧实验结果或研究表格。

### 72.82 2026-10-10: MoE metadata adapter gate 与单机 CUDA UCP 回归

按 Ponytail 保持修改范围最小：新增的 `Merak/core/moe/semantic_adapter.py` 继续只是
AdaCheck detector proposal 的 MoE 拓扑过滤器，不生成 detector hash、强 checksum 或 UCP
placement。修正 rank dataclass 坐标校验，并拒绝重复 rank 覆盖、畸形候选、未知 rank 和
缺失 expert-ID 映射；新增负向单测。它按 detector class、role、FQN、PP/TP、expert ID 与
完整 tensor slice 分组，shared 值允许跨 EP/replica-DP，expert 值只允许同 expert ID 的
replica-DP 候选。没有变更 `RedundancyChecker`、UCP 对象地址、optimizer canonicalization
或 dense SemanticFixed 路径。

验证：指定 conda Python 运行
`pytest -q test/core/moe/test_semantic_adapter.py test/core/checkpoint/test_reduced_checkpoint.py test/core/checkpoint/test_opportunity.py test/core/checkpoint/test_zero1_object_adapter.py test/core/moe/test_zero1_optimizer.py`
结果 **46 passed, 1 pynvml deprecation warning**。随后本机 8×A40 跑
`test/core/moe/test_production_topology_groups.py::test_cuda_qwen_tiny_pp4_ep2_dp1_role_aware_zero1_ucp_restore`
结果 **1 passed, 1 warning, 16.15 s**；该测试为 tiny Qwen PP4/EP2/replica-DP1/TP1
单机闭环，不是双机 16-rank benchmark。pytest 原始日志保存在
`runs/moe-semantic-gate/20261010-qwen-cuda-regression/pytest.log`。两项验证结束后没有留
pytest 或分布式 worker。

当前 MoE placement blocker 已进一步确认：`object_semantic_plan._canonical_objects()`
只按 FQN/PP/TP/DP 合并 detector class，schema 中没有 expert ID；role-aware ZeRO-1 的
optimizer/master/moment 仍按 rank/group/chunk 建立 UCP 地址，且没有反映 AdaCheck equivalence
class 的恢复 requirements。只接入 metadata adapter 或移除 `run_workload.py` 的 legacy
AdaCheck guard 都不安全。故本轮不启动双机 grouping 对比，不把 rank-local complete snapshot
结果转写为 detector-aware policy 结果。

结论状态不变：DeepSeek-V2-Lite 与 Qwen2-MoE 的双机 tiny fresh recovery smoke 可作为
correctness 证据；Qwen 另有上述单机 CUDA UCP closure。尚无 MoE 的四策略 matched experiment、
AdaCheck detector-to-UCP optimizer/state bridge，或与 dense 定义对齐的五项指标，因此不能得出
“MoE 指标优于 baseline、方向与 dense 一致”的结论。`docs/09-24.md` 未修改。

### 72.83 2026-10-10: MoE semantic evidence 到 reducer 的 CPU 闭环

按研究目标保持 policy identity 与物理 topology 解耦：MoE adapter 以 PP、TP slice、role、FQN、
expert ID 和精确 tensor schema 限制 detector 等价候选；rank-to-host 是单独显式输入，不参与
语义 identity。修复此前 adapter 只返回候选 groups、没有提供安全 reducer evidence 的缺口：
`MoEDetectorGroups.evidence` 现在把每个确认 component 限定到过滤后同语义组，再交给既有
`build_reduced_checkpoint`。并将 checkpoint path 查找改为支持 `Mapping`/sequence 和 Tensor
类型检查；拒绝不完整/多余的 replica-class 与 expert-ID metadata。

新增/扩展的 16-rank CPU fixture 使用 PP4/TP1/EP2/replica-DP2：合法对象共 12 个语义组，含
4 个 shared 四 rank 组及 8 个同 expert 双 rank 组。经过滤的 evidence 重建完整 rank state，
model、rank-local optimizer 和 runtime tensors 均逐项相等；两 host 中任一 host 失效时，独立
replication cost plan 的每个对象仍可恢复。过程中两次 pytest 失败均如实排查：第一次旧断言误把
shared 组也按双 rank 预期；第二次把 rank//8 错当 host identity。修正断言后最终 focused suite
为 **47 passed, 1 pynvml deprecation warning**，命令为：
`/opt/conda/envs/sjw_ckpt_grouping/bin/python -m pytest -q test/core/moe/test_semantic_adapter.py test/core/checkpoint/test_reduced_checkpoint.py test/core/checkpoint/test_opportunity.py test/core/checkpoint/test_zero1_object_adapter.py test/core/moe/test_zero1_optimizer.py`。
`git diff --check` 通过；本轮没有修改 `docs/09-24.md`，未启动 GPU 或双机实验。

证据边界：fixture 的 detector records 是合成输入，只验证过滤、精确确认、reducer 与恢复规划
接口，不是 Merak `RedundancyChecker` 的真实 MoE capture、placement publication 或性能测量；
也不证明 policy 优于 AdaCheck baseline。下一步必须让 MoE Trainer/checkpoint capture 导出真实
AdaCheck object evidence（包括 role-aware ZeRO-1 参数/master/moment chunk 到完整参数 slice 的
映射），然后以同一 immutable catalog 和完整恢复要求比较固定与拓扑感知 policy。Dense
SemanticFixed 继续保持不变；接入真实 producer 及 matched gate 通过前不启动长时双机矩阵，不填
五指标表，不声称 MoE 指标改善。

### 72.84 2026-10-10: 投稿证据链复核与 adapter 归因更正

按 conference evaluation 的可归因性复核 72.83：其中 16-rank CPU fixture 的 detector
evidence 是测试按预设语义手工构造的，不是 `RedundancyChecker` 输出。因此它只能证明通用
`confirm_detector_evidence`、reducer 和故障域 cost helper 在该合成输入下工作，不能记作 MoE
detector 集成或实验结果。72.83 的措辞“MoE semantic evidence 到 reducer 的 CPU 闭环”及其中组数、
恢复结论应按此限制解释；当时 focused suite 的 47 passed 仍是单测结果，不是系统 gate。

本轮进一步发现并修复 adapter 的两个 identity 风险：detector FQN 与 pipeline 本地 checkpoint
path 不总相同，现支持显式 `state_path_by_detector_path`，且 checksum confirmation 在映射后的
本地 tensor path 上完成；不同 detector equivalence components 不会因 FQN/shape 相同而合并，
component 只界定 proposal membership，不进入持久 semantic object identity。拒绝 detector path
重复、多个 detector path alias 同一 state tensor、缺失 expert ID 和未知 rank。删除了仅被测试
调用、且 grouping 契约与生产 filter 不一致的 `build_moe_candidates` helper，避免双份逻辑。

验证：
`/opt/conda/envs/sjw_ckpt_grouping/bin/python -m pytest -q test/core/moe/test_semantic_adapter.py test/core/checkpoint/test_reduced_checkpoint.py test/core/checkpoint/test_zero1_object_adapter.py test/core/moe/test_zero1_optimizer.py`
最终 **37 passed, 1 pynvml deprecation warning**。中间一轮由于测试预期错误文案导致 **35 passed,
1 failed**；改正匹配文本后重跑全套通过。`git diff --check` 通过。没有运行 GPU capture、真实
MoE detector、双机实验或指标测量，`docs/09-24.md` 未修改。

真实接入仍有结构性 blocker：Merak pipeline module 的每 rank `state_dict` 是 stage-local；
`RedundancyChecker.tensor_redundancy_evidence()` 导出的 path 基于检测 state 的 FQN，并不自动映射
到 UCP slice address；role-aware ZeRO-1 的现有恢复地址按 rank/group/chunk，而 detector 不导出
master/exp_avg/exp_avg_sq 与模型参数区间之间的完整映射。下一项实现应是独立 MoE capture adapter：
从每 rank module 参数名/EP marker/PP-TP 坐标生成经覆盖校验的 detector-path→canonical-object
映射，并基于 `iter_zero1_shards()` 生成 chunk→parameter-offset manifest；对 shared 与 expert
optimizer 使用各自真实 ZeRO process-group holder set。完成该 manifest 后先做真实单机 MoE
detector capture + CPU fault-closure，再冻结 catalog 比较 matched policies；在此之前 MoE 五指标
和对 AdaCheck baseline 的优越性仍为未测，不得填表。

### 72.85 2026-10-10: 双机真实 detector audit 接入与两模型恢复 gate

将 `RedundancyChecker(zero_stage=0)` 的真实 model-parameter proposal capture 接入可选的
`CKPT_MOE_ADACHECK_AUDIT=1` UCP source publish 路径。audit 通过 PP/EP/replica-DP/TP 坐标、
shared/expert 标记、专家 ID、精确 state path、强 checksum 二次确认；摘要包含正向 proposal
coverage 声明和 evidence SHA-256，并随 sealed control/locator 保存。它被明确标记
`placement_applied=false`，当前 `adacheck_fixed` 仍按既有 UCP 规则放置完整 rank-local
ZeRO-1 model/optimizer/runtime state；没有把 detector 证据消费为新 placement，也没有改变
dense SemanticFixed。locator reader 兼容缺少可选 audit 字段的旧 schema。

真实两节点 GPU gate 使用 tmux，在 node17 (172.20.17.2) 与 node8 (172.20.8.2) 各 8 GPU，
配置 PP4/TP1/DP4/EP2/replica-DP2、role-aware ZeRO-1；tiny Qwen2-MoE 与 tiny DeepSeek-V2-Lite
各执行 source 3 steps（step 2 发布）、独立 uninterrupted reference、从 epoch 2 新进程恢复后
继续到 step 3。完整恢复 oracle 比较全部 16 ranks 的 model、optimizer、lr_scheduler、runtime
checksum：两个模型均 `bitwise_equal=true`，source/reference/restore 均为 16 ranks，step 3
iteration/optimizer_iteration 均正确。有效日志和 oracle：
`runs/two-node-moe-zero1/20261010-qwenaudit-r2/`、
`runs/two-node-moe-zero1/20261010-dsv2audit-r1/`；两者均有 COMPLETE。审计摘要：Qwen 332
input records、978/978 candidate pairs confirmed、59 shared + 48 expert groups、0 rejected；
DeepSeek 284 records、1002/1002 confirmed、47 shared + 48 expert groups、0 rejected、1 unmatched
location。unmatched singleton/proposal coverage 受限项按 report 保留，不能据此声称 detector
召回率完整。两个成功 run 的 UCP arm 都是 `adacheck_fixed`，审计没有驱动 placement。

失败记录：首次 Qwen run `20261010-qwenaudit-r1` 的 source/reference 成功，但 restore 失败，
rank traceback 为 `ZeRO-1 restore epoch differs from requested epoch`。根因为本轮曾将 audit
塞入 rank-local `progress`，违反恢复 schema `{iteration, optimizer}`。移除该注入后加入回归
断言（audit capture 保持四个原 opaque roots，progress key 不变），完整 focused suite
`test_semantic_adapter.py test_zero1_object_adapter.py test_object_recovery_controller.py
test_object_recovery_service.py test_zero1_optimizer.py` 最终 **46 passed, 1 pynvml warning**；
之后 Qwen 和 DeepSeek 双机 gates 均完整通过。失败 run 日志仍保留，不覆盖或伪报。

便于 gate 诊断的单次 tiny-run 数值（不是论文五指标，也不用于策略优劣结论）：Qwen remote
logical object bytes 334,400、serialized remote payload 757,820 B（含对象/封包开销）、wire
779,407 B、remote edge payload 740,043 B、最大 data PUT-through-commit-ACK 21.977 ms、capture
0.445 s、publish 0.583 s、restore 0.969 s；DeepSeek 对应为 320,608 B、701,424 B、723,012 B、
683,699 B、24.372 ms、0.355 s、0.486 s、0.827 s。`max_edge` 是 client-observed data bundle
PUT-through-commit-ACK，不是纯网络传输时间；checkpoint exposure 包含 audit/capture/publish
流程。本 workload 仅 tiny 4-layer、3-step correctness fixture，不能与 dense 完整实验或
AdaCheck baseline 横比。虽然脚本采集了 step 样本，本轮不据此报告训练开销/性能改善。

双机进程和本轮 tmux service/training sessions 已退出，GPU 空闲。`git diff --check`、shell
syntax 与 touched Python `py_compile` 通过。`docs/09-24.md` 未修改：尚无 MoE 四策略 matched
run、故障注入恢复矩阵、与 AdaCheck detector/placement 公平对比或五指标定义下可比数据；因此
当前结论仅是两 tiny MoE 在上述 16-rank 配置下可训练并精确恢复，尚不能声称 MoE grouping 优于
baseline，也不能声称优化方向与 dense 一致。下一步是解决模型语义证据到 UCP tensor slice 与
ZeRO-1 master/moment chunk 的完整映射，再设计共同 catalog、共同失败集、共同五指标口径下的
匹配策略实验；不能将本次 audit-only capture 当作该桥接已完成。

### 72.86 2026-10-12: 分组接入审计与保守回退

本轮先尝试把真实 AdaCheck semantic group 接入 role-aware ZeRO-1 model object identity，并为
Trainer 建立独立 `CKPT_MOE_ADACHECK_GROUPING` 开关。代码审查发现该直接桥接不满足当前生产契约：
PP stage-local FQN 无法构成稳定跨 stage canonical identity；跨 rank 同一 shared model object
会产生重叠模型 fragment，而 `select_zero1_publications` 只按同地址/shard/version 选择来源，
没有消费 detector component owner 集；给每 rank 不同 alias 则不能形成分组。此外 model-only
相等性不能推出 ZeRO master/Adam moments 相等。故撤销 object-ID 修改及其测试挂钩，保留现有
rank-local model/optimizer/runtime identity 与 audit-only 路径；显式设置非零
`CKPT_MOE_ADACHECK_GROUPING` 现在在 Trainer 初始化时 fail-fast，不会生成看似成功的无效实验。

完整 focused 命令：
`/opt/conda/envs/sjw_ckpt_grouping/bin/python -m pytest -q test/core/moe/test_semantic_adapter.py test/core/checkpoint/test_zero1_object_adapter.py test/core/moe/test_zero1_optimizer.py test/core/moe/test_production_topology_groups.py::test_pp4_stage_local_zero1_roles_and_ucp_restore test/core/moe/test_production_topology_groups.py::test_16rank_device_mesh_role_aware_zero1_groups`
结果 **37 passed, 1 skipped, 1 pynvml deprecation warning**；skip 是需 16 张 CUDA 卡的 NCCL 参数化分支，Gloo 分支已执行。一次无效 CLI 尝试传入 pytest 不支持的 `--backend=gloo`，没有启动测试；测试中两轮失败均为 fixture 对 semantic path/rank coordinate 的预期写错，修正后全套通过。`compileall` 与 `git diff --check` 通过。Trainer 修改前备份为
`Merak/merak_trainer.py.pre_moe_semantic_20261012.bak`，保留未删。

随后增加 `test_zero1_trainer_glue.py` 复测初始化/role-aware ZeRO-1 选择路径，合并 focused suite
最终 **43 passed, 1 skipped, 1 pynvml deprecation warning**；`compileall`、`git diff --check`
再次通过。上述失败/修正历史保留，不以最终通过覆盖中间失败。

本轮没有 GPU/双机训练或性能实验，没有恢复分组 placement，未改 `docs/09-24.md`。MoE 五指标仍无可写入结果；下一步需实现并验证 component 到 canonical model slice、多个来源 owner 选择、PP-local restore binding 的完整映射，并保持 optimizer/runtime rank-local，然后再做服务恢复 gate 与 matched policy 实验。现有 tiny Qwen/DeepSeek audit-only correctness 结论不变，不代表 grouping policy 已应用或优于 AdaCheck baseline。

### 72.87 2026-10-13: Semantic ZeRO-1 model snapshot identity 一致性

复查 dormant `_merak_semantic_zero1` adapter 后发现 model capture 会给 EP 参数附加
`@expert=...` 身份，而 `zero1_model_requests()` / `apply_zero1_model_tensors()` 在 semantic
模式使用本地 FQN，导致同一对象的 capture 地址与 restore 请求不一致，最终完整性校验拒绝
恢复。`_model_object_id()` 在 semantic snapshot 下统一使用本地 FQN；普通 role-aware 模式仍
保持 rank-local identity，buffer 仍保持 rank-local。增加回归测试覆盖 EP expert parameter 的
capture/request 地址相同、apply 可恢复，以及缺少逐参数精确 detector evidence 时拒绝 capture。

指定环境 `/opt/conda/envs/sjw_ckpt_grouping` 下 focused 回归命令：
`python -m pytest -q test/core/checkpoint/test_zero1_object_adapter.py test/core/moe/test_semantic_adapter.py test/core/moe/test_zero1_optimizer.py test/core/moe/test_zero1_trainer_glue.py test/core/moe/test_production_topology_groups.py::test_pp4_stage_local_zero1_roles_and_ucp_restore test/core/moe/test_expert_parallel.py::test_qwen_real_adacheck_moe_detector_and_zero1_manifest`
结果 **45 passed, 1 pynvml deprecation warning**（39.03 s）；另外单独重跑 model identity
regression 为 **1 passed**。覆盖包含四卡真实 Qwen AdaCheck detector/manifest gate 与 PP4
stage-local ZeRO-1 Gloo UCP restore。没有跑双机训练或性能测量：本次改动限于未接入 Trainer
生产路径的 semantic snapshot identity，双机 run 不会验证 grouped placement 或性能。

此修复不改变 72.86 的生产接入结论：MoE grouping 仍 fail-fast，真实 AdaCheck proposal
仍为 positive-only audit，未驱动 publication placement；语义 capture 要求 detector 对每个
模型参数提供精确证据，而当前 detector 不能证明完整 checkpoint coverage。optimizer/runtime
仍须 rank-local。故仍缺 detector component → 完整 canonical model slices、source owner
selection 与 PP-local restore binding 的生产映射；MoE matched policies、故障矩阵及论文五指标
实验未完成，不报告收益。`git diff --check` 通过。

### 72.88 2026-10-13: MoE ZeRO-1 模型副本与互补 optimizer shard 证据

扩展 `test/core/moe/test_zero1_optimizer.py` 的现有 4-rank Gloo 集成测试：成功 ZeRO-1 更新后，
对 shared/router 与 expert 各自真实 optimizer process group gather 模型参数并要求逐位相等；
同时 gather `zero1_shard_ranges`，要求每个参数的局部区间无缝、无重叠覆盖完整张量，且每个
rank 对每个参数恰持有一个本地区间。这直接区分“可作为模型恢复副本的参数值”和“不可当作
副本的 rank-local FP32 master/Adam moments”：后者按 ZeRO-1 互补分片，不能用副本 hash
去重。group 范围严格来自 shared EP×DP 与同 expert 的 true-DP process groups，不把 EP 中
不同 expert 误判为副本。

验证命令 `/opt/conda/envs/sjw_ckpt_grouping/bin/python -m pytest -q
test/core/moe/test_zero1_optimizer.py` 结果 **2 passed, 1 pynvml deprecation warning**；
`py_compile` 与 `git diff --check` 通过。第一次新增断言失败，因为一个 rank 的参数区间可跨
多个 communication chunks，不能假设每 rank 每参数只有一段；去掉该错误假设后，要求所有
rank 聚合区间严格连续覆盖且不重叠，focused test 与文件全测均通过。该测试只验证 role-aware
ZeRO-1 的分片/副本语义，不改变 MoE detector audit-only 与 grouping fail-fast 的生产行为，
不代表 UCP 消费 detector grouping，也不是性能或论文五指标结果。

### 72.89 2026-10-13: MoE detector→UCP 接缝复审与真实 CUDA gate

重新从当前代码追踪 proposal 到恢复目标，确认不能通过只改 `ObjectAddress` 来开启 detector
grouping：`select_zero1_publications()` 只折叠完全相同的 `(address, shard, version)`；
`publish_distributed_zero1_epoch()` 对每个 source rank 上传其被选 fragment，要求每个 surviving
host 的 source catalog 完整；restore 再按 rank-local `TargetLayout` 检查 tensor/request 完整性。
MoE detector 的 semantic candidate identity 不等于现有 UCP 地址。安全跨 rank 共享模型片段至少
还需要 sealed detector-component→canonical-slice 映射、每个 slice 的真实 source/host holder
集合、每个目标 rank 的 alias/request binding，以及恢复覆盖证明。没有这些字段，删掉 rank-local
model fragment 会让 fresh target 缺对象；仅把每个 rank 的 fragment 改名则会造成 catalog 重叠。
ZeRO master/Adam moments 和 RNG/sampler/progress 仍为 rank-local，不能因模型副本而合并。

代码另一个论文测量点：当前 `capture_moe_adacheck_groups()` 复制每个 rank 完整参数 state 到
CPU 并 all-gather，再做精确确认；real detector test 也分别收集全量 state/evidence。该 audit 会
产生全量 checkpoint 级 CPU/Gloo 流量和额外驻留内存，必须计入 checkpoint exposure/peak host
memory；目前没有 audit-off/on 的匹配 overhead 结果，不能称为免费 placement metadata。

验证：在 `/opt/conda/envs/sjw_ckpt_grouping` 下运行
`python -m pytest -q -s test/core/moe/test_expert_parallel.py::test_qwen_real_adacheck_moe_detector_and_zero1_manifest --durations=3`，
4-GPU Gloo/NCCL 子进程内的 4-rank tiny Qwen real detector + ZeRO manifest gate **1 passed**；
pytest call 7.85 s、总 12.18 s，1 pynvml deprecation warning。它只证明小模型适配器正确性，测试
没有单独计时 detector、没有做 audit-off/on 对照，也没跑 service publish/restore，不能将 7.85 s
归为 detector latency 或端到端成本。双节点 GPU 本轮检查时空闲，但未重复此前已通过且
placement 仍 audit-only 的双机 restore gate；也未报告 MoE grouping 收益。

未把 semantic-zero1 dormant adapter 标记接入 Trainer：当前生产 UCP 路径保持 role-aware、
rank-local snapshot，`CKPT_MOE_ADACHECK_GROUPING!=0` 仍明确 fail-fast。下一项真正实现工作应在
distributed Zero1 control/locator seal alias/holder map、Trainer capture 产生 canonical model
slice、restore controller 据 sealed alias 构造 per-rank target requests，并做 survivor-host 故障闭包；
之后才值得运行两节点 matched policy 与五指标实验。`git diff --check` 本轮最终结果待检查。

### 72.90 2026-10-13: 跨 stage-local FQN detector canonicalization 与对象层 alias 恢复

修复 MoE semantic grouping 的根因：原 `MoECandidate.semantic_key` 把本地 FQN 纳入跨 rank key，
导致同一个 AdaCheck component 中 stage-local 命名不同、但 role/shape/slice/dtype/PP/TP 与
expert identity 一致且精确值 checksum 相同的 shared parameter 被拆成 singleton。现在 canonical
equivalence key 不含本地 name；candidate 仍保留 rank-local FQN供 target binding。加入 same-rank
多参数落入同一 detector component/canonical key 时 fail-closed，避免 hash bucket collision 把
两个本地参数错误合并。不同 expert ID仍形成独立 singleton，不因相同数值被 alias。

补上 object recovery CPU core 的显式 per-request source alias：`SliceRequest.source_address`
只允许 semantic ID 不同，role/global shape/dtype 必须完全匹配；`SliceRequest.key`（因此 target
layout 与 sealed plan hash）包含 source address。fragment coverage、source selection 与 payload
验证按 source address 执行，恢复输出仍用 target rank-local semantic ID。CPU 集成 gate 直接消费
detector-confirmed stage-local shared group，给 canonical source 在 host0/host1各存一份；
`adacheck_object_fixed`/`m1_object` 与失败集 `{0}`、`{1}` 均恢复两个不同本地 target key并逐位等于
reference。source/target shape不匹配被拒绝。该机制只在 object recovery core + 手工构造的
CPU target layout 闭环，不是 live UCP Trainer alias placement。

验证：指定环境运行
`python -m pytest -q test/core/moe/test_semantic_adapter.py test/core/checkpoint/test_object_recovery.py test/core/checkpoint/test_zero1_object_adapter.py test/core/moe/test_zero1_optimizer.py`
结果 **54 passed, 1 pynvml deprecation warning**；随后真实 4-GPU tiny Qwen AdaCheck detector
+ role-aware ZeRO manifest gate **1 passed, 1 warning**（11.92 s）。`py_compile`通过。测试过程
保留了两项失败历史：先移除 name 后 hash-bucket collision 测试暴露同 rank歧义，增加重复 rank
拒绝；首次 CPU recovery fixture 暴露 planner 没有 source alias，加入 sealed `source_address` 后
两种 policy/两个单 host failure均通过。expert 测试初始错误要求抛错，依据真实语义改成断言两
个独立 singleton group。

限制：`capture_moe_adacheck_groups()`仍有全量 CPU state gather/audit成本，尚无 audit-off/on
配对测量；distributed `zero1_service` catalog/publication 与 locator、`zero1_target_layout()`、
live controller和 Trainer 尚未传递 `source_address`，optimizer ZeRO shards 与 opaque roots仍
保持 rank-local。因此当前真实双机 MoE UCP仍不消费 detector group；没有运行新的双机训练，也
没有 placement gain/性能或五指标 claim。下一步需把 canonical source + 每目标 rank alias mapping
连同 holder/failure closure纳入 sealed service control，并做 production-path恢复故障测试；这比仅
开启环境变量重要。`git diff --check`本轮结果待收尾检查。

### 72.91 2026-10-13: live Zero1 alias gate 暂停，维持 audit-only

按请求在 `/opt/conda/envs/sjw_ckpt_grouping` 检查并尝试修复 live MoE Zero1 alias 链路。adapter
现有 key 是 rank-qualified model FQN（expert 另含 `@expert=<id>`），而
`MoEDetectorGroups.model_aliases()` 的 value 是跨 rank `moe/model/<sha256>` canonical ID；不能把
canonical ID 当参数名，也不能只改请求/fragment 一侧。source 与 target identity、fragment version、
detector checksum、sealed per-rank mapping、service catalog holder coverage 必须同一协议闭合。
本轮四 rank `test_role_aware_zero1_matches_adamw_with_clipping_overflow_and_empty_routes` 多次失败，
最终仍在 target rank 0 model source selection 报 `shared@rank=0::model` 无 source；因此没有
production alias/recovery 闭环，不得启用 `CKPT_MOE_ADACHECK_GROUPING=1`。不继续尝试靠修改测试
canonical 字符串掩盖此差距。审计开关与 grouping fail-fast 保持不变；用户确认日志无未解释错误前不跑
新实验。

本轮指定环境 CPU 回归：
`python -m pytest -q test/core/checkpoint/test_zero1_object_adapter.py test/core/moe/test_semantic_adapter.py`
结果 **38 passed, 1 pynvml deprecation warning**。四 rank live service/restore gate **失败**，故本轮
整体未通过。`py_compile` 曾通过；后续局部 adapter 改动后需再验证。没有运行 node8/node17 或 GPU训练，
也没有更改 grouping 开关。后续应先定下 alias schema（FQN→canonical object ID、detector checksum、
holder ranks/hosts、每 rank target binding），使 capture、catalog selection、两 host seal、restore
controller 与 apply 使用同一 schema；再补 poison/missing holder 和任一 host 故障闭包测试。性能实验
需等 CPU production service gate 通过后开展。

### 72.92 2026-10-13: MoE Zero1 alias service→live apply CPU 闭环修复

继续复现 72.91 gate 后，首个缺 source 错误定位为测试 closure layout 没带 capture 时使用的 rank-local
alias row；统一从 `filter_moe_detector_evidence(...).model_aliases()` 生成并复用 alias rows 后，暴露
service 对 canonical 模型地址校验时遗漏 `::model` role suffix，导致 descriptor 中找不到 canonical
fragment。修正为按 canonical `ObjectAddress.semantic_id`、model role 及 alias owner ranks 精确匹配；
仍检查跨 host owner、shape/dtype、checksum/version，防止 alias 漂移或异值被合并。

service survivor restore 返回 target request 的 rank-local semantic IDs，这是 object recovery 的既定
契约。`apply_live_zero1_state()` 与 `apply_zero1_model_tensors()` 原先却期待 canonical source IDs，
导致服务恢复完整但 live apply 拒绝。现统一由 source alias 只负责 planner 选数，restore/apply 使用
rank-local target IDs；canonical→target 映射仍由 sealed `SliceRequest.source_address` 携带。四 rank
测试改为用真实 MoE adapter 的 detector proposal graph + checksum confirmation 生成 alias，再经过
双 TCP service publication、locator 校验、survivor restore、fresh optimizer controller 和下一步训练；
原有非 alias closure oracle 使用独立 rank-local capture，避免混用两种 target layout。

验证：指定环境运行
`/opt/conda/envs/sjw_ckpt_grouping/bin/python -m pytest -q test/core/checkpoint/test_zero1_object_adapter.py test/core/moe/test_semantic_adapter.py test/core/moe/test_zero1_optimizer.py`
结果 **40 passed, 1 pynvml deprecation warning**；四 rank live/service gate 单独 **1 passed**；
`git diff --check` 通过。过程中保留并逐项解释了 alias 缺失、service suffix 校验、旧测试 assertion
容器、live apply key mismatch、测试向 controller 传 unsupported 参数、legacy closure 混用 alias catalog
以及错误位置的 collective 等失败；最终 suite 全量重跑通过。

范围限制：本 gate 为 CPU/Gloo + localhost TCP service，alias proposal 是 adapter 输入形状的 AdaCheck
候选图，不是实际模型 `RedundancyChecker` GPU capture，也未运行 node8/node17。Trainer 中
`CKPT_MOE_ADACHECK_GROUPING=1` 仍 fail-fast；未证明 PP/TP 多维生产模型、detector proposal coverage、
optimizer manifest 一致性在真实模型、host failure 网络、matched placement 收益或论文五指标。没有启动
新训练实验，未将结果写成性能结论。下一步先增加 canonical alias 的 poisoned checksum/owner mismatch
负例与 live apply target-key 回归，然后才评估是否解除 grouping fail-fast；在该门槛之前不跑 MoE
grouping GPU 实验。

### 72.93 2026-10-13: MoE alias preflight 负例及真实 detector capture 对齐

修复四-rank gate 的 missing-owner 负例：测试曾按 semantic ID 从所有 rank 删除同名副本，且仅在单个
rank 包异常断言，导致 rank-local preflight 抛出的 `ValueError` 逃出测试；现在 manifest 保持原样，只在
一个 owner rank 删除其一条 alias fragment，四 rank 同步确认 service 在上传前拒绝，随后用正确输入继续
发布。补充 M1 与 AdaCheck fixed 的 alias model bytes 严格比较，以及 optimizer role fragments 在两个
host 计划中完整保留的断言。Trainer 指标 scope 改为准确说明 canonical model fragments 按 host planner
发布，optimizer/runtime roots 仍 rank-local。

Qwen GPU gate 改为调用生产 `capture_moe_adacheck_groups()`，并对真实 detector alias 调用
`capture_zero1_rank()`，断言每个 alias FQN 恰好绑定一个 `::model` fragment；4-rank GPU gate 中 host map
为均衡模拟，不是双机实验。没有解除 `CKPT_MOE_ADACHECK_GROUPING=1` fail-fast：Trainer 生产路径尚未由
双节点 capture→publish→locator→fresh-process survivor restore 完整覆盖，且没有 matched placement
收益数据。TP1 EP2/DP2 只是已覆盖子集；不得将 AdaCheck 论文的 MoE 可用性等同于本文 UCP grouping
恢复已获一般 MoE/TP 拓扑支持。

指定环境 `/opt/conda/envs/sjw_ckpt_grouping` 验证：
`test_zero1_object_adapter.py test_semantic_adapter.py test_zero1_trainer_glue.py` **44 passed**；四-rank
CPU/Gloo + 双 localhost TCP service alias publication、survivor restore/live apply gate **1 passed**；
`CKPT_TEST_REAL_MOE_DETECTOR=1 ...test_qwen_real_adacheck_moe_detector_and_zero1_manifest` **1 passed**；
`py_compile` 与 `git diff --check` 均通过。仅有既存 pynvml deprecation warning。node8/node17 双节点及
新的 MoE grouping 训练实验本轮未运行；上述 GPU gate 只是一节点四卡 detector/capture mapping 检查，不是
service/recovery 或性能 gate。

### 72.94 2026-10-13: detector 歧义审计字段与 grouping replica-DP guard

`filter_moe_detector_evidence()`遇到同一 AdaCheck `(rank, local_index)` 对应多个 state FQN 时会丢弃
相关 proposal，避免把 hash 去重后的同值异名参数错误绑定到一个 detector location。旧审计字段
`same_rank_ambiguous_components`实际计的是被丢弃的 `(rank,path)` proposal 数，不是 component 数；改为
`ambiguous_proposals_dropped`。碰撞测试现在只断言无 group/alias/evidence 和丢弃数，不再把 content-only
reducer 的相等值去重混作 detector alias 语义。

`CKPT_MOE_ADACHECK_GROUPING=1` 的 Trainer preflight 额外要求 `EP>1, replica DP>1, TP=1`。这是因为
仅有 EP 轴而 replica-DP=1 时，当前 MoE shared replica closure 不具备可验证的跨 EP 数据并行副本关系。
`test/core/moe/test_zero1_trainer_glue.py`覆盖可接受 EP2/DP2、TP2 拒绝与 DP1 拒绝。该约束仅保护
grouping 开关，不影响单独的 detector audit / UCP training。

模型盘点：`/data/haiqwa/zevin_nfs/andy/Auto-Parallelization/nnscaler_group1/llama-model-config/Meta-Llama-3.1-70B`
只有 config/index，无权重 shard；未找到 DeepSeek-V2 236B 权重目录。找到完整
`/data/haiqwa/zevin_nfs/wwh/model-downloads/Qwen3.5-122B-A10B`，39/39 safetensors shard，index 声明
250,173,007,840 bytes，node8 可读。它是 `Qwen3_5MoeForConditionalGeneration` 多模态架构，含 48 层
文本配置、256 experts、linear/full attention 混合及视觉塔；当前环境 Transformers 4.42.4，本仓库
`workloads.py` 仅实现 Qwen2-MoE 与 DeepSeek-V2-Lite 的模型/EP 转换，`run_workload.py` 没有 checkpoint
权重加载路径。故没有将不兼容架构硬套到旧模型类，也没有执行无意义或高风险的 122B 初始化；真实
较大 MoE load/train 尚未完成，是当前实验缺口。

指定环境 `/opt/conda/envs/sjw_ckpt_grouping`：
`test_zero1_object_adapter.py test_semantic_adapter.py test_zero1_optimizer.py test_zero1_trainer_glue.py`
**48 passed**；`test_qwen_real_adacheck_moe_detector_and_zero1_manifest`真实四卡 GPU **1 passed**；
`py_compile` 与 `git diff --check` 通过。仅有既存 pynvml deprecation warning。

复核双节点 artifact `runs/two-node-moe-zero1/20261013-dsv2ep-r5`：node17/node8 各 8 张 A40，PP4/EP2/
replica-DP2/TP1 tiny DeepSeek-V2-Lite，source UCP publication、uninterrupted reference、epoch2 survivor
restore 三阶段完成；16 ranks 在 step3 的 model、optimizer、lr_scheduler、runtime 全部 bitwise equal。
epoch2 capture 0.339s、publish 0.550s、跨主机 data logical 320,608B、data/control 合计 payload
703,215B，观测 PUT→commit ACK 44.226ms。detector 有 232 records、252 candidate/confirmed pairs，
无 reject/unmatched；52 个 ambiguous proposal 过滤，34 shared + 48 expert semantic groups，group size 为
2或4。注意该 artifact 是更早的审计字段版本（显示 52 为
`same_rank_ambiguous_components`）；placement 明确记录为 rank-local，detector audit 未被消费，因此
bitwise recovery 只证明现有 UCP ZeRO1 路径正确，不证明 detector-driven grouping 或 placement/performance
收益。无新的双节点 run；节点当前空闲。

### 72.95 2026-10-13: alias 标记自描述与双机 gate false-negative 修正

复核本轮 `20261013-dsv2ep-group-r1`：node17/node8 source、uninterrupted reference、fresh survivor
restore 三阶段均 exit 0；16 ranks step3 的 model/optimizer/lr_scheduler/runtime oracle
`bitwise_equal=true`。source audit 为 232 records、252/252 confirmed、34 shared + 48 expert groups，
52 ambiguous proposals dropped；epoch2 data logical 273,440 B、remote payload 621,582 B、capture
0.321 s、publish 0.563 s、最大 client-observed data PUT→commit ACK 14.926 ms。这个 run 只作为 tiny
DeepSeek correctness artifact：locator/metrics provenance 与当前 checkout 的 alias 标记 schema不一致，
且 `CKPT_MOE_ADACHECK_GROUPING=1` 仍被 Trainer fail-fast；不把它列为成功的 grouping-on 双机实验，
不从其中声称 alias reduction/performance 收益。

修复 zero1 sealed locator 增加 `model_aliases_applied`，分布式/集中 publisher 都从已校验 alias rows
生成该值；旧 locator 读取默认 false。Trainer source metrics 同步写该字段。双机 smoke 验收将
`detector_audit.placement_applied=false` 与 alias-applied 独立校验，不再把合法的
“canonical model fragments + rank-local optimizer/runtime” scope 报成 detector audit placement 标签错；
locator manifest、applied flag、metrics flag 不一致仍 fail。原错误 artifact 和日志未覆盖。
runner 现在在任何 node/service preflight 前拒绝 `CKPT_MOE_ADACHECK_GROUPING=1`，明确说明 live Trainer
仍 fail-fast；这避免调用方误以为脚本已把 audit-only smoke 切换成 grouping-on。负向预检返回预期 exit 2，
未启动服务或训练。

验证：`/opt/conda/envs/sjw_ckpt_grouping/bin/python -m pytest -q
test/core/checkpoint/test_zero1_object_adapter.py test/core/moe/test_semantic_adapter.py
test/core/moe/test_zero1_optimizer.py test/core/moe/test_zero1_trainer_glue.py` **49 passed**；修改 locator
后针对 alias locator round-trip 和四-rank Gloo + TCP survivor live-apply gate 重跑 **2 passed**；真实
四卡 Qwen AdaCheck detector/manifest gate **1 passed**。`bash -n`、指定环境 `py_compile`、
`git diff --check` 通过。首次新 locator 回归因测试未 import `json` 失败，加入局部 import 后重跑通过；
有既存 pynvml deprecation warning。

结论边界：当前 CPU distributed gate 覆盖 alias manifest → locator → source-address target request →
survivor restore → live apply → 后续 optimizer step；GPU gate 覆盖真实 detector capture/fragment binding。
尚未有与新 locator schema 同一 checkout provenance 的 node8/node17 grouping-on完整重跑、真实模型
大权重 load、matched placement baseline、故障矩阵或论文五指标。生产 Trainer grouping fail-fast 保持，
双机脚本不再接受把 grouping 环境变量静默当 audit-only 的结果；本轮没有启动第二次双机实验。

### 72.96 2026-10-13: DeepSeek safetensors loader 审计

对完整 DeepSeek-V2-Lite checkpoint（index 5,291 keys、31,412,968,448 bytes）做只读 key audit。
更正先前初判：`load_sharded_safetensors()` 的 `cached_files` 从未写入，所以同 shard tensor 不会漏，
但每个命中 key 都重复打开 shard。按目标模型 keys 先分组，再每个实际命中的 shard 打开一次；新增
两 shard/多 tensor/未命中 shard 回归，指定环境 pytest **1 passed**。

主要 MoE blocker 仍在 key/schema：当前通用转换将 PipelineModule `N.layer...` 猜成
`model.layer...`，而 DeepSeek HF index 使用 `model.layers.<layer_id>...`、`model.embed_tokens...`、
`lm_head...`；真实示例 key probe 全部 miss。EP2 又只保留 local expert ModuleDict，须基于全局 expert ID
做分片读取和严格完整性校验。`check_state_dict` 另有宽松后缀匹配，不能当作正确加载证据。没有运行
真实权重训练、双节点或声称 MoE pretrained load 成功。下一步需实现/验证显式 DeepSeek HF→pipeline-local
映射及 EP expert 过滤，并检查加载结果的 missing/unexpected keys、参数 checksum 与前向数值，再进入
Zero1/EP/UCP 端到端实验。

### 72.97 2026-10-15: MoE M1 准入复核与 Codex TUI 保护边界

按 `moe_plan.md` 复核工作区和双机状态：node17/node8 各 8×A40 空闲，已有 tmux
session 中没有活跃 MoE 训练；两机 TCP/SSH 可达。工作区有大量既存未提交研究改动，均保留。
当时没有启动新的 tmux 实验：PP-first 16-rank Gloo gate 和相关回归尚未全部收口，
且 launcher 有 grouping=1 遗留拒绝条件；dp-pp tiny audit run 不能代替 PP-first grouping 闭环。

实验准入证据：
- 指定环境 `python -m pytest -q test/core/moe/test_semantic_adapter.py
  test/core/moe/test_production_topology_groups.py::test_device_mesh_grid_group_report`
  **22 passed**；另一次对语义与 8-rank mesh group 的聚焦重跑 **21 passed**。
- `test/core/checkpoint/test_zero1_object_adapter.py::test_zero1_alias_manifest_is_sealed_and_rejects_rank_local_collision`
  **1 passed**。
- `test/core/moe/test_expert_parallel.py::test_qwen_real_adacheck_moe_detector_and_zero1_manifest`
  **1 passed**（4 CUDA ranks）；这是 detector/capture→fragment mapping gate，不是双机 service/recovery。
- `tools/preflight_moe_ep_zero1.py --models qwen2_moe_14b` 报 EP ownership ready，但
  `checkpoint_ready=false`；脚本只覆盖 dp-pp 并显式声明不支持 SemanticFixed ZeRO-1，不能作为完整准入。
- 旧 artifact `runs/two-node-moe-zero1/20261013-dsv2ep-group-r1/locator.json` 的 arm 是
  `m1` 且 aliases 已保存，但其 code/schema provenance 被 §72.95 明确判旧，不能算本轮通过。

代码进度更新：`run_workload.py`、EP layout 与 MoE `ProcessTopology` 已统一支持 PP-first
`rank=4*pp+2*replica_dp+ep`；新增 layout helper 回归，避免 runner 与 topology tests 的轴顺序漂移。
16-rank production Gloo gate 已通过，核验 PP-first topology axes/coordinates、pipeline 与
composite-DP groups、EP dispatch、replica sampler、shared/expert ZeRO partitions、world overflow
consensus 及 optimizer step。另区分 grid DeepSpeed model group（固定 data 坐标，8 ranks）和
MPU model group（TP slice，singleton）；8-rank CUDA tiny Qwen train/restore 仍是 replica-DP1，
不能冒充该正式 gate。

selector ledger 更正：四个 stage 的 shared objects 合计跨 host logical bytes 为 fixed 6,400 B、
Host/M1 1,600 B（每 stage 1,600 B 对 400 B）。mixed shared+expert model-only ledger 的两 host
selected bytes 为 fixed 17,920 B、Host/M1 5,760 B；跨 host logical bytes 为 8,960 B 对 2,880 B。
该 ledger 不含真实 detector aliases、optimizer/runtime objects，也不是 serialized PUT/wire bytes；
真实完整账本与 service observation 仍是 E0 blocker。

当前顺序：16-rank Gloo gate 已通过；更正 ledger 并补足 grid-vs-MPU group 断言后，先跑相关
alias/selector tests 与 checker 语义修正后，已完成 PP-first EP2/replica-DP2 tiny grouping-on
三臂 source/reference/fresh-restore gate。详见 §72.98。当前不进入性能矩阵：locator 只保留
source-rank manifest checksum evidence，不含 committed object inventory；service 在 run 完成后
退出，不能事后对账计划集合、model/optimizer/runtime role bytes 或 maximum publisher load。下一步
仅需为 smoke artifact 导出最小 committed inventory，并证明 host catalog hash 和 planned set =
committed set。Codex SQLite/TUI 未触碰。

Codex session 保护审计（同日）：当前 VS Code remote `ptyHost` (PID 1325129) 托管 shell
`pts/33`，`codex resume` Node wrapper PID 1325992 与 Codex CLI PID 1325999 正在运行；当前
Codex CLI 直接打开 `/root/zevin_nfs/bc/codex_memory/codex/logs_2.sqlite` 及 WAL/SHM。进程不在
tmux 内。不能安全地将已运行 TUI 后置迁移到 tmux，也不应在线复制/修复 SQLite；继续保持
VS Code remote 窗口与当前 session，不要关闭 TTY/终止进程。另存的仓库续接点见
`SESSION_RECOVERY.md`。本次未改动 Codex 数据库。

### 72.98 2026-10-15: PP-first 三臂 tiny grouping 闭环与计量缺口

修复 `tools/run_two_node_moe_zero1_smoke.sh` 的 sampler checker：cursor 是 pipeline input-owner
stage 的 rank-local 状态，PP0/PP3 在 3 steps 后为 6，PP1/PP2 为 0；不能按 replica-DP rank
期待全局统一值。16-rank Gloo regression 增补 replica-DP 两侧实际取不同样本的断言，3 个相关
拓扑/selector tests 通过；shell syntax、py_compile、`git diff --check` 通过。

双机 PP-first `PP4×EP2×replica-DP2×TP1`、tiny Qwen、固定 seed、epoch2 publication、step3
reference/restore 三臂均通过，source/reference/restore 每 rank 的 model、optimizer、scheduler、
runtime checksum 16/16 完全一致，detector aliases 三臂均应用：

| Run | Placement | Remote data logical | Edge payload | Total remote payload | Remote wire | Bitwise oracle |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `20261015-ppfirst-fixed-g3` | AdaCheck fixed | 334,400 B | 748,890 B | 806,320 B | 827,906 B | pass |
| `20261015-ppfirst-host-g1` | Host-global fixed | 276,608 B | 579,795 B | 637,228 B | 658,817 B | pass |
| `20261015-ppfirst-m1-g1` | M1 | 276,608 B | 579,897 B | 637,315 B | 658,901 B | pass |

Detector 每臂相同：50 shared groups、48 expert groups、348/348 candidate pairs 确认、36 ambiguous
proposal dropped、0 unmatched。fixed g3 在 fresh restore 前停止 node8 service，仅用 node17 survivor
恢复，三阶段 node17/node8 进程均 exit 0。g2 旧 run 也经离线 digest/metrics 校验通过，但因旧
checker 误报保留为无 COMPLETE 的诊断 artifact；不覆盖或伪造状态。

字节结果仅是每臂一次 tiny aggregate 观测：Host-global/M1 的 remote data logical bytes 相对
fixed 少 17.3%，total payload 少约 21.0%；Host 与 M1 payload 差 87 B，不能据此声称 M1 优于
Host。capture/publish 和 maximum PUT/edge latency 波动明显，不形成时间收益结论。

E0 尚未完全关闭：sealed locator 保存 source-rank manifest IDs/checksums，不保存 manifest 的
object lists；run-state 也未持久化 endpoints，service 在脚本结束时退出。因此无法事后验证
planned object set = committed object set、model/optimizer/runtime 分项或 maximum publisher load。
下一步仅在 smoke artifact 中归档最小 committed object inventory（沿用现有 manifest，不新增
服务层），用 host content hash 对账，然后再决定是否开始 E1 timing。随机初始化 tiny 结果不写成
完整 Qwen 14B 性能，不外推 convergence；该流程仍是同步 full snapshot，不是 dense gradient-suffix
合同、异步 checkpoint 或 AdaCheck end-to-end 系统复现。Codex TUI/SQLite 未触碰。

### 72.99 2026-10-27: MoE committed inventory 首个 survivor run

遵循 `moe_plan.md` E0，给分布式 ZeRO-1 publication 增加 opt-in committed inventory：seal 后通过现有 replica-client GET 读取两 host 的 source-rank bundles，核对 planned/committed fragment identity 与 host catalog content hash，并写入已有 publication metrics。未新增服务协议；普通训练默认不导出。Smoke checker 要求 inventory 有两 host、全部 source ranks、role/publisher breakdown，并核对远端 logical bytes 与 edge payload 对账。当前管理员只提供 node17/node8，两机16 ranks；不再安排32 ranks。

验证进展：四-rank Gloo/TCP 的 `test_role_aware_zero1_matches_adamw_with_clipping_overflow_and_empty_routes` 在 inventory 开启后通过；新 16-rank Qwen tiny fixed run `runs/two-node-moe-zero1/20261027-inv-fix` 的 source/reference/restore 三阶段成功，restore 前停 node8 service，16/16 ranks 的 model、optimizer、lr_scheduler、runtime 均与 uninterrupted reference bitwise equal；inventory 记录两 host 各312 fragments/64 runtime roots，planned shard count 与 committed set 相同，host content hash 相同。观测 remote data logical 249,152 B、edge payload 497,094 B、total remote payload 554,530 B、remote wire 576,115 B，capture 0.424 s、publish 1.218 s、max data PUT→commit ACK 43.439 ms。该 run 是 single tiny correctness/accounting sample，不是 full Qwen 性能结果或 timing claim。

修订中的 byte reconciliation checker 发现 manifest `logical_bytes` 对自定义 object bundle 为0；ledger 必须单独按 fragments 与 opaque tensor logical size 汇总，不能拿 manifest field 代替。已据此修正 inventory schema/checker，下一轮 three-arm fresh runs 将验证确切 aggregate 对账；在修正版 checker 通过前，E0 仍未关闭，不开始 E1 timing。后续目标是在两节点资源约束下完成两种 MoE 模型、16-rank fixed/Host/M1 与专利/AE artifact；32-rank 暂不覆盖。Codex TUI/SQLite 未触碰。

### 72.100 2026-10-27: DP-first negative control and explicit PP-first campaign

Rechecked actual run-state and rank topology: `20261027-inv-fix2`,
`20261027-inv-host`, and `20261027-inv-m1` used the launcher's default
DP-first (`dp,pp,sp,tp`), not the intended PP-first candidate. All three
completed 16-rank source/reference/restore with bitwise equality and committed
inventory reconciliation. Fixed and Host each transferred 249,152 logical B
(model 1,152; optimizer 166,592; runtime 81,408). M1 transferred 262,336 B
(model 14,336; optimizer/runtime unchanged). Thus current M1 adds remote
model traffic on this negative-control layout; it does not show a benefit.
Results are independently reconstructed from stored bundle entries and locator
evidence by `tools/analyze_moe_snapshot.py`; output:
`runs/two-node-moe-zero1/20261027-dp-inventory-summary.json`.

The earlier unpacking error was an exception-path tuple-length bug masking the
inventory ownership error, not an environment propagation disagreement. Both
were fixed before the successful CPU/TCP and GPU runs. Opaque records retain
storage_hosts=(0,1), so membership is the correct host check. Inventory role
field `role_logical_bytes_per_host` currently sums BOTH stored host copies;
use the analyzer's explicitly remote-only role breakdown for transfer claims.
Inventory readback is inside publication timing, making these timings diagnostic.

Started `moe-pp-inventory-20261027` tmux campaign with explicit
CKPT_MOE_MESH_ORDER=pp-dp, grouping enabled, and node8 service stopped before
every restore: Qwen and DeepSeek tiny, fixed/Host/M1, sequentially on node17/node8.
Run IDs: `20261027-pp-{qwen,ds}-{fixed,host,m1}-i1`. Success is pending actual
COMPLETE/oracle/ledger checks; this entry does not claim the campaign passed.

### 72.101 2026-10-27: Six PP-first MoE inventory and survivor gates passed

The `moe-pp-inventory-20261027` campaign finished with exit_code=0 at
2026-10-27T16:27:46Z. Both tiny models, three policies each, used explicit
PP-first PP4/TP1/EP2/replica-DP2 on node17/node8. Every source/reference/restore
phase exited 0 on both nodes. Before every restore node8's checkpoint service
was stopped; all 16 ranks' model/optimizer/scheduler/runtime checksums matched
the uninterrupted step-3 reference bitwise. Each run has COMPLETE, sealed
locator evidence, committed object inventory and matched planned identities
and host catalog hashes. Independent offline reconstruction matched logical
and serialized remote data byte counters for all six runs.

All run IDs below have prefix `20261027-pp-` and suffix `-i1`:

| Run | Remote logical B | Total remote payload B | PUT/commit wire B | Max remote publisher logical B | Max remote publisher payload B |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen-fixed | 334400 | 806318 | 827899 | 34188 | 73141 |
| qwen-host | 276608 | 637230 | 658818 | 34188 | 73125 |
| qwen-m1 | 276608 | 637313 | 658895 | 30892 | 63098 |
| ds-fixed | 320608 | 739640 | 761226 | 32464 | 65776 |
| ds-host | 267040 | 601554 | 623142 | 32464 | 65762 |
| ds-m1 | 267040 | 601613 | 623205 | 28368 | 54345 |

Qwen remote model bytes fell 86400 -> 28608; optimizer 166592 and runtime
81408 were unchanged. DeepSeek model bytes fell 80896 -> 27328; optimizer
158304 and runtime 81408 were unchanged. Thus complete-state logical traffic
fell 17.28% and 16.71%, respectively, from fixed to Host/M1. Host and M1 have
equal logical bytes on this layout; maximum publisher load is lower for M1.
Payload includes control, while publisher maxima cover data bundles only.
These are single tiny correctness/accounting runs. Inventory GET readback is
inside publication timing, so no throughput or stall improvement is claimed.

Artifacts: `runs/two-node-moe-zero1/20261027-pp-inventory-summary.json`,
campaign log/status alongside it, `tools/analyze_moe_snapshot.py`, and
`docs/moe_snapshot_ae.md`. Analyzer compiled and ran on all six primary runs
and all three DP-first negative controls; git diff --check passed.

This closes the two-model PP-first E0 committed-ledger/restore gap. Full-model
E1/E2 performance, master-to-BF16 encoding control, and E3 changed-layout
continuation remain outstanding. The DP-first M1 remote-publisher regression
is recorded in 72.100 and must be resolved before claiming byte-optimal policy.
Only node17/node8 and 16 ranks are authorized by the current resource scope.

### 72.102 2026-10-27: M1 local-first publisher correction

Changed M1 to select local holders before balancing publisher load. The
selector otherwise considered remote holders even when a local exact replica
existed, contradicting the plan's bytes-first objective. Three focused
selector tests passed, including zero remote traffic for fully local-covered
objects and preservation of PP-first opportunity totals.
`20261027-dp-m1-local-i1` then completed the actual 16-rank Qwen source,
reference and node17-survivor restore after stopping node8's service. All
state components matched bitwise; independent inventory analysis reports
249152 remote logical B, model 1152/optimizer 166592/runtime 81408, matching
Host/fixed rather than the previous M1's262336 B. Summary:
`runs/two-node-moe-zero1/20261027-dp-m1-local-summary.json`.

Started full Qwen 14.3B one-step training-capacity run
`runs/moe-capacity/20261027-full-qwen-c1` using the new minimal
`tools/run_two_node_moe_capacity.sh`, under `moe-full-qwen-cap-20261027` tmux.
PP4/TP1/EP2/DP2, BF16, seq128, GA8; no tiny config and no checkpoint
publication. Actual default partition_method=uniform_floor gives
[0,6,13,20,27], differing from the earlier ceiling-partition static estimate.
GPU sampling is saved every polling interval on both nodes. Completion and
training-memory admission remain pending the actual process exit and metrics.

### 72.103 2026-10-27: Both full MoE architectures trained on 16 A40s

Full-model capacity runs completed under tmux, both nodes exit_code=0:
`runs/moe-capacity/20261027-full-qwen-c1` and
`runs/moe-capacity/20261027-full-ds-c1`. No tiny flags or pretrained weights;
full Qwen2-MoE14.3B / DeepSeek-V2-Lite15.7B architecture, random initialization,
PP4/TP1/EP2/replica-DP2, BF16, seq128, microbatch1, GA8, one optimizer step.
Each artifact has16 rank-step records at step1 and complete workload/rank
metadata. Qwen rank-step wall range25.504--25.564s; DeepSeek28.455--28.473s.
Sampled GPU-memory maxima in MiB: Qwen node17=35734/node8=37628;
DeepSeek node17=39940/node8=41838. These are sampled nvidia-smi observations,
not exact allocator peaks; cold one-step times are not stable performance.

Both jobs finished and released their GPUs. Reconciled summary:
`runs/moe-capacity/20261027-capacity-summary.json`. Reproduce with
`bash tools/run_two_node_moe_capacity.sh <fresh-run-id> <model>` under
`tools/run_in_tmux.sh`; actual command and sampled memory are in each artifact.
This establishes real full-model training feasibility, NOT full-model
checkpoint publication/recovery. Next: admit one fixed-arm full snapshot
within measured host RAM, then run the matched policy/recovery experiments.
The estimated220/239GiB per-host checkpoint lower bounds and additional
capture/serialization buffers must be budgeted before enabling publication.

### 72.104 2026-10-27: Remove full-model memory amplification before publication gate

Code inspection found the live MoE detector all-gathered actual model tensors
to every rank. Changed capture to exchange CheckpointObject summaries and use
the existing confirm_detector_object_evidence path. The filtering API retains
its tensor-state input for callers/tests, converts it once to summaries, and
uses identical SHA-256/shape/dtype confirmation and semantic grouping. This
also removes repeated full-state rehashing inside the candidate loop.
19 semantic tests passed, including exact equality of metadata-vs-tensor
confirmation outputs on the same evidence.

Recovery previously retained every source bundle's tensors in every target
rank. It now validates each fragment, retains tensor data only for source
addresses requested by the target layout, preserves descriptors for all other
fragments, and releases each bundle/payload before fetching the next. Full
host content hash and coverage checks remain. Existing four-rank Gloo/TCP
publication/restore/live-continuation regression passed.

Added opt-in CKPT_ZERO1_SERIAL_PUBLISH=1 to bound simultaneous rank-bundle
serialization for capacity runs. All ranks advance through the same barriers;
upload failures still propagate through the existing gathered evidence check.
Default concurrency is unchanged. The distributed TCP regression also passed
with this mode enabled. Smoke launcher explicitly forwards this flag to both
nodes (outer tmux environment alone is not sufficient).

`20261027-pp-bounded-i2` ran the combined changes on16 GPUs with explicit
PP-first, grouping and serial publication. All source/reference/restore jobs
exited0; node8 service was stopped before restore; the oracle was bitwise equal
for every rank/component. COMPLETE and outer exit_code=0 at16:49:00Z.
The entire detector_audit equals `20261027-pp-qwen-fixed-i1` exactly, and
remote logical bytes remain334400. Offline inventory reconciliation passed;
summary: `runs/two-node-moe-zero1/20261027-bounded-summary.json`.

Retained failed predecessor `20261027-pp-bounded-i1`: training phases exited0
but editing the shell script while it ran caused Bash read-offset corruption
(`e: command not found`, exit127). It has no COMPLETE and is excluded. Never
edit a running launcher. Full-model checkpoint save/restore is still pending;
these tests verify the memory-reduction changes, not full-model peak memory.

### 72.105 2026-10-27: Full Qwen snapshot capacity trial and snapshot ownership

Extended the existing capacity launcher with opt-in CKPT_CAPACITY_PUBLISH=1:
fixed policy, epoch1, serial publishers, grouping enabled, inventory readback
disabled, two256GiB services/one retained epoch. It records host available RAM
and stops the owned trial below24GiB available; normal training-only mode stays
available. Full-model source configuration remains PP4/EP2/DP2 BF16 seq128 GA8.

First run `20261027-full-qwen-save-c1` reached actual service PUTs (observed
node17 two bundles/26.90GiB and node8 one bundle/13.45GiB) after training, but
completed publishers retained their CPU snapshots while waiting for global
seal. Per-rank RSS was roughly24--37GiB during publishing. The run was
intentionally stopped before completion to fix this avoidable retention; this
was NOT an observed OOM or an asserted successful checkpoint. FAILED preserves
the reason and raw logs. Stopping tmux alone left torchrun descendants, so
the exact owned run's Python processes were terminated on both nodes and GPU
memory returned to0. Cleanup now also terminates descendants matching the
unique run directory, rather than assuming tmux shell exit is sufficient.

Trainer now transfers ownership of its captured CPU snapshot into the
publisher; the publisher drops its payload references after its two commits,
before waiting for remaining ranks. It does not mutate caller tensors; existing
callers retaining their own references remain valid. Serial four-rank Gloo/TCP
publication/restore/continuation regression passed, compilation and diff check
passed. Retry `20261027-full-qwen-save-c2` is running under
`moe-full-save-qwen2-20261027`. Exact runtime sources are archived in
`source-code.tar.gz` with SHA-256, and dependency versions in python-packages.txt.
Service records/bytes and host memory are being sampled into
service-progress.jsonl; completion, seal and full-model peak budget are still
pending actual measurements. No full-model restore is claimed.

### 72.106 2026-10-27: Caller-frame retention fixed; c3 remains live

c2 was deliberately stopped after observing completed-publisher RSS remain
high. A weakref reproduction proved that passing box.pop() as an argument
keeps the popped object alive on the caller evaluation stack until return;
deleting the callee parameter cannot release it. Passing the container and
popping inside the callee allows immediate release instead. This corrects
the provisional ownership claim in72.105; c2 is FAILED, not OOM/complete.

The publisher now accepts optional captured_snapshot=[fragments, opaque],
unpacks and clears it internally, and drops its payload references after both
commits. Trainer passes the container itself. Legacy explicit sequences remain
supported without mutation. The serial Gloo/TCP recovery/continuation test
passes and checks that the consumed container is empty. An initial test change
shadowed an existing model-snapshot variable; renaming that fixture variable
fixed the test, without changing the implementation.

ACTIVE RUN: `runs/moe-capacity/20261027-full-qwen-save-c3`, outer tmux
`moe-full-save-qwen3-20261027`, outer log
`runs/moe-capacity/20261027-full-qwen-save3-campaign.log` and `.status`.
Started with CKPT_CAPACITY_PUBLISH=1; full Qwen fixed snapshot, serial publishers,
same model/topology/seq128/GA8. Runtime source tar/SHA-256 and pip freeze are
stored inside c3. At17:14:14Z, tmux pane PID2654586 was live, training step1 had
completed (22.071s), and capture/publication was still running. Available RAM
was474206600/369865280 KiB on node17/node8. No COMPLETE or global-seal result
has been claimed. Continue polling THIS run; do not launch a replacement due
to observation timeout. Do not edit its launcher or runtime code while live.

c3 verified progress at17:19:27Z: node17 service stored2 bundles totaling
28878519412 B; node8 stored1 bundle totaling14439259706 B. No global seal yet.
At17:19:58Z the first publisher process PID2654856 had RSS13563344 KiB,
down from28445504 KiB during capture/serialization. This is measured evidence
that consumed-container ownership permits payload release during serial
publication, unlike c2's caller-argument retention. Host available RAM then
was333089588/237475460 KiB. Outer tmux remains live; continue this c3 run.
Before full-model restore, address its current all-source-bundle read fanout:
target filtering bounds retained tensors but still downloads every bundle per
rank. A sealed descriptor index can select needed bundles without new service
protocol, but that change is not part of the running c3 source archive.

### 72.107 2026-10-27: Full Qwen epoch1 sealed; direct survivor restore active

c3 SOURCE PASSED: node17/node8 training groups exited0 at17:55:07/08Z.
Epoch1 locator is sealed, aliases applied, and both services have17 records
(16 data bundles plus control), each255009847862 B =237.496428 GiB.
Remote data logical bytes255002439520; remote total payload255009847862 B.
Rank0 capture64.306s and serial publication2397.930s; whole step2484.499s.
This is a single full-model fixed-policy capacity result, not a comparative
performance claim. Actual service storage exceeds the earlier220.25GiB
tensor estimate; use measured237.50GiB for subsequent capacity planning.
Raw source metrics, service-stats.json and PUBLICATION_COMPLETE are archived.

To reuse this already-uploaded snapshot, the source cleanup controller
PID2654588 was SIGSTOP'ed before it removed the services. Training/upload
continued independently to completion. Its outer tmux is
moe-full-save-qwen3-20261027; it must NOT be resumed while restore needs the
survivor service. A temporary memory watcher took over monitoring, then was
stopped once the fresh restore launcher provided its own monitoring. Source
worker exit status and PUBLICATION_COMPLETE, not the held outer supervisor,
are the evidence for publication success.

The new optional target-only reader prioritizes the target rank's bundle and
stops when all requested full tensor addresses and runtime roots are covered.
It explicitly binds returned control/data manifests to the sealed locator
evidence and verifies payload/fragment checksums. It validates complete target
coverage rather than recomputing an unrelated whole-host content catalog hash.
Legacy whole-host audit remains the default. Serial restore applies each rank
and releases its temporary state before advancing the next rank; topology
gather occurs before this serialized section. Existing four-rank distributed
restore/continuation test passed with both options enabled.

ACTIVE RESTORE: runs/moe-restore/20261027-full-qwen-r1, outer tmux
moe-full-qwen-restore-20261027, outer log/status
runs/moe-restore/20261027-full-qwen-restore-campaign.log{,.status}.
Node8 checkpoint service was explicitly stopped before launch. All16 fresh
workers use only node17's original c3 service, restore epoch1 and attempt
step2, full Qwen PP4/EP2/DP2 seq128 GA8. This restore-only code was changed
after source processes had loaded their archived source version; source
publication code and the running source shell launcher were not edited.
The reader code has its own source-code.tar.gz/SHA-256 in the restore artifact.
At18:09:46Z ranks0 and1 had each read exactly one14439259706 B bundle;
restore is still active, and success/next-step correctness is not yet claimed.
An uninterrupted full-model reference comparison is still outstanding.

Final requested deliverable: after necessary experiments, create docs/10-27.md
summarizing dense+MoE16-rank results and research value using docs/09-24.md
section9 metrics. Preserve encoding/layout distinctions and do not substitute
RTO for GEMINI recovery-wasted time. The final synthesis has not been written
early or marked complete.

## 72.108 October 28: AE source archive and full-model recovery status

The user requested code1028.zip before further experiments. Created a curated
source-only archive using tools/package_ae.py: 308 entries, 1069126 bytes,
290 Python, 7 shell and 11 Markdown files; extension/path and ZIP CRC checks
passed. AE_README.md explains prepared-environment prerequisites, external
DeepSeek/dense detector configurations, current entrypoints, and evidence
limits. AE_MANIFEST.md inside the archive binds each included file by SHA-256.
The original LICENSE is preserved verbatim under the allowed LICENSE.md name.
No logs, weights, checkpoint payloads, raw results, caches or vendor trees are
included. This is not the final results report or a self-contained installer.

Full Qwen restore r2 (runs/moe-restore/20261027-full-qwen-r2) completed:
all16 ranks fetched exactly one target bundle, total255008523112 payload bytes,
then step2 finite loss12.2768. Restore-controller time1235.389363s; max step2
wall24.19626068s. No uninterrupted step2 reference exists for that source,
so this is integrity/coverage plus continuation evidence, not bitwise final
state equivalence. Qwen source c3 services were subsequently released; outer
source exit143 was intentional supervisor cleanup after successful restore.

ACTIVE: moe-ds-fixed-chain-20261028, source20261028-ds-fixed-p1, restore
20261028-ds-fixed-p1-r. Full DeepSeek fixed source sealed epoch1 and completed
step2, preserving full state digests. Remote publication bytes267682992393,
step1 wall1695.803s; step2 loss11.9445 and wall23.727s. Publication and restore
use one active rank per host. At03:41UTC fresh restore had read bundles for
10/16 ranks, with ample host memory and no observed failure. Parent performs
the strict source/restored step2 digest comparison; do not claim it passed
until recovery-oracle.json and COMPLETE establish this. Node8 service is
stopped; node17 survivor service remains required. Do not edit running shells
or launch competing GPU jobs. Remaining full-model policy matrix and final
docs/10-27.md are still outstanding.

### 72.109 October 28: Full-policy queue and master-to-BF16 boundary

Started tmux moe-full-policy-queue-20261028 using
tools/run_moe_full_policy_queue.sh (bash syntax check passed), log/status
runs/moe-capacity/20261028-policy-queue.log{,.status}. It waits for the current
DeepSeek fixed chain's successful final status and COMPLETE before starting
DeepSeek Host/M1 then Qwen fixed/Host/M1, all with two source steps, epoch1
snapshot, fresh survivor restore, and strict step2 state-digest comparison.
It snapshots source code per run, runs sequentially, and stops on the first
failure. This closes full-policy admission only, not the E1 repeated timing,
E2 sustainability/fault, or E3 layout-change requirements. At queue launch
the prior capacity/restore processes were confirmed live; no job was restarted.

The required master-to-BF16 check now has executable evidence:
test/core/moe/test_zero1_optimizer.py::
test_bf16_model_reconstructs_exactly_from_role_sharded_fp32_masters,
1 passed in9.55s, tmux moe-master-bf16-gate-20261028, log/status
runs/moe-capacity/20261028-master-bf16-gate.log{,.status}.
Four CPU/Gloo ranks use real role-aware ZeRO-1, EP2/DP2, distinct expert IDs,
three AdamW updates with BF16 parameters. Reconstruct each shared/router/expert
parameter exclusively from its owners' FP32 master ranges cast to BF16;
assert exactly-once range coverage and bitwise equality after every update.
This confirms the need for the plan's common stronger encoder control.
It is NOT a full-model service/restore test with model tensors omitted.
The present positive grouping result remains conditional on explicitly stored
model snapshots; saved master tensors can make that redundancy unnecessary.
Do not generalize it to an optimal snapshot encoder or full AdaCheck system.

The already-delivered code1028.zip remains the earlier immutable source
snapshot; this queue/test and later research edits are not retroactively
claimed to be in that ZIP. Repackage explicitly at final handoff if needed.

### 72.110 October 28: Full-model analysis path

Extended the existing tools/analyze_moe_snapshot.py rather than creating a
second reporting framework. Full two-step chains are admitted only when both
source and restore are COMPLETE without FAILED, the16-rank step2 oracle
passes, the raw per-rank state/component digests agree, and epoch/alias/step
coverage matches the chain contract. Output includes remote logical/payload/
wire bytes, PUT and publication timing, maximum per-rank step wall and exposed
stall, and the restore-controller event. Controller timing explicitly excludes
startup and is not mislabeled as RTO or recovery-wasted time. Role breakdown
is null for full runs without inventory export, not inferred from tiny runs.

Checked the existing Qwen tiny fixed artifact: unchanged86400 model +166592
optimizer +81408 runtime remote logical bytes and recovery pass. The active
DeepSeek full chain was correctly rejected as incomplete. Its successful
full-analysis path still awaits the actual final chain verdict.

### 72.111 October 28: Full DeepSeek mismatch and deterministic dispatch fix

The p1 full DeepSeek fresh restore finished all16 target bundle reads and
continued step2. Restore-controller time1501.9863724894822s, continued step
wall about31.616s, loss11.9446 versus source11.9445. All16 final model and
optimizer component digests differ; scheduler/runtime digests agree. Source
recovery-oracle.json records bitwise_equal=false; source FAILED and outer
status exit1 at03:51:58Z. Restore-only COMPLETE means continuation, not the
parent's strict recovery equivalence. The queued policy matrix stopped with
exit1 as intended. This run is excluded from successful full-model results.

Isolated a real nondeterminism bug with no checkpoint involved:
test_top6_dispatch_repeats_bitwise_forward_and_backward uses one CUDA GPU,
256 tokens,128 hidden,8 experts,top6, BF16 expert compute and identical input,
weights and upstream gradients across10 evaluations. Original code fails on
repeated forward output (component0), log20261028-top6-repeat-before.log,
1 failed in5.49s. dispatch_to_experts used repeated token index_select and
index_add, allowing CUDA atomic addition ordering to affect values.

Changed only dispatch assignment expansion and combination: each token/expert
assignment has a unique index for permutation, and weighted outputs reshape
to [tokens,top_k,hidden] then sum across top_k. Expansion backward also avoids
duplicate token-index atomic accumulation. The recurrence test plus existing
CPU/Gloo and two-GPU NCCL EP forward/backward/empty-route tests passed:
3 passed in18.03s, log20261028-top6-repeat-after.log. This demonstrates the
isolated bug/fix, not yet that it explains every p1 full-state difference.

Because the arithmetic order changes, old source continuation cannot serve
as the new-code oracle. Released only the p1 node17 survivor service after
preserving logs, digests, manifests and source archive. Both old workers and
old queue were terminal before this action. The old payload is no longer
available in RAM; do not attempt to reuse its locator as a live checkpoint.

Started fresh tmux moe-full-policy-p2-20261028, outer log/status
runs/moe-capacity/20261028-p2-policy-queue.log{,.status}. Updated the stopped
queue script to accept a fresh campaign ID and include all six arms, beginning
20261028-p2-ds-fixed, then ds-host,ds-m1,qwen-fixed,qwen-host,qwen-m1. Each
run archives current source and uses identical corrected arithmetic, host-
serial publication/restoration, two source steps, epoch1 snapshot and fresh
step2 bitwise comparison. No full-model pass is claimed before this rerun.

### 72.112 October 28: Optimizer-only serialized BF16 reconstruction gate

Advanced the existing four-rank CPU master reconstruction test without
changing the live GPU runtime or shell launchers. After each of three AdamW
updates, capture_zero1_fragments and capture_zero1_step_fragments serialize
only optimizer.master/exp_avg/exp_avg_sq/step tensors through torch.save.
Explicitly zero live model parameters, masters, moments and scalar clocks;
load the tensor-only payload and restore using apply_zero1_fragments. Gather
the restored master parameter ranges over their actual shared/expert owner
groups, require exactly-once coverage, cast to BF16, compare bitwise against
the pre-clear model, and copy reconstructed values into the live parameters
before the next update. All three reconstruction boundaries passed.

Command: pytest -q test/core/moe/test_zero1_optimizer.py::
test_bf16_model_reconstructs_exactly_from_role_sharded_fp32_masters
under the prepared environment, CPU/Gloo, OMP/MKL threads1.
Evidence: runs/moe-capacity/20261028-master-restore-gate.log{,.status},
tmux moe-master-restore-gate-20261028;1 passed in8.86s, exit0.
This is optimizer-only serialized parameter reconstruction in the same live
process group, not a complete runtime/RNG/scheduler checkpoint, fresh-process
service recovery, or the three-policy stronger-encoder experiment. Those
boundaries remain explicit; no full-model savings are inferred from this gate.

At03:58UTC the p2 full DeepSeek source was live after step1 loss11.9452,
in snapshot publication. Queue moe-full-policy-p2-20261028 and both worker
launches remain active; sampled host available memory remained above the
24GiB floor. No p2 completion or recovery outcome is yet claimed.

### 72.113 October 28: Restore read accounting and live publication

The existing MoE analyzer now extracts all16 ZERO1_TARGET_RESTORE records
from the two restore logs, preserving per-rank bundle counts and payload
bytes, and reports summed actual read bytes separately from publication.
Checked against the archived Qwen r2 per-rank analysis: exact agreement,
255008523112 bytes. DeepSeek p1 diagnostic reads total267681521756 bytes,
16 bundles; its source FAILED still excludes it from successful analysis.
These are data payload bytes, excluding restore control reads/protocol wire.

At approximately04:02UTC verified tmux queue and child capacity process live;
read-only stats from the p2 services at172.20.17.2:34701 and172.20.8.2:36193
reported3 and2 records respectively. Publication is making progress. Do not
infer a sealed checkpoint from these partial service counts. No restart or
runtime-code change was made while this p2 experiment was running.

### 72.114 October 28: Matched-step wall and sampled resource reporting

Added reporting fields to the existing MoE analyzer for maximum over ranks
of each rank's sum of step walls (not sum of per-step maxima), sealed epoch1
in two source optimizer steps, and sampled source/restore GPU memory and host
availability. The one-commit/two-step observation is explicitly not sustained
cadence. Whole-device usage/host availability is not job-attributed peak RSS.

Parsed completed historical diagnostics and stored input CSV SHA-256 hashes
in runs/research-summary-20261027/full-memory-diagnostics.json. DeepSeek p1
source sampled single-GPU maxima node17/node8=41454/41878 MiB; minimum host
available=169320980/55635704 KiB. Its restore maxima=40476/42378 MiB,
minimum available=212925308/247331136 KiB. These remain failed-oracle
diagnostics, not successful performance rows. Qwen r2 continuation maxima
36326/37282 MiB, minimum available239719136/371757212 KiB, still without an
uninterrupted reference. Earlier training-only c1 runs lack the host CSV and
are not silently admitted by this full-chain resource parser.

P2 source/queue processes remained live at04:08UTC. Its current publication
has not yet sealed, and no partial p2 timings enter the successful result set.

### 72.115 October 28: Refresh the AE package after the dispatch fix

Preserved the initially delivered code1028.zip verbatim as
runs/research-summary-20261027/code1028-initial.zip,
SHA-256 fa138255ad7930ce79974151f57d2bd20f72ec10b3efe1d95e7804d431607b6a.
Explicitly refreshed the public code1028.zip so the delivered code includes
the demonstrated dispatch determinism fix, its tests, the full six-arm queue,
and current analysis helpers. AE_README now documents the p1 failed oracle,
p2 pending verdict, old tiny evidence's code-version boundary, and the
optimizer-derived encoder limitation; no full-model success is implied.

Refreshed archive:309 files (290.py,8.sh,11.md),1073785 bytes;
SHA-256 c84d2929cf048c83bcd451b925a22fda5cb5a4f65b0c02cf1d53d95230b426fe.
ZIP CRC and extension/path whitelist checks passed. The initial snapshot is
retained outside this archive; no weights/logs/results/non-approved types
were bundled. This explicit refresh supersedes the earlier public archive
while preserving its historical identity for reproducibility.

P2 remained live during packaging; service record counts were8/7 at04:10UTC.
No running experiment scripts or runtime code were edited in this refresh.

### 72.116 October 28: Corrected DeepSeek p2 source sealed; recovery launched

Full DeepSeek p2 fixed source (runs/moe-capacity/20261028-p2-ds-fixed)
sealed epoch1 and completed step2. PUBLICATION_COMPLETE exists, both node
worker statuses are exit0, and all16 final rank digests record iteration2.
The unchanged queue stopped node8's checkpoint service and launched fresh
restore20261028-p2-ds-fixed-r at approximately04:24UTC, using node17 survivor
data. Restore workers and the parent queue remain active; final equivalence
after the dispatch fix is still unproven.

Source metrics: remote logical data267674810112 B, data payload267681521756 B,
total payload267682992393 B, wire267683014258 B; aliases applied. Rank0 capture
53.7652296666s, publication1530.8048863132s; maximum client-observed remote PUT
87755.3287167ms, PUT-through-commit-ACK87758.4921438ms. Source log step1 wall
1613.204s, step2 wall23.800s and loss11.9441. These are single-run diagnostics
until the full chain succeeds, not a claimed stable timing benefit versus p1
(whose arithmetic differs and oracle failed).

ACTIVE restore artifact: runs/moe-restore/20261028-p2-ds-fixed-r.
Owning queue: moe-full-policy-p2-20261028. Preserve survivor service and do
not launch other GPU work while recovery/next-step oracle is pending.

### 72.117 October 28: Full DeepSeek fixed bitwise recovery passed

The corrected full DeepSeek fixed p2 chain passed. Source and fresh restore
both have COMPLETE; recovery-oracle.json records16 ranks,step2,
bitwise_equal=true,mismatches={}. Independently compared all16 raw source
and restored state/component checksums: model,optimizer,scheduler,runtime
all exactly equal. Source/restored step2 losses both print11.9441. Node8
checkpoint service was stopped; all16 fresh workers used node17's survivor.

Reproducible summary: runs/moe-capacity/20261028-p2-ds-fixed/analysis.json,
generated by tools/analyze_moe_snapshot.py. Its successful full-chain path
is now exercised on real evidence. Publication metrics remain as72.116.
Maximum rank sum of two source step walls1637.0065553s; step1 maximum
1613.2044862s including exposed checkpoint stall1584.7007953s; step2 maximum
23.8148567s. Restore controller1407.4672946s excludes process startup;
restored step2 maximum30.7272665s. Read16 bundles totaling267681521756 bytes.
This is not an end-to-end RTO or recovery-wasted-time claim.

Source single-GPU sampled maxima node17/node8=41454/41876 MiB; minimum host
available172095668/57908404 KiB. Restore maxima40476/42376 MiB; minimum host
available214913204/244163356 KiB. Whole-device/host samples are not per-job
peak RSS. One synchronous checkpoint in two source steps establishes full-
model admission, not sustained cadence or repeated timing evidence. No
Host/M1 speedup is inferred before matching arms finish.

At04:50UTC the queue cleaned up fixed services and automatically started
runs/moe-capacity/20261028-p2-ds-host (host_global_fixed). Its services and
workers are launched under tmux; queue remains moe-full-policy-p2-20261028.
Remaining order: DS M1, Qwen fixed/Host/M1. The overall goal remains active.

### 72.118 October 28: Queue matched two-step NoCheckpoint controls

The active six-arm queue does not include NoCheckpoint. Added a separate
minimal follower tools/run_moe_full_nockpt_after_queue.sh, reusing the same
capacity launcher and model/seed/layout/optimizer configuration, two source
steps, publication disabled. It waits for the existing policy queue's exit0
status AND COMPLETE before running DS and Qwen controls sequentially. It
stops on policy-queue failure; no competing GPU work is launched while waiting.
This is a two-step timing control, not a new independent full-state oracle.
The six checkpoint arms retain their source-versus-fresh-restore oracle.

bash syntax check passed. Live follower tmux moe-nockpt-p2-20261028;
log/status runs/moe-capacity/20261028-p2-nockpt-controls.log{,.status}.
Future run IDs:20261028-p2-ds-nockpt and20261028-p2-qwen-nockpt. Source archives
are created before each control run. No live shell or runtime was edited.
The current DS Host publication remains active; controls have not run yet.
The frozen309-file AE ZIP predates this follower and can be refreshed at the
next results handoff rather than being silently claimed to contain it.

### 72.119 October 28: Full DeepSeek Host published; recovery pending

Host source20261028-p2-ds-host sealed epoch1 and both workers exited0
(node17 05:17:10Z,node8 05:17:26Z). All16 source step2 full-state/component
digests exactly match fixed p2 source; detector evidence SHA-256 also matches.
Compared every archived Merak Python file across fixed/Host source tarballs:
identical. This supports matched implementation/input evidence for publication.

Publication-only comparison saved to the Host run's publication-comparison.json,
explicitly marked recovery pending. Fixed -> Host remote logical bytes:
267674810112 ->231016068864 (-13.695253%); total payload:
267682992393 ->231021006150 (-13.696046%); wire:
267683014258 ->231021028016. Rank0 capture53.765230 ->56.581514s,
publish1530.804886 ->1383.192262s (-9.642811%); maximum remote PUT
87755.328717 ->80904.704884ms (-7.806505%). Single observations only;
do not claim stable speedup or successful recoverable Host savings yet.
Source Host log step1 wall1468.072s,step2 wall23.199s,loss11.9441.

The queue stopped node8's service and launched fresh restore
runs/moe-restore/20261028-p2-ds-host-r. Node17 survivor service remains needed.
Owning queue moe-full-policy-p2-20261028; next arm DS M1 only after the Host
full-state continuation oracle passes. NoCheckpoint follower remains waiting.

### 72.120 October 28: Host recovery read-amplification observation

Host fresh restore remains live. rank0 read one14002182463-byte data bundle;
rank1 required two totaling27110950535 bytes. rank8 initially read one
19660522926-byte bundle. These are completed per-target reads only, not the
final job total or a failed oracle.

Read-only review of restore_zero1_from_host shows a concrete efficiency
limitation: target_only puts the target rank first, then scans remaining source
ranks in global ascending order until all requested addresses/roots are
covered. Pooled model objects owned by a later pipeline stage can therefore
cause unrelated earlier-stage bundles to be downloaded and validated first.
This is not a reason to loosen integrity/coverage checks or claim a failure;
it is a potential recovery cost of the current coarse bundle reader. Prioritize
same sealed PP/TP coordinates before global fallback if optimization becomes
necessary, and preserve reader-version/timing comparability across arms.
No runtime change or restart was made in this observation. Do not conflate
the measured13.696% publication reduction with recovery-byte/time reduction.

### 72.121 October 28: Patent patch delivered; stage-first reader regression passed

Created docs/patent_patch.md after reviewing all15 pages and5 figures of
docs/patent_v4.pdf. It provides page-specific replacement text and diagram
edits for process grouping versus object protection, stored checkpoint sources,
prospective failure coverage, global commit, layout mapping, and terminology.
The PDF itself was not edited.

The Host old-reader run now confirms the predicted amplification: rank9 read
10 bundles totaling154898701691 bytes, versus fixed's one19660522926-byte
bundle for rank9. Original CPU/TCP regression failed with actual read order
[3,0,1,2] rather than [3,2], while restored values were correct. This is a
reader efficiency bug, not evidence of corrupt checkpoint state.

Changed target-only source ordering in zero1_service.py: own rank first,
then matching sealed PP/TP coordinates, then all remaining sources. No source
is removed, and manifest, checksum and full target-coverage checks remain.
CPU/TCP test verifies same-stage recovery reads only [3,2], and an actual
cross-stage target still succeeds through fallback [3,2,0]. Together with the
existing four-rank role-aware ZeRO-1 service/restore/continuation gate:
2 passed in13.83s, exit0. Logs20261028-stage-read-before.log and
20261028-stage-read-after.log preserve the failing/passing evidence.

Current Host restore workers loaded the OLD reader and remain live; the
change does not hot-patch them. Future queue processes load the NEW reader,
whose archive records the change. Publication logic and arithmetic are
unchanged. Do not mix old/new reader recovery times into a policy-speedup
claim; matching reader-version recovery runs are still required. No GPU
restart or running-shell edit was performed in this change. The public ZIP
predates this reader change and patent patch; refresh at next AE handoff.

### 72.122 October 28: Retry Host recovery without republishing

At05:50UTC deliberately terminated old Host restore launcher PID2835365,
after proving rank9's154898701691-byte unrelated-stage scan and passing the
reader regression. Its EXIT cleanup removed only owned restore workers on
both nodes; GPU process lists became empty. Source parent preserved node17's
service as designed, recorded FAILED for restore, and exited; p2 policy and
NoCheckpoint waiting queues both exited1. Old restore INTERRUPTED.md records
the intervention. No old failure/partial result was overwritten as success.

Confirmed survivor172.20.17.2:42283 still has17 records, then started fresh
restore20261028-p2-ds-host-r2 under tmux moe-ds-host-r2-20261028, log/status
runs/moe-restore/20261028-p2-ds-host-r2-campaign.log{,.status}. Same source
checkpoint and step2 oracle, stage-first reader, host-serial schedule. Reader
source archive:runs/moe-restore/20261028-p2-ds-host-r2-source.tar.gz,
SHA-256 b84173a383229a8e4e58be8a459bc71639917b3aa82b4d156f04b0073ac6cad5.

Extended the existing analyzer with explicit --restore-run selection. This
requires published source, successful source/restore node exits, matching
source-run binding, completed restore and exact16-rank step2 state/component
digests. It reports explicit_restore_retry and original_source_chain_failed
rather than erasing the interrupted parent status. Verified against successful
DS p2 fixed and rejected known mismatching DS p1 raw digests.

New waiting owner:moe-full-policy-p3-20261028, log/status
runs/moe-capacity/20261028-p3-policy-queue.log{,.status}, executing archived
one-off runs/moe-capacity/20261028-reader-retry-continue.sh. After r2 exit0 it
runs the strict analyzer into r2/analysis.json; only on success releases the
old Host survivor service and starts the remaining DS M1,Qwen fixed/Host/M1
via run_moe_full_policy_queue.sh 20261028-p3 ds-m1. The stopped queue script
now accepts a validated starting arm; syntax checks passed. No live shell
was edited. The new NoCheckpoint follower moe-nockpt-p3-20261028 waits for
that queue before matched DS/Qwen controls. Do not launch competing GPU jobs.

Publication implementation/arithmetic remain unchanged; source archived code
now differs only where reader/reporting work changed. Existing fixed recovery
timing is old-reader evidence; pooled-policy new-reader timing must not be
used to claim a matched reader-version speedup until that comparison is run.

### 72.123 October 28: Stage-first full-model read reduction observed

At06:01UTC Host r2 rank9 finished with2 bundles and38821886836 data-payload
bytes, compared with10 bundles and154898701691 bytes in the interrupted old
reader. Both read the same sealed Host checkpoint. This directly confirms
removal of unrelated-stage scans for this target, not a policy byte-reduction
claim or a complete recovery verdict. Completed targets so far:0,1,8,9.
The r2 process and p3 continuation waiter remain live; full16-rank step2
oracle is still pending. Keep the node17 survivor service available.

### 72.124 October 28: Full DeepSeek Host recovery passed; M1 started

Host r2 finished with exit0 at06:29:01Z and COMPLETE. The explicit retry
analyzer independently verified all16 step2 raw full-state/component digests
against the sealed Host source continuation: bitwise_equal=true. Source and
restored losses both print11.9441. Result:
runs/moe-restore/20261028-p2-ds-host-r2/analysis.json. The original source
parent FAILED remains preserved for the deliberately interrupted old reader;
analysis explicitly records the successful retry and prior failure history.

Host publication231021006150 total payload bytes is13.696046% below fixed
267682992393, with identical detector evidence and source step2 state. Host
maximum rank sum source step wall1491.2770015s; step1 maximum1468.0720320s,
exposed stall1439.8958965s; step2 maximum23.2229441s. Single observations only.

Host restore read32 bundles totaling496077919396 bytes; ranks at each stage
read1,2,2,3 bundles respectively. Controller2089.7560227s excludes startup;
restored step2 maximum29.3878130s. Whole-bundle reads still amplify recovery
payload despite lower publication bytes. Fixed observed16 bundles and
267681521756 read bytes, controller1407.4672946s, with the older reader.
Report these observed costs and version distinction; do not claim matched
reader-version RTO non-inferiority or recovery speedup.

Validated two-arm aggregation with analysis hashes:
runs/research-summary-20261027/deepseek-full-fixed-host.json.
Source single-GPU maxima17/8=41454/42132 MiB, minimum host availability
216695568/88648636 KiB. Restore maxima40476/42376 MiB, minimum availability
245961992/325568956 KiB. These remain sampled whole-device/host metrics.

After strict retry analysis passed, continuation wrapper released the Host
survivor service and automatically started20261028-p3-ds-m1 under
moe-full-policy-p3-20261028. Next:Qwen fixed/Host/M1, then NoCheckpoint
controls. Host service is no longer a live recovery source. Overall goal,
remaining policies/models, stronger encoder controls and final report remain
unfinished; no completion claim is made.

### 72.125 October 28: User-directed final run and repository cleanup

Latest user scope supersedes the outstanding experiment plan: the cluster
administrator is reclaiming the machines. Finish ONLY the currently running
DeepSeek M1 experiment (20261028-p3-ds-m1, including its existing restore and
oracle), then end GPU experiments. Do not start Qwen or NoCheckpoint runs.
Prepare final docs/10-27.md, retain necessary reproducible code/configuration
and evidence, delete unnecessary generated trees/large files, and git add the
retained work. Do not modify .gitignore and do not make a commit.

At06:33UTC SIGSTOP'ed only policy scheduler PID2859852, leaving its current
capacity child PID2869218 running. Terminated the waiting NoCheckpoint
scheduler PID2859864; it is no longer live. Paused policy scheduler must be
terminated after the current child finishes, not resumed into more jobs.
Current M1 remains source publication, then automatic restore/bitwise oracle.

Inventory: experiments about796GiB, runs about21GiB. Prepared
tools/retain_ae_evidence.py to preserve selected dense/tiny/full-MoE raw
configs, commands, digests, logs and source archives under artifacts/ae1028
with SHA-256 before deleting original trees. It must run after the current
M1 restore artifact exists and must be verified before deletion. No generated
trees have been deleted yet. Copied the exact small DeepSeek architecture
definitions, config and upstream license to third_party/deepseek_v2_lite
(no weights), removing a future dependency on the departing cluster's external
model directory. Runtime default path is not changed until the current job ends.

Original .gitignore SHA-256 at this request:
9a6582694824fafe69a5dce44276b67c71db482d5cf9193ee38fc4f7ab426c1b.
Necessary ignored JSON/shell/evidence files will be explicitly force-added;
the ignore file contents must remain unchanged. User explicitly authorized
deletion of unnecessary workspace artifacts; no additional permission needed.
