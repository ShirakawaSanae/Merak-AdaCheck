# Standalone Communication Profile

## Boundary

`Merak/core/checkpoint/communication_profile.py` is an optional observability
module. It uses only the Python standard library and has no dependency on the
AdaCheck detector, semantic encoder, grouping solver, transport, global commit,
or recovery implementation. Those components may submit already measured
monotonic timestamps, but the profiler cannot select or mutate their actions.

The profiler deliberately does not call `time.*`, synchronize a CUDA stream,
or enter a distributed collective. Disabled profiling returns before timestamp
validation or filesystem creation. Enabled profiling appends compact JSONL, so
raw evidence remains separate from rank metrics and can be reanalyzed without
rerunning training.

## Current Signals

The live Merak adapter records the following interval types. Each interval's start and end
must come from one process-local monotonic clock domain:

| Interval | Meaning | Critical-path interpretation |
| --- | --- | --- |
| `training_compute` | `train_batch` including the optimizer hook | End-to-end foreground compute boundary. |
| `checkpoint_capture` | Hook metadata plus asynchronous snapshot enqueue | Nested foreground checkpoint work; no remote ACK wait is implied. |
| `checkpoint_poll` | Completion polling and coordinator advancement | Exposed post-compute checkpoint work. |
| `checkpoint_final_drain` | Publisher close, global completion, seal wait, and optional oracle | Completion/recovery-readiness tail, excluded from steady step time. |
| `checkpoint_publisher_drain` | Wait for queued snapshot publishers to commit | Separates sustained service backlog from validation. |
| `checkpoint_global_publish_seal` | Final global control publication and seal ACK | Global recovery-point completion cost. |
| `checkpoint_materialized_reader_wait` | Wait for a bounded node-local full-state reader slot | Host memory/admission tail, not network transfer. |
| `checkpoint_materialized_fetch_deserialize` | Service response plus checkpoint decode | Validation/restore data retrieval cost. |
| `checkpoint_materialized_oracle_compare` | Local full-state numerical comparison | Validation-only CPU/memory work. |
| `checkpoint_materialized_validation_barrier` | Wait for every rank to finish validation | Straggler tail after local validation. |
| `checkpoint_restore_validate_load` | Binding validation and framework state load | Recovery CPU/load work after retrieval. |
| `checkpoint_materialized_restore_barrier` | Wait for all ranks to finish materialized restore | Recovery straggler tail. |
| `checkpoint_finalize_total`, `checkpoint_restore_total` | Whole-operation compatibility totals | Never substituted for steady training time. |

`tools/analyze_merak_incremental.py` reports all-rank and per-epoch rank-maximum
P50/P95/P99/max tails. Completed remote data-plane `max_time` remains a separate
transport metric: it is the longest edge, never the sum of rank durations.
Logical bytes, application payload bytes, wire bytes, observed global commit
rate, and boundary lag are also kept separate.

Absolute monotonic timestamps from different ranks or hosts are not assumed to
be aligned. Cross-rank tail aggregation compares completed durations, not raw
start or end timestamps. Likewise, directional idle-slack analysis is valid only
when both interval streams share an explicitly established clock domain; the
current live adapter does not infer cross-host overlap from unsynchronized clocks.

The library additionally provides deterministic interval union and directional
idle-slack primitives. They are not yet used to schedule traffic. A future
adapter may add absolute send/receive intervals, provided it preserves the same
clock domain and does not introduce synchronization merely to obtain a timestamp.

## Relation To The nnScaler Prototype

The earlier EIMC profiler under
`../../eimc/Enhanced-In-Memory-Checkpoint/nnscaler/runtime/memckpt/` is a design
reference for communication-pattern and idle-time analysis. Its timing methods
call `stream.synchronize()` and its layout logic is coupled to nnScaler. Reusing
those calls in the current Merak measurement would change the checkpointed
training critical path. The standalone module therefore reuses the concepts,
not the synchronization or framework-specific implementation.

## Paper Claim

The implemented, defensible system-design point is a non-perturbing
observability plane that distinguishes background service time from foreground
exposure and final recovery readiness. It supports fair diagnosis and policy
ablation, but it does not reduce bytes or latency by itself.

A stronger `profile-guided scheduling` contribution remains proposed. It may be
claimed only after one scheduler consumes a frozen or held-out profile and,
under the same model, parallel layout, detector, encoder, grouping, transport,
and recovery constraints, improves exposed tail or sustainable checkpoint
frequency against the profiler-disabled scheduling baseline. Report profiler
overhead in that ablation and keep grouping-byte gains attributed to M1.

## Live Evidence

The r101 16-rank SemanticFixed gate produced one JSONL file per rank: 160
`training_compute`, 160 `checkpoint_capture`, 160 `checkpoint_poll`, and 16
`checkpoint_final_drain` intervals. Reanalysis reports per-epoch rank-maximum
capture and poll maxima of `2.944 s` and `0.726 s`, while the remote data-plane
edge maximum remains separately measured as `34.815 s`. The final-drain maximum
is `3,040.828 s` and is excluded from steady step latency. This validates the
standalone measurement path and exposes where the tail lives; it does not show
that profiling itself improves the tail.

## Verification

Focused behavior is covered by
`test/core/checkpoint/test_communication_profile.py` and aggregation by
`test/core/checkpoint/test_analyze_merak_incremental.py`. End-to-end runs store
one `checkpoint/communication-profile/rank_XXXXX.jsonl` file per rank.
