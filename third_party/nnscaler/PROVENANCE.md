# Vendored nnScaler provenance

- Source path: `../../eimc/nnscaler-clean`
- Snapshot date: 2026-08-29
- Reported version: `0.5`
- Source-only SHA-256 tree digest:
  `a862749fd9c84625f9620eb5cf89abd445478308919a920396e332bb6dfa4b0d`
- License: Microsoft MIT; the license text was recovered from the adjacent
  `../../eimc/Enhanced-In-Memory-Checkpoint/LICENSE` because the source-only
  directory contained MIT file headers but no standalone license file.
- Upstream commit: unknown.  The source directory has no Git metadata, so this
  snapshot must not be attributed to a specific Microsoft nnScaler commit.

The digest covers relative paths and contents for regular files after excluding
`__pycache__/`, `*.pyc`, `.rendered.*`, and `*.so`.  `LICENSE`, this provenance
file, and the compatibility files below are not part of the source digest.

## Local compatibility change

The source snapshot retained imports of
`nnscaler.runtime.memckpt.memckpt_config.comm_profiler` in
`runtime/adapter/collectives.py` and `cli/trainer.py`, but contained no
`runtime/memckpt` directory.  This repository supplies a deliberately no-op
`comm_profiler` compatibility object so the planner/runtime can import without
silently importing the modified MemCkpt fork.  It implements no save, restore,
profiling, or recovery behavior.

## Research boundary

Vendored nnScaler is an offline plan producer.  It may generate a versioned
parallel-plan JSON containing rank placement and tensor/shard metadata.  Merak
and AdaCheck consume that JSON through an explicit adapter; they do not import
nnScaler in the checkpoint hot path.  Results must report the snapshot digest
and any generated-plan digest.
