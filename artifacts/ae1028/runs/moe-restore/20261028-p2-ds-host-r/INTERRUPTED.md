# Intentionally interrupted diagnostic

2026-10-28 approximately05:50UTC: terminated only this restore launcher after
confirming excessive unrelated-stage reads. rank9 read10 bundles totaling
154898701691 bytes. The stage-first reader passed the targeted TCP test and
four-rank continuation gate before this intervention.

Source publication and16 step2 source digests remain valid. The owning source
launcher recorded FAILED for this interrupted restore and retained node17's
sealed service. The old policy and NoCheckpoint queues stopped with exit1.
Do not count this partial restore as success or infer a payload integrity error.

Replacement: ../20261028-p2-ds-host-r2 using the same source checkpoint with
the revised reader. Its full-state verdict is independent and must be checked.
