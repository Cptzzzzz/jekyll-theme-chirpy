# Chapter 06 experiment summary

## Native Group semantics

- Native rank rows: `76`
- Semantic variants: `9`
- Full-tensor and return-code checks: `PASS`

| variant | rank rows | result |
|---|---:|---|
| async-error-after-success | 4 | PASS |
| group-end-without-start | 4 | PASS |
| invalid-root-group-cleanup | 4 | PASS |
| nested-depth-two | 4 | PASS |
| nonblocking-finalize-poll | 4 | PASS |
| nonblocking-group-poll | 4 | PASS |
| nonblocking-init-poll | 4 | PASS |
| two-user-streams | 4 | PASS |
| valid-op-after-group-error | 4 | PASS |

## GroupEnd versus device completion

Payload: one in-place `256 MiB` AllReduce, 3 warmups and 10 measured cycles.
Per-cycle values use the maximum across four ranks.

| metric | median (us) | P95 (us) | CV |
|---|---:|---:|---:|
| host GroupStart/API/GroupEnd interval | 14.945 | 26.219 | 26.88% |
| CUDA event completion interval | 3548.625 | 3561.929 | 0.25% |

- CUDA end events already ready immediately after GroupEnd: `0/40` rank-cycles.
- Observed median device/host interval ratio: `237.44x`.
- Host interval CV is `26.88%`; the host median and ratio are descriptive evidence only, not a precision performance baseline.

## ProcessGroup ordering matrix

| case | debug | return | duration (s) | classified outcome | result |
|---|---|---:|---:|---|---|
| ordered_off | OFF | 0 | 4.827 | success | PASS |
| op_mismatch_detail | DETAIL | 1 | 4.994 | fingerprint-rejected | PASS |
| size_mismatch_detail | DETAIL | 1 | 4.833 | fingerprint-rejected | PASS |
| op_mismatch_raw | OFF | 1 | 11.055 | partial-return-then-watchdog | PASS |
| rank_skip_raw | OFF | 1 | 11.316 | watchdog-timeout | PASS |

## Assertions

- Nested Group depth two launched three correctly ordered AllReduces.
- GroupEnd without GroupStart returned `ncclInvalidUsage` on every rank.
- Invalid root was returned by both the inner API and balanced outer GroupEnd.
- A valid collective succeeded after group error cleanup on the blocking communicator.
- Nonblocking init, first Group submission, and Finalize reached terminal success through async-error polling.
- `TORCH_DISTRIBUTED_DEBUG=DETAIL` rejected op and shape mismatch through fingerprints.
- With debug OFF, the mismatch was not reclassified as a fingerprint failure.
- A skipped rank reached the timeout path after a healthy communicator warmup.
