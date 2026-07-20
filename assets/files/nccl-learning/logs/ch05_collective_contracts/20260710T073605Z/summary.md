# Chapter 05 experiment summary

- Rank rows: `56`
- Contract variants: `14`
- Full-tensor/reference checks: `PASS`

| operation | variant | rank rows | result |
|---|---|---:|---|
| all_gather | into-tensor-rank-major | 4 | PASS |
| all_gather | list-rank-major | 4 | PASS |
| all_reduce | max-float32-in-place | 4 | PASS |
| all_reduce | sum-int64-in-place | 4 | PASS |
| all_reduce | zero-count | 4 | PASS |
| all_to_all | equal-splits-source-major-output | 4 | PASS |
| all_to_all | uneven-splits-source-major-output | 4 | PASS |
| broadcast | invalid-root-rejected | 4 | PASS |
| broadcast | root-0 | 4 | PASS |
| broadcast | root-2 | 4 | PASS |
| reduce | sum-root-0 | 4 | PASS |
| reduce | sum-root-3 | 4 | PASS |
| reduce_scatter | sum-contiguous-rank-chunk | 4 | PASS |
| send_recv | grouped-ring | 4 | PASS |

## Assertions

- Rooted operations were checked on two different roots.
- Reduce non-root receive contents were treated as outside the contract.
- Equal and uneven AllToAll layouts were checked element by element.
- Grouped ring Send/Recv delivered source-encoded payloads on all ranks.
- Out-of-range broadcast root was rejected on every rank.
