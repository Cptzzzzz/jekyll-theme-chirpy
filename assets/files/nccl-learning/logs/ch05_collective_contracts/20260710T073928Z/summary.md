# Chapter 05 experiment summary

- PyTorch rank rows: `56`
- PyTorch contract variants: `14`
- Native NCCL rank rows: `36`
- Native NCCL pointer/layout variants: `10`
- Full-tensor/reference and pointer-layout checks: `PASS`

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

## Native NCCL C API

| operation | variant | pointer relation | rank rows | result |
|---|---|---|---:|---|
| all_gather | in-place-rank-offset | `sendbuff==recvbuff+rank*sendcount` | 4 | PASS |
| all_gather | out-of-place | `sendbuff separate;recvbuff[nranks*sendcount]` | 4 | PASS |
| all_reduce | in-place | `sendbuff==recvbuff` | 4 | PASS |
| all_reduce | out-of-place | `sendbuff!=recvbuff` | 4 | PASS |
| broadcast | in-place-root-1 | `sendbuff==recvbuff` | 4 | PASS |
| broadcast | out-of-place-root-2 | `sendbuff!=recvbuff;root is communicator rank` | 4 | PASS |
| reduce | out-of-place-root-3 | `non-root recvbuff==nullptr` | 3 | PASS |
| reduce | out-of-place-root-3 | `root recvbuff valid` | 1 | PASS |
| reduce_scatter | in-place-rank-offset | `recvbuff==sendbuff+rank*recvcount` | 4 | PASS |
| reduce_scatter | out-of-place | `sendbuff[nranks*recvcount];recvbuff separate` | 4 | PASS |

## Assertions

- Rooted operations were checked on two different roots.
- Reduce non-root receive contents were treated as outside the contract.
- Equal and uneven AllToAll layouts were checked element by element.
- Grouped ring Send/Recv delivered source-encoded payloads on all ranks.
- Out-of-range broadcast root was rejected on every rank.
- Native AllGather and ReduceScatter in-place pointer offsets were exercised directly.
- Native Reduce accepted a null receive pointer on every non-root rank.
