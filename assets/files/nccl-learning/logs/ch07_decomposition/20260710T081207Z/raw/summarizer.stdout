# Chapter 07 experiment summary

## Contract matrix

- Contract rank rows: `20`
- Exact full-tensor checks: `PASS`
- Negative transforms rejected: `PASS`

| contract | rank rows | result |
|---|---:|---|
| exact-sum-equivalence | 4 | PASS |
| nondivisible-without-padding-rejected | 4 | PASS |
| padding-and-trim-equivalence | 4 | PASS |
| wrong-reduction-op-rejected | 4 | PASS |
| wrong-shard-order-rejected | 4 | PASS |

## Timing

Each mode used 3 warmups and 10 alternating-order cycles. Per-cycle latency is the maximum across four ranks.

| size | mode | median us | P95 us | CV | payload GB/s | RS+AG / AR |
|---:|---|---:|---:|---:|---:|---:|
| 1 KiB | direct | 86.272 | 113.674 | 15.19% | 0.012 | - |
| 1 KiB | decomposed | 133.920 | 145.674 | 5.03% | 0.008 | unstable |
| 64 KiB | direct | 93.328 | 370.274 | 86.53% | 0.702 | - |
| 64 KiB | decomposed | 132.272 | 455.768 | 72.39% | 0.495 | unstable |
| 1 MiB | direct | 113.600 | 510.243 | 85.38% | 9.230 | - |
| 1 MiB | decomposed | 152.176 | 190.872 | 11.76% | 6.891 | unstable |
| 16 MiB | direct | 381.440 | 408.765 | 3.28% | 43.984 | - |
| 16 MiB | decomposed | 431.568 | 445.875 | 1.95% | 38.875 | 1.131x |
| 256 MiB | direct | 3611.792 | 3633.774 | 0.48% | 74.322 | - |
| 256 MiB | decomposed | 3990.800 | 4026.605 | 0.62% | 67.264 | 1.105x |

Only size pairs where both modes have CV <= 5% receive a numeric ratio. Other ratios remain in the CSV as descriptive observations and are not accepted as precision conclusions.

## Allocated memory

Payload is `256 MiB` per rank. Values are deltas from each rank's clean ProcessGroup baseline and use the maximum rank.

| mode | steady allocated MiB | peak allocated MiB |
|---|---:|---:|
| allreduce-full | 256.0 | 256.0 |
| reduce-scatter-shard | 64.0 | 320.0 |
| rs-ag-naive-lifetime | 256.0 | 576.0 |
| rs-ag-optimized-lifetime | 256.0 | 320.0 |

## Nsight Systems kernel attribution

One process managed four GPUs; the measured NVTX range contained 5 iterations at 64 MiB.

- Direct AllReduce NCCL kernels: `20` (`5 x 4 GPUs`).
- ReduceScatter + AllGather NCCL kernels: `40` (`5 x 2 collectives x 4 GPUs`).
- Correctness endpoints: `PASS`.
