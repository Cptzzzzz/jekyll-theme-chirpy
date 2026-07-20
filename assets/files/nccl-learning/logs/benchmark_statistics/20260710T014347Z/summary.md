# NCCL Benchmark Statistics

Run ID: `20260710T014347Z`

## Summary

| bytes | median us | P95 us | cycle CV | algbw GB/s | busbw GB/s | in-place us |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 25.96 | 27.18 | 2.87% | 0.04 | 0.06 | 24.73 |
| 4096 | 25.15 | 25.96 | 1.88% | 0.16 | 0.24 | 25.06 |
| 16384 | 25.75 | 27.05 | 3.10% | 0.64 | 0.95 | 24.60 |
| 65536 | 26.74 | 27.53 | 2.28% | 2.45 | 3.68 | 26.30 |
| 262144 | 27.65 | 29.51 | 3.51% | 9.48 | 14.22 | 26.90 |
| 1048576 | 53.61 | 56.32 | 1.54% | 19.56 | 29.34 | 51.44 |
| 4194304 | 104.94 | 107.20 | 0.70% | 39.97 | 59.95 | 103.71 |
| 16777216 | 261.65 | 265.93 | 0.53% | 64.12 | 96.19 | 261.90 |
| 67108864 | 856.64 | 860.07 | 0.15% | 78.34 | 117.50 | 858.35 |
| 268435456 | 3290.12 | 3291.73 | 0.03% | 81.59 | 122.38 | 3294.68 |

## Model Checks

- Median latency for 1 KiB through 64 KiB: `25.86 us`.
- Large-message fitted algorithm bandwidth: `82.99 GB/s`.
- Large-message fitted intercept: `54.40 us`.
- Large-message linear fit R-squared: `0.999987`.
- Maximum CV across ten cycle-level means: `3.51%`.
- All out-of-place and in-place correctness checks reported `#wrong=0`.
- Recomputed AllReduce busbw used `algbw * 2 * (N-1) / N = algbw * 1.5` for four ranks.

## Interpretation

Small messages are dominated by a nearly fixed launch/synchronization cost. Large messages approach a linear bytes/time relation, while the transition region reflects protocol, channel and algorithm decisions. Cycle-level stability and per-iteration jitter are different quantities and must be reported separately.
