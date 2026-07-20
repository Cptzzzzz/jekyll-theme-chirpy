# NCCL Algorithm and Protocol Sweep

Run ID: `20260710T014532Z`

## Selected Results

| algorithm | protocol | 1 KiB us | 64 KiB us | 1 MiB us | 16 MiB GB/s | 256 MiB GB/s | max CV |
|---|---|---:|---:|---:|---:|---:|---:|
| Ring | Simple | 33.52 | 45.23 | 60.42 | 97.61 | 122.51 | 5.96% |
| Ring | LL | 18.58 | 19.57 | 41.78 | 45.89 | 42.90 | 5.21% |
| Ring | LL128 | 26.01 | 32.78 | 50.10 | 78.85 | 95.43 | 0.63% |
| Tree | Simple | 37.01 | 48.52 | 92.73 | 48.42 | 76.02 | 0.31% |
| Tree | LL | 20.62 | 23.71 | 57.98 | 30.96 | 29.72 | 2.55% |
| Tree | LL128 | 21.49 | 24.67 | 59.41 | 58.99 | 68.37 | 0.74% |

## Best Combination by Message Size

| bytes | best | median us | busbw GB/s |
|---:|---|---:|---:|
| 1024 | Ring + LL | 18.58 | 0.08 |
| 65536 | Ring + LL | 19.57 | 5.02 |
| 1048576 | Ring + LL | 41.78 | 37.64 |
| 16777216 | Ring + Simple | 257.83 | 97.61 |
| 268435456 | Ring + Simple | 3286.68 | 122.51 |

## Validation

- Every case completed all out-of-place and in-place checks with `#wrong=0`.
- Algorithm and protocol were forced only for isolation; these settings are not production recommendations.
- The best combination changes with message size, so a single 64 MiB comparison cannot describe NCCL tuning behavior.
