# DDP Bucket Sweep

Run ID: `20260710T015205Z`

| cap MiB | actual bucket sizes | median step ms | P95 ms | comm ms | overlap ms | peak MiB | replicas equal |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `201326592` | 7.506 | 8.030 | 6.275 | 2.353 | 593.0 | 1 |
| 4 | `201326592` | 7.281 | 7.624 | 13.644 | 2.592 | 593.0 | 1 |
| 16 | `201326592` | 7.142 | 7.785 | 10.422 | 2.833 | 593.0 | 1 |
| 64 | `201326592` | 6.854 | 7.185 | 5.541 | 1.634 | 593.0 | 1 |

## Fixed Facts

- Total parameter size: `192.0 MiB`.
- One weight tensor size: `16.0 MiB`.
- DDP cannot split one parameter tensor across buckets; the configured cap is therefore not a strict maximum.
- Every configuration ended with identical parameter replicas across all ranks.
