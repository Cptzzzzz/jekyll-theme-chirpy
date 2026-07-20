# DDP Bucket Sweep

Run ID: `20260710T015313Z`

| cap MiB | rebuilt bucket sizes | median step ms | P95 ms | comm ms | overlap ms | peak MiB | replicas equal |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216` | 7.786 | 8.124 | 10.479 | 2.451 | 593.0 | 1 |
| 4 | `16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216` | 7.258 | 7.442 | 5.046 | 2.361 | 593.0 | 1 |
| 16 | `16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216` | 7.080 | 7.799 | 7.656 | 2.495 | 593.0 | 1 |
| 64 | `67108864, 67108864, 67108864` | 6.806 | 7.444 | 6.640 | 1.681 | 593.0 | 1 |

## Fixed Facts

- Total parameter size: `192.0 MiB`.
- One weight tensor size: `16.0 MiB`.
- DDP cannot split one parameter tensor across buckets; the configured cap is therefore not a strict maximum.
- Every configuration ended with identical parameter replicas across all ranks.
