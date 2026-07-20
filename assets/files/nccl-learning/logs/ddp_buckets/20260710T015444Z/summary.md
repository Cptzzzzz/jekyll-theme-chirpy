# DDP Bucket Sweep

Run ID: `20260710T015444Z`

| cap MiB | rebuilt bucket sizes | median step ms | P95 ms | cycle CV | comm ms | overlap ms | peak MiB | replicas equal |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216` | 7.340 | 7.812 | 3.82% | 6.996 | 2.403 | 593.0 | 1 |
| 4 | `16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216` | 7.208 | 7.430 | 2.56% | 9.893 | 2.282 | 593.0 | 1 |
| 16 | `16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216, 16777216` | 7.263 | 7.675 | 2.56% | 9.244 | 2.781 | 593.0 | 1 |
| 64 | `67108864, 67108864, 67108864` | 6.876 | 7.011 | 1.29% | 7.848 | 1.697 | 593.0 | 1 |

## Fixed Facts

- Total parameter size: `192.0 MiB`.
- One weight tensor size: `16.0 MiB`.
- DDP cannot split one parameter tensor across buckets; the configured cap is therefore not a strict maximum.
- Every configuration ended with identical parameter replicas across all ranks.
