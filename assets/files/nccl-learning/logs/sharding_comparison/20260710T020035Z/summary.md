# DDP and FSDP Sharding Comparison

Run ID: `20260710T020035Z`

| mode | cycles | median step ms | P95 ms | cycle CV | allocated MiB | peak MiB | replicas equal |
|---|---:|---:|---:|---:|---:|---:|---:|
| ddp | 5 | 11.808 | 11.885 | 1.11% | 976.5 | 1168.5 | 1 |
| fsdp | 5 | 32.957 | 39.873 | 8.83% | 208.5 | 400.5 | 1 |

- Steady allocated-memory reduction of FSDP versus DDP: `78.65%`.
- Median-step change of FSDP versus DDP: `179.11%`.
- FSDP uses per-linear FULL_SHARD, so it exchanges parameter shards around each wrapped layer instead of retaining a full replica.
- This is a mechanism experiment on one model shape, not a universal FSDP performance recommendation.
