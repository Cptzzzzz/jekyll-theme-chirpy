# DDP and FSDP Sharding Comparison

Run ID: `20260710T015937Z`

| mode | cycles | median step ms | P95 ms | cycle CV | allocated MiB | peak MiB | replicas equal |
|---|---:|---:|---:|---:|---:|---:|---:|
| ddp | 3 | 11.746 | 11.910 | 0.94% | 976.5 | 1168.5 | 1 |
| fsdp | 3 | 39.358 | 39.402 | 9.72% | 208.5 | 400.5 | 1 |

- Steady allocated-memory reduction of FSDP versus DDP: `78.65%`.
- Median-step change of FSDP versus DDP: `235.07%`.
- FSDP uses per-linear FULL_SHARD, so it exchanges parameter shards around each wrapped layer instead of retaining a full replica.
- This is a mechanism experiment on one model shape, not a universal FSDP performance recommendation.
