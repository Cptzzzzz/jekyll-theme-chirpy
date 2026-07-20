# Model Parallel Communication Patterns

Generated: `2026-07-10T02:04:27.883030+00:00`

| pattern | size MiB | median ms | P95 ms | payload GB/s | correct |
|---|---:|---:|---:|---:|---:|
| tp_all_reduce | 1 | 0.249 | 0.259 | 4.20 | 1 |
| sharded_rs_ag | 1 | 0.249 | 0.259 | 4.22 | 1 |
| ep_all_to_all | 1 | 0.387 | 0.743 | 2.71 | 1 |
| pp_send_recv | 1 | 0.254 | 0.366 | 4.13 | 1 |
| tp_all_reduce | 16 | 0.412 | 0.415 | 40.74 | 1 |
| sharded_rs_ag | 16 | 0.467 | 0.483 | 35.96 | 1 |
| ep_all_to_all | 16 | 0.333 | 0.342 | 50.41 | 1 |
| pp_send_recv | 16 | 0.504 | 0.892 | 33.26 | 1 |
| tp_all_reduce | 64 | 0.955 | 1.092 | 70.28 | 1 |
| sharded_rs_ag | 64 | 1.055 | 1.203 | 63.64 | 1 |
| ep_all_to_all | 64 | 0.582 | 0.604 | 115.26 | 1 |
| pp_send_recv | 64 | 1.508 | 1.620 | 44.49 | 1 |

`payload_GBs` is application payload divided by critical-rank latency. It is not NCCL busbw and must not be compared as a physical-link utilization metric across different collective semantics.
