# Chapter 03 experiment summary

## Host and device completion

| blocking | bytes | mode | host call us | host wait/sync us | stream dependency us | completed after wait |
|---:|---:|---|---:|---:|---:|---:|
| 0 | 1024 | async_device_sync | 88.832 | 14.276 | 107.776 | 1.00 |
| 0 | 1024 | async_wait | 78.729 | 4.151 | 99.040 | 0.75 |
| 0 | 1024 | sync_api | 92.450 | n/a | 130.512 | n/a |
| 0 | 1048576 | async_device_sync | 118.770 | 43.196 | 138.160 | 1.00 |
| 0 | 1048576 | async_wait | 77.254 | 4.066 | 111.232 | 0.00 |
| 0 | 1048576 | sync_api | 71.391 | n/a | 112.368 | n/a |
| 0 | 67108864 | async_device_sync | 998.681 | 920.496 | 1019.408 | 1.00 |
| 0 | 67108864 | async_wait | 80.561 | 4.130 | 990.032 | 0.00 |
| 0 | 67108864 | sync_api | 77.829 | n/a | 993.168 | n/a |
| 0 | 268435456 | async_device_sync | 3585.577 | 3508.934 | 3606.304 | 1.00 |
| 0 | 268435456 | async_wait | 80.366 | 4.149 | 3573.712 | 0.00 |
| 0 | 268435456 | sync_api | 74.236 | n/a | 3584.432 | n/a |
| 1 | 1024 | async_device_sync | 466.450 | 239.002 | 520.480 | 1.00 |
| 1 | 1024 | async_wait | 10455.075 | 10187.285 | 10576.272 | 1.00 |
| 1 | 1024 | sync_api | 10409.753 | n/a | 10519.376 | n/a |
| 1 | 1048576 | async_device_sync | 486.223 | 303.018 | 532.016 | 1.00 |
| 1 | 1048576 | async_wait | 10456.067 | 10180.289 | 10569.248 | 1.00 |
| 1 | 1048576 | sync_api | 10429.816 | n/a | 10539.168 | n/a |
| 1 | 67108864 | async_device_sync | 1176.670 | 989.712 | 1233.776 | 1.00 |
| 1 | 67108864 | async_wait | 20597.929 | 20336.819 | 20716.720 | 1.00 |
| 1 | 67108864 | sync_api | 10494.294 | n/a | 10612.352 | n/a |
| 1 | 268435456 | async_device_sync | 3706.673 | 3546.216 | 3752.832 | 1.00 |
| 1 | 268435456 | async_wait | 10526.634 | 10192.050 | 10646.048 | 1.00 |
| 1 | 268435456 | sync_api | 15469.667 | n/a | 15586.176 | n/a |

## Stream-scoped wait

- Samples: `40`
- Median host `work.wait()`: `5.441 us`
- Median queued GPU sleep: `130135.376 us`
- Free stream completed before wait-dependent stream: `100.0%`
- Work already complete immediately after nonblocking wait: `0.0%`

## Assertions

- Every timed output passed full-tensor AllReduce correctness.
- Nonblocking `wait()` establishes a CUDA-stream dependency without requiring host completion.
- Blocking-wait mode polls Work completion on the host.
