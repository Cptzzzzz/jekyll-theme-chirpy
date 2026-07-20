# Chapter 14 experiment summary

- Main measurement rows: 1740
- Alignment measurement rows: 160
- Out-of-place correctness: PASS
- In-place correctness: PASS
- Nsight kernel-path profiles: 6

## Small-message stability repeat

The original matrix had 19 groups above 5% CV at 64 KiB or below. The five
affected algorithm/protocol combinations were repeated in two fresh processes,
20 cycles each, producing 3000 additional rows and 40 samples per group.

- Ring protocol winner at every repeated size: LL.
- Ring LL advantage over the next protocol: 37.05%-104.34%.
- Tree protocol winner at every repeated size: LL.
- Tree LL advantage over the next protocol: 32.80%-47.95%.
- 45/75 repeated groups still exceeded 5% CV because of long tails or process-state drift.
- Absolute LL128/Simple medians shifted roughly 14%-16% at selected sizes, so they are not stable microsecond baselines.

| bytes | best | runner | runner margin |
|---:|---|---|---:|
| 4 | Tree+LL 17.750 us | Ring+LL 18.115 us | 2.06% |
| 1024 | Ring+LL 17.880 us | Tree+LL 18.200 us | 1.79% |
| 16384 | Ring+LL 18.185 us | Tree+LL 19.365 us | 6.49% |
| 65536 | Ring+LL 19.220 us | Tree+LL 21.555 us | 12.15% |

## Protocol layout

| protocol | line B | data B | efficiency | default buffer | data/step | sync |
|---|---:|---:|---:|---:|---:|---|
| LL | 16 | 8 | 50.00% | 524288 | 32768 | two_32bit_flags_per_16B_line |
| LL128 | 128 | 120 | 93.75% | 4915200 | 576000 | one_64bit_flag_per_128B_line |
| Simple | 0 | 0 | 100.00% | 4194304 | 524288 | separate_connection_step_head_tail |

## Selected performance points

| algo | proto | bytes | median us | P95 | CV | busbw |
|---|---|---:|---:|---:|---:|---:|
| Ring | LL | 4 | 18.445 | 20.022 | 4.34% | 0.00 |
| Ring | LL | 1024 | 18.545 | 20.700 | 5.06% | 0.08 |
| Ring | LL | 65536 | 19.020 | 20.757 | 4.64% | 5.17 |
| Ring | LL | 1048576 | 41.675 | 41.831 | 0.18% | 37.74 |
| Ring | LL | 16777216 | 537.625 | 561.103 | 2.32% | 46.81 |
| Ring | LL | 1073741824 | 37803.400 | 37885.260 | 0.18% | 42.61 |
| Ring | LL128 | 4 | 21.355 | 21.456 | 0.26% | 0.00 |
| Ring | LL128 | 1024 | 26.035 | 26.145 | 0.30% | 0.06 |
| Ring | LL128 | 65536 | 32.985 | 34.722 | 2.72% | 2.98 |
| Ring | LL128 | 1048576 | 50.425 | 50.535 | 0.27% | 31.20 |
| Ring | LL128 | 16777216 | 313.045 | 316.075 | 0.45% | 80.39 |
| Ring | LL128 | 1073741824 | 16373.000 | 16403.435 | 0.10% | 98.37 |
| Ring | Simple | 4 | 30.300 | 34.759 | 7.68% | 0.00 |
| Ring | Simple | 1024 | 33.380 | 33.538 | 0.35% | 0.05 |
| Ring | Simple | 65536 | 45.250 | 45.596 | 0.49% | 2.17 |
| Ring | Simple | 1048576 | 60.730 | 61.495 | 0.64% | 25.90 |
| Ring | Simple | 16777216 | 257.880 | 258.991 | 0.25% | 97.59 |
| Ring | Simple | 1073741824 | 13043.200 | 13054.420 | 0.06% | 123.48 |
| Tree | LL | 4 | 19.145 | 20.781 | 4.54% | 0.00 |
| Tree | LL | 1024 | 18.870 | 21.704 | 7.64% | 0.08 |
| Tree | LL | 65536 | 22.350 | 28.992 | 14.01% | 4.40 |
| Tree | LL | 1048576 | 58.130 | 58.584 | 0.46% | 27.05 |
| Tree | LL | 16777216 | 801.500 | 804.746 | 0.31% | 31.40 |
| Tree | LL | 1073741824 | 53936.600 | 54065.815 | 0.18% | 29.86 |
| Tree | LL128 | 4 | 21.835 | 22.177 | 0.79% | 0.00 |
| Tree | LL128 | 1024 | 21.955 | 25.697 | 7.23% | 0.07 |
| Tree | LL128 | 65536 | 24.980 | 25.073 | 0.22% | 3.94 |
| Tree | LL128 | 1048576 | 59.550 | 60.005 | 0.37% | 26.41 |
| Tree | LL128 | 16777216 | 426.520 | 427.226 | 0.11% | 59.01 |
| Tree | LL128 | 1073741824 | 22441.000 | 22473.075 | 0.15% | 71.77 |
| Tree | Simple | 4 | 36.380 | 36.895 | 0.63% | 0.00 |
| Tree | Simple | 1024 | 36.485 | 36.736 | 0.37% | 0.04 |
| Tree | Simple | 65536 | 48.155 | 48.288 | 0.25% | 2.04 |
| Tree | Simple | 1048576 | 92.745 | 93.251 | 0.31% | 16.96 |
| Tree | Simple | 16777216 | 519.500 | 521.049 | 0.18% | 48.45 |
| Tree | Simple | 1073741824 | 20884.550 | 20897.915 | 0.04% | 77.12 |

## Best overall by size

| bytes | best | runner | margin | decision |
|---:|---|---|---:|---|
| 4 | Ring+LL | Tree+LL | 3.80% | practical_tie |
| 8 | Ring+LL | Tree+LL | 0.46% | practical_tie |
| 16 | Ring+LL | Tree+LL | 3.40% | practical_tie |
| 32 | Ring+LL | Tree+LL | 1.54% | practical_tie |
| 64 | Ring+LL | Tree+LL | 0.46% | practical_tie |
| 128 | Tree+LL | Ring+LL | 0.08% | practical_tie |
| 256 | Ring+LL | Tree+LL | 1.14% | practical_tie |
| 512 | Ring+LL | Tree+LL | 1.54% | practical_tie |
| 1024 | Ring+LL | Tree+LL | 1.75% | practical_tie |
| 2048 | Ring+LL | Tree+LL | 2.31% | practical_tie |
| 4096 | Ring+LL | Tree+LL | 4.42% | practical_tie |
| 8192 | Ring+LL | Tree+LL | 1.29% | practical_tie |
| 16384 | Ring+LL | Tree+LL | 5.50% | winner |
| 32768 | Ring+LL | Tree+LL | 15.30% | winner |
| 65536 | Ring+LL | Tree+LL | 17.51% | winner |
| 131072 | Ring+LL | Tree+LL | 13.90% | winner |
| 262144 | Ring+LL | Tree+LL | 21.10% | winner |
| 524288 | Ring+LL | Tree+LL | 42.62% | winner |
| 1048576 | Ring+LL | Ring+LL128 | 21.00% | winner |
| 2097152 | Ring+LL | Ring+LL128 | 0.24% | practical_tie |
| 4194304 | Ring+Simple | Ring+LL128 | 12.55% | winner |
| 8388608 | Ring+Simple | Ring+LL128 | 23.65% | winner |
| 16777216 | Ring+Simple | Ring+LL128 | 21.39% | winner |
| 33554432 | Ring+Simple | Ring+LL128 | 28.42% | winner |
| 67108864 | Ring+Simple | Ring+LL128 | 30.76% | winner |
| 134217728 | Ring+Simple | Ring+LL128 | 29.30% | winner |
| 268435456 | Ring+Simple | Ring+LL128 | 27.81% | winner |
| 536870912 | Ring+Simple | Ring+LL128 | 26.91% | winner |
| 1073741824 | Ring+Simple | Ring+LL128 | 25.53% | winner |

## LL128 alignment

| unalign float elements | offset B | bytes | median us | CV | vs aligned |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 1048576 | 50.340 | 0.40% | +0.00% |
| 0 | 0 | 67108864 | 1116.085 | 0.18% | +0.00% |
| 1 | 4 | 1048576 | 54.775 | 0.47% | +8.81% |
| 1 | 4 | 67108864 | 1204.765 | 0.20% | +7.95% |
| 2 | 8 | 1048576 | 54.750 | 0.47% | +8.76% |
| 2 | 8 | 67108864 | 1205.905 | 0.24% | +8.05% |
| 3 | 12 | 1048576 | 54.790 | 0.20% | +8.84% |
| 3 | 12 | 67108864 | 1204.470 | 0.19% | +7.92% |

## Kernel-path proof

| algo | requested proto | selected id | launch family | instances | median kernel us | grid | block |
|---|---|---:|---|---:|---:|---|---|
| Ring | LL | 0 | `_RING_LL` | 56 | 44.111 | 12 | 512 |
| Ring | LL128 | 1 | `_RING_LL` | 56 | 57.232 | 12 | 640 |
| Ring | Simple | 2 | `_RING_LL` | 56 | 66.111 | 12 | 544 |
| Tree | LL | 0 | `_TREE_LL` | 56 | 58.895 | 12 | 640 |
| Tree | LL128 | 1 | `_TREE_LL` | 56 | 65.151 | 12 | 640 |
| Tree | Simple | 2 | `_TREE_LL` | 56 | 93.823 | 12 | 640 |
