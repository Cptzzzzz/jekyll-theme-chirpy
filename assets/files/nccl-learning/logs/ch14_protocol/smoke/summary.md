# Chapter 14 experiment summary

- Main measurement rows: 348
- Alignment measurement rows: 16
- Out-of-place correctness: PASS
- In-place correctness: PASS
- Nsight kernel-path profiles: 0

## Protocol layout

| protocol | line B | data B | efficiency | default buffer | data/step | sync |
|---|---:|---:|---:|---:|---:|---|
| LL | 16 | 8 | 50.00% | 524288 | 32768 | two_32bit_flags_per_16B_line |
| LL128 | 128 | 120 | 93.75% | 4915200 | 576000 | one_64bit_flag_per_128B_line |
| Simple | 0 | 0 | 100.00% | 4194304 | 524288 | separate_connection_step_head_tail |

## Selected performance points

| algo | proto | bytes | median us | P95 | CV | busbw |
|---|---|---:|---:|---:|---:|---:|
| Ring | LL | 4 | 31.255 | 31.448 | 0.69% | 0.00 |
| Ring | LL | 1024 | 32.790 | 33.546 | 2.56% | 0.05 |
| Ring | LL | 65536 | 38.920 | 41.197 | 6.50% | 2.54 |
| Ring | LL | 1048576 | 62.475 | 62.947 | 0.84% | 25.18 |
| Ring | LL | 16777216 | 578.950 | 584.494 | 1.06% | 43.47 |
| Ring | LL | 1073741824 | 36973.350 | 37215.585 | 0.73% | 43.56 |
| Ring | LL128 | 4 | 45.130 | 46.210 | 2.66% | 0.00 |
| Ring | LL128 | 1024 | 47.490 | 48.786 | 3.03% | 0.03 |
| Ring | LL128 | 65536 | 55.660 | 56.308 | 1.29% | 1.77 |
| Ring | LL128 | 1048576 | 71.545 | 71.945 | 0.62% | 21.98 |
| Ring | LL128 | 16777216 | 331.685 | 331.815 | 0.04% | 75.88 |
| Ring | LL128 | 1073741824 | 16432.000 | 16501.030 | 0.47% | 98.02 |
| Ring | Simple | 4 | 50.620 | 51.385 | 1.68% | 0.00 |
| Ring | Simple | 1024 | 56.725 | 59.358 | 5.16% | 0.03 |
| Ring | Simple | 65536 | 67.240 | 68.887 | 2.72% | 1.46 |
| Ring | Simple | 1048576 | 83.555 | 83.686 | 0.17% | 18.82 |
| Ring | Simple | 16777216 | 282.895 | 283.601 | 0.28% | 88.96 |
| Ring | Simple | 1073741824 | 13101.300 | 13119.930 | 0.16% | 122.94 |
| Tree | LL | 4 | 32.025 | 32.047 | 0.08% | 0.00 |
| Tree | LL | 1024 | 31.065 | 31.988 | 3.30% | 0.05 |
| Tree | LL | 65536 | 41.295 | 41.372 | 0.21% | 2.38 |
| Tree | LL | 1048576 | 78.630 | 79.854 | 1.73% | 20.01 |
| Tree | LL | 16777216 | 833.360 | 835.880 | 0.34% | 30.20 |
| Tree | LL | 1073741824 | 54018.700 | 54033.370 | 0.03% | 29.81 |
| Tree | LL128 | 4 | 42.670 | 42.967 | 0.77% | 0.00 |
| Tree | LL128 | 1024 | 39.955 | 40.077 | 0.34% | 0.04 |
| Tree | LL128 | 65536 | 48.585 | 48.599 | 0.03% | 2.02 |
| Tree | LL128 | 1048576 | 82.385 | 83.532 | 1.55% | 19.09 |
| Tree | LL128 | 16777216 | 450.785 | 451.978 | 0.29% | 55.83 |
| Tree | LL128 | 1073741824 | 22449.850 | 22459.525 | 0.05% | 71.75 |
| Tree | Simple | 4 | 54.910 | 54.973 | 0.13% | 0.00 |
| Tree | Simple | 1024 | 51.040 | 51.400 | 0.78% | 0.03 |
| Tree | Simple | 65536 | 63.325 | 63.527 | 0.36% | 1.56 |
| Tree | Simple | 1048576 | 125.830 | 128.278 | 2.16% | 12.51 |
| Tree | Simple | 16777216 | 534.815 | 537.529 | 0.56% | 47.05 |
| Tree | Simple | 1073741824 | 20911.500 | 20930.580 | 0.10% | 77.02 |

## Best overall by size

| bytes | best | runner | margin | decision |
|---:|---|---|---:|---|
| 4 | Ring+LL | Tree+LL | 2.46% | practical_tie |
| 8 | Ring+LL | Tree+LL | 6.45% | winner |
| 16 | Tree+LL | Ring+LL | 1.68% | practical_tie |
| 32 | Ring+LL | Tree+LL | 4.92% | practical_tie |
| 64 | Ring+LL | Tree+LL | 0.35% | practical_tie |
| 128 | Tree+LL | Ring+LL | 1.97% | practical_tie |
| 256 | Tree+LL | Ring+LL | 0.00% | practical_tie |
| 512 | Ring+LL | Tree+LL | 3.27% | practical_tie |
| 1024 | Tree+LL | Ring+LL | 5.55% | winner |
| 2048 | Ring+LL | Tree+LL | 1.46% | practical_tie |
| 4096 | Tree+LL | Ring+LL | 1.55% | practical_tie |
| 8192 | Ring+LL | Tree+LL | 0.41% | practical_tie |
| 16384 | Ring+LL | Tree+LL | 5.79% | winner |
| 32768 | Ring+LL | Tree+LL | 3.50% | practical_tie |
| 65536 | Ring+LL | Tree+LL | 6.10% | winner |
| 131072 | Tree+LL | Ring+LL | 1.32% | practical_tie |
| 262144 | Ring+LL | Tree+LL | 33.97% | winner |
| 524288 | Ring+LL | Tree+LL | 23.60% | winner |
| 1048576 | Ring+LL | Ring+LL128 | 14.52% | winner |
| 2097152 | Ring+LL128 | Ring+Simple | 5.11% | winner |
| 4194304 | Ring+Simple | Ring+LL128 | 4.84% | practical_tie |
| 8388608 | Ring+Simple | Ring+LL128 | 18.92% | winner |
| 16777216 | Ring+Simple | Ring+LL128 | 17.25% | winner |
| 33554432 | Ring+Simple | Ring+LL128 | 25.73% | winner |
| 67108864 | Ring+Simple | Ring+LL128 | 28.85% | winner |
| 134217728 | Ring+Simple | Ring+LL128 | 27.75% | winner |
| 268435456 | Ring+Simple | Ring+LL128 | 27.57% | winner |
| 536870912 | Ring+Simple | Ring+LL128 | 26.83% | winner |
| 1073741824 | Ring+Simple | Ring+LL128 | 25.42% | winner |

## LL128 alignment

| unalign float elements | offset B | bytes | median us | CV | vs aligned |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 1048576 | 81.815 | 9.43% | +0.00% |
| 0 | 0 | 67108864 | 1177.660 | 3.33% | +0.00% |
| 1 | 4 | 1048576 | 83.560 | 0.36% | +2.13% |
| 1 | 4 | 67108864 | 1339.820 | 0.12% | +13.77% |
| 2 | 8 | 1048576 | 84.420 | 0.05% | +3.18% |
| 2 | 8 | 67108864 | 1341.555 | 0.12% | +13.92% |
| 3 | 12 | 1048576 | 92.235 | 10.96% | +12.74% |
| 3 | 12 | 67108864 | 1343.765 | 0.01% | +14.10% |
