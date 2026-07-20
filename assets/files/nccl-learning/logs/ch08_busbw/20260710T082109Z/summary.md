# Chapter 08 experiment summary

- Raw out-of-place measurement rows: `360`
- Formula max absolute error after two-decimal report quantization: algbw `0.00891` GB/s, busbw `0.00912` GB/s
- Out-of-place correctness: `PASS` for all rows
- Supported in-place correctness: `PASS` for all rows

## Bus bandwidth factors

| collective | N=2 | N=3 | N=4 |
|---|---:|---:|---:|
| all_reduce | 1.000000 | 1.333333 | 1.500000 |
| all_gather | 0.500000 | 0.666667 | 0.750000 |
| reduce_scatter | 0.500000 | 0.666667 | 0.750000 |
| broadcast | 1.000000 | 1.000000 | 1.000000 |
| reduce | 1.000000 | 1.000000 | 1.000000 |
| alltoall | 0.500000 | 0.666667 | 0.750000 |

## Large-message observations

The table uses the actual aligned size printed by nccl-tests, ten cycles, and out-of-place results.

| collective | ranks | actual bytes | factor | median us | CV | algbw GB/s | busbw GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_gather | 2 | 67108864 | 0.500000 | 791.410 | 0.07% | 84.79 | 42.40 |
| all_gather | 3 | 67108848 | 0.666667 | 621.610 | 0.34% | 107.96 | 71.97 |
| all_gather | 4 | 67108864 | 0.750000 | 469.425 | 0.23% | 142.96 | 107.22 |
| all_reduce | 2 | 67108864 | 1.000000 | 1533.285 | 0.40% | 43.77 | 43.77 |
| all_reduce | 3 | 67108864 | 1.333333 | 1107.190 | 1.06% | 60.61 | 80.81 |
| all_reduce | 4 | 67108864 | 1.500000 | 859.710 | 0.75% | 78.06 | 117.09 |
| alltoall | 2 | 67108864 | 0.500000 | 842.290 | 0.09% | 79.67 | 39.84 |
| alltoall | 3 | 67108848 | 0.666667 | 684.665 | 2.15% | 98.02 | 65.34 |
| alltoall | 4 | 67108864 | 0.750000 | 535.965 | 0.08% | 125.22 | 93.91 |
| broadcast | 2 | 67108864 | 1.000000 | 1494.360 | 0.02% | 44.91 | 44.91 |
| broadcast | 3 | 67108864 | 1.000000 | 858.130 | 0.15% | 78.20 | 78.20 |
| broadcast | 4 | 67108864 | 1.000000 | 648.795 | 1.85% | 103.44 | 103.44 |
| reduce | 2 | 67108864 | 1.000000 | 1488.855 | 0.01% | 45.07 | 45.07 |
| reduce | 3 | 67108864 | 1.000000 | 852.540 | 0.06% | 78.72 | 78.72 |
| reduce | 4 | 67108864 | 1.000000 | 695.360 | 0.21% | 96.51 | 96.51 |
| reduce_scatter | 2 | 67108864 | 0.500000 | 1138.075 | 1.26% | 58.97 | 29.48 |
| reduce_scatter | 3 | 67108848 | 0.666667 | 639.390 | 0.09% | 104.95 | 69.97 |
| reduce_scatter | 4 | 67108864 | 0.750000 | 524.570 | 0.10% | 127.93 | 95.95 |

## Assertions

- Recomputed algbw used the actual reported bytes divided by reported time.
- Recomputed busbw applied the exact GetBw factor for the collective and world size.
- Three-rank alignment-adjusted sizes were not replaced with the requested CLI size.
- busbw is a modeled normalization and no hardware link counter was read in this experiment.
