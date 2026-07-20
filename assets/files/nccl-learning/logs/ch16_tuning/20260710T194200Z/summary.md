# Chapter 16 experiment summary

- Auto measurements: 290 rows, out-of-place/in-place correctness PASS
- Rounded INFO model table reproduces all 29 logged auto selections: PASS
- Exact selection hits against Chapter 14 forced sweep: 26/29
- Practical hits (regret < 5%): 28/29
- Worst regret: 1048576 B, 21.00%
- Tuner plugin override: Tree/Simple, channels=2, median slowdown 3.48x
- Nsight plugin geometry: gridX=2

## Model table

| algorithm | protocol | latency us | algorithm BW GB/s |
|---|---|---:|---:|
| Tree | LL | 10.4 | 14.5 |
| Tree | LL128 | 21.5 | 42.9 |
| Tree | Simple | 168.0 | 55.2 |
| Ring | LL | 10.2 | 26.0 |
| Ring | LL128 | 25.4 | 73.6 |
| Ring | Simple | 28.8 | 80.0 |

## Auto-selection regret

| bytes | auto | measured best | regret | model abs error |
|---:|---|---|---:|---:|
| 4 | Ring+LL | Ring+LL | 0.00% | 44.70% |
| 8 | Ring+LL | Ring+LL | 0.00% | 45.12% |
| 16 | Ring+LL | Ring+LL | 0.00% | 44.98% |
| 32 | Ring+LL | Ring+LL | 0.00% | 44.96% |
| 64 | Ring+LL | Ring+LL | 0.00% | 44.70% |
| 128 | Ring+LL | Tree+LL | 0.08% | 45.36% |
| 256 | Ring+LL | Ring+LL | 0.00% | 44.41% |
| 512 | Ring+LL | Ring+LL | 0.00% | 44.79% |
| 1024 | Ring+LL | Ring+LL | 0.00% | 44.79% |
| 2048 | Ring+LL | Ring+LL | 0.00% | 43.55% |
| 4096 | Ring+LL | Ring+LL | 0.00% | 42.82% |
| 8192 | Ring+LL | Ring+LL | 0.00% | 43.47% |
| 16384 | Ring+LL | Ring+LL | 0.00% | 42.19% |
| 32768 | Ring+LL | Ring+LL | 0.00% | 38.57% |
| 65536 | Ring+LL | Ring+LL | 0.00% | 33.12% |
| 131072 | Ring+LL | Ring+LL | 0.00% | 23.24% |
| 262144 | Ring+LL | Ring+LL | 0.00% | 1.73% |
| 524288 | Ring+LL | Ring+LL | 0.00% | 20.00% |
| 1048576 | Ring+LL128 | Ring+LL | 21.00% | 21.37% |
| 2097152 | Ring+LL128 | Ring+LL | 0.24% | 27.63% |
| 4194304 | Ring+Simple | Ring+Simple | 0.00% | 20.38% |
| 8388608 | Ring+Simple | Ring+Simple | 0.00% | 9.68% |
| 16777216 | Ring+Simple | Ring+Simple | 0.00% | 7.51% |
| 33554432 | Ring+Simple | Ring+Simple | 0.00% | 0.96% |
| 67108864 | Ring+Simple | Ring+Simple | 0.00% | 1.90% |
| 134217728 | Ring+Simple | Ring+Simple | 0.00% | 3.17% |
| 268435456 | Ring+Simple | Ring+Simple | 0.00% | 2.80% |
| 536870912 | Ring+Simple | Ring+Simple | 0.00% | 3.26% |
| 1073741824 | Ring+Simple | Ring+Simple | 0.00% | 3.12% |

## Plugin override

| policy | median us | P95 us | CV | busbw GB/s |
|---|---:|---:|---:|---:|
| auto_64M | 849.825 | 851.000 | 0.09% | 118.45 |
| plugin_tree_simple_ch2 | 2955.280 | 2956.914 | 0.06% | 34.06 |
