# Chapter 09 experiment summary

- Config process runs: 46
- Out-of-place measurement rows: 920
- Out-of-place correctness: PASS for all rows
- In-place correctness: PASS for all rows

## Aggregated results

| matrix | variant | bytes | n | median us | P95 us | CV | ref delta | first-cycle penalty |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| affinity | cpu0_numa0 | 1048576 | 20 | 61.725 | 62.500 | 1.10% | +0.00% | +0.18% |
| affinity | cpu0_numa0 | 67108864 | 20 | 864.620 | 869.474 | 0.32% | +0.00% | +0.09% |
| affinity | cpu1_numa1 | 1048576 | 20 | 54.355 | 60.194 | 4.15% | -11.94% | +5.34% |
| affinity | cpu1_numa1 | 67108864 | 20 | 854.025 | 861.593 | 0.37% | -1.23% | +0.51% |
| blocking | z0 | 1048576 | 20 | 60.730 | 62.461 | 5.22% | +0.00% | +0.73% |
| blocking | z0 | 67108864 | 20 | 862.645 | 867.817 | 0.59% | +0.00% | +0.42% |
| blocking | z1 | 1048576 | 20 | 77.315 | 79.490 | 1.58% | +27.31% | +2.18% |
| blocking | z1 | 67108864 | 20 | 874.620 | 879.106 | 0.26% | +1.39% | +0.45% |
| blocking | z2 | 1048576 | 20 | 77.905 | 79.052 | 1.18% | +28.28% | +0.62% |
| blocking | z2 | 67108864 | 20 | 874.545 | 878.290 | 0.18% | +1.38% | +0.28% |
| blocking | z3 | 1048576 | 20 | 88.210 | 100.778 | 5.87% | +45.25% | +7.00% |
| blocking | z3 | 67108864 | 20 | 885.650 | 898.853 | 2.49% | +2.67% | -0.07% |
| instrumentation | I0 | 1048576 | 20 | 50.385 | 51.045 | 0.55% | +0.00% | +0.69% |
| instrumentation | I0 | 67108864 | 20 | 850.330 | 850.757 | 0.08% | +0.00% | -0.04% |
| instrumentation | I1 | 1048576 | 20 | 54.125 | 54.715 | 0.50% | +7.42% | -0.05% |
| instrumentation | I1 | 67108864 | 20 | 853.855 | 854.786 | 0.08% | +0.41% | -0.11% |
| iterations | n1 | 1048576 | 20 | 85.975 | 96.039 | 8.62% | +59.11% | -0.65% |
| iterations | n1 | 67108864 | 20 | 888.720 | 901.072 | 1.06% | +4.07% | -0.18% |
| iterations | n100 | 1048576 | 20 | 53.570 | 53.717 | 0.19% | -0.86% | -0.15% |
| iterations | n100 | 67108864 | 20 | 852.740 | 853.271 | 0.05% | -0.15% | +0.05% |
| iterations | n20 | 1048576 | 20 | 54.035 | 62.020 | 5.55% | +0.00% | +7.47% |
| iterations | n20 | 67108864 | 20 | 853.985 | 866.160 | 0.54% | +0.00% | +0.66% |
| iterations | n5 | 1048576 | 20 | 61.180 | 65.208 | 5.85% | +13.22% | -0.92% |
| iterations | n5 | 67108864 | 20 | 863.110 | 870.532 | 0.78% | +1.07% | -0.12% |
| noise | idle | 1048576 | 20 | 54.210 | 59.131 | 4.49% | +0.00% | +11.79% |
| noise | idle | 67108864 | 20 | 853.685 | 859.894 | 0.36% | +0.00% | +0.90% |
| noise | remote_numa | 1048576 | 20 | 53.990 | 55.045 | 2.85% | -0.41% | +6.33% |
| noise | remote_numa | 67108864 | 20 | 853.290 | 854.162 | 0.07% | -0.05% | -0.02% |
| noise | same_core | 1048576 | 20 | 59.935 | 584.271 | 143.94% | +10.56% | -2.95% |
| noise | same_core | 67108864 | 20 | 865.610 | 1157.343 | 11.34% | +1.40% | -0.71% |
| warmup | w0 | 1048576 | 40 | 56.155 | 63.172 | 6.75% | +3.65% | +15.96% |
| warmup | w0 | 67108864 | 40 | 855.135 | 869.897 | 0.81% | +0.16% | +1.29% |
| warmup | w1 | 1048576 | 40 | 54.195 | 62.114 | 6.17% | +0.03% | +7.10% |
| warmup | w1 | 67108864 | 40 | 854.220 | 869.198 | 0.79% | +0.05% | +1.61% |
| warmup | w20 | 1048576 | 40 | 54.065 | 61.968 | 5.32% | -0.21% | +5.60% |
| warmup | w20 | 67108864 | 40 | 853.955 | 864.668 | 0.47% | +0.02% | +0.46% |
| warmup | w5 | 1048576 | 40 | 54.180 | 62.327 | 6.46% | +0.00% | +0.26% |
| warmup | w5 | 67108864 | 40 | 853.780 | 868.692 | 0.75% | +0.00% | +0.85% |

## Method assertions

- Warmup=0 still has BenchTime's untimed priming collective.
- Blocking modes intentionally have different synchronization boundaries.
- Per-iteration CUDA event reporting is treated as instrumentation and measured against I=0.
- CPU affinity and noise runs use taskset; affinity variants use mirrored order.
- Exact claims must be rejected for any group whose cycle CV exceeds 5%.
