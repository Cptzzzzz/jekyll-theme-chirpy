# NCCL Interview Detail Experiments

Run ID: `20260717T090915Z`

All 784 per-rank contracts passed; failures: 0.
Critical-path samples use the slowest rank in each cycle.

## Async completion

| bytes | host enqueue us | Work.wait host us | GPU dependency us | event ready after wait | CV |
|---:|---:|---:|---:|---:|---:|
| 4096 | 73.322 | 8.539 | 103.152 | 0.0% | 16.94% |
| 1048576 | 70.806 | 7.617 | 110.432 | 0.0% | 10.98% |
| 67108864 | 70.898 | 7.841 | 990.240 | 0.0% | 37.13% |

## AllReduce versus explicit ReduceScatter plus AllGather

| bytes | mode | median us | overhead vs AllReduce | CV |
|---:|---|---:|---:|---:|
| 4096 | direct | 88.400 | 0.00% | 5.61% |
| 4096 | rs_ag | 132.896 | 50.33% | 6.00% |
| 524288 | direct | 87.808 | 0.00% | 6.60% |
| 524288 | rs_ag | 141.232 | 60.84% | 10.24% |
| 67108864 | direct | 1002.192 | 0.00% | 0.54% |
| 67108864 | rs_ag | 1096.864 | 9.45% | 0.87% |

## Same 80 MiB logical payload split into multiple collectives

| calls | bytes/call | median total us | slowdown vs one call | per call us | CV |
|---:|---:|---:|---:|---:|---:|
| 1 | 83886080 | 1201.568 | 1.00x | 1201.568 | 0.67% |
| 10 | 8388608 | 1727.120 | 1.44x | 172.712 | 37.61% |
| 40 | 2097152 | 3590.240 | 2.99x | 89.756 | 12.15% |
| 160 | 524288 | 7446.576 | 6.20x | 46.541 | 17.46% |

## Decode communication-only replay

| TP | tokens | bytes/call | calls | median total us | per call us | CV |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | 8192 | 160 | 6317.792 | 39.486 | 20.98% |
| 2 | 8 | 65536 | 160 | 6420.080 | 40.125 | 21.91% |
| 2 | 32 | 262144 | 160 | 6568.544 | 41.053 | 24.87% |
| 2 | 128 | 1048576 | 160 | 8979.056 | 56.119 | 11.80% |
| 4 | 1 | 8192 | 160 | 7603.232 | 47.520 | 15.50% |
| 4 | 8 | 65536 | 160 | 7703.520 | 48.147 | 27.59% |
| 4 | 32 | 262144 | 160 | 7469.136 | 46.682 | 22.15% |
| 4 | 128 | 1048576 | 160 | 8469.200 | 52.933 | 18.28% |

## nccl-tests control for Decode-sized messages

| ranks | bytes | median us | P95 us | algbw GB/s | busbw GB/s | CV | wrong |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 8192 | 16.530 | 17.510 | 0.495 | 0.495 | 3.17% | 0 |
| 2 | 65536 | 17.130 | 18.300 | 3.825 | 3.825 | 3.50% | 0 |
| 2 | 262144 | 23.065 | 23.620 | 11.365 | 11.365 | 0.74% | 0 |
| 2 | 1048576 | 49.295 | 49.370 | 21.270 | 21.270 | 0.06% | 0 |
| 4 | 8192 | 32.710 | 33.510 | 0.250 | 0.375 | 2.76% | 0 |
| 4 | 65536 | 33.175 | 35.150 | 1.975 | 2.965 | 2.31% | 0 |
| 4 | 262144 | 34.050 | 39.970 | 7.695 | 11.550 | 5.69% | 0 |
| 4 | 1048576 | 59.580 | 59.650 | 17.595 | 26.400 | 0.16% | 0 |

## FP16 reduction counterexamples

| scenario | permutations | FP16 mismatches | FP16 nonfinite | FP32 mismatches | FP16 outputs |
|---|---:|---:|---:|---:|---|
| finite-overflow | 24 | 8 | 8 | 0 | -inf/0.0/inf |
| lost-unit-10000 | 24 | 16 | 0 | 0 | 0.0/1.0 |
| lost-unit-2048 | 24 | 8 | 0 | 0 | 0.0/1.0 |

## Evidence boundary

- These are single-node 4x V100 NV2 results for NCCL 2.22.3.
- Decode replay contains communication only; it excludes GEMM, scheduling, CUDA Graph, and Custom AllReduce.
- FP16 counterexamples prove that callers cannot assume implicit FP32 accumulation; they do not specify every GPU architecture's internal instruction sequence.
- Exact performance percentages are valid only for the recorded hardware, software, and message matrix.
