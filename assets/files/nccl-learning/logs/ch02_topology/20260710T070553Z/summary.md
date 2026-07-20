# Chapter 02 experiment summary

- GPUs: `4`
- Directed NVLink XML entries: `12`
- NVLinks per GPU pair: `2`
- `nvidia-smi` per-link status: `25.781 GB/s`
- Reported two-link pair capacity: `51.562 GB/s`
- 256 MiB measured median: `48.334 GB/s`
- Payload/capacity ratio: `93.74%`

## Size curve

| bytes | median us | median GB/s | pair min | pair max | max CV | max reverse asymmetry |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 17.380 | 0.059 | 0.058 | 0.060 | 3.10% | 2.90% |
| 65536 | 17.785 | 3.685 | 3.641 | 3.723 | 1.99% | 1.91% |
| 1048576 | 39.041 | 26.858 | 26.133 | 27.223 | 0.94% | 2.49% |
| 16777216 | 359.289 | 46.696 | 46.645 | 46.713 | 0.16% | 0.14% |
| 268435456 | 5553.777 | 48.334 | 48.322 | 48.372 | 0.05% | 0.10% |

## Concurrent patterns

| CPU NUMA | pattern | edges | median aggregate GB/s | CV |
|---:|---|---:|---:|---:|
| 0 | bidirectional_0_1 | 2 | 48.372 | 0.03% |
| 0 | disjoint_pairs | 2 | 96.583 | 0.13% |
| 0 | fanin_to_0 | 3 | 48.403 | 0.06% |
| 0 | ring_4_edges | 4 | 48.356 | 0.05% |
| 0 | single_0_to_1 | 1 | 48.310 | 0.01% |
| 1 | bidirectional_0_1 | 2 | 48.356 | 0.04% |
| 1 | disjoint_pairs | 2 | 96.590 | 0.16% |
| 1 | fanin_to_0 | 3 | 48.413 | 0.07% |
| 1 | ring_4_edges | 4 | 48.363 | 0.06% |
| 1 | single_0_to_1 | 1 | 48.310 | 0.02% |

## Assertions

- All 12 directed pairs support CUDA peer access and passed data checks.
- All matrix and concurrent outputs passed full-tensor correctness checks.
- Topology XML contains two direct NVLinks for every directed GPU pair.
- NUMA results measure host submission affinity, not a change in the NVLink data path.
