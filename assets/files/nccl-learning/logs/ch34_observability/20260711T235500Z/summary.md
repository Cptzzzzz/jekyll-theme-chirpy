# Chapter 34 experiment summary

- Measurements: 18
- NCCL debug evidence: 32
- Flight-recorder AllReduce rows: 12
- Nsight kernels: 6; NVTX ranges: 6
- Source model: 32/32 PASS

| bytes | median GPU us |
|---:|---:|
| 4096 | 168.6 |
| 1048576 | 226.0 |
| 67108864 | 1967.9 |
