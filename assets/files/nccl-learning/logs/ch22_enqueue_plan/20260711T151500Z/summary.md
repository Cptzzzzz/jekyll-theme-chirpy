# Chapter 22 experiment summary

- Measurements: 190
- Correctness: PASS
- Nsight acceptance: 3/3 PASS

- sequential4 median total: 289.513 us
- grouped4 median total: 236.562 us
- grouped4 vs sequential4: -18.29%

| mode | GPU ops in NVTX range | gridX | blockX |
|---|---:|---|---|
| sequential | 16 | 12 | 640 |
| grouped | 4 | 12 | 544 |
| graph | 20 | 12 | 544 |
