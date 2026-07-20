# Chapter 30 experiment summary

- Iteration rows: 320
- Bucket-ready rows: 1960
- Correctness and replica equality: PASS
- Bucket layouts: 8/8 PASS
- Unused-parameter expected failure: PASS
- Nsight overlap: 8/8 PASS
- Source model: 30/30 PASS

| config | buckets | step GPU us | backward GPU us | logger overlap us |
|---|---:|---:|---:|---:|
| coarse4_cap16 | 3 | 4120.5 | 2658.0 | 2624.8 |
| coarse4_cap4 | 12 | 5600.6 | 3727.6 | 3966.3 |
| coarse4_cap64 | 1 | 4336.2 | 3042.5 | 1603.8 |
| delay_cap4 | 12 | 81121.9 | 79612.7 | 62946.0 |
| delay_cap64 | 1 | 81839.0 | 80161.2 | 188.1 |
| dynamic_unused | 4 | 4133.6 | 2382.8 | 2780.8 |
| fine1_cap4 | 12 | 8890.2 | 5060.6 | 5955.1 |
| static_unused | 4 | 4580.3 | 3178.9 | 1325.5 |
