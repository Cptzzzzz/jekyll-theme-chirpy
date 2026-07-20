# Chapter 32 experiment summary

- Iteration rows: 336
- Communication operation rows: 2112
- Pipeline timeline rows: 864
- Numeric reference and DP replica equality: PASS
- Source model: 39/39 PASS
- Nsight paths: 20/20 PASS

| config | critical GPU us | critical host us | host CV |
|---|---:|---:|---:|---:|
| ep_balanced | 2777.4 | 2850.1 | 15.33% |
| ep_skewed | 2581.7 | 2625.6 | 18.55% |
| pp_mb1 | 1111.3 | 1139.6 | 12.62% |
| pp_mb8 | 5906.7 | 6968.7 | 51.60% |
| tp_ar_b128 | 2139.6 | 2276.1 | 80.95% |
| tp_ar_b32 | 1802.4 | 2020.8 | 54.95% |
| tp_sp_b128 | 2827.5 | 3000.9 | 48.82% |
