# Chapter 31 experiment summary

- Iteration rows: 480
- Rank state rows: 48
- Correctness and numeric equivalence: PASS
- State accounting: 6/6 PASS
- Nsight collective paths: 24/24 PASS
- Source model: 30/30 PASS
- DeepSpeed runtime: NOT INSTALLED; fixed source only

| config | step us | CV | steady MiB | peak MiB |
|---|---:|---:|---:|---:|
| zero0_ddp | 8319.6 | 1.96% | 528.2 | 784.5 |
| zero1_zro | 7353.9 | 2.68% | 336.2 | 496.5 |
| zero2_fsdp | 18737.9 | 9.71% | 144.2 | 273.5 |
| zero3_none | 23123.3 | 11.51% | 144.2 | 188.5 |
| zero3_post | 20583.3 | 6.92% | 144.2 | 188.5 |
| zero3_pre | 20573.1 | 10.84% | 144.2 | 200.8 |
