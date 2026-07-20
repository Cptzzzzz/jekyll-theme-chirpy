# Chapter 36 Capstone summary

- Prediction acceptance: 8/8 PASS
- Call graph model: 35/35 PASS
- Nsight AllReduce kernels: 4/4 PASS
- Replica equality: PASS
- Channel1 step regression: 245.93%

| variant | bucket | algo/proto | channels | step GPU us |
|---|---:|---|---:|---:|
| baseline | 67108864 | 1/2 | 12 | 2330.0 |
| channel1 | 67108864 | 1/2 | 1 | 8060.2 |
