# Chapter 11 experiment summary

- Performance process runs: 14
- Measurement rows: 420
- Out-of-place correctness: PASS
- In-place correctness: PASS

## Performance fingerprints

| variant | ref | bytes | median us | P95 us | CV | time delta | busbw retention |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | baseline | 1024 | 18.900 | 20.310 | 4.03% | +0.00% | 100.00% |
| baseline | baseline | 262144 | 22.210 | 23.057 | 3.05% | +0.00% | 100.00% |
| baseline | baseline | 67108864 | 863.850 | 867.963 | 0.85% | +0.00% | 100.00% |
| channel1 | baseline | 1024 | 18.575 | 19.966 | 4.08% | -1.72% | 100.00% |
| channel1 | baseline | 262144 | 83.950 | 85.149 | 3.49% | +277.98% | 26.46% |
| channel1 | baseline | 67108864 | 4433.115 | 4463.614 | 2.76% | +413.18% | 19.49% |
| no_p2p_numa0 | baseline | 1024 | 19.870 | 23.258 | 5.61% | +5.13% | 100.00% |
| no_p2p_numa0 | baseline | 262144 | 223.745 | 224.380 | 0.18% | +907.41% | 9.94% |
| no_p2p_numa0 | baseline | 67108864 | 39278.350 | 39303.115 | 0.04% | +4446.89% | 2.20% |
| no_p2p_numa1 | no_p2p_numa0 | 1024 | 22.005 | 23.366 | 4.38% | +10.74% | 87.50% |
| no_p2p_numa1 | no_p2p_numa0 | 262144 | 226.400 | 227.726 | 1.22% | +1.19% | 98.86% |
| no_p2p_numa1 | no_p2p_numa0 | 67108864 | 39265.150 | 39298.625 | 0.05% | -0.03% | 100.00% |
| remote_numa | baseline | 1024 | 20.175 | 22.214 | 6.77% | +6.75% | 100.00% |
| remote_numa | baseline | 262144 | 22.635 | 24.694 | 5.08% | +1.91% | 98.14% |
| remote_numa | baseline | 67108864 | 854.735 | 866.687 | 0.84% | -1.06% | 101.07% |
| same_core_noise | single_core_idle | 1024 | 19.315 | 35.607 | 49.33% | +0.52% | 100.00% |
| same_core_noise | single_core_idle | 262144 | 21.190 | 460.376 | 208.99% | -1.19% | 101.20% |
| same_core_noise | single_core_idle | 67108864 | 851.295 | 1207.086 | 14.00% | +0.04% | 99.96% |
| single_core_idle | single_core_idle | 1024 | 19.215 | 21.117 | 5.73% | +0.00% | 100.00% |
| single_core_idle | single_core_idle | 262144 | 21.445 | 22.837 | 3.87% | +0.00% | 100.00% |
| single_core_idle | single_core_idle | 67108864 | 850.975 | 859.174 | 2.06% | +0.00% | 100.00% |

## Log evidence

| variant | coll channels | P2P | SHM | selections |
|---|---|---|---|---|
| baseline | 12 | True | False | 1024:Ring+LL;262144:Ring+LL;67108864:Ring+Simple |
| channel1 | 1 | True | False | 1024:Ring+LL;262144:Ring+LL;67108864:Ring+Simple |
| no_p2p_numa0 | 2 | False | True | 1024:Tree+LL;262144:Tree+Simple;67108864:Tree+Simple |
| no_p2p_numa1 | 2 | False | True | 1024:Tree+LL;262144:Tree+Simple;67108864:Tree+Simple |
| remote_numa | 12 | True | False | 1024:Ring+LL;262144:Ring+LL;67108864:Ring+Simple |
| single_core_idle | 12 | True | False | 1024:Ring+LL;262144:Ring+LL;67108864:Ring+Simple |
| same_core_noise | 12 | True | False | 1024:Ring+LL;262144:Ring+LL;67108864:Ring+Simple |

## Attribution rules

- CV above 5% rejects an exact slowdown percentage.
- A path override may also change graph/channel/tuning; evidence must report every changed layer.
- same_core_noise is compared with single_core_idle, not the four-core baseline.
- no_p2p_numa1 is compared with no_p2p_numa0 to isolate CPU/SHM placement within the fallback family.
