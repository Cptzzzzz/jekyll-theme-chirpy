# Chapter 20 experiment summary

- Path/correctness acceptance: 8/8 PASS
- No-transport negative case: PASS
- Performance measurements: 240

| config | transport | 64 MiB median us | busbw GB/s | vs baseline |
|---|---|---:|---:|---:|
| baseline | P2P | 857.465 | 117.41 | +0.00% |
| shm_disabled | P2P | 851.030 | 118.28 | -0.75% |
| p2p_disabled | SHM | 39278.850 | 2.56 | +4480.81% |
| p2p_shm_disabled | NET | 47161.000 | 2.13 | +5400.05% |
