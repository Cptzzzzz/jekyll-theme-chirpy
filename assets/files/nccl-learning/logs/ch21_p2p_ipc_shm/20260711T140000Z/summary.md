# Chapter 21 experiment summary

- Path acceptance: 12/12 PASS
- Performance rows: 600
- Pair rows: 120
- Correctness: PASS

| config | variant | 64 MiB us | busbw GB/s | vs direct |
|---|---|---:|---:|---:|
| direct_write | P2P/direct pointer | 861.455 | 116.85 | +0.00% |
| direct_read | P2P/direct pointer/read | 969.160 | 103.87 | +12.50% |
| cumem | P2P/CUMEM | 903.965 | 111.35 | +4.93% |
| legacy_ipc | P2P/IPC | 902.745 | 111.50 | +4.79% |
| p2p_ce | P2P/CUMEM/CE | 1644.435 | 61.22 | +90.89% |
| shm_recv_local | SHM/direct/direct | 39269.700 | 2.56 | +4458.53% |
| shm_send_local | SHM/direct/direct | 39285.050 | 2.56 | +4460.31% |
| shm_ce_send | SHM/CE/direct | 41759.050 | 2.41 | +4747.50% |
| shm_ce_recv | SHM/direct/CE | 44718.600 | 2.25 | +5091.05% |
| shm_ce_both | SHM/CE/CE | 43854.800 | 2.30 | +4990.78% |
