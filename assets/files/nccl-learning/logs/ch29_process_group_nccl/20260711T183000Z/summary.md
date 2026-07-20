# Chapter 29 experiment summary

- Communicator cache rows: 36
- Communicator creation: 12/12 PASS
- Work semantics rows: 160
- Correctness: PASS
- Nsight dedicated stream separation: 4/4 PASS
- Source model: 24/24 PASS

| config | sleep GPU us | action host us | device sync host us | completed after action |
|---|---:|---:|---:|---:|
| device_sync_no_wait | 52062.3 | 0.6 | 52385.1 | 0.00 |
| future_wait_nonblocking | 52065.1 | 5.5 | 52381.3 | 0.00 |
| work_wait_blocking | 52084.6 | 60948.8 | 56.8 | 1.00 |
| work_wait_nonblocking | 53620.8 | 4.9 | 54283.3 | 0.00 |
