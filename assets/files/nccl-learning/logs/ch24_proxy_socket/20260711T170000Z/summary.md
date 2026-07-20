# Chapter 24 experiment summary

- Performance rows: 150
- Delay rows: 20
- Affinity rows: 15
- Correctness: PASS
- Forced transport: NET/Socket PASS

| delay us/call | median total us |
|---:|---:|
| 0 | 201828.578 |
| 20 | 201909.748 |
| 100 | 200712.107 |
| 1000 | 620215.938 |

| affinity case | median total us |
|---|---:|
| proxy_cpu0 | 3834233.987 |
| proxy_cpu0_contended | 3512841.103 |
| wide | 201863.031 |
