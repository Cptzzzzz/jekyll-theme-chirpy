# Chapter 04 experiment summary

## Native communicator and split

| global rank | CUDA device | PCI BDF | color | key | split rank | world sum | split sum |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 0 | 0000:1A:00.0 | 0 | 4 | 1 | 10.000 | 4.000 |
| 1 | 1 | 0000:1C:00.0 | 1 | 3 | 1 | 10.000 | 6.000 |
| 2 | 2 | 0000:1D:00.0 | 0 | 2 | 0 | 10.000 | 4.000 |
| 3 | 3 | 0000:1E:00.0 | 1 | 1 | 0 | 10.000 | 6.000 |

## CUDA_VISIBLE_DEVICES mapping

| rank/local ordinal | default PCI BDF | remapped PCI BDF |
|---:|---|---|
| 0 | 00000000:1a:00.0 | 00000000:1d:00.0 |
| 1 | 00000000:1c:00.0 | 00000000:1a:00.0 |
| 2 | 00000000:1d:00.0 | 00000000:1e:00.0 |
| 3 | 00000000:1e:00.0 | 00000000:1c:00.0 |

## Assertions

- All native ranks received one identical ncclUniqueId payload.
- Parent communicator rank equals global rank; split rank follows key ordering within color.
- World, parity and pair ProcessGroups produced independent expected sums.
- CUDA_VISIBLE_DEVICES changed every rank's physical BDF without changing its local ordinal.
- Binding two ranks to one visible GPU produced the expected duplicate-GPU failure.

- Default NCCL Init COMPLETE counts: `{2: 8, 4: 4}`
- Remapped NCCL Init COMPLETE counts: `{2: 8, 4: 4}`
