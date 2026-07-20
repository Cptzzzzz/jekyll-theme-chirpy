# NCCL Topology and Graph Summary

Run ID: `20260710T014639Z`

## Detected Topology

- Four V100 GPUs share one PCIe switch under NUMA node 0.
- Every GPU has `count=2` NVLink connections to each of the other three GPUs.
- NCCL assigns each NVLink GPU path `40.0 GB/s` in its internal topology model.
- `eth0` is represented as a 10 Gbit/s network device with `gdr=0`.
- NVLS multicast is unavailable on every V100 device.

## Graph Search

- Ring graph: pattern `4`, six base channels, `typeintra=NVL`, `crossnic=0`.
- Balanced-tree graph: pattern `1`, six base channels, `typeintra=NVL`, `crossnic=0`.
- NVLS and CollNet graphs have zero channels on this machine.
- Communicator initialization duplicates/expands the graph to 12 collective channels and 16 P2P channels.

The six base Ring channels enumerate all six permutations that start at GPU 0:

```text
0-1-2-3
0-1-3-2
0-2-3-1
0-2-1-3
0-3-1-2
0-3-2-1
```

Every connected channel reports `via P2P/direct pointer`; no SHM data path is selected.

## Initialization Timing

Each rank reports approximately:

```text
total        0.64 s
kernels      0.45 s
bootstrap    0.13 s
topology     0.02 s
connections  0.02 s
```

## Performance Check

The 64 MiB AllReduce completed with `#wrong=0` and approximately `116.18 GB/s` out-of-place bus bandwidth. This run validates that the exported graph is usable, but it does not prove that every workload will use the same algorithm or protocol.
