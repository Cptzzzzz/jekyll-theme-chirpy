# Chapter 18 experiment summary

- Performance rows: 300
- Out-of-place/in-place correctness: PASS
- Path/graph acceptance: 5/5 PASS

| fixture | Ring channels | graph speed | graph type | 64 MiB median us | busbw | vs native |
|---|---:|---:|---|---:|---:|---:|
| native_nv2 | 6 | 20.0 | NVL | 862.815 | 116.66 | +0.00% |
| all_to_all_nv1 | 6 | 10.0 | NVL | 850.185 | 118.40 | -1.46% |
| sparse_nv2_ring | 4 | 20.0 | NVL | 1205.090 | 83.53 | +39.67% |
| pcie_x16 | 1 | 12.0 | PIX | 2854.955 | 35.26 | +230.89% |
| pcie_x4 | 1 | 3.0 | PIX | 2852.815 | 35.28 | +230.64% |
