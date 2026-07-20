# Chapter 19 experiment summary

- Performance rows: 360
- Correctness: PASS
- Search/replay/fallback acceptance: 7/7 PASS
- Invalid graph failure: PASS

| config | Ring base | Tree base | final coll | fallback | first ring | 64 MiB us | busbw | vs native |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| search_native | 6 | 6 | 12 | 0 | 0-1-2-3 | 859.840 | 117.07 | +0.00% |
| replay_exact | 6 | 6 | 12 | 0 | 0-1-2-3 | 850.740 | 118.32 | -1.06% |
| replay_custom_one | 1 | 1 | 2 | 0 | 0-2-1-3 | 2554.540 | 39.41 | +197.09% |
| replay_ring_only | 1 | 1 | 2 | 0 | 0-2-1-3 | n/a | n/a | n/a |
| weak_fallback | 1 | 1 | 2 | 1 | 0-1-2-3 | 2852.990 | 35.28 | +231.80% |
| postset_max4 | 6 | 6 | 4 | 0 | 0-1-2-3 | 1329.480 | 75.72 | +54.62% |
| postset_min16 | 6 | 6 | 16 | 0 | 0-1-2-3 | 870.535 | 115.64 | +1.24% |
