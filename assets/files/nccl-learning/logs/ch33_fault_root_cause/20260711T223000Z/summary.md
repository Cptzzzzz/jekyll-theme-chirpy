# Chapter 33 experiment summary

- Cases: 9/9 PASS
- Causal event rows: 80
- Source model: 37/37 PASS
- Real network interruption: BLOCKED (single node, no HCA)

| case | root cause | duration s | events |
|---|---|---:|---:|
| control | success | 4.952 | 8 |
| op_detail | fingerprint | 4.998 | 10 |
| shape_detail | fingerprint | 4.886 | 10 |
| order_detail | fingerprint | 4.907 | 10 |
| op_raw | raw_mismatch | 9.226 | 11 |
| steady_skip | skip_timeout | 8.991 | 10 |
| lazy_skip | lazy_init_external_timeout | 16.459 | 4 |
| rank_exit | rank_exit | 4.293 | 8 |
| peer_exception | application_exception | 4.815 | 9 |
