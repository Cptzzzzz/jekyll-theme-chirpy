# Foundations Experiment Summary

## Version Matrix

- PyTorch: `2.5.0a0+872d972e41.nv24.08`
- CUDA used by PyTorch: `12.6`
- NCCL used by PyTorch: `2.22.3`

## GPU P2P Matrix

- Directed GPU pairs: `12`
- Median bandwidth range: `48.31 - 48.37 GB/s`
- Maximum CV across pairs: `0.06%`
- Every measured pair reported CUDA peer access and passed copy correctness.

## Asynchronous Completion

- Mean host enqueue time: `122.84 us`
- Mean GPU completion time: `3634.49 us`
- GPU completion / host enqueue ratio: `29.59x`
- `work.wait()` established stream ordering, while final CUDA event synchronization observed device completion.

## Communicator and Collective Correctness

- Subgroup experiment: `PASS`
- Seven-operation collective reference checks: `PASS`

## Conclusion

The four V100 GPUs form a uniform CUDA-P2P-capable NVLink domain. A process group is a collective communication domain rather than merely a rank list, and host-side NCCL submission must not be confused with GPU-side completion.
