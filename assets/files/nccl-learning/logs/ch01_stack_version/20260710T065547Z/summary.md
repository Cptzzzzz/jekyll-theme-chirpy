# Chapter 01 experiment summary

- Run ID: `20260710T065547Z`
- Header version code: `22203`
- System runtime: `2.22.3` (`22203`)
- PyTorch compile-time NCCL macros: `2.22.3` (`22203`)
- NCCL runtime loaded in the PyTorch process: `22203`
- System object reported by `dladdr`: `/lib/x86_64-linux-gnu/libnccl.so.2`
- Isolated override runtime: `2.99.99` (`29999`)
- Override object reported by `dladdr`: `/root/nccl-learning/logs/ch01_stack_version/20260710T065547Z/artifacts/fake/libnccl.so.2`

## Assertions

- System header and PyTorch compile-time macros agree.
- Native probe and the NCCL object loaded in the PyTorch process agree.
- The probe ELF requests the SONAME `libnccl.so.2`, not an absolute file.
- `LD_LIBRARY_PATH` can satisfy that SONAME with a different object.
- `dladdr`, loader trace and `/proc/<pid>/maps` independently identify the loaded object.
