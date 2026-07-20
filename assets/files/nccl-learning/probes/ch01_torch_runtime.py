#!/usr/bin/env python3
"""Report PyTorch's compile-time NCCL macros and the loaded runtime version."""

import ctypes
from pathlib import Path

import torch


def version_code(version: tuple[int, ...]) -> int:
    major, minor, patch = version[:3]
    if major <= 2 and minor <= 8:
        return major * 1000 + minor * 100 + patch
    return major * 10000 + minor * 100 + patch


compiled_version = tuple(torch.cuda.nccl.version())
runtime_library = ctypes.CDLL("libnccl.so.2")
runtime_code = ctypes.c_int()
status = runtime_library.ncclGetVersion(ctypes.byref(runtime_code))
if status != 0:
    raise RuntimeError(f"ncclGetVersion failed with status {status}")

print(f"torch_version={torch.__version__}")
print(f"torch_git_version={torch.version.git_version}")
print(f"torch_cuda_version={torch.version.cuda}")
print(f"torch_compiled_nccl_version={'.'.join(map(str, compiled_version))}")
print(f"torch_compiled_nccl_version_code={version_code(compiled_version)}")
print(f"ctypes_runtime_nccl_version_code={runtime_code.value}")
print(f"torch_module={torch.__file__}")

mapped = [line.strip() for line in Path("/proc/self/maps").read_text().splitlines() if "libnccl" in line]
print(f"mapped_nccl_lines={len(mapped)}")
for index, line in enumerate(mapped):
    print(f"mapped_nccl_{index}={line}")
