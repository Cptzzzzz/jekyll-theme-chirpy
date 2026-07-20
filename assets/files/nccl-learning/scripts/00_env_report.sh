#!/usr/bin/env bash
set -euo pipefail

echo "== System =="
date -u
hostname
uname -a

echo
echo "== NVIDIA GPUs =="
nvidia-smi -L
nvidia-smi

echo
echo "== GPU topology =="
nvidia-smi topo -m

echo
echo "== CUDA toolkit =="
command -v nvcc || true
/usr/local/cuda/bin/nvcc --version 2>/dev/null || true
ls -ld /usr/local/cuda* 2>/dev/null || true

echo
echo "== PyTorch / CUDA / NCCL =="
python3 - <<'PY'
import importlib.util

print("python: available")
if importlib.util.find_spec("torch") is None:
    print("torch: not installed")
else:
    import torch
    print("torch:", torch.__version__)
    print("torch cuda available:", torch.cuda.is_available())
    print("torch cuda version:", torch.version.cuda)
    print("torch nccl version:", torch.cuda.nccl.version())
    print("cuda device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"cuda:{i}:", torch.cuda.get_device_name(i))
PY

echo
echo "== NCCL shared libraries visible to linker =="
ldconfig -p 2>/dev/null | grep -i 'libnccl' || true
find /usr /usr/local -name 'libnccl.so*' 2>/dev/null | sort || true

echo
echo "== RDMA / InfiniBand probes =="
command -v ibv_devinfo || true
command -v ibstat || true
command -v rdma || true
ls -la /dev/infiniband 2>/dev/null || true
ibv_devinfo -l 2>/dev/null || true
