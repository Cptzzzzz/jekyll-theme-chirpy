#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/nccl-learning}"
NCCL_TEST_BIN="${NCCL_TEST_BIN:-${ROOT}/third_party/nccl-tests/build/all_reduce_perf}"

echo "== Experiment metadata =="
date -u +%Y-%m-%dT%H:%M:%SZ
hostname
uname -a

echo
echo "== Driver and CUDA toolkit =="
nvidia-smi --query-gpu=driver_version --format=csv,noheader | sort -u
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
  echo "nvcc is not on PATH; using /usr/local/cuda/bin/nvcc"
  /usr/local/cuda/bin/nvcc --version
else
  echo "nvcc: not found"
fi

echo
echo "== PyTorch runtime =="
python3 - <<'PY'
import os
import sys
import torch

print("python_executable:", sys.executable)
print("python_version:", sys.version.replace("\n", " "))
print("torch_version:", torch.__version__)
print("torch_cuda_version:", torch.version.cuda)
print("torch_nccl_version:", ".".join(map(str, torch.cuda.nccl.version())))
print("torch_path:", torch.__file__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
for key in sorted(k for k in os.environ if k.startswith("NCCL_") or k.startswith("TORCH_NCCL_")):
    print(f"env_{key}: {os.environ[key]}")
PY

echo
echo "== Dynamic linker NCCL entries =="
ldconfig -p 2>/dev/null | grep -E 'libnccl(\.so|_net|_wrapper)' || true

echo
echo "== nccl-tests linkage =="
if [[ -x "${NCCL_TEST_BIN}" ]]; then
  echo "binary: ${NCCL_TEST_BIN}"
  ldd "${NCCL_TEST_BIN}" | grep -E 'nccl|cuda|cudart|stdc\+\+' || true
else
  echo "nccl-tests binary not found: ${NCCL_TEST_BIN}"
fi

echo
echo "== Installed NCCL packages =="
dpkg-query -W -f='${Package}\t${Version}\n' 2>/dev/null | grep -E '^libnccl|^nccl' || true

echo
echo "== Source revisions =="
for repo in "${ROOT}/third_party/nccl-2.22.3" "${ROOT}/third_party/nccl" "${ROOT}/third_party/nccl-tests"; do
  if git -C "${repo}" rev-parse --git-dir >/dev/null 2>&1; then
    printf '%s\t' "${repo}"
    git -C "${repo}" log -1 --format='%H %ad %s' --date=iso-strict
  fi
done
