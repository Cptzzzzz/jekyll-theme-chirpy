#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/nccl-learning}
REPO=${REPO:-$ROOT/third_party/nccl-tests}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

if [[ ! -d "$REPO/.git" ]]; then
  git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git "$REPO"
fi

make -C "$REPO" -j"$(nproc)" CUDA_HOME="$CUDA_HOME"

echo "Built nccl-tests under $REPO/build"
ls -1 "$REPO"/build/*_perf
