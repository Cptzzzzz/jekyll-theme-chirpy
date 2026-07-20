#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/nccl-learning}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-${ROOT}/logs/parallel_patterns/${RUN_ID}}"
mkdir -p "${OUT_DIR}/raw"

{
  echo "run_id=${RUN_ID}"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "sizes_mib=1,16,64"
  echo "warmup=5"
  echo "iterations=20"
} > "${OUT_DIR}/manifest.txt"

torchrun --standalone --nproc_per_node=4 "${ROOT}/scripts/24_parallel_patterns.py" \
  --sizes-mib 1,16,64 --warmup 5 --iters 20 --out-dir "${OUT_DIR}" \
  |& tee "${OUT_DIR}/raw/run.log"

echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT_DIR}/manifest.txt"
echo "out_dir=${OUT_DIR}"
