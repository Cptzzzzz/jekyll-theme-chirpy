#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/nccl-learning}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-${ROOT}/logs/nsys_allreduce/${RUN_ID}}"
RAW_DIR="${OUT_DIR}/raw"
ARTIFACT_DIR="${OUT_DIR}/artifacts"
BIN="${ROOT}/third_party/nccl-tests/build/all_reduce_perf"
mkdir -p "${RAW_DIR}" "${ARTIFACT_DIR}"

{
  echo "run_id=${RUN_ID}"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "command=nsys profile --trace=cuda,nvtx,osrt ${BIN} -b 64M -e 64M -g 4 -w 2 -n 5 -c 1"
} > "${OUT_DIR}/manifest.txt"

nsys profile --force-overwrite=true --sample=none --trace=cuda,nvtx,osrt \
  --output="${ARTIFACT_DIR}/allreduce_64m" \
  "${BIN}" -b 64M -e 64M -g 4 -w 2 -n 5 -c 1 \
  |& tee "${RAW_DIR}/profile.log"

nsys stats --force-export=true \
  --report=cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum \
  "${ARTIFACT_DIR}/allreduce_64m.nsys-rep" \
  |& tee "${OUT_DIR}/stats.txt"

echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT_DIR}/manifest.txt"
echo "out_dir=${OUT_DIR}"
