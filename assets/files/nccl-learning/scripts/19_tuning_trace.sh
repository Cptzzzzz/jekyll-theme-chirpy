#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/nccl-learning}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-${ROOT}/logs/tuning_trace/${RUN_ID}}"
RAW_DIR="${OUT_DIR}/raw"
BIN="${ROOT}/third_party/nccl-tests/build/all_reduce_perf"
mkdir -p "${RAW_DIR}"

{
  echo "run_id=${RUN_ID}"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "command=NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING ${BIN} -b 1K -e 256M -f 4 -g 4 -w 0 -n 1 -c 1"
} > "${OUT_DIR}/manifest.txt"

NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING \
  "${BIN}" -b 1K -e 256M -f 4 -g 4 -w 0 -n 1 -c 1 \
  |& tee "${RAW_DIR}/tuning.log"

grep -E 'Bytes -> Algo|threadThresholds|#wrong|Out of bounds|Avg bus bandwidth' \
  "${RAW_DIR}/tuning.log" > "${OUT_DIR}/selection.txt" || true
echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT_DIR}/manifest.txt"
echo "out_dir=${OUT_DIR}"
