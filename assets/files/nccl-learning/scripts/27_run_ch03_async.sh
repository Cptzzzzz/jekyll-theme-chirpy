#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/27_run_ch03_async.sh [--run-id ID]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "${1-}" == "--run-id" ]]; then RUN_ID="$2"; shift 2; fi
if [[ $# -ne 0 ]]; then echo "usage: $0 [--run-id ID]" >&2; exit 2; fi

RUN_DIR="${ROOT}/logs/ch03_async/${RUN_ID}"
mkdir -p "${RUN_DIR}/raw/nonblocking" "${RUN_DIR}/raw/blocking" "${RUN_DIR}/raw/stream_scope"

{
  echo "experiment=ch03_async"
  echo "run_id=${RUN_ID}"
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "nccl_source_commit=$(git -C "${ROOT}/third_party/nccl-2.22.3" rev-parse HEAD)"
  echo "torch=$(python3 -c 'import torch; print(torch.__version__)')"
  echo "world_size=4"
  echo "sizes_kib=1,1024,65536,262144"
  echo "cycles=10"
} > "${RUN_DIR}/manifest.txt"

NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL TORCH_NCCL_BLOCKING_WAIT=0 \
  torchrun --standalone --nproc_per_node=4 \
  "${ROOT}/scripts/27_ch03_async_semantics.py" \
  --mode timing --cycles 10 --sizes-kib 1,1024,65536,262144 \
  --output-dir "${RUN_DIR}/raw/nonblocking" \
  > "${RUN_DIR}/raw/nonblocking.stdout" 2>&1

NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL TORCH_NCCL_BLOCKING_WAIT=1 \
  torchrun --standalone --nproc_per_node=4 \
  "${ROOT}/scripts/27_ch03_async_semantics.py" \
  --mode timing --cycles 10 --sizes-kib 1,1024,65536,262144 \
  --output-dir "${RUN_DIR}/raw/blocking" \
  > "${RUN_DIR}/raw/blocking.stdout" 2>&1

NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL TORCH_NCCL_BLOCKING_WAIT=0 \
  torchrun --standalone --nproc_per_node=4 \
  "${ROOT}/scripts/27_ch03_async_semantics.py" \
  --mode stream-scope --cycles 10 --bytes-mib 64 --sleep-cycles 200000000 \
  --output-dir "${RUN_DIR}/raw/stream_scope" \
  > "${RUN_DIR}/raw/stream_scope.stdout" 2>&1

python3 "${ROOT}/probes/ch03_summarize.py" --run-dir "${RUN_DIR}" > "${RUN_DIR}/raw/summarizer.stdout"
cat "${RUN_DIR}/summary.md"
echo "run_dir=${RUN_DIR}"
