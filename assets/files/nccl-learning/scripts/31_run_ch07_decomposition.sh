#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/31_run_ch07_decomposition.sh [--run-id ID]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "${1-}" == "--run-id" ]]; then RUN_ID="$2"; shift 2; fi
if [[ $# -ne 0 ]]; then echo "usage: $0 [--run-id ID]" >&2; exit 2; fi
RUN_DIR="${ROOT}/logs/ch07_decomposition/${RUN_ID}"
mkdir -p "${RUN_DIR}/raw/results" "${RUN_DIR}/raw/nccl" "${RUN_DIR}/artifacts"

TRACE_BIN="${RUN_DIR}/artifacts/ch07_trace_decomposition"
TRACE_BYTES=$((64 * 1024 * 1024))
TRACE_WARMUPS=2
TRACE_ITERATIONS=5

{
  echo "experiment=ch07_decomposition"
  echo "run_id=${RUN_ID}"
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "world_size=4"
  echo "sizes_bytes=1024,65536,1048576,16777216,268435456"
  echo "warmups=3"
  echo "cycles=10"
  echo "memory_bytes=268435456"
  echo "trace_bytes=${TRACE_BYTES}"
  echo "trace_warmups=${TRACE_WARMUPS}"
  echo "trace_iterations=${TRACE_ITERATIONS}"
  echo "nccl_source_commit=$(git -C "${ROOT}/third_party/nccl-2.22.3" rev-parse HEAD)"
  echo "torch=$(python3 -c 'import torch; print(torch.__version__)')"
  echo "nsys=$(nsys --version | head -1)"
} > "${RUN_DIR}/manifest.txt"

NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL \
NCCL_DEBUG_FILE="${RUN_DIR}/raw/nccl/nccl_%h_%p.log" \
  torchrun --standalone --nproc_per_node=4 \
  "${ROOT}/scripts/31_ch07_decomposition.py" \
  --output-dir "${RUN_DIR}/raw/results" \
  > "${RUN_DIR}/raw/benchmark.stdout" 2>&1

g++ -O2 -Wall -Wextra -I/usr/local/cuda/include \
  "${ROOT}/probes/ch07_trace_decomposition.cc" \
  -L/usr/local/cuda/lib64 -lcudart -L/lib/x86_64-linux-gnu -lnccl -lnvToolsExt \
  -o "${TRACE_BIN}"

for mode in direct decomposed; do
  nsys profile --force-overwrite=true --sample=none --trace=cuda,nvtx,osrt \
    --output="${RUN_DIR}/artifacts/${mode}" \
    "${TRACE_BIN}" "${mode}" "${TRACE_BYTES}" "${TRACE_WARMUPS}" "${TRACE_ITERATIONS}" \
    > "${RUN_DIR}/raw/nsys_${mode}.stdout" 2>&1
  nsys stats --force-export=true --report=nvtx_kern_sum,cuda_gpu_kern_sum \
    --format=csv "${RUN_DIR}/artifacts/${mode}.nsys-rep" \
    > "${RUN_DIR}/raw/nsys_${mode}_stats.csv" 2>&1
done

python3 "${ROOT}/probes/ch07_parse_nsys.py" \
  --direct "${RUN_DIR}/raw/nsys_direct_stats.csv" \
  --decomposed "${RUN_DIR}/raw/nsys_decomposed_stats.csv" \
  --iterations "${TRACE_ITERATIONS}" --gpus 4 \
  --output "${RUN_DIR}/kernel_summary.csv" \
  > "${RUN_DIR}/raw/nsys_parser.stdout"

python3 "${ROOT}/probes/ch07_summarize.py" --run-dir "${RUN_DIR}" \
  > "${RUN_DIR}/raw/summarizer.stdout"
cat "${RUN_DIR}/summary.md"
echo "run_dir=${RUN_DIR}"
