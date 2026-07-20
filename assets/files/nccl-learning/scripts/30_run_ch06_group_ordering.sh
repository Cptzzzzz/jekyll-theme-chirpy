#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/30_run_ch06_group_ordering.sh [--run-id ID]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "${1-}" == "--run-id" ]]; then RUN_ID="$2"; shift 2; fi
if [[ $# -ne 0 ]]; then echo "usage: $0 [--run-id ID]" >&2; exit 2; fi
RUN_DIR="${ROOT}/logs/ch06_group_ordering/${RUN_ID}"
mkdir -p "${RUN_DIR}/raw/native" "${RUN_DIR}/artifacts"

OMPI_PREFIX=/opt/hpcx/ompi
MPI_LD_LIBRARY_PATH="${OMPI_PREFIX}/lib:/opt/hpcx/ucx/lib:/usr/local/cuda/lib64:/lib/x86_64-linux-gnu"
MPICXX="${OMPI_PREFIX}/bin/mpicxx"
MPIRUN="${OMPI_PREFIX}/bin/mpirun"
NATIVE_BIN="${RUN_DIR}/artifacts/ch06_native_group_semantics"

{
  echo "experiment=ch06_group_ordering"
  echo "run_id=${RUN_ID}"
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "world_size_native=4"
  echo "world_size_failure_matrix=2"
  echo "process_group_timeout_seconds=6"
  echo "large_payload_bytes=$((256 * 1024 * 1024))"
  echo "warmups=3"
  echo "cycles=10"
  echo "nccl_source_commit=$(git -C "${ROOT}/third_party/nccl-2.22.3" rev-parse HEAD)"
  echo "torch=$(python3 -c 'import torch; print(torch.__version__)')"
} > "${RUN_DIR}/manifest.txt"

OPAL_PREFIX="${OMPI_PREFIX}" LD_LIBRARY_PATH="${MPI_LD_LIBRARY_PATH}" \
  "${MPICXX}" -O2 -Wall -Wextra \
  -I/usr/local/cuda/include "${ROOT}/probes/ch06_native_group_semantics.cc" \
  -L/usr/local/cuda/lib64 -lcudart -L/lib/x86_64-linux-gnu -lnccl \
  -o "${NATIVE_BIN}"

OPAL_PREFIX="${OMPI_PREFIX}" LD_LIBRARY_PATH="${MPI_LD_LIBRARY_PATH}" \
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL \
NCCL_DEBUG_FILE="${RUN_DIR}/raw/native/nccl_%h_%p.log" \
  "${MPIRUN}" --allow-run-as-root --bind-to none \
  --mca pml ob1 --mca btl self,vader,tcp --mca coll_hcoll_enable 0 \
  -np 4 "${NATIVE_BIN}" "${RUN_DIR}/raw/native" \
  > "${RUN_DIR}/raw/native.stdout" 2>&1

python3 "${ROOT}/probes/ch06_run_failures.py" \
  --root "${ROOT}" --run-dir "${RUN_DIR}" \
  > "${RUN_DIR}/raw/failure_runner.stdout" 2>&1

python3 "${ROOT}/probes/ch06_summarize.py" --run-dir "${RUN_DIR}" \
  > "${RUN_DIR}/raw/summarizer.stdout"
cat "${RUN_DIR}/summary.md"
echo "run_dir=${RUN_DIR}"
