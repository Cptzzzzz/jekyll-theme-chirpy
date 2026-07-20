#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/29_run_ch05_contracts.sh [--run-id ID]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "${1-}" == "--run-id" ]]; then RUN_ID="$2"; shift 2; fi
if [[ $# -ne 0 ]]; then echo "usage: $0 [--run-id ID]" >&2; exit 2; fi
RUN_DIR="${ROOT}/logs/ch05_collective_contracts/${RUN_ID}"
mkdir -p "${RUN_DIR}/raw/results" "${RUN_DIR}/raw/native" "${RUN_DIR}/artifacts"

OMPI_PREFIX=/opt/hpcx/ompi
MPI_LD_LIBRARY_PATH="${OMPI_PREFIX}/lib:/opt/hpcx/ucx/lib:/usr/local/cuda/lib64:/lib/x86_64-linux-gnu"
MPICXX="${OMPI_PREFIX}/bin/mpicxx"
MPIRUN="${OMPI_PREFIX}/bin/mpirun"
NATIVE_BIN="${RUN_DIR}/artifacts/ch05_native_collective_contracts"

{
  echo "experiment=ch05_collective_contracts"
  echo "run_id=${RUN_ID}"
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "world_size=4"
  echo "nccl_source_commit=$(git -C "${ROOT}/third_party/nccl-2.22.3" rev-parse HEAD)"
  echo "torch=$(python3 -c 'import torch; print(torch.__version__)')"
  echo "mpi_prefix=${OMPI_PREFIX}"
} > "${RUN_DIR}/manifest.txt"

OPAL_PREFIX="${OMPI_PREFIX}" LD_LIBRARY_PATH="${MPI_LD_LIBRARY_PATH}" \
  "${MPICXX}" -O2 -Wall -Wextra \
  -I/usr/local/cuda/include "${ROOT}/probes/ch05_native_collective_contracts.cc" \
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

NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL,P2P \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
  torchrun --standalone --nproc_per_node=4 \
  "${ROOT}/scripts/29_ch05_collective_contracts.py" \
  --output-dir "${RUN_DIR}/raw/results" \
  > "${RUN_DIR}/raw/torchrun.stdout" 2>&1

python3 "${ROOT}/probes/ch05_summarize.py" --run-dir "${RUN_DIR}" > "${RUN_DIR}/raw/summarizer.stdout"
cat "${RUN_DIR}/summary.md"
echo "run_dir=${RUN_DIR}"
