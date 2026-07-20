#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/28_run_ch04_communicators.sh [--run-id ID]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "${1-}" == "--run-id" ]]; then RUN_ID="$2"; shift 2; fi
if [[ $# -ne 0 ]]; then echo "usage: $0 [--run-id ID]" >&2; exit 2; fi

RUN_DIR="${ROOT}/logs/ch04_communicators/${RUN_ID}"
mkdir -p "${RUN_DIR}/raw/native" "${RUN_DIR}/raw/pytorch_default" "${RUN_DIR}/raw/pytorch_remap" "${RUN_DIR}/artifacts"

OMPI_PREFIX=/opt/hpcx/ompi
MPI_LD_LIBRARY_PATH="${OMPI_PREFIX}/lib:/opt/hpcx/ucx/lib:/usr/local/cuda/lib64:/lib/x86_64-linux-gnu"
MPICXX="${OMPI_PREFIX}/bin/mpicxx"
MPIRUN="${OMPI_PREFIX}/bin/mpirun"
NATIVE_BIN="${RUN_DIR}/artifacts/ch04_native_communicator"

{
  echo "experiment=ch04_communicators"
  echo "run_id=${RUN_ID}"
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "nccl_source_commit=$(git -C "${ROOT}/third_party/nccl-2.22.3" rev-parse HEAD)"
  echo "torch=$(python3 -c 'import torch; print(torch.__version__)')"
  echo "mpi_prefix=${OMPI_PREFIX}"
  echo "default_cuda_visible_devices=0,1,2,3"
  echo "remapped_cuda_visible_devices=2,0,3,1"
} > "${RUN_DIR}/manifest.txt"

OPAL_PREFIX="${OMPI_PREFIX}" LD_LIBRARY_PATH="${MPI_LD_LIBRARY_PATH}" \
  "${MPICXX}" -O2 -Wall -Wextra \
  -I/usr/local/cuda/include "${ROOT}/probes/ch04_native_communicator.cc" \
  -L/usr/local/cuda/lib64 -lcudart -L/lib/x86_64-linux-gnu -lnccl \
  -o "${NATIVE_BIN}"

OPAL_PREFIX="${OMPI_PREFIX}" LD_LIBRARY_PATH="${MPI_LD_LIBRARY_PATH}" \
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL \
NCCL_DEBUG_FILE="${RUN_DIR}/raw/native/nccl_%h_%p.log" \
  "${MPIRUN}" --allow-run-as-root --bind-to none \
  --mca pml ob1 --mca btl self,vader,tcp --mca coll_hcoll_enable 0 \
  -np 4 "${NATIVE_BIN}" "${RUN_DIR}/raw/native" \
  > "${RUN_DIR}/raw/native.stdout" 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL \
NCCL_DEBUG_FILE="${RUN_DIR}/raw/pytorch_default/nccl_%h_%p.log" \
  torchrun --standalone --nproc_per_node=4 \
  "${ROOT}/scripts/28_ch04_process_groups.py" \
  --output-dir "${RUN_DIR}/raw/pytorch_default" \
  > "${RUN_DIR}/raw/pytorch_default.stdout" 2>&1

CUDA_VISIBLE_DEVICES=2,0,3,1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL \
NCCL_DEBUG_FILE="${RUN_DIR}/raw/pytorch_remap/nccl_%h_%p.log" \
  torchrun --standalone --nproc_per_node=4 \
  "${ROOT}/scripts/28_ch04_process_groups.py" \
  --output-dir "${RUN_DIR}/raw/pytorch_remap" \
  > "${RUN_DIR}/raw/pytorch_remap.stdout" 2>&1

set +e
CUDA_VISIBLE_DEVICES=0 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,COLL \
  timeout 45s torchrun --standalone --nproc_per_node=2 \
  "${ROOT}/scripts/28_ch04_process_groups.py" \
  --device-mode duplicate-zero --output-dir "${RUN_DIR}/raw/duplicate" \
  > "${RUN_DIR}/raw/duplicate.stdout" 2>&1
duplicate_status=$?
set -e
echo "duplicate_exit_status=${duplicate_status}" > "${RUN_DIR}/raw/duplicate_status.txt"
if [[ ${duplicate_status} -eq 0 ]]; then
  echo "duplicate GPU experiment unexpectedly succeeded" >&2
  exit 1
fi

python3 "${ROOT}/probes/ch04_summarize.py" --run-dir "${RUN_DIR}" > "${RUN_DIR}/raw/summarizer.stdout"
cat "${RUN_DIR}/summary.md"
echo "run_dir=${RUN_DIR}"
