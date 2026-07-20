#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/nccl-learning}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-${ROOT}/logs/foundations/${RUN_ID}}"
RAW_DIR="${OUT_DIR}/raw"
ARTIFACT_DIR="${OUT_DIR}/artifacts"
mkdir -p "${RAW_DIR}" "${ARTIFACT_DIR}"

{
  echo "run_id=${RUN_ID}"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "command=$0"
  echo "p2p_bytes_mib=256"
  echo "p2p_warmup=5"
  echo "p2p_iters=20"
  echo "p2p_cycles=10"
  echo "async_all_reduce_bytes_mib=256"
  env | grep -E '^(NCCL_|TORCH_NCCL_|CUDA_VISIBLE_DEVICES=)' | sort || true
} > "${OUT_DIR}/manifest.txt"

"${ROOT}/scripts/10_version_matrix.sh" |& tee "${RAW_DIR}/version_matrix.log"
nvidia-smi topo -m |& tee "${ARTIFACT_DIR}/nvidia_smi_topo.txt"
numactl --hardware |& tee "${ARTIFACT_DIR}/numa.txt"

python3 "${ROOT}/scripts/11_p2p_matrix.py" \
  --bytes-mib 256 --warmup 5 --iters 20 --cycles 10 \
  --csv "${OUT_DIR}/p2p_raw.csv" --summary-csv "${OUT_DIR}/p2p_summary.csv" \
  |& tee "${RAW_DIR}/p2p_matrix.log"

torchrun --standalone --nproc_per_node=4 "${ROOT}/scripts/12_async_collective.py" \
  --bytes-mib 256 --csv "${OUT_DIR}/async_collective.csv" \
  |& tee "${RAW_DIR}/async_collective.log"

NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM,NET \
  torchrun --standalone --nproc_per_node=4 "${ROOT}/scripts/13_subgroup_communicators.py" \
  |& tee "${RAW_DIR}/subgroup.log"

torchrun --standalone --nproc_per_node=4 "${ROOT}/scripts/14_collective_correctness.py" \
  |& tee "${RAW_DIR}/collective_correctness.log"

python3 "${ROOT}/scripts/15_foundations_summary.py" "${OUT_DIR}"
echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT_DIR}/manifest.txt"
echo "out_dir=${OUT_DIR}"
