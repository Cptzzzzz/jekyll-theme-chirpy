#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/nccl-learning}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-${ROOT}/logs/topology_graph/${RUN_ID}}"
RAW_DIR="${OUT_DIR}/raw"
ARTIFACT_DIR="${OUT_DIR}/artifacts"
BIN="${ROOT}/third_party/nccl-tests/build/all_reduce_perf"
mkdir -p "${RAW_DIR}" "${ARTIFACT_DIR}"

{
  echo "run_id=${RUN_ID}"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "command=NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,NET,P2P,SHM ${BIN} -b 64M -e 64M -g 4 -w 5 -n 20 -c 1"
} > "${OUT_DIR}/manifest.txt"

nvidia-smi topo -m > "${ARTIFACT_DIR}/nvidia_smi_topo.txt"
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV,NET,P2P,SHM \
NCCL_TOPO_DUMP_FILE="${ARTIFACT_DIR}/topology.xml" \
NCCL_GRAPH_DUMP_FILE="${ARTIFACT_DIR}/graph.xml" \
  "${BIN}" -b 64M -e 64M -g 4 -w 5 -n 20 -c 1 \
  |& tee "${RAW_DIR}/topology_graph.log"

test -s "${ARTIFACT_DIR}/topology.xml"
test -s "${ARTIFACT_DIR}/graph.xml"
echo "completed_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${OUT_DIR}/manifest.txt"
echo "out_dir=${OUT_DIR}"
