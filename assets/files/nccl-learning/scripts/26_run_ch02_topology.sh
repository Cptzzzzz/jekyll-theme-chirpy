#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/26_run_ch02_topology.sh [--run-id ID]

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "${1-}" == "--run-id" ]]; then RUN_ID="$2"; shift 2; fi
if [[ $# -ne 0 ]]; then echo "usage: $0 [--run-id ID]" >&2; exit 2; fi

RUN_DIR="${ROOT}/logs/ch02_topology/${RUN_ID}"
RAW_DIR="${RUN_DIR}/raw"
ARTIFACT_DIR="${RUN_DIR}/artifacts"
mkdir -p "${RAW_DIR}" "${ARTIFACT_DIR}"

{
  echo "experiment=ch02_topology"
  echo "run_id=${RUN_ID}"
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "nccl_source_commit=$(git -C "${ROOT}/third_party/nccl-2.22.3" rev-parse HEAD)"
  echo "torch=$(python3 -c 'import torch; print(torch.__version__)')"
  echo "matrix_sizes_kib=1,64,1024,16384,262144"
  echo "cycles=10"
  echo "warmup=5"
  env | grep -E '^(NCCL|TORCH_NCCL|CUDA)_' | sort || true
} > "${RUN_DIR}/manifest.txt"

nvidia-smi topo -m > "${RAW_DIR}/nvidia_smi_topo.log"
nvidia-smi nvlink --status > "${RAW_DIR}/nvlink_status.log"
nvidia-smi -q > "${RAW_DIR}/nvidia_smi_query.log"
lscpu > "${RAW_DIR}/lscpu.log"
numactl --hardware > "${RAW_DIR}/numactl_hardware.log"
if command -v lspci >/dev/null 2>&1; then
  lspci -tv > "${RAW_DIR}/lspci_tree.log"
else
  echo "lspci is not installed" > "${RAW_DIR}/lspci_tree.log"
fi

{
  for bdf in 0000:1a:00.0 0000:1c:00.0 0000:1d:00.0 0000:1e:00.0; do
    echo "bdf=${bdf} numa=$(cat "/sys/bus/pci/devices/${bdf}/numa_node") path=$(readlink -f "/sys/bus/pci/devices/${bdf}")"
  done
} > "${RAW_DIR}/gpu_pci_sysfs.log"

NCCL_TOPO_DUMP_FILE="${ARTIFACT_DIR}/topology.xml" \
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH \
  "${ROOT}/third_party/nccl-tests/build/all_reduce_perf" \
  -b 8 -e 8 -f 2 -g 4 -w 1 -n 1 \
  > "${RAW_DIR}/nccl_topology_dump.log" 2>&1

numactl --cpunodebind=0 \
  python3 "${ROOT}/scripts/26_ch02_topology_p2p.py" \
  --mode matrix --cycles 10 --warmup 5 \
  --sizes-kib 1,64,1024,16384,262144 \
  --output "${RAW_DIR}/matrix.csv" \
  > "${RAW_DIR}/matrix.stdout"

for node in 0 1; do
  numactl --cpunodebind="${node}" \
    python3 "${ROOT}/scripts/26_ch02_topology_p2p.py" \
    --mode concurrency --cycles 10 --warmup 5 --iterations 20 --bytes-mib 256 \
    --output "${RAW_DIR}/concurrent_numa${node}.csv" \
    > "${RAW_DIR}/concurrent_numa${node}.stdout"
done

python3 "${ROOT}/probes/ch02_summarize.py" --run-dir "${RUN_DIR}" > "${RAW_DIR}/summarizer.stdout"
cat "${RUN_DIR}/summary.md"
echo "run_dir=${RUN_DIR}"
