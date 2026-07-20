#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/25_ch01_stack_version.sh [--output-root DIR] [--run-id ID]
#
# The experiment compiles a native NCCL version probe, proves its real loader
# path, and then uses an isolated fake SONAME to demonstrate LD_LIBRARY_PATH
# precedence. It never modifies the system linker configuration.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${ROOT}/logs/ch01_stack_version"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"

usage() {
  sed -n '4,10p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
RAW_DIR="${RUN_DIR}/raw"
ARTIFACT_DIR="${RUN_DIR}/artifacts"
BUILD_DIR="${ARTIFACT_DIR}/build"
FAKE_DIR="${ARTIFACT_DIR}/fake"
mkdir -p "${RAW_DIR}" "${BUILD_DIR}" "${FAKE_DIR}"

PROBE="${BUILD_DIR}/nccl_version_probe"
FAKE_LIB="${FAKE_DIR}/libnccl.so.2"
CUDA_INCLUDE="${CUDA_HOME:-/usr/local/cuda}/include"

cc -O2 -Wall -Wextra \
  -I"${CUDA_INCLUDE}" \
  "${ROOT}/probes/ch01_version_probe.c" \
  -L/lib/x86_64-linux-gnu -lnccl -ldl \
  -o "${PROBE}"

cc -O2 -Wall -Wextra -fPIC -shared \
  -Wl,-soname,libnccl.so.2 \
  "${ROOT}/probes/ch01_fake_nccl.c" \
  -o "${FAKE_LIB}"

{
  echo "experiment=ch01_stack_version"
  echo "run_id=${RUN_ID}"
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "kernel=$(uname -srmo)"
  echo "probe_compiler=$(cc --version | head -1)"
  echo "runtime_source_commit=$(git -C "${ROOT}/third_party/nccl-2.22.3" rev-parse HEAD)"
  echo "runtime_source_tag=$(git -C "${ROOT}/third_party/nccl-2.22.3" describe --tags --exact-match 2>/dev/null || true)"
  echo "ld_library_path=${LD_LIBRARY_PATH-<unset>}"
  env | grep -E '^(NCCL|TORCH_NCCL|CUDA)_' | sort || true
} > "${RUN_DIR}/manifest.txt"

{
  nvidia-smi
  echo
  nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version --format=csv,noheader
  echo
  if command -v nvcc >/dev/null 2>&1; then nvcc --version; else /usr/local/cuda/bin/nvcc --version; fi
  echo
  dpkg-query -W -f='${Package}\t${Version}\n' 2>/dev/null | grep -E '^libnccl' || true
  echo
  ldconfig -p | grep 'libnccl\.so' || true
} > "${RAW_DIR}/system_versions.log"

python3 "${ROOT}/probes/ch01_torch_runtime.py" > "${RAW_DIR}/torch_runtime.log"
"${PROBE}" > "${RAW_DIR}/probe_system.log"
LD_DEBUG=libs "${PROBE}" > "${RAW_DIR}/probe_system_ld_debug.stdout" 2> "${RAW_DIR}/probe_system_ld_debug.stderr"

"${PROBE}" 3 > "${RAW_DIR}/probe_system_hold.log" &
system_pid=$!
for _ in $(seq 1 30); do
  [[ -s "${RAW_DIR}/probe_system_hold.log" ]] && break
  sleep 0.1
done
grep 'libnccl' "/proc/${system_pid}/maps" > "${RAW_DIR}/probe_system_maps.log"
wait "${system_pid}"

LD_LIBRARY_PATH="${FAKE_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${PROBE}" > "${RAW_DIR}/probe_fake.log"
LD_LIBRARY_PATH="${FAKE_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  LD_DEBUG=libs "${PROBE}" > "${RAW_DIR}/probe_fake_ld_debug.stdout" 2> "${RAW_DIR}/probe_fake_ld_debug.stderr"

LD_LIBRARY_PATH="${FAKE_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  "${PROBE}" 3 > "${RAW_DIR}/probe_fake_hold.log" &
fake_pid=$!
for _ in $(seq 1 30); do
  [[ -s "${RAW_DIR}/probe_fake_hold.log" ]] && break
  sleep 0.1
done
grep 'libnccl' "/proc/${fake_pid}/maps" > "${RAW_DIR}/probe_fake_maps.log"
wait "${fake_pid}"

readelf -d "${PROBE}" > "${RAW_DIR}/probe_dynamic_section.log"
readelf -d /lib/x86_64-linux-gnu/libnccl.so.2 > "${RAW_DIR}/system_nccl_dynamic_section.log"
readelf -Ws /lib/x86_64-linux-gnu/libnccl.so.2 | grep 'ncclGetVersion' > "${RAW_DIR}/system_nccl_version_symbol.log"
ldd "${PROBE}" > "${RAW_DIR}/probe_ldd.log"

extract_value() {
  local key="$1"
  local file="$2"
  awk -F= -v key="${key}" '$1 == key {print substr($0, length(key) + 2); exit}' "${file}"
}

header_code="$(extract_value header_version_code "${RAW_DIR}/probe_system.log")"
system_code="$(extract_value runtime_version_code "${RAW_DIR}/probe_system.log")"
system_version="$(extract_value runtime_version "${RAW_DIR}/probe_system.log")"
system_object="$(extract_value nccl_get_version_object "${RAW_DIR}/probe_system.log")"
fake_code="$(extract_value runtime_version_code "${RAW_DIR}/probe_fake.log")"
fake_version="$(extract_value runtime_version "${RAW_DIR}/probe_fake.log")"
fake_object="$(extract_value nccl_get_version_object "${RAW_DIR}/probe_fake.log")"
torch_compiled_code="$(extract_value torch_compiled_nccl_version_code "${RAW_DIR}/torch_runtime.log")"
torch_compiled_version="$(extract_value torch_compiled_nccl_version "${RAW_DIR}/torch_runtime.log")"
torch_runtime_code="$(extract_value ctypes_runtime_nccl_version_code "${RAW_DIR}/torch_runtime.log")"

test "${header_code}" = "${system_code}"
test "${header_code}" = "${torch_compiled_code}"
test "${system_code}" = "${torch_runtime_code}"
test "${fake_code}" = "29999"
test "${fake_code}" != "${header_code}"
grep -Fq "${FAKE_LIB}" "${RAW_DIR}/probe_fake_maps.log"
grep -Fq 'libnccl.so.2' "${RAW_DIR}/probe_dynamic_section.log"

{
  echo 'metric,value'
  echo "header_version_code,${header_code}"
  echo "system_runtime_version_code,${system_code}"
  echo "system_runtime_version,${system_version}"
  echo "torch_compiled_version_code,${torch_compiled_code}"
  echo "torch_compiled_version,${torch_compiled_version}"
  echo "torch_loaded_runtime_version_code,${torch_runtime_code}"
  echo "fake_runtime_version_code,${fake_code}"
  echo "fake_runtime_version,${fake_version}"
  echo "header_matches_system,true"
  echo "header_matches_torch_compile_time,true"
  echo "system_matches_torch_loaded_runtime,true"
  echo "fake_override_proved,true"
  echo "system_object,${system_object}"
  echo "fake_object,${fake_object}"
} > "${RUN_DIR}/summary.csv"

{
  echo '# Chapter 01 experiment summary'
  echo
  echo "- Run ID: \`${RUN_ID}\`"
  echo "- Header version code: \`${header_code}\`"
  echo "- System runtime: \`${system_version}\` (\`${system_code}\`)"
  echo "- PyTorch compile-time NCCL macros: \`${torch_compiled_version}\` (\`${torch_compiled_code}\`)"
  echo "- NCCL runtime loaded in the PyTorch process: \`${torch_runtime_code}\`"
  echo "- System object reported by \`dladdr\`: \`${system_object}\`"
  echo "- Isolated override runtime: \`${fake_version}\` (\`${fake_code}\`)"
  echo "- Override object reported by \`dladdr\`: \`${fake_object}\`"
  echo
  echo '## Assertions'
  echo
  echo '- System header and PyTorch compile-time macros agree.'
  echo '- Native probe and the NCCL object loaded in the PyTorch process agree.'
  echo '- The probe ELF requests the SONAME `libnccl.so.2`, not an absolute file.'
  echo '- `LD_LIBRARY_PATH` can satisfy that SONAME with a different object.'
  echo '- `dladdr`, loader trace and `/proc/<pid>/maps` independently identify the loaded object.'
} > "${RUN_DIR}/summary.md"

echo "run_dir=${RUN_DIR}"
cat "${RUN_DIR}/summary.md"
