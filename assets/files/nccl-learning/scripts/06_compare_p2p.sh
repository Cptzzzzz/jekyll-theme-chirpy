#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/nccl-learning}
REPO=${REPO:-$ROOT/third_party/nccl-tests}
LOG_DIR=${LOG_DIR:-$ROOT/logs}
MAX_BYTES=${MAX_BYTES:-1G}
GPUS=${GPUS:-4}

mkdir -p "$LOG_DIR"

if [[ ! -x "$REPO/build/all_reduce_perf" ]]; then
  "$ROOT/scripts/04_build_nccl_tests.sh"
fi

run_case() {
  local name=$1
  shift
  local log="$LOG_DIR/all_reduce_${name}_$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "== $name =="
  env "$@" "$REPO/build/all_reduce_perf" -b 8M -e "$MAX_BYTES" -f 2 -g "$GPUS" | tee "$log"
  echo "log: $log"
  awk '($1 !~ /^#/ && NF >= 12) {last=$0} END {if (last) print "last data row:", last}' "$log"
}

run_case baseline
run_case no_p2p NCCL_P2P_DISABLE=1
run_case no_shm NCCL_SHM_DISABLE=1
