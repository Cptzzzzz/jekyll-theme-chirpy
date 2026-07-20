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

for test in all_reduce all_gather reduce_scatter; do
  bin="$REPO/build/${test}_perf"
  log="$LOG_DIR/${test}_g${GPUS}_$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "== $test: -b 8 -e $MAX_BYTES -f 2 -g $GPUS =="
  "$bin" -b 8 -e "$MAX_BYTES" -f 2 -g "$GPUS" | tee "$log"
  echo "log: $log"
done
