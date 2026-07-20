#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/nccl-learning}
LOG_DIR=${LOG_DIR:-$ROOT/logs}
NPROC=${NPROC:-4}

mkdir -p "$LOG_DIR"
log="$LOG_DIR/ddp_nccl_debug_$(date -u +%Y%m%dT%H%M%SZ).log"

NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,COLL,P2P,SHM,ENV \
torchrun --standalone --nproc_per_node="$NPROC" "$ROOT/scripts/01_collectives.py" >"$log" 2>&1

echo
echo "== Useful log lines =="
grep -Ei 'Bootstrap|NCCL version|NET/|Using network|Init START|Init COMPLETE|Channel|Connected|P2P|SHM|NVL|GRAPH|cudaDev|busId|AllReduce|AllGather|ReduceScatter' "$log" | sed -n '1,240p' || true
echo
echo "If you need the full log, inspect:"
echo "log: $log"
