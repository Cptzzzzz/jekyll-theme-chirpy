#!/usr/bin/env bash
set -euo pipefail

echo "== RDMA devices =="
ls -la /dev/infiniband 2>/dev/null || true
ibv_devinfo -l 2>/dev/null || true
ibv_devinfo 2>/dev/null | sed -n '1,160p' || true

echo
echo "== RDMA links =="
if command -v rdma >/dev/null 2>&1; then
  rdma link || true
else
  echo "rdma command not installed"
fi

echo
echo "== Network interfaces =="
if command -v ip >/dev/null 2>&1; then
  ip -br addr || true
  ip route || true
else
  echo "ip command not installed"
  hostname -I 2>/dev/null || true
fi

echo
echo "== GPU / NIC topology =="
nvidia-smi topo -m

echo
echo "== NCCL multi-node env template =="
cat <<'EOF'
# Set these only after confirming your fabric and interface names.
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,COLL,ENV
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=<bootstrap_iface_or_prefix>
export NCCL_IB_HCA=<mlx5_device_or_prefix>
export NCCL_IB_GID_INDEX=<roce_gid_index_if_needed>
EOF
