#!/usr/bin/env python3
"""Validate rank/device mapping and overlapping ProcessGroupNCCL domains.

Run with torchrun. `--device-mode duplicate-zero` intentionally binds every
rank to cuda:0 and is expected to fail when more than one rank is launched.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def physical_gpu_map() -> dict[str, str]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader,nounits"],
        text=True,
    )
    result = {}
    for line in output.splitlines():
        uuid, pci = [part.strip() for part in line.split(",")]
        result[uuid.removeprefix("GPU-").lower()] = pci.lower()
    return result


def reduce_value(rank: int, group: dist.ProcessGroup, expected: float, device: torch.device) -> float:
    value = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(value, group=group)
    torch.cuda.synchronize(device)
    actual = float(value.item())
    if actual != expected:
        raise RuntimeError(f"rank {rank}: expected {expected}, got {actual}")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-mode", choices=("local-rank", "duplicate-zero"), default="local-rank")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    device_index = local_rank if args.device_mode == "local-rank" else 0
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    dist.init_process_group("nccl", timeout=timedelta(seconds=30))
    rank = dist.get_rank()
    world = dist.get_world_size()

    if args.device_mode == "duplicate-zero":
        value = torch.tensor([float(rank + 1)], device=device)
        dist.all_reduce(value)
        torch.cuda.synchronize(device)
        raise RuntimeError("duplicate-zero experiment unexpectedly succeeded")

    if world != 4:
        raise RuntimeError("normal experiment requires exactly four ranks")

    uuid = str(torch.cuda.get_device_properties(device_index).uuid).lower()
    pci_bus_id = physical_gpu_map()[uuid]
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")

    print(f"MARK rank={rank} phase=before_new_group", flush=True)
    group_even = dist.new_group([0, 2], backend="nccl")
    group_odd = dist.new_group([1, 3], backend="nccl")
    group_low = dist.new_group([0, 1], backend="nccl")
    group_high = dist.new_group([2, 3], backend="nccl")
    print(f"MARK rank={rank} phase=groups_created_no_collective", flush=True)

    parity_group = group_even if rank % 2 == 0 else group_odd
    parity_members = [0, 2] if rank % 2 == 0 else [1, 3]
    pair_group = group_low if rank < 2 else group_high
    pair_members = [0, 1] if rank < 2 else [2, 3]

    parity_sum = reduce_value(rank, parity_group, 4.0 if rank % 2 == 0 else 6.0, device)
    pair_sum = reduce_value(rank, pair_group, 3.0 if rank < 2 else 7.0, device)
    world_sum = reduce_value(rank, dist.group.WORLD, 10.0, device)
    print(f"MARK rank={rank} phase=collectives_complete", flush=True)

    row = {
        "pid": os.getpid(),
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world,
        "visible_device_ordinal": device_index,
        "physical_uuid": uuid,
        "physical_pci_bus_id": pci_bus_id,
        "cuda_visible_devices": visible_devices,
        "parity_members": "-".join(map(str, parity_members)),
        "parity_group_rank": dist.get_rank(parity_group),
        "parity_group_size": dist.get_world_size(parity_group),
        "parity_sum": parity_sum,
        "pair_members": "-".join(map(str, pair_members)),
        "pair_group_rank": dist.get_rank(pair_group),
        "pair_group_size": dist.get_world_size(pair_group),
        "pair_sum": pair_sum,
        "world_sum": world_sum,
        "correct": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / f"rank{rank}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    dist.barrier()
    for group in (group_even, group_odd, group_low, group_high):
        if group != dist.GroupMember.NON_GROUP_MEMBER:
            dist.destroy_process_group(group)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
