#!/usr/bin/env python3
"""Run one ordered or intentionally invalid ProcessGroupNCCL sequence."""

from __future__ import annotations

import argparse
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=["ordered", "op_mismatch", "size_mismatch", "rank_skip"],
        required=True,
    )
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=6))
    rank = dist.get_rank()
    if dist.get_world_size() != 2:
        raise RuntimeError("chapter 6 order worker requires exactly two ranks")

    warmup = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(warmup)
    torch.cuda.synchronize(device)
    if warmup.item() != 3.0:
        raise RuntimeError("communicator warmup failed")
    dist.barrier()
    print(f"rank={rank} case={args.case} phase=warmup-complete", flush=True)

    if args.case == "ordered":
        broadcast = torch.tensor([10.0 + rank], device=device)
        reduced = torch.tensor([float(rank + 1)], device=device)
        dist.broadcast(broadcast, src=0)
        dist.all_reduce(reduced)
        torch.cuda.synchronize(device)
        correct = broadcast.item() == 10.0 and reduced.item() == 3.0
        print(
            f"rank={rank} case=ordered broadcast={broadcast.item():.1f} "
            f"allreduce={reduced.item():.1f} correct={int(correct)}",
            flush=True,
        )
        if not correct:
            raise RuntimeError("ordered sequence produced incorrect output")
        dist.destroy_process_group()
        return

    if args.case == "op_mismatch":
        value = torch.tensor([10.0 + rank], device=device)
        if rank == 0:
            print("rank=0 seq=1 operation=broadcast", flush=True)
            dist.broadcast(value, src=0)
        else:
            print("rank=1 seq=1 operation=all_reduce", flush=True)
            dist.all_reduce(value)
        torch.cuda.synchronize(device)
        print(f"rank={rank} op_mismatch_returned value={value.item()}", flush=True)
        # Keep a locally completed rank alive so its peer can expose the real
        # timeout path instead of being terminated early by torchrun.
        time.sleep(10)
        raise RuntimeError("mismatched operations unexpectedly completed")

    if args.case == "size_mismatch":
        count = 4 if rank == 0 else 8
        value = torch.full((count,), float(rank + 1), device=device)
        print(f"rank={rank} seq=1 operation=all_reduce count={count}", flush=True)
        dist.all_reduce(value)
        torch.cuda.synchronize(device)
        print(f"rank={rank} size_mismatch_returned", flush=True)
        time.sleep(10)
        raise RuntimeError("mismatched counts unexpectedly completed")

    if rank == 0:
        print("rank=0 intentionally skips seq=1 collective", flush=True)
        time.sleep(15)
    else:
        value = torch.tensor([2.0], device=device)
        print("rank=1 seq=1 operation=all_reduce while rank0 skips", flush=True)
        dist.all_reduce(value)
        torch.cuda.synchronize(device)
        raise RuntimeError("rank-skip collective unexpectedly completed")


if __name__ == "__main__":
    main()
