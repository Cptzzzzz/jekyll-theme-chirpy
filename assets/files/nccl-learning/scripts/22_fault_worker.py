#!/usr/bin/env python3
import argparse
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject distributed collective failures after communicator warmup")
    parser.add_argument("--case", choices=["skip", "op_mismatch", "size_mismatch", "rank_exit"], required=True)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=6))
    rank = dist.get_rank()
    if dist.get_world_size() != 2:
        raise RuntimeError("fault worker requires exactly two ranks")

    warmup = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(warmup)
    torch.cuda.synchronize(device)
    if warmup.item() != 3.0:
        raise RuntimeError("warmup collective failed")
    dist.barrier()
    print(f"rank={rank} case={args.case} communicator_warm=1", flush=True)

    if args.case == "skip":
        if rank == 0:
            print("rank=0 intentionally skips the second all_reduce", flush=True)
            time.sleep(15)
        else:
            value = torch.tensor([2.0], device=device)
            print("rank=1 enters the second all_reduce", flush=True)
            dist.all_reduce(value)
            torch.cuda.synchronize(device)
    elif args.case == "op_mismatch":
        value = torch.tensor([float(rank)], device=device)
        if rank == 0:
            print("rank=0 enters broadcast", flush=True)
            dist.broadcast(value, src=0)
        else:
            print("rank=1 enters all_reduce", flush=True)
            dist.all_reduce(value)
        torch.cuda.synchronize(device)
    elif args.case == "size_mismatch":
        elements = 4 if rank == 0 else 8
        value = torch.full((elements,), float(rank + 1), device=device)
        print(f"rank={rank} enters all_reduce elements={elements}", flush=True)
        dist.all_reduce(value)
        torch.cuda.synchronize(device)
    elif args.case == "rank_exit":
        if rank == 0:
            print("rank=0 exits before the second all_reduce", flush=True)
            os._exit(42)
        value = torch.tensor([2.0], device=device)
        print("rank=1 enters all_reduce after peer exit", flush=True)
        dist.all_reduce(value)
        torch.cuda.synchronize(device)

    print(f"rank={rank} case={args.case} UNEXPECTED_SUCCESS", flush=True)
    dist.destroy_process_group()
    raise SystemExit(2)


if __name__ == "__main__":
    main()
