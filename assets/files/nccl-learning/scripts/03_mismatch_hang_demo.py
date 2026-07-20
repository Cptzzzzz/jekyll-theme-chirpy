#!/usr/bin/env python3
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl", timeout=timedelta(seconds=12))
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    x = torch.ones(1, device=f"cuda:{local_rank}") * (rank + 1)

    if rank == 0:
        print("[rank 0] intentionally skipping all_reduce", flush=True)
        time.sleep(20)
    else:
        print(f"[rank {rank}] entering all_reduce and waiting for rank 0", flush=True)
        dist.all_reduce(x)
        print(f"[rank {rank}] this line should not normally be reached: {x.item()}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
