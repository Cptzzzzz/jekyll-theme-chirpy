#!/usr/bin/env python3
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def local_rank() -> int:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return int(os.environ.get("RANK", "0")) % torch.cuda.device_count()


def main() -> None:
    dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    rank = dist.get_rank()
    world = dist.get_world_size()
    lr = local_rank()
    torch.cuda.set_device(lr)
    device = torch.device("cuda", lr)

    # all_reduce: every rank starts with a different tensor, every rank receives
    # the same reduced result. DDP uses this pattern for gradient synchronization.
    x = torch.full((4,), float(rank + 1), device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    print(f"[rank {rank}] all_reduce SUM -> {x.tolist()}", flush=True)

    # all_gather: every rank contributes one tensor; every rank receives all
    # tensors. Tensor/pipeline parallel and some checkpoint flows use this often.
    y = torch.tensor([rank, lr], dtype=torch.float32, device=device)
    gathered = [torch.empty_like(y) for _ in range(world)]
    dist.all_gather(gathered, y)
    gathered_cpu = [t.cpu().tolist() for t in gathered]
    print(f"[rank {rank}] all_gather -> {gathered_cpu}", flush=True)

    # reduce_scatter_tensor: reduce a concatenated input, then scatter shards.
    # FSDP/ZeRO-style sharding commonly relies on reduce-scatter/all-gather.
    if hasattr(dist, "reduce_scatter_tensor"):
        inp = torch.arange(world * 2, dtype=torch.float32, device=device) + rank * 100
        out = torch.empty(2, dtype=torch.float32, device=device)
        dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM)
        print(
            f"[rank {rank}] reduce_scatter_tensor input={inp.cpu().tolist()} out={out.cpu().tolist()}",
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
