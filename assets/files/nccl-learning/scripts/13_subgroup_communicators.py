#!/usr/bin/env python3
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 4:
        raise RuntimeError("this experiment requires exactly four ranks")

    group_a_ranks = [0, 1]
    group_b_ranks = [2, 3]
    group_a = dist.new_group(group_a_ranks, backend="nccl")
    group_b = dist.new_group(group_b_ranks, backend="nccl")
    own_group = group_a if rank in group_a_ranks else group_b
    own_ranks = group_a_ranks if rank in group_a_ranks else group_b_ranks

    subgroup_value = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(subgroup_value, group=own_group)
    expected_subgroup = float(sum(r + 1 for r in own_ranks))
    if subgroup_value.item() != expected_subgroup:
        raise RuntimeError(
            f"rank {rank}: subgroup expected {expected_subgroup}, got {subgroup_value.item()}"
        )

    world_value = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(world_value)
    expected_world = float(world * (world + 1) // 2)
    if world_value.item() != expected_world:
        raise RuntimeError(f"rank {rank}: world expected {expected_world}, got {world_value.item()}")

    print(
        f"rank={rank} local_rank={local_rank} subgroup={own_ranks} "
        f"subgroup_sum={subgroup_value.item():.0f} world_sum={world_value.item():.0f} correct=1",
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
