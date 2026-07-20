#!/usr/bin/env python3
import os
from datetime import timedelta

import torch
import torch.distributed as dist


def assert_equal(name: str, actual: torch.Tensor, expected: torch.Tensor, rank: int) -> None:
    if not torch.equal(actual.cpu(), expected.cpu()):
        raise AssertionError(
            f"rank {rank} {name} mismatch: actual={actual.cpu().tolist()} expected={expected.cpu().tolist()}"
        )


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    rank = dist.get_rank()
    world = dist.get_world_size()
    passed = []

    value = torch.full((4,), rank + 1, dtype=torch.int64, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    assert_equal("all_reduce", value, torch.full((4,), world * (world + 1) // 2, dtype=torch.int64), rank)
    passed.append("all_reduce")

    value = torch.full((4,), rank + 10, dtype=torch.int64, device=device)
    dist.broadcast(value, src=0)
    assert_equal("broadcast", value, torch.full((4,), 10, dtype=torch.int64), rank)
    passed.append("broadcast")

    value = torch.tensor([rank, rank + 100], dtype=torch.int64, device=device)
    gathered = [torch.empty_like(value) for _ in range(world)]
    dist.all_gather(gathered, value)
    actual = torch.stack(gathered).cpu()
    expected = torch.tensor([[r, r + 100] for r in range(world)], dtype=torch.int64)
    assert_equal("all_gather", actual, expected, rank)
    passed.append("all_gather")

    chunk = 2
    rs_input = torch.arange(world * chunk, dtype=torch.int64, device=device) + rank * 100
    rs_output = torch.empty(chunk, dtype=torch.int64, device=device)
    dist.reduce_scatter_tensor(rs_output, rs_input, op=dist.ReduceOp.SUM)
    rank_slice = torch.arange(rank * chunk, (rank + 1) * chunk, dtype=torch.int64)
    rs_expected = rank_slice * world + 100 * world * (world - 1) // 2
    assert_equal("reduce_scatter", rs_output, rs_expected, rank)
    passed.append("reduce_scatter")

    a2a_input = torch.empty(world * chunk, dtype=torch.int64, device=device)
    for dst in range(world):
        a2a_input[dst * chunk : (dst + 1) * chunk] = torch.tensor(
            [rank * 100 + dst * 10, rank * 100 + dst * 10 + 1], device=device
        )
    a2a_output = torch.empty_like(a2a_input)
    dist.all_to_all_single(a2a_output, a2a_input)
    a2a_expected = torch.tensor(
        [item for src in range(world) for item in (src * 100 + rank * 10, src * 100 + rank * 10 + 1)],
        dtype=torch.int64,
    )
    assert_equal("all_to_all", a2a_output, a2a_expected, rank)
    passed.append("all_to_all")

    reduced = torch.full((2,), rank + 1, dtype=torch.int64, device=device)
    dist.reduce(reduced, dst=0, op=dist.ReduceOp.SUM)
    if rank == 0:
        assert_equal(
            "reduce", reduced, torch.full((2,), world * (world + 1) // 2, dtype=torch.int64), rank
        )
    passed.append("reduce")

    if rank == 0:
        send = torch.tensor([20260710], dtype=torch.int64, device=device)
        dist.send(send, dst=1)
    elif rank == 1:
        recv = torch.empty(1, dtype=torch.int64, device=device)
        dist.recv(recv, src=0)
        assert_equal("send_recv", recv, torch.tensor([20260710], dtype=torch.int64), rank)
    passed.append("send_recv")

    dist.barrier()
    print(f"rank={rank} passed={','.join(passed)} correct=1", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
