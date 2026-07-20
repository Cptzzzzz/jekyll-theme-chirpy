#!/usr/bin/env python3
import argparse
import csv
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def gather_metrics(values: list[float], device: torch.device) -> list[list[float]]:
    local = torch.tensor(values, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return [tensor.cpu().tolist() for tensor in gathered]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare host enqueue and GPU completion time")
    parser.add_argument("--bytes-mib", type=int, default=256)
    parser.add_argument("--csv", default="")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    rank = dist.get_rank()
    world = dist.get_world_size()

    elements = args.bytes_mib * 1024 * 1024 // 4
    x = torch.full((elements,), float(rank + 1), device=device)

    dist.all_reduce(x)
    torch.cuda.synchronize(device)
    x.fill_(float(rank + 1))
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    host_start = time.perf_counter_ns()
    work = dist.all_reduce(x, async_op=True)
    enqueue_end = time.perf_counter_ns()
    completed_immediately = work.is_completed()
    work.wait()
    wait_end = time.perf_counter_ns()
    end.record()
    end.synchronize()
    sync_end = time.perf_counter_ns()

    expected = world * (world + 1) / 2
    correct = bool(torch.all(x[:1024] == expected).item())
    metrics = gather_metrics(
        [
            (enqueue_end - host_start) / 1000.0,
            (wait_end - enqueue_end) / 1000.0,
            (sync_end - host_start) / 1000.0,
            start.elapsed_time(end) * 1000.0,
            float(completed_immediately),
            float(correct),
        ],
        device,
    )

    if rank == 0:
        fieldnames = [
            "rank",
            "host_enqueue_us",
            "host_wait_us",
            "host_total_to_sync_us",
            "gpu_elapsed_us",
            "is_completed_immediate",
            "correct",
        ]
        rows = []
        print(",".join(fieldnames))
        for r, row in enumerate(metrics):
            print(f"{r}," + ",".join(f"{value:.3f}" for value in row))
            rows.append(dict(zip(fieldnames, [r, *row])))
        if args.csv:
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"csv={args.csv}")

    if not correct:
        raise RuntimeError(f"rank {rank}: expected {expected}, got {x[0].item()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
