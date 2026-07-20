#!/usr/bin/env python3
"""Measure ProcessGroupNCCL host, Work and CUDA-stream completion semantics.

Run with torchrun. Examples:
  torchrun --standalone --nproc_per_node=4 scripts/27_ch03_async_semantics.py \
    --mode timing --output-dir /tmp/ch03
  torchrun --standalone --nproc_per_node=4 scripts/27_ch03_async_semantics.py \
    --mode stream-scope --output-dir /tmp/ch03-stream
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


@dataclass
class TimingResult:
    rank: int
    cycle: int
    size_bytes: int
    mode: str
    blocking_wait: bool
    host_call_us: float
    host_enqueue_us: float
    host_wait_or_sync_us: float
    stream_dependency_us: float
    completed_after_enqueue: float
    completed_after_wait: float
    correct: bool


@dataclass
class StreamScopeResult:
    rank: int
    cycle: int
    size_bytes: int
    sleep_cycles: int
    host_wait_us: float
    sleep_gpu_us: float
    completed_after_wait: bool
    free_stream_finished_before_wait_stream: bool
    correct: bool


def write_rows(path: Path, rows: list[TimingResult] | list[StreamScopeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def reset_and_align(tensor: torch.Tensor, value: float, device: torch.device) -> None:
    tensor.fill_(value)
    torch.cuda.synchronize(device)
    dist.barrier()
    torch.cuda.synchronize(device)


def run_timing(
    rank: int,
    world: int,
    device: torch.device,
    sizes: list[int],
    cycles: int,
) -> list[TimingResult]:
    blocking_wait = os.environ.get("TORCH_NCCL_BLOCKING_WAIT", "0") == "1"
    expected = world * (world + 1) / 2
    rows: list[TimingResult] = []

    for nbytes in sizes:
        tensor = torch.full((nbytes // 4,), float(rank + 1), dtype=torch.float32, device=device)
        dist.all_reduce(tensor)
        torch.cuda.synchronize(device)

        for mode in ("sync_api", "async_wait", "async_device_sync"):
            for cycle in range(cycles):
                reset_and_align(tensor, float(rank + 1), device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                call_start = time.perf_counter_ns()
                enqueue_us = math.nan
                wait_us = math.nan
                completed_after_enqueue = math.nan
                completed_after_wait = math.nan

                if mode == "sync_api":
                    dist.all_reduce(tensor, async_op=False)
                else:
                    work = dist.all_reduce(tensor, async_op=True)
                    enqueue_end = time.perf_counter_ns()
                    enqueue_us = (enqueue_end - call_start) / 1000.0
                    completed_after_enqueue = float(work.is_completed())
                    if mode == "async_wait":
                        wait_start = time.perf_counter_ns()
                        work.wait()
                        wait_end = time.perf_counter_ns()
                        wait_us = (wait_end - wait_start) / 1000.0
                        completed_after_wait = float(work.is_completed())
                    else:
                        sync_start = time.perf_counter_ns()
                        torch.cuda.synchronize(device)
                        sync_end = time.perf_counter_ns()
                        wait_us = (sync_end - sync_start) / 1000.0
                        completed_after_wait = float(work.is_completed())

                call_end = time.perf_counter_ns()
                end.record()
                end.synchronize()
                gpu_us = start.elapsed_time(end) * 1000.0
                correct = bool(torch.all(tensor == expected).item())
                rows.append(
                    TimingResult(
                        rank=rank,
                        cycle=cycle,
                        size_bytes=nbytes,
                        mode=mode,
                        blocking_wait=blocking_wait,
                        host_call_us=(call_end - call_start) / 1000.0,
                        host_enqueue_us=enqueue_us,
                        host_wait_or_sync_us=wait_us,
                        stream_dependency_us=gpu_us,
                        completed_after_enqueue=completed_after_enqueue,
                        completed_after_wait=completed_after_wait,
                        correct=correct,
                    )
                )
        del tensor
    return rows


def run_stream_scope(
    rank: int,
    world: int,
    device: torch.device,
    nbytes: int,
    sleep_cycles: int,
    cycles: int,
) -> list[StreamScopeResult]:
    if os.environ.get("TORCH_NCCL_BLOCKING_WAIT", "0") == "1":
        raise RuntimeError("stream-scope mode requires TORCH_NCCL_BLOCKING_WAIT=0")
    if not hasattr(torch.cuda, "_sleep"):
        raise RuntimeError("torch.cuda._sleep is unavailable in this PyTorch build")

    expected = world * (world + 1) / 2
    tensor = torch.full((nbytes // 4,), float(rank + 1), dtype=torch.float32, device=device)
    wait_stream = torch.cuda.Stream(device=device)
    free_stream = torch.cuda.Stream(device=device)
    dist.all_reduce(tensor)
    torch.cuda.synchronize(device)

    rows: list[StreamScopeResult] = []
    for cycle in range(cycles):
        reset_and_align(tensor, float(rank + 1), device)
        sleep_start = torch.cuda.Event(enable_timing=True)
        sleep_end = torch.cuda.Event(enable_timing=True)
        sleep_start.record()
        torch.cuda._sleep(sleep_cycles)
        sleep_end.record()

        work = dist.all_reduce(tensor, async_op=True)
        wait_marker = torch.cuda.Event()
        free_marker = torch.cuda.Event()

        with torch.cuda.stream(wait_stream):
            host_wait_start = time.perf_counter_ns()
            work.wait()
            host_wait_end = time.perf_counter_ns()
            wait_marker.record(wait_stream)

        with torch.cuda.stream(free_stream):
            free_marker.record(free_stream)

        deadline = time.monotonic() + 2.0
        while not free_marker.query() and time.monotonic() < deadline:
            time.sleep(0.0001)
        free_before_wait = free_marker.query() and not wait_marker.query()
        completed_after_wait = work.is_completed()

        wait_marker.synchronize()
        sleep_end.synchronize()
        correct = bool(torch.all(tensor == expected).item())
        rows.append(
            StreamScopeResult(
                rank=rank,
                cycle=cycle,
                size_bytes=nbytes,
                sleep_cycles=sleep_cycles,
                host_wait_us=(host_wait_end - host_wait_start) / 1000.0,
                sleep_gpu_us=sleep_start.elapsed_time(sleep_end) * 1000.0,
                completed_after_wait=completed_after_wait,
                free_stream_finished_before_wait_stream=free_before_wait,
                correct=correct,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("timing", "stream-scope"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes-kib", default="1,1024,65536,262144")
    parser.add_argument("--bytes-mib", type=int, default=64)
    parser.add_argument("--sleep-cycles", type=int, default=200_000_000)
    parser.add_argument("--cycles", type=int, default=10)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    rank = dist.get_rank()
    world = dist.get_world_size()

    if args.mode == "timing":
        sizes = [int(value) * 1024 for value in args.sizes_kib.split(",")]
        rows = run_timing(rank, world, device, sizes, args.cycles)
    else:
        rows = run_stream_scope(
            rank,
            world,
            device,
            args.bytes_mib * 1024 * 1024,
            args.sleep_cycles,
            args.cycles,
        )

    output = args.output_dir / f"rank{rank}.csv"
    write_rows(output, rows)
    if not all(row.correct for row in rows):
        raise RuntimeError(f"rank {rank}: correctness failure")
    dist.barrier()
    if rank == 0:
        print(f"mode={args.mode} output_dir={args.output_dir} rows_per_rank={len(rows)}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
