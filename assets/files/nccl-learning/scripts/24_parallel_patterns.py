#!/usr/bin/env python3
import argparse
import csv
import math
import os
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import torch.distributed as dist


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(q * len(ordered)) - 1]


def measure(fn, reset, warmup: int, iters: int, device: torch.device, rank: int) -> list[float]:
    for _ in range(warmup):
        reset()
        fn()
    torch.cuda.synchronize(device)
    times = []
    for _ in range(iters):
        reset()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        critical = torch.tensor([start.elapsed_time(end)], dtype=torch.float64, device=device)
        dist.all_reduce(critical, op=dist.ReduceOp.MAX)
        if rank == 0:
            times.append(critical.item())
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure communication patterns used by model parallel strategies")
    parser.add_argument("--sizes-mib", default="1,16,64")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 4:
        raise RuntimeError("parallel pattern experiment requires exactly four ranks")
    rows = []

    for size_mib in [int(value) for value in args.sizes_mib.split(",")]:
        nbytes = size_mib * 1024 * 1024
        elements = nbytes // 4

        ar = torch.empty(elements, dtype=torch.float32, device=device)
        ar_times = measure(
            lambda: dist.all_reduce(ar),
            lambda: ar.fill_(rank + 1),
            args.warmup,
            args.iters,
            device,
            rank,
        )
        ar.fill_(rank + 1)
        dist.all_reduce(ar)
        if not torch.all(ar[:1024] == 10).item():
            raise RuntimeError("AllReduce correctness failed")
        if rank == 0:
            rows.append(("tp_all_reduce", size_mib, ar_times))

        rs_input = torch.empty(elements, dtype=torch.float32, device=device)
        rs_output = torch.empty(elements // world, dtype=torch.float32, device=device)
        ag_output = torch.empty(elements, dtype=torch.float32, device=device)

        def rs_ag() -> None:
            dist.reduce_scatter_tensor(rs_output, rs_input)
            dist.all_gather_into_tensor(ag_output, rs_output)

        rs_ag_times = measure(
            rs_ag,
            lambda: rs_input.fill_(rank + 1),
            args.warmup,
            args.iters,
            device,
            rank,
        )
        rs_input.fill_(rank + 1)
        rs_ag()
        if not torch.all(ag_output[:1024] == 10).item():
            raise RuntimeError("ReduceScatter+AllGather correctness failed")
        if rank == 0:
            rows.append(("sharded_rs_ag", size_mib, rs_ag_times))

        a2a_input = torch.empty(elements, dtype=torch.float32, device=device)
        a2a_output = torch.empty_like(a2a_input)
        chunk = elements // world

        def reset_a2a() -> None:
            for dst in range(world):
                a2a_input[dst * chunk : (dst + 1) * chunk].fill_(rank * 100 + dst)

        a2a_times = measure(
            lambda: dist.all_to_all_single(a2a_output, a2a_input),
            reset_a2a,
            args.warmup,
            args.iters,
            device,
            rank,
        )
        reset_a2a()
        dist.all_to_all_single(a2a_output, a2a_input)
        for src in range(world):
            expected = src * 100 + rank
            if not torch.all(a2a_output[src * chunk : src * chunk + min(chunk, 1024)] == expected).item():
                raise RuntimeError(f"AllToAll correctness failed for src={src} rank={rank}")
        if rank == 0:
            rows.append(("ep_all_to_all", size_mib, a2a_times))

        p2p = torch.empty(elements, dtype=torch.float32, device=device)
        peer = rank + 1 if rank % 2 == 0 else rank - 1

        def send_recv() -> None:
            if rank % 2 == 0:
                dist.send(p2p, dst=peer)
            else:
                dist.recv(p2p, src=peer)

        p2p_times = measure(
            send_recv,
            lambda: p2p.fill_(rank + 1) if rank % 2 == 0 else p2p.zero_(),
            args.warmup,
            args.iters,
            device,
            rank,
        )
        if rank % 2 == 1 and not torch.all(p2p[:1024] == peer + 1).item():
            raise RuntimeError(f"Send/Recv correctness failed on rank {rank}")
        if rank == 0:
            rows.append(("pp_send_recv", size_mib, p2p_times))

    if rank == 0:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        structured = []
        for pattern, size_mib, times in rows:
            median_ms = statistics.median(times)
            structured.append(
                {
                    "pattern": pattern,
                    "size_mib": size_mib,
                    "iterations": len(times),
                    "median_ms": median_ms,
                    "p95_ms": percentile(times, 0.95),
                    "min_ms": min(times),
                    "max_ms": max(times),
                    "payload_GBs": size_mib * 1024 * 1024 / (median_ms / 1000.0) / 1e9,
                    "correct": 1,
                }
            )
        with (args.out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(structured[0].keys()))
            writer.writeheader()
            writer.writerows(structured)
        lines = [
            "# Model Parallel Communication Patterns",
            "",
            f"Generated: `{datetime.now(timezone.utc).isoformat()}`",
            "",
            "| pattern | size MiB | median ms | P95 ms | payload GB/s | correct |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in structured:
            lines.append(
                f"| {row['pattern']} | {row['size_mib']} | {row['median_ms']:.3f} | {row['p95_ms']:.3f} | "
                f"{row['payload_GBs']:.2f} | {row['correct']} |"
            )
        lines.extend(
            [
                "",
                "`payload_GBs` is application payload divided by critical-rank latency. It is not NCCL busbw and must not be compared as a physical-link utilization metric across different collective semantics.",
            ]
        )
        (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"out_dir={args.out_dir}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
