#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass, asdict

import torch


@dataclass
class Result:
    cycle: int
    src: int
    dst: int
    peer_access: bool
    bytes: int
    time_us: float
    bandwidth_GBs: float
    correct: bool


def measure(cycle: int, src_id: int, dst_id: int, nbytes: int, warmup: int, iters: int) -> Result:
    elements = nbytes // torch.tensor([], dtype=torch.float32).element_size()
    src = torch.full((elements,), float(src_id + 1), device=f"cuda:{src_id}")
    dst = torch.empty(elements, dtype=torch.float32, device=f"cuda:{dst_id}")
    stream = torch.cuda.Stream(device=dst_id)

    with torch.cuda.device(dst_id), torch.cuda.stream(stream):
        for _ in range(warmup):
            dst.copy_(src, non_blocking=True)
        stream.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(iters):
            dst.copy_(src, non_blocking=True)
        end.record(stream)
        end.synchronize()

    time_ms = start.elapsed_time(end) / iters
    sample = dst[:1024].cpu()
    correct = bool(torch.all(sample == float(src_id + 1)).item())
    return Result(
        cycle=cycle,
        src=src_id,
        dst=dst_id,
        peer_access=torch.cuda.can_device_access_peer(src_id, dst_id),
        bytes=nbytes,
        time_us=time_ms * 1000.0,
        bandwidth_GBs=nbytes / (time_ms / 1000.0) / 1e9,
        correct=correct,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure directed GPU-to-GPU copy bandwidth")
    parser.add_argument("--bytes-mib", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--csv", default="")
    parser.add_argument("--summary-csv", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise SystemExit("at least two CUDA GPUs are required")

    nbytes = args.bytes_mib * 1024 * 1024
    results = []
    for src in range(torch.cuda.device_count()):
        for dst in range(torch.cuda.device_count()):
            if src == dst:
                continue
            for cycle in range(args.cycles):
                result = measure(cycle, src, dst, nbytes, args.warmup, args.iters)
                results.append(result)
                print(
                    f"cycle={cycle} src={src} dst={dst} peer={int(result.peer_access)} "
                    f"time_us={result.time_us:.2f} bandwidth_GBs={result.bandwidth_GBs:.2f} "
                    f"correct={int(result.correct)}",
                    flush=True,
                )

    if not all(r.correct for r in results):
        raise SystemExit("P2P correctness validation failed")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(r) for r in results)
        print(f"csv={args.csv}")

    summaries = []
    for src in range(torch.cuda.device_count()):
        for dst in range(torch.cuda.device_count()):
            pair = [r.bandwidth_GBs for r in results if r.src == src and r.dst == dst]
            if not pair:
                continue
            ordered = sorted(pair)
            p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
            mean = statistics.fmean(pair)
            summaries.append(
                {
                    "src": src,
                    "dst": dst,
                    "cycles": len(pair),
                    "median_bandwidth_GBs": statistics.median(pair),
                    "p95_bandwidth_GBs": p95,
                    "min_bandwidth_GBs": min(pair),
                    "max_bandwidth_GBs": max(pair),
                    "cv_percent": statistics.pstdev(pair) / mean * 100.0 if len(pair) > 1 else 0.0,
                }
            )
    print("src,dst,cycles,median_bandwidth_GBs,p95_bandwidth_GBs,min_bandwidth_GBs,max_bandwidth_GBs,cv_percent")
    for row in summaries:
        print(
            f"{row['src']},{row['dst']},{row['cycles']},"
            f"{row['median_bandwidth_GBs']:.3f},{row['p95_bandwidth_GBs']:.3f},"
            f"{row['min_bandwidth_GBs']:.3f},{row['max_bandwidth_GBs']:.3f},"
            f"{row['cv_percent']:.3f}"
        )

    if args.summary_csv:
        with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        print(f"summary_csv={args.summary_csv}")


if __name__ == "__main__":
    main()
