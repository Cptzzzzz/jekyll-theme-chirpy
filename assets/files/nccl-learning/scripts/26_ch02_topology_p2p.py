#!/usr/bin/env python3
"""Measure directed P2P curves or concurrent GPU-copy patterns.

Examples:
  python3 scripts/26_ch02_topology_p2p.py --mode matrix --output matrix.csv
  python3 scripts/26_ch02_topology_p2p.py --mode concurrency --output concurrent.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class MatrixResult:
    size_bytes: int
    src: int
    dst: int
    cycle: int
    iterations: int
    peer_access: bool
    time_us: float
    bandwidth_GBs: float
    correct: bool


@dataclass
class ConcurrentResult:
    pattern: str
    edges: int
    size_bytes_per_edge: int
    cycle: int
    iterations: int
    time_us_per_iteration: float
    aggregate_bandwidth_GBs: float
    correct: bool


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def iterations_for_size(nbytes: int) -> int:
    if nbytes <= 64 * 1024:
        return 2000
    if nbytes <= 1024 * 1024:
        return 500
    if nbytes <= 16 * 1024 * 1024:
        return 100
    return 20


def directed_pair(
    src_id: int,
    dst_id: int,
    nbytes: int,
    warmup: int,
    cycles: int,
) -> list[MatrixResult]:
    iterations = iterations_for_size(nbytes)
    value = (src_id * 17 + dst_id + 1) % 251
    src = torch.full((nbytes,), value, dtype=torch.uint8, device=f"cuda:{src_id}")
    dst = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{dst_id}")
    stream = torch.cuda.Stream(device=dst_id)

    with torch.cuda.device(dst_id), torch.cuda.stream(stream):
        for _ in range(warmup):
            dst.copy_(src, non_blocking=True)
        stream.synchronize()

    rows: list[MatrixResult] = []
    for cycle in range(cycles):
        with torch.cuda.device(dst_id), torch.cuda.stream(stream):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            for _ in range(iterations):
                dst.copy_(src, non_blocking=True)
            end.record(stream)
            end.synchronize()
            elapsed_ms = start.elapsed_time(end)

        time_us = elapsed_ms * 1000.0 / iterations
        rows.append(
            MatrixResult(
                size_bytes=nbytes,
                src=src_id,
                dst=dst_id,
                cycle=cycle,
                iterations=iterations,
                peer_access=torch.cuda.can_device_access_peer(src_id, dst_id),
                time_us=time_us,
                bandwidth_GBs=nbytes / (time_us / 1e6) / 1e9,
                correct=True,
            )
        )

    full_correct = bool(torch.all(dst == value).item())
    for row in rows:
        row.correct = full_correct
    del src, dst, stream
    return rows


PATTERNS: dict[str, list[tuple[int, int]]] = {
    "single_0_to_1": [(0, 1)],
    "bidirectional_0_1": [(0, 1), (1, 0)],
    "disjoint_pairs": [(0, 1), (2, 3)],
    "ring_4_edges": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "fanin_to_0": [(1, 0), (2, 0), (3, 0)],
}


def concurrent_pattern(
    name: str,
    edges: list[tuple[int, int]],
    nbytes: int,
    warmup: int,
    iterations: int,
    cycles: int,
) -> list[ConcurrentResult]:
    transfers = []
    devices = sorted({device for edge in edges for device in edge})
    for index, (src_id, dst_id) in enumerate(edges):
        value = (index + 1) * 29 % 251
        src = torch.full((nbytes,), value, dtype=torch.uint8, device=f"cuda:{src_id}")
        dst = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{dst_id}")
        stream = torch.cuda.Stream(device=dst_id)
        transfers.append((src_id, dst_id, value, src, dst, stream))

    for _ in range(warmup):
        for _, dst_id, _, src, dst, stream in transfers:
            with torch.cuda.device(dst_id), torch.cuda.stream(stream):
                dst.copy_(src, non_blocking=True)
    for device in devices:
        torch.cuda.synchronize(device)

    rows: list[ConcurrentResult] = []
    for cycle in range(cycles):
        for device in devices:
            torch.cuda.synchronize(device)
        start = time.perf_counter_ns()
        for _ in range(iterations):
            for _, dst_id, _, src, dst, stream in transfers:
                with torch.cuda.device(dst_id), torch.cuda.stream(stream):
                    dst.copy_(src, non_blocking=True)
        for device in devices:
            torch.cuda.synchronize(device)
        elapsed_s = (time.perf_counter_ns() - start) / 1e9

        per_iteration_s = elapsed_s / iterations
        rows.append(
            ConcurrentResult(
                pattern=name,
                edges=len(edges),
                size_bytes_per_edge=nbytes,
                cycle=cycle,
                iterations=iterations,
                time_us_per_iteration=per_iteration_s * 1e6,
                aggregate_bandwidth_GBs=(nbytes * len(edges)) / per_iteration_s / 1e9,
                correct=True,
            )
        )

    full_correct = all(bool(torch.all(dst == value).item()) for _, _, value, _, dst, _ in transfers)
    for row in rows:
        row.correct = full_correct
    return rows


def write_rows(path: Path, rows: list[MatrixResult] | list[ConcurrentResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("matrix", "concurrency"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes-kib", default="1,64,1024,16384,262144")
    parser.add_argument("--bytes-mib", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--cycles", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        raise SystemExit("this experiment requires at least four CUDA GPUs")

    if args.mode == "matrix":
        rows: list[MatrixResult] = []
        sizes = [int(value) * 1024 for value in args.sizes_kib.split(",")]
        for nbytes in sizes:
            for src in range(torch.cuda.device_count()):
                for dst in range(torch.cuda.device_count()):
                    if src == dst:
                        continue
                    pair_rows = directed_pair(src, dst, nbytes, args.warmup, args.cycles)
                    rows.extend(pair_rows)
                    values = [row.bandwidth_GBs for row in pair_rows]
                    print(
                        f"size={nbytes} src={src} dst={dst} "
                        f"median_GBs={statistics.median(values):.3f} "
                        f"p95_GBs={percentile(values, 0.95):.3f}",
                        flush=True,
                    )
    else:
        rows: list[ConcurrentResult] = []
        nbytes = args.bytes_mib * 1024 * 1024
        for name, edges in PATTERNS.items():
            pattern_rows = concurrent_pattern(
                name,
                edges,
                nbytes,
                args.warmup,
                args.iterations,
                args.cycles,
            )
            rows.extend(pattern_rows)
            values = [row.aggregate_bandwidth_GBs for row in pattern_rows]
            print(
                f"pattern={name} edges={len(edges)} "
                f"median_aggregate_GBs={statistics.median(values):.3f} "
                f"p95_GBs={percentile(values, 0.95):.3f}",
                flush=True,
            )

    if not all(row.correct for row in rows):
        raise SystemExit("copy correctness validation failed")
    write_rows(args.output, rows)
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
