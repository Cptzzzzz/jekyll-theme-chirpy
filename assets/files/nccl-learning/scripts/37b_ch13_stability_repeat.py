#!/usr/bin/env python3
"""Repeat noisy Chapter 13 groups without replacing the original samples."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


ROW_RE = re.compile(r"^\s*\d+\s+")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def cv_percent(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean * 100.0


def parse_rows(
    text: str,
    world_size: int,
    algorithm: str,
    size_bytes: int,
    replicate: int,
    cycles: int,
) -> list[dict[str, object]]:
    rows = []
    for line in text.splitlines():
        if not ROW_RE.match(line):
            continue
        fields = line.split()
        if len(fields) != 13 or int(fields[0]) != size_bytes:
            continue
        rows.append(
            {
                "world_size": world_size,
                "algorithm": algorithm,
                "protocol": "Simple",
                "channels": 12,
                "size_bytes": size_bytes,
                "replicate": replicate,
                "cycle": len(rows),
                "time_us": float(fields[5]),
                "busbw_GBs": float(fields[7]),
                "wrong": int(fields[8]),
                "in_place_time_us": float(fields[9]),
                "in_place_busbw_GBs": float(fields[11]),
                "in_place_wrong": int(fields[12]),
            }
        )
    if len(rows) != cycles:
        raise RuntimeError(f"expected {cycles} rows, got {len(rows)}")
    if any(
        int(row["wrong"]) != 0 or int(row["in_place_wrong"]) != 0
        for row in rows
    ):
        raise RuntimeError("correctness failure")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"empty output: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    with (args.run_dir / "summary.csv").open(encoding="utf-8") as handle:
        noisy = [
            row
            for row in csv.DictReader(handle)
            if float(row["cycle_cv_percent"]) > args.threshold
        ]
    if not noisy:
        raise RuntimeError("no noisy groups above threshold")

    root = Path("/root/nccl-learning")
    binary = root / "third_party/nccl-tests/build/all_reduce_perf"
    log_dir = args.run_dir / "raw/stability_repeat"
    log_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    commands = []
    run_index = 0
    for group in noisy:
        world_size = int(group["world_size"])
        algorithm = group["algorithm"]
        size_bytes = int(group["size_bytes"])
        for replicate in (1, 2):
            run_index += 1
            command = [
                str(binary),
                "-b", str(size_bytes),
                "-e", str(size_bytes),
                "-g", str(world_size),
                "-w", "5",
                "-n", str(args.iterations),
                "-N", str(args.cycles),
                "-c", "1",
                "-I", "0",
                "-z", "0",
                "-C", "0",
                "-a", "3",
                "-d", "float",
                "-o", "sum",
            ]
            command_text = (
                f"NCCL_ALGO={algorithm} NCCL_PROTO=Simple "
                "NCCL_MIN_NCHANNELS=12 NCCL_MAX_NCHANNELS=12 "
                + " ".join(command)
            )
            commands.append(command_text)
            process = subprocess.run(
                command,
                cwd=root / "third_party/nccl-tests",
                env={
                    **os.environ,
                    "NCCL_ALGO": algorithm,
                    "NCCL_PROTO": "Simple",
                    "NCCL_MIN_NCHANNELS": "12",
                    "NCCL_MAX_NCHANNELS": "12",
                    "NCCL_DEBUG": "WARN",
                },
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            log_path = log_dir / (
                f"{run_index:02d}_n{world_size}_{algorithm.lower()}_"
                f"{size_bytes}_r{replicate}.log"
            )
            log_path.write_text(process.stdout, encoding="utf-8")
            if process.returncode != 0:
                raise RuntimeError(f"failed; see {log_path}")
            all_rows.extend(
                parse_rows(
                    process.stdout,
                    world_size,
                    algorithm,
                    size_bytes,
                    replicate,
                    args.cycles,
                )
            )

    groups: dict[tuple[int, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in all_rows:
        groups[
            (
                int(row["world_size"]),
                str(row["algorithm"]),
                int(row["size_bytes"]),
            )
        ].append(row)
    summaries = []
    for (world_size, algorithm, size_bytes), rows in sorted(groups.items()):
        times = [float(row["time_us"]) for row in rows]
        median_time = statistics.median(times)
        summaries.append(
            {
                "world_size": world_size,
                "algorithm": algorithm,
                "size_bytes": size_bytes,
                "samples": len(times),
                "median_time_us": median_time,
                "p95_time_us": percentile(times, 0.95),
                "cycle_cv_percent": cv_percent(times),
                "min_time_us": min(times),
                "max_time_us": max(times),
                "samples_over_2x_median": sum(
                    time > 2 * median_time for time in times
                ),
                "median_busbw_GBs": statistics.median(
                    float(row["busbw_GBs"]) for row in rows
                ),
            }
        )
    write_csv(args.run_dir / "stability_repeat_raw.csv", all_rows)
    write_csv(args.run_dir / "stability_repeat_summary.csv", summaries)
    manifest = [
        f"source_run={args.run_dir}",
        f"threshold_percent={args.threshold}",
        f"cycles={args.cycles}",
        f"iterations={args.iterations}",
        f"groups={len(noisy)}",
        f"rows={len(all_rows)}",
    ]
    manifest.extend(
        f"command_{index:02d}={command}"
        for index, command in enumerate(commands, 1)
    )
    (args.run_dir / "stability_repeat_manifest.txt").write_text(
        "\n".join(manifest) + "\n", encoding="utf-8"
    )
    for row in summaries:
        print(row)


if __name__ == "__main__":
    main()
