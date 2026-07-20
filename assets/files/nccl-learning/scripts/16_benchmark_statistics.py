#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROW_RE = re.compile(r"^\s*\d+\s+")


def parse_rows(text: str) -> list[dict[str, float | int | str]]:
    rows = []
    seen = defaultdict(int)
    for line in text.splitlines():
        if not ROW_RE.match(line):
            continue
        parts = line.split()
        if len(parts) != 21:
            continue
        size = int(parts[0])
        cycle = seen[size]
        seen[size] += 1
        rows.append(
            {
                "cycle": cycle,
                "bytes": size,
                "count": int(parts[1]),
                "type": parts[2],
                "redop": parts[3],
                "time_us": float(parts[5]),
                "algbw_GBs": float(parts[6]),
                "busbw_GBs": float(parts[7]),
                "wrong": int(parts[8]),
                "iter_min_us": float(parts[9]),
                "iter_max_us": float(parts[10]),
                "iter_p99_us": float(parts[11]),
                "iter_cv_percent": float(parts[12]),
                "inplace_time_us": float(parts[13]),
                "inplace_algbw_GBs": float(parts[14]),
                "inplace_busbw_GBs": float(parts[15]),
                "inplace_wrong": int(parts[16]),
                "inplace_iter_min_us": float(parts[17]),
                "inplace_iter_max_us": float(parts[18]),
                "inplace_iter_p99_us": float(parts[19]),
                "inplace_iter_cv_percent": float(parts[20]),
            }
        )
    return rows


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(q * len(ordered)) - 1]


def cv_percent(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean * 100.0 if len(values) > 1 else 0.0


def linear_fit(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator
    intercept = y_mean - slope * x_mean
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in points)
    total = sum((y - y_mean) ** 2 for y in ys)
    r_squared = 1.0 - residual / total if total else 1.0
    return intercept, slope, r_squared


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and summarize a statistically useful nccl-tests baseline")
    parser.add_argument("--root", type=Path, default=Path("/root/nccl-learning"))
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    out_dir = args.root / "logs/benchmark_statistics" / args.run_id
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    binary = args.root / "third_party/nccl-tests/build/all_reduce_perf"
    command = [
        str(binary),
        "-b", "1K",
        "-e", "256M",
        "-f", "4",
        "-g", "4",
        "-w", "10",
        "-n", str(args.iters),
        "-N", str(args.cycles),
        "-c", "1",
        "-I", "1",
    ]

    manifest = [
        f"run_id={args.run_id}",
        f"started_at={datetime.now(timezone.utc).isoformat()}",
        f"hostname={os.uname().nodename}",
        f"command={' '.join(command)}",
    ]
    (out_dir / "manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    proc = subprocess.run(
        command,
        cwd=args.root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (raw_dir / "all_reduce_statistics.log").write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(f"nccl-tests failed with exit code {proc.returncode}")

    rows = parse_rows(proc.stdout)
    expected_rows = args.cycles * 10
    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} rows, parsed {len(rows)}")
    if any(int(r["wrong"]) != 0 or int(r["inplace_wrong"]) != 0 for r in rows):
        raise RuntimeError("correctness check failed")

    write_csv(out_dir / "raw_measurements.csv", rows)
    grouped: dict[int, list[dict[str, float | int | str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["bytes"])].append(row)

    summaries = []
    for size, group in sorted(grouped.items()):
        times = [float(r["time_us"]) for r in group]
        algbws = [float(r["algbw_GBs"]) for r in group]
        busbws = [float(r["busbw_GBs"]) for r in group]
        inplace_times = [float(r["inplace_time_us"]) for r in group]
        median_time = statistics.median(times)
        median_algbw = statistics.median(algbws)
        median_busbw = statistics.median(busbws)
        calculated_algbw = size / (median_time * 1e-6) / 1e9
        calculated_busbw = calculated_algbw * 1.5
        summaries.append(
            {
                "bytes": size,
                "cycles": len(group),
                "median_time_us": median_time,
                "p95_time_us": percentile(times, 0.95),
                "cycle_cv_percent": cv_percent(times),
                "median_algbw_GBs": median_algbw,
                "median_busbw_GBs": median_busbw,
                "calculated_algbw_GBs": calculated_algbw,
                "calculated_busbw_GBs": calculated_busbw,
                "median_iter_cv_percent": statistics.median(float(r["iter_cv_percent"]) for r in group),
                "median_inplace_time_us": statistics.median(inplace_times),
                "inplace_cycle_cv_percent": cv_percent(inplace_times),
            }
        )

    for row in summaries:
        if int(row["bytes"]) >= 65536:
            relative_error = abs(float(row["median_busbw_GBs"]) - float(row["calculated_busbw_GBs"])) / float(
                row["calculated_busbw_GBs"]
            )
            if relative_error > 0.01:
                raise RuntimeError(f"busbw formula check failed for {row['bytes']} bytes: {relative_error}")

    write_csv(out_dir / "summary.csv", summaries)
    large = [
        (float(r["bytes"]), float(r["median_time_us"]) * 1e-6)
        for r in summaries
        if int(r["bytes"]) >= 16 * 1024 * 1024
    ]
    intercept_s, slope_s_per_byte, r_squared = linear_fit(large)
    fitted_algbw = 1.0 / slope_s_per_byte / 1e9
    small_latency = statistics.median(
        float(r["median_time_us"]) for r in summaries if int(r["bytes"]) <= 64 * 1024
    )
    max_cycle_cv = max(float(r["cycle_cv_percent"]) for r in summaries)

    lines = [
        "# NCCL Benchmark Statistics",
        "",
        f"Run ID: `{args.run_id}`",
        "",
        "## Summary",
        "",
        "| bytes | median us | P95 us | cycle CV | algbw GB/s | busbw GB/s | in-place us |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['bytes']} | {float(row['median_time_us']):.2f} | {float(row['p95_time_us']):.2f} | "
            f"{float(row['cycle_cv_percent']):.2f}% | {float(row['median_algbw_GBs']):.2f} | "
            f"{float(row['median_busbw_GBs']):.2f} | {float(row['median_inplace_time_us']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Model Checks",
            "",
            f"- Median latency for 1 KiB through 64 KiB: `{small_latency:.2f} us`.",
            f"- Large-message fitted algorithm bandwidth: `{fitted_algbw:.2f} GB/s`.",
            f"- Large-message fitted intercept: `{intercept_s * 1e6:.2f} us`.",
            f"- Large-message linear fit R-squared: `{r_squared:.6f}`.",
            f"- Maximum CV across ten cycle-level means: `{max_cycle_cv:.2f}%`.",
            "- All out-of-place and in-place correctness checks reported `#wrong=0`.",
            "- Recomputed AllReduce busbw used `algbw * 2 * (N-1) / N = algbw * 1.5` for four ranks.",
            "",
            "## Interpretation",
            "",
            "Small messages are dominated by a nearly fixed launch/synchronization cost. Large messages approach a linear bytes/time relation, while the transition region reflects protocol, channel and algorithm decisions. Cycle-level stability and per-iteration jitter are different quantities and must be reported separately.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (out_dir / "manifest.txt").open("a", encoding="utf-8") as f:
        f.write(f"completed_at={datetime.now(timezone.utc).isoformat()}\n")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
