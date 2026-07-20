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
CASES = [
    ("Ring", "Simple"),
    ("Ring", "LL"),
    ("Ring", "LL128"),
    ("Tree", "Simple"),
    ("Tree", "LL"),
    ("Tree", "LL128"),
]


def parse_rows(text: str) -> list[dict[str, float | int | str]]:
    rows = []
    seen = defaultdict(int)
    for line in text.splitlines():
        if not ROW_RE.match(line):
            continue
        parts = line.split()
        if len(parts) != 13:
            continue
        size = int(parts[0])
        cycle = seen[size]
        seen[size] += 1
        rows.append(
            {
                "cycle": cycle,
                "bytes": size,
                "time_us": float(parts[5]),
                "algbw_GBs": float(parts[6]),
                "busbw_GBs": float(parts[7]),
                "wrong": int(parts[8]),
                "inplace_time_us": float(parts[9]),
                "inplace_algbw_GBs": float(parts[10]),
                "inplace_busbw_GBs": float(parts[11]),
                "inplace_wrong": int(parts[12]),
            }
        )
    return rows


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(q * len(ordered)) - 1]


def cv_percent(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean * 100.0 if len(values) > 1 else 0.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep NCCL AllReduce algorithms and protocols")
    parser.add_argument("--root", type=Path, default=Path("/root/nccl-learning"))
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    out_dir = args.root / "logs/protocol_algorithm" / args.run_id
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    binary = args.root / "third_party/nccl-tests/build/all_reduce_perf"
    all_rows = []
    manifest = [
        f"run_id={args.run_id}",
        f"started_at={datetime.now(timezone.utc).isoformat()}",
        f"hostname={os.uname().nodename}",
        f"cycles={args.cycles}",
        f"iters={args.iters}",
    ]

    for algorithm, protocol in CASES:
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
        ]
        env = os.environ.copy()
        env["NCCL_ALGO"] = algorithm
        env["NCCL_PROTO"] = protocol
        case_name = f"{algorithm.lower()}_{protocol.lower()}"
        manifest.append(f"case_{case_name}=NCCL_ALGO={algorithm} NCCL_PROTO={protocol} {' '.join(command)}")
        proc = subprocess.run(
            command,
            cwd=args.root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (raw_dir / f"{case_name}.log").write_text(proc.stdout, encoding="utf-8")
        print(f"case={case_name} exit_code={proc.returncode}")
        if proc.returncode != 0:
            print(proc.stdout[-4000:])
            raise SystemExit(f"case {case_name} failed")
        rows = parse_rows(proc.stdout)
        if len(rows) != args.cycles * 10:
            raise RuntimeError(f"case {case_name}: expected {args.cycles * 10} rows, parsed {len(rows)}")
        for row in rows:
            row["algorithm"] = algorithm
            row["protocol"] = protocol
            all_rows.append(row)

    if any(int(r["wrong"]) != 0 or int(r["inplace_wrong"]) != 0 for r in all_rows):
        raise RuntimeError("at least one correctness check failed")
    write_csv(out_dir / "raw_measurements.csv", all_rows)

    grouped = defaultdict(list)
    for row in all_rows:
        grouped[(str(row["algorithm"]), str(row["protocol"]), int(row["bytes"]))].append(row)
    summaries = []
    for (algorithm, protocol, size), group in sorted(grouped.items()):
        times = [float(r["time_us"]) for r in group]
        summaries.append(
            {
                "algorithm": algorithm,
                "protocol": protocol,
                "bytes": size,
                "cycles": len(group),
                "median_time_us": statistics.median(times),
                "p95_time_us": percentile(times, 0.95),
                "cycle_cv_percent": cv_percent(times),
                "median_algbw_GBs": statistics.median(float(r["algbw_GBs"]) for r in group),
                "median_busbw_GBs": statistics.median(float(r["busbw_GBs"]) for r in group),
            }
        )
    write_csv(out_dir / "summary.csv", summaries)

    selected_sizes = [1024, 65536, 1048576, 16777216, 268435456]
    lines = [
        "# NCCL Algorithm and Protocol Sweep",
        "",
        f"Run ID: `{args.run_id}`",
        "",
        "## Selected Results",
        "",
        "| algorithm | protocol | 1 KiB us | 64 KiB us | 1 MiB us | 16 MiB GB/s | 256 MiB GB/s | max CV |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for algorithm, protocol in CASES:
        case = [r for r in summaries if r["algorithm"] == algorithm and r["protocol"] == protocol]
        by_size = {int(r["bytes"]): r for r in case}
        max_cv = max(float(r["cycle_cv_percent"]) for r in case)
        lines.append(
            f"| {algorithm} | {protocol} | {float(by_size[1024]['median_time_us']):.2f} | "
            f"{float(by_size[65536]['median_time_us']):.2f} | {float(by_size[1048576]['median_time_us']):.2f} | "
            f"{float(by_size[16777216]['median_busbw_GBs']):.2f} | "
            f"{float(by_size[268435456]['median_busbw_GBs']):.2f} | {max_cv:.2f}% |"
        )

    lines.extend(["", "## Best Combination by Message Size", "", "| bytes | best | median us | busbw GB/s |", "|---:|---|---:|---:|"])
    for size in selected_sizes:
        candidates = [r for r in summaries if int(r["bytes"]) == size]
        best = min(candidates, key=lambda r: float(r["median_time_us"]))
        lines.append(
            f"| {size} | {best['algorithm']} + {best['protocol']} | {float(best['median_time_us']):.2f} | "
            f"{float(best['median_busbw_GBs']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            "- Every case completed all out-of-place and in-place checks with `#wrong=0`.",
            "- Algorithm and protocol were forced only for isolation; these settings are not production recommendations.",
            "- The best combination changes with message size, so a single 64 MiB comparison cannot describe NCCL tuning behavior.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest.append(f"completed_at={datetime.now(timezone.utc).isoformat()}")
    (out_dir / "manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
