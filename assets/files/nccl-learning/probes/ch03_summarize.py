#!/usr/bin/env python3
"""Aggregate chapter 3 timing and stream-scope CSV files."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_many(directory: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(directory.glob("rank*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"no rank CSV files in {directory}")
    return rows


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def finite(rows: list[dict[str, str]], key: str) -> list[float]:
    return [value for row in rows if math.isfinite(value := float(row[key]))]


def stats(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.fmean(values)
    cv = statistics.pstdev(values) / mean * 100.0 if len(values) > 1 and mean else 0.0
    return statistics.median(values), percentile(values, 0.95), cv


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir

    timing_sets = {
        False: read_many(run_dir / "raw/nonblocking"),
        True: read_many(run_dir / "raw/blocking"),
    }
    stream_rows = read_many(run_dir / "raw/stream_scope")
    if not all(row["correct"].lower() == "true" for rows in timing_sets.values() for row in rows):
        raise SystemExit("timing correctness failure")
    if not all(row["correct"].lower() == "true" for row in stream_rows):
        raise SystemExit("stream-scope correctness failure")

    summary_rows: list[dict[str, object]] = []
    for blocking, rows in timing_sets.items():
        groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            groups[(int(row["size_bytes"]), row["mode"])].append(row)
        for (size_bytes, mode), selected in sorted(groups.items()):
            host_median, host_p95, host_cv = stats(finite(selected, "host_call_us"))
            stream_median, stream_p95, stream_cv = stats(finite(selected, "stream_dependency_us"))
            enqueue_values = finite(selected, "host_enqueue_us")
            wait_values = finite(selected, "host_wait_or_sync_us")
            summary_rows.append(
                {
                    "blocking_wait": int(blocking),
                    "size_bytes": size_bytes,
                    "mode": mode,
                    "samples": len(selected),
                    "median_host_call_us": f"{host_median:.6f}",
                    "p95_host_call_us": f"{host_p95:.6f}",
                    "host_cv_percent": f"{host_cv:.6f}",
                    "median_host_enqueue_us": f"{statistics.median(enqueue_values):.6f}" if enqueue_values else "nan",
                    "median_host_wait_or_sync_us": f"{statistics.median(wait_values):.6f}" if wait_values else "nan",
                    "median_stream_dependency_us": f"{stream_median:.6f}",
                    "p95_stream_dependency_us": f"{stream_p95:.6f}",
                    "stream_cv_percent": f"{stream_cv:.6f}",
                    "completed_after_enqueue_rate": f"{statistics.fmean(finite(selected, 'completed_after_enqueue')):.6f}" if enqueue_values else "nan",
                    "completed_after_wait_rate": f"{statistics.fmean(finite(selected, 'completed_after_wait')):.6f}" if wait_values else "nan",
                }
            )
    write_csv(run_dir / "timing_summary.csv", summary_rows)

    stream_host = finite(stream_rows, "host_wait_us")
    stream_sleep = finite(stream_rows, "sleep_gpu_us")
    free_rate = statistics.fmean(row["free_stream_finished_before_wait_stream"].lower() == "true" for row in stream_rows)
    completed_rate = statistics.fmean(row["completed_after_wait"].lower() == "true" for row in stream_rows)
    stream_summary = [
        {"metric": "samples", "value": len(stream_rows)},
        {"metric": "median_host_wait_us", "value": f"{statistics.median(stream_host):.6f}"},
        {"metric": "median_sleep_gpu_us", "value": f"{statistics.median(stream_sleep):.6f}"},
        {"metric": "free_stream_before_wait_stream_rate", "value": f"{free_rate:.6f}"},
        {"metric": "completed_after_wait_rate", "value": f"{completed_rate:.6f}"},
    ]
    write_csv(run_dir / "stream_scope_summary.csv", stream_summary)

    sizes = sorted({int(row["size_bytes"]) for row in summary_rows})
    lines = [
        "# Chapter 03 experiment summary",
        "",
        "## Host and device completion",
        "",
        "| blocking | bytes | mode | host call us | host wait/sync us | stream dependency us | completed after wait |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        if int(row["size_bytes"]) in sizes:
            wait_value = row["median_host_wait_or_sync_us"]
            completed = row["completed_after_wait_rate"]
            lines.append(
                f"| {row['blocking_wait']} | {row['size_bytes']} | {row['mode']} | "
                f"{float(row['median_host_call_us']):.3f} | "
                f"{float(wait_value):.3f} | " if wait_value != "nan" else
                f"| {row['blocking_wait']} | {row['size_bytes']} | {row['mode']} | {float(row['median_host_call_us']):.3f} | n/a | "
            )
            lines[-1] += f"{float(row['median_stream_dependency_us']):.3f} | {float(completed):.2f} |" if completed != "nan" else f"{float(row['median_stream_dependency_us']):.3f} | n/a |"

    lines.extend([
        "",
        "## Stream-scoped wait",
        "",
        f"- Samples: `{len(stream_rows)}`",
        f"- Median host `work.wait()`: `{statistics.median(stream_host):.3f} us`",
        f"- Median queued GPU sleep: `{statistics.median(stream_sleep):.3f} us`",
        f"- Free stream completed before wait-dependent stream: `{free_rate * 100:.1f}%`",
        f"- Work already complete immediately after nonblocking wait: `{completed_rate * 100:.1f}%`",
        "",
        "## Assertions",
        "",
        "- Every timed output passed full-tensor AllReduce correctness.",
        "- Nonblocking `wait()` establishes a CUDA-stream dependency without requiring host completion.",
        "- Blocking-wait mode polls Work completion on the host.",
    ])
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
