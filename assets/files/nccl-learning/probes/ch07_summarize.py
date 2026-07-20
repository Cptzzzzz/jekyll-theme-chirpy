#!/usr/bin/env python3
"""Aggregate chapter 7 contracts, timings, memory, and kernel evidence."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    return {
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "cv": statistics.pstdev(values) / mean * 100.0 if mean else 0.0,
    }


def read_rank_csv(directory: Path, suffix: str) -> list[dict[str, str]]:
    rows = []
    for path in sorted(directory.glob(f"rank*_{suffix}.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def size_label(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes // (1024 * 1024)} MiB"
    if size_bytes >= 1024:
        return f"{size_bytes // 1024} KiB"
    return f"{size_bytes} B"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    results_dir = args.run_dir / "raw/results"

    contracts = read_rank_csv(results_dir, "contracts")
    timings = read_rank_csv(results_dir, "timings")
    memory = read_rank_csv(results_dir, "memory")
    if len(contracts) != 20 or not all(row["correct"].lower() == "true" for row in contracts):
        raise SystemExit("contract matrix must contain 20 passing rank rows")
    if len(timings) != 400 or not all(row["correct"].lower() == "true" for row in timings):
        raise SystemExit("timing matrix must contain 400 passing rank rows")
    if len(memory) != 16 or not all(row["correct"].lower() == "true" for row in memory):
        raise SystemExit("memory matrix must contain 16 passing rank rows")

    rank_cycles: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for row in timings:
        rank_cycles[(int(row["size_bytes"]), row["mode"], int(row["cycle"]))].append(
            float(row["duration_us"])
        )
    if any(len(values) != 4 for values in rank_cycles.values()):
        raise SystemExit("each timing cycle must have four rank values")

    mode_cycles: dict[tuple[int, str], list[float]] = defaultdict(list)
    for (size_bytes, mode, _cycle), values in rank_cycles.items():
        mode_cycles[(size_bytes, mode)].append(max(values))
    if any(len(values) != 10 for values in mode_cycles.values()):
        raise SystemExit("each size/mode must have ten cycles")

    timing_summary = []
    for (size_bytes, mode), values in sorted(
        mode_cycles.items(), key=lambda item: (item[0][0], 0 if item[0][1] == "direct" else 1)
    ):
        measured = describe(values)
        timing_summary.append(
            {
                "size_bytes": size_bytes,
                "mode": mode,
                "cycles": len(values),
                "median_us": measured["median"],
                "p95_us": measured["p95"],
                "cv_percent": measured["cv"],
                "payload_GBs": size_bytes / (measured["median"] / 1e6) / 1e9,
            }
        )
    by_key = {(row["size_bytes"], row["mode"]): row for row in timing_summary}
    for size_bytes in sorted({int(row["size_bytes"]) for row in timing_summary}):
        direct = by_key[(size_bytes, "direct")]
        decomposed = by_key[(size_bytes, "decomposed")]
        direct["decomposed_over_direct"] = ""
        decomposed["decomposed_over_direct"] = (
            float(decomposed["median_us"]) / float(direct["median_us"])
        )
    with (args.run_dir / "timing_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timing_summary[0]))
        writer.writeheader()
        writer.writerows(timing_summary)

    memory_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in memory:
        memory_groups[row["mode"]].append(row)
    memory_summary = []
    for mode, rows in sorted(memory_groups.items()):
        if len(rows) != 4:
            raise SystemExit(f"memory mode {mode} must have four ranks")
        memory_summary.append(
            {
                "mode": mode,
                "steady_MiB_max_rank": max(int(row["steady_delta_bytes"]) for row in rows)
                / (1024 * 1024),
                "peak_MiB_max_rank": max(int(row["peak_delta_bytes"]) for row in rows)
                / (1024 * 1024),
            }
        )
    with (args.run_dir / "memory_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(memory_summary[0]))
        writer.writeheader()
        writer.writerows(memory_summary)

    with (args.run_dir / "kernel_summary.csv").open(newline="", encoding="utf-8") as handle:
        kernels = list(csv.DictReader(handle))
    kernel_totals = defaultdict(int)
    for row in kernels:
        kernel_totals[row["mode"]] += int(row["kernel_instances"])
    if kernel_totals != {"direct": 20, "decomposed": 40}:
        raise SystemExit(f"unexpected kernel totals: {dict(kernel_totals)}")

    lines = [
        "# Chapter 07 experiment summary",
        "",
        "## Contract matrix",
        "",
        "- Contract rank rows: `20`",
        "- Exact full-tensor checks: `PASS`",
        "- Negative transforms rejected: `PASS`",
        "",
        "| contract | rank rows | result |",
        "|---|---:|---|",
    ]
    contract_groups = defaultdict(list)
    for row in contracts:
        contract_groups[row["name"]].append(row)
    for name, rows in sorted(contract_groups.items()):
        lines.append(f"| {name} | {len(rows)} | PASS |")

    lines.extend(
        [
            "",
            "## Timing",
            "",
            "Each mode used 3 warmups and 10 alternating-order cycles. Per-cycle latency is the maximum across four ranks.",
            "",
            "| size | mode | median us | P95 us | CV | payload GB/s | RS+AG / AR |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in timing_summary:
        ratio = row["decomposed_over_direct"]
        if ratio == "":
            ratio_text = "-"
        else:
            direct_row = by_key[(int(row["size_bytes"]), "direct")]
            stable = (
                float(row["cv_percent"]) <= 5.0
                and float(direct_row["cv_percent"]) <= 5.0
            )
            ratio_text = f"{float(ratio):.3f}x" if stable else "unstable"
        lines.append(
            f"| {size_label(int(row['size_bytes']))} | {row['mode']} | "
            f"{float(row['median_us']):.3f} | {float(row['p95_us']):.3f} | "
            f"{float(row['cv_percent']):.2f}% | {float(row['payload_GBs']):.3f} | {ratio_text} |"
        )
    lines.extend(
        [
            "",
            "Only size pairs where both modes have CV <= 5% receive a numeric ratio. Other ratios remain in the CSV as descriptive observations and are not accepted as precision conclusions.",
        ]
    )

    lines.extend(
        [
            "",
            "## Allocated memory",
            "",
            "Payload is `256 MiB` per rank. Values are deltas from each rank's clean ProcessGroup baseline and use the maximum rank.",
            "",
            "| mode | steady allocated MiB | peak allocated MiB |",
            "|---|---:|---:|",
        ]
    )
    for row in memory_summary:
        lines.append(
            f"| {row['mode']} | {float(row['steady_MiB_max_rank']):.1f} | "
            f"{float(row['peak_MiB_max_rank']):.1f} |"
        )

    lines.extend(
        [
            "",
            "## Nsight Systems kernel attribution",
            "",
            "One process managed four GPUs; the measured NVTX range contained 5 iterations at 64 MiB.",
            "",
            f"- Direct AllReduce NCCL kernels: `{kernel_totals['direct']}` (`5 x 4 GPUs`).",
            f"- ReduceScatter + AllGather NCCL kernels: `{kernel_totals['decomposed']}` (`5 x 2 collectives x 4 GPUs`).",
            "- Correctness endpoints: `PASS`.",
        ]
    )
    (args.run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
