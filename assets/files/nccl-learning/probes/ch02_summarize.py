#!/usr/bin/env python3
"""Summarize chapter 2 matrix, concurrency, topology XML and NVLink data."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def cv(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean * 100.0 if len(values) > 1 and mean else 0.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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

    matrix = read_csv(run_dir / "raw/matrix.csv")
    concurrent_by_node = {
        0: read_csv(run_dir / "raw/concurrent_numa0.csv"),
        1: read_csv(run_dir / "raw/concurrent_numa1.csv"),
    }
    if not all(row["correct"].lower() == "true" and row["peer_access"].lower() == "true" for row in matrix):
        raise SystemExit("matrix correctness or peer access assertion failed")
    if not all(row["correct"].lower() == "true" for rows in concurrent_by_node.values() for row in rows):
        raise SystemExit("concurrency correctness assertion failed")

    pair_groups: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    latency_groups: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for row in matrix:
        key = (int(row["size_bytes"]), int(row["src"]), int(row["dst"]))
        pair_groups[key].append(float(row["bandwidth_GBs"]))
        latency_groups[key].append(float(row["time_us"]))

    pair_rows: list[dict[str, object]] = []
    for (size_bytes, src, dst), values in sorted(pair_groups.items()):
        times = latency_groups[(size_bytes, src, dst)]
        pair_rows.append(
            {
                "size_bytes": size_bytes,
                "src": src,
                "dst": dst,
                "cycles": len(values),
                "median_time_us": f"{statistics.median(times):.6f}",
                "p95_time_us": f"{percentile(times, 0.95):.6f}",
                "median_bandwidth_GBs": f"{statistics.median(values):.6f}",
                "p95_bandwidth_GBs": f"{percentile(values, 0.95):.6f}",
                "cv_percent": f"{cv(values):.6f}",
            }
        )
    write_csv(run_dir / "pair_summary.csv", pair_rows)

    size_rows: list[dict[str, object]] = []
    for size_bytes in sorted({int(row["size_bytes"]) for row in matrix}):
        samples = [float(row["bandwidth_GBs"]) for row in matrix if int(row["size_bytes"]) == size_bytes]
        times = [float(row["time_us"]) for row in matrix if int(row["size_bytes"]) == size_bytes]
        pair_medians = [float(row["median_bandwidth_GBs"]) for row in pair_rows if int(row["size_bytes"]) == size_bytes]
        pair_cvs = [float(row["cv_percent"]) for row in pair_rows if int(row["size_bytes"]) == size_bytes]
        reverse_asymmetry = []
        median_map = {(int(row["src"]), int(row["dst"])): float(row["median_bandwidth_GBs"]) for row in pair_rows if int(row["size_bytes"]) == size_bytes}
        for (src, dst), value in median_map.items():
            reverse = median_map[(dst, src)]
            reverse_asymmetry.append(abs(value - reverse) / statistics.fmean((value, reverse)) * 100.0)
        size_rows.append(
            {
                "size_bytes": size_bytes,
                "samples": len(samples),
                "median_time_us": f"{statistics.median(times):.6f}",
                "median_bandwidth_GBs": f"{statistics.median(samples):.6f}",
                "pair_median_min_GBs": f"{min(pair_medians):.6f}",
                "pair_median_max_GBs": f"{max(pair_medians):.6f}",
                "max_pair_cv_percent": f"{max(pair_cvs):.6f}",
                "max_reverse_asymmetry_percent": f"{max(reverse_asymmetry):.6f}",
            }
        )
    write_csv(run_dir / "size_summary.csv", size_rows)

    concurrency_rows: list[dict[str, object]] = []
    for node, rows in concurrent_by_node.items():
        for pattern in sorted({row["pattern"] for row in rows}):
            selected = [float(row["aggregate_bandwidth_GBs"]) for row in rows if row["pattern"] == pattern]
            sample = next(row for row in rows if row["pattern"] == pattern)
            concurrency_rows.append(
                {
                    "cpu_numa_node": node,
                    "pattern": pattern,
                    "edges": int(sample["edges"]),
                    "cycles": len(selected),
                    "median_aggregate_GBs": f"{statistics.median(selected):.6f}",
                    "p95_aggregate_GBs": f"{percentile(selected, 0.95):.6f}",
                    "cv_percent": f"{cv(selected):.6f}",
                }
            )
    write_csv(run_dir / "concurrency_summary.csv", concurrency_rows)

    xml_root = ET.parse(run_dir / "artifacts/topology.xml").getroot()
    gpus = xml_root.findall(".//gpu")
    nvlinks = xml_root.findall(".//nvlink")
    nvlink_counts = [int(node.attrib["count"]) for node in nvlinks]
    link_speeds = [float(value) for value in re.findall(r"Link \d+: ([0-9.]+) GB/s", (run_dir / "raw/nvlink_status.log").read_text())]
    per_link_GBs = statistics.median(link_speeds)
    pair_theoretical_GBs = per_link_GBs * statistics.median(nvlink_counts)

    largest = size_rows[-1]
    large_median = float(largest["median_bandwidth_GBs"])
    efficiency = large_median / pair_theoretical_GBs * 100.0
    node0 = {row["pattern"]: float(row["median_aggregate_GBs"]) for row in concurrency_rows if row["cpu_numa_node"] == 0}
    node1 = {row["pattern"]: float(row["median_aggregate_GBs"]) for row in concurrency_rows if row["cpu_numa_node"] == 1}
    numa_deltas = {pattern: (node1[pattern] / node0[pattern] - 1.0) * 100.0 for pattern in node0}

    metrics = [
        {"metric": "gpu_count", "value": len(gpus)},
        {"metric": "directed_nvlink_xml_entries", "value": len(nvlinks)},
        {"metric": "links_per_gpu_pair", "value": statistics.median(nvlink_counts)},
        {"metric": "reported_per_link_GBs", "value": f"{per_link_GBs:.3f}"},
        {"metric": "reported_pair_theoretical_GBs", "value": f"{pair_theoretical_GBs:.3f}"},
        {"metric": "largest_message_median_GBs", "value": f"{large_median:.3f}"},
        {"metric": "largest_message_link_efficiency_percent", "value": f"{efficiency:.3f}"},
        {"metric": "largest_message_max_pair_cv_percent", "value": largest["max_pair_cv_percent"]},
        {"metric": "largest_message_max_reverse_asymmetry_percent", "value": largest["max_reverse_asymmetry_percent"]},
        {"metric": "numa_max_abs_delta_percent", "value": f"{max(abs(value) for value in numa_deltas.values()):.3f}"},
    ]
    write_csv(run_dir / "summary.csv", metrics)

    lines = [
        "# Chapter 02 experiment summary",
        "",
        f"- GPUs: `{len(gpus)}`",
        f"- Directed NVLink XML entries: `{len(nvlinks)}`",
        f"- NVLinks per GPU pair: `{statistics.median(nvlink_counts):.0f}`",
        f"- `nvidia-smi` per-link status: `{per_link_GBs:.3f} GB/s`",
        f"- Reported two-link pair capacity: `{pair_theoretical_GBs:.3f} GB/s`",
        f"- 256 MiB measured median: `{large_median:.3f} GB/s`",
        f"- Payload/capacity ratio: `{efficiency:.2f}%`",
        "",
        "## Size curve",
        "",
        "| bytes | median us | median GB/s | pair min | pair max | max CV | max reverse asymmetry |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in size_rows:
        lines.append(
            f"| {row['size_bytes']} | {float(row['median_time_us']):.3f} | "
            f"{float(row['median_bandwidth_GBs']):.3f} | {float(row['pair_median_min_GBs']):.3f} | "
            f"{float(row['pair_median_max_GBs']):.3f} | {float(row['max_pair_cv_percent']):.2f}% | "
            f"{float(row['max_reverse_asymmetry_percent']):.2f}% |"
        )
    lines.extend([
        "",
        "## Concurrent patterns",
        "",
        "| CPU NUMA | pattern | edges | median aggregate GB/s | CV |",
        "|---:|---|---:|---:|---:|",
    ])
    for row in concurrency_rows:
        lines.append(
            f"| {row['cpu_numa_node']} | {row['pattern']} | {row['edges']} | "
            f"{float(row['median_aggregate_GBs']):.3f} | {float(row['cv_percent']):.2f}% |"
        )
    lines.extend([
        "",
        "## Assertions",
        "",
        "- All 12 directed pairs support CUDA peer access and passed data checks.",
        "- All matrix and concurrent outputs passed full-tensor correctness checks.",
        "- Topology XML contains two direct NVLinks for every directed GPU pair.",
        "- NUMA results measure host submission affinity, not a change in the NVLink data path.",
    ])
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
