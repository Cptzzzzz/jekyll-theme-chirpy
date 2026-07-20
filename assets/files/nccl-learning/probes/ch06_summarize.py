#!/usr/bin/env python3
"""Aggregate native Group evidence and ProcessGroup ordering failures."""

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


def stats(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    return {
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "cv": statistics.pstdev(values) / mean * 100.0 if mean else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    native_rows = []
    for path in sorted((args.run_dir / "raw/native").glob("rank*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            native_rows.extend(csv.DictReader(handle))
    if len(native_rows) != 76:
        raise SystemExit(f"expected 76 native rows, got {len(native_rows)}")
    if not all(row["correct"].lower() == "true" for row in native_rows):
        raise SystemExit("native Group semantic or data correctness failure")

    semantics = defaultdict(list)
    completion_by_cycle: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in native_rows:
        if row["kind"] == "semantic":
            semantics[row["variant"]].append(row)
        else:
            completion_by_cycle[int(row["cycle"])].append(row)
    if len(semantics) != 9 or any(len(rows) != 4 for rows in semantics.values()):
        raise SystemExit("unexpected native semantic coverage")
    if len(completion_by_cycle) != 10 or any(len(rows) != 4 for rows in completion_by_cycle.values()):
        raise SystemExit("unexpected completion timing coverage")

    cycle_rows = []
    for cycle, selected in sorted(completion_by_cycle.items()):
        cycle_rows.append(
            {
                "cycle": cycle,
                "max_host_us": max(float(row["host_us"]) for row in selected),
                "max_device_us": max(float(row["device_us"]) for row in selected),
                "events_ready_immediately": sum(
                    int(row["event_ready_after_group_end"]) for row in selected
                ),
            }
        )
    with (args.run_dir / "completion_cycles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cycle_rows[0]))
        writer.writeheader()
        writer.writerows(cycle_rows)

    host_stats = stats([row["max_host_us"] for row in cycle_rows])
    device_stats = stats([row["max_device_us"] for row in cycle_rows])
    ready_count = sum(row["events_ready_immediately"] for row in cycle_rows)

    with (args.run_dir / "failure_summary.csv").open(newline="", encoding="utf-8") as handle:
        failures = list(csv.DictReader(handle))
    if len(failures) != 5 or not all(row["accepted"] == "1" for row in failures):
        raise SystemExit("ordering failure matrix did not satisfy expectations")

    lines = [
        "# Chapter 06 experiment summary",
        "",
        "## Native Group semantics",
        "",
        f"- Native rank rows: `{len(native_rows)}`",
        f"- Semantic variants: `{len(semantics)}`",
        "- Full-tensor and return-code checks: `PASS`",
        "",
        "| variant | rank rows | result |",
        "|---|---:|---|",
    ]
    for variant, selected in sorted(semantics.items()):
        lines.append(f"| {variant} | {len(selected)} | PASS |")
    lines.extend(
        [
            "",
            "## GroupEnd versus device completion",
            "",
            "Payload: one in-place `256 MiB` AllReduce, 3 warmups and 10 measured cycles.",
            "Per-cycle values use the maximum across four ranks.",
            "",
            "| metric | median (us) | P95 (us) | CV |",
            "|---|---:|---:|---:|",
            f"| host GroupStart/API/GroupEnd interval | {host_stats['median']:.3f} | {host_stats['p95']:.3f} | {host_stats['cv']:.2f}% |",
            f"| CUDA event completion interval | {device_stats['median']:.3f} | {device_stats['p95']:.3f} | {device_stats['cv']:.2f}% |",
            "",
            f"- CUDA end events already ready immediately after GroupEnd: `{ready_count}/40` rank-cycles.",
            f"- Observed median device/host interval ratio: `{device_stats['median'] / host_stats['median']:.2f}x`.",
            f"- Host interval CV is `{host_stats['cv']:.2f}%`; the host median and ratio are descriptive evidence only, not a precision performance baseline.",
            "",
            "## ProcessGroup ordering matrix",
            "",
            "| case | debug | return | duration (s) | classified outcome | result |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for row in failures:
        lines.append(
            f"| {row['label']} | {row['distributed_debug']} | {row['returncode']} | "
            f"{row['duration_s']} | {row['outcome']} | PASS |"
        )
    lines.extend(
        [
            "",
            "## Assertions",
            "",
            "- Nested Group depth two launched three correctly ordered AllReduces.",
            "- GroupEnd without GroupStart returned `ncclInvalidUsage` on every rank.",
            "- Invalid root was returned by both the inner API and balanced outer GroupEnd.",
            "- A valid collective succeeded after group error cleanup on the blocking communicator.",
            "- Nonblocking init, first Group submission, and Finalize reached terminal success through async-error polling.",
            "- `TORCH_DISTRIBUTED_DEBUG=DETAIL` rejected op and shape mismatch through fingerprints.",
            "- With debug OFF, the mismatch was not reclassified as a fingerprint failure.",
            "- A skipped rank reached the timeout path after a healthy communicator warmup.",
        ]
    )
    (args.run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
