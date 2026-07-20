#!/usr/bin/env python3
"""Aggregate chapter 5 collective contract results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted((args.run_dir / "raw/results").glob("rank*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if len(rows) != 56:
        raise SystemExit(f"expected 56 rows, got {len(rows)}")
    if not all(row["correct"].lower() == "true" for row in rows):
        raise SystemExit("collective contract correctness failure")

    native_rows = []
    for path in sorted((args.run_dir / "raw/native").glob("rank*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            native_rows.extend(csv.DictReader(handle))
    if len(native_rows) != 36:
        raise SystemExit(f"expected 36 native rows, got {len(native_rows)}")
    if not all(row["correct"].lower() == "true" for row in native_rows):
        raise SystemExit("native NCCL contract correctness failure")

    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["operation"], row["variant"])].append(row)
    summary_rows = [
        {
            "operation": operation,
            "variant": variant,
            "rank_rows": len(selected),
            "all_correct": all(row["correct"].lower() == "true" for row in selected),
        }
        for (operation, variant), selected in sorted(groups.items())
    ]
    with (args.run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    native_groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in native_rows:
        native_groups[(row["operation"], row["variant"], row["pointer_relation"])].append(row)
    native_summary_rows = [
        {
            "operation": operation,
            "variant": variant,
            "pointer_relation": pointer_relation,
            "rank_rows": len(selected),
            "all_correct": all(row["correct"].lower() == "true" for row in selected),
        }
        for (operation, variant, pointer_relation), selected in sorted(native_groups.items())
    ]
    with (args.run_dir / "native_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(native_summary_rows[0]))
        writer.writeheader()
        writer.writerows(native_summary_rows)

    lines = [
        "# Chapter 05 experiment summary",
        "",
        f"- PyTorch rank rows: `{len(rows)}`",
        f"- PyTorch contract variants: `{len(summary_rows)}`",
        f"- Native NCCL rank rows: `{len(native_rows)}`",
        f"- Native NCCL pointer/layout variants: `{len(native_summary_rows)}`",
        "- Full-tensor/reference and pointer-layout checks: `PASS`",
        "",
        "| operation | variant | rank rows | result |",
        "|---|---|---:|---|",
    ]
    for row in summary_rows:
        lines.append(f"| {row['operation']} | {row['variant']} | {row['rank_rows']} | PASS |")
    lines.extend([
        "",
        "## Native NCCL C API",
        "",
        "| operation | variant | pointer relation | rank rows | result |",
        "|---|---|---|---:|---|",
    ])
    for row in native_summary_rows:
        lines.append(
            f"| {row['operation']} | {row['variant']} | `{row['pointer_relation']}` | "
            f"{row['rank_rows']} | PASS |"
        )
    lines.extend([
        "",
        "## Assertions",
        "",
        "- Rooted operations were checked on two different roots.",
        "- Reduce non-root receive contents were treated as outside the contract.",
        "- Equal and uneven AllToAll layouts were checked element by element.",
        "- Grouped ring Send/Recv delivered source-encoded payloads on all ranks.",
        "- Out-of-range broadcast root was rejected on every rank.",
        "- Native AllGather and ReduceScatter in-place pointer offsets were exercised directly.",
        "- Native Reduce accepted a null receive pointer on every non-root rank.",
    ])
    (args.run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
