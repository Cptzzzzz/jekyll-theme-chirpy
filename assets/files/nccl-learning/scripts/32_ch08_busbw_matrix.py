#!/usr/bin/env python3
"""Run nccl-tests across collectives/world sizes and independently verify bandwidth formulas."""

from __future__ import annotations

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
COLLECTIVES = {
    "all_reduce": "all_reduce_perf",
    "all_gather": "all_gather_perf",
    "reduce_scatter": "reduce_scatter_perf",
    "broadcast": "broadcast_perf",
    "reduce": "reduce_perf",
    "alltoall": "alltoall_perf",
}


def factor(collective: str, nranks: int) -> float:
    if collective == "all_reduce":
        return 2.0 * (nranks - 1) / nranks
    if collective in {"all_gather", "reduce_scatter", "alltoall"}:
        return (nranks - 1) / nranks
    return 1.0


def parse_rows(text: str, collective: str, nranks: int) -> list[dict[str, object]]:
    rows = []
    seen: dict[int, int] = defaultdict(int)
    for line in text.splitlines():
        if not ROW_RE.match(line):
            continue
        parts = line.split()
        if len(parts) != 21:
            continue
        size_bytes = int(parts[0])
        time_us = float(parts[5])
        printed_algbw = float(parts[6])
        printed_busbw = float(parts[7])
        calculated_algbw = size_bytes / (time_us * 1e-6) / 1e9
        calculated_busbw = calculated_algbw * factor(collective, nranks)
        cycle = seen[size_bytes]
        seen[size_bytes] += 1
        inplace_wrong = parts[16]
        rows.append(
            {
                "collective": collective,
                "nranks": nranks,
                "cycle": cycle,
                "size_bytes": size_bytes,
                "count_column": int(parts[1]),
                "type": parts[2],
                "redop": parts[3],
                "time_us": time_us,
                "printed_algbw_GBs": printed_algbw,
                "printed_busbw_GBs": printed_busbw,
                "factor": factor(collective, nranks),
                "calculated_algbw_GBs": calculated_algbw,
                "calculated_busbw_GBs": calculated_busbw,
                "algbw_abs_error": abs(printed_algbw - calculated_algbw),
                "busbw_abs_error": abs(printed_busbw - calculated_busbw),
                "wrong": int(parts[8]),
                "inplace_wrong": inplace_wrong,
                "iter_cv_percent": float(parts[12]),
            }
        )
    return rows


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def cv_percent(values: list[float]) -> float:
    mean = statistics.mean(values)
    return statistics.pstdev(values) / mean * 100.0 if mean else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/nccl-learning"))
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    run_dir = args.root / "logs/ch08_busbw" / args.run_id
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tests_dir = args.root / "third_party/nccl-tests"
    commands = []
    all_rows = []

    for collective, binary_name in COLLECTIVES.items():
        binary = tests_dir / "build" / binary_name
        if not binary.is_file():
            raise FileNotFoundError(binary)
        for nranks in (2, 3, 4):
            command = [
                str(binary),
                "-b", "1M",
                "-e", "64M",
                "-f", "64",
                "-g", str(nranks),
                "-w", "5",
                "-n", str(args.iters),
                "-N", str(args.cycles),
                "-c", "1",
                "-I", "1",
                "-z", "0",
                "-d", "float",
                "-o", "sum",
                "-r", "0",
            ]
            commands.append(" ".join(command))
            environment = os.environ.copy()
            environment["NCCL_DEBUG"] = "WARN"
            process = subprocess.run(
                command,
                cwd=tests_dir,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            log_path = raw_dir / f"{collective}_n{nranks}.log"
            log_path.write_text(process.stdout, encoding="utf-8")
            if process.returncode != 0:
                raise RuntimeError(f"{collective} n={nranks} failed; see {log_path}")
            rows = parse_rows(process.stdout, collective, nranks)
            if len(rows) != args.cycles * 2:
                raise RuntimeError(
                    f"{collective} n={nranks}: expected {args.cycles * 2} rows, got {len(rows)}"
                )
            if any(int(row["wrong"]) != 0 for row in rows):
                raise RuntimeError(f"{collective} n={nranks}: out-of-place correctness failed")
            if collective != "alltoall" and any(row["inplace_wrong"] != "0" for row in rows):
                raise RuntimeError(f"{collective} n={nranks}: in-place correctness failed")
            if collective == "alltoall" and any(row["inplace_wrong"] != "N/A" for row in rows):
                raise RuntimeError("alltoall in-place result should be N/A in this nccl-tests version")
            all_rows.extend(rows)
            print(
                f"collective={collective} nranks={nranks} rows={len(rows)} "
                f"max_bus_formula_error={max(float(row['busbw_abs_error']) for row in rows):.5f}",
                flush=True,
            )

    if len(all_rows) != len(COLLECTIVES) * 3 * 2 * args.cycles:
        raise RuntimeError("unexpected total measurement row count")
    max_alg_error = max(float(row["algbw_abs_error"]) for row in all_rows)
    max_bus_error = max(float(row["busbw_abs_error"]) for row in all_rows)
    # nccl-tests prints only two decimals, so independent calculations cannot
    # match more closely than the report's quantization.
    if max_alg_error > 0.011 or max_bus_error > 0.016:
        raise RuntimeError(
            f"formula verification exceeded print-rounding tolerance: alg={max_alg_error} bus={max_bus_error}"
        )

    with (run_dir / "raw_measurements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    groups: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in all_rows:
        groups[(str(row["collective"]), int(row["nranks"]), int(row["size_bytes"]))].append(row)
    summaries = []
    for (collective, nranks, size_bytes), rows in sorted(groups.items()):
        times = [float(row["time_us"]) for row in rows]
        algbws = [float(row["printed_algbw_GBs"]) for row in rows]
        busbws = [float(row["printed_busbw_GBs"]) for row in rows]
        summaries.append(
            {
                "collective": collective,
                "nranks": nranks,
                "size_bytes": size_bytes,
                "cycles": len(rows),
                "factor": factor(collective, nranks),
                "median_time_us": statistics.median(times),
                "p95_time_us": percentile(times, 0.95),
                "cycle_cv_percent": cv_percent(times),
                "median_algbw_GBs": statistics.median(algbws),
                "median_busbw_GBs": statistics.median(busbws),
            }
        )
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    manifest = [
        "experiment=ch08_busbw",
        f"run_id={args.run_id}",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"hostname={os.uname().nodename}",
        f"cycles={args.cycles}",
        f"iterations={args.iters}",
        "world_sizes=2,3,4",
        "requested_sizes=1M,64M",
        f"nccl_tests_commit={subprocess.check_output(['git', '-C', str(tests_dir), 'rev-parse', 'HEAD'], text=True).strip()}",
        f"nccl_source_commit={subprocess.check_output(['git', '-C', str(args.root / 'third_party/nccl-2.22.3'), 'rev-parse', 'HEAD'], text=True).strip()}",
    ]
    manifest.extend(f"command_{index:02d}={command}" for index, command in enumerate(commands, 1))
    (run_dir / "manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    large_rows = [row for row in summaries if int(row["size_bytes"]) > 60 * 1024 * 1024]
    lines = [
        "# Chapter 08 experiment summary",
        "",
        f"- Raw out-of-place measurement rows: `{len(all_rows)}`",
        f"- Formula max absolute error after two-decimal report quantization: algbw `{max_alg_error:.5f}` GB/s, busbw `{max_bus_error:.5f}` GB/s",
        "- Out-of-place correctness: `PASS` for all rows",
        "- Supported in-place correctness: `PASS` for all rows",
        "",
        "## Bus bandwidth factors",
        "",
        "| collective | N=2 | N=3 | N=4 |",
        "|---|---:|---:|---:|",
    ]
    for collective in COLLECTIVES:
        lines.append(
            f"| {collective} | {factor(collective, 2):.6f} | "
            f"{factor(collective, 3):.6f} | {factor(collective, 4):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Large-message observations",
            "",
            "The table uses the actual aligned size printed by nccl-tests, ten cycles, and out-of-place results.",
            "",
            "| collective | ranks | actual bytes | factor | median us | CV | algbw GB/s | busbw GB/s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in large_rows:
        lines.append(
            f"| {row['collective']} | {row['nranks']} | {row['size_bytes']} | "
            f"{float(row['factor']):.6f} | {float(row['median_time_us']):.3f} | "
            f"{float(row['cycle_cv_percent']):.2f}% | {float(row['median_algbw_GBs']):.2f} | "
            f"{float(row['median_busbw_GBs']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Assertions",
            "",
            "- Recomputed algbw used the actual reported bytes divided by reported time.",
            "- Recomputed busbw applied the exact GetBw factor for the collective and world size.",
            "- Three-rank alignment-adjusted sizes were not replaced with the requested CLI size.",
            "- busbw is a modeled normalization and no hardware link counter was read in this experiment.",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
