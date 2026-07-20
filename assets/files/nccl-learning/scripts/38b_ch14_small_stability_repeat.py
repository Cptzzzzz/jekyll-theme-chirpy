#!/usr/bin/env python3
"""Repeat noisy Chapter 14 small-message groups in independent processes."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def load_ch14_module(root: Path):
    path = root / "scripts/38_ch14_protocol_wire_sync.py"
    spec = importlib.util.spec_from_file_location("ch14_protocol", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--root", type=Path, default=Path("/root/nccl-learning"))
    parser.add_argument("--threshold", type=float, default=5.0)
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    ch14 = load_ch14_module(args.root)
    with (args.run_dir / "summary.csv").open(encoding="utf-8") as handle:
        original = list(csv.DictReader(handle))
    original_by_key = {
        (row["algorithm"], row["protocol"], int(row["size_bytes"])): row
        for row in original
    }
    noisy_combinations = sorted(
        {
            (row["algorithm"], row["protocol"])
            for row in original
            if int(row["size_bytes"]) <= 65536
            and float(row["cycle_cv_percent"]) > args.threshold
        }
    )
    if not noisy_combinations:
        raise RuntimeError("no noisy small-message combinations")

    binary = args.root / "third_party/nccl-tests/build/all_reduce_perf"
    log_dir = args.run_dir / "raw/small_stability_repeat"
    log_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    commands = []
    run_index = 0
    for replicate, combinations in (
        (1, noisy_combinations),
        (2, tuple(reversed(noisy_combinations))),
    ):
        for algorithm, protocol in combinations:
            run_index += 1
            command = [
                str(binary),
                "-b", "4",
                "-e", "64K",
                "-f", "2",
                "-g", "4",
                "-w", "5",
                "-n", str(args.iterations),
                "-N", str(args.cycles),
                "-c", "1",
                "-I", "0",
                "-z", "0",
                "-u", "0",
                "-C", "0",
                "-a", "3",
                "-d", "float",
                "-o", "sum",
            ]
            command_text = (
                f"NCCL_ALGO={algorithm} NCCL_PROTO={protocol} "
                "NCCL_MIN_NCHANNELS=12 NCCL_MAX_NCHANNELS=12 "
                + " ".join(command)
            )
            commands.append(command_text)
            process = subprocess.run(
                command,
                cwd=args.root / "third_party/nccl-tests",
                env={
                    **os.environ,
                    "NCCL_ALGO": algorithm,
                    "NCCL_PROTO": protocol,
                    "NCCL_MIN_NCHANNELS": "12",
                    "NCCL_MAX_NCHANNELS": "12",
                    "NCCL_DEBUG": "WARN",
                },
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            path = log_dir / (
                f"{run_index:02d}_{algorithm.lower()}_{protocol.lower()}_r{replicate}.log"
            )
            path.write_text(process.stdout, encoding="utf-8")
            if process.returncode != 0:
                raise RuntimeError(f"failed; see {path}")
            rows.extend(
                ch14.parse_perf(
                    process.stdout,
                    algorithm,
                    protocol,
                    replicate,
                    args.cycles,
                    15,
                )
            )
            print(
                f"[{run_index:02d}/{2*len(noisy_combinations)}] "
                f"{algorithm}+{protocol} r={replicate} PASS",
                flush=True,
            )

    summaries = ch14.summarize(
        rows, ("algorithm", "protocol", "size_bytes")
    )
    for row in summaries:
        key = (row["algorithm"], row["protocol"], int(row["size_bytes"]))
        old = original_by_key[key]
        old_median = float(old["median_time_us"])
        median = float(row["median_time_us"])
        group_rows = [
            item
            for item in rows
            if item["algorithm"] == row["algorithm"]
            and item["protocol"] == row["protocol"]
            and int(item["size_bytes"]) == int(row["size_bytes"])
        ]
        row["original_samples"] = int(old["samples"])
        row["original_median_us"] = old_median
        row["original_cv_percent"] = float(old["cycle_cv_percent"])
        row["median_shift_percent"] = (median / old_median - 1.0) * 100.0
        row["samples_over_2x_median"] = sum(
            float(item["time_us"]) > 2 * median for item in group_rows
        )

    ch14.write_csv(args.run_dir / "small_repeat_raw.csv", rows)
    ch14.write_csv(args.run_dir / "small_repeat_summary.csv", summaries)
    manifest = [
        "experiment=ch14_small_stability_repeat",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"source_run={args.run_dir}",
        f"threshold_percent={args.threshold}",
        f"combinations={','.join('+'.join(combo) for combo in noisy_combinations)}",
        "sizes=4B..64KiB_factor2",
        f"cycles={args.cycles}",
        f"iterations={args.iterations}",
        f"rows={len(rows)}",
    ]
    manifest.extend(
        f"command_{index:02d}={command}"
        for index, command in enumerate(commands, 1)
    )
    (args.run_dir / "small_repeat_manifest.txt").write_text(
        "\n".join(manifest) + "\n", encoding="utf-8"
    )
    noisy = [
        row for row in summaries if float(row["cycle_cv_percent"]) > args.threshold
    ]
    print(
        f"rows={len(rows)} groups={len(summaries)} "
        f"groups_cv_gt_{args.threshold:g}={len(noisy)}"
    )
    for row in sorted(
        noisy,
        key=lambda item: float(item["cycle_cv_percent"]),
        reverse=True,
    )[:10]:
        print(
            row["algorithm"],
            row["protocol"],
            row["size_bytes"],
            f"median={float(row['median_time_us']):.3f}",
            f"cv={float(row['cycle_cv_percent']):.2f}%",
            f"outliers={row['samples_over_2x_median']}",
        )


if __name__ == "__main__":
    main()
