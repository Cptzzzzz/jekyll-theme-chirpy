#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


RESULT_RE = re.compile(r"^RESULT_JSON=(\{.*\})$", re.MULTILINE)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(q * len(ordered)) - 1]


def cv_percent(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean * 100.0 if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DDP bucket sweep")
    parser.add_argument("--root", type=Path, default=Path("/root/nccl-learning"))
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--cycles", type=int, default=5)
    args = parser.parse_args()

    out_dir = args.root / "logs/ddp_buckets" / args.run_id
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = []
    manifest = [
        f"run_id={args.run_id}",
        f"started_at={datetime.now(timezone.utc).isoformat()}",
        f"hostname={os.uname().nodename}",
        "model=12x Linear(2048,2048,bias=False) + ReLU",
        "warmup_steps=5",
        "measured_steps=15",
        f"cycles={args.cycles}",
    ]

    for bucket in [1, 4, 16, 64]:
        for cycle in range(args.cycles):
            command = [
                "torchrun",
                "--standalone",
                "--nproc_per_node=4",
                str(args.root / "scripts/21_ddp_bucket_worker.py"),
                "--bucket-cap-mb",
                str(bucket),
            ]
            if cycle == 0:
                manifest.append(f"bucket_{bucket}_command={' '.join(command)}")
            proc = subprocess.run(
                command,
                cwd=args.root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=300,
            )
            (raw_dir / f"bucket_{bucket}mb_cycle_{cycle}.log").write_text(proc.stdout, encoding="utf-8")
            print(f"bucket_cap_mb={bucket} cycle={cycle} exit_code={proc.returncode}")
            if proc.returncode != 0:
                print(proc.stdout[-6000:])
                raise SystemExit(f"bucket {bucket} MiB cycle {cycle} failed")
            match = RESULT_RE.search(proc.stdout)
            if not match:
                raise RuntimeError(f"bucket {bucket} MiB cycle {cycle} did not emit RESULT_JSON")
            row = json.loads(match.group(1))
            row["cycle"] = cycle
            if not row["parameters_equal"]:
                raise RuntimeError(f"bucket {bucket} MiB cycle {cycle} produced divergent parameters")
            raw_rows.append(row)

    with (out_dir / "raw_measurements.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    rows = []
    for bucket in [1, 4, 16, 64]:
        group = [r for r in raw_rows if r["bucket_cap_mb"] == bucket]
        step_medians = [float(r["median_step_ms"]) for r in group]
        layouts = {str(r["rebuilt_bucket_sizes"]) for r in group}
        if len(layouts) != 1:
            raise RuntimeError(f"bucket {bucket} MiB rebuild layout changed between cycles: {layouts}")
        rows.append(
            {
                "bucket_cap_mb": bucket,
                "cycles": len(group),
                "rebuilt_bucket_sizes": layouts.pop(),
                "median_step_ms": statistics.median(step_medians),
                "p95_step_ms": percentile(step_medians, 0.95),
                "cycle_cv_percent": cv_percent(step_medians),
                "median_backward_comm_ms": statistics.median(
                    float(r["avg_backward_comm_time_ns"]) / 1e6 for r in group
                ),
                "median_backward_overlap_ms": statistics.median(
                    float(r["avg_backward_overlap_time_ns"]) / 1e6 for r in group
                ),
                "peak_memory_bytes": max(int(r["peak_memory_bytes"]) for r in group),
                "parameters_equal": all(bool(r["parameters_equal"]) for r in group),
            }
        )

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# DDP Bucket Sweep",
        "",
        f"Run ID: `{args.run_id}`",
        "",
        "| cap MiB | rebuilt bucket sizes | median step ms | P95 ms | cycle CV | comm ms | overlap ms | peak MiB | replicas equal |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['bucket_cap_mb']} | `{row['rebuilt_bucket_sizes']}` | {row['median_step_ms']:.3f} | "
            f"{row['p95_step_ms']:.3f} | {row['cycle_cv_percent']:.2f}% | {row['median_backward_comm_ms']:.3f} | "
            f"{row['median_backward_overlap_ms']:.3f} | {row['peak_memory_bytes'] / 1024 / 1024:.1f} | "
            f"{int(row['parameters_equal'])} |"
        )
    lines.extend(
        [
            "",
            "## Fixed Facts",
            "",
            f"- Total parameter size: `{raw_rows[0]['parameter_bytes'] / 1024 / 1024:.1f} MiB`.",
            f"- One weight tensor size: `{raw_rows[0]['single_weight_bytes'] / 1024 / 1024:.1f} MiB`.",
            "- DDP cannot split one parameter tensor across buckets; the configured cap is therefore not a strict maximum.",
            "- Every configuration ended with identical parameter replicas across all ranks.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest.append(f"completed_at={datetime.now(timezone.utc).isoformat()}")
    (out_dir / "manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
