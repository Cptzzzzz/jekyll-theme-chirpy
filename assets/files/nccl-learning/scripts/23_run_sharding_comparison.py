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
    parser = argparse.ArgumentParser(description="Run repeated DDP/FSDP comparisons")
    parser.add_argument("--root", type=Path, default=Path("/root/nccl-learning"))
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--cycles", type=int, default=3)
    args = parser.parse_args()

    out_dir = args.root / "logs/sharding_comparison" / args.run_id
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = []
    manifest = [
        f"run_id={args.run_id}",
        f"started_at={datetime.now(timezone.utc).isoformat()}",
        f"hostname={os.uname().nodename}",
        f"cycles={args.cycles}",
        "model=12x Linear(2048,2048,bias=False) + ReLU",
        "optimizer=AdamW",
        "fsdp=FULL_SHARD with per-linear auto wrap",
    ]

    for mode in ["ddp", "fsdp"]:
        for cycle in range(args.cycles):
            command = [
                "torchrun",
                "--standalone",
                "--nproc_per_node=4",
                str(args.root / "scripts/23_sharding_worker.py"),
                "--mode",
                mode,
            ]
            if cycle == 0:
                manifest.append(f"mode_{mode}_command={' '.join(command)}")
            proc = subprocess.run(
                command,
                cwd=args.root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=300,
            )
            (raw_dir / f"{mode}_cycle_{cycle}.log").write_text(proc.stdout, encoding="utf-8")
            print(f"mode={mode} cycle={cycle} exit_code={proc.returncode}")
            if proc.returncode != 0:
                print(proc.stdout[-8000:])
                raise SystemExit(f"{mode} cycle {cycle} failed")
            match = RESULT_RE.search(proc.stdout)
            if not match:
                raise RuntimeError(f"{mode} cycle {cycle} did not emit RESULT_JSON")
            row = json.loads(match.group(1))
            row["cycle"] = cycle
            raw_rows.append(row)

    with (out_dir / "raw_measurements.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    rows = []
    for mode in ["ddp", "fsdp"]:
        group = [r for r in raw_rows if r["mode"] == mode]
        medians = [float(r["median_step_ms"]) for r in group]
        rows.append(
            {
                "mode": mode,
                "cycles": len(group),
                "median_step_ms": statistics.median(medians),
                "p95_step_ms": percentile(medians, 0.95),
                "cycle_cv_percent": cv_percent(medians),
                "max_allocated_mib": max(int(r["allocated_bytes"]) for r in group) / 1024 / 1024,
                "max_peak_allocated_mib": max(int(r["peak_allocated_bytes"]) for r in group) / 1024 / 1024,
                "parameters_equal": all(bool(r["parameters_equal"]) for r in group),
            }
        )
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ddp = next(r for r in rows if r["mode"] == "ddp")
    fsdp = next(r for r in rows if r["mode"] == "fsdp")
    memory_reduction = 1.0 - fsdp["max_allocated_mib"] / ddp["max_allocated_mib"]
    slowdown = fsdp["median_step_ms"] / ddp["median_step_ms"] - 1.0
    lines = [
        "# DDP and FSDP Sharding Comparison",
        "",
        f"Run ID: `{args.run_id}`",
        "",
        "| mode | cycles | median step ms | P95 ms | cycle CV | allocated MiB | peak MiB | replicas equal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['cycles']} | {row['median_step_ms']:.3f} | {row['p95_step_ms']:.3f} | "
            f"{row['cycle_cv_percent']:.2f}% | {row['max_allocated_mib']:.1f} | "
            f"{row['max_peak_allocated_mib']:.1f} | {int(row['parameters_equal'])} |"
        )
    lines.extend(
        [
            "",
            f"- Steady allocated-memory reduction of FSDP versus DDP: `{memory_reduction * 100:.2f}%`.",
            f"- Median-step change of FSDP versus DDP: `{slowdown * 100:.2f}%`.",
            "- FSDP uses per-linear FULL_SHARD, so it exchanges parameter shards around each wrapped layer instead of retaining a full replica.",
            "- This is a mechanism experiment on one model shape, not a universal FSDP performance recommendation.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest.append(f"completed_at={datetime.now(timezone.utc).isoformat()}")
    (out_dir / "manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
