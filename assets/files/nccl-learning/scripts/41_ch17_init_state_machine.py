#!/usr/bin/env python3
"""Measure NCCL init phases and collect bootstrap failure signatures."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


TIMING_RE = re.compile(
    r"Init timings: rank (\d+) nranks (\d+) total ([0-9.]+) "
    r"\(kernels ([0-9.]+), bootstrap ([0-9.]+), allgathers ([0-9.]+), "
    r"topo ([0-9.]+), graphs ([0-9.]+), connections ([0-9.]+), rest ([0-9.]+)\)"
)
ROW_RE = re.compile(r"^\s*\d+\s+")
PHASES = ("kernels_s", "bootstrap_s", "allgathers_s", "topo_s", "graphs_s", "connections_s", "rest_s")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith("NCCL_") or key.startswith("TORCH_NCCL_"):
            env.pop(key)
    return env


def run(command: list[str], env: dict[str, str], path: Path,
        cwd: Path, timeout: float | None = None) -> tuple[int, str, float]:
    start = time.perf_counter()
    try:
        process = subprocess.run(command, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            check=False, timeout=timeout)
        status, text = process.returncode, process.stdout
    except subprocess.TimeoutExpired as error:
        status = 124
        parts = []
        for part in (error.stdout, error.stderr):
            if isinstance(part, bytes):
                parts.append(part.decode(errors="replace"))
            elif part:
                parts.append(part)
        text = "".join(parts)
    elapsed = time.perf_counter() - start
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return status, text, elapsed


def parse_timing(text: str, cycle: int, wall_s: float) -> list[dict[str, object]]:
    rows = []
    for match in TIMING_RE.finditer(text):
        values = [float(value) for value in match.groups()[2:]]
        row: dict[str, object] = {
            "world_size": int(match.group(2)), "cycle": cycle,
            "rank": int(match.group(1)), "command_wall_s": wall_s,
            "total_s": values[0],
        }
        for phase, value in zip(PHASES, values[1:]):
            row[phase] = value
        row["phase_sum_s"] = sum(values[1:])
        row["phase_sum_error_ms"] = (sum(values[1:]) - values[0]) * 1000
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[int(row["world_size"])].append(row)
    output = []
    for world, group in sorted(groups.items()):
        totals = [float(row["total_s"]) for row in group]
        result: dict[str, object] = {
            "world_size": world, "rank_samples": len(group),
            "cycles": len({int(row["cycle"]) for row in group}),
            "median_total_ms": statistics.median(totals) * 1000,
            "p95_total_ms": percentile(totals, .95) * 1000,
            "total_cv_percent": statistics.pstdev(totals) / statistics.fmean(totals) * 100,
        }
        for phase in PHASES:
            vals = [float(row[phase]) for row in group]
            result[f"median_{phase[:-2]}_ms"] = statistics.median(vals) * 1000
        output.append(result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/nccl-learning"))
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--cycles", type=int, default=10)
    args = parser.parse_args()
    run_dir = args.root / "logs/ch17_init" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    binary = args.root / "third_party/nccl-tests/build/all_reduce_perf"
    cwd = args.root / "third_party/nccl-tests"
    all_rows = []
    commands = []

    order = (1, 2, 3, 4, 4, 3, 2, 1, 2, 4, 1, 3, 3, 1, 4, 2,
             1, 3, 2, 4, 4, 2, 3, 1, 2, 1, 4, 3, 3, 4, 1, 2,
             2, 1, 3, 4, 4, 3, 1, 2)
    counts = defaultdict(int)
    for index, world in enumerate(order, 1):
        counts[world] += 1
        command = [str(binary), "-b", "4K", "-e", "4K", "-g", str(world),
                   "-w", "1", "-n", "1", "-N", "1", "-c", "1", "-I", "0",
                   "-z", "0", "-u", "0", "-C", "0", "-a", "3", "-d", "float", "-o", "sum"]
        env = clean_env()
        env.update({"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT"})
        status, text, wall = run(command, env, run_dir / f"raw/scaling/{index:02d}_n{world}.log", cwd, 60)
        if status != 0 or "# Out of bounds values : 0 OK" not in text:
            raise RuntimeError(f"scaling run failed: index={index} world={world}")
        rows = parse_timing(text, counts[world], wall)
        if len(rows) != world:
            raise RuntimeError(f"expected {world} timing rows, got {len(rows)}")
        all_rows.extend(rows)
        commands.append(" ".join(command))
        print(f"[scaling {index:02d}/40] N={world} PASS", flush=True)

    probe = args.root / "private/ch17_init" / args.run_id / "ch17_bootstrap_failures"
    probe.parent.mkdir(parents=True, exist_ok=True)
    compile_command = ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra",
        "-I/usr/local/cuda/include", str(args.root / "probes/ch17_bootstrap_failures.cc"),
        "-L/usr/local/cuda/lib64", "-lcudart", "-L/lib/x86_64-linux-gnu", "-lnccl", "-o", str(probe)]
    status, _, _ = run(compile_command, clean_env(), run_dir / "raw/failures/build.log", args.root, 60)
    if status != 0:
        raise RuntimeError("probe compilation failed")

    failure_rows = []
    env = clean_env(); env.update({"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "ENV,INIT"})
    bad_command = [str(binary), "-b", "4K", "-e", "4K", "-g", "1", "-n", "1"]
    bad_env = dict(env); bad_env["NCCL_COMM_ID"] = "not-an-address"
    status, text, wall = run(bad_command, bad_env, run_dir / "raw/failures/invalid_comm_id.log", cwd, 10)
    invalid_pass = status != 0 and "Invalid NCCL_COMM_ID" in text
    failure_rows.append({"case": "invalid_comm_id", "exit_status": status, "wall_s": wall,
                         "expected_signature": "Invalid NCCL_COMM_ID", "observed": int(invalid_pass)})

    status, text, wall = run([str(probe), "duplicate-gpu"], env,
                             run_dir / "raw/failures/duplicate_gpu.log", args.root, 15)
    duplicate_pass = status == 0 and "invalid usage" in text and "code=5" in text
    failure_rows.append({"case": "duplicate_gpu", "exit_status": status, "wall_s": wall,
                         "expected_signature": "ncclCommInitAll early invalid usage code=5", "observed": int(duplicate_pass)})

    trace_env = clean_env(); trace_env.update({"NCCL_DEBUG": "TRACE", "NCCL_DEBUG_SUBSYS": "INIT"})
    status, text, wall = run([str(probe), "missing-rank"], trace_env,
                             run_dir / "raw/failures/missing_rank.log", args.root, 5)
    missing_pass = status == 124 and "MISSING_RANK entering" in text and "Init COMPLETE" not in text
    failure_rows.append({"case": "missing_rank", "exit_status": status, "wall_s": wall,
                         "expected_signature": "external timeout while bootstrap root waits rank 1", "observed": int(missing_pass)})
    if not all(int(row["observed"]) for row in failure_rows):
        raise RuntimeError(f"failure signature mismatch: {failure_rows}")

    summaries = summarize(all_rows)
    write_csv(run_dir / "init_timings_raw.csv", all_rows)
    write_csv(run_dir / "init_timings_summary.csv", summaries)
    write_csv(run_dir / "failure_matrix.csv", failure_rows)
    manifest = ["experiment=ch17_init_state_machine", f"run_id={args.run_id}",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}", f"hostname={os.uname().nodename}",
        "world_sizes=1,2,3,4", f"cycles_per_world={args.cycles}", f"timing_rows={len(all_rows)}",
        "scaling_correctness=PASS", "failure_cases=invalid_comm_id,duplicate_gpu,missing_rank",
        f"nccl_source_commit={subprocess.check_output(['git','-C',str(args.root/'third_party/nccl-2.22.3'),'rev-parse','HEAD'],text=True).strip()}"]
    (run_dir / "manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    lines = ["# Chapter 17 experiment summary", "", f"- Timing rows: {len(all_rows)}", "- Correctness: PASS", "- Failure signatures: 3/3 PASS", "",
             "| N | samples | total median ms | P95 | CV | kernels | bootstrap | allgathers | topo | graphs | connections | rest |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| {row['world_size']} | {row['rank_samples']} | {row['median_total_ms']:.2f} | {row['p95_total_ms']:.2f} | {row['total_cv_percent']:.2f}% | {row['median_kernels_ms']:.2f} | {row['median_bootstrap_ms']:.2f} | {row['median_allgathers_ms']:.2f} | {row['median_topo_ms']:.2f} | {row['median_graphs_ms']:.2f} | {row['median_connections_ms']:.2f} | {row['median_rest_ms']:.2f} |")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
