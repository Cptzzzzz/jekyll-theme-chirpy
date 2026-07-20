#!/usr/bin/env python3
"""Validate NCCL transport priority, connector indexes, fallback, and cost."""

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


CONFIGS = ("baseline", "shm_disabled", "p2p_disabled", "p2p_shm_disabled")
EXPECTED = {
    "baseline": "P2P",
    "shm_disabled": "P2P",
    "p2p_disabled": "SHM",
    "p2p_shm_disabled": "NET",
}
ROW_RE = re.compile(r"^\s*\d+\s+")
CONNECTION_RE = re.compile(r"Channel\s+(\d+)(?:/(\d+))?\s+.*?\bvia\s+(P2P|SHM|NET)/([^\s]+)")


def clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith("NCCL_") or key.startswith("TORCH_NCCL_"):
            env.pop(key)
    return env


def config_env(config: str, debug: str = "WARN") -> dict[str, str]:
    env = clean_env()
    if config in ("p2p_disabled", "p2p_shm_disabled"):
        env["NCCL_P2P_DISABLE"] = "1"
    if config in ("shm_disabled", "p2p_shm_disabled"):
        env["NCCL_SHM_DISABLE"] = "1"
    env["NCCL_DEBUG"] = debug
    if debug == "INFO":
        env["NCCL_DEBUG_SUBSYS"] = "ENV,INIT,P2P,SHM,NET"
    return env


def run(command: list[str], env: dict[str, str], path: Path, cwd: Path,
        check: bool = True) -> tuple[int, str]:
    process = subprocess.run(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, check=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(process.stdout, encoding="utf-8")
    if check and process.returncode != 0:
        raise RuntimeError(f"command failed with {process.returncode}; see {path}")
    return process.returncode, process.stdout


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def allreduce_command(binary: Path, begin: str, end: str, cycles: int,
        iterations: int) -> list[str]:
    return [str(binary), "-b", begin, "-e", end, "-f", "128", "-g", "4",
        "-w", "1" if cycles == 1 else "5", "-n", str(iterations),
        "-N", str(cycles), "-c", "1", "-I", "0", "-z", "0", "-u", "0",
        "-C", "0", "-a", "3", "-d", "float", "-o", "sum"]


def sendrecv_command(binary: Path) -> list[str]:
    return [str(binary), "-b", "4K", "-e", "4K", "-g", "4", "-w", "1",
        "-n", "1", "-N", "1", "-c", "1", "-I", "0", "-z", "0",
        "-u", "0", "-C", "0", "-a", "3", "-d", "float", "-o", "sum"]


def parse_path(config: str, workload: str, text: str) -> dict[str, object]:
    connections = []
    for line in text.splitlines():
        match = CONNECTION_RE.search(line)
        if match:
            connections.append(match.groups())
    transports = sorted({item[2] for item in connections})
    indexes = sorted({int(item[1]) for item in connections if item[1] is not None})
    variants = sorted({item[3] for item in connections})
    expected_index = 0 if workload == "all_reduce" else 1
    return {
        "config": config,
        "workload": workload,
        "expected_transport": EXPECTED[config],
        "observed_transports": ",".join(transports),
        "connection_lines": len(connections),
        "logged_conn_indexes": ",".join(str(value) for value in indexes),
        "expected_conn_index": expected_index,
        "index_visible_in_transport_log": int(expected_index in indexes),
        "transport_variants": ",".join(variants),
        "correctness_pass": int("# Out of bounds values : 0 OK" in text),
    }


def parse_perf(config: str, replicate: int, text: str,
        cycles: int) -> list[dict[str, object]]:
    rows = []
    seen: dict[int, int] = defaultdict(int)
    for line in text.splitlines():
        if not ROW_RE.match(line):
            continue
        fields = line.split()
        if len(fields) != 13 or int(fields[0]) < 4096:
            continue
        size = int(fields[0])
        cycle = seen[size]
        seen[size] += 1
        rows.append({
            "config": config,
            "replicate": replicate,
            "cycle": cycle,
            "size_bytes": size,
            "time_us": float(fields[5]),
            "algbw_GBs": float(fields[6]),
            "busbw_GBs": float(fields[7]),
            "wrong": int(fields[8]),
            "in_place_time_us": float(fields[9]),
            "in_place_busbw_GBs": float(fields[11]),
            "in_place_wrong": int(fields[12]),
        })
    if len(rows) != 3 * cycles or len(seen) != 3 or set(seen.values()) != {cycles}:
        raise RuntimeError(f"{config} r={replicate}: unexpected rows {len(rows)} {seen}")
    if any(row["wrong"] or row["in_place_wrong"] for row in rows):
        raise RuntimeError(f"correctness failure for {config} r={replicate}")
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, object], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(row["config"], row["size_bytes"])].append(row)
    output = []
    for (config, size), group in sorted(groups.items()):
        times = [float(row["time_us"]) for row in group]
        output.append({
            "config": config,
            "transport": EXPECTED[str(config)],
            "size_bytes": size,
            "samples": len(group),
            "median_time_us": statistics.median(times),
            "p95_time_us": percentile(times, .95),
            "cycle_cv_percent": statistics.pstdev(times) / statistics.fmean(times) * 100,
            "median_busbw_GBs": statistics.median(float(row["busbw_GBs"]) for row in group),
        })
    baseline = {int(row["size_bytes"]): float(row["median_time_us"])
        for row in output if row["config"] == "baseline"}
    for row in output:
        row["time_vs_baseline_percent"] = (
            float(row["median_time_us"]) / baseline[int(row["size_bytes"])] - 1) * 100
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/nccl-learning"))
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    run_dir = args.root / "logs/ch20_transport_selection" / args.run_id
    raw = run_dir / "raw"
    tests = args.root / "third_party/nccl-tests"
    allreduce = tests / "build/all_reduce_perf"
    sendrecv = tests / "build/sendrecv_perf"

    path_rows = []
    path_commands = {
        "all_reduce": allreduce_command(allreduce, "4K", "4K", 1, 1),
        "sendrecv": sendrecv_command(sendrecv),
    }
    for config in CONFIGS:
        for workload, command in path_commands.items():
            _, text = run(command, config_env(config, "INFO"),
                raw / "paths" / f"{config}_{workload}.log", tests)
            row = parse_path(config, workload, text)
            path_rows.append(row)
            if not row["correctness_pass"]:
                raise RuntimeError(f"path correctness failed: {config} {workload}")
            if row["observed_transports"] != EXPECTED[config]:
                raise RuntimeError(f"path mismatch: {row}")
            # P2P and NET include /connIndex in INFO; SHM's INFO format does not.
            if EXPECTED[config] != "SHM" and not row["index_visible_in_transport_log"]:
                raise RuntimeError(f"connIndex mismatch: {row}")
            print(f"[path] {config} {workload} PASS", flush=True)

    fail_env = config_env("p2p_shm_disabled", "INFO")
    fail_env["NCCL_NET_DISABLE_INTRA"] = "1"
    status, text = run(path_commands["all_reduce"], fail_env,
        raw / "failures/no_transport.log", tests, check=False)
    failure_rows = [{
        "case": "p2p_shm_net_intra_disabled",
        "exit_status": status,
        "signature": "No transport found for rank",
        "signature_count": text.count("No transport found for rank"),
        "nccl_system_error_visible": int("ncclSystemError" in text or "System error" in text),
        "observed": int(status != 0 and "No transport found for rank" in text),
    }]
    if not failure_rows[0]["observed"]:
        raise RuntimeError(f"no-transport failure not observed: {failure_rows}")

    perf_rows = []
    perf_command = allreduce_command(allreduce, "4K", "64M", args.cycles, args.iterations)
    for replicate, ordered in ((1, CONFIGS), (2, tuple(reversed(CONFIGS)))):
        for config in ordered:
            _, text = run(perf_command, config_env(config),
                raw / "performance" / f"{config}_r{replicate}.log", tests)
            perf_rows.extend(parse_perf(config, replicate, text, args.cycles))
            print(f"[perf] {config} r={replicate} PASS", flush=True)
    perf_summary = summarize(perf_rows)

    write_csv(run_dir / "path_summary.csv", path_rows)
    write_csv(run_dir / "failure_matrix.csv", failure_rows)
    write_csv(run_dir / "raw_measurements.csv", perf_rows)
    write_csv(run_dir / "performance_summary.csv", perf_summary)

    source = args.root / "third_party/nccl-2.22.3"
    manifest = [
        "experiment=ch20_transport_selection",
        f"run_id={args.run_id}",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"hostname={os.uname().nodename}",
        "world_size=4",
        f"configs={','.join(CONFIGS)}",
        f"cycles={args.cycles}",
        f"iterations={args.iterations}",
        f"measurement_rows={len(perf_rows)}",
        "path_acceptance=8/8 PASS",
        "correctness=PASS",
        "no_transport_failure=PASS",
        "nccl_runtime=2.22.3",
        "nccl_source_commit=" + subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
    ]
    (run_dir / "manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    by64 = {(str(row["config"]), int(row["size_bytes"])): row for row in perf_summary}
    lines = [
        "# Chapter 20 experiment summary", "",
        "- Path/correctness acceptance: 8/8 PASS",
        "- No-transport negative case: PASS",
        f"- Performance measurements: {len(perf_rows)}", "",
        "| config | transport | 64 MiB median us | busbw GB/s | vs baseline |",
        "|---|---|---:|---:|---:|",
    ]
    for config in CONFIGS:
        row = by64[(config, 64 << 20)]
        lines.append(f"| {config} | {EXPECTED[config]} | {row['median_time_us']:.3f} | "
            f"{row['median_busbw_GBs']:.2f} | {row['time_vs_baseline_percent']:+.2f}% |")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
