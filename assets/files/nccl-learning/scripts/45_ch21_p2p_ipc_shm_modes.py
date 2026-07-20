#!/usr/bin/env python3
"""Validate NCCL P2P mapping/read/CE modes, SHM CE modes, and pair effects."""

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


PATH_CONFIGS = (
    "direct_write", "direct_read", "cumem", "legacy_ipc", "p2p_ce",
    "level_nvl", "level_loc", "shm_recv_local", "shm_send_local",
    "shm_ce_send", "shm_ce_recv", "shm_ce_both",
)
PERF_CONFIGS = (
    "direct_write", "direct_read", "cumem", "legacy_ipc", "p2p_ce",
    "shm_recv_local", "shm_send_local", "shm_ce_send", "shm_ce_recv",
    "shm_ce_both",
)
PAIR_CONFIGS = ("direct_write", "direct_read", "cumem", "legacy_ipc")
EXPECTED_VARIANT = {
    "direct_write": "P2P/direct pointer",
    "direct_read": "P2P/direct pointer/read",
    "cumem": "P2P/CUMEM",
    "legacy_ipc": "P2P/IPC",
    "p2p_ce": "P2P/CUMEM/CE",
    "level_nvl": "P2P/direct pointer",
    "level_loc": "SHM/direct/direct",
    "shm_recv_local": "SHM/direct/direct",
    "shm_send_local": "SHM/direct/direct",
    "shm_ce_send": "SHM/CE/direct",
    "shm_ce_recv": "SHM/direct/CE",
    "shm_ce_both": "SHM/CE/CE",
}
ROW_RE = re.compile(r"^\s*\d+\s+")
VARIANT_RE = re.compile(
    r"via (P2P/(?:direct pointer|CUMEM|IPC)(?:/read)?(?:/CE)?|"
    r"SHM/(?:direct|CE)/(?:direct|CE))"
)


def clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith("NCCL_") or key.startswith("TORCH_NCCL_"):
            env.pop(key)
    return env


def config_env(config: str, debug: str = "WARN") -> dict[str, str]:
    env = clean_env()
    if config == "direct_read":
        env["NCCL_P2P_READ_ENABLE"] = "1"
    elif config == "cumem":
        env["NCCL_P2P_DIRECT_DISABLE"] = "1"
    elif config == "legacy_ipc":
        env.update({"NCCL_P2P_DIRECT_DISABLE": "1", "NCCL_CUMEM_ENABLE": "0"})
    elif config == "p2p_ce":
        env["NCCL_P2P_USE_CUDA_MEMCPY"] = "1"
    elif config == "level_nvl":
        env["NCCL_P2P_LEVEL"] = "NVL"
    elif config == "level_loc":
        env["NCCL_P2P_LEVEL"] = "LOC"
    elif config.startswith("shm_"):
        env["NCCL_P2P_DISABLE"] = "1"
        if config == "shm_send_local":
            env["NCCL_SHM_LOCALITY"] = "1"
        elif config.startswith("shm_ce_"):
            env["NCCL_SHM_USE_CUDA_MEMCPY"] = "1"
            env["NCCL_SHM_MEMCPY_MODE"] = {
                "shm_ce_send": "1", "shm_ce_recv": "2", "shm_ce_both": "3"
            }[config]
    env["NCCL_DEBUG"] = debug
    if debug == "INFO":
        env["NCCL_DEBUG_SUBSYS"] = "ENV,INIT,P2P,SHM"
    return env


def command(binary: Path, gpus: int, begin: str, end: str,
            cycles: int, iterations: int) -> list[str]:
    return [str(binary), "-b", begin, "-e", end, "-f", "128", "-g", str(gpus),
        "-w", "1" if cycles == 1 else "5", "-n", str(iterations),
        "-N", str(cycles), "-c", "1", "-I", "0", "-z", "0", "-u", "0",
        "-C", "0", "-a", "3", "-d", "float", "-o", "sum"]


def run(cmd: list[str], env: dict[str, str], path: Path,
        cwd: Path) -> str:
    process = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, check=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(process.stdout, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"failed with {process.returncode}; see {path}")
    return process.stdout


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"empty CSV {path}")
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


def path_row(config: str, text: str) -> dict[str, object]:
    variants = sorted(set(VARIANT_RE.findall(text)))
    expected = EXPECTED_VARIANT[config]
    row = {
        "config": config,
        "expected_variant": expected,
        "observed_variants": ",".join(variants),
        "variant_count": len(variants),
        "connection_lines": sum(1 for line in text.splitlines()
            if VARIANT_RE.search(line)),
        "correctness_pass": int("# Out of bounds values : 0 OK" in text),
        "read_mode": int("/read" in expected),
        "copy_engine_sides": expected.count("CE"),
        "shm_locality": 1 if config == "shm_send_local" else
            2 if config.startswith("shm_") else 0,
    }
    if variants != [expected] or not row["correctness_pass"]:
        raise RuntimeError(f"path acceptance failed: {row}")
    return row


def parse_perf(config: str, replicate: int, text: str,
               cycles: int, pair: str = "all4") -> list[dict[str, object]]:
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
            "config": config, "pair": pair, "replicate": replicate,
            "cycle": cycle, "size_bytes": size, "time_us": float(fields[5]),
            "algbw_GBs": float(fields[6]), "busbw_GBs": float(fields[7]),
            "wrong": int(fields[8]), "in_place_time_us": float(fields[9]),
            "in_place_busbw_GBs": float(fields[11]),
            "in_place_wrong": int(fields[12]),
        })
    expected_sizes = 1 if pair != "all4" else 3
    if len(rows) != expected_sizes * cycles or set(seen.values()) != {cycles}:
        raise RuntimeError(f"parse {config} {pair}: {len(rows)} {seen}")
    if any(row["wrong"] or row["in_place_wrong"] for row in rows):
        raise RuntimeError(f"correctness {config} {pair}")
    return rows


def summarize(rows: list[dict[str, object]],
              keys: tuple[str, ...]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        times = [float(row["time_us"]) for row in group]
        record = {name: value for name, value in zip(keys, key)}
        record.update({
            "samples": len(group),
            "median_time_us": statistics.median(times),
            "p95_time_us": percentile(times, .95),
            "cycle_cv_percent": statistics.pstdev(times) /
                statistics.fmean(times) * 100,
            "median_busbw_GBs": statistics.median(
                float(row["busbw_GBs"]) for row in group),
        })
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/nccl-learning"))
    parser.add_argument("--run-id",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--pair-cycles", type=int, default=5)
    args = parser.parse_args()

    run_dir = args.root / "logs/ch21_p2p_ipc_shm" / args.run_id
    raw = run_dir / "raw"
    tests = args.root / "third_party/nccl-tests"
    binary = tests / "build/all_reduce_perf"

    topo = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    (raw / "nvidia_smi_topo.log").parent.mkdir(parents=True, exist_ok=True)
    (raw / "nvidia_smi_topo.log").write_text(topo, encoding="utf-8")

    path_rows = []
    smoke = command(binary, 4, "4K", "4K", 1, 1)
    for config in PATH_CONFIGS:
        text = run(smoke, config_env(config, "INFO"),
            raw / "paths" / f"{config}.log", tests)
        path_rows.append(path_row(config, text))
        print(f"[path] {config} PASS", flush=True)

    perf_rows = []
    perf_cmd = command(binary, 4, "4K", "64M", args.cycles, args.iterations)
    for replicate, ordered in ((1, PERF_CONFIGS), (2, tuple(reversed(PERF_CONFIGS)))):
        for config in ordered:
            text = run(perf_cmd, config_env(config),
                raw / "performance" / f"{config}_r{replicate}.log", tests)
            perf_rows.extend(parse_perf(config, replicate, text, args.cycles))
            print(f"[perf] {config} r={replicate} PASS", flush=True)

    pair_rows = []
    pair_cmd = command(binary, 2, "64M", "64M", args.pair_cycles,
        args.iterations)
    pairs = tuple(f"{a}-{b}" for a in range(4) for b in range(a + 1, 4))
    for config in PAIR_CONFIGS:
        for pair in pairs:
            env = config_env(config)
            env["CUDA_VISIBLE_DEVICES"] = pair.replace("-", ",")
            text = run(pair_cmd, env,
                raw / "pairs" / f"{config}_{pair}.log", tests)
            pair_rows.extend(parse_perf(config, 1, text,
                args.pair_cycles, pair))
            print(f"[pair] {config} {pair} PASS", flush=True)

    perf_summary = summarize(perf_rows, ("config", "size_bytes"))
    pair_summary = summarize(pair_rows, ("config", "pair", "size_bytes"))
    stale_shm = list(Path("/dev/shm").glob("nccl-*"))
    if stale_shm:
        raise RuntimeError(f"stale NCCL SHM files after run: {stale_shm}")
    baseline = {int(row["size_bytes"]): float(row["median_time_us"])
        for row in perf_summary if row["config"] == "direct_write"}
    for row in perf_summary:
        row["time_vs_direct_write_percent"] = (
            float(row["median_time_us"]) /
            baseline[int(row["size_bytes"])] - 1) * 100

    write_csv(run_dir / "path_summary.csv", path_rows)
    write_csv(run_dir / "raw_measurements.csv", perf_rows)
    write_csv(run_dir / "performance_summary.csv", perf_summary)
    write_csv(run_dir / "pair_measurements.csv", pair_rows)
    write_csv(run_dir / "pair_summary.csv", pair_summary)

    source = args.root / "third_party/nccl-2.22.3"
    manifest = [
        "experiment=ch21_p2p_ipc_shm_modes", f"run_id={args.run_id}",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"hostname={os.uname().nodename}", "world_size=4",
        f"path_configs={len(PATH_CONFIGS)}",
        f"performance_configs={len(PERF_CONFIGS)}",
        f"performance_rows={len(perf_rows)}",
        f"pair_rows={len(pair_rows)}", "path_acceptance=12/12 PASS",
        "correctness=PASS", "shm_cleanup=PASS", "stale_shm_files=0",
        "nccl_runtime=2.22.3",
        "nccl_source_commit=" + subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True).strip(),
    ]
    (run_dir / "manifest.txt").write_text(
        "\n".join(manifest) + "\n", encoding="utf-8")

    at64 = {(str(row["config"]), int(row["size_bytes"])): row
        for row in perf_summary}
    lines = ["# Chapter 21 experiment summary", "",
        "- Path acceptance: 12/12 PASS",
        f"- Performance rows: {len(perf_rows)}",
        f"- Pair rows: {len(pair_rows)}", "- Correctness: PASS", "",
        "| config | variant | 64 MiB us | busbw GB/s | vs direct |",
        "|---|---|---:|---:|---:|"]
    for config in PERF_CONFIGS:
        row = at64[(config, 64 << 20)]
        lines.append(f"| {config} | {EXPECTED_VARIANT[config]} | "
            f"{row['median_time_us']:.3f} | {row['median_busbw_GBs']:.2f} | "
            f"{row['time_vs_direct_write_percent']:+.2f}% |")
    (run_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print(f"run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
