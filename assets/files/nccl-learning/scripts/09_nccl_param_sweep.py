#!/usr/bin/env python3
import csv
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ.get("ROOT", "/root/nccl-learning"))
REPO = Path(os.environ.get("REPO", ROOT / "third_party/nccl-tests"))
BUILD = REPO / "build"
OUT_ROOT = Path(os.environ.get("OUT_DIR", ROOT / "logs/sweeps"))


ROW_RE = re.compile(r"^\s*(\d+)\s+")


def run_cmd(name, bin_name, args, env=None):
    env = env or {}
    out_dir = run_cmd.out_dir
    log = out_dir / f"{name}.log"
    cmd = [str(BUILD / bin_name)] + args
    run_env = os.environ.copy()
    run_env.update(env)

    print(f"== {name} ==")
    print("env:", " ".join(f"{k}={v}" for k, v in env.items()) or "<none>")
    print("cmd:", " ".join(cmd))

    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        env=run_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log.write_text(proc.stdout)
    print(f"exit_code={proc.returncode} log={log}")
    if proc.returncode != 0:
        print(proc.stdout[-4000:])
    return proc.returncode, log, proc.stdout


def parse_rows(text):
    rows = []
    for line in text.splitlines():
        if not ROW_RE.match(line):
            continue
        parts = line.split()
        if len(parts) < 13:
            continue
        # nccl-tests prints out-of-place and in-place metrics in the same row.
        # Use out-of-place columns by default because all experiments are comparable.
        rows.append(
            {
                "bytes": int(parts[0]),
                "count": parts[1],
                "type": parts[2],
                "redop": parts[3],
                "root": parts[4],
                "time_us": float(parts[5]),
                "algbw_GBs": float(parts[6]),
                "busbw_GBs": float(parts[7]),
                "wrong": int(parts[8]),
                "inplace_time_us": float(parts[9]),
                "inplace_algbw_GBs": float(parts[10]),
                "inplace_busbw_GBs": float(parts[11]),
                "inplace_wrong": int(parts[12]),
            }
        )
    return rows


def append_rows(all_rows, experiment, param, value, op, gpus, log, rows):
    for row in rows:
        all_rows.append(
            {
                "experiment": experiment,
                "param": param,
                "value": value,
                "op": op,
                "gpus": gpus,
                "log": str(log.relative_to(ROOT)),
                **row,
            }
        )


def write_csv(rows, path):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(x):
    return f"{x:.2f}"


def write_summary(rows, path):
    lines = []
    lines.append("# NCCL Parameter Sweep Summary")
    lines.append("")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    def latest(experiment):
        return [r for r in rows if r["experiment"] == experiment]

    # Message size: one row per scanned size.
    size_rows = latest("message_size")
    if size_rows:
        lines.append("## Message Size Sweep")
        lines.append("")
        lines.append("| bytes | time_us | algbw_GBs | busbw_GBs |")
        lines.append("|---:|---:|---:|---:|")
        for r in sorted(size_rows, key=lambda x: x["bytes"]):
            lines.append(
                f"| {r['bytes']} | {fmt_float(r['time_us'])} | {fmt_float(r['algbw_GBs'])} | {fmt_float(r['busbw_GBs'])} |"
            )
        lines.append("")

    # GPU count: compare same message size.
    gpu_rows = latest("gpu_count")
    if gpu_rows:
        lines.append("## GPU Count Sweep")
        lines.append("")
        lines.append("| gpus | bytes | time_us | algbw_GBs | busbw_GBs |")
        lines.append("|---:|---:|---:|---:|---:|")
        for r in sorted(gpu_rows, key=lambda x: x["gpus"]):
            lines.append(
                f"| {r['gpus']} | {r['bytes']} | {fmt_float(r['time_us'])} | {fmt_float(r['algbw_GBs'])} | {fmt_float(r['busbw_GBs'])} |"
            )
        lines.append("")

    coll_rows = latest("collective_type")
    if coll_rows:
        lines.append("## Collective Type Sweep")
        lines.append("")
        lines.append("| op | bytes | time_us | algbw_GBs | busbw_GBs |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in sorted(coll_rows, key=lambda x: x["op"]):
            lines.append(
                f"| {r['op']} | {r['bytes']} | {fmt_float(r['time_us'])} | {fmt_float(r['algbw_GBs'])} | {fmt_float(r['busbw_GBs'])} |"
            )
        lines.append("")

    transport_rows = latest("transport")
    if transport_rows:
        lines.append("## Transport Isolation")
        lines.append("")
        lines.append("| case | bytes | time_us | algbw_GBs | busbw_GBs |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in sorted(transport_rows, key=lambda x: x["value"]):
            lines.append(
                f"| {r['value']} | {r['bytes']} | {fmt_float(r['time_us'])} | {fmt_float(r['algbw_GBs'])} | {fmt_float(r['busbw_GBs'])} |"
            )
        lines.append("")

    channel_rows = latest("channels")
    if channel_rows:
        lines.append("## Channel Count Sweep")
        lines.append("")
        lines.append("| channels | bytes | time_us | algbw_GBs | busbw_GBs |")
        lines.append("|---:|---:|---:|---:|---:|")
        for r in sorted(channel_rows, key=lambda x: int(x["value"])):
            lines.append(
                f"| {r['value']} | {r['bytes']} | {fmt_float(r['time_us'])} | {fmt_float(r['algbw_GBs'])} | {fmt_float(r['busbw_GBs'])} |"
            )
        lines.append("")

    algo_rows = latest("algorithm")
    if algo_rows:
        lines.append("## Algorithm Sweep")
        lines.append("")
        lines.append("| algo | bytes | time_us | algbw_GBs | busbw_GBs |")
        lines.append("|---|---:|---:|---:|---:|")
        for r in sorted(algo_rows, key=lambda x: x["value"]):
            lines.append(
                f"| {r['value']} | {r['bytes']} | {fmt_float(r['time_us'])} | {fmt_float(r['algbw_GBs'])} | {fmt_float(r['busbw_GBs'])} |"
            )
        lines.append("")

    path.write_text("\n".join(lines) + "\n")


def main():
    if not (BUILD / "all_reduce_perf").exists():
        print("nccl-tests is not built. Run scripts/04_build_nccl_tests.sh first.", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cmd.out_dir = out_dir

    all_rows = []

    # 1. Message size sweep: latency dominated -> bandwidth dominated.
    code, log, text = run_cmd(
        "message_size_all_reduce_g4",
        "all_reduce_perf",
        ["-b", "8K", "-e", "256M", "-f", "8", "-g", "4"],
    )
    if code == 0:
        append_rows(all_rows, "message_size", "bytes", "scan", "all_reduce", 4, log, parse_rows(text))

    # 2. GPU count sweep: same payload, different world size.
    for g in [1, 2, 4]:
        code, log, text = run_cmd(
            f"gpu_count_all_reduce_g{g}",
            "all_reduce_perf",
            ["-b", "256M", "-e", "256M", "-f", "2", "-g", str(g)],
        )
        if code == 0:
            append_rows(all_rows, "gpu_count", "gpus", str(g), "all_reduce", g, log, parse_rows(text))

    # 3. Collective type comparison at same total size.
    for op, bin_name in [
        ("all_reduce", "all_reduce_perf"),
        ("all_gather", "all_gather_perf"),
        ("reduce_scatter", "reduce_scatter_perf"),
    ]:
        code, log, text = run_cmd(
            f"collective_{op}_g4_256M",
            bin_name,
            ["-b", "256M", "-e", "256M", "-f", "2", "-g", "4"],
        )
        if code == 0:
            append_rows(all_rows, "collective_type", "op", op, op, 4, log, parse_rows(text))

    # 4. Transport isolation.
    for case, env in [
        ("baseline", {}),
        ("no_p2p", {"NCCL_P2P_DISABLE": "1"}),
        ("no_shm", {"NCCL_SHM_DISABLE": "1"}),
    ]:
        code, log, text = run_cmd(
            f"transport_{case}_all_reduce_g4_256M",
            "all_reduce_perf",
            ["-b", "256M", "-e", "256M", "-f", "2", "-g", "4"],
            env=env,
        )
        if code == 0:
            append_rows(all_rows, "transport", "case", case, "all_reduce", 4, log, parse_rows(text))

    # 5. Channel count. Fixing channels is a diagnostic tool, not a production default.
    for channels in [1, 2, 4, 8, 12]:
        code, log, text = run_cmd(
            f"channels_{channels}_all_reduce_g4_64M",
            "all_reduce_perf",
            ["-b", "64M", "-e", "64M", "-f", "2", "-g", "4"],
            env={
                "NCCL_MIN_NCHANNELS": str(channels),
                "NCCL_MAX_NCHANNELS": str(channels),
            },
        )
        if code == 0:
            append_rows(all_rows, "channels", "channels", str(channels), "all_reduce", 4, log, parse_rows(text))

    # 6. Algorithm choice. Tree may be better for some sizes/topologies; Ring often
    # wins on bandwidth-heavy NVLink cases. Measure instead of assuming.
    for algo in ["Ring", "Tree"]:
        code, log, text = run_cmd(
            f"algorithm_{algo.lower()}_all_reduce_g4_64M",
            "all_reduce_perf",
            ["-b", "64M", "-e", "64M", "-f", "2", "-g", "4"],
            env={"NCCL_ALGO": algo},
        )
        if code == 0:
            append_rows(all_rows, "algorithm", "algo", algo, "all_reduce", 4, log, parse_rows(text))

    csv_path = out_dir / "summary.csv"
    md_path = out_dir / "summary.md"
    write_csv(all_rows, csv_path)
    write_summary(all_rows, md_path)
    print(f"\nCSV: {csv_path}")
    print(f"Markdown summary: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
