#!/usr/bin/env python3
"""Sweep channels, Simple buffer size, worker threads, and interactions."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sqlite3
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROW_RE = re.compile(r"^\s*\d+\s+")
CHANNEL_VALUES = (1, 2, 4, 8, 12)
BUFFER_VALUES = (64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20)
THREAD_VALUES = (64, 128, 256, 512)
DEFAULT_CHANNELS = 12
DEFAULT_BUFFER = 4 << 20
DEFAULT_THREADS = 512


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def cv_percent(values: list[float]) -> float:
    mean = statistics.fmean(values)
    if not mean or len(values) < 2:
        return 0.0
    return statistics.pstdev(values) / mean * 100.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def configurations() -> list[dict[str, object]]:
    configs = []
    for channels in CHANNEL_VALUES:
        configs.append(
            {
                "sweep": "channels",
                "channels": channels,
                "buffer_bytes": DEFAULT_BUFFER,
                "nthreads": DEFAULT_THREADS,
                "min_size": "4K",
                "max_size": "64M",
                "factor": "128",
                "size_count": 3,
            }
        )
    for buffer_bytes in BUFFER_VALUES:
        configs.append(
            {
                "sweep": "buffer",
                "channels": DEFAULT_CHANNELS,
                "buffer_bytes": buffer_bytes,
                "nthreads": DEFAULT_THREADS,
                "min_size": "512K",
                "max_size": "64M",
                "factor": "128",
                "size_count": 2,
            }
        )
    for nthreads in THREAD_VALUES:
        configs.append(
            {
                "sweep": "threads",
                "channels": DEFAULT_CHANNELS,
                "buffer_bytes": DEFAULT_BUFFER,
                "nthreads": nthreads,
                "min_size": "4K",
                "max_size": "64M",
                "factor": "128",
                "size_count": 3,
            }
        )
    for channels in (2, 12):
        for buffer_bytes in (256 << 10, 4 << 20):
            for nthreads in (128, 512):
                configs.append(
                    {
                        "sweep": "interaction",
                        "channels": channels,
                        "buffer_bytes": buffer_bytes,
                        "nthreads": nthreads,
                        "min_size": "512K",
                        "max_size": "64M",
                        "factor": "128",
                        "size_count": 2,
                    }
                )
    if len(configs) != 22:
        raise RuntimeError(f"expected 22 configs, got {len(configs)}")
    return configs


def parse_perf(
    text: str,
    config: dict[str, object],
    replicate: int,
    cycles: int,
) -> list[dict[str, object]]:
    seen: dict[int, int] = defaultdict(int)
    rows = []
    for line in text.splitlines():
        if not ROW_RE.match(line):
            continue
        fields = line.split()
        if len(fields) != 13:
            continue
        size_bytes = int(fields[0])
        if size_bytes < 4:
            continue
        cycle = seen[size_bytes]
        seen[size_bytes] += 1
        rows.append(
            {
                "sweep": config["sweep"],
                "channels": config["channels"],
                "buffer_bytes": config["buffer_bytes"],
                "nthreads": config["nthreads"],
                "replicate": replicate,
                "cycle": cycle,
                "size_bytes": size_bytes,
                "time_us": float(fields[5]),
                "algbw_GBs": float(fields[6]),
                "busbw_GBs": float(fields[7]),
                "wrong": int(fields[8]),
                "in_place_time_us": float(fields[9]),
                "in_place_busbw_GBs": float(fields[11]),
                "in_place_wrong": int(fields[12]),
            }
        )
    expected = int(config["size_count"]) * cycles
    if len(rows) != expected:
        raise RuntimeError(
            f"{config} replicate={replicate}: expected {expected}, got {len(rows)}"
        )
    if len(seen) != int(config["size_count"]) or set(seen.values()) != {cycles}:
        raise RuntimeError(f"unexpected cycle counts: {seen}")
    if any(
        int(row["wrong"]) != 0 or int(row["in_place_wrong"]) != 0
        for row in rows
    ):
        raise RuntimeError(f"correctness failure: {config}")
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    keys = ("sweep", "channels", "buffer_bytes", "nthreads", "size_bytes")
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    summaries = []
    for key_values, group in sorted(groups.items()):
        times = [float(row["time_us"]) for row in group]
        result = {key: value for key, value in zip(keys, key_values)}
        result.update(
            {
                "samples": len(times),
                "median_time_us": statistics.median(times),
                "p95_time_us": percentile(times, 0.95),
                "cycle_cv_percent": cv_percent(times),
                "min_time_us": min(times),
                "max_time_us": max(times),
                "median_algbw_GBs": statistics.median(
                    float(row["algbw_GBs"]) for row in group
                ),
                "median_busbw_GBs": statistics.median(
                    float(row["busbw_GBs"]) for row in group
                ),
            }
        )
        summaries.append(result)
    return summaries


def add_reference_columns(summaries: list[dict[str, object]]) -> None:
    references = {}
    for row in summaries:
        sweep = str(row["sweep"])
        is_reference = (
            sweep == "channels" and int(row["channels"]) == DEFAULT_CHANNELS
            or sweep == "buffer" and int(row["buffer_bytes"]) == DEFAULT_BUFFER
            or sweep == "threads" and int(row["nthreads"]) == DEFAULT_THREADS
            or sweep == "interaction"
            and int(row["channels"]) == DEFAULT_CHANNELS
            and int(row["buffer_bytes"]) == DEFAULT_BUFFER
            and int(row["nthreads"]) == DEFAULT_THREADS
        )
        if is_reference:
            references[(sweep, int(row["size_bytes"]))] = float(
                row["median_time_us"]
            )
    for row in summaries:
        reference = references[(str(row["sweep"]), int(row["size_bytes"]))]
        row["reference_median_us"] = reference
        row["time_vs_reference_percent"] = (
            float(row["median_time_us"]) / reference - 1.0
        ) * 100.0


def interaction_rows(
    summaries: list[dict[str, object]],
) -> list[dict[str, object]]:
    lookup = {
        (
            str(row["sweep"]),
            int(row["channels"]),
            int(row["buffer_bytes"]),
            int(row["nthreads"]),
            int(row["size_bytes"]),
        ): row
        for row in summaries
    }
    results = []
    for row in summaries:
        if row["sweep"] != "interaction":
            continue
        size = int(row["size_bytes"])
        baseline = float(
            lookup[
                (
                    "interaction",
                    DEFAULT_CHANNELS,
                    DEFAULT_BUFFER,
                    DEFAULT_THREADS,
                    size,
                )
            ]["median_time_us"]
        )
        channel_time = float(
            lookup[
                (
                    "channels",
                    int(row["channels"]),
                    DEFAULT_BUFFER,
                    DEFAULT_THREADS,
                    size,
                )
            ]["median_time_us"]
        )
        channel_base = float(
            lookup[
                (
                    "channels",
                    DEFAULT_CHANNELS,
                    DEFAULT_BUFFER,
                    DEFAULT_THREADS,
                    size,
                )
            ]["median_time_us"]
        )
        buffer_time = float(
            lookup[
                (
                    "buffer",
                    DEFAULT_CHANNELS,
                    int(row["buffer_bytes"]),
                    DEFAULT_THREADS,
                    size,
                )
            ]["median_time_us"]
        )
        buffer_base = float(
            lookup[
                (
                    "buffer",
                    DEFAULT_CHANNELS,
                    DEFAULT_BUFFER,
                    DEFAULT_THREADS,
                    size,
                )
            ]["median_time_us"]
        )
        thread_time = float(
            lookup[
                (
                    "threads",
                    DEFAULT_CHANNELS,
                    DEFAULT_BUFFER,
                    int(row["nthreads"]),
                    size,
                )
            ]["median_time_us"]
        )
        thread_base = float(
            lookup[
                (
                    "threads",
                    DEFAULT_CHANNELS,
                    DEFAULT_BUFFER,
                    DEFAULT_THREADS,
                    size,
                )
            ]["median_time_us"]
        )
        predicted = baseline * (channel_time / channel_base)
        predicted *= buffer_time / buffer_base
        predicted *= thread_time / thread_base
        actual = float(row["median_time_us"])
        results.append(
            {
                "channels": row["channels"],
                "buffer_bytes": row["buffer_bytes"],
                "nthreads": row["nthreads"],
                "size_bytes": size,
                "actual_median_us": actual,
                "independent_effect_prediction_us": predicted,
                "interaction_residual_percent": (actual / predicted - 1.0) * 100.0,
                "actual_vs_default_percent": (actual / baseline - 1.0) * 100.0,
            }
        )
    return results


def chunk_layout_rows() -> list[dict[str, object]]:
    rows = []
    for buffer_bytes in BUFFER_VALUES:
        step = buffer_bytes // 8
        chunk = step * 4
        for message_bytes in (512 << 10, 64 << 20):
            rows.append(
                {
                    "buffer_bytes": buffer_bytes,
                    "message_bytes": message_bytes,
                    "fifo_steps": 8,
                    "simple_data_bytes_per_step": step,
                    "allreduce_chunk_steps": 4,
                    "allreduce_slice_steps": 2,
                    "slices_per_chunk": 2,
                    "nominal_chunk_bytes": chunk,
                    "nominal_slice_bytes": step * 2,
                    "simple_grain_bytes": 512,
                    "estimated_ring_loops_world4_ch12": math.ceil(
                        message_bytes / (12 * 4 * chunk)
                    ),
                }
            )
    return rows


def run_geometry_profiles(
    root: Path,
    run_dir: Path,
    private_dir: Path,
    binary: Path,
) -> list[dict[str, object]]:
    profile_configs = (
        ("channels_1", 1, DEFAULT_BUFFER, DEFAULT_THREADS, "1M", 1 << 20),
        ("channels_4", 4, DEFAULT_BUFFER, DEFAULT_THREADS, "1M", 1 << 20),
        ("channels_12", 12, DEFAULT_BUFFER, DEFAULT_THREADS, "1M", 1 << 20),
        ("threads_64", 12, DEFAULT_BUFFER, 64, "1M", 1 << 20),
        ("threads_128", 12, DEFAULT_BUFFER, 128, "1M", 1 << 20),
        ("threads_256", 12, DEFAULT_BUFFER, 256, "1M", 1 << 20),
        ("tiny_4k_requested_ch12_t512", 12, DEFAULT_BUFFER, 512, "4K", 4 << 10),
    )
    private_dir.mkdir(parents=True, exist_ok=True)
    public_dir = run_dir / "raw/nsys"
    public_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (
        label, channels, buffer_bytes, nthreads, message_arg, message_bytes
    ) in enumerate(
        profile_configs, 1
    ):
        stem = f"{index:02d}_{label}"
        report_stem = private_dir / stem
        command = [
            "nsys",
            "profile",
            "--force-overwrite=true",
            "--sample=none",
            "--trace=cuda,nvtx",
            f"--output={report_stem}",
            str(binary),
            "-b", message_arg,
            "-e", message_arg,
            "-g", "4",
            "-w", "2",
            "-n", "5",
            "-N", "1",
            "-c", "0",
            "-I", "0",
            "-z", "0",
            "-u", "0",
            "-C", "0",
            "-a", "3",
            "-d", "float",
            "-o", "sum",
        ]
        process = subprocess.run(
            command,
            cwd=root / "third_party/nccl-tests",
            env={
                **os.environ,
                "NCCL_ALGO": "Ring",
                "NCCL_PROTO": "Simple",
                "NCCL_MIN_NCHANNELS": str(channels),
                "NCCL_MAX_NCHANNELS": str(channels),
                "NCCL_BUFFSIZE": str(buffer_bytes),
                "NCCL_NTHREADS": str(nthreads),
                "NCCL_DEBUG": "INFO",
                "NCCL_DEBUG_SUBSYS": "ENV,TUNING",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        (public_dir / f"{stem}_profile.log").write_text(
            process.stdout, encoding="utf-8"
        )
        if process.returncode != 0:
            raise RuntimeError(f"nsys profile failed: {stem}")
        selection = re.findall(
            rf"{message_bytes} Bytes -> Algo (\d+) proto (\d+)",
            process.stdout,
        )
        if not selection or any(pair != ("1", "2") for pair in selection):
            raise RuntimeError(f"selection mismatch {label}: {selection}")

        report_path = report_stem.with_suffix(".nsys-rep")
        sqlite_path = report_stem.with_suffix(".sqlite")
        export = subprocess.run(
            [
                "nsys",
                "export",
                "--force-overwrite=true",
                "--type=sqlite",
                f"--output={sqlite_path}",
                str(report_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        (public_dir / f"{stem}_export.log").write_text(
            export.stdout, encoding="utf-8"
        )
        if export.returncode != 0:
            raise RuntimeError(f"nsys export failed: {stem}")
        connection = sqlite3.connect(sqlite_path)
        kernel_rows = connection.execute(
            """
            SELECT s.value, k.end-k.start, k.gridX, k.blockX
            FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
            JOIN StringIds AS s ON s.id = k.shortName
            WHERE s.value LIKE 'ncclDevKernel_AllReduce%'
            """
        ).fetchall()
        connection.close()
        matched = [row for row in kernel_rows if "_RING_LL" in row[0]]
        if not matched:
            raise RuntimeError(f"no AllReduce launch kernel: {label}")
        durations = [int(row[1]) / 1000.0 for row in matched]
        grid_values = sorted({int(row[2]) for row in matched})
        block_values = sorted({int(row[3]) for row in matched})
        rows.append(
            {
                "label": label,
                "message_bytes": message_bytes,
                "requested_channels": channels,
                "buffer_bytes": buffer_bytes,
                "requested_nthreads": nthreads,
                "selected_algo_id": 1,
                "selected_proto_id": 2,
                "kernel_name": sorted({row[0] for row in matched})[0],
                "instances": len(matched),
                "grid_x_values": ",".join(map(str, grid_values)),
                "block_x_values": ",".join(map(str, block_values)),
                "grid_matches_requested_channels": int(grid_values == [channels]),
                "median_kernel_us": statistics.median(durations),
                "p95_kernel_us": percentile(durations, 0.95),
                "private_report_retained": 1,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("/root/nccl-learning")
    )
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--skip-nsys", action="store_true")
    args = parser.parse_args()

    run_dir = args.root / "logs/ch15_pipeline" / args.run_id
    raw_dir = run_dir / "raw/performance"
    raw_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.root / "private/ch15_pipeline" / args.run_id
    binary = args.root / "third_party/nccl-tests/build/all_reduce_perf"
    configs = configurations()

    all_rows = []
    commands = []
    run_index = 0
    for replicate, ordered in (
        (1, configs),
        (2, list(reversed(configs))),
    ):
        for config in ordered:
            run_index += 1
            command = [
                str(binary),
                "-b", str(config["min_size"]),
                "-e", str(config["max_size"]),
                "-f", str(config["factor"]),
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
            environment = {
                **os.environ,
                "NCCL_ALGO": "Ring",
                "NCCL_PROTO": "Simple",
                "NCCL_MIN_NCHANNELS": str(config["channels"]),
                "NCCL_MAX_NCHANNELS": str(config["channels"]),
                "NCCL_BUFFSIZE": str(config["buffer_bytes"]),
                "NCCL_NTHREADS": str(config["nthreads"]),
                "NCCL_DEBUG": "WARN",
            }
            command_text = " ".join(
                [
                    "NCCL_ALGO=Ring",
                    "NCCL_PROTO=Simple",
                    f"NCCL_MIN_NCHANNELS={config['channels']}",
                    f"NCCL_MAX_NCHANNELS={config['channels']}",
                    f"NCCL_BUFFSIZE={config['buffer_bytes']}",
                    f"NCCL_NTHREADS={config['nthreads']}",
                    *command,
                ]
            )
            commands.append(command_text)
            process = subprocess.run(
                command,
                cwd=args.root / "third_party/nccl-tests",
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            path = raw_dir / (
                f"{run_index:02d}_{config['sweep']}_ch{config['channels']}_"
                f"b{config['buffer_bytes']}_t{config['nthreads']}_r{replicate}.log"
            )
            path.write_text(process.stdout, encoding="utf-8")
            if process.returncode != 0:
                raise RuntimeError(f"failed; see {path}")
            all_rows.extend(
                parse_perf(process.stdout, config, replicate, args.cycles)
            )
            print(
                f"[perf {run_index:02d}/44] {config['sweep']} "
                f"ch={config['channels']} b={config['buffer_bytes']} "
                f"t={config['nthreads']} r={replicate} PASS",
                flush=True,
            )

    expected_rows_per_round = sum(
        int(config["size_count"]) * args.cycles for config in configs
    )
    if len(all_rows) != 2 * expected_rows_per_round:
        raise RuntimeError(
            f"expected {2*expected_rows_per_round}, got {len(all_rows)}"
        )
    summaries = summarize(all_rows)
    add_reference_columns(summaries)
    interactions = interaction_rows(summaries)
    chunk_rows = chunk_layout_rows()
    geometry_rows = []
    if not args.skip_nsys:
        geometry_rows = run_geometry_profiles(
            args.root, run_dir, private_dir, binary
        )
        for row in geometry_rows:
            print(
                f"[nsys] {row['label']} grid={row['grid_x_values']} "
                f"block={row['block_x_values']} PASS",
                flush=True,
            )

    write_csv(run_dir / "raw_measurements.csv", all_rows)
    write_csv(run_dir / "summary.csv", summaries)
    write_csv(run_dir / "interaction_model.csv", interactions)
    write_csv(run_dir / "chunk_layout.csv", chunk_rows)
    if geometry_rows:
        write_csv(run_dir / "kernel_geometry.csv", geometry_rows)

    tests_dir = args.root / "third_party/nccl-tests"
    nccl_dir = args.root / "third_party/nccl-2.22.3"
    manifest = [
        "experiment=ch15_channels_chunks_buffers",
        f"run_id={args.run_id}",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"hostname={os.uname().nodename}",
        "world_size=4",
        "algorithm=Ring",
        "protocol=Simple",
        f"cycles={args.cycles}",
        f"iterations={args.iterations}",
        f"process_configs={2*len(configs)}",
        f"measurement_rows={len(all_rows)}",
        f"nsys_profiles={len(geometry_rows)}",
        f"private_nsys_dir={private_dir if geometry_rows else 'skipped'}",
        f"nccl_tests_commit={subprocess.check_output(['git', '-C', str(tests_dir), 'rev-parse', 'HEAD'], text=True).strip()}",
        f"nccl_source_commit={subprocess.check_output(['git', '-C', str(nccl_dir), 'rev-parse', 'HEAD'], text=True).strip()}",
    ]
    manifest.extend(
        f"command_{index:02d}={command}"
        for index, command in enumerate(commands, 1)
    )
    (run_dir / "manifest.txt").write_text(
        "\n".join(manifest) + "\n", encoding="utf-8"
    )

    lines = [
        "# Chapter 15 experiment summary",
        "",
        f"- Measurement rows: {len(all_rows)}",
        "- Out-of-place correctness: PASS",
        "- In-place correctness: PASS",
        f"- Nsight geometry profiles: {len(geometry_rows)}",
        "",
        "## Layered sweeps",
        "",
        "| sweep | ch | buffer | threads | bytes | median us | CV | busbw | vs reference |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['sweep']} | {row['channels']} | {row['buffer_bytes']} | "
            f"{row['nthreads']} | {row['size_bytes']} | "
            f"{float(row['median_time_us']):.3f} | "
            f"{float(row['cycle_cv_percent']):.2f}% | "
            f"{float(row['median_busbw_GBs']):.2f} | "
            f"{float(row['time_vs_reference_percent']):+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Interaction residuals",
            "",
            "| ch | buffer | threads | bytes | actual us | independent prediction | residual |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in interactions:
        lines.append(
            f"| {row['channels']} | {row['buffer_bytes']} | "
            f"{row['nthreads']} | {row['size_bytes']} | "
            f"{float(row['actual_median_us']):.3f} | "
            f"{float(row['independent_effect_prediction_us']):.3f} | "
            f"{float(row['interaction_residual_percent']):+.2f}% |"
        )
    if geometry_rows:
        lines.extend(
            [
                "",
                "## Kernel geometry",
                "",
                "| label | bytes | requested ch | requested threads | gridX | blockX | matches channels | median kernel us |",
                "|---|---:|---:|---:|---|---|---:|---:|",
            ]
        )
        for row in geometry_rows:
            lines.append(
                f"| {row['label']} | {row['message_bytes']} | "
                f"{row['requested_channels']} | "
                f"{row['requested_nthreads']} | {row['grid_x_values']} | "
                f"{row['block_x_values']} | "
                f"{row['grid_matches_requested_channels']} | "
                f"{float(row['median_kernel_us']):.3f} |"
            )
    (run_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
