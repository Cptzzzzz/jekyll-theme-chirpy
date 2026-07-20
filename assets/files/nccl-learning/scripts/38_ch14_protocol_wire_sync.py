#!/usr/bin/env python3
"""Measure LL, LL128, and Simple wire layouts, performance, and kernels."""

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
COMBINATIONS = (
    ("Ring", "LL"),
    ("Ring", "LL128"),
    ("Ring", "Simple"),
    ("Tree", "LL"),
    ("Tree", "LL128"),
    ("Tree", "Simple"),
)
SIZE_COUNT = 29  # 4 B through 1 GiB.
CHANNELS = 12
PRACTICAL_TIE_PERCENT = 5.0
ALIGN_SIZES = (1024 * 1024, 64 * 1024 * 1024)
UNALIGN_ELEMENTS = (0, 1, 2, 3)


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


def parse_perf(
    text: str,
    algorithm: str,
    protocol: str,
    replicate: int,
    cycles: int,
    expected_size_count: int,
    unalign_elements: int = 0,
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
                "algorithm": algorithm,
                "protocol": protocol,
                "channels": CHANNELS,
                "replicate": replicate,
                "cycle": cycle,
                "size_bytes": size_bytes,
                "unalign_elements": unalign_elements,
                "unalign_bytes_float": unalign_elements * 4,
                "time_us": float(fields[5]),
                "algbw_GBs": float(fields[6]),
                "busbw_GBs": float(fields[7]),
                "wrong": int(fields[8]),
                "in_place_time_us": float(fields[9]),
                "in_place_busbw_GBs": float(fields[11]),
                "in_place_wrong": int(fields[12]),
            }
        )
    expected = expected_size_count * cycles
    if len(rows) != expected:
        raise RuntimeError(
            f"{algorithm}/{protocol} r={replicate} u={unalign_elements}: "
            f"expected {expected}, got {len(rows)}"
        )
    if len(seen) != expected_size_count or set(seen.values()) != {cycles}:
        raise RuntimeError(f"unexpected cycle counts: {seen}")
    if any(
        int(row["wrong"]) != 0 or int(row["in_place_wrong"]) != 0
        for row in rows
    ):
        raise RuntimeError(
            f"{algorithm}/{protocol} r={replicate}: correctness"
        )
    return rows


def summarize(
    rows: list[dict[str, object]],
    keys: tuple[str, ...],
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    summaries = []
    for key_values, group in sorted(groups.items()):
        times = [float(row["time_us"]) for row in group]
        summary = {key: value for key, value in zip(keys, key_values)}
        summary.update(
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
        summaries.append(summary)
    return summaries


def rank_candidates(
    summaries: list[dict[str, object]],
    group_keys: tuple[str, ...],
    candidate_keys: tuple[str, ...],
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in summaries:
        groups[tuple(row[key] for key in group_keys)].append(row)
    ranked_rows = []
    for group_values, candidates in sorted(groups.items()):
        ordered = sorted(candidates, key=lambda row: float(row["median_time_us"]))
        best, runner = ordered[:2]
        best_name = "+".join(str(best[key]) for key in candidate_keys)
        runner_name = "+".join(str(runner[key]) for key in candidate_keys)
        margin = (
            float(runner["median_time_us"]) / float(best["median_time_us"]) - 1.0
        ) * 100.0
        result = {key: value for key, value in zip(group_keys, group_values)}
        result.update(
            {
                "best": best_name,
                "best_median_us": best["median_time_us"],
                "runner_up": runner_name,
                "runner_up_median_us": runner["median_time_us"],
                "runner_margin_percent": margin,
                "decision": (
                    "winner"
                    if margin >= PRACTICAL_TIE_PERCENT
                    else "practical_tie"
                ),
            }
        )
        ranked_rows.append(result)
    return ranked_rows


def protocol_layout_rows() -> list[dict[str, object]]:
    nccl_steps = 8
    ll_line_bytes = 16
    ll_data_bytes = 8
    ll_buffer = 8 * 512 * nccl_steps * ll_line_bytes
    ll128_line_bytes = 128
    ll128_data_bytes = 120
    ll128_buffer = 120 * 640 * nccl_steps * 8
    simple_buffer = 1 << 22
    return [
        {
            "protocol": "LL",
            "wire_line_bytes": ll_line_bytes,
            "data_bytes_per_line": ll_data_bytes,
            "payload_efficiency_percent": ll_data_bytes / ll_line_bytes * 100,
            "default_buffer_bytes": ll_buffer,
            "wire_bytes_per_step": ll_buffer // nccl_steps,
            "data_bytes_per_step": ll_buffer // nccl_steps // 2,
            "sync_metadata": "two_32bit_flags_per_16B_line",
        },
        {
            "protocol": "LL128",
            "wire_line_bytes": ll128_line_bytes,
            "data_bytes_per_line": ll128_data_bytes,
            "payload_efficiency_percent": ll128_data_bytes / ll128_line_bytes * 100,
            "default_buffer_bytes": ll128_buffer,
            "wire_bytes_per_step": ll128_buffer // nccl_steps,
            "data_bytes_per_step": (
                ll128_buffer // nccl_steps * 15 // 16
            ),
            "sync_metadata": "one_64bit_flag_per_128B_line",
        },
        {
            "protocol": "Simple",
            "wire_line_bytes": 0,
            "data_bytes_per_line": 0,
            "payload_efficiency_percent": 100.0,
            "default_buffer_bytes": simple_buffer,
            "wire_bytes_per_step": simple_buffer // nccl_steps,
            "data_bytes_per_step": simple_buffer // nccl_steps,
            "sync_metadata": "separate_connection_step_head_tail",
        },
    ]


def run_nsys_profiles(
    root: Path,
    run_dir: Path,
    private_dir: Path,
    binary: Path,
) -> list[dict[str, object]]:
    private_dir.mkdir(parents=True, exist_ok=True)
    public_logs = run_dir / "raw/nsys"
    public_logs.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (algorithm, protocol) in enumerate(COMBINATIONS, 1):
        stem = f"{index:02d}_{algorithm.lower()}_{protocol.lower()}_1m"
        report_stem = private_dir / stem
        command = [
            "nsys",
            "profile",
            "--force-overwrite=true",
            "--sample=none",
            "--trace=cuda,nvtx",
            f"--output={report_stem}",
            str(binary),
            "-b", "1M",
            "-e", "1M",
            "-g", "4",
            "-w", "2",
            "-n", "5",
            "-N", "1",
            "-c", "0",
            "-I", "0",
            "-z", "0",
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
                "NCCL_ALGO": algorithm,
                "NCCL_PROTO": protocol,
                "NCCL_MIN_NCHANNELS": str(CHANNELS),
                "NCCL_MAX_NCHANNELS": str(CHANNELS),
                "NCCL_DEBUG": "INFO",
                "NCCL_DEBUG_SUBSYS": "ENV,TUNING",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        (public_logs / f"{stem}_profile.log").write_text(
            process.stdout, encoding="utf-8"
        )
        if process.returncode != 0:
            raise RuntimeError(f"nsys profile failed: {stem}")
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
        (public_logs / f"{stem}_export.log").write_text(
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
        selection_matches = re.findall(
            r"1048576 Bytes -> Algo (\d+) proto (\d+)", process.stdout
        )
        expected_algo_id = 1 if algorithm == "Ring" else 0
        expected_proto_id = {"LL": 0, "LL128": 1, "Simple": 2}[protocol]
        if not selection_matches or any(
            (int(algo_id), int(proto_id))
            != (expected_algo_id, expected_proto_id)
            for algo_id, proto_id in selection_matches
        ):
            raise RuntimeError(
                f"selection mismatch for {algorithm}+{protocol}: "
                f"{selection_matches}"
            )

        # generate.py::best_kernel maps every AllReduce protocol to an
        # algorithm-specific LL launch family. ncclKernelMain dispatches the
        # requested non-LL funcId through ncclDevFuncTable inside that kernel.
        marker = f"_{algorithm.upper()}_LL"
        matched = [row for row in kernel_rows if marker in row[0]]
        if not matched:
            raise RuntimeError(
                f"no kernel containing {marker}; got {[row[0] for row in kernel_rows]}"
            )
        names = sorted({row[0] for row in matched})
        durations_us = [int(row[1]) / 1000.0 for row in matched]
        rows.append(
            {
                "algorithm": algorithm,
                "requested_protocol": protocol,
                "message_bytes": 1024 * 1024,
                "selected_algo_id": expected_algo_id,
                "selected_proto_id": expected_proto_id,
                "launch_family_marker": marker,
                "launch_kernel_names": " | ".join(names),
                "kernel_name_encodes_requested_protocol": int(protocol == "LL"),
                "instances": len(matched),
                "median_kernel_us": statistics.median(durations_us),
                "p95_kernel_us": percentile(durations_us, 0.95),
                "grid_x_values": ",".join(
                    str(value) for value in sorted({int(row[2]) for row in matched})
                ),
                "block_x_values": ",".join(
                    str(value) for value in sorted({int(row[3]) for row in matched})
                ),
                "private_report": str(report_path),
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
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--alignment-cycles", type=int, default=10)
    parser.add_argument("--skip-nsys", action="store_true")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Parse existing raw logs and rerun only derived/trace stages.",
    )
    args = parser.parse_args()

    run_dir = args.root / "logs/ch14_protocol" / args.run_id
    perf_dir = run_dir / "raw/performance"
    align_dir = run_dir / "raw/alignment"
    perf_dir.mkdir(parents=True, exist_ok=True)
    align_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.root / "private/ch14_protocol" / args.run_id
    binary = args.root / "third_party/nccl-tests/build/all_reduce_perf"

    main_rows: list[dict[str, object]] = []
    commands: list[str] = []
    schedules = (
        (1, COMBINATIONS),
        (2, tuple(reversed(COMBINATIONS))),
    )
    run_index = 0
    for replicate, combinations in schedules:
        for algorithm, protocol in combinations:
            run_index += 1
            command = [
                str(binary),
                "-b", "4",
                "-e", "1G",
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
            environment = {
                **os.environ,
                "NCCL_ALGO": algorithm,
                "NCCL_PROTO": protocol,
                "NCCL_MIN_NCHANNELS": str(CHANNELS),
                "NCCL_MAX_NCHANNELS": str(CHANNELS),
                "NCCL_DEBUG": "WARN",
            }
            command_text = (
                f"NCCL_ALGO={algorithm} NCCL_PROTO={protocol} "
                f"NCCL_MIN_NCHANNELS={CHANNELS} "
                f"NCCL_MAX_NCHANNELS={CHANNELS} "
                + " ".join(command)
            )
            commands.append(command_text)
            path = perf_dir / (
                f"{run_index:02d}_{algorithm.lower()}_{protocol.lower()}_r{replicate}.log"
            )
            if args.resume_existing:
                output = path.read_text(encoding="utf-8")
            else:
                process = subprocess.run(
                    command,
                    cwd=args.root / "third_party/nccl-tests",
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                path.write_text(process.stdout, encoding="utf-8")
                if process.returncode != 0:
                    raise RuntimeError(f"failed; see {path}")
                output = process.stdout
            main_rows.extend(
                parse_perf(
                    output,
                    algorithm,
                    protocol,
                    replicate,
                    args.cycles,
                    SIZE_COUNT,
                )
            )
            print(
                f"[perf {run_index:02d}/12] {algorithm}+{protocol} "
                f"r={replicate} PASS",
                flush=True,
            )

    expected_rows = 2 * len(COMBINATIONS) * SIZE_COUNT * args.cycles
    if len(main_rows) != expected_rows:
        raise RuntimeError(
            f"expected {expected_rows} main rows, got {len(main_rows)}"
        )
    main_summary = summarize(
        main_rows, ("algorithm", "protocol", "size_bytes")
    )
    best_overall = rank_candidates(
        main_summary, ("size_bytes",), ("algorithm", "protocol")
    )
    best_by_algorithm = rank_candidates(
        main_summary,
        ("algorithm", "size_bytes"),
        ("protocol",),
    )

    alignment_rows: list[dict[str, object]] = []
    align_commands: list[str] = []
    align_index = 0
    for replicate, values in (
        (1, UNALIGN_ELEMENTS),
        (2, tuple(reversed(UNALIGN_ELEMENTS))),
    ):
        for unalign in values:
            align_index += 1
            command = [
                str(binary),
                "-b", "1M",
                "-e", "64M",
                "-f", "64",
                "-g", "4",
                "-w", "5",
                "-n", str(args.iterations),
                "-N", str(args.alignment_cycles),
                "-c", "1",
                "-I", "0",
                "-z", "0",
                "-u", str(unalign),
                "-C", "0",
                "-a", "3",
                "-d", "float",
                "-o", "sum",
            ]
            command_text = (
                "NCCL_ALGO=Ring NCCL_PROTO=LL128 "
                f"NCCL_MIN_NCHANNELS={CHANNELS} "
                f"NCCL_MAX_NCHANNELS={CHANNELS} "
                + " ".join(command)
            )
            align_commands.append(command_text)
            path = align_dir / (
                f"{align_index:02d}_u{unalign}_r{replicate}.log"
            )
            if args.resume_existing:
                output = path.read_text(encoding="utf-8")
            else:
                process = subprocess.run(
                    command,
                    cwd=args.root / "third_party/nccl-tests",
                    env={
                        **os.environ,
                        "NCCL_ALGO": "Ring",
                        "NCCL_PROTO": "LL128",
                        "NCCL_MIN_NCHANNELS": str(CHANNELS),
                        "NCCL_MAX_NCHANNELS": str(CHANNELS),
                        "NCCL_DEBUG": "WARN",
                    },
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                path.write_text(process.stdout, encoding="utf-8")
                if process.returncode != 0:
                    raise RuntimeError(f"alignment failed; see {path}")
                output = process.stdout
            alignment_rows.extend(
                parse_perf(
                    output,
                    "Ring",
                    "LL128",
                    replicate,
                    args.alignment_cycles,
                    len(ALIGN_SIZES),
                    unalign,
                )
            )
            print(
                f"[align {align_index:02d}/8] u={unalign} "
                f"r={replicate} PASS",
                flush=True,
            )
    alignment_summary = summarize(
        alignment_rows,
        ("unalign_elements", "unalign_bytes_float", "size_bytes"),
    )
    aligned_reference = {
        int(row["size_bytes"]): float(row["median_time_us"])
        for row in alignment_summary
        if int(row["unalign_elements"]) == 0
    }
    for row in alignment_summary:
        size = int(row["size_bytes"])
        row["time_vs_aligned_percent"] = (
            float(row["median_time_us"]) / aligned_reference[size] - 1.0
        ) * 100.0

    nsys_rows: list[dict[str, object]] = []
    if not args.skip_nsys:
        nsys_rows = run_nsys_profiles(
            args.root, run_dir, private_dir, binary
        )
        for row in nsys_rows:
            print(
                f"[nsys] {row['algorithm']}+{row['requested_protocol']} "
                f"proto={row['selected_proto_id']} "
                f"family={row['launch_family_marker']} PASS",
                flush=True,
            )

    layout_rows = protocol_layout_rows()
    write_csv(run_dir / "raw_measurements.csv", main_rows)
    write_csv(run_dir / "summary.csv", main_summary)
    write_csv(run_dir / "best_overall.csv", best_overall)
    write_csv(run_dir / "best_by_algorithm.csv", best_by_algorithm)
    write_csv(run_dir / "alignment_raw.csv", alignment_rows)
    write_csv(run_dir / "alignment_summary.csv", alignment_summary)
    write_csv(run_dir / "protocol_layout.csv", layout_rows)
    if nsys_rows:
        write_csv(run_dir / "kernel_trace_summary.csv", nsys_rows)

    tests_dir = args.root / "third_party/nccl-tests"
    nccl_dir = args.root / "third_party/nccl-2.22.3"
    manifest = [
        "experiment=ch14_protocol_wire_sync",
        f"run_id={args.run_id}",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"hostname={os.uname().nodename}",
        "world_size=4",
        "algorithms=Ring,Tree",
        "protocols=LL,LL128,Simple",
        "sizes=4B..1GiB_factor2",
        f"channels={CHANNELS}",
        f"cycles={args.cycles}",
        f"iterations={args.iterations}",
        f"main_rows={len(main_rows)}",
        f"alignment_rows={len(alignment_rows)}",
        f"alignment_cycles={args.alignment_cycles}",
        "alignment_protocol=Ring+LL128",
        "unalign_float_elements=0,1,2,3",
        f"practical_tie_percent={PRACTICAL_TIE_PERCENT}",
        f"nsys_profiles={len(nsys_rows)}",
        f"resume_existing_raw={int(args.resume_existing)}",
        f"private_nsys_dir={private_dir if nsys_rows else 'skipped'}",
        f"nccl_tests_commit={subprocess.check_output(['git', '-C', str(tests_dir), 'rev-parse', 'HEAD'], text=True).strip()}",
        f"nccl_source_commit={subprocess.check_output(['git', '-C', str(nccl_dir), 'rev-parse', 'HEAD'], text=True).strip()}",
    ]
    manifest.extend(
        f"command_{index:02d}={command}"
        for index, command in enumerate(commands, 1)
    )
    manifest.extend(
        f"alignment_command_{index:02d}={command}"
        for index, command in enumerate(align_commands, 1)
    )
    (run_dir / "manifest.txt").write_text(
        "\n".join(manifest) + "\n", encoding="utf-8"
    )

    lines = [
        "# Chapter 14 experiment summary",
        "",
        f"- Main measurement rows: {len(main_rows)}",
        f"- Alignment measurement rows: {len(alignment_rows)}",
        "- Out-of-place correctness: PASS",
        "- In-place correctness: PASS",
        f"- Nsight kernel-path profiles: {len(nsys_rows)}",
        "",
        "## Protocol layout",
        "",
        "| protocol | line B | data B | efficiency | default buffer | data/step | sync |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in layout_rows:
        lines.append(
            f"| {row['protocol']} | {row['wire_line_bytes']} | "
            f"{row['data_bytes_per_line']} | "
            f"{float(row['payload_efficiency_percent']):.2f}% | "
            f"{row['default_buffer_bytes']} | {row['data_bytes_per_step']} | "
            f"{row['sync_metadata']} |"
        )
    lines.extend(
        [
            "",
            "## Selected performance points",
            "",
            "| algo | proto | bytes | median us | P95 | CV | busbw |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    selected_sizes = {4, 1024, 65536, 1048576, 16777216, 1073741824}
    for row in main_summary:
        if int(row["size_bytes"]) not in selected_sizes:
            continue
        lines.append(
            f"| {row['algorithm']} | {row['protocol']} | "
            f"{row['size_bytes']} | {float(row['median_time_us']):.3f} | "
            f"{float(row['p95_time_us']):.3f} | "
            f"{float(row['cycle_cv_percent']):.2f}% | "
            f"{float(row['median_busbw_GBs']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Best overall by size",
            "",
            "| bytes | best | runner | margin | decision |",
            "|---:|---|---|---:|---|",
        ]
    )
    for row in best_overall:
        lines.append(
            f"| {row['size_bytes']} | {row['best']} | {row['runner_up']} | "
            f"{float(row['runner_margin_percent']):.2f}% | {row['decision']} |"
        )
    lines.extend(
        [
            "",
            "## LL128 alignment",
            "",
            "| unalign float elements | offset B | bytes | median us | CV | vs aligned |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in alignment_summary:
        lines.append(
            f"| {row['unalign_elements']} | {row['unalign_bytes_float']} | "
            f"{row['size_bytes']} | {float(row['median_time_us']):.3f} | "
            f"{float(row['cycle_cv_percent']):.2f}% | "
            f"{float(row['time_vs_aligned_percent']):+.2f}% |"
        )
    if nsys_rows:
        lines.extend(
            [
                "",
                "## Kernel-path proof",
                "",
                "| algo | requested proto | selected id | launch family | instances | median kernel us | grid | block |",
                "|---|---|---:|---|---:|---:|---|---|",
            ]
        )
        for row in nsys_rows:
            lines.append(
                f"| {row['algorithm']} | {row['requested_protocol']} | "
                f"{row['selected_proto_id']} | "
                f"`{row['launch_family_marker']}` | {row['instances']} | "
                f"{float(row['median_kernel_us']):.3f} | "
                f"{row['grid_x_values']} | {row['block_x_values']} |"
            )
    (run_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
