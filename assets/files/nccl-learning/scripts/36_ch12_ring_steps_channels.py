#!/usr/bin/env python3
"""Validate Ring AllReduce source steps across world size, message size, and channels."""

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
RING_RE = re.compile(
    r"Channel\s+(\d+)/(\d+)\s*:\s*((?:\s*\d+)+)"
)
CHANNELS = (1, 2, 4, 8, 12)


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


def parse_rows(
    text: str,
    world_size: int,
    channels: int,
    replicate: int,
    cycles: int,
) -> list[dict[str, object]]:
    rows = []
    seen: dict[int, int] = defaultdict(int)
    for line in text.splitlines():
        if not ROW_RE.match(line):
            continue
        fields = line.split()
        if len(fields) != 13:
            continue
        size_bytes = int(fields[0])
        cycle = seen[size_bytes]
        seen[size_bytes] += 1
        rows.append(
            {
                "world_size": world_size,
                "requested_channels": channels,
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
    expected = cycles * 2
    if len(rows) != expected:
        raise RuntimeError(
            f"N={world_size} ch={channels} r={replicate}: "
            f"expected {expected}, got {len(rows)}"
        )
    if any(
        int(row["wrong"]) != 0 or int(row["in_place_wrong"]) != 0
        for row in rows
    ):
        raise RuntimeError(
            f"N={world_size} ch={channels} r={replicate}: correctness"
        )
    return rows


def parse_rings(
    text: str, world_size: int, requested_channels: int
) -> list[dict[str, object]]:
    rings: dict[int, tuple[int, tuple[int, ...]]] = {}
    for match in RING_RE.finditer(text):
        channel_id = int(match.group(1))
        total_channels = int(match.group(2))
        order = tuple(int(value) for value in match.group(3).split())
        if len(order) != world_size or sorted(order) != list(
            range(world_size)
        ):
            continue
        value = (total_channels, order)
        if channel_id in rings and rings[channel_id] != value:
            raise RuntimeError(
                f"inconsistent ring {channel_id}: {rings[channel_id]} vs {value}"
            )
        rings[channel_id] = value
    if not rings:
        raise RuntimeError(
            f"no ring rows for N={world_size} ch={requested_channels}"
        )
    result = []
    for channel_id, (total_channels, order) in sorted(rings.items()):
        result.append(
            {
                "world_size": world_size,
                "requested_channels": requested_channels,
                "channel_id": channel_id,
                "reported_total_channels": total_channels,
                "ring_order": "-".join(str(rank) for rank in order),
            }
        )
    return result


def build_schedule(
    ring_order: list[int],
) -> list[dict[str, object]]:
    nranks = len(ring_order)
    rows = []
    for ring_index, user_rank in enumerate(ring_order):
        prev_rank = ring_order[(ring_index - 1) % nranks]
        next_rank = ring_order[(ring_index + 1) % nranks]
        sequence = 0

        def append(
            phase: str,
            operation: str,
            chunk_index: int,
            receives: bool,
            sends: bool,
            final_chunk_created: bool,
        ) -> None:
            nonlocal sequence
            rows.append(
                {
                    "ring_order": "-".join(
                        str(rank) for rank in ring_order
                    ),
                    "ring_index": ring_index,
                    "user_rank": user_rank,
                    "prev_rank": prev_rank,
                    "next_rank": next_rank,
                    "sequence": sequence,
                    "phase": phase,
                    "operation": operation,
                    "chunk_index": chunk_index,
                    "chunk_owner_user_rank": ring_order[chunk_index],
                    "receives_from_prev": receives,
                    "sends_to_next": sends,
                    "final_chunk_created": final_chunk_created,
                }
            )
            sequence += 1

        append(
            "reduce_scatter",
            "send",
            (ring_index + nranks - 1) % nranks,
            False,
            True,
            False,
        )
        for j in range(2, nranks):
            append(
                "reduce_scatter",
                "recvReduceSend",
                (ring_index + nranks - j) % nranks,
                True,
                True,
                False,
            )
        append(
            "rs_ag_fused_boundary",
            "directRecvReduceCopySend",
            ring_index,
            True,
            True,
            True,
        )
        for j in range(1, nranks - 1):
            append(
                "all_gather",
                "directRecvCopySend",
                (ring_index + nranks - j) % nranks,
                True,
                True,
                False,
            )
        append(
            "all_gather",
            "directRecv",
            (ring_index + 1) % nranks,
            True,
            False,
            False,
        )

    for user_rank in ring_order:
        rank_rows = [
            row for row in rows if int(row["user_rank"]) == user_rank
        ]
        sends = sum(bool(row["sends_to_next"]) for row in rank_rows)
        receives = sum(
            bool(row["receives_from_prev"]) for row in rank_rows
        )
        if sends != 2 * (nranks - 1):
            raise RuntimeError(
                f"rank {user_rank}: sends={sends}, expected {2*(nranks-1)}"
            )
        if receives != 2 * (nranks - 1):
            raise RuntimeError(
                f"rank {user_rank}: recv={receives}, expected {2*(nranks-1)}"
            )
        chunks = {int(row["chunk_index"]) for row in rank_rows}
        if chunks != set(range(nranks)):
            raise RuntimeError(
                f"rank {user_rank}: missing chunk operations {chunks}"
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
    args = parser.parse_args()

    run_dir = args.root / "logs/ch12_ring" / args.run_id
    perf_dir = run_dir / "raw/performance"
    graph_dir = run_dir / "raw/graph"
    perf_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    binary = args.root / "third_party/nccl-tests/build/all_reduce_perf"

    all_rows: list[dict[str, object]] = []
    commands: list[str] = []
    run_index = 0
    for world_size in (2, 3, 4):
        for replicate, order in (
            (1, CHANNELS),
            (2, tuple(reversed(CHANNELS))),
        ):
            for channels in order:
                run_index += 1
                command = [
                    str(binary),
                    "-b", "1M",
                    "-e", "64M",
                    "-f", "64",
                    "-g", str(world_size),
                    "-w", "5",
                    "-n", str(args.iterations),
                    "-N", str(args.cycles),
                    "-c", "1",
                    "-I", "0",
                    "-z", "0",
                    "-C", "0",
                    "-a", "3",
                    "-d", "float",
                    "-o", "sum",
                ]
                environment = {
                    **os.environ,
                    "NCCL_ALGO": "Ring",
                    "NCCL_PROTO": "Simple",
                    "NCCL_MIN_NCHANNELS": str(channels),
                    "NCCL_MAX_NCHANNELS": str(channels),
                    "NCCL_DEBUG": "WARN",
                }
                commands.append(
                    f"NCCL_ALGO=Ring NCCL_PROTO=Simple "
                    f"NCCL_MIN_NCHANNELS={channels} "
                    f"NCCL_MAX_NCHANNELS={channels} "
                    + " ".join(command)
                )
                process = subprocess.run(
                    command,
                    cwd=args.root / "third_party/nccl-tests",
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                path = (
                    perf_dir
                    / f"{run_index:02d}_n{world_size}_ch{channels}_r{replicate}.log"
                )
                path.write_text(process.stdout, encoding="utf-8")
                if process.returncode != 0:
                    raise RuntimeError(f"failed; see {path}")
                all_rows.extend(
                    parse_rows(
                        process.stdout,
                        world_size,
                        channels,
                        replicate,
                        args.cycles,
                    )
                )
                print(
                    f"[{run_index:02d}/30] N={world_size} "
                    f"ch={channels} r={replicate} PASS",
                    flush=True,
                )

    expected_rows = 3 * 2 * len(CHANNELS) * args.cycles * 2
    if len(all_rows) != expected_rows:
        raise RuntimeError(
            f"expected {expected_rows} rows, got {len(all_rows)}"
        )

    groups: dict[
        tuple[int, int, int], list[dict[str, object]]
    ] = defaultdict(list)
    for row in all_rows:
        groups[
            (
                int(row["world_size"]),
                int(row["requested_channels"]),
                int(row["size_bytes"]),
            )
        ].append(row)
    reference = {
        (world, size): statistics.median(
            float(row["time_us"])
            for row in groups[(world, 12, size)]
        )
        for world in (2, 3, 4)
        for size in (1024 * 1024, 64 * 1024 * 1024)
    }
    summaries = []
    for (world, channels, size), rows in sorted(groups.items()):
        times = [float(row["time_us"]) for row in rows]
        median_time = statistics.median(times)
        summaries.append(
            {
                "world_size": world,
                "requested_channels": channels,
                "size_bytes": size,
                "samples": len(times),
                "median_time_us": median_time,
                "p95_time_us": percentile(times, 0.95),
                "cycle_cv_percent": cv_percent(times),
                "median_algbw_GBs": statistics.median(
                    float(row["algbw_GBs"]) for row in rows
                ),
                "median_busbw_GBs": statistics.median(
                    float(row["busbw_GBs"]) for row in rows
                ),
                "time_vs_ch12_percent":
                    (median_time / reference[(world, size)] - 1.0) * 100.0,
            }
        )

    ring_rows: list[dict[str, object]] = []
    graph_index = 0
    for world_size in (2, 3, 4):
        for channels in (1, 12):
            graph_index += 1
            command = [
                str(binary),
                "-b", "1M",
                "-e", "1M",
                "-g", str(world_size),
                "-w", "0",
                "-n", "1",
                "-N", "1",
                "-c", "0",
                "-I", "0",
                "-z", "0",
                "-d", "float",
                "-o", "sum",
            ]
            process = subprocess.run(
                command,
                cwd=args.root / "third_party/nccl-tests",
                env={
                    **os.environ,
                    "NCCL_ALGO": "Ring",
                    "NCCL_PROTO": "Simple",
                    "NCCL_MIN_NCHANNELS": str(channels),
                    "NCCL_MAX_NCHANNELS": str(channels),
                    "NCCL_DEBUG": "INFO",
                    "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,TUNING",
                },
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            path = (
                graph_dir
                / f"{graph_index:02d}_n{world_size}_ch{channels}.log"
            )
            path.write_text(process.stdout, encoding="utf-8")
            if process.returncode != 0:
                raise RuntimeError(f"graph run failed; see {path}")
            ring_rows.extend(
                parse_rings(process.stdout, world_size, channels)
            )

    world4_channel0 = next(
        row
        for row in ring_rows
        if int(row["world_size"]) == 4
        and int(row["requested_channels"]) == 12
        and int(row["channel_id"]) == 0
    )
    ring_order = [
        int(value)
        for value in str(world4_channel0["ring_order"]).split("-")
    ]
    schedule = build_schedule(ring_order)

    write_csv(run_dir / "raw_measurements.csv", all_rows)
    write_csv(run_dir / "summary.csv", summaries)
    write_csv(run_dir / "rings.csv", ring_rows)
    write_csv(run_dir / "step_schedule.csv", schedule)

    tests_dir = args.root / "third_party/nccl-tests"
    nccl_dir = args.root / "third_party/nccl-2.22.3"
    manifest = [
        "experiment=ch12_ring_steps_channels",
        f"run_id={args.run_id}",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"hostname={os.uname().nodename}",
        f"cycles={args.cycles}",
        f"iterations={args.iterations}",
        "world_sizes=2,3,4",
        "sizes=1M,64M",
        "channels=1,2,4,8,12",
        "algorithm=Ring",
        "protocol=Simple",
        f"measurement_rows={len(all_rows)}",
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
        "# Chapter 12 experiment summary",
        "",
        f"- Measurement rows: {len(all_rows)}",
        "- Out-of-place correctness: PASS",
        "- In-place correctness: PASS",
        f"- World-4 channel-0 ring: {' -> '.join(map(str, ring_order))} -> {ring_order[0]}",
        f"- Source-operation schedule rows: {len(schedule)}",
        "- Each rank sends and receives exactly 2*(N-1) chunks per loop.",
        "",
        "## Channel sweep",
        "",
        "| N | channels | bytes | median us | P95 | CV | algbw | busbw | vs ch12 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['world_size']} | {row['requested_channels']} | "
            f"{row['size_bytes']} | "
            f"{float(row['median_time_us']):.3f} | "
            f"{float(row['p95_time_us']):.3f} | "
            f"{float(row['cycle_cv_percent']):.2f}% | "
            f"{float(row['median_algbw_GBs']):.2f} | "
            f"{float(row['median_busbw_GBs']):.2f} | "
            f"{float(row['time_vs_ch12_percent']):+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Reconstructed rings",
            "",
            "| N | requested channels | channel | total | order |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for row in ring_rows:
        lines.append(
            f"| {row['world_size']} | {row['requested_channels']} | "
            f"{row['channel_id']} | {row['reported_total_channels']} | "
            f"{row['ring_order']} |"
        )
    (run_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
