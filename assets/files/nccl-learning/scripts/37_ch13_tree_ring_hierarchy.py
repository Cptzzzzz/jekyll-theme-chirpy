#!/usr/bin/env python3
"""Compare Ring and Tree and reconstruct NCCL's actual and synthetic trees."""

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
TREE_LINE_RE = re.compile(r"\[(\d+)\] NCCL INFO Trees (.*)")
TREE_ENTRY_RE = re.compile(
    r"\[(\d+)\]\s+(-?\d+)/(-?\d+)/(-?\d+)->(\d+)->(-?\d+)"
)
ALGORITHMS = ("Ring", "Tree")
WORLD_SIZES = (2, 3, 4)
SIZE_COUNT = 29  # 4 B through 1 GiB, powers of two.
CHANNELS = 12
PRACTICAL_TIE_PERCENT = 5.0


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
    world_size: int,
    algorithm: str,
    replicate: int,
    cycles: int,
) -> list[dict[str, object]]:
    seen: dict[int, int] = defaultdict(int)
    rows: list[dict[str, object]] = []
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
                "world_size": world_size,
                "algorithm": algorithm,
                "protocol": "Simple",
                "channels": CHANNELS,
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
    expected = SIZE_COUNT * cycles
    if len(rows) != expected:
        raise RuntimeError(
            f"N={world_size} {algorithm} r={replicate}: "
            f"expected {expected}, got {len(rows)}"
        )
    if set(seen.values()) != {cycles}:
        raise RuntimeError(f"unexpected cycle counts: {seen}")
    if any(
        int(row["wrong"]) != 0 or int(row["in_place_wrong"]) != 0
        for row in rows
    ):
        raise RuntimeError(
            f"N={world_size} {algorithm} r={replicate}: correctness"
        )
    return rows


def parse_active_trees(
    text: str, world_size: int
) -> list[dict[str, object]]:
    trees: dict[tuple[int, int], tuple[int, tuple[int, int, int]]] = {}
    for line in text.splitlines():
        line_match = TREE_LINE_RE.search(line)
        if not line_match:
            continue
        log_rank = int(line_match.group(1))
        for match in TREE_ENTRY_RE.finditer(line_match.group(2)):
            channel = int(match.group(1))
            children = (
                int(match.group(2)),
                int(match.group(3)),
                int(match.group(4)),
            )
            rank = int(match.group(5))
            up = int(match.group(6))
            if rank != log_rank:
                raise RuntimeError(
                    f"tree log rank {log_rank} contains rank {rank}"
                )
            key = (channel, rank)
            value = (up, children)
            if key in trees and trees[key] != value:
                raise RuntimeError(f"inconsistent active tree row {key}")
            trees[key] = value

    expected = CHANNELS * world_size
    if len(trees) != expected:
        raise RuntimeError(
            f"N={world_size}: expected {expected} tree rows, got {len(trees)}"
        )
    return [
        {
            "world_size": world_size,
            "channel": channel,
            "rank": rank,
            "up": value[0],
            "down0": value[1][0],
            "down1": value[1][1],
            "down2": value[1][2],
        }
        for (channel, rank), value in sorted(trees.items())
    ]


def validate_tree(
    nodes: dict[int, tuple[int, tuple[int, ...]]]
) -> tuple[int, int, int, str, int]:
    ranks = set(nodes)
    roots = [rank for rank, (up, _) in nodes.items() if up == -1]
    if len(roots) != 1:
        raise RuntimeError(f"expected one root, got {roots}")
    root = roots[0]
    edge_count = 0
    max_out_degree = 0
    for rank, (up, children) in nodes.items():
        valid_children = [child for child in children if child != -1]
        max_out_degree = max(max_out_degree, len(valid_children))
        edge_count += len(valid_children)
        if len(valid_children) != len(set(valid_children)):
            raise RuntimeError(f"rank {rank}: duplicate child")
        if up != -1:
            if up not in ranks or rank not in nodes[up][1]:
                raise RuntimeError(f"rank {rank}: parent edge is not mutual")
        for child in valid_children:
            if child not in ranks or nodes[child][0] != rank:
                raise RuntimeError(f"rank {rank}: child edge is not mutual")
    if edge_count != len(nodes) - 1:
        raise RuntimeError(f"edges={edge_count}, nodes={len(nodes)}")

    depths: dict[int, int] = {}
    for rank in nodes:
        current = rank
        visited = set()
        depth = 0
        while current != root:
            if current in visited:
                raise RuntimeError("cycle in tree")
            visited.add(current)
            current = nodes[current][0]
            depth += 1
            if current == -1:
                raise RuntimeError(f"rank {rank}: disconnected")
        depths[rank] = depth

    order = []
    current = root
    while current != -1:
        order.append(current)
        children = [child for child in nodes[current][1] if child != -1]
        current = children[0] if len(children) == 1 else -1
    chain_order = "-".join(map(str, order)) if len(order) == len(nodes) else ""
    return root, max(depths.values()), max_out_degree, chain_order, edge_count


def summarize_active_trees(
    rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    groups: dict[tuple[int, int], dict[int, tuple[int, tuple[int, ...]]]] = (
        defaultdict(dict)
    )
    for row in rows:
        groups[(int(row["world_size"]), int(row["channel"]))][
            int(row["rank"])
        ] = (
            int(row["up"]),
            (
                int(row["down0"]),
                int(row["down1"]),
                int(row["down2"]),
            ),
        )
    summaries = []
    for (world_size, channel), nodes in sorted(groups.items()):
        root, depth, arity, order, edges = validate_tree(nodes)
        summaries.append(
            {
                "world_size": world_size,
                "channel": channel,
                "root": root,
                "max_depth": depth,
                "max_out_degree": arity,
                "edges": edges,
                "is_chain": int(bool(order)),
                "root_to_leaf_order": order,
            }
        )
    return summaries


def run_dtree_probe(
    root: Path, run_dir: Path
) -> tuple[list[dict[str, object]], list[dict[str, object]], str]:
    nccl_dir = root / "third_party/nccl-2.22.3"
    probe_source = root / "probes/ch13_dtree_probe.cc"
    probe_binary = run_dir / "ch13_dtree_probe"
    compile_command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-I/usr/include",
        "-I/usr/local/cuda-12.6/targets/x86_64-linux/include",
        str(probe_source),
        str(nccl_dir / "src/graph/trees.cc"),
        "-o",
        str(probe_binary),
    ]
    subprocess.run(compile_command, check=True, cwd=root)
    output = subprocess.check_output([str(probe_binary), "16"], text=True)
    raw_path = run_dir / "dtree_raw.csv"
    raw_path.write_text(output, encoding="utf-8")
    raw_rows = list(csv.DictReader(output.splitlines()))

    summaries: list[dict[str, object]] = []
    for nodes_count in range(1, 17):
        selected = [
            row for row in raw_rows if int(row["nodes"]) == nodes_count
        ]
        tree_nodes = []
        internals = []
        roots = []
        depths = []
        for tree_id in (0, 1):
            nodes = {
                int(row["rank"]): (
                    int(row[f"t{tree_id}_up"]),
                    (
                        int(row[f"t{tree_id}_down0"]),
                        int(row[f"t{tree_id}_down1"]),
                    ),
                )
                for row in selected
            }
            root_rank, depth, _, _, _ = validate_tree(nodes)
            internal = {
                rank
                for rank, (_, children) in nodes.items()
                if any(child != -1 for child in children)
            }
            tree_nodes.append(nodes)
            internals.append(internal)
            roots.append(root_rank)
            depths.append(depth)
        summaries.append(
            {
                "nodes": nodes_count,
                "construction": (
                    "trivial"
                    if nodes_count == 1
                    else "shift"
                    if nodes_count % 2
                    else "mirror"
                ),
                "tree0_root": roots[0],
                "tree1_root": roots[1],
                "tree0_depth": depths[0],
                "tree1_depth": depths[1],
                "tree0_internal": "-".join(map(str, sorted(internals[0]))),
                "tree1_internal": "-".join(map(str, sorted(internals[1]))),
                "internal_overlap": "-".join(
                    map(str, sorted(internals[0] & internals[1]))
                ),
                "internal_overlap_count": len(internals[0] & internals[1]),
            }
        )
    return raw_rows, summaries, " ".join(compile_command)


def build_winner_segments(
    summaries: list[dict[str, object]]
) -> list[dict[str, object]]:
    by_world_size: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    for row in summaries:
        by_world_size[(int(row["world_size"]), int(row["size_bytes"]))][
            str(row["algorithm"])
        ] = float(row["median_time_us"])

    segments = []
    for world_size in WORLD_SIZES:
        points = []
        for (world, size), times in sorted(by_world_size.items()):
            if world != world_size or set(times) != set(ALGORITHMS):
                continue
            winner = min(times, key=times.get)
            loser = "Tree" if winner == "Ring" else "Ring"
            margin = (times[loser] / times[winner] - 1.0) * 100.0
            if margin < PRACTICAL_TIE_PERCENT:
                winner = "Tie"
            points.append((size, winner, margin))
        start = 0
        while start < len(points):
            end = start
            while end + 1 < len(points) and points[end + 1][1] == points[start][1]:
                end += 1
            margins = [point[2] for point in points[start : end + 1]]
            segments.append(
                {
                    "world_size": world_size,
                    "winner": points[start][1],
                    "start_bytes": points[start][0],
                    "end_bytes": points[end][0],
                    "points": end - start + 1,
                    "median_winner_margin_percent": statistics.median(margins),
                    "max_winner_margin_percent": max(margins),
                }
            )
            start = end + 1
    return segments


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
    args = parser.parse_args()

    run_dir = args.root / "logs/ch13_tree" / args.run_id
    perf_dir = run_dir / "raw/performance"
    graph_dir = run_dir / "raw/graph"
    perf_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)
    binary = args.root / "third_party/nccl-tests/build/all_reduce_perf"

    all_rows: list[dict[str, object]] = []
    commands: list[str] = []
    schedules = (
        (1, WORLD_SIZES, ALGORITHMS),
        (2, tuple(reversed(WORLD_SIZES)), tuple(reversed(ALGORITHMS))),
    )
    run_index = 0
    for replicate, worlds, algorithms in schedules:
        for world_size in worlds:
            for algorithm in algorithms:
                run_index += 1
                command = [
                    str(binary),
                    "-b", "4",
                    "-e", "1G",
                    "-f", "2",
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
                    "NCCL_ALGO": algorithm,
                    "NCCL_PROTO": "Simple",
                    "NCCL_MIN_NCHANNELS": str(CHANNELS),
                    "NCCL_MAX_NCHANNELS": str(CHANNELS),
                    "NCCL_DEBUG": "WARN",
                }
                command_text = (
                    f"NCCL_ALGO={algorithm} NCCL_PROTO=Simple "
                    f"NCCL_MIN_NCHANNELS={CHANNELS} "
                    f"NCCL_MAX_NCHANNELS={CHANNELS} "
                    + " ".join(command)
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
                path = perf_dir / (
                    f"{run_index:02d}_n{world_size}_{algorithm.lower()}_r{replicate}.log"
                )
                path.write_text(process.stdout, encoding="utf-8")
                if process.returncode != 0:
                    raise RuntimeError(f"failed; see {path}")
                all_rows.extend(
                    parse_perf(
                        process.stdout,
                        world_size,
                        algorithm,
                        replicate,
                        args.cycles,
                    )
                )
                print(
                    f"[{run_index:02d}/12] N={world_size} "
                    f"{algorithm} r={replicate} PASS",
                    flush=True,
                )

    expected_rows = 2 * len(WORLD_SIZES) * len(ALGORITHMS) * SIZE_COUNT * args.cycles
    if len(all_rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} rows, got {len(all_rows)}")

    groups: dict[tuple[int, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in all_rows:
        groups[
            (
                int(row["world_size"]),
                str(row["algorithm"]),
                int(row["size_bytes"]),
            )
        ].append(row)
    summaries = []
    for (world_size, algorithm, size_bytes), rows in sorted(groups.items()):
        times = [float(row["time_us"]) for row in rows]
        summaries.append(
            {
                "world_size": world_size,
                "algorithm": algorithm,
                "protocol": "Simple",
                "channels": CHANNELS,
                "size_bytes": size_bytes,
                "samples": len(times),
                "median_time_us": statistics.median(times),
                "p95_time_us": percentile(times, 0.95),
                "cycle_cv_percent": cv_percent(times),
                "median_algbw_GBs": statistics.median(
                    float(row["algbw_GBs"]) for row in rows
                ),
                "median_busbw_GBs": statistics.median(
                    float(row["busbw_GBs"]) for row in rows
                ),
            }
        )
    winner_segments = build_winner_segments(summaries)

    active_tree_rows: list[dict[str, object]] = []
    for graph_index, world_size in enumerate(WORLD_SIZES, 1):
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
                "NCCL_ALGO": "Tree",
                "NCCL_PROTO": "Simple",
                "NCCL_MIN_NCHANNELS": str(CHANNELS),
                "NCCL_MAX_NCHANNELS": str(CHANNELS),
                "NCCL_DEBUG": "INFO",
                "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,TUNING",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        path = graph_dir / f"{graph_index:02d}_n{world_size}_tree_ch12.log"
        path.write_text(process.stdout, encoding="utf-8")
        if process.returncode != 0:
            raise RuntimeError(f"graph run failed; see {path}")
        active_tree_rows.extend(parse_active_trees(process.stdout, world_size))
    active_tree_summaries = summarize_active_trees(active_tree_rows)

    dtree_raw, dtree_summaries, compile_command = run_dtree_probe(
        args.root, run_dir
    )

    write_csv(run_dir / "raw_measurements.csv", all_rows)
    write_csv(run_dir / "summary.csv", summaries)
    write_csv(run_dir / "winner_segments.csv", winner_segments)
    write_csv(run_dir / "active_trees.csv", active_tree_rows)
    write_csv(run_dir / "active_tree_summary.csv", active_tree_summaries)
    write_csv(run_dir / "dtree_summary.csv", dtree_summaries)

    tests_dir = args.root / "third_party/nccl-tests"
    nccl_dir = args.root / "third_party/nccl-2.22.3"
    manifest = [
        "experiment=ch13_tree_ring_hierarchy",
        f"run_id={args.run_id}",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"hostname={os.uname().nodename}",
        f"cycles={args.cycles}",
        f"iterations={args.iterations}",
        "world_sizes=2,3,4",
        "sizes=4B..1GiB_factor2",
        "algorithms=Ring,Tree",
        "protocol=Simple",
        f"channels={CHANNELS}",
        f"practical_tie_percent={PRACTICAL_TIE_PERCENT}",
        f"measurement_rows={len(all_rows)}",
        f"active_tree_rows={len(active_tree_rows)}",
        f"dtree_rows={len(dtree_raw)}",
        f"dtree_compile={compile_command}",
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
        "# Chapter 13 experiment summary",
        "",
        f"- Measurement rows: {len(all_rows)}",
        "- Out-of-place correctness: PASS",
        "- In-place correctness: PASS",
        f"- Active single-node tree rows: {len(active_tree_rows)}",
        "- All active single-node trees are valid connected trees.",
        "- The synthetic DTree probe directly compiles src/graph/trees.cc.",
        "",
        "## Winner segments",
        "",
        "| N | winner | start B | end B | points | median margin | max margin |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in winner_segments:
        lines.append(
            f"| {row['world_size']} | {row['winner']} | "
            f"{row['start_bytes']} | {row['end_bytes']} | {row['points']} | "
            f"{float(row['median_winner_margin_percent']):.2f}% | "
            f"{float(row['max_winner_margin_percent']):.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Selected performance points",
            "",
            "| N | algo | bytes | median us | P95 | CV | busbw |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    selected_sizes = {4, 4096, 1048576, 67108864, 1073741824}
    for row in summaries:
        if int(row["size_bytes"]) not in selected_sizes:
            continue
        lines.append(
            f"| {row['world_size']} | {row['algorithm']} | "
            f"{row['size_bytes']} | {float(row['median_time_us']):.3f} | "
            f"{float(row['p95_time_us']):.3f} | "
            f"{float(row['cycle_cv_percent']):.2f}% | "
            f"{float(row['median_busbw_GBs']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Active tree shapes",
            "",
            "| N | ch | root | depth | max arity | chain | order |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in active_tree_summaries:
        lines.append(
            f"| {row['world_size']} | {row['channel']} | {row['root']} | "
            f"{row['max_depth']} | {row['max_out_degree']} | "
            f"{row['is_chain']} | {row['root_to_leaf_order']} |"
        )
    lines.extend(
        [
            "",
            "## Source DTree construction",
            "",
            "| nodes | mode | roots | depths | internal overlap |",
            "|---:|---|---|---|---|",
        ]
    )
    for row in dtree_summaries:
        lines.append(
            f"| {row['nodes']} | {row['construction']} | "
            f"{row['tree0_root']}/{row['tree1_root']} | "
            f"{row['tree0_depth']}/{row['tree1_depth']} | "
            f"{row['internal_overlap'] or '-'} |"
        )
    (run_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
