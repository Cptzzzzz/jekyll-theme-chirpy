#!/usr/bin/env python3
"""Run and summarize the focused NCCL interview experiment suite.

Usage:
  python3 scripts/61_run_interview_detail_labs.py [--run-id ID] [--cycles 10]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def run_capture(command: list[str], env: dict[str, str] | None = None) -> tuple[int, str]:
    process = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return process.returncode, process.stdout


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csvs(directory: Path, suffix: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(directory.glob(f"rank*_{suffix}.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def cv_percent(values: list[float]) -> float:
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean * 100.0 if len(values) > 1 else 0.0


def truthy(value: str) -> bool:
    return value.lower() == "true"


def parse_nccl_tests(text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cycles: dict[int, int] = defaultdict(int)
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 13 or not parts[0].isdigit():
            continue
        size_bytes = int(parts[0])
        rows.append(
            {
                "cycle": cycles[size_bytes],
                "size_bytes": size_bytes,
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
        cycles[size_bytes] += 1
    return rows


def critical_cycles(
    rows: list[dict[str, str]],
    case_fields: tuple[str, ...],
    metric: str,
) -> dict[tuple[str, ...], list[float]]:
    per_cycle: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in case_fields) + (row["cycle"],)
        per_cycle[key].append(float(row[metric]))
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for key, rank_values in per_cycle.items():
        grouped[key[:-1]].append(max(rank_values))
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="NCCL learning asset root",
    )
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    )
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--decode-collectives", type=int, default=160)
    parser.add_argument("--cpu-affinity", default="0,2,4,6")
    parser.add_argument(
        "--nccl-tests",
        type=Path,
        default=Path("/root/nccl-learning/third_party/nccl-tests/build/all_reduce_perf"),
    )
    parser.add_argument(
        "--nccl-source",
        type=Path,
        default=Path("/root/nccl-learning/third_party/nccl-2.22.3"),
    )
    args = parser.parse_args()

    run_dir = args.root / "logs" / "interview_details" / args.run_id
    raw_dir = run_dir / "raw"
    result_root = raw_dir / "results"
    nccl_log_dir = raw_dir / "nccl"
    artifact_dir = run_dir / "artifacts"
    for directory in (result_root, nccl_log_dir, artifact_dir):
        directory.mkdir(parents=True, exist_ok=True)

    worker = args.root / "scripts" / "61_interview_detail_worker.py"
    if not worker.exists():
        raise FileNotFoundError(worker)

    artifact_commands = {
        "nvidia_smi_query.txt": [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id",
            "--format=csv,noheader",
        ],
        "nvidia_smi_topo.txt": ["nvidia-smi", "topo", "-m"],
        "ibv_devinfo.txt": ["ibv_devinfo"],
    }
    for filename, command in artifact_commands.items():
        _, output = run_capture(command)
        (artifact_dir / filename).write_text(output, encoding="utf-8")

    command_lines: list[str] = []

    def run_torch_case(name: str, world_size: int, sections: str) -> None:
        output_dir = result_root / name
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={world_size}",
            str(worker),
            "--output-dir",
            str(output_dir),
            "--sections",
            sections,
            "--cycles",
            str(args.cycles),
            "--warmups",
            str(args.warmups),
            "--decode-collectives",
            str(args.decode_collectives),
            "--cpu-affinity",
            args.cpu_affinity,
        ]
        env = os.environ.copy()
        env.update(
            {
                "NCCL_DEBUG": "INFO",
                "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,COLL",
                "NCCL_DEBUG_FILE": str(nccl_log_dir / f"{name}_%h_%p.log"),
            }
        )
        command_lines.append(
            f"NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL {' '.join(command)}"
        )
        return_code, output = run_capture(command, env=env)
        (raw_dir / f"{name}.stdout").write_text(output, encoding="utf-8")
        if return_code != 0:
            raise RuntimeError(f"{name} failed with code {return_code}:\n{output[-6000:]}")

    run_torch_case(
        "world4_full",
        4,
        "async,decomposition,fragmentation,decode,numerics",
    )
    run_torch_case("world2_decode", 2, "decode")

    nccl_tests_raw: list[dict[str, object]] = []
    for world_size in (2, 4):
        command = [
            str(args.nccl_tests),
            "-b",
            "8K",
            "-e",
            "1M",
            "-f",
            "2",
            "-g",
            str(world_size),
            "-w",
            "20",
            "-n",
            "100",
            "-N",
            str(args.cycles),
            "-c",
            "1",
            "-d",
            "half",
            "-a",
            "3",
        ]
        env = os.environ.copy()
        env.update(
            {
                "NCCL_DEBUG": "INFO",
                "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,COLL",
                "NCCL_DEBUG_FILE": str(
                    nccl_log_dir / f"nccl_tests_g{world_size}_%h_%p.log"
                ),
            }
        )
        command_lines.append(
            f"NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL {' '.join(command)}"
        )
        return_code, output = run_capture(command, env=env)
        (raw_dir / f"nccl_tests_g{world_size}.stdout").write_text(
            output, encoding="utf-8"
        )
        if return_code != 0:
            raise RuntimeError(
                f"nccl-tests g{world_size} failed with code {return_code}:\n{output[-6000:]}"
            )
        rows = parse_nccl_tests(output)
        if not rows:
            raise RuntimeError(f"nccl-tests g{world_size} produced no parseable rows")
        for row in rows:
            row["world_size"] = world_size
            nccl_tests_raw.append(row)
    if any(
        int(row["wrong"]) != 0 or int(row["inplace_wrong"]) != 0
        for row in nccl_tests_raw
    ):
        raise RuntimeError("nccl-tests correctness failure")
    write_csv(run_dir / "nccl_tests_raw.csv", nccl_tests_raw)

    world4 = result_root / "world4_full"
    world2 = result_root / "world2_decode"
    all_contracts = read_csvs(world4, "contracts") + read_csvs(world2, "contracts")
    failures = [row for row in all_contracts if not truthy(row["correct"])]
    if failures:
        raise RuntimeError(f"contract failures: {failures[:3]}")

    unified_rows: list[dict[str, object]] = []

    async_raw = read_csvs(world4, "async")
    async_gpu = critical_cycles(async_raw, ("world_size", "size_bytes"), "gpu_dependency_us")
    async_enqueue = critical_cycles(async_raw, ("world_size", "size_bytes"), "host_enqueue_us")
    async_wait = critical_cycles(async_raw, ("world_size", "size_bytes"), "host_wait_us")
    async_summary: list[dict[str, object]] = []
    for key, gpu_values in sorted(async_gpu.items(), key=lambda item: int(item[0][1])):
        world_size, size_bytes = key
        enqueue_values = async_enqueue[key]
        wait_values = async_wait[key]
        case_rows = [
            row
            for row in async_raw
            if row["world_size"] == world_size and row["size_bytes"] == size_bytes
        ]
        summary = {
            "world_size": int(world_size),
            "size_bytes": int(size_bytes),
            "cycles": len(gpu_values),
            "median_host_enqueue_us": statistics.median(enqueue_values),
            "median_host_wait_us": statistics.median(wait_values),
            "median_gpu_dependency_us": statistics.median(gpu_values),
            "p95_gpu_dependency_us": percentile(gpu_values, 0.95),
            "gpu_cv_percent": cv_percent(gpu_values),
            "work_completed_after_return_rate": statistics.fmean(
                truthy(row["work_completed_after_return"]) for row in case_rows
            ),
            "event_ready_after_wait_rate": statistics.fmean(
                truthy(row["event_ready_after_wait"]) for row in case_rows
            ),
            "correct": all(truthy(row["correct"]) for row in case_rows),
        }
        async_summary.append(summary)
        unified_rows.append(
            {
                "section": "async",
                "world_size": world_size,
                "case": f"bytes={size_bytes}",
                "metric": "gpu_dependency_us",
                "median": summary["median_gpu_dependency_us"],
                "p95": summary["p95_gpu_dependency_us"],
                "cv_percent": summary["gpu_cv_percent"],
                "samples": summary["cycles"],
                "correct": summary["correct"],
            }
        )
    write_csv(run_dir / "async_summary.csv", async_summary)

    decomposition_raw = read_csvs(world4, "decomposition")
    decomposition_groups = critical_cycles(
        decomposition_raw, ("world_size", "size_bytes", "mode"), "duration_us"
    )
    decomposition_medians = {
        key: statistics.median(values) for key, values in decomposition_groups.items()
    }
    decomposition_summary: list[dict[str, object]] = []
    for key, values in sorted(
        decomposition_groups.items(), key=lambda item: (int(item[0][1]), item[0][2])
    ):
        world_size, size_bytes, mode = key
        direct_median = decomposition_medians[(world_size, size_bytes, "direct")]
        median = statistics.median(values)
        overhead = (median / direct_median - 1.0) * 100.0
        summary = {
            "world_size": int(world_size),
            "size_bytes": int(size_bytes),
            "mode": mode,
            "cycles": len(values),
            "median_duration_us": median,
            "p95_duration_us": percentile(values, 0.95),
            "cv_percent": cv_percent(values),
            "overhead_vs_allreduce_percent": overhead,
            "correct": True,
        }
        decomposition_summary.append(summary)
        unified_rows.append(
            {
                "section": "decomposition",
                "world_size": world_size,
                "case": f"bytes={size_bytes};mode={mode}",
                "metric": "duration_us",
                "median": median,
                "p95": summary["p95_duration_us"],
                "cv_percent": summary["cv_percent"],
                "samples": summary["cycles"],
                "correct": True,
            }
        )
    write_csv(run_dir / "decomposition_summary.csv", decomposition_summary)

    fragmentation_raw = read_csvs(world4, "fragmentation")
    fragmentation_groups = critical_cycles(
        fragmentation_raw,
        ("world_size", "total_bytes", "fragment_count", "fragment_bytes"),
        "duration_us",
    )
    base_key = next(key for key in fragmentation_groups if key[2] == "1")
    base_median = statistics.median(fragmentation_groups[base_key])
    fragmentation_summary: list[dict[str, object]] = []
    for key, values in sorted(fragmentation_groups.items(), key=lambda item: int(item[0][2])):
        world_size, total_bytes, fragment_count, fragment_bytes = key
        median = statistics.median(values)
        summary = {
            "world_size": int(world_size),
            "total_bytes": int(total_bytes),
            "fragment_count": int(fragment_count),
            "fragment_bytes": int(fragment_bytes),
            "cycles": len(values),
            "median_duration_us": median,
            "p95_duration_us": percentile(values, 0.95),
            "min_duration_us": min(values),
            "max_duration_us": max(values),
            "cv_percent": cv_percent(values),
            "slowdown_vs_one_collective": median / base_median,
            "median_per_collective_us": median / int(fragment_count),
            "strictly_slower_than_one_in_all_cycles": (
                True
                if int(fragment_count) == 1
                else min(values) > max(fragmentation_groups[base_key])
            ),
            "correct": True,
        }
        fragmentation_summary.append(summary)
        unified_rows.append(
            {
                "section": "fragmentation",
                "world_size": world_size,
                "case": f"count={fragment_count};fragment_bytes={fragment_bytes}",
                "metric": "duration_us_for_80MiB",
                "median": median,
                "p95": summary["p95_duration_us"],
                "cv_percent": summary["cv_percent"],
                "samples": summary["cycles"],
                "correct": True,
            }
        )
    write_csv(run_dir / "fragmentation_summary.csv", fragmentation_summary)

    decode_summary: list[dict[str, object]] = []
    for directory in (world2, world4):
        decode_raw = read_csvs(directory, "decode")
        decode_groups = critical_cycles(
            decode_raw,
            ("world_size", "tokens", "message_bytes", "collectives"),
            "duration_us",
        )
        for key, values in sorted(
            decode_groups.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))
        ):
            world_size, tokens, message_bytes, collectives = key
            median = statistics.median(values)
            summary = {
                "world_size": int(world_size),
                "tokens": int(tokens),
                "hidden_size": 4096,
                "message_bytes": int(message_bytes),
                "collectives": int(collectives),
                "cycles": len(values),
                "median_total_us": median,
                "p95_total_us": percentile(values, 0.95),
                "min_total_us": min(values),
                "max_total_us": max(values),
                "cv_percent": cv_percent(values),
                "median_per_collective_us": median / int(collectives),
                "correct": True,
            }
            decode_summary.append(summary)
            unified_rows.append(
                {
                    "section": "decode",
                    "world_size": world_size,
                    "case": f"tokens={tokens};bytes={message_bytes};calls={collectives}",
                    "metric": "replay_total_us",
                    "median": median,
                    "p95": summary["p95_total_us"],
                    "cv_percent": summary["cv_percent"],
                    "samples": summary["cycles"],
                    "correct": True,
                }
            )
    write_csv(run_dir / "decode_summary.csv", decode_summary)

    selected_sizes = {8192, 65536, 262144, 1048576}
    nccl_tests_groups: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in nccl_tests_raw:
        size_bytes = int(row["size_bytes"])
        if size_bytes in selected_sizes:
            nccl_tests_groups[(int(row["world_size"]), size_bytes)].append(row)
    nccl_tests_summary: list[dict[str, object]] = []
    for (world_size, size_bytes), rows in sorted(nccl_tests_groups.items()):
        values = [float(row["time_us"]) for row in rows]
        summary = {
            "world_size": world_size,
            "size_bytes": size_bytes,
            "cycles": len(values),
            "median_time_us": statistics.median(values),
            "p95_time_us": percentile(values, 0.95),
            "min_time_us": min(values),
            "max_time_us": max(values),
            "cv_percent": cv_percent(values),
            "median_algbw_GBs": statistics.median(
                float(row["algbw_GBs"]) for row in rows
            ),
            "median_busbw_GBs": statistics.median(
                float(row["busbw_GBs"]) for row in rows
            ),
            "wrong_total": sum(
                int(row["wrong"]) + int(row["inplace_wrong"]) for row in rows
            ),
        }
        nccl_tests_summary.append(summary)
        unified_rows.append(
            {
                "section": "nccl_tests_decode_sizes",
                "world_size": world_size,
                "case": f"bytes={size_bytes}",
                "metric": "time_us",
                "median": summary["median_time_us"],
                "p95": summary["p95_time_us"],
                "cv_percent": summary["cv_percent"],
                "samples": summary["cycles"],
                "correct": summary["wrong_total"] == 0,
            }
        )
    write_csv(run_dir / "nccl_tests_summary.csv", nccl_tests_summary)

    numeric_raw = [
        row for row in read_csvs(world4, "numerics") if row["rank"] == "0"
    ]
    numeric_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in numeric_raw:
        numeric_groups[row["scenario"]].append(row)
    numeric_summary: list[dict[str, object]] = []
    for scenario, rows in sorted(numeric_groups.items()):
        fp16_mismatches = sum(not truthy(row["fp16_matches_real_sum"]) for row in rows)
        fp32_mismatches = sum(not truthy(row["fp32_matches_real_sum"]) for row in rows)
        fp16_nonfinite = sum(not math.isfinite(float(row["fp16_result"])) for row in rows)
        unique_fp16 = sorted({row["fp16_result"] for row in rows})
        summary = {
            "scenario": scenario,
            "cases": len(rows),
            "fp16_mismatch_count": fp16_mismatches,
            "fp16_nonfinite_count": fp16_nonfinite,
            "fp32_mismatch_count": fp32_mismatches,
            "unique_fp16_results": "/".join(unique_fp16),
            "correct_replication_and_fp32_reference": fp32_mismatches == 0,
        }
        numeric_summary.append(summary)
        unified_rows.append(
            {
                "section": "numerics",
                "world_size": 4,
                "case": scenario,
                "metric": "fp16_mismatch_count",
                "median": fp16_mismatches,
                "p95": "",
                "cv_percent": "",
                "samples": len(rows),
                "correct": fp32_mismatches == 0,
            }
        )
    write_csv(run_dir / "numeric_summary.csv", numeric_summary)
    write_csv(run_dir / "summary.csv", unified_rows)

    torch_info = subprocess.check_output(
        [
            "python3",
            "-c",
            "import torch; print(torch.__version__); print(torch.version.cuda); print('.'.join(map(str, torch.cuda.nccl.version())))",
        ],
        text=True,
    ).splitlines()
    source_commit = "unavailable"
    if (args.nccl_source / ".git").exists():
        source_commit = subprocess.check_output(
            ["git", "-C", str(args.nccl_source), "rev-parse", "HEAD"], text=True
        ).strip()
    worker_sha = hashlib.sha256(worker.read_bytes()).hexdigest()
    manifest = [
        "experiment=interview_details",
        f"run_id={args.run_id}",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"hostname={os.uname().nodename}",
        "gpu=4x Tesla V100-SXM2-32GB",
        "topology=all GPU pairs NV2",
        f"torch={torch_info[0]}",
        f"cuda_runtime={torch_info[1]}",
        f"nccl_runtime={torch_info[2]}",
        f"nccl_source_commit={source_commit}",
        f"worker_sha256={worker_sha}",
        f"cycles={args.cycles}",
        f"warmups={args.warmups}",
        f"decode_collectives={args.decode_collectives}",
        f"cpu_affinity={args.cpu_affinity} (local rank i uses entry i)",
        f"nccl_tests={args.nccl_tests}",
        "dtype=FP32 for async/decomposition; FP16 for fragmentation/decode/numeric counterexample/nccl-tests control",
        "critical_path_summary=max rank duration per cycle, then median/P95/CV across cycles",
        "NCCL_DEBUG=INFO",
        "NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL",
        "commands:",
        *[f"  {line}" for line in command_lines],
    ]
    (run_dir / "manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    lines = [
        "# NCCL Interview Detail Experiments",
        "",
        f"Run ID: `{args.run_id}`",
        "",
        f"All {len(all_contracts)} per-rank contracts passed; failures: {len(failures)}.",
        "Critical-path samples use the slowest rank in each cycle.",
        "",
        "## Async completion",
        "",
        "| bytes | host enqueue us | Work.wait host us | GPU dependency us | event ready after wait | CV |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in async_summary:
        lines.append(
            f"| {row['size_bytes']} | {row['median_host_enqueue_us']:.3f} | "
            f"{row['median_host_wait_us']:.3f} | {row['median_gpu_dependency_us']:.3f} | "
            f"{row['event_ready_after_wait_rate']:.1%} | {row['gpu_cv_percent']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## AllReduce versus explicit ReduceScatter plus AllGather",
            "",
            "| bytes | mode | median us | overhead vs AllReduce | CV |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in decomposition_summary:
        lines.append(
            f"| {row['size_bytes']} | {row['mode']} | {row['median_duration_us']:.3f} | "
            f"{row['overhead_vs_allreduce_percent']:.2f}% | {row['cv_percent']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Same 80 MiB logical payload split into multiple collectives",
            "",
            "| calls | bytes/call | median total us | slowdown vs one call | per call us | CV |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in fragmentation_summary:
        lines.append(
            f"| {row['fragment_count']} | {row['fragment_bytes']} | {row['median_duration_us']:.3f} | "
            f"{row['slowdown_vs_one_collective']:.2f}x | {row['median_per_collective_us']:.3f} | "
            f"{row['cv_percent']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Decode communication-only replay",
            "",
            "| TP | tokens | bytes/call | calls | median total us | per call us | CV |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in decode_summary:
        lines.append(
            f"| {row['world_size']} | {row['tokens']} | {row['message_bytes']} | "
            f"{row['collectives']} | {row['median_total_us']:.3f} | "
            f"{row['median_per_collective_us']:.3f} | {row['cv_percent']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## nccl-tests control for Decode-sized messages",
            "",
            "| ranks | bytes | median us | P95 us | algbw GB/s | busbw GB/s | CV | wrong |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in nccl_tests_summary:
        lines.append(
            f"| {row['world_size']} | {row['size_bytes']} | {row['median_time_us']:.3f} | "
            f"{row['p95_time_us']:.3f} | {row['median_algbw_GBs']:.3f} | "
            f"{row['median_busbw_GBs']:.3f} | {row['cv_percent']:.2f}% | "
            f"{row['wrong_total']} |"
        )
    lines.extend(
        [
            "",
            "## FP16 reduction counterexamples",
            "",
            "| scenario | permutations | FP16 mismatches | FP16 nonfinite | FP32 mismatches | FP16 outputs |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in numeric_summary:
        lines.append(
            f"| {row['scenario']} | {row['cases']} | {row['fp16_mismatch_count']} | "
            f"{row['fp16_nonfinite_count']} | {row['fp32_mismatch_count']} | "
            f"{row['unique_fp16_results']} |"
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "- These are single-node 4x V100 NV2 results for NCCL 2.22.3.",
            "- Decode replay contains communication only; it excludes GEMM, scheduling, CUDA Graph, and Custom AllReduce.",
            "- FP16 counterexamples prove that callers cannot assume implicit FP32 accumulation; they do not specify every GPU architecture's internal instruction sequence.",
            "- Exact performance percentages are valid only for the recorded hardware, software, and message matrix.",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((run_dir / "summary.md").read_text(encoding="utf-8"))
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
