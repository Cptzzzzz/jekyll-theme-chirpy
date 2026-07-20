#!/usr/bin/env python3
"""Measure how nccl-tests methodology choices change stability and reported time."""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pynvml


@dataclass(frozen=True)
class RunSpec:
    matrix: str
    variant: str
    replicate: int
    warmup: int = 5
    iterations: int = 20
    blocking: int = 0
    per_iter: int = 1
    cpu: int = 0
    noise_cpu: int | None = None

    @property
    def config_id(self) -> str:
        return f"{self.matrix}_{self.variant}_r{self.replicate}"


def specs() -> list[RunSpec]:
    result: list[RunSpec] = []
    # Latin-square order distributes startup clock drift across warmup values.
    warmup_orders = (
        (0, 1, 5, 20),
        (1, 5, 20, 0),
        (5, 20, 0, 1),
        (20, 0, 1, 5),
    )
    for replicate, order in enumerate(warmup_orders, 1):
        result.extend(
            RunSpec("warmup", f"w{value}", replicate, warmup=value)
            for value in order
        )

    # Mirrored orders prevent iteration count from being identical to run order.
    for replicate, order in enumerate(
        ((1, 5, 20, 100), (100, 20, 5, 1)), 1
    ):
        result.extend(
            RunSpec("iterations", f"n{value}", replicate, iterations=value)
            for value in order
        )
    for replicate, order in enumerate(
        ((0, 1, 2, 3), (3, 2, 1, 0)), 1
    ):
        result.extend(
            RunSpec("blocking", f"z{value}", replicate, blocking=value)
            for value in order
        )
    result.extend(
        [
            RunSpec("instrumentation", "I0", 1, per_iter=0),
            RunSpec("instrumentation", "I1", 1, per_iter=1),
            RunSpec("instrumentation", "I1", 2, per_iter=1),
            RunSpec("instrumentation", "I0", 2, per_iter=0),
        ]
    )
    # Mirrored order limits a monotonic clock/temperature drift confound.
    result.extend(
        [
            RunSpec("affinity", "cpu0_numa0", 1, cpu=0),
            RunSpec("affinity", "cpu1_numa1", 1, cpu=1),
            RunSpec("affinity", "cpu1_numa1", 2, cpu=1),
            RunSpec("affinity", "cpu0_numa0", 2, cpu=0),
        ]
    )
    result.extend(
        [
            RunSpec("noise", "idle", 1, cpu=0),
            RunSpec("noise", "same_core", 1, cpu=0, noise_cpu=0),
            RunSpec("noise", "remote_numa", 1, cpu=0, noise_cpu=1),
            RunSpec("noise", "remote_numa", 2, cpu=0, noise_cpu=1),
            RunSpec("noise", "same_core", 2, cpu=0, noise_cpu=0),
            RunSpec("noise", "idle", 2, cpu=0),
        ]
    )
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def cv_percent(values: list[float]) -> float:
    mean = statistics.fmean(values)
    if not mean or len(values) < 2:
        return 0.0
    return statistics.pstdev(values) / mean * 100.0


def parse_rows(text: str, spec: RunSpec) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: dict[int, int] = defaultdict(int)
    for line in text.splitlines():
        fields = line.split()
        if not fields or not fields[0].isdigit():
            continue
        expected_fields = 21 if spec.per_iter else 13
        if len(fields) != expected_fields:
            continue
        size_bytes = int(fields[0])
        cycle = seen[size_bytes]
        seen[size_bytes] += 1
        if spec.per_iter:
            in_place_time = float(fields[13])
            in_place_busbw = float(fields[15])
            in_place_wrong = int(fields[16])
            iter_min = float(fields[9])
            iter_max = float(fields[10])
            iter_p99 = float(fields[11])
            iter_cv = float(fields[12])
        else:
            in_place_time = float(fields[9])
            in_place_busbw = float(fields[11])
            in_place_wrong = int(fields[12])
            iter_min = iter_max = iter_p99 = iter_cv = math.nan
        rows.append(
            {
                **asdict(spec),
                "config_id": spec.config_id,
                "cycle": cycle,
                "size_bytes": size_bytes,
                "count": int(fields[1]),
                "time_us": float(fields[5]),
                "algbw_GBs": float(fields[6]),
                "busbw_GBs": float(fields[7]),
                "wrong": int(fields[8]),
                "iter_min_us": iter_min,
                "iter_max_us": iter_max,
                "iter_p99_us": iter_p99,
                "iter_cv_percent": iter_cv,
                "in_place_time_us": in_place_time,
                "in_place_busbw_GBs": in_place_busbw,
                "in_place_wrong": in_place_wrong,
            }
        )
    return rows


def start_burner(cpu: int | None) -> subprocess.Popen[str] | None:
    if cpu is None:
        return None
    process = subprocess.Popen(
        ["taskset", "-c", str(cpu), "sh", "-c", "while :; do :; done"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    time.sleep(0.25)
    return process


def stop_burner(process: subprocess.Popen[str] | None) -> None:
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


class NvmlSampler:
    def __init__(self, config_id: str, interval_s: float = 0.1) -> None:
        self.config_id = config_id
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.rows: list[dict[str, object]] = []
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        handles = [
            pynvml.nvmlDeviceGetHandleByIndex(index)
            for index in range(pynvml.nvmlDeviceGetCount())
        ]
        started = time.monotonic()
        while not self.stop_event.is_set():
            for index, handle in enumerate(handles):
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.rows.append(
                    {
                        "config_id": self.config_id,
                        "elapsed_s": time.monotonic() - started,
                        "gpu": index,
                        "pstate": pynvml.nvmlDeviceGetPowerState(handle),
                        "sm_clock_MHz": pynvml.nvmlDeviceGetClockInfo(
                            handle, pynvml.NVML_CLOCK_SM
                        ),
                        "memory_clock_MHz": pynvml.nvmlDeviceGetClockInfo(
                            handle, pynvml.NVML_CLOCK_MEM
                        ),
                        "temperature_C": pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        ),
                        "power_W": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
                        "gpu_util_percent": utilization.gpu,
                        "memory_util_percent": utilization.memory,
                    }
                )
            self.stop_event.wait(self.interval_s)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> list[dict[str, object]]:
        self.stop_event.set()
        self.thread.join(timeout=3)
        return self.rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["matrix"]),
                str(row["variant"]),
                int(row["size_bytes"]),
            )
        ].append(row)

    references = {
        "warmup": "w5",
        "iterations": "n20",
        "blocking": "z0",
        "instrumentation": "I0",
        "affinity": "cpu0_numa0",
        "noise": "idle",
    }
    medians = {
        key: statistics.median(float(row["time_us"]) for row in group_rows)
        for key, group_rows in groups.items()
    }
    result: list[dict[str, object]] = []
    for (matrix, variant, size_bytes), group_rows in sorted(groups.items()):
        times = [float(row["time_us"]) for row in group_rows]
        in_place_times = [float(row["in_place_time_us"]) for row in group_rows]
        first = [
            float(row["time_us"])
            for row in group_rows
            if int(row["cycle"]) == 0
        ]
        steady = [
            float(row["time_us"])
            for row in group_rows
            if int(row["cycle"]) > 0
        ]
        median_time = statistics.median(times)
        reference = medians[(matrix, references[matrix], size_bytes)]
        first_median = statistics.median(first)
        steady_median = statistics.median(steady)
        result.append(
            {
                "matrix": matrix,
                "variant": variant,
                "size_bytes": size_bytes,
                "samples": len(times),
                "median_time_us": median_time,
                "p95_time_us": percentile(times, 0.95),
                "cycle_cv_percent": cv_percent(times),
                "median_busbw_GBs": statistics.median(
                    float(row["busbw_GBs"]) for row in group_rows
                ),
                "median_in_place_time_us": statistics.median(in_place_times),
                "relative_to_reference_percent":
                    (median_time / reference - 1.0) * 100.0,
                "first_cycle_median_us": first_median,
                "steady_cycle_median_us": steady_median,
                "first_cycle_penalty_percent":
                    (first_median / steady_median - 1.0) * 100.0,
            }
        )
    return result


def telemetry_summary(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["config_id"])].append(row)
    result = []
    for config_id, samples in sorted(groups.items()):
        active = [
            row for row in samples if int(row["gpu_util_percent"]) > 0
        ]
        selected = active or samples
        result.append(
            {
                "config_id": config_id,
                "samples": len(samples),
                "active_samples": len(active),
                "max_pstate": max(int(row["pstate"]) for row in selected),
                "min_sm_clock_MHz": min(
                    int(row["sm_clock_MHz"]) for row in selected
                ),
                "median_sm_clock_MHz": statistics.median(
                    int(row["sm_clock_MHz"]) for row in selected
                ),
                "max_sm_clock_MHz": max(
                    int(row["sm_clock_MHz"]) for row in selected
                ),
                "max_temperature_C": max(
                    int(row["temperature_C"]) for row in selected
                ),
                "max_power_W": max(float(row["power_W"]) for row in selected),
            }
        )
    return result


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
    args = parser.parse_args()

    run_dir = args.root / "logs/ch09_benchmark" / args.run_id
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    binary = args.root / "third_party/nccl-tests/build/all_reduce_perf"
    if not binary.is_file():
        raise FileNotFoundError(binary)

    pynvml.nvmlInit()
    all_rows: list[dict[str, object]] = []
    all_telemetry: list[dict[str, object]] = []
    commands: list[str] = []
    started_at = datetime.now(timezone.utc)
    try:
        for index, spec in enumerate(specs(), 1):
            command = [
                "taskset", "-c", str(spec.cpu),
                str(binary),
                "-b", "1M",
                "-e", "64M",
                "-f", "64",
                "-g", "4",
                "-w", str(spec.warmup),
                "-n", str(spec.iterations),
                "-N", str(args.cycles),
                "-c", "1",
                "-I", str(spec.per_iter),
                "-z", str(spec.blocking),
                "-d", "float",
                "-o", "sum",
                "-a", "3",
                "-C", "0",
            ]
            commands.append(" ".join(command))
            burner = start_burner(spec.noise_cpu)
            sampler = NvmlSampler(spec.config_id)
            sampler.start()
            try:
                process = subprocess.run(
                    command,
                    cwd=args.root / "third_party/nccl-tests",
                    env={**os.environ, "NCCL_DEBUG": "WARN"},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            finally:
                all_telemetry.extend(sampler.stop())
                stop_burner(burner)
            log_path = raw_dir / f"{index:02d}_{spec.config_id}.log"
            log_path.write_text(process.stdout, encoding="utf-8")
            if process.returncode != 0:
                raise RuntimeError(
                    f"{spec.config_id} failed with {process.returncode}; "
                    f"see {log_path}"
                )
            rows = parse_rows(process.stdout, spec)
            expected = args.cycles * 2
            if len(rows) != expected:
                raise RuntimeError(
                    f"{spec.config_id}: expected {expected} rows, got {len(rows)}"
                )
            if any(
                int(row["wrong"]) != 0
                or int(row["in_place_wrong"]) != 0
                for row in rows
            ):
                raise RuntimeError(f"{spec.config_id}: correctness failure")
            all_rows.extend(rows)
            print(
                f"[{index:02d}/{len(specs())}] {spec.config_id} "
                f"rows={len(rows)} PASS",
                flush=True,
            )
    finally:
        pynvml.nvmlShutdown()

    expected_total = len(specs()) * args.cycles * 2
    if len(all_rows) != expected_total:
        raise RuntimeError(
            f"expected {expected_total} total rows, got {len(all_rows)}"
        )

    summaries = summarize(all_rows)
    telemetry_summaries = telemetry_summary(all_telemetry)
    write_csv(run_dir / "raw_measurements.csv", all_rows)
    write_csv(run_dir / "summary.csv", summaries)
    write_csv(run_dir / "telemetry_raw.csv", all_telemetry)
    write_csv(run_dir / "telemetry_summary.csv", telemetry_summaries)

    tests_dir = args.root / "third_party/nccl-tests"
    nccl_dir = args.root / "third_party/nccl-2.22.3"
    manifest = [
        "experiment=ch09_benchmark_methodology",
        f"run_id={args.run_id}",
        f"started_at_utc={started_at.isoformat()}",
        f"completed_at_utc={datetime.now(timezone.utc).isoformat()}",
        f"hostname={platform.node()}",
        f"kernel={platform.release()}",
        f"cycles={args.cycles}",
        f"total_measurement_rows={len(all_rows)}",
        "sizes=1M,64M",
        "world_size=4",
        "gpu_affinity=all GPUs on NUMA0; CPU0 NUMA0; CPU1 NUMA1",
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
        "# Chapter 09 experiment summary",
        "",
        f"- Config process runs: {len(specs())}",
        f"- Out-of-place measurement rows: {len(all_rows)}",
        "- Out-of-place correctness: PASS for all rows",
        "- In-place correctness: PASS for all rows",
        "",
        "## Aggregated results",
        "",
        "| matrix | variant | bytes | n | median us | P95 us | CV | ref delta | first-cycle penalty |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['matrix']} | {row['variant']} | {row['size_bytes']} | "
            f"{row['samples']} | {float(row['median_time_us']):.3f} | "
            f"{float(row['p95_time_us']):.3f} | "
            f"{float(row['cycle_cv_percent']):.2f}% | "
            f"{float(row['relative_to_reference_percent']):+.2f}% | "
            f"{float(row['first_cycle_penalty_percent']):+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Method assertions",
            "",
            "- Warmup=0 still has BenchTime's untimed priming collective.",
            "- Blocking modes intentionally have different synchronization boundaries.",
            "- Per-iteration CUDA event reporting is treated as instrumentation and measured against I=0.",
            "- CPU affinity and noise runs use taskset; affinity variants use mirrored order.",
            "- Exact claims must be rejected for any group whose cycle CV exceeds 5%.",
        ]
    )
    (run_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
