#!/usr/bin/env python3
"""Fit measured AllReduce latency/bandwidth regions and compare NCCL tuner estimates."""

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
SELECT_RE = re.compile(
    r"(\d+) Bytes -> Algo (\d+) proto (\d+) time ([0-9.]+)"
)
ALGORITHMS = {
    0: "Tree",
    1: "Ring",
    2: "CollNetDirect",
    3: "CollNetChain",
    4: "NVLS",
    5: "NVLSTree",
}
PROTOCOLS = {0: "LL", 1: "LL128", 2: "Simple"}


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


def parse_performance(text: str, cycles: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
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
                "cycle": cycle,
                "size_bytes": size_bytes,
                "count": int(fields[1]),
                "type": fields[2],
                "time_us": float(fields[5]),
                "algbw_GBs": float(fields[6]),
                "busbw_GBs": float(fields[7]),
                "wrong": int(fields[8]),
                "in_place_time_us": float(fields[9]),
                "in_place_algbw_GBs": float(fields[10]),
                "in_place_busbw_GBs": float(fields[11]),
                "in_place_wrong": int(fields[12]),
            }
        )
    expected_sizes = 31
    expected_rows = expected_sizes * cycles
    if len(rows) != expected_rows:
        raise RuntimeError(
            f"expected {expected_rows} performance rows, got {len(rows)}"
        )
    if any(
        int(row["wrong"]) != 0 or int(row["in_place_wrong"]) != 0
        for row in rows
    ):
        raise RuntimeError("performance correctness failed")
    return rows


def parse_selections(text: str) -> list[dict[str, object]]:
    candidates: dict[int, set[tuple[int, int, float]]] = defaultdict(set)
    for match in SELECT_RE.finditer(text):
        size_bytes = int(match.group(1))
        candidates[size_bytes].add(
            (
                int(match.group(2)),
                int(match.group(3)),
                float(match.group(4)),
            )
        )
    if len(candidates) != 31:
        raise RuntimeError(
            f"expected selections for 31 sizes, got {len(candidates)}"
        )
    result = []
    for size_bytes, choices in sorted(candidates.items()):
        if len(choices) != 1:
            raise RuntimeError(
                f"size {size_bytes} has inconsistent choices: {choices}"
            )
        algorithm, protocol, predicted_us = next(iter(choices))
        result.append(
            {
                "size_bytes": size_bytes,
                "algorithm_id": algorithm,
                "algorithm": ALGORITHMS[algorithm],
                "protocol_id": protocol,
                "protocol": PROTOCOLS[protocol],
                "tuner_predicted_us": predicted_us,
            }
        )
    return result


def linear_fit(
    points: list[tuple[float, float]],
) -> dict[str, float]:
    if len(points) < 2:
        raise ValueError("linear fit requires at least two points")
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = sum(
        (x - x_mean) * (y - y_mean) for x, y in points
    ) / denominator
    intercept = y_mean - slope * x_mean
    predictions = [intercept + slope * x for x in xs]
    residual_sum = sum(
        (actual - predicted) ** 2
        for actual, predicted in zip(ys, predictions)
    )
    total_sum = sum((actual - y_mean) ** 2 for actual in ys)
    r_squared = 1.0 - residual_sum / total_sum if total_sum else 1.0
    mape = statistics.fmean(
        abs(predicted / actual - 1.0) * 100.0
        for actual, predicted in zip(ys, predictions)
    )
    max_abs_relative = max(
        abs(predicted / actual - 1.0) * 100.0
        for actual, predicted in zip(ys, predictions)
    )
    # slope is microseconds/byte. Convert to decimal GB/s.
    beta_GBs = 1.0 / (slope * 1e3) if slope > 0 else math.inf
    return {
        "points": len(points),
        "min_bytes": min(xs),
        "max_bytes": max(xs),
        "alpha_us": intercept,
        "slope_us_per_byte": slope,
        "beta_GBs": beta_GBs,
        "r_squared": r_squared,
        "mape_percent": mape,
        "max_abs_relative_error_percent": max_abs_relative,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
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
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    run_dir = args.root / "logs/ch10_latency_model" / args.run_id
    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    binary = args.root / "third_party/nccl-tests/build/all_reduce_perf"
    if not binary.is_file():
        raise FileNotFoundError(binary)

    common = [
        "taskset", "-c", "0,2,4,6",
        str(binary),
        "-b", "1",
        "-e", "1G",
        "-f", "2",
        "-g", "4",
        "-d", "int8",
        "-o", "sum",
        "-z", "0",
        "-C", "0",
        "-a", "3",
    ]
    performance_command = common + [
        "-w", "5",
        "-n", str(args.iterations),
        "-N", str(args.cycles),
        "-c", "1",
        "-I", "0",
    ]
    tuning_command = common + [
        "-w", "0",
        "-n", "1",
        "-N", "1",
        "-c", "0",
        "-I", "0",
    ]

    performance = subprocess.run(
        performance_command,
        cwd=args.root / "third_party/nccl-tests",
        env={**os.environ, "NCCL_DEBUG": "WARN"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    (raw_dir / "performance.log").write_text(
        performance.stdout, encoding="utf-8"
    )
    if performance.returncode != 0:
        raise RuntimeError(
            f"performance command failed with {performance.returncode}"
        )
    raw_rows = parse_performance(performance.stdout, args.cycles)

    tuning = subprocess.run(
        tuning_command,
        cwd=args.root / "third_party/nccl-tests",
        env={
            **os.environ,
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "TUNING",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    (raw_dir / "tuning.log").write_text(tuning.stdout, encoding="utf-8")
    if tuning.returncode != 0:
        raise RuntimeError(
            f"tuning command failed with {tuning.returncode}"
        )
    selections = parse_selections(tuning.stdout)
    selection_by_size = {
        int(row["size_bytes"]): row for row in selections
    }
    write_csv(run_dir / "selections.csv", selections)

    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        groups[int(row["size_bytes"])].append(row)
    summaries: list[dict[str, object]] = []
    for size_bytes, rows in sorted(groups.items()):
        times = [float(row["time_us"]) for row in rows]
        selection = selection_by_size[size_bytes]
        median_time = statistics.median(times)
        summaries.append(
            {
                "size_bytes": size_bytes,
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
                "median_in_place_time_us": statistics.median(
                    float(row["in_place_time_us"]) for row in rows
                ),
                "algorithm_id": selection["algorithm_id"],
                "algorithm": selection["algorithm"],
                "protocol_id": selection["protocol_id"],
                "protocol": selection["protocol"],
                "tuner_predicted_us": selection["tuner_predicted_us"],
                "tuner_relative_error_percent":
                    (
                        float(selection["tuner_predicted_us"])
                        / median_time
                        - 1.0
                    )
                    * 100.0,
            }
        )

    fit_specs: list[tuple[str, list[dict[str, object]]]] = [
        ("global_all_sizes", summaries),
    ]
    for protocol in ("LL", "LL128", "Simple"):
        fit_specs.append(
            (
                f"auto_{protocol}",
                [row for row in summaries if row["protocol"] == protocol],
            )
        )
    fit_specs.append(
        (
            "large_simple_16MiB_plus",
            [
                row
                for row in summaries
                if row["protocol"] == "Simple"
                and int(row["size_bytes"]) >= 16 * 1024 * 1024
            ],
        )
    )

    fits: list[dict[str, object]] = []
    fit_by_name: dict[str, dict[str, object]] = {}
    for name, rows in fit_specs:
        fitted = linear_fit(
            [
                (float(row["size_bytes"]), float(row["median_time_us"]))
                for row in rows
            ]
        )
        result: dict[str, object] = {"model": name, **fitted}
        fits.append(result)
        fit_by_name[name] = result

    def predict(fit: dict[str, object], size_bytes: int) -> float:
        return float(fit["alpha_us"]) + float(
            fit["slope_us_per_byte"]
        ) * size_bytes

    previous: dict[str, object] | None = None
    for row in summaries:
        size_bytes = int(row["size_bytes"])
        measured = float(row["median_time_us"])
        global_prediction = predict(
            fit_by_name["global_all_sizes"], size_bytes
        )
        segment_prediction = predict(
            fit_by_name[f"auto_{row['protocol']}"], size_bytes
        )
        row["global_predicted_us"] = global_prediction
        row["global_residual_us"] = measured - global_prediction
        row["global_abs_relative_error_percent"] = (
            abs(global_prediction / measured - 1.0) * 100.0
        )
        row["segment_predicted_us"] = segment_prediction
        row["segment_residual_us"] = measured - segment_prediction
        row["segment_abs_relative_error_percent"] = (
            abs(segment_prediction / measured - 1.0) * 100.0
        )
        if previous is None:
            row["interval_effective_GBs"] = math.nan
        else:
            delta_bytes = size_bytes - int(previous["size_bytes"])
            delta_us = measured - float(previous["median_time_us"])
            row["interval_effective_GBs"] = (
                delta_bytes / delta_us / 1e3
                if delta_us > 0
                else math.nan
            )
        previous = row

    write_csv(run_dir / "raw_measurements.csv", raw_rows)
    write_csv(run_dir / "summary.csv", summaries)
    write_csv(run_dir / "model_fits.csv", fits)

    transitions = []
    for previous_row, row in zip(selections, selections[1:]):
        if (
            row["algorithm_id"] != previous_row["algorithm_id"]
            or row["protocol_id"] != previous_row["protocol_id"]
        ):
            transitions.append(
                f"{row['size_bytes']}:{row['algorithm']}+{row['protocol']}"
            )
    stable = [
        row
        for row in summaries
        if float(row["cycle_cv_percent"]) <= 5.0
    ]
    unstable = [
        row
        for row in summaries
        if float(row["cycle_cv_percent"]) > 5.0
    ]
    max_alg = max(float(row["median_algbw_GBs"]) for row in summaries)
    saturation = next(
        row
        for index, row in enumerate(summaries)
        if all(
            float(later["median_algbw_GBs"]) >= 0.8 * max_alg
            for later in summaries[index:]
        )
    )

    tests_dir = args.root / "third_party/nccl-tests"
    nccl_dir = args.root / "third_party/nccl-2.22.3"
    manifest = [
        "experiment=ch10_latency_bandwidth_model",
        f"run_id={args.run_id}",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"hostname={os.uname().nodename}",
        f"cycles={args.cycles}",
        f"iterations={args.iterations}",
        "sizes=1B..1GiB factor2",
        "datatype=int8",
        "world_size=4",
        "cpu_affinity=0,2,4,6 (NUMA0)",
        f"performance_command={' '.join(performance_command)}",
        f"tuning_command=NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=TUNING {' '.join(tuning_command)}",
        f"nccl_tests_commit={subprocess.check_output(['git', '-C', str(tests_dir), 'rev-parse', 'HEAD'], text=True).strip()}",
        f"nccl_source_commit={subprocess.check_output(['git', '-C', str(nccl_dir), 'rev-parse', 'HEAD'], text=True).strip()}",
    ]
    (run_dir / "manifest.txt").write_text(
        "\n".join(manifest) + "\n", encoding="utf-8"
    )

    lines = [
        "# Chapter 10 experiment summary",
        "",
        f"- Performance rows: {len(raw_rows)}",
        "- Out-of-place correctness: PASS",
        "- In-place correctness: PASS",
        f"- Stable size groups (CV <= 5%): {len(stable)}/31",
        f"- Unstable size groups: {len(unstable)}/31",
        f"- Automatic selection transitions: {', '.join(transitions)}",
        f"- First size keeping >=80% of maximum algbw for all later points: {saturation['size_bytes']} bytes",
        "",
        "## Model fits",
        "",
        "| model | points | range | alpha us | beta GB/s | R2 | MAPE | max rel error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fit in fits:
        lines.append(
            f"| {fit['model']} | {fit['points']} | "
            f"{int(float(fit['min_bytes']))}-{int(float(fit['max_bytes']))} | "
            f"{float(fit['alpha_us']):.3f} | "
            f"{float(fit['beta_GBs']):.3f} | "
            f"{float(fit['r_squared']):.6f} | "
            f"{float(fit['mape_percent']):.2f}% | "
            f"{float(fit['max_abs_relative_error_percent']):.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Per-size results",
            "",
            "| bytes | median us | P95 | CV | algbw | selection | tuner us | tuner err | segment err |",
            "|---:|---:|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['size_bytes']} | {float(row['median_time_us']):.3f} | "
            f"{float(row['p95_time_us']):.3f} | "
            f"{float(row['cycle_cv_percent']):.2f}% | "
            f"{float(row['median_algbw_GBs']):.2f} | "
            f"{row['algorithm']}+{row['protocol']} | "
            f"{float(row['tuner_predicted_us']):.3f} | "
            f"{float(row['tuner_relative_error_percent']):+.2f}% | "
            f"{float(row['segment_abs_relative_error_percent']):.2f}% |"
        )
    (run_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
