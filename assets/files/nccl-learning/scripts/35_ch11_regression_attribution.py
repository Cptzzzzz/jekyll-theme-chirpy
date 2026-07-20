#!/usr/bin/env python3
"""Inject layered NCCL regressions and extract performance/log fingerprints."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


ROW_RE = re.compile(r"^\s*\d+\s+")
COLL_CHANNEL_RE = re.compile(r"(\d+) coll channels")
SELECT_RE = re.compile(
    r"(\d+) Bytes -> Algo (\d+) proto (\d+) time ([0-9.]+)"
)
ALGORITHMS = {0: "Tree", 1: "Ring"}
PROTOCOLS = {0: "LL", 1: "LL128", 2: "Simple"}


@dataclass(frozen=True)
class Variant:
    name: str
    cpu_set: str
    env_items: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    noise_cpu: int | None = None
    reference: str = "baseline"

    @property
    def env(self) -> dict[str, str]:
        return dict(self.env_items)


VARIANTS = {
    "baseline": Variant("baseline", "0,2,4,6"),
    "channel1": Variant(
        "channel1",
        "0,2,4,6",
        (
            ("NCCL_MIN_NCHANNELS", "1"),
            ("NCCL_MAX_NCHANNELS", "1"),
        ),
    ),
    "no_p2p_numa0": Variant(
        "no_p2p_numa0",
        "0,2,4,6",
        (("NCCL_P2P_DISABLE", "1"),),
    ),
    "no_p2p_numa1": Variant(
        "no_p2p_numa1",
        "1,3,5,7",
        (("NCCL_P2P_DISABLE", "1"),),
        reference="no_p2p_numa0",
    ),
    "remote_numa": Variant("remote_numa", "1,3,5,7"),
    "single_core_idle": Variant(
        "single_core_idle", "0", reference="single_core_idle"
    ),
    "same_core_noise": Variant(
        "same_core_noise",
        "0",
        noise_cpu=0,
        reference="single_core_idle",
    ),
}


def ordered_specs() -> list[tuple[Variant, int]]:
    forward = [
        "baseline",
        "channel1",
        "no_p2p_numa0",
        "no_p2p_numa1",
        "remote_numa",
        "single_core_idle",
        "same_core_noise",
    ]
    reverse = list(reversed(forward))
    return (
        [(VARIANTS[name], 1) for name in forward]
        + [(VARIANTS[name], 2) for name in reverse]
    )


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
    text: str, variant: Variant, replicate: int, cycles: int
) -> list[dict[str, object]]:
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
                "variant": variant.name,
                "replicate": replicate,
                "cpu_set": variant.cpu_set,
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
    expected = cycles * 3
    if len(rows) != expected:
        raise RuntimeError(
            f"{variant.name} r{replicate}: expected {expected}, got {len(rows)}"
        )
    if any(
        int(row["wrong"]) != 0 or int(row["in_place_wrong"]) != 0
        for row in rows
    ):
        raise RuntimeError(
            f"{variant.name} r{replicate}: correctness failed"
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["variant"]), int(row["size_bytes"]))].append(row)
    medians = {
        key: statistics.median(float(row["time_us"]) for row in values)
        for key, values in groups.items()
    }
    bus_medians = {
        key: statistics.median(float(row["busbw_GBs"]) for row in values)
        for key, values in groups.items()
    }
    result = []
    for (variant_name, size_bytes), values in sorted(groups.items()):
        variant = VARIANTS[variant_name]
        times = [float(row["time_us"]) for row in values]
        reference_key = (variant.reference, size_bytes)
        reference_time = medians[reference_key]
        reference_bus = bus_medians[reference_key]
        median_time = medians[(variant_name, size_bytes)]
        median_bus = bus_medians[(variant_name, size_bytes)]
        result.append(
            {
                "variant": variant_name,
                "reference": variant.reference,
                "size_bytes": size_bytes,
                "samples": len(times),
                "median_time_us": median_time,
                "p95_time_us": percentile(times, 0.95),
                "cycle_cv_percent": cv_percent(times),
                "median_busbw_GBs": median_bus,
                "time_delta_percent":
                    (median_time / reference_time - 1.0) * 100.0,
                "busbw_retention_percent":
                    median_bus / reference_bus * 100.0
                    if reference_bus
                    else math.nan,
            }
        )
    return result


def parse_evidence(
    variant: Variant, text: str
) -> dict[str, object]:
    coll_channels = sorted(
        {int(value) for value in COLL_CHANNEL_RE.findall(text)}
    )
    selections: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for match in SELECT_RE.finditer(text):
        selections[int(match.group(1))].add(
            (int(match.group(2)), int(match.group(3)))
        )
    selection_parts = []
    for size_bytes, choices in sorted(selections.items()):
        if len(choices) != 1:
            raise RuntimeError(
                f"{variant.name}: inconsistent tuning choices {choices}"
            )
        algorithm, protocol = next(iter(choices))
        selection_parts.append(
            f"{size_bytes}:{ALGORITHMS.get(algorithm, str(algorithm))}"
            f"+{PROTOCOLS.get(protocol, str(protocol))}"
        )
    return {
        "variant": variant.name,
        "cpu_set": variant.cpu_set,
        "env": ";".join(f"{key}={value}" for key, value in variant.env_items)
        or "none",
        "coll_channels": ";".join(str(value) for value in coll_channels),
        "has_p2p_direct": "via P2P/direct pointer" in text,
        "has_shm_direct": "via SHM/direct/direct" in text,
        "p2p_connection_lines": text.count("via P2P/direct pointer"),
        "shm_connection_lines": text.count("via SHM/direct/direct"),
        "tuning_selections": ";".join(selection_parts),
    }


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

    run_dir = args.root / "logs/ch11_regression" / args.run_id
    raw_perf_dir = run_dir / "raw/performance"
    raw_evidence_dir = run_dir / "raw/evidence"
    raw_perf_dir.mkdir(parents=True, exist_ok=True)
    raw_evidence_dir.mkdir(parents=True, exist_ok=True)
    binary = args.root / "third_party/nccl-tests/build/all_reduce_perf"
    if not binary.is_file():
        raise FileNotFoundError(binary)

    base_args = [
        str(binary),
        "-b", "1K",
        "-e", "64M",
        "-f", "256",
        "-g", "4",
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
    all_rows: list[dict[str, object]] = []
    commands: list[str] = []
    for index, (variant, replicate) in enumerate(ordered_specs(), 1):
        command = ["taskset", "-c", variant.cpu_set, *base_args]
        environment = {
            **os.environ,
            **variant.env,
            "NCCL_DEBUG": "WARN",
        }
        commands.append(
            " ".join(
                [
                    *(f"{key}={value}" for key, value in variant.env_items),
                    *command,
                ]
            )
        )
        burner = start_burner(variant.noise_cpu)
        try:
            process = subprocess.run(
                command,
                cwd=args.root / "third_party/nccl-tests",
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        finally:
            stop_burner(burner)
        log_path = (
            raw_perf_dir
            / f"{index:02d}_{variant.name}_r{replicate}.log"
        )
        log_path.write_text(process.stdout, encoding="utf-8")
        if process.returncode != 0:
            raise RuntimeError(
                f"{variant.name} r{replicate} failed; see {log_path}"
            )
        rows = parse_rows(
            process.stdout, variant, replicate, args.cycles
        )
        all_rows.extend(rows)
        print(
            f"[{index:02d}/{len(ordered_specs())}] "
            f"{variant.name} r{replicate} PASS",
            flush=True,
        )

    expected_total = len(ordered_specs()) * args.cycles * 3
    if len(all_rows) != expected_total:
        raise RuntimeError(
            f"expected {expected_total} rows, got {len(all_rows)}"
        )
    summaries = summarize(all_rows)
    write_csv(run_dir / "raw_measurements.csv", all_rows)
    write_csv(run_dir / "summary.csv", summaries)

    evidence_rows = []
    evidence_args = [
        str(binary),
        "-b", "1K",
        "-e", "64M",
        "-f", "256",
        "-g", "4",
        "-w", "0",
        "-n", "1",
        "-N", "1",
        "-c", "0",
        "-I", "0",
        "-z", "0",
        "-C", "0",
        "-a", "3",
        "-d", "float",
        "-o", "sum",
    ]
    for variant in VARIANTS.values():
        command = ["taskset", "-c", variant.cpu_set, *evidence_args]
        process = subprocess.run(
            command,
            cwd=args.root / "third_party/nccl-tests",
            env={
                **os.environ,
                **variant.env,
                "NCCL_DEBUG": "INFO",
                "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,TUNING",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        path = raw_evidence_dir / f"{variant.name}.log"
        path.write_text(process.stdout, encoding="utf-8")
        if process.returncode != 0:
            raise RuntimeError(
                f"evidence run {variant.name} failed; see {path}"
            )
        evidence_rows.append(parse_evidence(variant, process.stdout))
    write_csv(run_dir / "evidence.csv", evidence_rows)

    tests_dir = args.root / "third_party/nccl-tests"
    nccl_dir = args.root / "third_party/nccl-2.22.3"
    manifest = [
        "experiment=ch11_regression_attribution",
        f"run_id={args.run_id}",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"hostname={os.uname().nodename}",
        f"cycles={args.cycles}",
        f"iterations={args.iterations}",
        "sizes=1K,256K,64M",
        "world_size=4",
        f"total_measurement_rows={len(all_rows)}",
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
        "# Chapter 11 experiment summary",
        "",
        f"- Performance process runs: {len(ordered_specs())}",
        f"- Measurement rows: {len(all_rows)}",
        "- Out-of-place correctness: PASS",
        "- In-place correctness: PASS",
        "",
        "## Performance fingerprints",
        "",
        "| variant | ref | bytes | median us | P95 us | CV | time delta | busbw retention |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['variant']} | {row['reference']} | "
            f"{row['size_bytes']} | "
            f"{float(row['median_time_us']):.3f} | "
            f"{float(row['p95_time_us']):.3f} | "
            f"{float(row['cycle_cv_percent']):.2f}% | "
            f"{float(row['time_delta_percent']):+.2f}% | "
            f"{float(row['busbw_retention_percent']):.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Log evidence",
            "",
            "| variant | coll channels | P2P | SHM | selections |",
            "|---|---|---|---|---|",
        ]
    )
    for row in evidence_rows:
        lines.append(
            f"| {row['variant']} | {row['coll_channels']} | "
            f"{row['has_p2p_direct']} | {row['has_shm_direct']} | "
            f"{row['tuning_selections']} |"
        )
    lines.extend(
        [
            "",
            "## Attribution rules",
            "",
            "- CV above 5% rejects an exact slowdown percentage.",
            "- A path override may also change graph/channel/tuning; evidence must report every changed layer.",
            "- same_core_noise is compared with single_core_idle, not the four-core baseline.",
            "- no_p2p_numa1 is compared with no_p2p_numa0 to isolate CPU/SHM placement within the fallback family.",
        ]
    )
    (run_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
