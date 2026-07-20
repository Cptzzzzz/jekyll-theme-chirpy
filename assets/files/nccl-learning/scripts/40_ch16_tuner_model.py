#!/usr/bin/env python3
"""Recompute NCCL's cost model and prove tuner-plugin overrides end to end."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
SELECTION_RE = re.compile(
    r"(\d+) Bytes -> Algo (\d+) proto (\d+) time (-?[0-9.]+)"
)
PAIR_RE = re.compile(r"([0-9.]+)/\s*([0-9.]+)")
ALGORITHMS = ("Tree", "Ring")
PROTOCOLS = ("LL", "LL128", "Simple")
ALGORITHM_ID = {"Tree": 0, "Ring": 1}
PROTOCOL_ID = {"LL": 0, "LL128": 1, "Simple": 2}
TREE_CORRECTION = {
    "LL": (1, 1, 1, 1, .9, .8, .7, .7, .7, .7, .6, .5, .4,
           .4, .5, .6, .7, .8, .9, 1, 1, 1, 1),
    "LL128": (1, 1, 1, 1, 1, .9, .8, .8, .8, .7, .6, .6, .6,
              .6, .6, .6, .8, .9, .9, .9, .9, 1, 1),
    "Simple": (.9, .9, .9, .9, .9, .9, .9, .8, .7, .6, .6, .5,
               .5, .5, .5, .6, .7, .8, .7, .7, .8, .9, .9),
}
SIZE_COUNT = 29
TARGET_BYTES = 64 << 20
PRACTICAL_TIE_PERCENT = 5.0


def clean_environment() -> dict[str, str]:
    env = dict(os.environ)
    for name in tuple(env):
        if name.startswith("NCCL_") or name.startswith("TORCH_NCCL_"):
            env.pop(name)
    return env


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def cv_percent(values: list[float]) -> float:
    mean = statistics.fmean(values)
    if mean == 0 or len(values) < 2:
        return 0.0
    return statistics.pstdev(values) / mean * 100.0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_logged(
    command: list[str], env: dict[str, str], path: Path, cwd: Path
) -> str:
    process = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(process.stdout, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"command failed; see {path}")
    return process.stdout


def perf_command(binary: Path, begin: str, end: str, cycles: int,
                 iterations: int) -> list[str]:
    return [
        str(binary), "-b", begin, "-e", end, "-f", "2", "-g", "4",
        "-w", "5", "-n", str(iterations), "-N", str(cycles), "-c", "1",
        "-I", "0", "-z", "0", "-u", "0", "-C", "0", "-a", "3",
        "-d", "float", "-o", "sum",
    ]


def parse_perf(text: str, experiment: str, replicate: int,
               expected_sizes: int, cycles: int) -> list[dict[str, object]]:
    rows = []
    seen: dict[int, int] = defaultdict(int)
    for line in text.splitlines():
        if not ROW_RE.match(line):
            continue
        fields = line.split()
        if len(fields) != 13 or int(fields[0]) < 4:
            continue
        size = int(fields[0])
        cycle = seen[size]
        seen[size] += 1
        rows.append({
            "experiment": experiment,
            "replicate": replicate,
            "cycle": cycle,
            "size_bytes": size,
            "time_us": float(fields[5]),
            "algbw_GBs": float(fields[6]),
            "busbw_GBs": float(fields[7]),
            "wrong": int(fields[8]),
            "in_place_time_us": float(fields[9]),
            "in_place_busbw_GBs": float(fields[11]),
            "in_place_wrong": int(fields[12]),
        })
    if len(rows) != expected_sizes * cycles:
        raise RuntimeError(
            f"{experiment} r={replicate}: expected {expected_sizes*cycles}, "
            f"got {len(rows)}"
        )
    if len(seen) != expected_sizes or set(seen.values()) != {cycles}:
        raise RuntimeError(f"unexpected size/cycle counts: {seen}")
    if any(row["wrong"] or row["in_place_wrong"] for row in rows):
        raise RuntimeError(f"correctness failure: {experiment}")
    return rows


def summarize_perf(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["experiment"]), int(row["size_bytes"]))].append(row)
    output = []
    for (experiment, size), group in sorted(groups.items()):
        times = [float(row["time_us"]) for row in group]
        output.append({
            "experiment": experiment,
            "size_bytes": size,
            "samples": len(group),
            "median_time_us": statistics.median(times),
            "p95_time_us": percentile(times, .95),
            "cycle_cv_percent": cv_percent(times),
            "min_time_us": min(times),
            "max_time_us": max(times),
            "median_busbw_GBs": statistics.median(
                float(row["busbw_GBs"]) for row in group
            ),
        })
    return output


def parse_model_table(text: str) -> list[dict[str, object]]:
    candidate_lines = [line for line in text.splitlines()
                       if "NCCL INFO" in line and "AllReduce |" in line]
    for line in candidate_lines:
        pairs = [(float(lat), float(bw)) for lat, bw in PAIR_RE.findall(line)]
        if len(pairs) >= 6 and pairs[0][1] > 0 and pairs[3][1] > 0:
            rows = []
            for index, (algorithm, protocol) in enumerate(
                (a, p) for a in ALGORITHMS for p in PROTOCOLS
            ):
                latency, bandwidth = pairs[index]
                rows.append({
                    "algorithm": algorithm,
                    "algorithm_id": ALGORITHM_ID[algorithm],
                    "protocol": protocol,
                    "protocol_id": PROTOCOL_ID[protocol],
                    "base_latency_us": latency,
                    "algorithm_bandwidth_GBs": bandwidth,
                    "available": int(bandwidth > 0),
                })
            return rows
    raise RuntimeError("could not parse Tree/Ring AllReduce model table")


def parse_selections(text: str, expected_sizes: set[int]) -> dict[int, tuple[int, int, float]]:
    grouped: dict[int, set[tuple[int, int, float]]] = defaultdict(set)
    for size, algorithm, protocol, time_us in SELECTION_RE.findall(text):
        size_i = int(size)
        if size_i in expected_sizes:
            grouped[size_i].add((int(algorithm), int(protocol), float(time_us)))
    if set(grouped) != expected_sizes:
        missing = sorted(expected_sizes - set(grouped))
        raise RuntimeError(f"selection log missing sizes: {missing}")
    result = {}
    for size, values in grouped.items():
        identities = {(a, p) for a, p, _ in values}
        if len(identities) != 1:
            raise RuntimeError(f"inconsistent selections for {size}: {values}")
        # Different ranks print identical float values in this environment.
        result[size] = min(values, key=lambda value: value[2])
    return result


def tree_factor(protocol: str, nbytes: int) -> tuple[int, float]:
    shifted = nbytes >> 6
    log_size = -1 if shifted == 0 else shifted.bit_length() - 1
    factors = TREE_CORRECTION[protocol]
    factor = factors[log_size] if 0 <= log_size < len(factors) else 1.0
    return log_size, factor


def recompute_model(
    table: list[dict[str, object]],
    selections: dict[int, tuple[int, int, float]],
) -> list[dict[str, object]]:
    rows = []
    for size in sorted(selections):
        logged_algo, logged_proto, logged_time = selections[size]
        candidates = []
        for entry in table:
            algorithm = str(entry["algorithm"])
            protocol = str(entry["protocol"])
            bw = float(entry["algorithm_bandwidth_GBs"])
            if bw == 0:
                continue
            log_size, correction = tree_factor(protocol, size)
            effective_bw = bw * (correction if algorithm == "Tree" else 1.0)
            predicted = float(entry["base_latency_us"]) + size / (1000 * effective_bw)
            candidate = {
                "size_bytes": size,
                "algorithm": algorithm,
                "algorithm_id": int(entry["algorithm_id"]),
                "protocol": protocol,
                "protocol_id": int(entry["protocol_id"]),
                "base_latency_us": entry["base_latency_us"],
                "base_bandwidth_GBs": bw,
                "tree_log_size": log_size if algorithm == "Tree" else "",
                "tree_correction": correction if algorithm == "Tree" else 1.0,
                "effective_bandwidth_GBs": effective_bw,
                "predicted_time_us": predicted,
                "logged_selected": int(
                    int(entry["algorithm_id"]) == logged_algo
                    and int(entry["protocol_id"]) == logged_proto
                ),
                "logged_selected_time_us": logged_time if (
                    int(entry["algorithm_id"]) == logged_algo
                    and int(entry["protocol_id"]) == logged_proto
                ) else "",
            }
            candidates.append(candidate)
            rows.append(candidate)
        predicted_best = min(candidates, key=lambda row: float(row["predicted_time_us"]))
        if (int(predicted_best["algorithm_id"]), int(predicted_best["protocol_id"])) != (
            logged_algo, logged_proto
        ):
            raise RuntimeError(
                f"rounded-table recomputation mismatch at {size}: "
                f"{predicted_best} vs {(logged_algo, logged_proto)}"
            )
        predicted_time = float(predicted_best["predicted_time_us"])
        tolerance = max(.15, logged_time * .005)
        if abs(predicted_time - logged_time) > tolerance:
            raise RuntimeError(
                f"time mismatch at {size}: {predicted_time} vs {logged_time}"
            )
    return rows


def compare_to_measurement(
    root: Path,
    candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    ch14 = root / "logs/ch14_protocol/20260710T161632Z/summary.csv"
    with ch14.open(newline="", encoding="utf-8") as handle:
        measured = list(csv.DictReader(handle))
    measured_lookup = {
        (int(row["size_bytes"]), row["algorithm"], row["protocol"]): float(row["median_time_us"])
        for row in measured
    }
    by_size: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in candidates:
        by_size[int(row["size_bytes"])].append(row)
    output = []
    for size, group in sorted(by_size.items()):
        selected = next(row for row in group if int(row["logged_selected"]))
        measured_group = []
        for row in group:
            key = (size, str(row["algorithm"]), str(row["protocol"]))
            measured_group.append((row, measured_lookup[key]))
        best_row, best_time = min(measured_group, key=lambda pair: pair[1])
        selected_time = measured_lookup[
            (size, str(selected["algorithm"]), str(selected["protocol"]))
        ]
        regret = (selected_time / best_time - 1.0) * 100.0
        predicted = float(selected["predicted_time_us"])
        output.append({
            "size_bytes": size,
            "auto_algorithm": selected["algorithm"],
            "auto_protocol": selected["protocol"],
            "model_time_us": predicted,
            "auto_measured_median_us": selected_time,
            "measured_best_algorithm": best_row["algorithm"],
            "measured_best_protocol": best_row["protocol"],
            "measured_best_median_us": best_time,
            "exact_selection_hit": int(
                selected["algorithm"] == best_row["algorithm"]
                and selected["protocol"] == best_row["protocol"]
            ),
            "practical_selection_hit_5pct": int(regret < PRACTICAL_TIE_PERCENT),
            "selection_regret_percent": regret,
            "selected_model_abs_error_percent": abs(predicted / selected_time - 1.0) * 100.0,
        })
    return output


def compile_plugin(root: Path, run_dir: Path, private_dir: Path) -> Path:
    source = root / "probes/ch16_force_tree_simple_tuner.cc"
    library = private_dir / "ch16_force_tree_simple_tuner.so"
    private_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "g++", "-std=c++17", "-O2", "-fPIC", "-shared", "-Wall", "-Wextra",
        f"-I{root / 'third_party/nccl-2.22.3/src/include'}",
        "-I/usr/local/cuda/include", str(source), "-o", str(library),
    ]
    run_logged(command, clean_environment(), run_dir / "raw/plugin/build.log", root)
    return library


def profile_plugin(
    root: Path, run_dir: Path, private_dir: Path, binary: Path, plugin: Path
) -> dict[str, object]:
    stem = private_dir / "force_tree_simple_64M"
    env = clean_environment()
    env.update({
        "NCCL_TUNER_PLUGIN": str(plugin),
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "ENV,TUNING",
    })
    command = [
        "nsys", "profile", "--force-overwrite=true", "--sample=none",
        "--trace=cuda,nvtx", f"--output={stem}", str(binary),
        "-b", "64M", "-e", "64M", "-g", "4", "-w", "2", "-n", "5",
        "-N", "1", "-c", "1", "-I", "0", "-z", "0", "-u", "0",
        "-C", "0", "-a", "3", "-d", "float", "-o", "sum",
    ]
    text = run_logged(
        command, env, run_dir / "raw/plugin/nsys_profile.log",
        root / "third_party/nccl-tests"
    )
    if "Using tuner plugin ch16-force-tree-simple" not in text:
        raise RuntimeError("plugin load was not logged")
    selections = parse_selections(text, {TARGET_BYTES})
    if selections[TARGET_BYTES][:2] != (0, 2):
        raise RuntimeError(f"plugin selection mismatch: {selections}")
    report = stem.with_suffix(".nsys-rep")
    sqlite_path = stem.with_suffix(".sqlite")
    export_command = [
        "nsys", "export", "--force-overwrite=true", "--type=sqlite",
        f"--output={sqlite_path}", str(report),
    ]
    run_logged(
        export_command, clean_environment(),
        run_dir / "raw/plugin/nsys_export.log", root
    )
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
    if not kernel_rows:
        raise RuntimeError("no AllReduce kernels in plugin profile")
    grids = sorted({int(row[2]) for row in kernel_rows})
    blocks = sorted({int(row[3]) for row in kernel_rows})
    if grids != [2]:
        raise RuntimeError(f"plugin requested channels=2 but gridX={grids}")
    durations = [int(row[1]) / 1000.0 for row in kernel_rows]
    return {
        "message_bytes": TARGET_BYTES,
        "selected_algorithm": "Tree",
        "selected_algorithm_id": 0,
        "selected_protocol": "Simple",
        "selected_protocol_id": 2,
        "requested_channels": 2,
        "kernel_names": ";".join(sorted({str(row[0]) for row in kernel_rows})),
        "kernel_instances": len(kernel_rows),
        "grid_x_values": ",".join(map(str, grids)),
        "block_x_values": ",".join(map(str, blocks)),
        "median_kernel_us": statistics.median(durations),
        "p95_kernel_us": percentile(durations, .95),
        "private_profile_retained": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/nccl-learning"))
    parser.add_argument(
        "--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--plugin-cycles", type=int, default=10)
    parser.add_argument("--skip-nsys", action="store_true")
    args = parser.parse_args()

    root = args.root
    run_dir = root / "logs/ch16_tuning" / args.run_id
    private_dir = root / "private/ch16_tuning" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    binary = root / "third_party/nccl-tests/build/all_reduce_perf"
    tests_cwd = root / "third_party/nccl-tests"
    commands = []

    auto_rows = []
    auto_command = perf_command(binary, "4", "1G", args.cycles, args.iterations)
    for replicate in (1, 2):
        env = clean_environment()
        env["NCCL_DEBUG"] = "WARN"
        commands.append("NCCL_DEBUG=WARN " + " ".join(auto_command))
        output = run_logged(
            auto_command, env,
            run_dir / f"raw/auto/performance_r{replicate}.log", tests_cwd
        )
        auto_rows.extend(parse_perf(
            output, "auto", replicate, SIZE_COUNT, args.cycles
        ))
        print(f"[auto performance] replicate={replicate} PASS", flush=True)

    info_command = perf_command(binary, "4", "1G", 1, 1)
    info_env = clean_environment()
    info_env.update({"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "ENV,TUNING"})
    commands.append("NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=ENV,TUNING " + " ".join(info_command))
    info_text = run_logged(
        info_command, info_env, run_dir / "raw/auto/model_selection.log", tests_cwd
    )
    if "using internal tuner instead" not in info_text:
        raise RuntimeError("internal tuner fallback was not observed")
    sizes = {4 << exponent for exponent in range(SIZE_COUNT)}
    selections = parse_selections(info_text, sizes)
    model_table = parse_model_table(info_text)
    model_candidates = recompute_model(model_table, selections)
    selection_comparison = compare_to_measurement(root, model_candidates)
    print("[model] 29 selections reproduced from source formula PASS", flush=True)

    plugin = compile_plugin(root, run_dir, private_dir)
    plugin_rows = []
    target_command = perf_command(
        binary, "64M", "64M", args.plugin_cycles, args.iterations
    )
    for experiment, use_plugin in (("auto_64M", False), ("plugin_tree_simple_ch2", True)):
        env = clean_environment()
        env["NCCL_DEBUG"] = "WARN"
        if use_plugin:
            env["NCCL_TUNER_PLUGIN"] = str(plugin)
        commands.append(
            (f"NCCL_TUNER_PLUGIN={plugin} " if use_plugin else "")
            + "NCCL_DEBUG=WARN " + " ".join(target_command)
        )
        output = run_logged(
            target_command, env, run_dir / f"raw/plugin/{experiment}.log", tests_cwd
        )
        plugin_rows.extend(parse_perf(
            output, experiment, 1, 1, args.plugin_cycles
        ))
        print(f"[plugin performance] {experiment} PASS", flush=True)

    geometry = None
    if not args.skip_nsys:
        geometry = profile_plugin(root, run_dir, private_dir, binary, plugin)
        print(
            f"[plugin nsys] algo=Tree proto=Simple gridX={geometry['grid_x_values']} PASS",
            flush=True,
        )

    auto_summary = summarize_perf(auto_rows)
    plugin_summary = summarize_perf(plugin_rows)
    baseline = next(row for row in plugin_summary if row["experiment"] == "auto_64M")
    overridden = next(
        row for row in plugin_summary if row["experiment"] == "plugin_tree_simple_ch2"
    )
    slowdown = float(overridden["median_time_us"]) / float(baseline["median_time_us"])

    write_csv(run_dir / "auto_raw_measurements.csv", auto_rows)
    write_csv(run_dir / "auto_summary.csv", auto_summary)
    write_csv(run_dir / "model_table.csv", model_table)
    write_csv(run_dir / "model_candidates.csv", model_candidates)
    write_csv(run_dir / "selection_vs_measured.csv", selection_comparison)
    write_csv(run_dir / "plugin_raw_measurements.csv", plugin_rows)
    write_csv(run_dir / "plugin_summary.csv", plugin_summary)
    if geometry:
        write_csv(run_dir / "plugin_kernel_geometry.csv", [geometry])

    exact_hits = sum(int(row["exact_selection_hit"]) for row in selection_comparison)
    practical_hits = sum(
        int(row["practical_selection_hit_5pct"]) for row in selection_comparison
    )
    worst = max(selection_comparison, key=lambda row: float(row["selection_regret_percent"]))
    source = root / "probes/ch16_force_tree_simple_tuner.cc"
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    nccl_dir = root / "third_party/nccl-2.22.3"
    manifest = [
        "experiment=ch16_tuning_model_plugin",
        f"run_id={args.run_id}",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"hostname={os.uname().nodename}",
        "world_size=4",
        "collective=AllReduce",
        "datatype=float",
        f"auto_cycles_per_replicate={args.cycles}",
        "auto_replicates=2",
        f"iterations={args.iterations}",
        f"auto_measurement_rows={len(auto_rows)}",
        f"plugin_cycles={args.plugin_cycles}",
        f"plugin_measurement_rows={len(plugin_rows)}",
        f"plugin_source_sha256={source_sha}",
        f"private_artifacts={private_dir}",
        f"nccl_tests_commit={subprocess.check_output(['git', '-C', str(tests_cwd), 'rev-parse', 'HEAD'], text=True).strip()}",
        f"nccl_source_commit={subprocess.check_output(['git', '-C', str(nccl_dir), 'rev-parse', 'HEAD'], text=True).strip()}",
    ]
    manifest.extend(
        f"command_{index:02d}={command}" for index, command in enumerate(commands, 1)
    )
    (run_dir / "manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    lines = [
        "# Chapter 16 experiment summary", "",
        f"- Auto measurements: {len(auto_rows)} rows, out-of-place/in-place correctness PASS",
        "- Rounded INFO model table reproduces all 29 logged auto selections: PASS",
        f"- Exact selection hits against Chapter 14 forced sweep: {exact_hits}/29",
        f"- Practical hits (regret < 5%): {practical_hits}/29",
        f"- Worst regret: {worst['size_bytes']} B, {float(worst['selection_regret_percent']):.2f}%",
        f"- Tuner plugin override: Tree/Simple, channels=2, median slowdown {slowdown:.2f}x",
        f"- Nsight plugin geometry: {'gridX=' + str(geometry['grid_x_values']) if geometry else 'skipped'}",
        "", "## Model table", "",
        "| algorithm | protocol | latency us | algorithm BW GB/s |",
        "|---|---|---:|---:|",
    ]
    for row in model_table:
        lines.append(
            f"| {row['algorithm']} | {row['protocol']} | {float(row['base_latency_us']):.1f} | "
            f"{float(row['algorithm_bandwidth_GBs']):.1f} |"
        )
    lines.extend([
        "", "## Auto-selection regret", "",
        "| bytes | auto | measured best | regret | model abs error |",
        "|---:|---|---|---:|---:|",
    ])
    for row in selection_comparison:
        lines.append(
            f"| {row['size_bytes']} | {row['auto_algorithm']}+{row['auto_protocol']} | "
            f"{row['measured_best_algorithm']}+{row['measured_best_protocol']} | "
            f"{float(row['selection_regret_percent']):.2f}% | "
            f"{float(row['selected_model_abs_error_percent']):.2f}% |"
        )
    lines.extend([
        "", "## Plugin override", "",
        "| policy | median us | P95 us | CV | busbw GB/s |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in plugin_summary:
        lines.append(
            f"| {row['experiment']} | {float(row['median_time_us']):.3f} | "
            f"{float(row['p95_time_us']):.3f} | {float(row['cycle_cv_percent']):.2f}% | "
            f"{float(row['median_busbw_GBs']):.2f} |"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
