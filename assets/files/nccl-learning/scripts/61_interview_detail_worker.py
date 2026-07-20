#!/usr/bin/env python3
"""Run focused NCCL interview experiments under torchrun.

Usage:
  torchrun --standalone --nproc_per_node=4 61_interview_detail_worker.py \
    --output-dir /tmp/results \
    --sections async,decomposition,fragmentation,decode,numerics
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def all_equal(tensor: torch.Tensor, expected: float) -> bool:
    return bool(torch.all(tensor == expected).item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sections",
        default="async,decomposition,fragmentation,decode,numerics",
        help="Comma-separated subset of async,decomposition,fragmentation,decode,numerics",
    )
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--decode-collectives", type=int, default=160)
    parser.add_argument(
        "--cpu-affinity",
        default="0,2,4,6",
        help="Comma-separated CPU list; local rank i is pinned to entry i",
    )
    args = parser.parse_args()

    sections = {item.strip() for item in args.sections.split(",") if item.strip()}
    allowed = {"async", "decomposition", "fragmentation", "decode", "numerics"}
    unknown = sections - allowed
    if unknown:
        raise ValueError(f"unknown sections: {sorted(unknown)}")

    local_rank = int(os.environ["LOCAL_RANK"])
    affinity = [int(cpu) for cpu in args.cpu_affinity.split(",") if cpu]
    if len(affinity) <= local_rank:
        raise ValueError("cpu-affinity must contain one CPU for every local rank")
    os.sched_setaffinity(0, {affinity[local_rank]})
    torch.set_num_threads(1)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    rank = dist.get_rank()
    world = dist.get_world_size()
    expected_sum = world * (world + 1) / 2

    async_rows: list[dict[str, object]] = []
    decomposition_rows: list[dict[str, object]] = []
    fragmentation_rows: list[dict[str, object]] = []
    decode_rows: list[dict[str, object]] = []
    numeric_rows: list[dict[str, object]] = []
    contract_rows: list[dict[str, object]] = []

    def contract(section: str, name: str, correct: bool, detail: str) -> None:
        contract_rows.append(
            {
                "rank": rank,
                "world_size": world,
                "section": section,
                "name": name,
                "correct": correct,
                "detail": detail,
            }
        )
        if not correct:
            raise AssertionError(f"{section}/{name}: {detail}")

    def sync_ranks() -> None:
        dist.barrier()
        torch.cuda.synchronize(device)

    def event_time_us(operation: Callable[[], None]) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000.0

    if "async" in sections:
        for size_bytes in (4096, 1 << 20, 64 << 20):
            elements = size_bytes // 4
            tensor = torch.empty(elements, dtype=torch.float32, device=device)
            for _ in range(args.warmups):
                tensor.fill_(rank + 1)
                work = dist.all_reduce(tensor, async_op=True)
                work.wait()
            torch.cuda.synchronize(device)

            for cycle in range(args.cycles):
                tensor.fill_(rank + 1)
                sync_ranks()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                host_start_ns = time.perf_counter_ns()
                work = dist.all_reduce(tensor, async_op=True)
                host_return_ns = time.perf_counter_ns()
                completed_after_return = work.is_completed()
                work.wait()
                host_wait_return_ns = time.perf_counter_ns()
                end_event.record()
                event_ready_after_wait = end_event.query()
                end_event.synchronize()
                host_complete_ns = time.perf_counter_ns()
                correct = all_equal(tensor, expected_sum)
                async_rows.append(
                    {
                        "rank": rank,
                        "world_size": world,
                        "size_bytes": size_bytes,
                        "cycle": cycle,
                        "host_enqueue_us": (host_return_ns - host_start_ns) / 1000.0,
                        "host_wait_us": (host_wait_return_ns - host_return_ns) / 1000.0,
                        "host_to_gpu_complete_us": (host_complete_ns - host_start_ns) / 1000.0,
                        "gpu_dependency_us": start_event.elapsed_time(end_event) * 1000.0,
                        "work_completed_after_return": completed_after_return,
                        "event_ready_after_wait": event_ready_after_wait,
                        "correct": correct,
                    }
                )
                contract("async", f"size-{size_bytes}-cycle-{cycle}", correct, "sum result")
            del tensor

    if "decomposition" in sections:
        if world not in (2, 4):
            raise RuntimeError("decomposition suite expects world size 2 or 4")
        for size_bytes in (4096, 512 << 10, 64 << 20):
            if size_bytes % (world * 4) != 0:
                raise ValueError("size must divide by world_size*sizeof(float32)")
            elements = size_bytes // 4
            direct_input = torch.empty(elements, dtype=torch.float32, device=device)
            decomp_input = torch.empty_like(direct_input)
            shard = torch.empty(elements // world, dtype=torch.float32, device=device)
            gathered = torch.empty_like(direct_input)

            def direct() -> None:
                dist.all_reduce(direct_input)

            def decomposed() -> None:
                dist.reduce_scatter_tensor(shard, decomp_input)
                dist.all_gather_into_tensor(gathered, shard)

            for _ in range(args.warmups):
                direct_input.fill_(rank + 1)
                direct()
                decomp_input.fill_(rank + 1)
                decomposed()
            torch.cuda.synchronize(device)

            for cycle in range(args.cycles):
                order = ("direct", "rs_ag") if cycle % 2 == 0 else ("rs_ag", "direct")
                for mode in order:
                    if mode == "direct":
                        direct_input.fill_(rank + 1)
                        operation = direct
                    else:
                        decomp_input.fill_(rank + 1)
                        operation = decomposed
                    sync_ranks()
                    duration_us = event_time_us(operation)
                    output = direct_input if mode == "direct" else gathered
                    correct = all_equal(output, expected_sum)
                    decomposition_rows.append(
                        {
                            "rank": rank,
                            "world_size": world,
                            "size_bytes": size_bytes,
                            "mode": mode,
                            "cycle": cycle,
                            "duration_us": duration_us,
                            "correct": correct,
                        }
                    )
                    contract(
                        "decomposition",
                        f"size-{size_bytes}-{mode}-cycle-{cycle}",
                        correct,
                        "direct and RS+AG must produce the same sum",
                    )
            contract(
                "decomposition",
                f"size-{size_bytes}-equivalence",
                bool(torch.equal(direct_input, gathered)),
                "AllReduce output equals ReduceScatter plus AllGather output",
            )
            del direct_input, decomp_input, shard, gathered

    if "fragmentation" in sections:
        total_bytes = 80 << 20
        fragment_counts = (1, 10, 40, 160)
        buffers = {
            count: torch.zeros(total_bytes // count // 2, dtype=torch.float16, device=device)
            for count in fragment_counts
        }
        for count, tensor in buffers.items():
            for _ in range(count):
                dist.all_reduce(tensor)
        torch.cuda.synchronize(device)

        for cycle in range(args.cycles):
            order = fragment_counts if cycle % 2 == 0 else tuple(reversed(fragment_counts))
            for count in order:
                tensor = buffers[count]
                sync_ranks()

                def fragmented() -> None:
                    for _ in range(count):
                        dist.all_reduce(tensor)

                duration_us = event_time_us(fragmented)
                correct = all_equal(tensor, 0.0)
                fragmentation_rows.append(
                    {
                        "rank": rank,
                        "world_size": world,
                        "total_bytes": total_bytes,
                        "fragment_count": count,
                        "fragment_bytes": total_bytes // count,
                        "cycle": cycle,
                        "duration_us": duration_us,
                        "per_collective_us": duration_us / count,
                        "logical_payload_GBs": total_bytes / duration_us / 1000.0,
                        "correct": correct,
                    }
                )
                contract(
                    "fragmentation",
                    f"count-{count}-cycle-{cycle}",
                    correct,
                    "zero is invariant under SUM",
                )
        del buffers

    if "decode" in sections:
        hidden_size = 4096
        collectives = args.decode_collectives
        token_counts = (1, 8, 32, 128)
        buffers = {
            tokens: torch.zeros(tokens * hidden_size, dtype=torch.float16, device=device)
            for tokens in token_counts
        }
        for tensor in buffers.values():
            for _ in range(collectives):
                dist.all_reduce(tensor)
        torch.cuda.synchronize(device)

        for cycle in range(args.cycles):
            order = token_counts if cycle % 2 == 0 else tuple(reversed(token_counts))
            for tokens in order:
                tensor = buffers[tokens]
                size_bytes = tensor.numel() * tensor.element_size()
                sync_ranks()

                def replay() -> None:
                    for _ in range(collectives):
                        dist.all_reduce(tensor)

                duration_us = event_time_us(replay)
                correct = all_equal(tensor, 0.0)
                decode_rows.append(
                    {
                        "rank": rank,
                        "world_size": world,
                        "tokens": tokens,
                        "hidden_size": hidden_size,
                        "dtype": "float16",
                        "message_bytes": size_bytes,
                        "collectives": collectives,
                        "cycle": cycle,
                        "duration_us": duration_us,
                        "per_collective_us": duration_us / collectives,
                        "correct": correct,
                    }
                )
                contract(
                    "decode",
                    f"tokens-{tokens}-cycle-{cycle}",
                    correct,
                    "zero is invariant under SUM",
                )
        del buffers

    if "numerics" in sections:
        if world != 4:
            raise RuntimeError("numerics suite requires exactly four ranks")
        scenarios = {
            "lost-unit-2048": (2048.0, 1.0, -2048.0, 0.0),
            "lost-unit-10000": (10000.0, 1.0, -10000.0, 0.0),
            "finite-overflow": (65504.0, 65504.0, -65504.0, -65504.0),
        }
        local_fp16: list[float] = []
        local_fp32: list[float] = []
        metadata: list[tuple[str, tuple[float, ...], float]] = []
        for scenario, values in scenarios.items():
            for permutation in itertools.permutations(values):
                local_fp16.append(permutation[rank])
                local_fp32.append(permutation[rank])
                metadata.append((scenario, permutation, float(math.fsum(permutation))))

        reduced_fp16 = torch.tensor(local_fp16, dtype=torch.float16, device=device)
        reduced_fp32 = torch.tensor(local_fp32, dtype=torch.float32, device=device)
        dist.all_reduce(reduced_fp16)
        dist.all_reduce(reduced_fp32)
        gathered_fp16 = torch.empty(world * reduced_fp16.numel(), dtype=torch.float16, device=device)
        gathered_fp32 = torch.empty(world * reduced_fp32.numel(), dtype=torch.float32, device=device)
        dist.all_gather_into_tensor(gathered_fp16, reduced_fp16)
        dist.all_gather_into_tensor(gathered_fp32, reduced_fp32)
        torch.cuda.synchronize(device)
        fp16_replicas = gathered_fp16.reshape(world, -1)
        fp32_replicas = gathered_fp32.reshape(world, -1)
        fp16_equal = (fp16_replicas == fp16_replicas[0]) | (
            torch.isnan(fp16_replicas) & torch.isnan(fp16_replicas[0])
        )
        contract(
            "numerics",
            "fp16-replicas-equal",
            bool(torch.all(fp16_equal).item()),
            "all ranks must receive the same FP16 result",
        )
        contract(
            "numerics",
            "fp32-replicas-equal",
            bool(torch.all(fp32_replicas == fp32_replicas[0]).item()),
            "all ranks must receive the same FP32 result",
        )
        fp16_cpu = reduced_fp16.cpu().tolist()
        fp32_cpu = reduced_fp32.cpu().tolist()
        fp32_matches = True
        for case_id, ((scenario, permutation, expected), value16, value32) in enumerate(
            zip(metadata, fp16_cpu, fp32_cpu)
        ):
            value16_float = float(value16)
            value32_float = float(value32)
            fp16_finite = math.isfinite(value16_float)
            fp16_error = abs(value16_float - expected) if fp16_finite else math.inf
            fp32_error = abs(value32_float - expected)
            fp32_matches = fp32_matches and fp32_error == 0.0
            numeric_rows.append(
                {
                    "rank": rank,
                    "world_size": world,
                    "case_id": case_id,
                    "scenario": scenario,
                    "rank_values": "/".join(f"{value:g}" for value in permutation),
                    "expected_real_sum": expected,
                    "fp16_result": value16_float,
                    "fp32_result": value32_float,
                    "fp16_abs_error": fp16_error,
                    "fp32_abs_error": fp32_error,
                    "fp16_matches_real_sum": fp16_error == 0.0,
                    "fp32_matches_real_sum": fp32_error == 0.0,
                }
            )
        contract(
            "numerics",
            "fp32-matches-reference",
            fp32_matches,
            "FP32 reduction matches math.fsum for all constructed cases",
        )
        del reduced_fp16, reduced_fp32, gathered_fp16, gathered_fp32

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("async", async_rows),
        ("decomposition", decomposition_rows),
        ("fragmentation", fragmentation_rows),
        ("decode", decode_rows),
        ("numerics", numeric_rows),
        ("contracts", contract_rows),
    ):
        write_csv(args.output_dir / f"rank{rank}_{name}.csv", rows)

    sync_ranks()
    if rank == 0:
        print(
            f"world={world} sections={','.join(sorted(sections))} "
            f"async={len(async_rows)} decomposition={len(decomposition_rows)} "
            f"fragmentation={len(fragmentation_rows)} decode={len(decode_rows)} "
            f"numerics={len(numeric_rows)} contracts={len(contract_rows)}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
