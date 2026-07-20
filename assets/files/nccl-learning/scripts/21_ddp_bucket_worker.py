#!/usr/bin/env python3
import argparse
import json
import math
import os
import statistics
from datetime import timedelta

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class DeepMLP(torch.nn.Module):
    def __init__(self, width: int, layers: int) -> None:
        super().__init__()
        modules = []
        for _ in range(layers):
            modules.append(torch.nn.Linear(width, width, bias=False))
            modules.append(torch.nn.ReLU())
        self.layers = torch.nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(q * len(ordered)) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure DDP bucket behavior")
    parser.add_argument("--bucket-cap-mb", type=int, required=True)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=15)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    rank = dist.get_rank()

    torch.manual_seed(20260710)
    model = DeepMLP(args.width, args.layers).to(device)
    ddp = DDP(model, device_ids=[local_rank], bucket_cap_mb=args.bucket_cap_mb)
    optimizer = torch.optim.SGD(ddp.parameters(), lr=1e-4)
    generator = torch.Generator(device=device).manual_seed(20260710 + rank)
    torch.cuda.reset_peak_memory_stats(device)
    step_times = []

    for step in range(args.warmup + args.steps):
        x = torch.randn(args.batch, args.width, generator=generator, device=device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        optimizer.zero_grad(set_to_none=True)
        output = ddp(x)
        loss = output.square().mean()
        loss.backward()
        optimizer.step()
        end.record()
        end.synchronize()
        if step >= args.warmup:
            step_times.append(start.elapsed_time(end))

    first_value = next(ddp.parameters()).detach().flatten()[0].clone()
    min_value = first_value.clone()
    max_value = first_value.clone()
    dist.all_reduce(min_value, op=dist.ReduceOp.MIN)
    dist.all_reduce(max_value, op=dist.ReduceOp.MAX)
    parameters_equal = bool((max_value - min_value).abs().item() < 1e-7)

    peak_memory = torch.tensor([torch.cuda.max_memory_allocated(device)], dtype=torch.int64, device=device)
    dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
    logging_data = ddp._get_ddp_logging_data()

    if rank == 0:
        result = {
            "bucket_cap_mb": args.bucket_cap_mb,
            "parameter_bytes": sum(p.numel() * p.element_size() for p in ddp.parameters()),
            "single_weight_bytes": args.width * args.width * 4,
            "median_step_ms": statistics.median(step_times),
            "p95_step_ms": percentile(step_times, 0.95),
            "min_step_ms": min(step_times),
            "max_step_ms": max(step_times),
            "peak_memory_bytes": int(peak_memory.item()),
            "parameters_equal": parameters_equal,
            "initial_bucket_sizes": str(logging_data.get("bucket_sizes", "unknown")),
            "rebuilt_bucket_sizes": str(logging_data.get("rebuilt_bucket_sizes", "unknown")),
            "bucket_cap_bytes": int(logging_data.get("bucket_cap_bytes", 0)),
            "avg_backward_compute_time_ns": int(logging_data.get("avg_backward_compute_time", 0)),
            "avg_backward_comm_time_ns": int(logging_data.get("avg_backward_comm_time", 0)),
            "avg_backward_overlap_time_ns": int(
                logging_data.get("avg_backward_compute_comm_overlap_time", 0)
            ),
            "has_rebuilt_buckets": int(logging_data.get("has_rebuilt_buckets", 0)),
        }
        print("RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)

    if not parameters_equal:
        raise RuntimeError("DDP parameter replicas diverged")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
