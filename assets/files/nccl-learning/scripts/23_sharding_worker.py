#!/usr/bin/env python3
import argparse
import json
import math
import os
import statistics
from datetime import timedelta
from functools import partial

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
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
    parser = argparse.ArgumentParser(description="Compare replicated DDP and fully sharded FSDP")
    parser.add_argument("--mode", choices=["ddp", "fsdp"], required=True)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=240))
    rank = dist.get_rank()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(20260710)
    model = DeepMLP(args.width, args.layers).to(device)
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    if args.mode == "ddp":
        wrapped = DDP(model, device_ids=[local_rank], bucket_cap_mb=64)
    else:
        auto_wrap = partial(size_based_auto_wrap_policy, min_num_params=args.width * args.width)
        wrapped = FSDP(
            model,
            auto_wrap_policy=auto_wrap,
            device_id=device,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
        )
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-4)
    generator = torch.Generator(device=device).manual_seed(20260710 + rank)
    step_times = []

    for step in range(args.warmup + args.steps):
        x = torch.randn(args.batch, args.width, generator=generator, device=device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        optimizer.zero_grad(set_to_none=True)
        output = wrapped(x)
        loss = output.square().mean()
        loss.backward()
        optimizer.step()
        end.record()
        end.synchronize()
        if step >= args.warmup:
            step_times.append(start.elapsed_time(end))

    if args.mode == "fsdp":
        with FSDP.summon_full_params(wrapped, recurse=True, writeback=False):
            first_value = next(wrapped.parameters()).detach().flatten()[0].clone()
    else:
        first_value = next(wrapped.parameters()).detach().flatten()[0].clone()
    min_value = first_value.clone()
    max_value = first_value.clone()
    dist.all_reduce(min_value, op=dist.ReduceOp.MIN)
    dist.all_reduce(max_value, op=dist.ReduceOp.MAX)
    parameters_equal = bool((max_value - min_value).abs().item() < 1e-7)

    metrics = torch.tensor(
        [torch.cuda.memory_allocated(device), torch.cuda.max_memory_allocated(device)],
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    if rank == 0:
        result = {
            "mode": args.mode,
            "parameter_bytes": parameter_bytes,
            "median_step_ms": statistics.median(step_times),
            "p95_step_ms": percentile(step_times, 0.95),
            "min_step_ms": min(step_times),
            "max_step_ms": max(step_times),
            "allocated_bytes": int(metrics[0].item()),
            "peak_allocated_bytes": int(metrics[1].item()),
            "parameters_equal": parameters_equal,
        }
        print("RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)

    if not parameters_equal:
        raise RuntimeError(f"{args.mode} parameters are not equal across ranks")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
