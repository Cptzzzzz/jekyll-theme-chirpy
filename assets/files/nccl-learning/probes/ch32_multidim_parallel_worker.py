#!/usr/bin/env python3
"""Train small TP, sequence-parallel, pipeline, and expert-parallel models."""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F


def make_groups(rank: int):
    # Every rank creates every group in the same global order.
    specs = (("tp0", (0, 1)), ("tp1", (2, 3)), ("dp0", (0, 2)), ("dp1", (1, 3)))
    groups = {name: dist.new_group(list(ranks)) for name, ranks in specs}
    tp_name = "tp0" if rank < 2 else "tp1"
    dp_name = "dp0" if rank % 2 == 0 else "dp1"
    return groups, tp_name, dp_name, rank % 2, rank // 2


def deterministic_weight(rows: int, cols: int, seed: int, device: torch.device):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(rows, cols, generator=gen, dtype=torch.float32).mul_(0.02).to(device)


def sync_grads(parameters, group):
    for p in parameters:
        dist.all_reduce(p.grad, group=group)
        p.grad.div_(2)


def replicas_equal(parameters, group):
    for p in parameters:
        copies = [torch.empty_like(p) for _ in range(2)]
        dist.all_gather(copies, p.detach(), group=group)
        if not torch.allclose(copies[0], copies[1], rtol=1e-5, atol=1e-6):
            return False
    return True


def append_op(rows, rank, config, iteration, phase, op, group_name, payload, note):
    rows.append({"rank": rank, "config": config, "iteration": iteration, "phase": phase,
                 "operation": op, "group": group_name, "payload_bytes": payload, "note": note})


class TensorParallelMLP:
    def __init__(self, d_model, hidden, tp_rank, device):
        assert hidden % 2 == 0
        w1 = deterministic_weight(hidden, d_model, 32001, device).chunk(2, dim=0)[tp_rank].contiguous()
        w2 = deterministic_weight(d_model, hidden, 32002, device).chunk(2, dim=1)[tp_rank].contiguous()
        self.w1 = torch.nn.Parameter(w1)
        self.w2 = torch.nn.Parameter(w2)

    def parameters(self):
        return (self.w1, self.w2)

    def partial(self, x):
        return F.linear(F.gelu(F.linear(x, self.w1)), self.w2)


def tp_forward(model, x, tp_group, mode):
    partial = model.partial(x)
    if mode == "tp_ar":
        return dist_nn.all_reduce(partial, group=tp_group)
    chunks = [x.contiguous() for x in partial.chunk(2, dim=0)]
    shard = torch.empty_like(chunks[0])
    shard = dist_nn.reduce_scatter(shard, chunks, group=tp_group)
    return torch.cat(dist_nn.all_gather(shard, group=tp_group), dim=0)


def run_tp(a, dev, rank, groups, tp_name, dp_name, tp_rank, dp_rank, rows, timeline):
    tp_group, dp_group = groups[tp_name], groups[dp_name]
    model = TensorParallelMLP(a.d_model, a.hidden, tp_rank, dev)
    opt = torch.optim.SGD(model.parameters(), lr=2e-3)
    gen = torch.Generator(device=dev).manual_seed(32100 + dp_rank)
    activation_bytes = a.batch * a.d_model * 4
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    check_gen = torch.Generator(device=dev).manual_seed(32900 + dp_rank)
    check_x = torch.randn(a.batch, a.d_model, generator=check_gen, device=dev)
    check_out = tp_forward(model, check_x, tp_group, a.mode)
    full_w1 = deterministic_weight(a.hidden, a.d_model, 32001, dev)
    full_w2 = deterministic_weight(a.d_model, a.hidden, 32002, dev)
    reference = F.linear(F.gelu(F.linear(check_x, full_w1)), full_w2)
    numeric_max_abs = float((check_out-reference).abs().max())
    if numeric_max_abs > 2e-5: raise RuntimeError(f"TP numeric mismatch {numeric_max_abs}")

    def one(iteration, measured):
        x = torch.randn(a.batch, a.d_model, generator=gen, device=dev)
        opt.zero_grad(set_to_none=True)
        out = tp_forward(model, x, tp_group, a.mode)
        loss = out.square().mean()
        loss.backward()
        sync_grads(model.parameters(), dp_group)
        opt.step()
        if measured:
            if a.mode == "tp_ar":
                append_op(rows, rank, a.config, iteration, "forward", "AllReduce", tp_name, activation_bytes, "row-parallel partial output")
                append_op(rows, rank, a.config, iteration, "backward", "AllReduce", tp_name, activation_bytes, "autograd dual of AllReduce")
            else:
                shard_bytes = activation_bytes // 2
                append_op(rows, rank, a.config, iteration, "forward", "ReduceScatter", tp_name, activation_bytes, "partial output to sequence shard")
                append_op(rows, rank, a.config, iteration, "forward", "AllGather", tp_name, shard_bytes, "sequence shard to replicated loss input")
                append_op(rows, rank, a.config, iteration, "backward", "ReduceScatter", tp_name, activation_bytes, "AllGather backward")
                append_op(rows, rank, a.config, iteration, "backward", "AllGather", tp_name, shard_bytes, "ReduceScatter backward")
            for p, layer in zip(model.parameters(), ("w1", "w2")):
                append_op(rows, rank, a.config, iteration, "dp_gradient", "AllReduce", dp_name,
                          p.numel() * p.element_size(), f"{layer} shard replica gradient")
        return loss

    for i in range(a.warmup):
        one(-i - 1, False)
    measurements = []
    for i in range(a.cycles):
        if a.profile_capture and i == 0:
            torch.cuda.cudart().cudaProfilerStart()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        host0 = time.perf_counter_ns()
        start.record()
        loss = one(i, True)
        end.record(); end.synchronize()
        host1 = time.perf_counter_ns()
        if a.profile_capture and i == 0:
            torch.cuda.cudart().cudaProfilerStop()
        measurements.append({"rank": rank, "config": a.config, "mode": a.mode, "iteration": i,
                             "step_gpu_us": start.elapsed_time(end) * 1000, "step_host_us": (host1-host0)/1000,
                             "loss": float(loss), "correct": 1})
    equal = True if a.profile_capture else replicas_equal(model.parameters(), dp_group)
    return measurements, equal, {
        "activation_bytes": activation_bytes, "local_parameter_bytes": parameter_bytes,
        "tp_group": tp_name, "dp_group": dp_name, "numeric_max_abs": numeric_max_abs,
    }


def run_ep(a, dev, rank, groups, ep_name, dp_name, ep_rank, dp_rank, rows, timeline):
    ep_group, dp_group = groups[ep_name], groups[dp_name]
    weight = torch.nn.Parameter(deterministic_weight(a.d_model, a.d_model, 32200 + ep_rank, dev))
    opt = torch.optim.SGD((weight,), lr=1e-3)
    gen = torch.Generator(device=dev).manual_seed(32300 + dp_rank * 10 + ep_rank)
    local_tokens = a.batch // 2
    first = local_tokens // 2 if a.mode == "ep_balanced" else local_tokens * 7 // 8
    input_splits = [first, local_tokens - first]
    output_splits = [first if ep_rank == 0 else local_tokens-first] * 2
    token_bytes = a.d_model * 4

    def forward(x):
        packed = torch.cat((x[:first], x[first:]), dim=0).contiguous()
        received = torch.empty(sum(output_splits), a.d_model, device=dev)
        received = dist_nn.all_to_all_single(received, packed, output_splits, input_splits, group=ep_group)
        expert_out = F.gelu(F.linear(received, weight))
        returned = torch.empty_like(packed)
        returned = dist_nn.all_to_all_single(returned, expert_out, input_splits, output_splits, group=ep_group)
        return returned

    check_gen = torch.Generator(device=dev).manual_seed(32910 + dp_rank * 10 + ep_rank)
    check_x = torch.randn(local_tokens, a.d_model, generator=check_gen, device=dev)
    check_out = forward(check_x)
    expert_weights = [torch.empty_like(weight) for _ in range(2)]
    dist.all_gather(expert_weights, weight.detach(), group=ep_group)
    reference = torch.cat((F.gelu(F.linear(check_x[:first], expert_weights[0])),
                           F.gelu(F.linear(check_x[first:], expert_weights[1]))), dim=0)
    numeric_max_abs = float((check_out-reference).abs().max())
    if numeric_max_abs > 2e-5: raise RuntimeError(f"EP numeric mismatch {numeric_max_abs}")

    def one(iteration, measured):
        # Treat tokens as an activation from a trainable upstream layer so the
        # dispatch AllToAll must also exchange input gradients in backward.
        x = torch.randn(local_tokens, a.d_model, generator=gen, device=dev, requires_grad=True)
        opt.zero_grad(set_to_none=True)
        out = forward(x)
        loss = out.square().mean()
        loss.backward()
        if x.grad is None or not torch.isfinite(x.grad).all():
            raise RuntimeError("EP upstream activation gradient missing")
        sync_grads((weight,), dp_group)
        opt.step()
        if measured:
            sent = local_tokens * token_bytes
            received = sum(output_splits) * token_bytes
            append_op(rows, rank, a.config, iteration, "dispatch", "AllToAll", ep_name, sent,
                      f"send_splits={input_splits};recv_splits={output_splits}")
            append_op(rows, rank, a.config, iteration, "combine", "AllToAll", ep_name, received,
                      f"send_splits={output_splits};recv_splits={input_splits}")
            append_op(rows, rank, a.config, iteration, "backward_combine", "AllToAll", ep_name, sent, "autograd reverse exchange")
            append_op(rows, rank, a.config, iteration, "backward_dispatch", "AllToAll", ep_name, received, "autograd reverse exchange")
            append_op(rows, rank, a.config, iteration, "dp_gradient", "AllReduce", dp_name,
                      weight.numel() * weight.element_size(), "expert replica gradient")
        return loss

    for i in range(a.warmup):
        one(-i - 1, False)
    measurements = []
    for i in range(a.cycles):
        if a.profile_capture and i == 0: torch.cuda.cudart().cudaProfilerStart()
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        host0 = time.perf_counter_ns(); start.record(); loss = one(i, True); end.record(); end.synchronize(); host1 = time.perf_counter_ns()
        if a.profile_capture and i == 0: torch.cuda.cudart().cudaProfilerStop()
        measurements.append({"rank": rank, "config": a.config, "mode": a.mode, "iteration": i,
                             "step_gpu_us": start.elapsed_time(end)*1000, "step_host_us": (host1-host0)/1000,
                             "loss": float(loss), "correct": 1})
    equal = True if a.profile_capture else replicas_equal((weight,), dp_group)
    return measurements, equal, {
        "local_tokens": local_tokens, "input_splits": input_splits, "output_splits": output_splits,
        "expert_received_tokens": sum(output_splits), "ep_group": ep_name, "dp_group": dp_name,
        "numeric_max_abs": numeric_max_abs,
    }


def run_pp(a, dev, rank, groups, dp_name, dp_rank, rows, timeline):
    stage = rank % 2
    peer = rank + 1 if stage == 0 else rank - 1
    dp_group = groups[dp_name]
    micro_batch = a.batch // a.microbatches
    if stage == 0:
        weight = torch.nn.Parameter(deterministic_weight(a.hidden, a.d_model, 32400, dev))
    else:
        weight = torch.nn.Parameter(deterministic_weight(a.d_model, a.hidden, 32401, dev))
    opt = torch.optim.SGD((weight,), lr=1e-3)
    gen = torch.Generator(device=dev).manual_seed(32500 + dp_rank)
    payload = micro_batch * a.hidden * 4
    check_gen = torch.Generator(device=dev).manual_seed(32920 + dp_rank)
    check_x = torch.randn(micro_batch, a.d_model, generator=check_gen, device=dev)
    if stage == 0:
        check_local = F.gelu(F.linear(check_x, weight))
        check_ref = F.gelu(F.linear(check_x, deterministic_weight(a.hidden, a.d_model, 32400, dev)))
    else:
        check_h = torch.randn(micro_batch, a.hidden, generator=check_gen, device=dev)
        check_local = F.linear(check_h, weight)
        check_ref = F.linear(check_h, deterministic_weight(a.d_model, a.hidden, 32401, dev))
    numeric_max_abs = float((check_local-check_ref).abs().max())
    if numeric_max_abs > 2e-5: raise RuntimeError(f"PP stage numeric mismatch {numeric_max_abs}")

    def one(iteration, measured):
        opt.zero_grad(set_to_none=True)
        t0 = time.perf_counter_ns()
        losses = []
        if stage == 0:
            activations = []
            for mb in range(a.microbatches):
                x = torch.randn(micro_batch, a.d_model, generator=gen, device=dev)
                h = F.gelu(F.linear(x, weight))
                dist.send(h.detach(), dst=peer)
                activations.append(h)
                if measured:
                    append_op(rows, rank, a.config, iteration, "forward", "Send", f"pp{dp_rank}", payload, f"microbatch={mb};stage=0->1")
                    timeline.append({"rank": rank, "config": a.config, "iteration": iteration, "microbatch": mb,
                                     "phase": "forward_send", "start_us": (time.perf_counter_ns()-t0)/1000})
            for mb in reversed(range(a.microbatches)):
                grad = torch.empty_like(activations[mb])
                dist.recv(grad, src=peer)
                activations[mb].backward(grad)
                if measured:
                    append_op(rows, rank, a.config, iteration, "backward", "Recv", f"pp{dp_rank}", payload, f"microbatch={mb};activation gradient")
                    timeline.append({"rank": rank, "config": a.config, "iteration": iteration, "microbatch": mb,
                                     "phase": "backward_recv", "start_us": (time.perf_counter_ns()-t0)/1000})
            loss_value = 0.0
        else:
            received = []
            for mb in range(a.microbatches):
                h = torch.empty(micro_batch, a.hidden, device=dev)
                dist.recv(h, src=peer); h.requires_grad_(True)
                target = torch.randn(micro_batch, a.d_model, generator=gen, device=dev)
                loss = F.mse_loss(F.linear(h, weight), target) / a.microbatches
                received.append((h, loss))
                if measured:
                    append_op(rows, rank, a.config, iteration, "forward", "Recv", f"pp{dp_rank}", payload, f"microbatch={mb};stage=0->1")
                    timeline.append({"rank": rank, "config": a.config, "iteration": iteration, "microbatch": mb,
                                     "phase": "forward_recv", "start_us": (time.perf_counter_ns()-t0)/1000})
            for mb in reversed(range(a.microbatches)):
                h, loss = received[mb]
                loss.backward()
                dist.send(h.grad, dst=peer)
                losses.append(float(loss))
                if measured:
                    append_op(rows, rank, a.config, iteration, "backward", "Send", f"pp{dp_rank}", payload, f"microbatch={mb};activation gradient")
                    timeline.append({"rank": rank, "config": a.config, "iteration": iteration, "microbatch": mb,
                                     "phase": "backward_send", "start_us": (time.perf_counter_ns()-t0)/1000})
            loss_value = sum(losses)
        sync_grads((weight,), dp_group)
        opt.step()
        if measured:
            append_op(rows, rank, a.config, iteration, "dp_gradient", "AllReduce", dp_name,
                      weight.numel()*weight.element_size(), f"pipeline stage {stage} replica gradient")
        return loss_value

    for i in range(a.warmup): one(-i - 1, False)
    measurements=[]
    for i in range(a.cycles):
        if a.profile_capture and i == 0: torch.cuda.cudart().cudaProfilerStart()
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        host0=time.perf_counter_ns();start.record();loss=one(i,True);end.record();end.synchronize();host1=time.perf_counter_ns()
        if a.profile_capture and i == 0: torch.cuda.cudart().cudaProfilerStop()
        measurements.append({"rank":rank,"config":a.config,"mode":a.mode,"iteration":i,
                             "step_gpu_us":start.elapsed_time(end)*1000,"step_host_us":(host1-host0)/1000,
                             "loss":loss,"correct":1})
    equal = True if a.profile_capture else replicas_equal((weight,), dp_group)
    return measurements, equal, {
        "stage":stage,"peer":peer,"micro_batch":micro_batch,"microbatches":a.microbatches,
        "activation_bytes_per_microbatch":payload,"dp_group":dp_name,"numeric_max_abs":numeric_max_abs,
    }


def write_csv(path, rows):
    if not rows: return
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,required=True);ap.add_argument("--config",required=True)
    ap.add_argument("--mode",choices=("tp_ar","tp_sp","pp","ep_balanced","ep_skewed"),required=True)
    ap.add_argument("--batch",type=int,default=64);ap.add_argument("--d-model",type=int,default=512);ap.add_argument("--hidden",type=int,default=1024)
    ap.add_argument("--microbatches",type=int,default=4);ap.add_argument("--warmup",type=int,default=3);ap.add_argument("--cycles",type=int,default=8)
    ap.add_argument("--profile-capture",action="store_true");a=ap.parse_args()
    local=int(os.environ["LOCAL_RANK"]);torch.cuda.set_device(local);dev=torch.device("cuda",local)
    dist.init_process_group("nccl",timeout=timedelta(seconds=180));rank=dist.get_rank()
    if dist.get_world_size()!=4:raise RuntimeError("chapter 32 requires world_size=4")
    groups,tp_name,dp_name,tp_rank,dp_rank=make_groups(rank);dist.barrier()
    ops=[];timeline=[]
    if a.mode.startswith("tp_"):
        measurements,equal,state=run_tp(a,dev,rank,groups,tp_name,dp_name,tp_rank,dp_rank,ops,timeline)
    elif a.mode.startswith("ep_"):
        measurements,equal,state=run_ep(a,dev,rank,groups,tp_name,dp_name,tp_rank,dp_rank,ops,timeline)
    else:
        measurements,equal,state=run_pp(a,dev,rank,groups,dp_name,dp_rank,ops,timeline)
    if not equal:raise RuntimeError("data-parallel replicas diverged")
    a.output_dir.mkdir(parents=True,exist_ok=True)
    write_csv(a.output_dir/f"iterations_rank{rank}.csv",measurements)
    write_csv(a.output_dir/f"operations_rank{rank}.csv",ops)
    write_csv(a.output_dir/f"timeline_rank{rank}.csv",timeline)
    state.update({"rank":rank,"config":a.config,"mode":a.mode,"tp_rank":tp_rank,"dp_rank":dp_rank,"replicas_equal":equal})
    (a.output_dir/f"state_rank{rank}.json").write_text(json.dumps(state,sort_keys=True)+"\n",encoding="utf-8")
    dist.barrier();dist.destroy_process_group()


if __name__=="__main__":main()
