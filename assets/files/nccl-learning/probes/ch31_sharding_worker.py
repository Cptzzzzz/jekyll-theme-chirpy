#!/usr/bin/env python3
"""Measure replicated, optimizer-sharded, gradient-sharded, and fully-sharded training."""
from __future__ import annotations
import argparse,csv,json,os,time
from datetime import timedelta
from functools import partial
from pathlib import Path
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION","python")
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,ShardingStrategy,BackwardPrefetch
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP

class DeepMLP(torch.nn.Module):
  def __init__(self,width,layers):
    super().__init__();self.layers=torch.nn.Sequential(*sum(([torch.nn.Linear(width,width,bias=False),torch.nn.ReLU()] for _ in range(layers)),[]))
  def forward(self,x):return self.layers(x)
def tensor_bytes(xs):return sum(x.numel()*x.element_size() for x in xs if isinstance(x,torch.Tensor))
def state_bytes(opt):
  local=opt.optim if isinstance(opt,ZeroRedundancyOptimizer) else opt;seen=set();total=0
  def visit(x):
    nonlocal total
    if isinstance(x,torch.Tensor):
      key=(x.untyped_storage().data_ptr(),x.untyped_storage().nbytes())
      if key not in seen:seen.add(key);total+=x.numel()*x.element_size()
    elif isinstance(x,dict):
      for v in x.values():visit(v)
    elif isinstance(x,(list,tuple)):
      for v in x:visit(v)
  visit(local.state);return total
def param_bytes(module):return tensor_bytes(list(module.parameters()))
def grad_bytes(module):return tensor_bytes([p.grad for p in module.parameters() if p.grad is not None])
def main():
  ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,required=True);ap.add_argument("--config",required=True)
  ap.add_argument("--mode",choices=("zero0","zero1","zero2","zero3"),required=True);ap.add_argument("--prefetch",choices=("pre","post","none"),default="pre")
  ap.add_argument("--width",type=int,default=2048);ap.add_argument("--layers",type=int,default=8);ap.add_argument("--batch",type=int,default=16)
  ap.add_argument("--warmup",type=int,default=3);ap.add_argument("--cycles",type=int,default=10);ap.add_argument("--profile-capture",action="store_true");a=ap.parse_args()
  local=int(os.environ["LOCAL_RANK"]);torch.cuda.set_device(local);dev=torch.device("cuda",local)
  dist.init_process_group("nccl",timeout=timedelta(seconds=180));rank=dist.get_rank();world=dist.get_world_size()
  torch.manual_seed(20260712);model=DeepMLP(a.width,a.layers).to(dev);logical=param_bytes(model)
  if a.mode in ("zero0","zero1"):
    wrapped=DDP(model,device_ids=[local],bucket_cap_mb=64)
  else:
    strategy=ShardingStrategy.SHARD_GRAD_OP if a.mode=="zero2" else ShardingStrategy.FULL_SHARD
    bp={"pre":BackwardPrefetch.BACKWARD_PRE,"post":BackwardPrefetch.BACKWARD_POST,"none":None}[a.prefetch]
    wrapped=FSDP(model,auto_wrap_policy=partial(size_based_auto_wrap_policy,min_num_params=a.width*a.width),device_id=dev,
      sharding_strategy=strategy,use_orig_params=True,backward_prefetch=bp,limit_all_gathers=True)
  after_wrap=torch.cuda.memory_allocated(dev)
  if a.mode=="zero1":opt=ZeroRedundancyOptimizer(wrapped.parameters(),optimizer_class=torch.optim.AdamW,lr=1e-4)
  else:opt=torch.optim.AdamW(wrapped.parameters(),lr=1e-4)
  gen=torch.Generator(device=dev).manual_seed(9100+rank)
  def one_step():
    x=torch.randn(a.batch,a.width,generator=gen,device=dev);out=wrapped(x);loss=out.square().mean();loss.backward();return loss
  # Materialize gradients and lazy Adam state once.
  opt.zero_grad(set_to_none=True);one_step();torch.cuda.synchronize(dev);local_grad=grad_bytes(wrapped);opt.step();torch.cuda.synchronize(dev)
  local_optim=state_bytes(opt);opt.zero_grad(set_to_none=True);torch.cuda.synchronize(dev)
  local_param=param_bytes(wrapped);steady=torch.cuda.memory_allocated(dev)
  for _ in range(a.warmup):one_step();opt.step();opt.zero_grad(set_to_none=True);torch.cuda.synchronize(dev)
  torch.cuda.reset_peak_memory_stats(dev);rows=[]
  for i in range(a.cycles):
    if a.profile_capture and i==0:torch.cuda.cudart().cudaProfilerStart()
    start=torch.cuda.Event(enable_timing=True);b0=torch.cuda.Event(enable_timing=True);b1=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record();opt.zero_grad(set_to_none=True);x=torch.randn(a.batch,a.width,generator=gen,device=dev);out=wrapped(x);loss=out.square().mean();b0.record()
    h0=time.perf_counter_ns();loss.backward();h1=time.perf_counter_ns();b1.record();opt.step();h2=time.perf_counter_ns();end.record();end.synchronize()
    if a.profile_capture and i==0:torch.cuda.cudart().cudaProfilerStop()
    rows.append({"rank":rank,"config":a.config,"iteration":i,"mode":a.mode,"prefetch":a.prefetch,"step_gpu_us":start.elapsed_time(end)*1000,
      "backward_gpu_us":b0.elapsed_time(b1)*1000,"backward_host_us":(h1-h0)/1000,"optimizer_host_us":(h2-h1)/1000,"correct":1})
  peak=torch.cuda.max_memory_allocated(dev)
  # Materialize a common full-parameter sample to check replicas and cross-mode numerics.
  ctx=FSDP.summon_full_params(wrapped,recurse=True,writeback=False) if isinstance(wrapped,FSDP) else __import__('contextlib').nullcontext()
  with ctx:
    vals=[]
    for p in wrapped.parameters():
      q=p.detach().reshape(-1);vals.extend((q[0],q[-1],q.mean()))
    sample=torch.stack(vals)
  gathered=[torch.empty_like(sample) for _ in range(world)];dist.all_gather(gathered,sample);torch.cuda.synchronize(dev)
  equal=all(torch.allclose(gathered[0],x,rtol=1e-5,atol=1e-6) for x in gathered[1:])
  if not equal:raise RuntimeError("replica sample mismatch")
  a.output_dir.mkdir(parents=True,exist_ok=True)
  with (a.output_dir/f"iterations_rank{rank}.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  state={"rank":rank,"config":a.config,"mode":a.mode,"prefetch":a.prefetch,"logical_parameter_bytes":logical,"local_parameter_bytes":local_param,
    "local_gradient_bytes":local_grad,"local_optimizer_state_bytes":local_optim,"after_wrap_allocated_bytes":after_wrap,"steady_allocated_bytes":steady,
    "peak_allocated_bytes":peak,"replicas_equal":equal,"sample":[float(x) for x in gathered[0].cpu()]}
  (a.output_dir/f"state_rank{rank}.json").write_text(json.dumps(state,sort_keys=True)+"\n",encoding="utf-8")
  dist.barrier();dist.destroy_process_group()
if __name__=="__main__":main()
