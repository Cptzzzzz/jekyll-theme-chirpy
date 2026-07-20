#!/usr/bin/env python3
"""DDP Reducer worker for bucket layout, delayed backward, and unused parameters."""
from __future__ import annotations
import argparse,csv,json,os,time
from dataclasses import dataclass,field
from datetime import timedelta
from pathlib import Path
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION","python")
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook

class SleepGrad(torch.autograd.Function):
  @staticmethod
  def forward(ctx,x,cycles):ctx.cycles=int(cycles);return x
  @staticmethod
  def backward(ctx,grad):
    if ctx.cycles:torch.cuda._sleep(ctx.cycles)
    return grad,None

class ParameterWorkload(torch.nn.Module):
  def __init__(self,count,elements,sleep_cycles,add_unused):
    super().__init__();self.params=torch.nn.ParameterList([torch.nn.Parameter(torch.full((elements,),.01)) for _ in range(count)])
    self.sleep_cycles=sleep_cycles
    if add_unused:self.unused=torch.nn.Parameter(torch.full((elements,),.02))
  def forward(self,_dummy):
    scale=1.0/sum(p.numel() for p in self.params);out=torch.zeros((),device=self.params[0].device)
    for p in self.params:out=out+SleepGrad.apply(p,self.sleep_cycles).sum()*scale
    return out

@dataclass
class HookState:
  group:object
  iteration:int=-999
  start_ns:int=0
  rows:list=field(default_factory=list)

def comm_hook(state,bucket):
  state.rows.append({"iteration":state.iteration,"bucket_index":bucket.index(),"is_last":int(bucket.is_last()),
    "bucket_bytes":bucket.buffer().numel()*bucket.buffer().element_size(),"parameter_count":len(bucket.parameters()),
    "ready_host_us":(time.perf_counter_ns()-state.start_ns)/1000})
  return allreduce_hook(state.group,bucket)

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,required=True);ap.add_argument("--config",required=True)
  ap.add_argument("--param-mib",type=int,required=True);ap.add_argument("--param-count",type=int,required=True);ap.add_argument("--bucket-cap-mb",type=int,required=True)
  ap.add_argument("--sleep-cycles",type=int,default=0);ap.add_argument("--warmup",type=int,default=3);ap.add_argument("--cycles",type=int,default=5)
  ap.add_argument("--find-unused",action="store_true");ap.add_argument("--static-graph",action="store_true");ap.add_argument("--add-unused",action="store_true")
  ap.add_argument("--profile-capture",action="store_true");a=ap.parse_args()
  local=int(os.environ["LOCAL_RANK"]);torch.cuda.set_device(local);dev=torch.device("cuda",local)
  dist.init_process_group("nccl",timeout=timedelta(seconds=90));rank=dist.get_rank();world=dist.get_world_size()
  if world!=4:raise RuntimeError("requires four ranks")
  elements=a.param_mib*(1<<20)//4;torch.manual_seed(1)
  model=ParameterWorkload(a.param_count,elements,a.sleep_cycles,a.add_unused).to(dev)
  ddp=DDP(model,device_ids=[local],bucket_cap_mb=a.bucket_cap_mb,find_unused_parameters=a.find_unused,static_graph=a.static_graph,gradient_as_bucket_view=True)
  state=HookState(dist.group.WORLD);ddp.register_comm_hook(state,comm_hook);opt=torch.optim.SGD(ddp.parameters(),lr=.01)
  rows=[];dummy=torch.ones((),device=dev)
  total=a.warmup+a.cycles
  for step in range(total):
    measured=step-a.warmup;state.iteration=measured;state.start_ns=time.perf_counter_ns()
    opt.zero_grad(set_to_none=True);torch.cuda.synchronize(dev)
    if a.profile_capture and measured==0:torch.cuda.cudart().cudaProfilerStart()
    start=torch.cuda.Event(enable_timing=True);bwd0=torch.cuda.Event(enable_timing=True);bwd1=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record();loss=ddp(dummy);bwd0.record();h0=time.perf_counter_ns();loss.backward();h1=time.perf_counter_ns();bwd1.record();opt.step();h2=time.perf_counter_ns();end.record();end.synchronize()
    if a.profile_capture and measured==0:torch.cuda.cudart().cudaProfilerStop()
    if measured>=0:rows.append({"rank":rank,"config":a.config,"iteration":measured,"param_mib":a.param_mib,"param_count":a.param_count,
      "bucket_cap_mb":a.bucket_cap_mb,"sleep_cycles":a.sleep_cycles,"find_unused":int(a.find_unused),"static_graph":int(a.static_graph),"add_unused":int(a.add_unused),
      "backward_host_us":(h1-h0)/1000,"optimizer_host_us":(h2-h1)/1000,"backward_gpu_us":bwd0.elapsed_time(bwd1)*1000,
      "step_gpu_us":start.elapsed_time(end)*1000,"correct":1})
  sample=torch.stack([v for p in model.parameters() for v in (p.detach()[0],p.detach()[-1],p.detach().mean())])
  gathered=[torch.empty_like(sample) for _ in range(world)];dist.all_gather(gathered,sample);torch.cuda.synchronize(dev)
  replicas_equal=all(torch.equal(gathered[0],x) for x in gathered[1:])
  if not replicas_equal:raise RuntimeError("replicas diverged")
  log=ddp._get_ddp_logging_data();a.output_dir.mkdir(parents=True,exist_ok=True)
  with (a.output_dir/f"iterations_rank{rank}.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  hooks=[]
  for x in state.rows:
    if x["iteration"]>=0:hooks.append({"rank":rank,"config":a.config,**x})
  with (a.output_dir/f"buckets_rank{rank}.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(hooks[0]));w.writeheader();w.writerows(hooks)
  info={"rank":rank,"config":a.config,"replicas_equal":replicas_equal,"parameter_bytes":sum(p.numel()*p.element_size() for p in model.parameters()),
    "bucket_sizes":str(log.get("bucket_sizes","")),"rebuilt_bucket_sizes":str(log.get("rebuilt_bucket_sizes","")),
    "has_rebuilt_buckets":int(log.get("has_rebuilt_buckets",0)),"unused_parameter_size":int(log.get("unused_parameter_size",0)),
    "avg_backward_compute_time_ns":int(log.get("avg_backward_compute_time",0)),"avg_backward_comm_time_ns":int(log.get("avg_backward_comm_time",0)),
    "avg_backward_overlap_time_ns":int(log.get("avg_backward_compute_comm_overlap_time",0)),"prev_iteration_grad_ready_order_indices":str(log.get("prev_iteration_grad_ready_order_indices",""))}
  (a.output_dir/f"logging_rank{rank}.json").write_text(json.dumps(info,sort_keys=True)+"\n",encoding="utf-8")
  dist.barrier();dist.destroy_process_group()
if __name__=="__main__":main()
