#!/usr/bin/env python3
"""Emit aligned CUDA/NVTX and ProcessGroupNCCL flight-recorder evidence."""
import argparse,csv,os,pickle
from datetime import timedelta
from pathlib import Path
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION","python")
import torch
import torch.distributed as dist
SIZES=(4096,1<<20,64<<20)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,required=True);ap.add_argument("--profile",action="store_true");a=ap.parse_args()
 local=int(os.environ["LOCAL_RANK"]);torch.cuda.set_device(local);dev=torch.device("cuda",local);dist.init_process_group("nccl",timeout=timedelta(seconds=90));rank=dist.get_rank();rows=[]
 tensors=[torch.full((n//4,),float(rank+1),device=dev) for n in SIZES]
 for x in tensors:dist.all_reduce(x);torch.cuda.synchronize(dev);x.fill_(rank+1)
 if a.profile:torch.cuda.cudart().cudaProfilerStart()
 for seq,(n,x) in enumerate(zip(SIZES,tensors)):
  name=f"ch34_seq{seq}_bytes{n}_allreduce";torch.cuda.nvtx.range_push(name);start=torch.cuda.Event(True);end=torch.cuda.Event(True);start.record();dist.all_reduce(x);end.record();end.synchronize();torch.cuda.nvtx.range_pop()
  correct=float(x[0])==3.0;rows.append({"rank":rank,"sequence":seq,"bytes":n,"nvtx_range":name,"gpu_us":start.elapsed_time(end)*1000,"correct":int(correct)})
  if not correct:raise RuntimeError("allreduce mismatch")
 if a.profile:torch.cuda.cudart().cudaProfilerStop()
 trace=torch._C._distributed_c10d._dump_nccl_trace(True,False,False)
 decoded=pickle.loads(trace)
 a.output_dir.mkdir(parents=True,exist_ok=True)
 with (a.output_dir/f"measurements_rank{rank}.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (a.output_dir/f"flight_rank{rank}.pkl").write_bytes(trace)
 (a.output_dir/f"flight_shape_rank{rank}.txt").write_text(f"type={type(decoded).__name__}\nkeys={sorted(decoded) if isinstance(decoded,dict) else 'n/a'}\n",encoding="utf-8")
 dist.destroy_process_group()
if __name__=="__main__":main()
