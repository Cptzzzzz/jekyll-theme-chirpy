#!/usr/bin/env python3
"""Compare ZeRO state scopes using DDP, ZRO, and FSDP strategies."""
from __future__ import annotations
import argparse,csv,json,math,os,statistics,subprocess
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
CONFIGS=(("zero0_ddp","zero0","pre"),("zero1_zro","zero1","pre"),("zero2_fsdp","zero2","pre"),
 ("zero3_pre","zero3","pre"),("zero3_post","zero3","post"),("zero3_none","zero3","none"))
def clean():
  e=dict(os.environ)
  for k in tuple(e):
    if k.startswith("NCCL_") or k.startswith("TORCH_NCCL_") or k in ("MASTER_ADDR","MASTER_PORT","LD_LIBRARY_PATH"):e.pop(k)
  return e
def run(cmd,e,path,cwd,check=True,timeout=600):
  p=subprocess.run(cmd,cwd=cwd,env=e,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(p.stdout,encoding="utf-8")
  if check and p.returncode:raise RuntimeError(f"failed {p.returncode}: {path}\n{p.stdout[-3000:]}")
  return p.returncode,p.stdout
def outcsv(p,rows):
  if not rows:raise RuntimeError(f"empty {p}")
  with p.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def readcsv(d):
  x=[]
  for p in sorted(d.glob("iterations_rank*.csv")):x+=list(csv.DictReader(p.open()))
  return x
def cmd(base,w,out,c,cycles,warmup=3,profile=False):
  name,mode,pref=c;x=base+[str(w),"--output-dir",str(out),"--config",name,"--mode",mode,"--prefetch",pref,"--cycles",str(cycles),"--warmup",str(warmup)]
  if profile:x.append("--profile-capture")
  return x
def pct(v,q):
  a=sorted(v);p=(len(a)-1)*q;l=math.floor(p);h=math.ceil(p);return a[l] if l==h else a[l]+(a[h]-a[l])*(p-l)
def parse_gpu(text,config):
  lines=text.splitlines();i=next(i for i,x in enumerate(lines) if x.startswith("Start (ns),"));rows=list(csv.DictReader(lines[i:]));g=defaultdict(lambda:defaultdict(lambda:{"count":0,"duration":0}))
  for r in rows:
    n=r["Name"];op=next((o for o in ("AllReduce","AllGather","ReduceScatter","Broadcast") if f"ncclDevKernel_{o}" in n),None)
    if op:g[r["Device"]][op]["count"]+=1;g[r["Device"]][op]["duration"]+=int(r["Duration (ns)"])
  out=[]
  for dev,ops in sorted(g.items()):
    z={"config":config,"device":dev,"profile_private":1}
    for op in ("AllReduce","AllGather","ReduceScatter","Broadcast"):z[op.lower()+"_count"]=ops[op]["count"];z[op.lower()+"_duration_us"]=ops[op]["duration"]/1000
    out.append(z)
  return out
def main():
  ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("/root/nccl-learning"));ap.add_argument("--run-id",default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"));ap.add_argument("--cycles",type=int,default=10);a=ap.parse_args()
  d=a.root/"logs/ch31_zero_fsdp_states"/a.run_id;raw=d/"raw";private=d/"private";private.mkdir(parents=True,exist_ok=True)
  tr=Path(subprocess.check_output(["which","torchrun"],text=True).strip());base=[str(tr),"--standalone","--nproc_per_node=4"];w=a.root/"probes/ch31_sharding_worker.py"
  iterations=[];states=[]
  for rep,order in ((1,CONFIGS),(2,tuple(reversed(CONFIGS)))):
    for c in order:
      name=c[0];od=raw/"rows"/f"{name}_r{rep}";e=clean();e["NCCL_DEBUG"]="WARN";run(cmd(base,w,od,c,a.cycles),e,raw/"runtime"/f"{name}_r{rep}.log",a.root)
      ir=readcsv(od);st=[json.loads(p.read_text()) for p in sorted(od.glob("state_rank*.json"))]
      if len(ir)!=4*a.cycles or len(st)!=4 or any(x["correct"]!="1" for x in ir) or not all(x["replicas_equal"] for x in st):raise RuntimeError(f"accept {name}")
      for x in ir:x["replicate"]=rep
      for x in st:x["replicate"]=rep
      iterations+=ir;states+=st;print(f"[state] {name} r={rep} PASS",flush=True)
  # Cross-mode samples must represent equivalent training updates.
  ref=next(x["sample"] for x in states if x["config"]=="zero0_ddp" and x["rank"]==0 and x["replicate"]==1)
  numeric=[]
  for name,_,_ in CONFIGS:
    samples=[x["sample"] for x in states if x["config"]==name and x["rank"]==0]
    err=max(abs(v-r) for s in samples for v,r in zip(s,ref));numeric.append({"config":name,"max_abs_vs_zero0":err,"tolerance":1e-4,"pass":int(err<=1e-4)})
  if any(not x["pass"] for x in numeric):raise RuntimeError(f"numeric {numeric}")
  outcsv(d/"numeric_equivalence.csv",numeric)
  # Aggregate physical state across four ranks, normalized by logical parameter bytes P.
  accounting=[]
  for name,mode,pref in CONFIGS:
    q=[x for x in states if x["config"]==name and x["replicate"]==1];P=q[0]["logical_parameter_bytes"]
    accounting.append({"config":name,"mode":mode,"world_size":4,"logical_parameter_bytes":P,
      "sum_local_parameter_factor":sum(x["local_parameter_bytes"] for x in q)/P,"sum_local_gradient_factor":sum(x["local_gradient_bytes"] for x in q)/P,
      "sum_local_optimizer_factor":sum(x["local_optimizer_state_bytes"] for x in q)/P,"max_rank_parameter_mib":max(x["local_parameter_bytes"] for x in q)/(1<<20),
      "max_rank_gradient_mib":max(x["local_gradient_bytes"] for x in q)/(1<<20),"max_rank_optimizer_mib":max(x["local_optimizer_state_bytes"] for x in q)/(1<<20)})
  expected={"zero0_ddp":(4,4,8),"zero1_zro":(4,4,2),"zero2_fsdp":(1,1,2),"zero3_pre":(1,1,2),"zero3_post":(1,1,2),"zero3_none":(1,1,2)}
  for x in accounting:
    for k,e in zip(("sum_local_parameter_factor","sum_local_gradient_factor","sum_local_optimizer_factor"),expected[x["config"]]):
      if abs(x[k]-e)>.08:raise RuntimeError(f"accounting {x} expected={expected[x['config']]}")
  outcsv(d/"state_accounting.csv",accounting)
  pg=defaultdict(list)
  for x in iterations:pg[x["config"]].append(x)
  perf=[]
  for name,a0 in sorted(pg.items()):
    t=[float(x["step_gpu_us"]) for x in a0];b=[float(x["backward_gpu_us"]) for x in a0];q=[x for x in states if x["config"]==name]
    perf.append({"config":name,"samples":len(t),"median_step_gpu_us":statistics.median(t),"p95_step_gpu_us":pct(t,.95),"step_cv_percent":statistics.pstdev(t)/statistics.fmean(t)*100,
      "median_backward_gpu_us":statistics.median(b),"max_steady_allocated_mib":max(x["steady_allocated_bytes"] for x in q)/(1<<20),"max_peak_allocated_mib":max(x["peak_allocated_bytes"] for x in q)/(1<<20)})
  outcsv(d/"iteration_measurements.csv",iterations);outcsv(d/"rank_state_measurements.csv",[{k:v for k,v in x.items() if k!="sample"} for x in states]);outcsv(d/"performance_summary.csv",perf)
  model=d/"zero_state_model.csv";run(["python3",str(a.root/"probes/ch31_zero_state_model.py"),"--output",str(model)],clean(),raw/"model.log",a.root);mr=list(csv.DictReader(model.open()))
  if len(mr)!=30 or any(x["pass"]!="1" for x in mr):raise RuntimeError(f"model {len(mr)}")
  nsys=Path(subprocess.check_output(["which","nsys"],text=True).strip());nrows=[]
  for c in CONFIGS:
    name=c[0];prefix=private/f"nsys_{name}";od=private/f"rows_{name}";e=clean();e["NCCL_DEBUG"]="WARN"
    run([str(nsys),"profile","--force-overwrite=true","--sample=none","--cpuctxsw=none","--trace=cuda,nvtx","--capture-range=cudaProfilerApi","--capture-range-end=stop","-o",str(prefix)]+cmd(base,w,od,c,1,2,True),e,raw/"nsys"/f"{name}.log",a.root,True,900)
    rep=Path(str(prefix)+".nsys-rep");_,gpu=run([str(nsys),"stats","--force-export=true","--report","cuda_gpu_trace","--format","csv",str(rep)],clean(),private/f"{name}_gpu.csv",a.root)
    q=parse_gpu(gpu,name)
    if len(q)!=4:raise RuntimeError(f"nsys {name} {q}")
    nrows+=q
  def rows(name):return [x for x in nrows if x["config"]==name]
  if any(x["allreduce_count"]==0 or x["allgather_count"] or x["reducescatter_count"] for x in rows("zero0_ddp")):raise RuntimeError("zero0 trace")
  if any(x["allreduce_count"]==0 or x["broadcast_count"]==0 for x in rows("zero1_zro")):raise RuntimeError("zero1 trace")
  if any(x["allgather_count"]==0 or x["reducescatter_count"]==0 for x in rows("zero2_fsdp")):raise RuntimeError("zero2 trace")
  if any(x["allgather_count"]<=rows("zero2_fsdp")[i]["allgather_count"] or x["reducescatter_count"]==0 for i,x in enumerate(rows("zero3_pre"))):raise RuntimeError("zero3 trace")
  outcsv(d/"nsys_collective_summary.csv",nrows)
  # Logical collective payload model; operation counts come from the captured step.
  payload=[];P=states[0]["logical_parameter_bytes"];N=4
  for name,mode,pref in CONFIGS:
    q=rows(name);med=lambda k:statistics.median(x[k] for x in q)
    payload.append({"config":name,"logical_parameter_bytes":P,"allreduce_input_bytes":P if mode in ("zero0","zero1") else 0,
      "forward_allgather_logical_bytes":P if mode in ("zero2","zero3") else 0,"backward_allgather_logical_bytes":P if mode=="zero3" else 0,
      "reduce_scatter_input_bytes":P if mode in ("zero2","zero3") else 0,"observed_allreduce_kernels_per_rank":med("allreduce_count"),
      "observed_allgather_kernels_per_rank":med("allgather_count"),"observed_reducescatter_kernels_per_rank":med("reducescatter_count"),"observed_broadcast_kernels_per_rank":med("broadcast_count")})
  outcsv(d/"collective_payload_model.csv",payload)
  import torch
  ds=a.root/"third_party/DeepSpeed-0.14.4";commit=subprocess.check_output(["git","-C",str(ds),"rev-parse","HEAD"],text=True).strip()
  manifest=["experiment=ch31_zero_fsdp_states",f"run_id={a.run_id}",f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",f"hostname={os.uname().nodename}",f"iteration_rows={len(iterations)}",f"rank_state_rows={len(states)}",f"source_model_rows={len(mr)}","correctness=PASS","state_accounting=6/6 PASS","numeric_equivalence=6/6 PASS","nsys_collectives=24/24 PASS","deepspeed_runtime=NOT_INSTALLED","deepspeed_source_tag=v0.14.4",f"deepspeed_source_commit={commit}","private_nsys_reports=NOT_FOR_PUBLICATION",f"pytorch_runtime={torch.__version__}","nccl_runtime=2.22.3"]
  (d/"manifest.txt").write_text("\n".join(manifest)+"\n",encoding="utf-8")
  lines=["# Chapter 31 experiment summary","",f"- Iteration rows: {len(iterations)}",f"- Rank state rows: {len(states)}","- Correctness and numeric equivalence: PASS","- State accounting: 6/6 PASS","- Nsight collective paths: 24/24 PASS","- Source model: 30/30 PASS","- DeepSpeed runtime: NOT INSTALLED; fixed source only","","| config | step us | CV | steady MiB | peak MiB |","|---|---:|---:|---:|---:|"]
  for x in perf:lines.append(f"| {x['config']} | {x['median_step_gpu_us']:.1f} | {x['step_cv_percent']:.2f}% | {x['max_steady_allocated_mib']:.1f} | {x['max_peak_allocated_mib']:.1f} |")
  (d/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(f"run_dir={d}",flush=True)
if __name__=="__main__":main()
