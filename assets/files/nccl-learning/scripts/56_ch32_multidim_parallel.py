#!/usr/bin/env python3
"""Run chapter 32 trainable TP/PP/EP experiments and trace their NCCL fingerprints."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,statistics,subprocess
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

CONFIGS=(
 ("tp_ar_b32","tp_ar",32,1),("tp_ar_b128","tp_ar",128,1),("tp_sp_b128","tp_sp",128,1),
 ("pp_mb1","pp",64,1),("pp_mb8","pp",64,8),
 ("ep_balanced","ep_balanced",128,1),("ep_skewed","ep_skewed",128,1),
)
PROFILE=("tp_ar_b128","tp_sp_b128","pp_mb8","ep_balanced","ep_skewed")

def clean():
  e=dict(os.environ)
  for k in tuple(e):
    if k.startswith("NCCL_") or k.startswith("TORCH_NCCL_") or k in ("MASTER_ADDR","MASTER_PORT","LD_LIBRARY_PATH"):e.pop(k)
  e["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python";return e
def run(cmd,e,path,cwd,check=True,timeout=600):
  p=subprocess.run(cmd,cwd=cwd,env=e,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(p.stdout,encoding="utf-8")
  if check and p.returncode:raise RuntimeError(f"failed {p.returncode}: {path}\n{p.stdout[-4000:]}")
  return p.returncode,p.stdout
def outcsv(p,rows):
  if not rows:raise RuntimeError(f"empty {p}")
  fields=list(rows[0]);fields.extend(k for r in rows for k in r if k not in fields)
  with p.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def readcsvs(d,prefix):
  rows=[]
  for p in sorted(d.glob(prefix+"*.csv")):rows+=list(csv.DictReader(p.open()))
  return rows
def command(base,worker,out,c,cycles,warmup=3,profile=False):
  name,mode,batch,micro=c
  x=base+[str(worker),"--output-dir",str(out),"--config",name,"--mode",mode,"--batch",str(batch),"--microbatches",str(micro),"--d-model","512","--hidden","1024","--cycles",str(cycles),"--warmup",str(warmup)]
  if profile:x.append("--profile-capture")
  return x
def pct(v,q):
  a=sorted(v);p=(len(a)-1)*q;l=math.floor(p);h=math.ceil(p);return a[l] if l==h else a[l]+(a[h]-a[l])*(p-l)
def parse_gpu(text,config):
  lines=text.splitlines();i=next(i for i,x in enumerate(lines) if x.startswith("Start (ns),"));rows=list(csv.DictReader(lines[i:]));g=defaultdict(lambda:defaultdict(lambda:{"count":0,"duration":0}))
  for r in rows:
    n=r["Name"]
    op=next((o for o in ("AllReduce","AllGather","ReduceScatter") if f"ncclDevKernel_{o}" in n),None)
    if op is None and "ncclDevKernel_SendRecv" in n:op="SendRecv"
    if op:g[r["Device"]][op]["count"]+=1;g[r["Device"]][op]["duration"]+=int(r["Duration (ns)"])
  out=[]
  for dev,ops in sorted(g.items()):
    z={"config":config,"device":dev,"profile_private":1}
    for op in ("AllReduce","AllGather","ReduceScatter","SendRecv"):
      key=op.lower();z[key+"_count"]=ops[op]["count"];z[key+"_duration_us"]=ops[op]["duration"]/1000
    out.append(z)
  return out
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("/root/nccl-learning"));ap.add_argument("--run-id",default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"));ap.add_argument("--cycles",type=int,default=6);a=ap.parse_args()
  d=a.root/"logs/ch32_multidim_parallel"/a.run_id;raw=d/"raw";private=d/"private";private.mkdir(parents=True,exist_ok=True)
  torchrun=Path(subprocess.check_output(["which","torchrun"],text=True).strip());base=[str(torchrun),"--standalone","--nproc_per_node=4"];worker=a.root/"probes/ch32_multidim_parallel_worker.py"
  iterations=[];operations=[];timeline=[];states=[]
  for rep,order in ((1,CONFIGS),(2,tuple(reversed(CONFIGS)))):
    for c in order:
      name=c[0];od=raw/"rows"/f"{name}_r{rep}";e=clean();e["NCCL_DEBUG"]="WARN"
      run(command(base,worker,od,c,a.cycles),e,raw/"runtime"/f"{name}_r{rep}.log",a.root,True,600)
      ir=readcsvs(od,"iterations_rank");op=readcsvs(od,"operations_rank");tl=readcsvs(od,"timeline_rank");st=[json.loads(p.read_text()) for p in sorted(od.glob("state_rank*.json"))]
      if len(ir)!=4*a.cycles or any(x["correct"]!="1" for x in ir) or len(st)!=4 or not all(x["replicas_equal"] for x in st) or max(x["numeric_max_abs"] for x in st)>2e-5:raise RuntimeError(f"accept {name}")
      for xs in (ir,op,tl,st):
        for x in xs:x["replicate"]=rep
      iterations+=ir;operations+=op;timeline+=tl;states+=st;print(f"[parallel] {name} r={rep} PASS",flush=True)
  outcsv(d/"iteration_measurements.csv",iterations);outcsv(d/"operation_payload_timeline.csv",operations);outcsv(d/"pipeline_microbatch_timeline.csv",timeline)
  outcsv(d/"rank_group_state.csv",states)

  # Report critical-rank time per logical iteration, rather than averaging asynchronous stages.
  critical=defaultdict(list)
  byiter=defaultdict(list)
  for x in iterations:byiter[(x["config"],x["replicate"],x["iteration"])].append(x)
  for (name,rep,it),xs in byiter.items():critical[name].append((max(float(x["step_gpu_us"]) for x in xs),max(float(x["step_host_us"]) for x in xs)))
  perf=[]
  for name,vals in sorted(critical.items()):
    gpu=[x[0] for x in vals];host=[x[1] for x in vals]
    perf.append({"config":name,"logical_iterations":len(vals),"median_critical_gpu_us":statistics.median(gpu),"p95_critical_gpu_us":pct(gpu,.95),
                 "median_critical_host_us":statistics.median(host),"p95_critical_host_us":pct(host,.95),"host_cv_percent":statistics.pstdev(host)/statistics.fmean(host)*100})
  outcsv(d/"performance_summary.csv",perf)

  grouped=defaultdict(list)
  for x in operations:grouped[(x["config"],x["phase"],x["operation"],x["group"])].append(x)
  opsum=[]
  logical_iterations=a.cycles*2
  for (name,phase,op,group),xs in sorted(grouped.items()):
    # Collective rows contain one record per participating rank; P2P rows contain
    # one record for the sending or receiving endpoint named by the operation.
    participants=1 if op in ("Send","Recv") else 2
    opsum.append({"config":name,"phase":phase,"operation":op,"group":group,"rows":len(xs),
                  "calls_per_group_iteration":len(xs)/(logical_iterations*participants),"median_payload_bytes":statistics.median(int(x["payload_bytes"]) for x in xs),
                  "max_payload_bytes":max(int(x["payload_bytes"]) for x in xs)})
  outcsv(d/"operation_summary.csv",opsum)

  model=d/"parallel_source_model.csv";run(["python3",str(a.root/"probes/ch32_parallel_source_model.py"),"--output",str(model)],clean(),raw/"source_model.log",a.root)
  mr=list(csv.DictReader(model.open()))
  if len(mr)!=39 or any(x["pass"]!="1" for x in mr):raise RuntimeError(f"source model {len(mr)}")

  nsys=Path(subprocess.check_output(["which","nsys"],text=True).strip());nrows=[]
  for name in PROFILE:
    c=next(x for x in CONFIGS if x[0]==name);prefix=private/f"nsys_{name}";od=private/f"rows_{name}";e=clean();e["NCCL_DEBUG"]="WARN"
    run([str(nsys),"profile","--force-overwrite=true","--sample=none","--cpuctxsw=none","--trace=cuda,nvtx","--capture-range=cudaProfilerApi","--capture-range-end=stop","-o",str(prefix)]+command(base,worker,od,c,1,2,True),e,raw/"nsys"/f"{name}.log",a.root,True,900)
    rep=Path(str(prefix)+".nsys-rep");_,gpu=run([str(nsys),"stats","--force-export=true","--report","cuda_gpu_trace","--format","csv",str(rep)],clean(),private/f"{name}_gpu.csv",a.root)
    q=parse_gpu(gpu,name)
    if len(q)!=4:raise RuntimeError(f"nsys devices {name} {q}")
    nrows+=q
  rows=lambda name:[x for x in nrows if x["config"]==name]
  if any(x["allreduce_count"]<3 for x in rows("tp_ar_b128")):raise RuntimeError("tp ar trace")
  if any(x["allgather_count"]<1 or x["reducescatter_count"]<1 or x["allreduce_count"]<2 for x in rows("tp_sp_b128")):raise RuntimeError("tp sp trace")
  for name in ("pp_mb8","ep_balanced","ep_skewed"):
    if any(x["sendrecv_count"]<1 or x["allreduce_count"]<1 for x in rows(name)):raise RuntimeError(f"p2p trace {name}")
  outcsv(d/"nsys_collective_summary.csv",nrows)

  pytorch=Path("/opt/pytorch/pytorch/torch/distributed/nn/functional.py");style=Path("/opt/pytorch/pytorch/torch/distributed/tensor/parallel/style.py")
  megatron=a.root/"third_party/Megatron-LM-core_v0.8.0";deepspeed=a.root/"third_party/DeepSpeed-0.14.4"
  meg_commit=subprocess.check_output(["git","-C",str(megatron),"rev-parse","HEAD"],text=True).strip();ds_commit=subprocess.check_output(["git","-C",str(deepspeed),"rev-parse","HEAD"],text=True).strip()
  import torch
  manifest=["experiment=ch32_multidim_parallel",f"run_id={a.run_id}",f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",f"hostname={os.uname().nodename}",f"iteration_rows={len(iterations)}",f"operation_rows={len(operations)}",f"pipeline_timeline_rows={len(timeline)}",f"source_model_rows={len(mr)}","correctness=PASS","replica_equality=PASS","numeric_reference=PASS","ep_upstream_activation_gradient=PASS","nsys_paths=20/20 PASS","private_nsys_reports=NOT_FOR_PUBLICATION",f"pytorch_runtime={torch.__version__}","nccl_runtime=2.22.3",f"pytorch_nn_functional_sha256={sha(pytorch)}",f"pytorch_tp_style_sha256={sha(style)}","megatron_source_tag=core_v0.8.0",f"megatron_source_commit={meg_commit}","deepspeed_source_tag=v0.14.4",f"deepspeed_source_commit={ds_commit}"]
  (d/"manifest.txt").write_text("\n".join(manifest)+"\n",encoding="utf-8")
  lines=["# Chapter 32 experiment summary","",f"- Iteration rows: {len(iterations)}",f"- Communication operation rows: {len(operations)}",f"- Pipeline timeline rows: {len(timeline)}","- Numeric reference and DP replica equality: PASS","- Source model: 39/39 PASS","- Nsight paths: 20/20 PASS","","| config | critical GPU us | critical host us | host CV |","|---|---:|---:|---:|---:|"]
  for x in perf:lines.append(f"| {x['config']} | {x['median_critical_gpu_us']:.1f} | {x['median_critical_host_us']:.1f} | {x['host_cv_percent']:.2f}% |")
  (d/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(f"run_dir={d}",flush=True)
if __name__=="__main__":main()
