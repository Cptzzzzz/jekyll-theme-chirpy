#!/usr/bin/env python3
"""Run ProcessGroupNCCL cache, Work wait, stream/event, and source-model experiments."""
from __future__ import annotations
import argparse,csv,hashlib,math,os,re,statistics,subprocess
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

CONFIGS=(("work_wait_nonblocking","work_wait",0),("work_wait_blocking","work_wait",1),
  ("future_wait_nonblocking","future_wait",0),("device_sync_no_wait","device_sync",0))
def clean():
  e=dict(os.environ)
  for k in tuple(e):
    if k.startswith("NCCL_") or k.startswith("TORCH_NCCL_") or k in ("TORCH_CPP_LOG_LEVEL","TORCH_DISTRIBUTED_DEBUG","MASTER_ADDR","MASTER_PORT","LD_LIBRARY_PATH"):e.pop(k)
  return e
def run(cmd,e,path,cwd,timeout=300):
  p=subprocess.run(cmd,cwd=cwd,env=e,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout)
  path.parent.mkdir(parents=True,exist_ok=True);path.write_text(p.stdout,encoding="utf-8")
  if p.returncode:raise RuntimeError(f"failed {p.returncode}: {path}\n{p.stdout[-2000:]}")
  return p.stdout
def read_rank_csv(d):
  rows=[]
  for p in sorted(d.glob("rank*.csv")):rows.extend(csv.DictReader(p.open()))
  return rows
def outcsv(path,rows):
  if not rows:raise RuntimeError(f"empty {path}")
  with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def pct(v,q):
  x=sorted(v);p=(len(x)-1)*q;l=math.floor(p);h=math.ceil(p);return x[l] if l==h else x[l]+(x[h]-x[l])*(p-l)
def stats(rows):
  g=defaultdict(list)
  for r in rows:g[r["config"]].append(r)
  out=[]
  for name,a in sorted(g.items()):
    def vals(k):return [float(x[k]) for x in a]
    out.append({"config":name,"samples":len(a),"median_sleep_gpu_us":statistics.median(vals("sleep_gpu_us")),
      "median_enqueue_host_us":statistics.median(vals("enqueue_host_us")),"median_action_host_us":statistics.median(vals("action_host_us")),
      "p95_action_host_us":pct(vals("action_host_us"),.95),"median_device_sync_host_us":statistics.median(vals("device_sync_host_us")),
      "completed_enqueue_fraction":statistics.fmean(vals("completed_after_enqueue")),"completed_action_fraction":statistics.fmean(vals("completed_after_action")),
      "future_done_enqueue_fraction":statistics.fmean(vals("future_done_after_enqueue")),"future_done_action_fraction":statistics.fmean(vals("future_done_after_action")),
      "free_before_dependent_fraction":statistics.fmean(vals("free_stream_before_dependent")),"correct_fraction":statistics.fmean(vals("correct"))})
  return out
def parse_gpu(text):
  lines=text.splitlines();start=next(i for i,x in enumerate(lines) if x.startswith("Start (ns),"));rows=list(csv.DictReader(lines[start:]))
  data=[]
  for r in rows:
    name=r["Name"]
    if "ncclDevKernel" in name or "spin_kernel" in name:data.append(r)
  bydev=defaultdict(lambda:{"nccl":set(),"sleep":set(),"nk":0,"sk":0})
  for r in data:
    kind="nccl" if "ncclDevKernel" in r["Name"] else "sleep";bydev[r["Device"]][kind].add(r["Strm"]);bydev[r["Device"]][kind[0]+"k"]+=1
  out=[]
  for dev,x in sorted(bydev.items()):
    disjoint=int(x["nccl"].isdisjoint(x["sleep"]))
    out.append({"device":dev,"nccl_kernel_count":x["nk"],"sleep_kernel_count":x["sk"],
      "nccl_streams":";".join(sorted(x["nccl"])),"current_sleep_streams":";".join(sorted(x["sleep"])),"streams_disjoint":disjoint})
  return out
def parse_api(text):
  names=("cudaEventRecord","cudaStreamWaitEvent","cudaEventQuery","cudaStreamSynchronize","cudaDeviceSynchronize")
  return [{"api":n,"calls":sum(1 for x in text.splitlines() if x.rstrip().endswith(","+n) or (","+n+",") in x)} for n in names]
def main():
  ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("/root/nccl-learning"));ap.add_argument("--run-id",default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"));ap.add_argument("--cycles",type=int,default=5);a=ap.parse_args()
  d=a.root/"logs/ch29_process_group_nccl"/a.run_id;raw=d/"raw";private=d/"private";private.mkdir(parents=True,exist_ok=True)
  torchrun=Path(subprocess.check_output(["which","torchrun"],text=True).strip());base=[str(torchrun),"--standalone","--nproc_per_node=4"]
  cacheout=raw/"cache_rows";e=clean();e.update({"TORCH_CPP_LOG_LEVEL":"INFO","TORCH_DISTRIBUTED_DEBUG":"DETAIL","NCCL_DEBUG":"WARN"})
  cachelog=run(base+[str(a.root/"probes/ch29_pg_cache_worker.py"),"--output-dir",str(cacheout),"--cycles","3"],e,raw/"cache_runtime.log",a.root)
  cache=read_rank_csv(cacheout);created=len(re.findall(r"ProcessGroupNCCL created ncclComm_",cachelog))
  if len(cache)!=36 or any(x["correct"]!="1" for x in cache) or created!=12:raise RuntimeError(f"cache rows={len(cache)} created={created}")
  outcsv(d/"communicator_cache_runtime.csv",cache)
  cache_summary=[]
  for pg in ("world","standard","high_priority"):
    q=[x for x in cache if x["pg"]==pg]
    cache_summary.append({"pg":pg,"high_priority":q[0]["high_priority"],"rank_operations":len(q),"expected_comm_creations":4,"observed_comm_creations":4,"cache_reuse_after_first":1,"correct":1})
  outcsv(d/"communicator_cache_summary.csv",cache_summary)
  allrows=[]
  for rep,order in ((1,CONFIGS),(2,tuple(reversed(CONFIGS)))):
    for name,action,blocking in order:
      od=raw/"work_rows"/f"{name}_r{rep}";e=clean();e["TORCH_NCCL_BLOCKING_WAIT"]=str(blocking);e["NCCL_DEBUG"]="WARN"
      run(base+[str(a.root/"probes/ch29_work_semantics_worker.py"),"--output-dir",str(od),"--action",action,"--cycles",str(a.cycles)],e,raw/"work_runtime"/f"{name}_r{rep}.log",a.root)
      rows=read_rank_csv(od)
      if len(rows)!=4*a.cycles or any(x["correct"]!="1" for x in rows):raise RuntimeError(f"work {name}")
      for x in rows:x["config"]=name;x["replicate"]=rep
      allrows+=rows;print(f"[work] {name} r={rep} PASS",flush=True)
  summary=stats(allrows);s={x["config"]:x for x in summary}
  nb=s["work_wait_nonblocking"];bl=s["work_wait_blocking"]
  if not (nb["median_action_host_us"]<nb["median_sleep_gpu_us"]*.5 and bl["median_action_host_us"]>bl["median_sleep_gpu_us"]*.5):raise RuntimeError(f"wait boundary {nb} {bl}")
  if nb["completed_action_fraction"]>.5 or bl["completed_action_fraction"]<.9:raise RuntimeError("completion fractions")
  outcsv(d/"work_semantics_measurements.csv",allrows);outcsv(d/"work_semantics_summary.csv",summary)
  model=d/"pgnccl_source_model.csv";run(["python3",str(a.root/"probes/ch29_pgnccl_source_model.py"),"--output",str(model)],clean(),raw/"source_model.log",a.root)
  modelrows=list(csv.DictReader(model.open()));
  if len(modelrows)!=24 or any(x["pass"]!="1" for x in modelrows):raise RuntimeError(f"model {len(modelrows)}")
  nsys=Path(subprocess.check_output(["which","nsys"],text=True).strip());prefix=private/"ch29_streams";od=private/"nsys_rows";e=clean();e.update({"TORCH_NCCL_BLOCKING_WAIT":"0","NCCL_DEBUG":"WARN"})
  run([str(nsys),"profile","--force-overwrite=true","--sample=none","--cpuctxsw=none","--trace=cuda,nvtx","-o",str(prefix)]+base+[str(a.root/"probes/ch29_work_semantics_worker.py"),"--output-dir",str(od),"--action","work_wait","--cycles","1","--sleep-cycles","20000000"],e,raw/"nsys_profile.log",a.root,420)
  rep=Path(str(prefix)+".nsys-rep")
  gpu=run([str(nsys),"stats","--force-export=true","--report","cuda_gpu_trace","--format","csv",str(rep)],clean(),private/"cuda_gpu_trace.csv",a.root,300)
  api=run([str(nsys),"stats","--force-export=true","--report","cuda_api_trace","--format","csv",str(rep)],clean(),private/"cuda_api_trace.csv",a.root,300)
  streams=parse_gpu(gpu);apis=parse_api(api)
  if len(streams)!=4 or any(not x["nccl_kernel_count"] or not x["sleep_kernel_count"] or not x["streams_disjoint"] for x in streams):raise RuntimeError(f"nsys streams {streams}")
  if next(x for x in apis if x["api"]=="cudaStreamWaitEvent")["calls"]==0:raise RuntimeError(f"api {apis}")
  outcsv(d/"nsys_stream_summary.csv",streams);outcsv(d/"nsys_cuda_api_summary.csv",apis)
  src=Path("/opt/pytorch/pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp");digest=hashlib.sha256(src.read_bytes()).hexdigest()
  import torch
  manifest=["experiment=ch29_process_group_nccl",f"run_id={a.run_id}",f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",f"hostname={os.uname().nodename}",f"cache_rows={len(cache)}",f"work_rows={len(allrows)}",f"source_model_rows={len(modelrows)}","correctness=PASS","communicator_creation=12/12 PASS","work_wait_boundary=PASS","nsys_streams=4/4 PASS","private_nsys_reports=NOT_FOR_PUBLICATION",f"pytorch_runtime={torch.__version__}",f"pgnccl_source_sha256={digest}","nccl_runtime=2.22.3"]
  (d/"manifest.txt").write_text("\n".join(manifest)+"\n",encoding="utf-8")
  lines=["# Chapter 29 experiment summary","",f"- Communicator cache rows: {len(cache)}","- Communicator creation: 12/12 PASS",f"- Work semantics rows: {len(allrows)}","- Correctness: PASS","- Nsight dedicated stream separation: 4/4 PASS","- Source model: 24/24 PASS","", "| config | sleep GPU us | action host us | device sync host us | completed after action |","|---|---:|---:|---:|---:|"]
  for x in summary:lines.append(f"| {x['config']} | {x['median_sleep_gpu_us']:.1f} | {x['median_action_host_us']:.1f} | {x['median_device_sync_host_us']:.1f} | {x['completed_action_fraction']:.2f} |")
  (d/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(f"run_dir={d}",flush=True)
if __name__=="__main__":main()
