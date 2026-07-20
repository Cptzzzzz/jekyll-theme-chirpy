#!/usr/bin/env python3
"""Run DDP Reducer bucket, compute-delay, graph-mode, and overlap experiments."""
from __future__ import annotations
import argparse,csv,json,math,os,statistics,subprocess
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
CONFIGS=(
 ("fine1_cap4",1,48,4,0,0,0,0),("coarse4_cap4",4,12,4,0,0,0,0),("coarse4_cap16",4,12,16,0,0,0,0),("coarse4_cap64",4,12,64,0,0,0,0),
 ("delay_cap4",4,12,4,10000000,0,0,0),("delay_cap64",4,12,64,10000000,0,0,0),
 ("dynamic_unused",4,12,16,0,1,0,1),("static_unused",4,12,16,0,0,1,1))
def clean():
  e=dict(os.environ)
  for k in tuple(e):
    if k.startswith("NCCL_") or k.startswith("TORCH_NCCL_") or k in ("MASTER_ADDR","MASTER_PORT","LD_LIBRARY_PATH"):e.pop(k)
  return e
def run(cmd,e,path,cwd,check=True,timeout=420):
  p=subprocess.run(cmd,cwd=cwd,env=e,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout)
  path.parent.mkdir(parents=True,exist_ok=True);path.write_text(p.stdout,encoding="utf-8")
  if check and p.returncode:raise RuntimeError(f"failed {p.returncode}: {path}\n{p.stdout[-2500:]}")
  return p.returncode,p.stdout
def outcsv(path,rows):
  if not rows:raise RuntimeError(f"empty {path}")
  with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def readcsvs(d,prefix):
  x=[]
  for p in sorted(d.glob(prefix+"*.csv")):x+=list(csv.DictReader(p.open()))
  return x
def cmd(base,worker,out,c,cycles=5,profile=False):
  name,pm,pc,cap,sleep,find,static,unused=c;x=base+[str(worker),"--output-dir",str(out),"--config",name,"--param-mib",str(pm),"--param-count",str(pc),"--bucket-cap-mb",str(cap),"--sleep-cycles",str(sleep),"--warmup","3","--cycles",str(cycles)]
  if find:x.append("--find-unused")
  if static:x.append("--static-graph")
  if unused:x.append("--add-unused")
  if profile:x.append("--profile-capture")
  return x
def pct(v,q):
  a=sorted(v);p=(len(a)-1)*q;l=math.floor(p);h=math.ceil(p);return a[l] if l==h else a[l]+(a[h]-a[l])*(p-l)
def perfsummary(rows,logs):
  g=defaultdict(list)
  for r in rows:g[r["config"]].append(r)
  out=[]
  for name,a in sorted(g.items()):
    v=lambda k:[float(x[k]) for x in a];lg=[x for x in logs if x["config"]==name]
    active=lg[0]["rebuilt_bucket_sizes"] or lg[0]["bucket_sizes"]
    out.append({"config":name,"samples":len(a),"median_step_gpu_us":statistics.median(v("step_gpu_us")),"p95_step_gpu_us":pct(v("step_gpu_us"),.95),
      "median_backward_gpu_us":statistics.median(v("backward_gpu_us")),"median_backward_host_us":statistics.median(v("backward_host_us")),
      "median_optimizer_host_us":statistics.median(v("optimizer_host_us")),"rebuilt_bucket_sizes":lg[0]["rebuilt_bucket_sizes"],
      "active_bucket_sizes":active,"bucket_count":len([z for z in active.split(",") if z.strip()]),"has_rebuilt_buckets":int(lg[0]["has_rebuilt_buckets"]),"median_logger_comm_us":statistics.median(int(x["avg_backward_comm_time_ns"])/1000 for x in lg),
      "median_logger_overlap_us":statistics.median(int(x["avg_backward_overlap_time_ns"])/1000 for x in lg),"replicas_equal":int(all(x["replicas_equal"] for x in lg))})
  return out
def parse_gpu(text,config):
  lines=text.splitlines();i=next(i for i,x in enumerate(lines) if x.startswith("Start (ns),"));rows=list(csv.DictReader(lines[i:]));by=defaultdict(lambda:{"spin":[],"nccl":[]})
  for r in rows:
    n=r["Name"];interval=(int(r["Start (ns)"]),int(r["Start (ns)"])+int(r["Duration (ns)"]))
    if "spin_kernel" in n:by[r["Device"]]["spin"].append(interval)
    elif "ncclDevKernel_AllReduce" in n and "f32" in n:by[r["Device"]]["nccl"].append(interval)
  out=[]
  for dev,x in sorted(by.items()):
    overlap=0
    for a,b in x["spin"]:
      for c,d in x["nccl"]:overlap+=max(0,min(b,d)-max(a,c))
    out.append({"config":config,"device":dev,"spin_kernels":len(x["spin"]),"allreduce_kernels":len(x["nccl"]),"kernel_overlap_us":overlap/1000,
      "spin_span_us":((max(b for a,b in x["spin"])-min(a for a,b in x["spin"]))/1000 if x["spin"] else 0),"profile_private":1})
  return out
def main():
  ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("/root/nccl-learning"));ap.add_argument("--run-id",default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"));ap.add_argument("--cycles",type=int,default=5);a=ap.parse_args()
  d=a.root/"logs/ch30_ddp_reducer_overlap"/a.run_id;raw=d/"raw";private=d/"private";private.mkdir(parents=True,exist_ok=True)
  torchrun=Path(subprocess.check_output(["which","torchrun"],text=True).strip());base=[str(torchrun),"--standalone","--nproc_per_node=4"];worker=a.root/"probes/ch30_ddp_reducer_worker.py"
  iterations=[];buckets=[];logs=[]
  for rep,order in ((1,CONFIGS),(2,tuple(reversed(CONFIGS)))):
    for c in order:
      name=c[0];od=raw/"rows"/f"{name}_r{rep}";e=clean();e["NCCL_DEBUG"]="WARN"
      run(cmd(base,worker,od,c,a.cycles),e,raw/"runtime"/f"{name}_r{rep}.log",a.root)
      ir=readcsvs(od,"iterations_rank");br=readcsvs(od,"buckets_rank")
      lr=[json.loads(p.read_text()) for p in sorted(od.glob("logging_rank*.json"))]
      if len(ir)!=4*a.cycles or any(x["correct"]!="1" for x in ir) or len(lr)!=4 or not all(x["replicas_equal"] for x in lr):raise RuntimeError(f"accept {name}")
      for x in ir:x["replicate"]=rep
      for x in br:x["replicate"]=rep
      for x in lr:x["replicate"]=rep
      iterations+=ir;buckets+=br;logs+=lr;print(f"[ddp] {name} r={rep} PASS",flush=True)
  # Using an actual unused parameter without either mode must fail on the next iteration.
  fail=("unused_without_detection",4,12,16,0,0,0,1);e=clean();e["NCCL_DEBUG"]="WARN"
  rc,txt=run(cmd(base,worker,raw/"expected_failure_rows",fail,3),e,raw/"expected_failure.log",a.root,False)
  sig="Expected to have finished reduction in the prior iteration"
  if rc==0 or sig not in txt:raise RuntimeError(f"unused failure rc={rc}")
  failure=[{"case":"unused_without_detection","returncode":rc,"signature":sig,"expected_failure":1,"pass":1}];outcsv(d/"expected_failure.csv",failure)
  summary=perfsummary(iterations,logs);outcsv(d/"iteration_measurements.csv",iterations);outcsv(d/"bucket_ready_timeline.csv",buckets);outcsv(d/"ddp_logging_data.csv",logs);outcsv(d/"performance_summary.csv",summary)
  expected={"fine1_cap4":12,"coarse4_cap4":12,"coarse4_cap16":3,"coarse4_cap64":1,"delay_cap4":12,"delay_cap64":1,"dynamic_unused":4,"static_unused":4}
  for x in summary:
    if x["bucket_count"]!=expected[x["config"]]:raise RuntimeError(f"layout {x}")
  model=d/"reducer_source_model.csv";run(["python3",str(a.root/"probes/ch30_reducer_source_model.py"),"--output",str(model)],clean(),raw/"source_model.log",a.root)
  mr=list(csv.DictReader(model.open()));
  if len(mr)!=30 or any(x["pass"]!="1" for x in mr):raise RuntimeError(f"model {len(mr)}")
  nsys=Path(subprocess.check_output(["which","nsys"],text=True).strip());nrows=[]
  for name in ("delay_cap4","delay_cap64"):
    c=next(x for x in CONFIGS if x[0]==name);prefix=private/f"nsys_{name}";od=private/f"rows_{name}";e=clean();e["NCCL_DEBUG"]="WARN"
    run([str(nsys),"profile","--force-overwrite=true","--sample=none","--cpuctxsw=none","--trace=cuda,nvtx","--capture-range=cudaProfilerApi","--capture-range-end=stop","-o",str(prefix)]+cmd(base,worker,od,c,1,True),e,raw/"nsys"/f"{name}.log",a.root,True,600)
    rep=Path(str(prefix)+".nsys-rep");_,gpu=run([str(nsys),"stats","--force-export=true","--report","cuda_gpu_trace","--format","csv",str(rep)],clean(),private/f"{name}_gpu.csv",a.root)
    q=parse_gpu(gpu,name)
    if len(q)!=4:raise RuntimeError(f"nsys devices {q}")
    nrows+=q
  cap4=[x for x in nrows if x["config"]=="delay_cap4"];cap64=[x for x in nrows if x["config"]=="delay_cap64"]
  if any(x["spin_kernels"]!=12 or x["allreduce_kernels"]!=12 or x["kernel_overlap_us"]<=0 for x in cap4):raise RuntimeError(f"cap4 nsys {cap4}")
  if any(x["spin_kernels"]!=12 or x["allreduce_kernels"]!=1 for x in cap64):raise RuntimeError(f"cap64 nsys {cap64}")
  outcsv(d/"nsys_overlap_summary.csv",nrows)
  import torch
  manifest=["experiment=ch30_ddp_reducer_overlap",f"run_id={a.run_id}",f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",f"hostname={os.uname().nodename}",f"iteration_rows={len(iterations)}",f"bucket_ready_rows={len(buckets)}",f"source_model_rows={len(mr)}","correctness=PASS","bucket_layout=8/8 PASS","unused_failure=1/1 PASS","nsys_overlap=8/8 PASS","private_nsys_reports=NOT_FOR_PUBLICATION",f"pytorch_runtime={torch.__version__}","nccl_runtime=2.22.3"]
  (d/"manifest.txt").write_text("\n".join(manifest)+"\n",encoding="utf-8")
  lines=["# Chapter 30 experiment summary","",f"- Iteration rows: {len(iterations)}",f"- Bucket-ready rows: {len(buckets)}","- Correctness and replica equality: PASS","- Bucket layouts: 8/8 PASS","- Unused-parameter expected failure: PASS","- Nsight overlap: 8/8 PASS","- Source model: 30/30 PASS","","| config | buckets | step GPU us | backward GPU us | logger overlap us |","|---|---:|---:|---:|---:|"]
  for x in summary:lines.append(f"| {x['config']} | {x['bucket_count']} | {x['median_step_gpu_us']:.1f} | {x['median_backward_gpu_us']:.1f} | {x['median_logger_overlap_us']:.1f} |")
  (d/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(f"run_dir={d}",flush=True)
if __name__=="__main__":main()
