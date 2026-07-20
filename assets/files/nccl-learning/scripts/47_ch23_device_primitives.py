#!/usr/bin/env python3
"""Measure NCCL device protocol geometry and validate synchronization models."""
from __future__ import annotations
import argparse,csv,math,os,re,statistics,subprocess
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

BASE=tuple((p,c,None) for p in ("LL","LL128","Simple") for c in (1,4,12))
BUFF=(("Simple",4,65536),("Simple",4,1048576),("Simple",4,4194304))
CONFIGS=BASE+BUFF
ROW_RE=re.compile(r"^\s*\d+\s+")
WORK_RE=re.compile(r"Collective AllReduce\(Sum, ncclFloat32, RING, (\w+)\) count=(\d+) devFuncId=(\d+) channel\{Lo..Hi\}=\{(\d+)..(\d+)\} count\{Lo,Mid,Hi\}=\{(\d+),(\d+),(\d+)\} chunkBytes\{Lo,Mid,Hi\}=\{(\d+),(\d+),(\d+)\}")

def name(x):
  p,c,b=x;return f"{p}_c{c}_"+("default" if b is None else f"b{b}")
def clean():
  e=dict(os.environ)
  for k in tuple(e):
    if k.startswith("NCCL_") or k.startswith("TORCH_NCCL_") or k=="LD_LIBRARY_PATH":e.pop(k)
  return e
def cfg_env(x,debug="WARN",trace=None):
  p,c,b=x;e=clean();e.update({"NCCL_ALGO":"Ring","NCCL_PROTO":p,
    "NCCL_MIN_NCHANNELS":str(c),"NCCL_MAX_NCHANNELS":str(c),"NCCL_DEBUG":debug})
  if b is not None:e["NCCL_BUFFSIZE"]=str(b)
  if trace:e.update({"LD_LIBRARY_PATH":str(trace),"NCCL_DEBUG_SUBSYS":"COLL,TUNING"})
  return e
def run(cmd,e,path,cwd):
  q=subprocess.run(cmd,cwd=cwd,env=e,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  path.parent.mkdir(parents=True,exist_ok=True);path.write_text(q.stdout,encoding="utf-8")
  if q.returncode:raise RuntimeError(f"failed {q.returncode}: {path}")
  return q.stdout
def write_csv(path,rows):
  if not rows:raise RuntimeError(f"empty {path}")
  with path.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def percentile(v,q):
  a=sorted(v);p=(len(a)-1)*q;lo=math.floor(p);hi=math.ceil(p)
  return a[lo] if lo==hi else a[lo]+(a[hi]-a[lo])*(p-lo)
def command(binary,cycles,iters):
  return [str(binary),"-b","4K","-e","64M","-f","128","-g","4","-w","5",
    "-n",str(iters),"-N",str(cycles),"-c","1","-I","0","-z","0","-u","0",
    "-C","0","-a","3","-d","float","-o","sum"]
def parse_perf(config,rep,text,cycles):
  rows=[];seen=defaultdict(int)
  for line in text.splitlines():
    if not ROW_RE.match(line):continue
    f=line.split()
    if len(f)!=13 or int(f[0])<4096:continue
    size=int(f[0]);cycle=seen[size];seen[size]+=1
    rows.append({"config":config,"replicate":rep,"cycle":cycle,"size_bytes":size,
      "time_us":float(f[5]),"algbw_GBs":float(f[6]),"busbw_GBs":float(f[7]),
      "wrong":int(f[8]),"in_place_time_us":float(f[9]),
      "in_place_busbw_GBs":float(f[11]),"in_place_wrong":int(f[12])})
  if len(rows)!=3*cycles or set(seen.values())!={cycles}:raise RuntimeError(f"parse {config} {len(rows)} {seen}")
  if any(r["wrong"] or r["in_place_wrong"] for r in rows):raise RuntimeError(f"correctness {config}")
  return rows
def summary(rows):
  g=defaultdict(list)
  for r in rows:g[(r["config"],r["size_bytes"])].append(r)
  out=[]
  for (c,s),a in sorted(g.items()):
    t=[r["time_us"] for r in a]
    out.append({"config":c,"size_bytes":s,"samples":len(a),
      "median_time_us":statistics.median(t),"p95_time_us":percentile(t,.95),
      "cycle_cv_percent":statistics.pstdev(t)/statistics.fmean(t)*100,
      "median_busbw_GBs":statistics.median(r["busbw_GBs"] for r in a)})
  return out
def parse_work(config,x,text):
  matches=[WORK_RE.search(line) for line in text.splitlines()];matches=[m for m in matches if m]
  if len(matches)<2:raise RuntimeError(f"work trace {config} {len(matches)}")
  m=matches[-1];p,c,b=x;g=m.groups()
  row={"config":config,"requested_protocol":p,"requested_channels":c,
    "buffer_bytes":b or 0,"observed_protocol":g[0],"count":int(g[1]),
    "dev_func_id":int(g[2]),"channel_lo":int(g[3]),"channel_hi":int(g[4]),
    "count_lo":int(g[5]),"count_mid":int(g[6]),"count_hi":int(g[7]),
    "chunk_bytes_lo":int(g[8]),"chunk_bytes_mid":int(g[9]),"chunk_bytes_hi":int(g[10])}
  if row["observed_protocol"].lower()!=p.lower() or row["channel_hi"]-row["channel_lo"]+1!=c:raise RuntimeError(f"geometry {row}")
  return row
def nsys_case(nsys,probe,x,d,root):
  config=name(x);prefix=d/"private"/f"nsys_{config}"
  run([str(nsys),"profile","--force-overwrite=true","--sample=none","--cpuctxsw=none",
    "--trace=cuda,nvtx","-o",str(prefix),str(probe),"--mode","grouped","--ops","1",
    "--bytes",str(64<<20),"--replays","1"],cfg_env(x),d/"raw/nsys"/f"{config}.log",root)
  rep=Path(str(prefix)+".nsys-rep")
  gpu=run([str(nsys),"stats","--force-export=true","--report","cuda_gpu_trace",
    "--format","csv",str(rep)],clean(),d/"private"/f"{config}_gpu.csv",root)
  kernels=[next(csv.reader([line])) for line in gpu.splitlines() if "ncclDevKernel_" in line][-4:]
  if len(kernels)!=4:raise RuntimeError(f"nsys kernels {config}")
  return {"config":config,"protocol":x[0],"channels":x[1],
    "kernel_instances":len(kernels),"median_duration_us":statistics.median(float(k[1]) for k in kernels)/1000,
    "grid_x":";".join(sorted({k[3] for k in kernels})),"block_x":";".join(sorted({k[6] for k in kernels})),
    "registers_per_thread":";".join(sorted({k[9] for k in kernels})),
    "static_smem_MB":";".join(sorted({k[10] for k in kernels})),
    "dynamic_smem_MB":";".join(sorted({k[11] for k in kernels})),
    "kernel_name_count":len({k[-1] for k in kernels}),"profile_private":1}
def main():
  ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("/root/nccl-learning"))
  ap.add_argument("--run-id",default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
  ap.add_argument("--cycles",type=int,default=10);ap.add_argument("--iterations",type=int,default=20);a=ap.parse_args()
  d=a.root/"logs/ch23_device_primitives"/a.run_id;(d/"private").mkdir(parents=True,exist_ok=True)
  tests=a.root/"third_party/nccl-tests";binary=tests/"build/all_reduce_perf"
  probe=a.root/"build/ch22_enqueue_plan_graph";trace=a.root/"build/nccl-trace/lib"
  if not probe.exists() or not trace.exists():raise RuntimeError("chapter22 probe/TRACE build required")
  work=[]
  for x in CONFIGS:
    config=name(x);text=run([str(probe),"--mode","grouped","--ops","1","--bytes",str(64<<20),"--replays","1"],
      cfg_env(x,"TRACE",trace),d/"raw/trace"/f"{config}.log",a.root)
    work.append(parse_work(config,x,text));print(f"[trace] {config} PASS",flush=True)
  perf=[];cmd=command(binary,a.cycles,a.iterations)
  for rep,order in ((1,CONFIGS),(2,tuple(reversed(CONFIGS)))):
    for x in order:
      config=name(x);text=run(cmd,cfg_env(x),d/"raw/performance"/f"{config}_r{rep}.log",tests)
      perf+=parse_perf(config,rep,text,a.cycles);print(f"[perf] {config} r={rep} PASS",flush=True)
  nsys=Path(subprocess.check_output(["which","nsys"],text=True).strip())
  nrows=[]
  for x in BASE:
    nrows.append(nsys_case(nsys,probe,x,d,a.root));print(f"[nsys] {name(x)} PASS",flush=True)
  model=a.root/"probes/ch23_protocol_state_model.py"
  run(["python3",str(model),"--output",str(d/"protocol_state_model.csv")],clean(),d/"raw/model.log",a.root)
  modelrows=list(csv.DictReader((d/"protocol_state_model.csv").open()))
  if len(modelrows)!=27 or any(r["pass"]!="1" for r in modelrows):raise RuntimeError("model acceptance")
  ps=summary(perf);write_csv(d/"work_geometry.csv",work);write_csv(d/"raw_measurements.csv",perf)
  write_csv(d/"performance_summary.csv",ps);write_csv(d/"nsys_kernel_geometry.csv",nrows)
  manifest=["experiment=ch23_device_primitives",f"run_id={a.run_id}",
    f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",f"hostname={os.uname().nodename}",
    f"measurement_rows={len(perf)}",f"work_geometry_rows={len(work)}",
    f"nsys_cases={len(nrows)}",f"model_rows={len(modelrows)}","correctness=PASS",
    "trace_acceptance=12/12 PASS","nsys_acceptance=9/9 PASS","model_acceptance=27/27 PASS",
    "private_nsys_reports=NOT_FOR_PUBLICATION","nccl_runtime=2.22.3",
    "nccl_source_commit="+subprocess.check_output(["git","-C",str(a.root/"third_party/nccl-2.22.3"),"rev-parse","HEAD"],text=True).strip()]
  (d/"manifest.txt").write_text("\n".join(manifest)+"\n",encoding="utf-8")
  by64={(r["config"],int(r["size_bytes"])):r for r in ps}
  lines=["# Chapter 23 experiment summary","",f"- Performance rows: {len(perf)}","- Correctness: PASS",
    "- TRACE geometry: 12/12 PASS","- Nsight: 9/9 PASS","- State model: 27/27 PASS","",
    "| config | 64 MiB us | busbw GB/s |","|---|---:|---:|"]
  for x in CONFIGS:
    r=by64[(name(x),64<<20)];lines.append(f"| {name(x)} | {r['median_time_us']:.3f} | {r['median_busbw_GBs']:.2f} |")
  (d/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
  print(f"run_dir={d}",flush=True)
if __name__=="__main__":main()
