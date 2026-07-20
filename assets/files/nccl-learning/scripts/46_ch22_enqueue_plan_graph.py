#!/usr/bin/env python3
"""Exercise NCCL enqueue tasks, grouped plans, work geometry, and CUDA Graphs."""
from __future__ import annotations
import argparse,csv,math,os,re,statistics,subprocess
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

RESULT_RE=re.compile(r"RESULT mode=(\w+) ops=(\d+) replays=(\d+) bytes=(\d+) host_us=([0-9.]+) total_us=([0-9.]+) graph_nodes=(\d+) correct=(\d+)")
WORK_RE=re.compile(r"Collective (\w+)\(([^,]+), ([^,]+), ([^,]+), ([^)]+)\) count=(\d+) devFuncId=(\d+) channel\{Lo..Hi\}=\{(\d+)..(\d+)\} count\{Lo,Mid,Hi\}=\{(\d+),(\d+),(\d+)\} chunkBytes\{Lo,Mid,Hi\}=\{(\d+),(\d+),(\d+)\}")
WIDTHS=(1,2,4,8,16)

def env_clean():
  e=dict(os.environ)
  for k in tuple(e):
    if k.startswith("NCCL_") or k.startswith("TORCH_NCCL_") or k=="LD_LIBRARY_PATH":e.pop(k)
  return e

def run(cmd,env,path,cwd,check=True):
  p=subprocess.run(cmd,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  path.parent.mkdir(parents=True,exist_ok=True);path.write_text(p.stdout,encoding="utf-8")
  if check and p.returncode:raise RuntimeError(f"failed {p.returncode}: {path}")
  return p.stdout

def write_csv(path,rows):
  if not rows:raise RuntimeError(f"empty {path}")
  with path.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def pct(v,q):
  a=sorted(v);p=(len(a)-1)*q;lo=math.floor(p);hi=math.ceil(p)
  return a[lo] if lo==hi else a[lo]+(a[hi]-a[lo])*(p-lo)

def parse_result(text,rep,sample):
  m=RESULT_RE.search(text)
  if not m:raise RuntimeError("missing RESULT")
  mode,ops,replays,size,host,total,nodes,correct=m.groups()
  if correct!="1":raise RuntimeError("correctness")
  return {"mode":mode,"ops":int(ops),"replays":int(replays),"bytes":int(size),
    "replicate":rep,"sample":sample,"host_us":float(host),"total_us":float(total),
    "graph_nodes":int(nodes),"correct":int(correct)}

def summarize(rows):
  g=defaultdict(list)
  for r in rows:g[(r["mode"],r["ops"],r["replays"],r["bytes"])].append(r)
  out=[]
  for key,a in sorted(g.items()):
    h=[r["host_us"] for r in a];t=[r["total_us"] for r in a]
    out.append({"mode":key[0],"ops":key[1],"replays":key[2],"bytes":key[3],
      "samples":len(a),"median_host_us":statistics.median(h),"p95_host_us":pct(h,.95),
      "median_total_us":statistics.median(t),"p95_total_us":pct(t,.95),
      "total_cv_percent":statistics.pstdev(t)/statistics.fmean(t)*100,
      "median_graph_nodes":statistics.median(r["graph_nodes"] for r in a)})
  return out

def parse_trace(case,text):
  rows=[]
  for line in text.splitlines():
    m=WORK_RE.search(line)
    if not m:continue
    x=m.groups()
    rows.append({"case":case,"work_index":len(rows),"function":x[0],"redop":x[1],
      "datatype":x[2],"algorithm":x[3],"protocol":x[4],"count":int(x[5]),
      "dev_func_id":int(x[6]),"channel_lo":int(x[7]),"channel_hi":int(x[8]),
      "count_lo":int(x[9]),"count_mid":int(x[10]),"count_hi":int(x[11]),
      "chunk_bytes_lo":int(x[12]),"chunk_bytes_mid":int(x[13]),"chunk_bytes_hi":int(x[14])})
  if len(rows)<5:raise RuntimeError(f"trace rows {case} {len(rows)}")
  return rows

def nsys_case(nsys,probe,mode,run_dir,cwd):
  replay=5 if mode=="graph" else 1;prefix=run_dir/"private"/f"nsys_{mode}"
  cmd=[str(nsys),"profile","--force-overwrite=true","--sample=none","--cpuctxsw=none",
    "--trace=cuda,nvtx","--cuda-graph-trace=node","-o",str(prefix),str(probe),
    "--mode",mode,"--ops","4","--bytes","1048576","--replays",str(replay)]
  run(cmd,env_clean(),run_dir/"raw/nsys"/f"{mode}_profile.log",cwd)
  report=Path(str(prefix)+".nsys-rep")
  nvtx=run([str(nsys),"stats","--force-export=true","--report","nvtx_gpu_proj_sum",
    "--format","csv",str(report)],env_clean(),run_dir/"private"/f"{mode}_nvtx.csv",cwd)
  gpu=run([str(nsys),"stats","--force-export=true","--report","cuda_gpu_trace",
    "--format","csv",str(report)],env_clean(),run_dir/"private"/f"{mode}_gpu.csv",cwd)
  line=next((x for x in nvtx.splitlines() if x.startswith(f":{mode}-ops4,")),None)
  if not line:raise RuntimeError(f"nvtx range {mode}")
  fields=next(csv.reader([line]));gpu_ops=int(fields[10])
  kernels=[next(csv.reader([x])) for x in gpu.splitlines() if "ncclDevKernel_" in x]
  scenario=kernels[-gpu_ops:]
  grids=sorted({int(x[3]) for x in scenario});blocks=sorted({int(x[6]) for x in scenario})
  expected={"sequential":16,"grouped":4,"graph":20}[mode]
  if gpu_ops!=expected:raise RuntimeError(f"{mode} gpu ops {gpu_ops}")
  return {"mode":mode,"ops":4,"replays":replay,"nvtx_gpu_ops":gpu_ops,
    "expected_gpu_ops":expected,"unique_grid_x":";".join(map(str,grids)),
    "unique_block_x":";".join(map(str,blocks)),"kernel_names":len({x[-1] for x in scenario}),
    "profile_private":1}

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("/root/nccl-learning"))
  ap.add_argument("--run-id",default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
  ap.add_argument("--samples",type=int,default=5);a=ap.parse_args()
  d=a.root/"logs/ch22_enqueue_plan"/a.run_id;raw=d/"raw";probe=a.root/"build/ch22_enqueue_plan_graph"
  (d/"private").mkdir(parents=True,exist_ok=True)
  src=a.root/"probes/ch22_enqueue_plan_graph.cc";trace=a.root/"build/nccl-trace/lib"
  run(["/usr/local/cuda/bin/nvcc","-std=c++17","-O2",str(src),"-o",str(probe),
    "-lnccl","-lnvToolsExt"],env_clean(),raw/"compile.log",a.root)

  trace_rows=[]
  for mode in ("sequential","grouped","graph"):
    e=env_clean();e.update({"LD_LIBRARY_PATH":str(trace),"NCCL_DEBUG":"TRACE","NCCL_DEBUG_SUBSYS":"COLL,TUNING"})
    text=run([str(probe),"--mode",mode,"--ops","4","--bytes","1048576","--replays","5"],
      e,raw/"trace"/f"{mode}.log",a.root)
    trace_rows.extend(parse_trace(mode,text))

  rows=[]
  configs=[(m,w,1) for m in ("sequential","grouped") for w in WIDTHS]
  configs += [("graph",w,r) for w in (1,4,16) for r in (1,5,20)]
  for rep,order in ((1,configs),(2,list(reversed(configs)))):
    for mode,ops,replays in order:
      for sample in range(a.samples):
        log=raw/"measurements"/f"{mode}_o{ops}_r{replays}_rep{rep}_s{sample}.log"
        text=log.read_text(encoding="utf-8") if log.exists() else run(
          [str(probe),"--mode",mode,"--ops",str(ops),"--bytes","1048576",
           "--replays",str(replays)],env_clean(),log,a.root)
        rows.append(parse_result(text,rep,sample))
      print(f"[measure] {mode} ops={ops} replay={replays} rep={rep} PASS",flush=True)

  nsys=Path(subprocess.check_output(["which","nsys"],text=True).strip())
  nsys_rows=[nsys_case(nsys,probe,m,d,a.root) for m in ("sequential","grouped","graph")]
  summary=summarize(rows)
  write_csv(d/"trace_work_geometry.csv",trace_rows);write_csv(d/"raw_measurements.csv",rows)
  write_csv(d/"performance_summary.csv",summary);write_csv(d/"nsys_summary.csv",nsys_rows)
  manifest=["experiment=ch22_enqueue_plan_graph",f"run_id={a.run_id}",
    f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",f"hostname={os.uname().nodename}",
    f"measurement_rows={len(rows)}",f"trace_work_rows={len(trace_rows)}",
    "correctness=PASS","nsys_acceptance=3/3 PASS","trace_build=ENABLE_TRACE",
    "private_nsys_reports=NOT_FOR_PUBLICATION","nccl_runtime=2.22.3",
    "nccl_source_commit="+subprocess.check_output(["git","-C",str(a.root/"third_party/nccl-2.22.3"),"rev-parse","HEAD"],text=True).strip()]
  (d/"manifest.txt").write_text("\n".join(manifest)+"\n",encoding="utf-8")
  s4=next(x for x in summary if x["mode"]=="sequential" and x["ops"]==4)
  g4=next(x for x in summary if x["mode"]=="grouped" and x["ops"]==4)
  lines=["# Chapter 22 experiment summary","",f"- Measurements: {len(rows)}","- Correctness: PASS","- Nsight acceptance: 3/3 PASS","",
    f"- sequential4 median total: {s4['median_total_us']:.3f} us",
    f"- grouped4 median total: {g4['median_total_us']:.3f} us",
    f"- grouped4 vs sequential4: {(g4['median_total_us']/s4['median_total_us']-1)*100:+.2f}%",
    "","| mode | GPU ops in NVTX range | gridX | blockX |","|---|---:|---|---|"]
  for x in nsys_rows:lines.append(f"| {x['mode']} | {x['nvtx_gpu_ops']} | {x['unique_grid_x']} | {x['unique_block_x']} |")
  (d/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
  print(f"run_dir={d}",flush=True)
if __name__=="__main__":main()
