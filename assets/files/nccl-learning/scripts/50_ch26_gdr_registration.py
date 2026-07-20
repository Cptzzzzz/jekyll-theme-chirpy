#!/usr/bin/env python3
"""Validate GDR capability gates, registration semantics, and Socket negative controls."""
from __future__ import annotations
import argparse,csv,os,re,statistics,subprocess
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

ROW_RE=re.compile(r"^\s*\d+\s+")
ATTR_RE=re.compile(r"CUDA_ATTR gpu=(\d+) gdr=(\d+) dmabuf=(-?\d+)")
TIME_RE=re.compile(r"TIMING bytes=(\d+) sample=(\d+) reg_us=([0-9.]+) dereg_us=([0-9.]+) handle_nonnull=(\d+)")
CONFIGS=(("baseline",{}),("gdr_level_loc",{"NCCL_NET_GDR_LEVEL":"LOC"}),
  ("gdr_level_sys",{"NCCL_NET_GDR_LEVEL":"SYS"}),("gdr_read_0",{"NCCL_NET_GDR_READ":"0"}),
  ("gdr_read_1",{"NCCL_NET_GDR_READ":"1"}),("dmabuf_0",{"NCCL_DMABUF_ENABLE":"0"}),
  ("dmabuf_1",{"NCCL_DMABUF_ENABLE":"1"}))
def clean():
  e=dict(os.environ)
  for k in tuple(e):
    if k.startswith("NCCL_") or k.startswith("TORCH_NCCL_") or k in ("LD_LIBRARY_PATH","LD_PRELOAD"):e.pop(k)
  return e
def env(extra=None,debug="INFO"):
  e=clean();e.update({"NCCL_P2P_DISABLE":"1","NCCL_SHM_DISABLE":"1","NCCL_ALGO":"Ring",
    "NCCL_PROTO":"Simple","NCCL_DEBUG":debug,"NCCL_DEBUG_SUBSYS":"INIT,NET"})
  if extra:e.update(extra)
  return e
def run(cmd,e,path,cwd,check=True):
  p=subprocess.run(cmd,cwd=cwd,env=e,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  path.parent.mkdir(parents=True,exist_ok=True);path.write_text(p.stdout,encoding="utf-8")
  if check and p.returncode:raise RuntimeError(f"failed {p.returncode}: {path}\n{p.stdout[-1800:]}")
  return p.returncode,p.stdout
def outcsv(path,rows):
  if not rows:raise RuntimeError(f"empty {path}")
  with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def parse_perf(config,rep,text,cycles):
  rows=[];seen=defaultdict(int)
  for line in text.splitlines():
    if not ROW_RE.match(line):continue
    f=line.split()
    if len(f)!=13 or int(f[0])<4096:continue
    size=int(f[0]);cy=seen[size];seen[size]+=1
    rows.append({"config":config,"replicate":rep,"cycle":cy,"size_bytes":size,"time_us":float(f[5]),
      "busbw_GBs":float(f[7]),"wrong":int(f[8]),"in_place_wrong":int(f[12])})
  if len(rows)!=2*cycles or set(seen.values())!={cycles}:raise RuntimeError(f"parse {config} {len(rows)} {seen}")
  if any(x["wrong"] or x["in_place_wrong"] for x in rows):raise RuntimeError("correctness")
  return rows
def main():
  ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("/root/nccl-learning"));ap.add_argument("--run-id",default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"));ap.add_argument("--cycles",type=int,default=5);ap.add_argument("--samples",type=int,default=20);a=ap.parse_args()
  d=a.root/"logs/ch26_gdr_registration"/a.run_id;raw=d/"raw";d.mkdir(parents=True,exist_ok=True)
  probe=a.root/"build/ch26_gdr_registration"
  run(["/usr/local/cuda/bin/nvcc","-std=c++17","-O2",str(a.root/"probes/ch26_gdr_registration.cc"),"-o",str(probe),"-lnccl","-lcuda"],clean(),raw/"compile.log",a.root)
  _,attrs=run([str(probe),"attrs"],clean(),raw/"attrs.log",a.root)
  caps=[{"gpu":int(g),"cuda_gdr_attr":int(x),"cuda_dmabuf_attr":int(b)} for g,x,b in ATTR_RE.findall(attrs)]
  if len(caps)!=4 or any(x["cuda_gdr_attr"]!=1 or x["cuda_dmabuf_attr"]!=0 for x in caps):raise RuntimeError(f"attributes {caps}")
  checks=(("dev_infiniband",Path("/dev/infiniband").exists()),("dev_gdrdrv",Path("/dev/gdrdrv").exists()),
    ("peermem_legacy",Path("/sys/kernel/mm/memory_peers/nv_mem/version").exists()),("peermem_new",Path("/sys/kernel/mm/memory_peers/nv_mem_nc/version").exists()),
    ("module_nvidia_peermem",Path("/sys/module/nvidia_peermem").exists()),("module_mlx4_ib",Path("/sys/module/mlx4_ib").exists()),
    ("sysfs_mlx4_0",Path("/sys/class/infiniband/mlx4_0").exists()),("libgdrapi",Path("/usr/lib/x86_64-linux-gnu/libgdrapi.so").exists()))
  inv=[]
  for c in caps:inv.append({"layer":"cuda_gpu", "item":f"gpu{c['gpu']}","available":c["cuda_gdr_attr"],"detail":f"gdr_attr={c['cuda_gdr_attr']};dmabuf_attr={c['cuda_dmabuf_attr']}"})
  for name,value in checks:inv.append({"layer":"host_container","item":name,"available":int(value),"detail":"present" if value else "absent"})
  inv.append({"layer":"end_to_end","item":"gdrdma_usable","available":0,"detail":"no /dev/infiniband and no peermem/DMA-BUF path"})
  outcsv(d/"capability_inventory.csv",inv)

  _,cache=run([str(probe),"cache"],env(),raw/"registration/cache.log",a.root)
  required=("exact_same=1","contained_same=1","disjoint_distinct=1","handles_nonnull=1","stale_is_invalid_usage=1","on 0 net devices")
  if any(x not in cache for x in required):raise RuntimeError("cache semantics")
  _,disabled=run([str(probe),"disabled"],env({"NCCL_LOCAL_REGISTER":"0"}),raw/"registration/disabled.log",a.root)
  if "result=0 handle_null=1" not in disabled:raise RuntimeError("disabled semantics")
  _,timing=run([str(probe),"timing",str(a.samples)],env(),raw/"registration/timing.log",a.root)
  times=[{"bytes":int(b),"sample":int(s),"reg_us":float(r),"dereg_us":float(u),"handle_nonnull":int(h),"backend_net_devices":0} for b,s,r,u,h in TIME_RE.findall(timing)]
  if len(times)!=3*a.samples or any(not x["handle_nonnull"] for x in times):raise RuntimeError(f"timings {len(times)}")
  outcsv(d/"registration_timings.csv",times)
  tg=defaultdict(list)
  for x in times:tg[x["bytes"]].append(x)
  tsum=[{"bytes":b,"samples":len(v),"median_reg_us":statistics.median(x["reg_us"] for x in v),"median_dereg_us":statistics.median(x["dereg_us"] for x in v),"actual_net_registrations":0} for b,v in sorted(tg.items())]
  outcsv(d/"registration_summary.csv",tsum)

  tests=a.root/"third_party/nccl-tests";binary=tests/"build/all_reduce_perf";cmd=[str(binary),"-b","4K","-e","64M","-f","16384","-g","4","-w","5","-n","10","-N",str(a.cycles),"-c","1","-I","0","-z","0","-u","0","-C","0","-a","3","-d","float","-o","sum"]
  perf=[];paths=[]
  for rep,order in ((1,CONFIGS),(2,tuple(reversed(CONFIGS)))):
    for name,extra in order:
      _,text=run(cmd,env(extra),raw/"negative_controls"/f"{name}_r{rep}.log",tests)
      if "GPU Direct RDMA Disabled for HCA" not in text or "/GDRDMA" in text or "via NET/Socket" not in text:raise RuntimeError(f"path {name}")
      perf+=parse_perf(name,rep,text,a.cycles);paths.append({"config":name,"replicate":rep,"gdr_disabled_log":1,"gdrdma_connector":0,"socket_connector":1,"correct":1});print(f"[negative] {name} r={rep} PASS",flush=True)
  outcsv(d/"negative_control_measurements.csv",perf);outcsv(d/"negative_control_paths.csv",paths)
  pg=defaultdict(list)
  for x in perf:pg[(x["config"],x["size_bytes"])].append(x)
  ps=[{"config":k[0],"size_bytes":k[1],"samples":len(v),"median_time_us":statistics.median(x["time_us"] for x in v),"median_busbw_GBs":statistics.median(x["busbw_GBs"] for x in v)} for k,v in sorted(pg.items())]
  outcsv(d/"negative_control_summary.csv",ps)
  model=d/"gdr_mr_source_model.csv";run(["python3",str(a.root/"probes/ch26_gdr_mr_model.py"),"--output",str(model)],clean(),raw/"model.log",a.root)
  mr=list(csv.DictReader(model.open()));
  if len(mr)!=30 or any(x["pass"]!="1" for x in mr):raise RuntimeError("model")
  manifest=["experiment=ch26_gdr_registration",f"run_id={a.run_id}",f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",f"hostname={os.uname().nodename}",
    f"negative_control_rows={len(perf)}",f"registration_timing_rows={len(times)}",f"source_model_rows={len(mr)}","correctness=PASS","cuda_gdr_attr=4/4",
    "cuda_dmabuf_attr=0/4","registration_cache=PASS","local_register_disabled=PASS","gdr_negative_paths=14/14 PASS","end_to_end_gdrdma=HARDWARE_BLOCKED",
    "nccl_runtime=2.22.3","nccl_source_commit="+subprocess.check_output(["git","-C",str(a.root/"third_party/nccl-2.22.3"),"rev-parse","HEAD"],text=True).strip()]
  (d/"manifest.txt").write_text("\n".join(manifest)+"\n",encoding="utf-8")
  lines=["# Chapter 26 experiment summary","",f"- Negative-control rows: {len(perf)}",f"- Registration timing rows: {len(times)}","- Correctness: PASS","- CUDA GDR attribute: 4/4 enabled","- CUDA DMA-BUF attribute: 0/4","- Registration cache semantics: PASS","- GDR negative connector paths: 14/14 PASS","- Source model: 30/30 PASS","- End-to-end GDRDMA: HARDWARE_BLOCKED"]
  (d/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(f"run_dir={d}")
if __name__=="__main__":main()
