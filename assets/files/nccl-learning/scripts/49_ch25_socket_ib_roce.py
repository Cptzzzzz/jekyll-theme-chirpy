#!/usr/bin/env python3
"""Validate Socket fallback and source-level IB/RoCE semantics without an HCA."""
from __future__ import annotations
import argparse,csv,os,re,statistics,subprocess
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

ROW_RE=re.compile(r"^\s*\d+\s+")
CONFIGS=(("auto",{}),("ib_disabled",{"NCCL_IB_DISABLE":"1"}),
  ("socket_forced",{"NCCL_NET":"Socket"}),("missing_hca",{"NCCL_IB_HCA":"definitely_missing"}))

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
  if check and p.returncode:raise RuntimeError(f"failed {p.returncode}: {path}\n{p.stdout[-1500:]}")
  return p.returncode,p.stdout
def csvout(path,rows):
  if not rows:raise RuntimeError(f"empty {path}")
  with path.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def perf_cmd(binary,cycles):
  return [str(binary),"-b","4K","-e","64M","-f","128","-g","4","-w","5","-n","10",
    "-N",str(cycles),"-c","1","-I","0","-z","0","-u","0","-C","0","-a","3","-d","float","-o","sum"]
def parse(config,rep,text,cycles):
  rows=[];seen=defaultdict(int)
  for line in text.splitlines():
    if not ROW_RE.match(line):continue
    f=line.split()
    if len(f)!=13 or int(f[0])<4096:continue
    size=int(f[0]);cycle=seen[size];seen[size]+=1
    rows.append({"config":config,"replicate":rep,"cycle":cycle,"size_bytes":size,
      "time_us":float(f[5]),"busbw_GBs":float(f[7]),"wrong":int(f[8]),"in_place_wrong":int(f[12])})
  if len(rows)!=3*cycles or set(seen.values())!={cycles}:raise RuntimeError(f"parse {config}: {len(rows)} {seen}")
  if any(r["wrong"] or r["in_place_wrong"] for r in rows):raise RuntimeError("correctness")
  return rows
def main():
  ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("/root/nccl-learning"))
  ap.add_argument("--run-id",default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"));ap.add_argument("--cycles",type=int,default=5);a=ap.parse_args()
  d=a.root/"logs/ch25_socket_ib_roce"/a.run_id;raw=d/"raw";d.mkdir(parents=True,exist_ok=True)
  inv=[]
  for name,cmd in (("ibv_devinfo",["ibv_devinfo"]),("dev_infiniband",["bash","-lc","ls -la /dev/infiniband"]),
      ("sys_class_infiniband",["bash","-lc","find -L /sys/class/infiniband -maxdepth 2 -printf '%p -> %l\\n'"]),
      ("loaded_modules",["bash","-lc","lsmod | grep -E 'mlx|ib_core|rdma' || true"])):
    rc,text=run(cmd,clean(),raw/"inventory"/f"{name}.log",a.root,False)
    inv.append({"check":name,"returncode":rc,"first_line":next((x for x in text.splitlines() if x.strip()),"EMPTY"),"hca_usable":0})
  csvout(d/"hardware_inventory.csv",inv)

  binary=a.root/"third_party/nccl-tests/build/all_reduce_perf";rows=[];paths=[];cmd=perf_cmd(binary,a.cycles)
  for rep,order in ((1,CONFIGS),(2,tuple(reversed(CONFIGS)))):
    for name,extra in order:
      _,text=run(cmd,env(extra),raw/"fallback"/f"{name}_r{rep}.log",binary.parent.parent)
      if "Using network Socket" not in text or "via NET/Socket" not in text:raise RuntimeError(f"Socket path absent {name}")
      rows+=parse(name,rep,text,a.cycles)
      paths.append({"config":name,"replicate":rep,"socket_selected":1,
        "ib_no_device":int("NET/IB : No device found" in text),"ib_disabled":int(name=="ib_disabled"),
        "correct":1});print(f"[fallback] {name} r={rep} PASS",flush=True)
  csvout(d/"raw_measurements.csv",rows);csvout(d/"backend_paths.csv",paths)
  g=defaultdict(list)
  for r in rows:g[(r["config"],r["size_bytes"])].append(r)
  summary=[{"config":k[0],"size_bytes":k[1],"samples":len(v),
    "median_time_us":statistics.median(x["time_us"] for x in v),
    "median_busbw_GBs":statistics.median(x["busbw_GBs"] for x in v)} for k,v in sorted(g.items())]
  csvout(d/"performance_summary.csv",summary)

  probe=a.root/"build/ch24_proxy_socket";ifaces=[]
  for name,ifname,ok in (("eth0","eth0",1),("loopback","lo",1),("missing","definitely_missing",0)):
    rc,text=run([str(probe),"--bytes",str(1<<20),"--iterations","1"],env({"NCCL_SOCKET_IFNAME":ifname}),raw/"interfaces"/f"{name}.log",a.root,False)
    passed=int((rc==0)==bool(ok) and (("correct=1" in text) if ok else True))
    if not passed:raise RuntimeError(f"interface case {name} rc={rc}")
    ifaces.append({"case":name,"ifname":ifname,"expected_success":ok,"returncode":rc,
      "socket_path":int("NET/Socket" in text),"failure_signature":next((x.strip() for x in text.splitlines() if "no socket interface" in x.lower() or "Bootstrap" in x and "no" in x),""),"pass":passed})
  csvout(d/"interface_cases.csv",ifaces)
  model=d/"ib_source_model.csv"
  run(["python3",str(a.root/"probes/ch25_ib_semantics_model.py"),"--output",str(model)],clean(),raw/"source_model.log",a.root)
  modelrows=list(csv.DictReader(model.open()));
  if not modelrows or any(x["pass"]!="1" for x in modelrows):raise RuntimeError("source model")
  manifest=["experiment=ch25_socket_ib_roce",f"run_id={a.run_id}",f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
    f"hostname={os.uname().nodename}",f"performance_rows={len(rows)}",f"path_rows={len(paths)}",f"source_model_rows={len(modelrows)}",
    "correctness=PASS","socket_fallback=8/8 PASS","interface_cases=3/3 PASS","hca_usable=NO",
    "ib_roce_performance=HARDWARE_BLOCKED","nccl_runtime=2.22.3",
    "nccl_source_commit="+subprocess.check_output(["git","-C",str(a.root/"third_party/nccl-2.22.3"),"rev-parse","HEAD"],text=True).strip()]
  (d/"manifest.txt").write_text("\n".join(manifest)+"\n",encoding="utf-8")
  lines=["# Chapter 25 experiment summary","",f"- Performance rows: {len(rows)}","- Correctness: PASS",
    "- Socket fallback: 8/8 PASS","- Interface cases: 3/3 PASS",f"- IB source model: {len(modelrows)}/{len(modelrows)} PASS",
    "- Usable HCA: NO","- IB/RoCE performance: HARDWARE_BLOCKED"]
  (d/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
  print(f"run_dir={d}")
if __name__=="__main__":main()
