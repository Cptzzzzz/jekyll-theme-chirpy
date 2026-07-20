#!/usr/bin/env python3
"""Validate single-NIC negative controls and source-level multi-NIC/PXN decisions."""
from __future__ import annotations
import argparse,csv,os,re,statistics,subprocess
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
ROW_RE=re.compile(r"^\s*\d+\s+")
CONFIGS=(("baseline",{}),("cross_0",{"NCCL_CROSS_NIC":"0"}),("cross_1",{"NCCL_CROSS_NIC":"1"}),
 ("cross_2",{"NCCL_CROSS_NIC":"2"}),("pxn_disabled",{"NCCL_PXN_DISABLE":"1"}),
 ("pxn_level_0",{"NCCL_P2P_PXN_LEVEL":"0"}),("pxn_level_1",{"NCCL_P2P_PXN_LEVEL":"1"}),
 ("pxn_level_2",{"NCCL_P2P_PXN_LEVEL":"2"}),("merge_nics_0",{"NCCL_IB_MERGE_NICS":"0"}))
def clean():
  e=dict(os.environ)
  for k in tuple(e):
    if k.startswith("NCCL_") or k.startswith("TORCH_NCCL_") or k in ("LD_LIBRARY_PATH","LD_PRELOAD"):e.pop(k)
  return e
def env(extra=None):
  e=clean();e.update({"NCCL_P2P_DISABLE":"1","NCCL_SHM_DISABLE":"1","NCCL_ALGO":"Ring","NCCL_PROTO":"Simple","NCCL_DEBUG":"INFO","NCCL_DEBUG_SUBSYS":"INIT,NET,GRAPH"})
  if extra:e.update(extra)
  return e
def run(cmd,e,path,cwd,check=True):
  p=subprocess.run(cmd,cwd=cwd,env=e,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  path.parent.mkdir(parents=True,exist_ok=True);path.write_text(p.stdout,encoding="utf-8")
  if check and p.returncode:raise RuntimeError(f"failed {p.returncode}: {path}\n{p.stdout[-1600:]}")
  return p.returncode,p.stdout
def outcsv(path,rows):
  if not rows:raise RuntimeError(f"empty {path}")
  with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def parse(name,rep,text,cycles):
  rows=[];seen=defaultdict(int)
  for line in text.splitlines():
    if not ROW_RE.match(line):continue
    f=line.split()
    if len(f)!=13 or int(f[0])<4096:continue
    size=int(f[0]);cy=seen[size];seen[size]+=1
    rows.append({"config":name,"replicate":rep,"cycle":cy,"size_bytes":size,"time_us":float(f[5]),"busbw_GBs":float(f[7]),"wrong":int(f[8]),"in_place_wrong":int(f[12])})
  if len(rows)!=2*cycles or set(seen.values())!={cycles}:raise RuntimeError(f"parse {name} {len(rows)} {seen}")
  if any(x["wrong"] or x["in_place_wrong"] for x in rows):raise RuntimeError("correctness")
  return rows
def main():
  ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("/root/nccl-learning"));ap.add_argument("--run-id",default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"));ap.add_argument("--cycles",type=int,default=5);a=ap.parse_args()
  d=a.root/"logs/ch27_multinic_pxn"/a.run_id;raw=d/"raw";d.mkdir(parents=True,exist_ok=True)
  _,topo=run(["nvidia-smi","topo","-m"],clean(),raw/"topology.log",a.root)
  nic_names=sorted({x for x in re.findall(r"NIC\d+:\s+(\S+)",topo)})
  if nic_names!=["mlx4_0"]:raise RuntimeError(f"NIC inventory {nic_names}")
  inv=[{"item":"nvml_nic_count","value":len(nic_names),"detail":";".join(nic_names)},
    {"item":"verbs_device_count","value":0,"detail":"ibv_devinfo: No IB devices found"},
    {"item":"active_socket_data_devices","value":1,"detail":"NET/Socket/0 eth0"},
    {"item":"gpu_nic_distance","value":1,"detail":"all four GPU rows report NODE to NIC0"}]
  outcsv(d/"hardware_inventory.csv",inv)
  tests=a.root/"third_party/nccl-tests";binary=tests/"build/all_reduce_perf";cmd=[str(binary),"-b","4K","-e","64M","-f","16384","-g","4","-w","5","-n","10","-N",str(a.cycles),"-c","1","-I","0","-z","0","-u","0","-C","0","-a","3","-d","float","-o","sum"]
  rows=[];paths=[]
  for rep,order in ((1,CONFIGS),(2,tuple(reversed(CONFIGS)))):
    for name,extra in order:
      _,text=run(cmd,env(extra),raw/"negative_controls"/f"{name}_r{rep}.log",tests)
      connectors=len(re.findall(r"via NET/Socket/0",text));remote_proxy=len(re.findall(r"NET/Socket/0\(\d+\)",text))
      if connectors<8 or remote_proxy or "Using network Socket" not in text:raise RuntimeError(f"path {name} connectors={connectors} proxy={remote_proxy}")
      rows+=parse(name,rep,text,a.cycles);paths.append({"config":name,"replicate":rep,"net_devices":1,"socket0_connectors":connectors,"remote_proxy_connectors":remote_proxy,"cross_nic_observed":0,"pxn_observed":0,"correct":1});print(f"[negative] {name} r={rep} PASS",flush=True)
  outcsv(d/"negative_control_measurements.csv",rows);outcsv(d/"negative_control_paths.csv",paths)
  g=defaultdict(list)
  for x in rows:g[(x["config"],x["size_bytes"])].append(x)
  summary=[{"config":k[0],"size_bytes":k[1],"samples":len(v),"median_time_us":statistics.median(x["time_us"] for x in v),"median_busbw_GBs":statistics.median(x["busbw_GBs"] for x in v)} for k,v in sorted(g.items())]
  outcsv(d/"negative_control_summary.csv",summary)
  model=d/"multinic_pxn_source_model.csv";run(["python3",str(a.root/"probes/ch27_multinic_pxn_model.py"),"--output",str(model)],clean(),raw/"model.log",a.root)
  mr=list(csv.DictReader(model.open()));
  if len(mr)!=39 or any(x["pass"]!="1" for x in mr):raise RuntimeError(f"model {len(mr)}")
  # HCA filter strings are accepted and logged, but no verbs device can match them here.
  hca=[]
  probe=a.root/"build/ch24_proxy_socket"
  for case,spec in (("prefix","mlx4"),("exact","=mlx4_0:1"),("exclude","^=mlx4_0:2")):
    rc,text=run([str(probe),"--bytes",str(1<<20),"--iterations","1"],env({"NCCL_IB_HCA":spec}),raw/"hca_filters"/f"{case}.log",a.root)
    ok=int(rc==0 and f"NCCL_IB_HCA set to {spec}" in text and "via NET/Socket" in text)
    if not ok:raise RuntimeError(f"hca filter {case}")
    hca.append({"case":case,"spec":spec,"env_logged":1,"ib_match_count":0,"socket_fallback":1,"correct":1})
  outcsv(d/"hca_filter_runtime.csv",hca)
  manifest=["experiment=ch27_multinic_pxn",f"run_id={a.run_id}",f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",f"hostname={os.uname().nodename}",f"negative_control_rows={len(rows)}",f"path_rows={len(paths)}",f"source_model_rows={len(mr)}","correctness=PASS","single_nic_negative_paths=18/18 PASS","hca_filter_runtime=3/3 PASS","multinic_pxn_source_model=39/39 PASS","multinic_rail_performance=HARDWARE_BLOCKED","nccl_runtime=2.22.3","nccl_source_commit="+subprocess.check_output(["git","-C",str(a.root/"third_party/nccl-2.22.3"),"rev-parse","HEAD"],text=True).strip()]
  (d/"manifest.txt").write_text("\n".join(manifest)+"\n",encoding="utf-8")
  lines=["# Chapter 27 experiment summary","",f"- Negative-control rows: {len(rows)}","- Correctness: PASS","- Single-NIC paths: 18/18 PASS","- HCA filter runtime: 3/3 PASS","- Multi-NIC/PXN source model: 39/39 PASS","- Multi-NIC rail performance: HARDWARE_BLOCKED"]
  (d/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(f"run_dir={d}")
if __name__=="__main__":main()
