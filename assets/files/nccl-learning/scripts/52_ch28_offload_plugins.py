#!/usr/bin/env python3
"""Validate network plugin rejection/fallback and CollNet/NVLS eligibility gates."""
from __future__ import annotations
import argparse,csv,os,re,statistics,subprocess
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
ROW_RE=re.compile(r"^\s*\d+\s+")
ATTR_RE=re.compile(r"NVLS_ATTR gpu=(\d+) multicast=(\d+) cc=(\d+)")
CONFIGS=(("baseline",{}),("collnet_0",{"NCCL_COLLNET_ENABLE":"0"}),("collnet_1",{"NCCL_COLLNET_ENABLE":"1"}),
 ("nvls_0",{"NCCL_NVLS_ENABLE":"0"}),("nvls_auto",{"NCCL_NVLS_ENABLE":"2"}),
 ("bad_plugin",{"NCCL_NET_PLUGIN":"BAD"}),("missing_plugin",{"NCCL_NET_PLUGIN":"/definitely/missing/libnccl-net.so"}))
FORCED=("CollNetDirect","CollNetChain","NVLS","NVLSTree")
def clean():
  e=dict(os.environ)
  for k in tuple(e):
    if k.startswith("NCCL_") or k.startswith("TORCH_NCCL_") or k in ("LD_LIBRARY_PATH","LD_PRELOAD"):e.pop(k)
  return e
def base(extra=None,forced_net=False,subsys="INIT,NET"):
  e=clean();e.update({"NCCL_DEBUG":"INFO","NCCL_DEBUG_SUBSYS":subsys})
  if forced_net:e.update({"NCCL_P2P_DISABLE":"1","NCCL_SHM_DISABLE":"1","NCCL_ALGO":"Ring","NCCL_PROTO":"Simple"})
  if extra:e.update(extra)
  return e
def run(cmd,e,path,cwd,check=True,timeout=180):
  p=subprocess.run(cmd,cwd=cwd,env=e,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout)
  path.parent.mkdir(parents=True,exist_ok=True);path.write_text(p.stdout,encoding="utf-8")
  if check and p.returncode:raise RuntimeError(f"failed {p.returncode}: {path}\n{p.stdout[-1800:]}")
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
  d=a.root/"logs/ch28_offload_plugins"/a.run_id;raw=d/"raw";d.mkdir(parents=True,exist_ok=True)
  bad=a.root/"build/libnccl-net-ch28bad.so";cap=a.root/"build/ch28_nvls_capability"
  run(["gcc","-O2","-shared","-fPIC",str(a.root/"probes/ch28_bad_net_plugin.c"),"-o",str(bad)],clean(),raw/"compile_bad.log",a.root)
  run(["/usr/local/cuda/bin/nvcc","-O2",str(a.root/"probes/ch28_nvls_capability.cc"),"-o",str(cap),"-lcuda"],clean(),raw/"compile_cap.log",a.root)
  _,ct=run([str(cap)],clean(),raw/"nvls_capability.log",a.root);caps=[{"gpu":int(g),"multicast_attr":int(m),"compute_capability":int(cc),"nvs_nodes":0} for g,m,cc in ATTR_RE.findall(ct)]
  if len(caps)!=4 or any(x["multicast_attr"]!=0 or x["compute_capability"]!=70 for x in caps):raise RuntimeError(f"caps {caps}")
  outcsv(d/"nvls_capability.csv",caps)
  plugin=Path("/opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so")
  _,nm=run(["nm","-D",str(plugin)],clean(),raw/"plugin_symbols.log",a.root)
  syms=[]
  for kind,prefix in (("net","ncclNetPlugin_v"),("collnet","ncclCollNetPlugin_v")):
    for v in range(5,9):syms.append({"kind":kind,"version":v,"symbol":prefix+str(v),"exported":int(prefix+str(v) in nm)})
  if any(not x["exported"] for x in syms):raise RuntimeError("SHARP ABI symbols")
  outcsv(d/"plugin_symbols.csv",syms)
  probe=a.root/"build/ch24_proxy_socket";plugin_cases=[]
  for name,extra,signature in (("sharp_default",{},"P2P plugin IBext_v8"),("bad_abi",{"NCCL_NET_PLUGIN":str(bad)},"Failed to find ncclNetPlugin symbol (>= v5)"),("missing",{"NCCL_NET_PLUGIN":"/definitely/missing/libnccl-net.so"},"Could not find:")):
    _,text=run([str(probe),"--bytes",str(1<<20),"--iterations","1"],base(extra,True),raw/"plugin_runtime"/f"{name}.log",a.root)
    ok=int(signature in text and "Using network Socket" in text and "correct=1" in text)
    if not ok:raise RuntimeError(f"plugin case {name}")
    plugin_cases.append({"case":name,"library_opened":int(name!="missing"),"net_v8_accepted":int(name=="sharp_default"),"devices_usable":0,"internal_socket_fallback":1,"correct":1})
  outcsv(d/"plugin_runtime.csv",plugin_cases)
  tests=a.root/"third_party/nccl-tests";binary=tests/"build/all_reduce_perf";perf_cmd=[str(binary),"-b","4K","-e","64M","-f","16384","-g","4","-w","5","-n","10","-N",str(a.cycles),"-c","1","-I","0","-z","0","-u","0","-C","0","-a","3","-d","float","-o","sum"]
  perf=[];paths=[]
  for rep,order in ((1,CONFIGS),(2,tuple(reversed(CONFIGS)))):
    for name,extra0 in order:
      extra={k:(str(bad) if v=="BAD" else v) for k,v in extra0.items()}
      _,text=run(perf_cmd,base(extra,True),raw/"negative_controls"/f"{name}_r{rep}.log",tests)
      if "via NET/Socket/0" not in text:raise RuntimeError(f"Socket path {name}")
      perf+=parse(name,rep,text,a.cycles);paths.append({"config":name,"replicate":rep,"socket_connector":1,"collnet_connector":int("via COLLNET/" in text),"nvls_connected":int("Connected NVLS" in text),"correct":1});print(f"[negative] {name} r={rep} PASS",flush=True)
  if any(x["collnet_connector"] or x["nvls_connected"] for x in paths):raise RuntimeError("offload unexpectedly active")
  outcsv(d/"negative_control_measurements.csv",perf);outcsv(d/"negative_control_paths.csv",paths)
  pg=defaultdict(list)
  for x in perf:pg[(x["config"],x["size_bytes"])].append(x)
  ps=[{"config":k[0],"size_bytes":k[1],"samples":len(v),"median_time_us":statistics.median(x["time_us"] for x in v),"median_busbw_GBs":statistics.median(x["busbw_GBs"] for x in v)} for k,v in sorted(pg.items())]
  outcsv(d/"negative_control_summary.csv",ps)
  forced=[];one=[str(binary),"-b","1M","-e","1M","-g","4","-w","1","-n","1","-N","1","-c","1","-I","0","-z","0","-u","0","-C","0","-a","3","-d","float","-o","sum"]
  for algo in FORCED:
    extra={"NCCL_ALGO":algo,"NCCL_COLLNET_ENABLE":"1","NCCL_NVLS_ENABLE":"1"}
    _,text=run(one,base(extra,False,"INIT,TUNING"),raw/"forced_algorithms"/f"{algo}.log",tests)
    selected={int(x) for x in re.findall(r"1048576 Bytes -> Algo (\d+) proto",text)}
    if selected!={1}:raise RuntimeError(f"forced {algo} selected={selected}")
    forced.append({"requested":algo,"returncode":0,"observed_algorithm_id":1,"observed_algorithm":"Ring","fallback":1,"wrong":0})
  outcsv(d/"forced_algorithm_fallback.csv",forced)
  model=d/"offload_source_model.csv";run(["python3",str(a.root/"probes/ch28_offload_model.py"),"--output",str(model)],clean(),raw/"model.log",a.root)
  mr=list(csv.DictReader(model.open()));
  if len(mr)!=28 or any(x["pass"]!="1" for x in mr):raise RuntimeError("model")
  manifest=["experiment=ch28_offload_plugins",f"run_id={a.run_id}",f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",f"hostname={os.uname().nodename}",f"negative_control_rows={len(perf)}",f"source_model_rows={len(mr)}","correctness=PASS","plugin_symbols=8/8 PASS","plugin_runtime=3/3 PASS","forced_algorithm_fallback=4/4 PASS","nvls_multicast_attr=0/4","offload_paths=0/14","collnet_sharp_performance=HARDWARE_BLOCKED","nvls_performance=HARDWARE_BLOCKED","nccl_runtime=2.22.3","nccl_source_commit="+subprocess.check_output(["git","-C",str(a.root/"third_party/nccl-2.22.3"),"rev-parse","HEAD"],text=True).strip()]
  (d/"manifest.txt").write_text("\n".join(manifest)+"\n",encoding="utf-8")
  lines=["# Chapter 28 experiment summary","",f"- Negative-control rows: {len(perf)}","- Correctness: PASS","- SHARP Net/CollNet symbols: 8/8 PASS","- Plugin accept/reject/missing runtime: 3/3 PASS","- Forced unsupported algorithm fallback: 4/4 -> Ring","- NVLS multicast attr: 0/4","- Source model: 28/28 PASS","- CollNet/SHARP and NVLS performance: HARDWARE_BLOCKED"]
  (d/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(f"run_dir={d}")
if __name__=="__main__":main()
