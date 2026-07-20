#!/usr/bin/env python3
"""Validate NCCL proxy, Socket helper, completion, and CPU-progress behavior."""
from __future__ import annotations
import argparse,csv,math,os,re,statistics,subprocess,time
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path

RESULT_RE=re.compile(r"RESULT iterations=(\d+) bytes=(\d+) host_us=([0-9.]+) total_us=([0-9.]+) correct=(\d+)")
SHIM_RE=re.compile(r"SOCKET_SHIM send_calls=(\d+) recv_calls=(\d+) send_bytes=(\d+) recv_bytes=(\d+) delayed_calls=(\d+) delay_us=(\d+) min_bytes=(\d+)")
ROW_RE=re.compile(r"^\s*\d+\s+")
SOCKET_CONFIGS=(("default",None,None),("t1_s1",1,1),("t1_s2",1,2),("t2_s1",2,1),("t2_s2",2,2))

def clean_env():
  e=dict(os.environ)
  for k in tuple(e):
    if k.startswith("NCCL_") or k.startswith("TORCH_NCCL_") or k in ("LD_LIBRARY_PATH","LD_PRELOAD"):e.pop(k)
  return e

def net_env(nt=None,ns=None,debug="WARN"):
  e=clean_env();e.update({"NCCL_P2P_DISABLE":"1","NCCL_SHM_DISABLE":"1",
    "NCCL_ALGO":"Ring","NCCL_PROTO":"Simple","NCCL_DEBUG":debug,
    "NCCL_SET_THREAD_NAME":"1"})
  if nt is not None:e["NCCL_SOCKET_NTHREADS"]=str(nt)
  if ns is not None:e["NCCL_NSOCKS_PERTHREAD"]=str(ns)
  return e

def run(cmd,e,path,cwd,check=True):
  p=subprocess.run(cmd,cwd=cwd,env=e,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  path.parent.mkdir(parents=True,exist_ok=True);path.write_text(p.stdout,encoding="utf-8")
  if check and p.returncode:raise RuntimeError(f"failed {p.returncode}: {path}\n{p.stdout[-2000:]}")
  return p.stdout

def write_csv(path,rows):
  if not rows:raise RuntimeError(f"empty {path}")
  with path.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def pct(v,q):
  a=sorted(v);p=(len(a)-1)*q;lo=math.floor(p);hi=math.ceil(p)
  return a[lo] if lo==hi else a[lo]+(a[hi]-a[lo])*(p-lo)

def parse_result(text,**extra):
  m=RESULT_RE.search(text)
  if not m:raise RuntimeError("missing RESULT")
  it,b,h,t,ok=m.groups()
  if ok!="1":raise RuntimeError("correctness")
  return {**extra,"iterations":int(it),"bytes":int(b),"host_us":float(h),"total_us":float(t),"correct":1}

def compile_tools(root,raw):
  probe=root/"build/ch24_proxy_socket";shim=root/"build/ch24_socket_delay.so"
  run(["/usr/local/cuda/bin/nvcc","-std=c++17","-O2",str(root/"probes/ch24_proxy_socket.cc"),
    "-o",str(probe),"-lnccl","-lnvToolsExt"],clean_env(),raw/"compile_probe.log",root)
  run(["gcc","-O2","-shared","-fPIC",str(root/"probes/ch24_socket_delay.c"),
    "-o",str(shim),"-ldl"],clean_env(),raw/"compile_shim.log",root)
  return probe,shim

def proc_task(pid,tid):
  base=Path(f"/proc/{pid}/task/{tid}")
  status=(base/"status").read_text(errors="replace").splitlines()
  fields={x.split(":",1)[0]:x.split(":",1)[1].strip() for x in status if ":" in x}
  stat=(base/"stat").read_text(errors="replace")
  tail=stat[stat.rfind(")")+2:].split()
  return fields.get("Name","?"),fields.get("Cpus_allowed_list","?"),int(tail[36])

def inventory(probe,d,root):
  e=net_env(2,2,"INFO")
  cmd=[str(probe),"--bytes",str(64<<20),"--iterations","20","--pause-ms","2500"]
  p=subprocess.Popen(cmd,cwd=root,env=e,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
  lines=[];pid=None
  while True:
    line=p.stdout.readline();lines.append(line)
    m=re.search(r"READY pid=(\d+)",line)
    if m:pid=int(m.group(1));break
    if not line and p.poll() is not None:raise RuntimeError("inventory exited before READY")
  samples=[]
  for sample in range(10):
    for task in Path(f"/proc/{pid}/task").iterdir():
      try:
        name,mask,cpu=proc_task(pid,task.name)
        samples.append({"sample":sample,"pid":pid,"tid":int(task.name),"thread_name":name,
          "cpus_allowed_list":mask,"observed_cpu":cpu})
      except FileNotFoundError:pass
    time.sleep(.1)
  rest=p.communicate(timeout=180)[0];text="".join(lines)+rest
  (d/"raw/thread_inventory.log").parent.mkdir(parents=True,exist_ok=True)
  (d/"raw/thread_inventory.log").write_text(text,encoding="utf-8")
  if p.returncode:raise RuntimeError("inventory probe failed")
  parse_result(text)
  counts=Counter((x["thread_name"],x["cpus_allowed_list"]) for x in samples)
  summary=[]
  for (name,mask),count in sorted(counts.items()):
    tids={x["tid"] for x in samples if x["thread_name"]==name and x["cpus_allowed_list"]==mask}
    cpus={x["observed_cpu"] for x in samples if x["thread_name"]==name and x["cpus_allowed_list"]==mask}
    summary.append({"thread_name":name,"unique_threads":len(tids),"samples":count,
      "cpus_allowed_list":mask,"observed_cpus":";".join(map(str,sorted(cpus)))})
  return samples,summary,text

def tests_command(binary,cycles,iters):
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
      "wrong":int(f[8]),"in_place_wrong":int(f[12])})
  if len(rows)!=cycles*3 or set(seen.values())!={cycles}:raise RuntimeError(f"perf parse {config} {len(rows)} {seen}")
  if any(x["wrong"] or x["in_place_wrong"] for x in rows):raise RuntimeError("perf correctness")
  return rows

def summarize_perf(rows):
  g=defaultdict(list)
  for r in rows:g[(r["config"],r["size_bytes"])].append(r)
  out=[]
  for (c,s),a in sorted(g.items()):
    t=[x["time_us"] for x in a]
    out.append({"config":c,"size_bytes":s,"samples":len(a),"median_time_us":statistics.median(t),
      "p95_time_us":pct(t,.95),"cv_percent":statistics.pstdev(t)/statistics.fmean(t)*100,
      "median_busbw_GBs":statistics.median(x["busbw_GBs"] for x in a)})
  return out

def delay_cases(probe,shim,d,root,samples):
  rows=[]
  for delay in (0,20,100,1000):
    for sample in range(samples):
      e=net_env(2,2);e.update({"LD_PRELOAD":str(shim),"NCCL_SOCKET_DELAY_US":str(delay),
        "NCCL_SOCKET_DELAY_MIN_BYTES":"65536"})
      text=run([str(probe),"--bytes",str(64<<20),"--iterations","5"],e,
        d/f"raw/delay/d{delay}_s{sample}.log",root)
      row=parse_result(text,delay_us=delay,sample=sample)
      m=SHIM_RE.search(text)
      if not m:raise RuntimeError("missing SOCKET_SHIM")
      for key,val in zip(("send_calls","recv_calls","send_bytes_actual","recv_bytes_actual","delayed_calls","shim_delay_us","min_bytes"),m.groups()):row[key]=int(val)
      rows.append(row)
    print(f"[delay] {delay} us PASS",flush=True)
  if not any(x["delayed_calls"] for x in rows if x["delay_us"]):raise RuntimeError("delay injection did not intercept data calls")
  return rows

def affinity_cases(probe,d,root,samples):
  rows=[]
  for case,pin,burn in (("wide",False,False),("proxy_cpu0",True,False),("proxy_cpu0_contended",True,True)):
    for sample in range(samples):
      burner=None
      try:
        cmd=[str(probe),"--bytes",str(64<<20),"--iterations","5","--pause-ms","2500"]
        p=subprocess.Popen(cmd,cwd=root,env=net_env(2,2),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        lines=[];pid=None
        while True:
          line=p.stdout.readline();lines.append(line)
          m=re.search(r"READY pid=(\d+)",line)
          if m:pid=int(m.group(1));break
          if not line and p.poll() is not None:raise RuntimeError(f"{case} exited before READY")
        targets=[]
        for task in Path(f"/proc/{pid}/task").iterdir():
          name=(task/"status").read_text(errors="replace").splitlines()[0].split(":",1)[1].strip()
          if name.startswith("NCCL Progress") or name.startswith("NCCL Sock"):targets.append(int(task.name))
        if len(targets)<8:raise RuntimeError(f"too few data-plane threads: {len(targets)}")
        if pin:
          for tid in targets:os.sched_setaffinity(tid,{0})
        masks={",".join(map(str,sorted(os.sched_getaffinity(tid)))) for tid in targets}
        if pin and masks!={"0"}:raise RuntimeError(f"affinity verification failed: {masks}")
        if burn:burner=subprocess.Popen(["taskset","-c","0","bash","-c","while :; do :; done"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        text="".join(lines)+p.communicate(timeout=240)[0]
        log=d/f"raw/affinity/{case}_s{sample}.log";log.parent.mkdir(parents=True,exist_ok=True);log.write_text(text,encoding="utf-8")
        if p.returncode:raise RuntimeError(f"{case} probe failed")
        rows.append(parse_result(text,case=case,sample=sample,target_threads=len(targets),
          target_masks=";".join(sorted(masks)),pin_verified=int(pin and masks=={"0"}),burner=int(burn)))
      finally:
        if burner:
          burner.terminate()
          try:burner.wait(timeout=2)
          except subprocess.TimeoutExpired:burner.kill();burner.wait()
    print(f"[affinity] {case} PASS",flush=True)
  return rows

def summarize_probe(rows,key):
  g=defaultdict(list)
  for r in rows:g[r[key]].append(r)
  out=[]
  for case,a in g.items():
    vals=[x["total_us"] for x in a]
    out.append({key:case,"samples":len(a),"median_host_us":statistics.median(x["host_us"] for x in a),
      "median_total_us":statistics.median(vals),"p95_total_us":pct(vals,.95),
      "cv_percent":statistics.pstdev(vals)/statistics.fmean(vals)*100})
  return sorted(out,key=lambda x:float(x[key]) if key=="delay_us" else str(x[key]))

def nsys_case(nsys,probe,d,root):
  prefix=d/"private/nsys_proxy_socket"
  e=net_env(2,2)
  run([str(nsys),"profile","--force-overwrite=true","--sample=none","--cpuctxsw=none",
    "--trace=cuda,nvtx,osrt","-o",str(prefix),str(probe),"--bytes",str(64<<20),
    "--iterations","5"],e,d/"raw/nsys/profile.log",root)
  rep=Path(str(prefix)+".nsys-rep")
  nvtx=run([str(nsys),"stats","--force-export=true","--report","nvtx_gpu_proj_sum","--format","csv",str(rep)],
    clean_env(),d/"private/nsys_nvtx.csv",root)
  gpu=run([str(nsys),"stats","--force-export=true","--report","cuda_gpu_trace","--format","csv",str(rep)],
    clean_env(),d/"private/nsys_gpu.csv",root)
  osrt=run([str(nsys),"stats","--force-export=true","--report","osrt_sum","--format","csv",str(rep)],
    clean_env(),d/"private/nsys_osrt.csv",root)
  line=next((x for x in nvtx.splitlines() if x.startswith(":net-loop,")),None)
  if not line:raise RuntimeError("missing net-loop NVTX")
  fields=next(csv.reader([line]));gpu_ops=int(fields[10])
  kernels=[next(csv.reader([x])) for x in gpu.splitlines() if "ncclDevKernel_" in x]
  calls=[]
  for x in osrt.splitlines():
    if not x or x.startswith('"') or x.startswith("Time") or x.startswith("Generating") or x.startswith("Processing") or x.startswith("Using") or x.startswith("SKIPPED"):continue
    try:
      f=next(csv.reader([x]));calls.append((float(f[0]),f[-1],int(f[2])))
    except (ValueError,IndexError):pass
  calls=sorted(calls,reverse=True)[:8]
  return [{"range":"net-loop","iterations":5,"gpu_ops":gpu_ops,"nccl_kernel_instances":len(kernels),
    "expected_min_gpu_ops":20,"top_os_runtime_calls":";".join(f"{name}:{count}" for _,name,count in calls),
    "profile_private":1}]

def main():
  ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("/root/nccl-learning"))
  ap.add_argument("--run-id",default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
  ap.add_argument("--samples",type=int,default=5);ap.add_argument("--cycles",type=int,default=5);a=ap.parse_args()
  d=a.root/"logs/ch24_proxy_socket"/a.run_id;(d/"private").mkdir(parents=True,exist_ok=True)
  probe,shim=compile_tools(a.root,d/"raw")
  inventory_rows,thread_summary,inventory_log=inventory(probe,d,a.root)
  if "NET/Socket" not in inventory_log:raise RuntimeError("forced NET/Socket path not observed")
  if not any(x["thread_name"].startswith("NCCL Progress") for x in thread_summary):raise RuntimeError("progress thread absent")
  if not any(x["thread_name"].startswith("NCCL Sock") for x in thread_summary):raise RuntimeError("socket helper absent")
  write_csv(d/"thread_inventory_raw.csv",inventory_rows);write_csv(d/"thread_summary.csv",thread_summary)

  binary=a.root/"third_party/nccl-tests/build/all_reduce_perf";perf=[];cmd=tests_command(binary,a.cycles,10)
  for rep,order in ((1,SOCKET_CONFIGS),(2,tuple(reversed(SOCKET_CONFIGS)))):
    for config,nt,ns in order:
      text=run(cmd,net_env(nt,ns),d/f"raw/performance/{config}_r{rep}.log",binary.parent.parent)
      perf+=parse_perf(config,rep,text,a.cycles);print(f"[perf] {config} r={rep} PASS",flush=True)
  perf_summary=summarize_perf(perf)
  write_csv(d/"raw_measurements.csv",perf);write_csv(d/"performance_summary.csv",perf_summary)

  delays=delay_cases(probe,shim,d,a.root,a.samples);affinity=affinity_cases(probe,d,a.root,a.samples)
  write_csv(d/"delay_measurements.csv",delays);write_csv(d/"delay_summary.csv",summarize_probe(delays,"delay_us"))
  write_csv(d/"affinity_measurements.csv",affinity);write_csv(d/"affinity_summary.csv",summarize_probe(affinity,"case"))
  nsys=Path(subprocess.check_output(["which","nsys"],text=True).strip())
  nsys_rows=nsys_case(nsys,probe,d,a.root);write_csv(d/"nsys_summary.csv",nsys_rows)

  delay_summary=summarize_probe(delays,"delay_us");aff_summary=summarize_probe(affinity,"case")
  manifest=["experiment=ch24_proxy_socket",f"run_id={a.run_id}",f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
    f"hostname={os.uname().nodename}",f"performance_rows={len(perf)}",f"delay_rows={len(delays)}",
    f"affinity_rows={len(affinity)}",f"thread_samples={len(inventory_rows)}","correctness=PASS",
    "transport_acceptance=NET/Socket PASS","progress_thread_acceptance=PASS","socket_helper_acceptance=PASS",
    "delay_interposition_acceptance=PASS","private_nsys_reports=NOT_FOR_PUBLICATION","nccl_runtime=2.22.3",
    "nccl_source_commit="+subprocess.check_output(["git","-C",str(a.root/"third_party/nccl-2.22.3"),"rev-parse","HEAD"],text=True).strip()]
  (d/"manifest.txt").write_text("\n".join(manifest)+"\n",encoding="utf-8")
  lines=["# Chapter 24 experiment summary","",f"- Performance rows: {len(perf)}",f"- Delay rows: {len(delays)}",
    f"- Affinity rows: {len(affinity)}","- Correctness: PASS","- Forced transport: NET/Socket PASS","",
    "| delay us/call | median total us |","|---:|---:|"]
  for x in delay_summary:lines.append(f"| {x['delay_us']} | {x['median_total_us']:.3f} |")
  lines += ["","| affinity case | median total us |","|---|---:|"]
  for x in aff_summary:lines.append(f"| {x['case']} | {x['median_total_us']:.3f} |")
  (d/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
  print(f"run_dir={d}",flush=True)

if __name__=="__main__":main()
