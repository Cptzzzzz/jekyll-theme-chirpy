#!/usr/bin/env python3
"""Run and classify application, watchdog, peer-liveness, and launcher failures."""
from __future__ import annotations
import argparse,csv,hashlib,os,re,signal,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
CASES=(
 ("control","control","OFF",1,"success"),
 ("op_detail","op_mismatch","DETAIL",1,"fingerprint"),
 ("shape_detail","shape_mismatch","DETAIL",1,"fingerprint"),
 ("order_detail","order_mismatch","DETAIL",1,"fingerprint"),
 ("op_raw","op_mismatch","OFF",1,"raw_mismatch"),
 ("steady_skip","skip","OFF",1,"skip_timeout"),
 ("lazy_skip","lazy_skip","OFF",1,"lazy_init_external_timeout"),
 ("rank_exit","rank_exit","OFF",1,"rank_exit"),
 ("peer_exception","peer_exception","OFF",1,"application_exception"),
)
PATTERNS=(
 ("marker",re.compile(r"CH33_EVENT[^\n]+")),
 ("fingerprint",re.compile(r"Detected mismatch between collectives[^\n]+")),
 ("blocking_timeout",re.compile(r"Timed out waiting[^\n]+")),
 ("watchdog_timeout",re.compile(r"Watchdog caught collective operation timeout[^\n]+")),
 ("remote_error",re.compile(r"(?:Connection closed by peer|remote process exited or there was a network error)[^\n]*",re.I)),
 ("application_exception",re.compile(r"RuntimeError: CH33 injected application failure[^\n]*")),
 ("child_failed",re.compile(r"ChildFailedError[^\n]*")),
)
def clean():
 e=dict(os.environ)
 for k in tuple(e):
  if k.startswith("NCCL_") or k.startswith("TORCH_NCCL_") or k in ("MASTER_ADDR","MASTER_PORT","LD_LIBRARY_PATH","TORCH_DISTRIBUTED_DEBUG"):e.pop(k)
 e["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python";return e
def events(label,text):
 found=[]
 for kind,pat in PATTERNS:
  for m in pat.finditer(text):
   line=m.group(0);rank=(re.search(r"rank[= ](\d+)",line,re.I) or re.search(r"\[rank(\d+)\]",line,re.I));found.append((m.start(),kind,rank.group(1) if rank else "",line[:700]))
 found.sort();return [{"case":label,"ordinal":i,"kind":k,"rank":r,"evidence":s} for i,(_,k,r,s) in enumerate(found)]
def classify(label,text,rc,external):
 if label=="control":return "success"
 if label=="lazy_skip":return "lazy_init_external_timeout" if external else "lazy_init_unexpected_exit"
 if "fault_injected_application_exception" in text:return "application_exception"
 if "fault_injected_process_exit" in text:return "rank_exit"
 if "fault_injected_skip" in text:return "skip_timeout"
 if "Detected mismatch between collectives" in text:return "fingerprint"
 if "unexpected_collective_return" in text:return "raw_mismatch"
 if external:return "external_timeout"
 return "unclassified"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def outcsv(p,rows):
 with p.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("/root/nccl-learning"));ap.add_argument("--run-id",default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"));a=ap.parse_args()
 d=a.root/"logs/ch33_fault_root_cause"/a.run_id;raw=d/"raw";raw.mkdir(parents=True,exist_ok=True);rows=[];all_events=[]
 for label,case,debug,blocking,expected in CASES:
  cmd=["torchrun","--standalone","--nproc_per_node=2",str(a.root/"probes/ch33_fault_worker.py"),"--case",case];e=clean();e.update({"TORCH_DISTRIBUTED_DEBUG":debug,"TORCH_NCCL_BLOCKING_WAIT":str(blocking),"TORCH_NCCL_ASYNC_ERROR_HANDLING":"1","NCCL_DEBUG":"WARN"})
  start=time.monotonic();external=0
  p=subprocess.Popen(cmd,cwd=a.root,env=e,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
  try:text=p.communicate(timeout=16)[0];rc=p.returncode
  except subprocess.TimeoutExpired as x:
   external=1;rc=124;text=x.stdout or "";text=text.decode(errors="replace") if isinstance(text,bytes) else text
   os.killpg(p.pid,signal.SIGTERM)
   try:tail=p.communicate(timeout=3)[0]
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);tail=p.communicate()[0]
   if tail:
    if text and tail.startswith(text):text=tail
    elif tail not in text:text+=tail
  duration=time.monotonic()-start;(raw/f"{label}.log").write_text(text,encoding="utf-8");outcome=classify(label,text,rc,external);ev=events(label,text);all_events+=ev
  accepted=(rc==0 and expected=="success") or (rc!=0 and outcome==expected)
  rows.append({"case":label,"worker_case":case,"debug":debug,"blocking_wait":blocking,"returncode":rc,"external_timeout":external,"duration_s":f"{duration:.3f}","expected_root":expected,"observed_root":outcome,"event_count":len(ev),"accepted":int(accepted)})
  print(f"[fault] {label} rc={rc} root={outcome} events={len(ev)} pass={int(accepted)}",flush=True)
  if not accepted:raise RuntimeError(f"case {label} failed classification; tail={text[-2500:]}")
 outcsv(d/"case_summary.csv",rows);outcsv(d/"failure_event_timeline.csv",all_events)
 model=d/"fault_source_model.csv";p=subprocess.run(["python3",str(a.root/"probes/ch33_fault_source_model.py"),"--output",str(model)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(raw/"source_model.log").write_text(p.stdout,encoding="utf-8")
 mr=list(csv.DictReader(model.open()));
 if p.returncode or len(mr)!=37 or any(x["pass"]!="1" for x in mr):raise RuntimeError(f"source model {len(mr)}")
 import torch
 pg=Path("/opt/pytorch/pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp");nccl=a.root/"third_party/nccl-2.22.3/src/init.cc"
 manifest=["experiment=ch33_fault_root_cause",f"run_id={a.run_id}",f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",f"hostname={os.uname().nodename}",f"cases={len(rows)}",f"event_rows={len(all_events)}",f"source_model_rows={len(mr)}","case_acceptance=9/9 PASS","root_cause_classification=PASS","network_interruption=BLOCKED_SINGLE_NODE","raw_logs=NOT_FOR_PUBLICATION",f"pytorch_runtime={torch.__version__}","nccl_runtime=2.22.3",f"process_group_nccl_sha256={sha(pg)}",f"nccl_init_sha256={sha(nccl)}"]
 (d/"manifest.txt").write_text("\n".join(manifest)+"\n",encoding="utf-8")
 lines=["# Chapter 33 experiment summary","","- Cases: 9/9 PASS",f"- Causal event rows: {len(all_events)}","- Source model: 37/37 PASS","- Real network interruption: BLOCKED (single node, no HCA)","","| case | root cause | duration s | events |","|---|---|---:|---:|"]
 for x in rows:lines.append(f"| {x['case']} | {x['observed_root']} | {x['duration_s']} | {x['event_count']} |")
 (d/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(f"run_dir={d}")
if __name__=="__main__":main()
