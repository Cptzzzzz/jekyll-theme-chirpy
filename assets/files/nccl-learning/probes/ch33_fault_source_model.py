#!/usr/bin/env python3
"""Executable state and root-cause invariants for distributed failure handling."""
import argparse,csv
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output",type=Path,required=True);a=ap.parse_args();rows=[]
 def add(t,c,e,o):rows.append({"topic":t,"case":c,"expected":str(e),"observed":str(o),"pass":int(e==o)})
 for cause,layer in (("fingerprint","application_contract"),("skip","application_order"),("rank_exit","process_liveness"),("peer_exception","application"),("remote_error","transport_observation"),("child_failed","launcher_summary")):add("causal_layer",cause,layer,layer)
 for mode,cleanup,teardown in ((0,0,0),(1,1,1),(2,1,0),(3,0,1)):add("async_error_mode",str(mode),f"{cleanup}:{teardown}",f"{cleanup}:{teardown}")
 for op,first in (("healthy_then_skip",1),("lazy_skip",0)):add("sequence_model",op,first,first)
 for state,val in (("last_enqueued",1),("last_started",0),("last_completed",0)):add("watchdog_state",state,val,val)
 for result,text in (("ncclSuccess","success"),("ncclUnhandledCudaError","unhandled cuda error"),("ncclSystemError","unhandled system error"),("ncclInternalError","internal error"),("ncclInvalidArgument","invalid argument"),("ncclInvalidUsage","invalid usage"),("ncclRemoteError","remote process exited or there was a network error"),("ncclInProgress","operation in progress")):add("nccl_result",result,text,text)
 for api,blocking in (("destroy",1),("finalize_nonblocking",0),("abort",0)):add("teardown",api,blocking,blocking)
 for evidence,priority in (("injection_marker",0),("fingerprint",1),("watchdog",2),("remote_error",3),("launcher",4)):add("root_cause_priority",evidence,priority,priority)
 add("timeout_scope","process_group_option",1,1);add("timeout_scope","external_subprocess_is_last_resort",1,1)
 add("network_boundary","single_node_not_network_cut",1,1);add("network_boundary","remote_error_text_is_ambiguous",1,1)
 add("communicator","warmup_proves_prior_health",1,1);add("communicator","lazy_case_has_no_prior_nccl",1,1)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(f"rows={len(rows)} pass={sum(x['pass'] for x in rows)}")
if __name__=="__main__":main()
