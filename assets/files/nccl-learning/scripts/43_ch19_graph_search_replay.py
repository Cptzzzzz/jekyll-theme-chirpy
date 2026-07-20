#!/usr/bin/env python3
"""Exercise NCCL graph search, XML replay, fallback, and final channel caps."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import statistics
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROW_RE = re.compile(r"^\s*\d+\s+")
PATTERN_RE = re.compile(r"Pattern (\d+), crossNic (\d+), nChannels (\d+), bw ([0-9.]+)/([0-9.]+), type ([A-Z]+)/([A-Z]+), sameChannels (\d+)")
FINAL_CHANNEL_RE = re.compile(r"(\d+) coll channels,.*?(\d+) p2p channels")
RING_RE = re.compile(r"Channel (\d+)/(\d+) :\s+((?:\d+\s*)+)$")
CONFIGS = ("search_native", "replay_exact", "replay_custom_one", "replay_ring_only",
           "weak_fallback", "postset_max4", "postset_min16")
PERF_CONFIGS = ("search_native", "replay_exact", "replay_custom_one",
                "weak_fallback", "postset_max4", "postset_min16")


def clean_env() -> dict[str, str]:
    env=dict(os.environ)
    for key in tuple(env):
        if key.startswith("NCCL_") or key.startswith("TORCH_NCCL_"): env.pop(key)
    return env


def run(command:list[str],env:dict[str,str],path:Path,cwd:Path,check:bool=True)->tuple[int,str]:
    process=subprocess.run(command,cwd=cwd,env=env,stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,text=True,check=False)
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(process.stdout,encoding="utf-8")
    if check and process.returncode!=0: raise RuntimeError(f"failed; see {path}")
    return process.returncode,process.stdout


def write_csv(path:Path,rows:list[dict[str,object]])->None:
    if not rows: raise RuntimeError(f"empty CSV {path}")
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def percentile(values:list[float],fraction:float)->float:
    ordered=sorted(values); pos=(len(ordered)-1)*fraction; lo,hi=math.floor(pos),math.ceil(pos)
    if lo==hi:return ordered[lo]
    return ordered[lo]+(ordered[hi]-ordered[lo])*(pos-lo)


def make_graph_variants(exact:Path,directory:Path)->dict[str,Path]:
    directory.mkdir(parents=True,exist_ok=True)
    outputs={"exact":directory/"exact.xml"}; outputs["exact"].write_bytes(exact.read_bytes())
    tree=ET.parse(exact); root=tree.getroot()
    for graph in list(root.findall("graph")):
        if graph.get("id") not in ("0","1"):
            root.remove(graph); continue
        graph.set("nchannels","1")
        for channel in list(graph.findall("channel")): graph.remove(channel)
        channel=ET.SubElement(graph,"channel")
        order=("0","0x2","0x1","0x3") if graph.get("id")=="0" else ("0","0x3","0x1","0x2")
        for dev in order: ET.SubElement(channel,"gpu",{"dev":dev})
    custom=directory/"custom_one.xml"; tree.write(custom,encoding="unicode"); outputs["custom"]=custom
    tree=ET.parse(custom); root=tree.getroot()
    for graph in list(root.findall("graph")):
        if graph.get("id")!="0": root.remove(graph)
    ring_only=directory/"ring_only.xml"; tree.write(ring_only,encoding="unicode"); outputs["ring_only"]=ring_only
    tree=ET.parse(custom); first=tree.find('.//graph[@id="0"]/channel/gpu')
    if first is None: raise RuntimeError("missing ring gpu")
    first.set("dev","0xff"); invalid=directory/"invalid_device.xml"; tree.write(invalid,encoding="unicode"); outputs["invalid"]=invalid
    return outputs


def make_weak_topology(native:Path,path:Path)->None:
    tree=ET.parse(native)
    for node in tree.findall(".//nvlink"): node.set("count","0")
    for pci in tree.findall(".//pci"):
        if pci.find("gpu") is not None:
            pci.set("link_width","1"); pci.set("link_speed","2.5 GT/s PCIe")
    tree.write(path,encoding="unicode")


def base_command(binary:Path,begin:str="4K",end:str="4K",cycles:int=1,iterations:int=1)->list[str]:
    return [str(binary),"-b",begin,"-e",end,"-f","128","-g","4","-w","1" if cycles==1 else "5",
        "-n",str(iterations),"-N",str(cycles),"-c","1","-I","0","-z","0","-u","0","-C","0","-a","3","-d","float","-o","sum"]


def config_env(config:str,variants:dict[str,Path],weak:Path)->dict[str,str]:
    env=clean_env()
    if config=="replay_exact": env["NCCL_GRAPH_FILE"]=str(variants["exact"])
    elif config=="replay_custom_one": env["NCCL_GRAPH_FILE"]=str(variants["custom"])
    elif config=="replay_ring_only": env["NCCL_GRAPH_FILE"]=str(variants["ring_only"])
    elif config=="weak_fallback": env["NCCL_TOPO_FILE"]=str(weak)
    elif config=="postset_max4": env["NCCL_MAX_NCHANNELS"]="4"
    elif config=="postset_min16": env["NCCL_MIN_NCHANNELS"]="16"
    return env


def parse_search(config:str,text:str)->tuple[dict[str,object],list[dict[str,object]]]:
    patterns={}
    for line in text.splitlines():
        if "[0] NCCL INFO" not in line: continue
        match=PATTERN_RE.search(line)
        if match and match.group(1) in ("4","1") and match.group(1) not in patterns:
            patterns[match.group(1)]=match.groups()
    if set(patterns)!={"4","1"}: raise RuntimeError(f"{config}: pattern rows {patterns}")
    final_matches=[]
    for line in text.splitlines():
        if "[0] NCCL INFO" in line:
            match=FINAL_CHANNEL_RE.search(line)
            if match: final_matches.append(match)
    if not final_matches: raise RuntimeError(f"{config}: no final channels")
    final_coll=int(final_matches[-1].group(1)); final_p2p=int(final_matches[-1].group(2))
    ring_orders=[]
    for line in text.splitlines():
        if "[0] NCCL INFO Channel " not in line: continue
        match=RING_RE.search(line)
        if match:
            ring_orders.append({"config":config,"channel":int(match.group(1)),
                "printed_total_channels":int(match.group(2)),
                "rank_order":"-".join(match.group(3).split())})
    ring=patterns["4"]; tree=patterns["1"]
    summary={"config":config,"ring_pattern":int(ring[0]),"ring_base_channels":int(ring[2]),
        "ring_bw_intra_GBs":float(ring[3]),"ring_bw_inter_GBs":float(ring[4]),
        "ring_type_intra":ring[5],"ring_same_channels":int(ring[7]),
        "tree_pattern":int(tree[0]),"tree_base_channels":int(tree[2]),
        "tree_bw_intra_GBs":float(tree[3]),"tree_type_intra":tree[5],
        "tree_same_channels":int(tree[7]),"final_coll_channels":final_coll,
        "final_p2p_channels":final_p2p,"fallback_warning":int("falling back to simple order" in text),
        "ring_xml_load_line":int("Search 0 :" in text and "channels loaded from XML graph" in text),
        "tree_xml_id_present":int(config in ("replay_exact","replay_custom_one")),
        "first_final_ring_order":ring_orders[0]["rank_order"] if ring_orders else ""}
    return summary,ring_orders


def parse_graph_xml(config:str,path:Path)->list[dict[str,object]]:
    rows=[]
    for index,graph in enumerate(ET.parse(path).getroot().findall("graph")):
        if graph.get("id") not in ("0","1"):continue
        rows.append({"config":config,"xml_index":index,"graph_id":int(graph.get("id","-1")),
            "pattern":int(graph.get("pattern","-1")),"channels":int(graph.get("nchannels","0")),
            "speed_intra_GBs":float(graph.get("speedintra","0")),"type_intra":graph.get("typeintra"),
            "same_channels":int(graph.get("samechannels","0")),
            "orders":";".join("-".join(g.get("dev","") for g in c.findall("gpu")) for c in graph.findall("channel"))})
    return rows


def parse_perf(config:str,replicate:int,text:str,cycles:int)->list[dict[str,object]]:
    rows=[];seen=defaultdict(int)
    for line in text.splitlines():
        if not ROW_RE.match(line):continue
        fields=line.split()
        if len(fields)!=13 or int(fields[0])<4096:continue
        size=int(fields[0]);cycle=seen[size];seen[size]+=1
        rows.append({"config":config,"replicate":replicate,"cycle":cycle,"size_bytes":size,
            "time_us":float(fields[5]),"algbw_GBs":float(fields[6]),"busbw_GBs":float(fields[7]),
            "wrong":int(fields[8]),"in_place_time_us":float(fields[9]),
            "in_place_busbw_GBs":float(fields[11]),"in_place_wrong":int(fields[12])})
    if len(rows)!=3*cycles or len(seen)!=3 or set(seen.values())!={cycles}:raise RuntimeError(f"perf parse {config} {len(rows)} {seen}")
    if any(r["wrong"] or r["in_place_wrong"] for r in rows):raise RuntimeError(f"correctness {config}")
    return rows


def summarize_perf(rows:list[dict[str,object]])->list[dict[str,object]]:
    groups=defaultdict(list)
    for row in rows:groups[(row["config"],row["size_bytes"])].append(row)
    output=[]
    for (config,size),group in sorted(groups.items()):
        times=[float(r["time_us"]) for r in group]
        output.append({"config":config,"size_bytes":size,"samples":len(group),
            "median_time_us":statistics.median(times),"p95_time_us":percentile(times,.95),
            "cycle_cv_percent":statistics.pstdev(times)/statistics.fmean(times)*100,
            "median_busbw_GBs":statistics.median(float(r["busbw_GBs"]) for r in group)})
    baseline={int(r["size_bytes"]):float(r["median_time_us"]) for r in output if r["config"]=="search_native"}
    for row in output:row["time_vs_search_native_percent"]=(float(row["median_time_us"])/baseline[int(row["size_bytes"])]-1)*100
    return output


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("/root/nccl-learning"))
    parser.add_argument("--run-id",default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--cycles",type=int,default=10);parser.add_argument("--iterations",type=int,default=20)
    args=parser.parse_args();run_dir=args.root/"logs/ch19_graph_search"/args.run_id
    raw=run_dir/"raw";artifacts=run_dir/"artifacts";artifacts.mkdir(parents=True,exist_ok=True)
    binary=args.root/"third_party/nccl-tests/build/all_reduce_perf";cwd=args.root/"third_party/nccl-tests"
    smoke=base_command(binary)

    native_env=clean_env();native_env.update({"NCCL_TOPO_DUMP_FILE":str(artifacts/"native_topology.xml"),
        "NCCL_GRAPH_DUMP_FILE":str(artifacts/"native_search_graphs.xml"),"NCCL_DEBUG":"INFO","NCCL_DEBUG_SUBSYS":"ENV,GRAPH,INIT"})
    _,native_text=run(smoke,native_env,raw/"search"/"search_native.log",cwd)
    if "# Out of bounds values : 0 OK" not in native_text:raise RuntimeError("native smoke")
    variants=make_graph_variants(artifacts/"native_search_graphs.xml",artifacts/"graph_fixtures")
    weak=artifacts/"weak_topology.xml";make_weak_topology(artifacts/"native_topology.xml",weak)
    (artifacts/"resolved_graphs").mkdir(parents=True,exist_ok=True)

    search_rows=[];order_rows=[];graph_rows=[]
    for config in CONFIGS:
        if config=="search_native":text=native_text;dump=artifacts/"native_search_graphs.xml"
        else:
            env=config_env(config,variants,weak);dump=artifacts/"resolved_graphs"/f"{config}.xml"
            env.update({"NCCL_GRAPH_DUMP_FILE":str(dump),"NCCL_DEBUG":"INFO","NCCL_DEBUG_SUBSYS":"ENV,GRAPH,INIT"})
            _,text=run(smoke,env,raw/"search"/f"{config}.log",cwd)
            if "# Out of bounds values : 0 OK" not in text:raise RuntimeError(f"smoke {config}")
        row,orders=parse_search(config,text);search_rows.append(row);order_rows.extend(orders);graph_rows.extend(parse_graph_xml(config,dump))
        print(f"[search] {config} PASS",flush=True)

    env=clean_env();env.update({"NCCL_GRAPH_FILE":str(variants["invalid"]),"NCCL_DEBUG":"INFO","NCCL_DEBUG_SUBSYS":"ENV,GRAPH,INIT"})
    status,text=run(smoke,env,raw/"failures"/"invalid_graph_device.log",cwd,check=False)
    failure=[{"case":"invalid_graph_device","exit_status":status,
        "signature":"XML Import Channel : dev 255 not found", "observed":int(status!=0 and "dev 255 not found" in text)}]
    if not failure[0]["observed"]:raise RuntimeError(f"invalid graph did not fail {failure}")

    expected={"search_native":(6,6,12,0),"replay_exact":(6,6,12,0),
        "replay_custom_one":(1,1,2,0),"replay_ring_only":(1,1,2,0),
        "weak_fallback":(1,1,2,1),"postset_max4":(6,6,4,0),"postset_min16":(6,6,16,0)}
    for row in search_rows:
        actual=(row["ring_base_channels"],row["tree_base_channels"],row["final_coll_channels"],row["fallback_warning"])
        if actual!=expected[str(row["config"])]:raise RuntimeError(f"acceptance {row} expected {expected[str(row['config'])]}")
    if next(r for r in search_rows if r["config"]=="replay_custom_one")["first_final_ring_order"]!="0-2-1-3":raise RuntimeError("custom order not honored")

    perf_rows=[];perf_cmd=base_command(binary,"4K","64M",args.cycles,args.iterations)
    for replicate,ordered in ((1,PERF_CONFIGS),(2,tuple(reversed(PERF_CONFIGS)))):
        for config in ordered:
            env=config_env(config,variants,weak);env["NCCL_DEBUG"]="WARN"
            _,text=run(perf_cmd,env,raw/"performance"/f"{config}_r{replicate}.log",cwd)
            perf_rows.extend(parse_perf(config,replicate,text,args.cycles));print(f"[perf] {config} r={replicate} PASS",flush=True)
    perf_summary=summarize_perf(perf_rows)

    write_csv(run_dir/"search_summary.csv",search_rows);write_csv(run_dir/"final_ring_orders.csv",order_rows)
    write_csv(run_dir/"resolved_graphs.csv",graph_rows);write_csv(run_dir/"failure_matrix.csv",failure)
    write_csv(run_dir/"raw_measurements.csv",perf_rows);write_csv(run_dir/"performance_summary.csv",perf_summary)
    manifest=["experiment=ch19_graph_search_replay",f"run_id={args.run_id}",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",f"hostname={os.uname().nodename}",
        "world_size=4",f"search_configs={','.join(CONFIGS)}",f"performance_configs={','.join(PERF_CONFIGS)}",
        f"cycles={args.cycles}",f"iterations={args.iterations}",f"measurement_rows={len(perf_rows)}",
        "correctness=PASS","invalid_graph_failure=PASS",
        f"nccl_source_commit={subprocess.check_output(['git','-C',str(args.root/'third_party/nccl-2.22.3'),'rev-parse','HEAD'],text=True).strip()}"]
    (run_dir/"manifest.txt").write_text("\n".join(manifest)+"\n",encoding="utf-8")
    perf64={(r["config"],int(r["size_bytes"])):r for r in perf_summary}
    lines=["# Chapter 19 experiment summary","",f"- Performance rows: {len(perf_rows)}","- Correctness: PASS","- Search/replay/fallback acceptance: 7/7 PASS","- Invalid graph failure: PASS","",
        "| config | Ring base | Tree base | final coll | fallback | first ring | 64 MiB us | busbw | vs native |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|"]
    for row in search_rows:
        perf=perf64.get((row["config"],64<<20))
        lines.append(f"| {row['config']} | {row['ring_base_channels']} | {row['tree_base_channels']} | {row['final_coll_channels']} | {row['fallback_warning']} | {row['first_final_ring_order']} | "+(f"{perf['median_time_us']:.3f} | {perf['median_busbw_GBs']:.2f} | {perf['time_vs_search_native_percent']:+.2f}% |" if perf else "n/a | n/a | n/a |"))
    (run_dir/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"run_dir={run_dir}",flush=True)


if __name__=="__main__":main()
