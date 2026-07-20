#!/usr/bin/env python3
"""Validate NCCL topology XML, path classes, graph outputs, and fixture effects."""

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


FIXTURES = ("native_nv2", "all_to_all_nv1", "sparse_nv2_ring", "pcie_x16", "pcie_x4")
ROW_RE = re.compile(r"^\s*\d+\s+")
PATH_LINE_RE = re.compile(r"GPU/([0-9A-F]+) :(.*)")
PATH_ENTRY_RE = re.compile(r"GPU/([^ ]+) \((\d+)/([0-9.]+)/([A-Z]+)\)")


def clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith("NCCL_") or key.startswith("TORCH_NCCL_"):
            env.pop(key)
    return env


def run(command: list[str], env: dict[str, str], path: Path, cwd: Path) -> str:
    process = subprocess.run(command, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(process.stdout, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"command failed; see {path}")
    return process.stdout


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values); pos = (len(ordered)-1)*fraction
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi: return ordered[lo]
    return ordered[lo] + (ordered[hi]-ordered[lo])*(pos-lo)


def make_fixtures(native: Path, directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name in FIXTURES:
        tree = ET.parse(native)
        if name == "all_to_all_nv1":
            for node in tree.findall(".//nvlink"): node.set("count", "1")
        elif name == "sparse_nv2_ring":
            allowed = {frozenset(pair) for pair in (
                ("0000:1a:00.0", "0000:1c:00.0"),
                ("0000:1c:00.0", "0000:1d:00.0"),
                ("0000:1d:00.0", "0000:1e:00.0"),
                ("0000:1e:00.0", "0000:1a:00.0"),
            )}
            for pci in tree.findall(".//pci"):
                gpu = pci.find("gpu")
                if gpu is None: continue
                for node in list(gpu.findall("nvlink")):
                    if frozenset((pci.get("busid"), node.get("target"))) not in allowed:
                        gpu.remove(node)
        elif name in ("pcie_x16", "pcie_x4"):
            # Keep zero-count nodes so ncclTopoGetXmlFromGpu does not refill
            # physical NVLinks through NVML when loading this synthetic fixture.
            for node in tree.findall(".//nvlink"): node.set("count", "0")
            if name == "pcie_x4":
                for pci in tree.findall(".//pci"):
                    if pci.find("gpu") is not None: pci.set("link_width", "4")
        path = directory / f"{name}.xml"
        tree.write(path, encoding="unicode")
        outputs[name] = path
    return outputs


def inventory(fixture: str, path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    gpu_pcis = [pci for pci in root.findall(".//pci") if pci.find("gpu") is not None]
    nvlinks = root.findall(".//nvlink")
    return {"fixture": fixture, "cpu_nodes": len(root.findall(".//cpu")),
        "pci_nodes": len(root.findall(".//pci")), "gpu_nodes": len(root.findall(".//gpu")),
        "nic_nodes": len(root.findall(".//nic")), "net_nodes": len(root.findall(".//net")),
        "nvlink_records": len(nvlinks),
        "positive_nvlink_records": sum(int(node.get("count", "0")) > 0 for node in nvlinks),
        "sum_directed_nvlink_count": sum(int(node.get("count", "0")) for node in nvlinks),
        "gpu_pci_widths": ",".join(sorted({pci.get("link_width", "") for pci in gpu_pcis})),
        "gpu_pci_speeds": ",".join(sorted({pci.get("link_speed", "") for pci in gpu_pcis}))}


def parse_paths(fixture: str, text: str) -> list[dict[str, object]]:
    rows = []
    seen = set()
    for line in text.splitlines():
        if "[0] NCCL INFO GPU/" not in line: continue
        match = PATH_LINE_RE.search(line)
        if not match: continue
        source = match.group(1)
        for target, hops, bw, path_type in PATH_ENTRY_RE.findall(match.group(2)):
            key = (source, target)
            if key in seen: continue
            seen.add(key)
            rows.append({"fixture": fixture, "source_gpu_id": source,
                "target_gpu_id": target, "hops": int(hops), "bandwidth_GBs": float(bw),
                "path_type": path_type})
    if len(rows) != 16:
        raise RuntimeError(f"{fixture}: expected 16 GPU path entries, got {len(rows)}")
    return rows


def parse_graphs(fixture: str, path: Path) -> list[dict[str, object]]:
    root = ET.parse(path).getroot(); rows = []
    names = {"0": "Ring", "1": "Tree", "2": "CollNet", "3": "NVLS"}
    for index, graph in enumerate(root.findall("graph")):
        rows.append({"fixture": fixture, "xml_index": index, "graph_id": graph.get("id"),
            "algorithm": names.get(graph.get("id", ""), "unknown"),
            "pattern": graph.get("pattern"), "channels": int(graph.get("nchannels", "0")),
            "speed_intra_GBs": float(graph.get("speedintra", "0")),
            "speed_inter_GBs": float(graph.get("speedinter", "0")),
            "type_intra": graph.get("typeintra"), "type_inter": graph.get("typeinter"),
            "same_channels": int(graph.get("samechannels", "0")),
            "channel_orders": ";".join(
                "-".join(node.get("dev", "") for node in channel.findall("gpu"))
                for channel in graph.findall("channel"))})
    return rows


def parse_perf(text: str, fixture: str, replicate: int, cycles: int) -> list[dict[str, object]]:
    rows=[]; seen=defaultdict(int)
    for line in text.splitlines():
        if not ROW_RE.match(line): continue
        fields=line.split()
        if len(fields)!=13 or int(fields[0])<4096: continue
        size=int(fields[0]); cycle=seen[size]; seen[size]+=1
        rows.append({"fixture":fixture,"replicate":replicate,"cycle":cycle,
            "size_bytes":size,"time_us":float(fields[5]),"algbw_GBs":float(fields[6]),
            "busbw_GBs":float(fields[7]),"wrong":int(fields[8]),
            "in_place_time_us":float(fields[9]),"in_place_busbw_GBs":float(fields[11]),
            "in_place_wrong":int(fields[12])})
    if len(rows)!=3*cycles or len(seen)!=3 or set(seen.values())!={cycles}:
        raise RuntimeError(f"{fixture} r={replicate}: unexpected performance rows {len(rows)} {seen}")
    if any(row["wrong"] or row["in_place_wrong"] for row in rows):
        raise RuntimeError(f"correctness failure: {fixture}")
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups=defaultdict(list)
    for row in rows: groups[(row["fixture"],row["size_bytes"])].append(row)
    output=[]
    for (fixture,size), group in sorted(groups.items()):
        times=[float(row["time_us"]) for row in group]
        output.append({"fixture":fixture,"size_bytes":size,"samples":len(group),
            "median_time_us":statistics.median(times),"p95_time_us":percentile(times,.95),
            "cycle_cv_percent":statistics.pstdev(times)/statistics.fmean(times)*100,
            "median_busbw_GBs":statistics.median(float(row["busbw_GBs"]) for row in group)})
    baseline={int(row["size_bytes"]):float(row["median_time_us"]) for row in output if row["fixture"]=="native_nv2"}
    for row in output:
        row["time_vs_native_percent"]=(float(row["median_time_us"])/baseline[int(row["size_bytes"])]-1)*100
    return output


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("/root/nccl-learning"))
    parser.add_argument("--run-id",default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--cycles",type=int,default=10)
    parser.add_argument("--iterations",type=int,default=20)
    args=parser.parse_args()
    run_dir=args.root/"logs/ch18_topology_paths"/args.run_id
    artifacts=run_dir/"artifacts"; raw=run_dir/"raw"; artifacts.mkdir(parents=True,exist_ok=True)
    binary=args.root/"third_party/nccl-tests/build/all_reduce_perf"; cwd=args.root/"third_party/nccl-tests"

    # Independent physical evidence.
    run(["nvidia-smi","topo","-m"],clean_env(),raw/"nvidia_smi_topo.log",args.root)
    buses=subprocess.check_output(["nvidia-smi","--query-gpu=pci.bus_id","--format=csv,noheader"],text=True).split()
    sysfs=[]
    for bus in buses:
        domain, pci_bus, device = bus.split(":")
        key=f"{domain[-4:]}:{pci_bus}:{device}".lower()
        sysfs.append(f"{key} max_link_speed="+(Path("/sys/bus/pci/devices")/key/"max_link_speed").read_text().strip())
        sysfs.append(f"{key} max_link_width="+(Path("/sys/bus/pci/devices")/key/"max_link_width").read_text().strip())
    (raw/"gpu_pci_sysfs.log").write_text("\n".join(sysfs)+"\n",encoding="utf-8")

    dump_env=clean_env(); dump_env.update({"NCCL_TOPO_DUMP_FILE":str(artifacts/"native_detected.xml"),"NCCL_DEBUG":"INFO","NCCL_DEBUG_SUBSYS":"ENV,GRAPH"})
    dump_cmd=[str(binary),"-b","4K","-e","4K","-g","4","-w","1","-n","1","-N","1","-c","1","-I","0","-z","0","-u","0","-C","0","-a","3","-d","float","-o","sum"]
    text=run(dump_cmd,dump_env,raw/"native_detection.log",cwd)
    if "# Out of bounds values : 0 OK" not in text: raise RuntimeError("native detection correctness")
    fixture_paths=make_fixtures(artifacts/"native_detected.xml",artifacts/"fixtures")

    inventories=[]; path_rows=[]; graph_rows=[]
    for fixture in FIXTURES:
        inventories.append(inventory(fixture,fixture_paths[fixture]))
        fixture_dir=artifacts/"resolved"/fixture; fixture_dir.mkdir(parents=True,exist_ok=True)
        env=clean_env(); env.update({"NCCL_TOPO_FILE":str(fixture_paths[fixture]),
            "NCCL_TOPO_DUMP_FILE":str(fixture_dir/"topology.xml"),
            "NCCL_GRAPH_DUMP_FILE":str(fixture_dir/"graphs.xml"),
            "NCCL_DEBUG":"INFO","NCCL_DEBUG_SUBSYS":"ENV,GRAPH,INIT"})
        output=run(dump_cmd,env,raw/"paths"/f"{fixture}.log",cwd)
        if "# Out of bounds values : 0 OK" not in output: raise RuntimeError(f"path correctness {fixture}")
        path_rows.extend(parse_paths(fixture,output)); graph_rows.extend(parse_graphs(fixture,fixture_dir/"graphs.xml"))
        print(f"[paths] {fixture} PASS",flush=True)

    perf_rows=[]
    perf_cmd=[str(binary),"-b","4K","-e","64M","-f","128","-g","4","-w","5","-n",str(args.iterations),"-N",str(args.cycles),"-c","1","-I","0","-z","0","-u","0","-C","0","-a","3","-d","float","-o","sum"]
    for replicate, ordered in ((1,FIXTURES),(2,tuple(reversed(FIXTURES)))):
        for fixture in ordered:
            env=clean_env(); env.update({"NCCL_TOPO_FILE":str(fixture_paths[fixture]),"NCCL_DEBUG":"WARN"})
            output=run(perf_cmd,env,raw/"performance"/f"{fixture}_r{replicate}.log",cwd)
            perf_rows.extend(parse_perf(output,fixture,replicate,args.cycles))
            print(f"[performance] {fixture} r={replicate} PASS",flush=True)
    perf_summary=summarize(perf_rows)

    ring_graphs=[row for row in graph_rows if row["graph_id"]=="0"]
    expected={"native_nv2":(6,20.0,"NVL"),"all_to_all_nv1":(6,10.0,"NVL"),
        "sparse_nv2_ring":(4,20.0,"NVL"),"pcie_x16":(1,12.0,"PIX"),"pcie_x4":(1,3.0,"PIX")}
    for row in ring_graphs:
        channels,speed,path_type=expected[str(row["fixture"])]
        if (row["channels"],row["speed_intra_GBs"],row["type_intra"])!=(channels,speed,path_type):
            raise RuntimeError(f"graph acceptance mismatch: {row}")

    write_csv(run_dir/"xml_inventory.csv",inventories); write_csv(run_dir/"gpu_path_matrix.csv",path_rows)
    write_csv(run_dir/"graph_summary.csv",graph_rows); write_csv(run_dir/"raw_measurements.csv",perf_rows)
    write_csv(run_dir/"performance_summary.csv",perf_summary)
    manifest=["experiment=ch18_topology_xml_paths",f"run_id={args.run_id}",
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",f"hostname={os.uname().nodename}",
        "world_size=4",f"fixtures={','.join(FIXTURES)}",f"cycles={args.cycles}",
        f"iterations={args.iterations}",f"measurement_rows={len(perf_rows)}",
        "physical_topology=4xV100_all_pairs_NV2", "fixture_performance_warning=XML changes planner view not physical wiring",
        f"nccl_source_commit={subprocess.check_output(['git','-C',str(args.root/'third_party/nccl-2.22.3'),'rev-parse','HEAD'],text=True).strip()}"]
    (run_dir/"manifest.txt").write_text("\n".join(manifest)+"\n",encoding="utf-8")
    lines=["# Chapter 18 experiment summary","",f"- Performance rows: {len(perf_rows)}","- Out-of-place/in-place correctness: PASS","- Path/graph acceptance: 5/5 PASS","",
        "| fixture | Ring channels | graph speed | graph type | 64 MiB median us | busbw | vs native |",
        "|---|---:|---:|---|---:|---:|---:|"]
    perf64={(row["fixture"],row["size_bytes"]):row for row in perf_summary}
    for row in ring_graphs:
        perf=perf64[(row["fixture"],64<<20)]
        lines.append(f"| {row['fixture']} | {row['channels']} | {row['speed_intra_GBs']:.1f} | {row['type_intra']} | {perf['median_time_us']:.3f} | {perf['median_busbw_GBs']:.2f} | {perf['time_vs_native_percent']:+.2f}% |")
    (run_dir/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"run_dir={run_dir}",flush=True)


if __name__=="__main__": main()
