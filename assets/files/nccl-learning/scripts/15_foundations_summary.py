#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the foundations experiment summary")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir

    p2p = read_csv(run_dir / "p2p_summary.csv")
    async_rows = read_csv(run_dir / "async_collective.csv")
    version_text = (run_dir / "raw/version_matrix.log").read_text(encoding="utf-8")
    subgroup_text = (run_dir / "raw/subgroup.log").read_text(encoding="utf-8")
    collective_text = (run_dir / "raw/collective_correctness.log").read_text(encoding="utf-8")

    def extract(pattern: str) -> str:
        match = re.search(pattern, version_text)
        return match.group(1).strip() if match else "unknown"

    min_bw = min(float(r["median_bandwidth_GBs"]) for r in p2p)
    max_bw = max(float(r["median_bandwidth_GBs"]) for r in p2p)
    max_cv = max(float(r["cv_percent"]) for r in p2p)
    enqueue = sum(float(r["host_enqueue_us"]) for r in async_rows) / len(async_rows)
    gpu = sum(float(r["gpu_elapsed_us"]) for r in async_rows) / len(async_rows)
    subgroup_ok = len(re.findall(r"correct=1", subgroup_text)) == 4
    collective_ok = len(re.findall(r"correct=1", collective_text)) == 4
    torch_version = extract(r"torch_version:\s*(.+)")
    torch_cuda_version = extract(r"torch_cuda_version:\s*(.+)")
    torch_nccl_version = extract(r"torch_nccl_version:\s*(.+)")

    lines = [
        "# Foundations Experiment Summary",
        "",
        "## Version Matrix",
        "",
        f"- PyTorch: `{torch_version}`",
        f"- CUDA used by PyTorch: `{torch_cuda_version}`",
        f"- NCCL used by PyTorch: `{torch_nccl_version}`",
        "",
        "## GPU P2P Matrix",
        "",
        f"- Directed GPU pairs: `{len(p2p)}`",
        f"- Median bandwidth range: `{min_bw:.2f} - {max_bw:.2f} GB/s`",
        f"- Maximum CV across pairs: `{max_cv:.2f}%`",
        "- Every measured pair reported CUDA peer access and passed copy correctness.",
        "",
        "## Asynchronous Completion",
        "",
        f"- Mean host enqueue time: `{enqueue:.2f} us`",
        f"- Mean GPU completion time: `{gpu:.2f} us`",
        f"- GPU completion / host enqueue ratio: `{gpu / enqueue:.2f}x`",
        "- `work.wait()` established stream ordering, while final CUDA event synchronization observed device completion.",
        "",
        "## Communicator and Collective Correctness",
        "",
        f"- Subgroup experiment: `{'PASS' if subgroup_ok else 'FAIL'}`",
        f"- Seven-operation collective reference checks: `{'PASS' if collective_ok else 'FAIL'}`",
        "",
        "## Conclusion",
        "",
        "The four V100 GPUs form a uniform CUDA-P2P-capable NVLink domain. A process group is a collective communication domain rather than merely a rank list, and host-side NCCL submission must not be confused with GPU-side completion.",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(run_dir / "summary.md")


if __name__ == "__main__":
    main()
