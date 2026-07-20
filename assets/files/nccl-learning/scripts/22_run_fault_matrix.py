#!/usr/bin/env python3
import argparse
import csv
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


CASES = [
    ("skip", "OFF"),
    ("op_mismatch", "DETAIL"),
    ("size_mismatch", "DETAIL"),
    ("rank_exit", "OFF"),
]
ERROR_PATTERNS = [
    r"Detected mismatch between collectives on ranks.*",
    r"Watchdog caught collective operation timeout.*",
    r"Timed out waiting.*",
    r"Connection closed by peer.*",
    r"DistBackendError:.*",
    r"ChildFailedError:.*",
]


def first_error(text: str) -> str:
    for pattern in ERROR_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(0).strip()[:500]
    lines = [line.strip() for line in text.splitlines() if "error" in line.lower() or "failed" in line.lower()]
    return lines[0][:500] if lines else "no classified error line"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run expected-failure distributed cases")
    parser.add_argument("--root", type=Path, default=Path("/root/nccl-learning"))
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()

    out_dir = args.root / "logs/fault_matrix" / args.run_id
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    manifest = [
        f"run_id={args.run_id}",
        f"started_at={datetime.now(timezone.utc).isoformat()}",
        f"hostname={os.uname().nodename}",
        "process_group_timeout_seconds=6",
        "communicator_warmup=1",
    ]

    for case, debug in CASES:
        command = [
            "torchrun",
            "--standalone",
            "--nproc_per_node=2",
            str(args.root / "scripts/22_fault_worker.py"),
            "--case",
            case,
        ]
        env = os.environ.copy()
        env["TORCH_DISTRIBUTED_DEBUG"] = debug
        env["TORCH_NCCL_BLOCKING_WAIT"] = "1"
        env["NCCL_DEBUG"] = "WARN"
        manifest.append(f"case_{case}=TORCH_DISTRIBUTED_DEBUG={debug} {' '.join(command)}")
        timed_out = False
        try:
            proc = subprocess.run(
                command,
                cwd=args.root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=25,
            )
            output = proc.stdout
            returncode = proc.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = 124
            output = exc.stdout or ""
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
        (raw_dir / f"{case}.log").write_text(output, encoding="utf-8")
        expected_failure = returncode != 0 and "UNEXPECTED_SUCCESS" not in output
        rows.append(
            {
                "case": case,
                "torch_distributed_debug": debug,
                "returncode": returncode,
                "external_timeout": int(timed_out),
                "expected_failure": int(expected_failure),
                "first_error": first_error(output),
            }
        )
        print(f"case={case} returncode={returncode} timeout={int(timed_out)} expected_failure={int(expected_failure)}")
        if not expected_failure:
            raise RuntimeError(f"case {case} did not fail in the expected way")

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Distributed Fault Matrix",
        "",
        f"Run ID: `{args.run_id}`",
        "",
        "| case | distributed debug | return code | external timeout | classified first error |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        error = str(row["first_error"]).replace("|", "\\|")
        lines.append(
            f"| {row['case']} | {row['torch_distributed_debug']} | {row['returncode']} | "
            f"{row['external_timeout']} | {error} |"
        )
    lines.extend(
        [
            "",
            "All cases first completed a valid AllReduce, so the communicator and P2P path were healthy before fault injection. The failure signature therefore belongs to collective ordering or peer liveness, not initial network setup.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest.append(f"completed_at={datetime.now(timezone.utc).isoformat()}")
    (out_dir / "manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
