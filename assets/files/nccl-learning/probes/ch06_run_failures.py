#!/usr/bin/env python3
"""Run chapter 6 expected-success and expected-failure order cases."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time
from pathlib import Path


CASES = [
    ("ordered_off", "ordered", "OFF", "success"),
    ("op_mismatch_detail", "op_mismatch", "DETAIL", "fingerprint"),
    ("size_mismatch_detail", "size_mismatch", "DETAIL", "fingerprint"),
    ("op_mismatch_raw", "op_mismatch", "OFF", "runtime-failure"),
    ("rank_skip_raw", "rank_skip", "OFF", "timeout"),
]


def classify(output: str, external_timeout: bool, returncode: int) -> tuple[str, str]:
    if "op_mismatch_returned" in output and "Watchdog caught collective operation timeout" in output:
        match = re.search(r"Watchdog caught collective operation timeout[^\n]*", output)
        return "partial-return-then-watchdog", match.group(0)[:500] if match else "partial return followed by watchdog"
    patterns = [
        ("fingerprint-rejected", r"Detected mismatch between collectives[^\n]*"),
        ("watchdog-timeout", r"Watchdog caught collective operation timeout[^\n]*"),
        ("blocking-wait-timeout", r"Timed out waiting[^\n]*"),
        ("local-mismatch-returned", r"op_mismatch_returned[^\n]*"),
        ("semantic-size-returned", r"mismatched counts unexpectedly completed"),
        ("child-failed", r"ChildFailedError[^\n]*"),
    ]
    for outcome, pattern in patterns:
        match = re.search(pattern, output)
        if match:
            return outcome, match.group(0)[:500]
    if external_timeout:
        return "external-timeout", "subprocess exceeded 25 seconds"
    if returncode == 0:
        return "success", "process exited zero"
    lines = [line.strip() for line in output.splitlines() if "error" in line.lower()]
    return "unclassified-failure", (lines[0][:500] if lines else "nonzero exit")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    raw_dir = args.run_dir / "raw/failures"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, worker_case, debug, expectation in CASES:
        command = [
            "torchrun",
            "--standalone",
            "--nproc_per_node=2",
            str(args.root / "scripts/30_ch06_order_worker.py"),
            "--case",
            worker_case,
        ]
        env = os.environ.copy()
        env.update(
            {
                "TORCH_DISTRIBUTED_DEBUG": debug,
                "TORCH_NCCL_BLOCKING_WAIT": "1",
                "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
                "NCCL_DEBUG": "WARN",
            }
        )
        started = time.monotonic()
        external_timeout = False
        try:
            process = subprocess.run(
                command,
                cwd=args.root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=25,
            )
            output = process.stdout
            returncode = process.returncode
        except subprocess.TimeoutExpired as error:
            external_timeout = True
            returncode = 124
            output = error.stdout or ""
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
        duration_s = time.monotonic() - started
        (raw_dir / f"{label}.log").write_text(output, encoding="utf-8")
        outcome, signature = classify(output, external_timeout, returncode)

        if expectation == "success":
            accepted = returncode == 0 and outcome == "success" and "correct=1" in output
        elif expectation == "fingerprint":
            accepted = returncode != 0 and outcome == "fingerprint-rejected"
        elif expectation == "timeout":
            accepted = returncode != 0 and outcome in {
                "watchdog-timeout",
                "blocking-wait-timeout",
                "external-timeout",
            }
        else:
            accepted = returncode != 0 and outcome != "fingerprint-rejected"

        rows.append(
            {
                "label": label,
                "worker_case": worker_case,
                "distributed_debug": debug,
                "expectation": expectation,
                "returncode": returncode,
                "external_timeout": int(external_timeout),
                "duration_s": f"{duration_s:.3f}",
                "outcome": outcome,
                "signature": signature.replace("\n", " "),
                "accepted": int(accepted),
            }
        )
        print(
            f"label={label} returncode={returncode} duration_s={duration_s:.3f} "
            f"outcome={outcome} accepted={int(accepted)}",
            flush=True,
        )
        if not accepted:
            raise RuntimeError(f"case {label} did not produce its required evidence")

    with (args.run_dir / "failure_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
