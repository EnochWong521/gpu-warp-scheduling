#!/usr/bin/env python3
"""Re-extract and validate cumulative GPGPU-Sim counters from feasibility logs.

Save as scripts/validate_feasibility.py in gpu-warp-scheduling and run:
    python scripts/validate_feasibility.py
"""

import csv
import math
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results" / "feasibility"
OUTPUT = RESULTS / "results_validated.csv"

WORKLOADS = [
    "bfs", "nw", "nn", "lud", "srad_v2", "hotspot", "backprop",
    "heartwall", "pathfinder", "streamcluster",
]
SCHEDULERS = {
    "lrr": "lrr",
    "gto": "gto",
    "rrr": "rrr",
    "oldest": "oldest",
    "wl4": "warp_limiting:2:4",
    "wl8": "warp_limiting:2:8",
    "wl16": "warp_limiting:2:16",
}

COUNTERS = {
    "cycles": "gpu_tot_sim_cycle",
    "instructions": "gpu_tot_sim_insn",
    "ipc": "gpu_tot_ipc",
}


def last_counter(log_text, name):
    """Return the last value and number of occurrences of a named counter."""
    pattern = rf"(?m)^{re.escape(name)}\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*$"
    values = re.findall(pattern, log_text)
    return (values[-1] if values else None), len(values)


def last_match(text, pattern):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return matches[-1] if matches else None


def extract_one(workload, scheduler, config):
    log_path = RESULTS / f"{workload}-{scheduler}.log"
    time_path = RESULTS / f"{workload}-{scheduler}.time"
    row = {
        "workload": workload,
        "scheduler": scheduler,
        "scheduler_config": config,
        "status": "",
        "cycles": "",
        "instructions": "",
        "ipc": "",
        "wall_time": "",
        "max_rss_kb": "",
        "exit_code": "",
        "stat_blocks": "",
        "validation_notes": "",
    }
    problems = []

    if not log_path.is_file():
        problems.append("missing log")
    else:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        counts = []
        for column, counter in COUNTERS.items():
            value, count = last_counter(text, counter)
            counts.append(count)
            if value is None:
                problems.append(f"missing {counter}")
            else:
                row[column] = int(value) if column != "ipc" else float(value)

        row["stat_blocks"] = max(counts)
        if len(set(counts)) != 1:
            problems.append("incomplete/mismatched stats blocks")

        if isinstance(row["cycles"], int) and isinstance(row["instructions"], int):
            if row["cycles"] <= 0 or row["instructions"] <= 0:
                problems.append("nonpositive cycles or instructions")
            elif isinstance(row["ipc"], float):
                computed_ipc = row["instructions"] / row["cycles"]
                # Simulator reports IPC to four decimal places.
                if not math.isclose(computed_ipc, row["ipc"], rel_tol=0, abs_tol=0.001):
                    problems.append("IPC does not equal instructions/cycles")

    if not time_path.is_file():
        problems.append("missing time file")
    else:
        text = time_path.read_text(encoding="utf-8", errors="replace")
        exit_code = last_match(text, r"^\s*Exit status:\s*(\d+)\s*$")
        row["exit_code"] = exit_code if exit_code is not None else ""
        if exit_code is None:
            problems.append("missing exit status")
        elif exit_code != "0":
            problems.append(f"nonzero exit code {exit_code}")

        row["wall_time"] = last_match(
            text, r"^\s*Elapsed \(wall clock\) time.*\):\s*(\S+)\s*$"
        ) or ""
        row["max_rss_kb"] = last_match(
            text, r"^\s*Maximum resident set size \(kbytes\):\s*(\d+)\s*$"
        ) or ""

    row["_problems"] = problems
    return row


def main():
    if not RESULTS.is_dir():
        raise SystemExit(f"Results directory not found: {RESULTS}")

    rows = [
        extract_one(workload, scheduler, config)
        for workload in WORKLOADS
        for scheduler, config in SCHEDULERS.items()
    ]

    # For a given workload/trace, the instruction count should not vary
    # between scheduling policies. Flag any discrepancy for investigation.
    for workload in WORKLOADS:
        subset = [row for row in rows if row["workload"] == workload]
        instruction_counts = {
            row["instructions"] for row in subset
            if isinstance(row["instructions"], int)
        }
        if len(instruction_counts) > 1:
            for row in subset:
                row["_problems"].append("instruction count differs across schedulers")

    for row in rows:
        issues = row.pop("_problems")
        row["validation_notes"] = "; ".join(issues)
        row["status"] = "CHECK" if issues else "PASS"

    columns = [
        "workload", "scheduler", "scheduler_config", "status",
        "cycles", "instructions", "ipc", "wall_time", "max_rss_kb",
        "exit_code", "stat_blocks", "validation_notes",
    ]
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    summary = Counter(row["status"] for row in rows)
    print(f"Wrote {len(rows)} rows to {OUTPUT}")
    print(f"PASS: {summary['PASS']}  CHECK: {summary['CHECK']}")
    for row in rows:
        if row["status"] != "PASS":
            print(f"  {row['workload']}/{row['scheduler']}: {row['validation_notes']}")
    if summary["CHECK"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
