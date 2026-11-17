"""Validate two diagnostic logs against paired stock controls and export launch deltas."""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path


def capture(pattern, block):
    matches = re.findall(pattern, block, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"Expected one match for {pattern}, found {len(matches)}")
    return matches[0]


def extract(run, workload, baseline):
    log = (run / f"{workload}.log").read_text()
    if "GPGPU-Sim: *** exit detected ***" not in log:
        raise ValueError(f"Incomplete {workload} log")
    blocks = re.split(r"(?m)^kernel_name[ \t]*=[ \t]*", log)[1:]
    snapshots = Counter()
    previous_hist = Counter()
    total_cycles = total_insn = 0
    for launch, (block, ref) in enumerate(zip(blocks, baseline, strict=True), 1):
        def stat(name):
            return int(capture(rf"^{name}[ \t]*=[ \t]*(\d+)$", block))

        kernel = block.splitlines()[0].strip()
        cycles, insn = stat("gpu_sim_cycle"), stat("gpu_sim_insn")
        total_cycles += cycles
        total_insn += insn
        if (total_cycles, total_insn) != (stat("gpu_tot_sim_cycle"), stat("gpu_tot_sim_insn")):
            raise ValueError(f"Inconsistent cumulative totals at {workload}/{launch}")
        if (launch, kernel, cycles, insn) != (
            int(ref["launch"]), ref["kernel"], int(ref["cycles"]), int(ref["instructions"])
        ):
            raise ValueError(f"Diagnostic differs from stock GTO at {workload}/{launch}")

        current = {key: stat(f"gto_probe_{key}") for key in
                   ("repeats", "bypasses", "unknown_cycles", "opportunities")}
        current.update(idle=int(capture(r"\bW0_Idle:(\d+)", block)),
                       scoreboard=int(capture(r"\bW0_Scoreboard:(\d+)", block)),
                       ldst_stalls=stat("gpgpu_n_stall_shd_mem"))
        delta = {key: value - snapshots[key] for key, value in current.items()}
        snapshots.update(delta)
        for key in ("idle", "scoreboard", "ldst_stalls"):
            if delta[key] != int(ref[key]):
                raise ValueError(f"Stock {key} differs at {workload}/{launch}")
        entries = capture(r"^gto_probe_wait_histogram =(.*)$", block).split()
        hist = Counter(dict(tuple(map(int, entry.split(":"))) for entry in entries))
        if previous_hist - hist:
            raise ValueError("Cumulative histogram decreased")
        waits = hist - previous_hist
        previous_hist = hist
        if (min(delta.values()) < 0 or delta["bypasses"] > delta["repeats"] or
                delta["unknown_cycles"] > delta["repeats"] or
                delta["opportunities"] < delta["bypasses"] or
                sum(k * v for k, v in waits.items()) != delta["opportunities"]):
            raise ValueError(f"Inconsistent counters or unclosed waits at {workload}/{launch}")
        yield dict(workload=workload, launch=launch, kernel=kernel, cycles=cycles,
                   instructions=insn, **{k: delta[k] for k in
                   ("repeats", "bypasses", "unknown_cycles", "opportunities")},
                   wait_histogram=json.dumps(dict(sorted(waits.items())), separators=(",", ":")))


def main(run, reference):
    with reference.open(newline="") as stream:
        controls = [r for r in csv.DictReader(stream) if r["policy"] == "gto"]
    rows = []
    for workload, count, cycles in [("nn", 4, 30482), ("hotspot", 7, 53797)]:
        baseline = sorted((r for r in controls if r["workload"] == workload),
                          key=lambda r: int(r["launch"]))
        if len(baseline) != count or sum(int(r["cycles"]) for r in baseline) != cycles:
            raise ValueError(f"Invalid {workload} reference")
        rows.extend(extract(run, workload, baseline))
    path = run / "gto_bypass.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{path} ({len(rows)} paired launches)")


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:]))
