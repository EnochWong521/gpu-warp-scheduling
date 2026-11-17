"""Export launch measurements after checking work and disabled GTO controls."""
import re
import sys
from pathlib import Path

from collect_gto_bypass import capture
from collect_gto_pairs import RESULTS, read_csv, write_csv


def extract(run, workload, enabled, controls):
    log = (run / f"{workload}-{enabled}.log").read_text()
    if "GPGPU-Sim: *** exit detected ***" not in log:
        raise ValueError(f"Incomplete {workload}/{enabled} run")
    blocks = re.split(r"(?m)^kernel_name[ \t]*=[ \t]*", log)[1:]
    previous = dict(scoreboard=0, ldst_stalls=0, overrides=0)
    total_cycles = total_insn = 0
    for i, (block, ref) in enumerate(zip(blocks, controls, strict=True), 1):
        def stat(key):
            return int(capture(rf"^{key}[ \t]*=[ \t]*(\d+)$", block))

        cycles, insn = stat("gpu_sim_cycle"), stat("gpu_sim_insn")
        kernel = block.splitlines()[0].strip()
        total_cycles += cycles
        total_insn += insn
        if (i != int(ref["launch"]) or kernel != ref["kernel"] or
                insn != int(ref["instructions"]) or cycles <= 0 or
                total_insn != stat("gpu_tot_sim_insn") or
                total_cycles != stat("gpu_tot_sim_cycle") or
                enabled != stat("gto_load_load_enabled")):
            raise ValueError(f"Work, mode or total mismatch at {workload}/{enabled}/{i}")
        current = dict(scoreboard=int(capture(r"\bW0_Scoreboard:(\d+)", block)),
                       ldst_stalls=stat("gpgpu_n_stall_shd_mem"),
                       overrides=stat("gto_load_load_overrides"))
        delta = {k: v - previous[k] for k, v in current.items()}
        previous = current
        if min(delta.values()) < 0:
            raise ValueError(f"Counter decreased at {workload}/{enabled}/{i}")
        if not enabled and (cycles != int(ref["cycles"]) or delta["overrides"] or
                            any(delta[k] != int(ref[k]) for k in ["scoreboard", "ldst_stalls"])):
            raise ValueError(f"Disabled mode differs from GTO at {workload}/{i}")
        yield dict(workload=workload, launch=i, enabled=enabled, kernel=kernel,
                   cycles=cycles, instructions=insn, **delta)


def main(run):
    archive = read_csv(RESULTS / "nn_hotspot_limit4.csv")
    rows = []
    for workload, count in [("nn", 4), ("hotspot", 7)]:
        refs = [r for r in archive if (r["workload"], r["policy"]) == (workload, "gto")]
        if len(refs) != count:
            raise ValueError(f"Incomplete stock reference for {workload}")
        for enabled in (0, 1):
            rows.extend(extract(run, workload, enabled, refs))
    write_csv(run / "gto_load_load.csv", rows)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
