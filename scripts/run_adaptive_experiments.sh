#!/usr/bin/env bash
set -eo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
results="$project_root/results/adaptive_policy"

usage() {
  cat <<'HELP'
Usage: bash scripts/run_adaptive_experiments.sh collect
       bash scripts/run_adaptive_experiments.sh run [backprop|nn|heartwall|hotspot ...]

collect   Extract launch counters from the existing feasibility logs.
run       Compare LRR, GTO, Oldest First, and GTO limit 4 on one rebuilt binary.
          Defaults to NN and Hotspot. Each run gets a fresh output directory.
HELP
}

collect_launches() {
  python3 - "$@" <<'PY'
import csv
import re
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

log_dir, output, validated_file = map(Path, sys.argv[1:4])
mode, *workloads = sys.argv[4:]
policies = ["lrr", "gto", "oldest"]
if mode == "run":
    policies.append("gto_limit4")

base_counters = {
    "idle": r"\bW0_Idle:(\d+)",
    "scoreboard": r"\bW0_Scoreboard:(\d+)",
    "ldst_stalls": r"^gpgpu_n_stall_shd_mem[ \t]*=[ \t]*(\d+)",
}
variant_counters = {
    "limit_active_cycles": r"^gto_limit_active_cycles[ \t]*=[ \t]*(\d+)",
    "limit_other_warp_cycles": r"^gto_limit_other_warp_cycles[ \t]*=[ \t]*(\d+)",
}

def extract(path, workload, policy):
    def field(pattern, block):
        hit = re.search(pattern, block, re.MULTILINE)
        if not hit:
            raise ValueError(f"{path}, launch {launch}: missing {pattern}")
        return hit.group(1)

    blocks = re.split(r"(?m)^kernel_name[ \t]*=[ \t]*", path.read_text(errors="replace"))[1:]
    if not blocks:
        raise ValueError(f"{path}: no kernel stats blocks")
    counters = dict(base_counters)
    if policy == "gto_limit4":
        counters.update(variant_counters)
    previous = dict.fromkeys(counters, 0)
    total_cycles = 0
    total_instructions = 0
    rows = []
    for launch, block in enumerate(blocks, 1):
        kernel = block.splitlines()[0].strip()
        if len(kernel.split()) != 1:
            raise ValueError(f"{path}, launch {launch}: expected one kernel, got {kernel!r}")
        cycles = int(field(r"^gpu_sim_cycle[ \t]*=[ \t]*(\d+)", block))
        instructions = int(field(r"^gpu_sim_insn[ \t]*=[ \t]*(\d+)", block))
        total_cycles += cycles
        total_instructions += instructions
        reported_cycles = int(field(r"^gpu_tot_sim_cycle[ \t]*=[ \t]*(\d+)", block))
        reported_instructions = int(field(r"^gpu_tot_sim_insn[ \t]*=[ \t]*(\d+)", block))
        if (cycles <= 0 or instructions <= 0 or
                (reported_cycles, reported_instructions) != (total_cycles, total_instructions)):
            raise ValueError(f"{path}, launch {launch}: inconsistent simulated totals")
        current = {name: int(field(pattern, block)) for name, pattern in counters.items()}
        deltas = {name: value - previous[name] for name, value in current.items()}
        if min(deltas.values()) < 0:
            raise ValueError(f"{path}, launch {launch}: cumulative counter decreased")
        if policy == "gto_limit4":
            limit = field(r"^gto_greedy_issue_limit[ \t]*=[ \t]*(\d+)", block)
            if limit != "4" or deltas["limit_other_warp_cycles"] > deltas["limit_active_cycles"]:
                raise ValueError(f"{path}, launch {launch}: invalid limit counters")
        rows.append(dict(workload=workload, policy=policy, launch=launch,
                         kernel=kernel, cycles=cycles, instructions=instructions, **deltas))
        previous = current
    return rows

def main():
    with validated_file.open(newline="") as stream:
        expected = {(r["workload"], r["scheduler"]): r for r in csv.DictReader(stream)
                    if r["status"] == "PASS"}
    rows = []
    for workload in workloads:
        reference_launches = None
        for policy in policies:
            launches = extract(log_dir / f"{workload}-{policy}.log", workload, policy)
            signature = [(r["kernel"], r["instructions"]) for r in launches]
            if reference_launches is None:
                reference_launches = signature
            if signature != reference_launches:
                raise ValueError(f"{workload}/{policy}: launch names or instruction counts differ")
            baseline = expected[workload, "lrr" if policy == "gto_limit4" else policy]
            cycles = sum(row["cycles"] for row in launches)
            instructions = sum(row["instructions"] for row in launches)
            if (len(launches), instructions) != (int(baseline["stat_blocks"]),
                                                 int(baseline["instructions"])):
                raise ValueError(f"{workload}/{policy}: launch count or work differs from feasibility run")
            if policy != "gto_limit4" and cycles != int(baseline["cycles"]):
                message = f"{workload}/{policy}: {cycles} cycles, feasibility run had {baseline['cycles']}"
                if mode == "collect":
                    raise ValueError(message)
                print(f"WARNING: {message}. Use the controls from this run.", file=sys.stderr)
            rows.extend(launches)

    fields = ["workload", "policy", "launch", "kernel", "cycles", "instructions",
              *base_counters, *variant_counters]
    with NamedTemporaryFile(mode="w", dir=output.parent, newline="", encoding="utf-8",
                            delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.close()
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"{output} ({len(rows)} launch records)")

try:
    main()
except (OSError, ValueError, KeyError) as exc:
    sys.exit(str(exc))
PY
}

mode="${1:---help}"
case "$mode" in
  -h|--help) usage; exit 0 ;;
  collect|run) shift ;;
  *) usage >&2; exit 2 ;;
esac

validated="$project_root/results/feasibility/results_validated.csv"
[[ -r "$validated" ]] || { printf 'Missing %s\n' "$validated" >&2; exit 1; }

if [[ "$mode" == collect ]]; then
  [[ $# -eq 0 ]] || { usage >&2; exit 2; }
  mkdir -p "$results"
  collect_launches "$project_root/results/feasibility" "$results/launch_counters.csv" \
    "$validated" collect backprop nn heartwall hotspot
  exit
fi

workloads=("$@")
if [[ ${#workloads[@]} -eq 0 ]]; then
  workloads=(nn hotspot)
fi
for workload in "${workloads[@]}"; do
  case "$workload" in
    backprop|nn|heartwall|hotspot) ;;
    *) printf 'Unknown workload: %s\n' "$workload" >&2; exit 2 ;;
  esac
done

framework="$project_root/accel-sim-framework"
simulator="$framework/gpu-simulator/bin/release/accel-sim.out"
trace_root="$framework/hw_run/rodinia_2.0-ft/11.0"
trace_config="$framework/gpu-simulator/configs/tested-cfgs/SM7_QV100/trace.config"
[[ -x "$simulator" && -d "$trace_root" && -r "$trace_config" ]] || {
  printf 'Missing simulator, Rodinia traces, or trace configuration under %s\n' "$framework" >&2
  exit 1
}
if [[ -z "${ACCELSIM_SETUP_ENVIRONMENT_WAS_RUN:-}" || "${ACCELSIM_CONFIG:-}" != release ]]; then
  source "$framework/gpu-simulator/setup_environment.sh" release || exit 1
fi
gpu_config="${GPGPUSIM_ROOT:?Accel-Sim setup did not set GPGPUSIM_ROOT}/configs/tested-cfgs/SM7_QV100/gpgpusim.config"
[[ -r "$gpu_config" ]] || { printf 'Missing %s\n' "$gpu_config" >&2; exit 1; }

mkdir -p "$results"
run_dir="$(mktemp -d "$results/run.XXXXXXXX")"
printf 'Run directory: %s\n' "$run_dir"
for workload in "${workloads[@]}"; do
  trace=""
  # Use the same first matching trace as the feasibility script.
  read -r -d '' trace < <(find "$trace_root" \
    -path "*${workload}-rodinia-2.0-ft*/traces/kernelslist.g" -print0 -quit) || true
  [[ -n "$trace" ]] || { printf 'No trace manifest for %s\n' "$workload" >&2; exit 1; }
  for policy in gto_limit4 gto oldest lrr; do
    name="$workload-$policy"
    mkdir -p "$run_dir/work/$name"
    printf 'Running %s\n' "$name"
    if ! (
      cd "$run_dir/work/$name" || exit 1
      TIMEFORMAT='Wall time: %3R seconds'
      {
        time "$simulator" -trace "$trace" -config "$gpu_config" \
          -config "$trace_config" -gpgpu_scheduler "$policy" > "$run_dir/$name.log"
      } 2> "$run_dir/$name.time"
    ); then
      printf 'Simulator failed. See %s/%s.log and .time.\n' "$run_dir" "$name" >&2
      exit 1
    fi
    if [[ "$policy" == gto_limit4 ]] && ! grep -qx 'gto_greedy_issue_limit = 4' "$run_dir/$name.log"; then
      printf 'The simulator did not report GTO limit 4. Apply the patch and rebuild.\n' >&2
      exit 1
    fi
  done
done
collect_launches "$run_dir" "$run_dir/launch_counters.csv" "$validated" run "${workloads[@]}"
