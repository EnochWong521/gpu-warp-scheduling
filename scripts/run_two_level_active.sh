#!/bin/bash

# ============================================================
# Two-Level Active (TLA) Scheduler Sweep
#
# Runs:
#   3 representative Rodinia workloads
#   x 3 TLA active-warp configurations
#   = 9 simulations
#
# Workloads:
#   backprop -> scheduler-sensitive
#   nn       -> Oldest/GTO performance difference
#   nw       -> scheduler-insensitive control
#
# TLA format:
#   two_level_active:<max_active_warps>:<inner_policy>:<outer_policy>
#
# Current GPGPU-Sim implementation:
#   inner policy 0 = LRR
#   outer policy 1 = SRR
# ============================================================

set -o pipefail


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ACCELSIM_ROOT="$PROJECT_ROOT/accel-sim-framework"

TRACE_ROOT="$ACCELSIM_ROOT/hw_run/rodinia_2.0-ft/11.0"

RESULTS="$PROJECT_ROOT/results/tla"

WORK_ROOT="$RESULTS/work"

SIM="$ACCELSIM_ROOT/gpu-simulator/bin/release/accel-sim.out"

TRACE_CONFIG="$ACCELSIM_ROOT/gpu-simulator/configs/tested-cfgs/SM7_QV100/trace.config"


# ============================================================
# Initial setup
# ============================================================

mkdir -p "$RESULTS"
mkdir -p "$WORK_ROOT"

if [ ! -f "$SIM" ]; then
    echo "ERROR: Accel-Sim executable not found:"
    echo "$SIM"
    exit 1
fi

if [ ! -d "$TRACE_ROOT" ]; then
    echo "ERROR: Trace directory not found:"
    echo "$TRACE_ROOT"
    exit 1
fi


# ============================================================
# Load Accel-Sim / GPGPU-Sim environment
# ============================================================

if [ -z "${GPGPUSIM_ROOT:-}" ]; then
    echo "Loading Accel-Sim environment..."

    source "$ACCELSIM_ROOT/gpu-simulator/setup_environment.sh"

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to load Accel-Sim environment."
        exit 1
    fi
fi

GPU_CONFIG="$GPGPUSIM_ROOT/configs/tested-cfgs/SM7_QV100/gpgpusim.config"

if [ ! -f "$GPU_CONFIG" ]; then
    echo "ERROR: GPU configuration not found:"
    echo "$GPU_CONFIG"
    exit 1
fi


# ============================================================
# Workloads
# ============================================================

apps=(
    "backprop"
    "nn"
    "nw"
)


# ============================================================
# Two-Level Active configurations
#
# Format:
#   two_level_active:<max_active>:<inner>:<outer>
#
# inner = 0 -> LRR
# outer = 1 -> SRR
# ============================================================

sched_names=(
    "tla4"
    "tla8"
    "tla16"
)

sched_configs=(
    "two_level_active:4:0:1"
    "two_level_active:8:0:1"
    "two_level_active:16:0:1"
)


if [ "${#sched_names[@]}" -ne "${#sched_configs[@]}" ]; then
    echo "ERROR: Scheduler arrays have different lengths."
    exit 1
fi


# ============================================================
# Save reproducibility information
# ============================================================

METADATA="$RESULTS/metadata.txt"

{
    echo "Two-Level Active Scheduler Sweep"
    echo
    echo "Date: $(date)"
    echo "Project root: $PROJECT_ROOT"
    echo
    echo "Accel-Sim:"
    echo "  $(git -C "$ACCELSIM_ROOT" rev-parse HEAD 2>/dev/null)"
    echo
    echo "GPGPU-Sim:"
    echo "  $(git -C "$GPGPUSIM_ROOT" rev-parse HEAD 2>/dev/null)"
    echo
    echo "GPU config:"
    echo "  $GPU_CONFIG"
    echo
    echo "Trace config:"
    echo "  $TRACE_CONFIG"
    echo
    echo "Trace set:"
    echo "  tesla-v100/rodinia_2.0-ft"
} > "$METADATA"


# ============================================================
# Run simulations
# ============================================================

TOTAL_RUNS=$(( ${#apps[@]} * ${#sched_names[@]} ))
run_number=0

echo
echo "=========================================="
echo "Two-Level Active Scheduler Sweep"
echo "=========================================="
echo "Workloads:      ${#apps[@]}"
echo "Configurations: ${#sched_names[@]}"
echo "Total runs:     $TOTAL_RUNS"
echo "Results:        $RESULTS"
echo "=========================================="
echo


for app in "${apps[@]}"
do

    trace=$(find "$TRACE_ROOT" \
        -path "*${app}-rodinia-2.0-ft*/traces/kernelslist.g" \
        | head -1)

    if [ -z "$trace" ]; then
        echo "ERROR: Could not find trace for $app"
        echo
        continue
    fi


    for i in "${!sched_names[@]}"
    do

        name="${sched_names[$i]}"
        config="${sched_configs[$i]}"

        run_number=$((run_number + 1))

        log="$RESULTS/${app}-${name}.log"
        timefile="$RESULTS/${app}-${name}.time"

        workdir="$WORK_ROOT/${app}-${name}"

        rm -rf "$workdir"
        mkdir -p "$workdir"

        echo "[$run_number/$TOTAL_RUNS]"
        echo "------------------------------------------"
        echo "Workload:  $app"
        echo "Scheduler: $name"
        echo "Config:    $config"
        echo "------------------------------------------"

        (
            cd "$workdir" || exit 1

            /usr/bin/time -v \
                "$SIM" \
                    -trace "$trace" \
                    -config "$GPU_CONFIG" \
                    -config "$TRACE_CONFIG" \
                    -gpgpu_scheduler "$config" \
                    > "$log" \
                    2> "$timefile"
        )

        status=$?

        if [ "$status" -eq 0 ]; then
            echo "PASS: $app / $name"
        else
            echo "FAIL: $app / $name (exit code $status)"
        fi

        echo

    done
done


# ============================================================
# Generate summary CSV
# ============================================================

CSV="$RESULTS/results.csv"

echo \
"workload,scheduler,scheduler_config,status,cycles,instructions,ipc,wall_time,max_rss_kb" \
> "$CSV"


for app in "${apps[@]}"
do

    for i in "${!sched_names[@]}"
    do

        name="${sched_names[$i]}"
        config="${sched_configs[$i]}"

        log="$RESULTS/${app}-${name}.log"
        timefile="$RESULTS/${app}-${name}.time"

        if [ ! -f "$log" ]; then

            echo \
"$app,$name,$config,MISSING,,,,," \
>> "$CSV"

            continue
        fi


        cycles=$(grep "^gpu_sim_cycle =" "$log" \
            | tail -1 \
            | awk '{print $3}')

        instructions=$(grep "^gpu_sim_insn =" "$log" \
            | tail -1 \
            | awk '{print $3}')

        ipc=$(grep "^gpu_ipc =" "$log" \
            | tail -1 \
            | awk '{print $3}')


        if [ -f "$timefile" ]; then

            wall_time=$(grep "Elapsed (wall clock) time" "$timefile" \
                | sed 's/^[[:space:]]*Elapsed (wall clock) time (h:mm:ss or m:ss): //')

            max_rss=$(grep "Maximum resident set size" "$timefile" \
                | awk '{print $6}')

        else
            wall_time=""
            max_rss=""
        fi


        if [ -n "$cycles" ] && \
           [ -n "$instructions" ] && \
           [ -n "$ipc" ]; then

            status="PASS"

        else
            status="FAIL"
        fi


        echo \
"$app,$name,$config,$status,$cycles,$instructions,$ipc,$wall_time,$max_rss" \
>> "$CSV"

    done
done


# ============================================================
# Print results
# ============================================================

echo
echo "=========================================="
echo "RESULTS"
echo "=========================================="
echo

column -s, -t "$CSV" 2>/dev/null || cat "$CSV"

echo
echo "=========================================="
echo "TLA sweep complete"
echo "=========================================="
echo
echo "CSV:"
echo "  $CSV"
echo
echo "Raw logs:"
echo "  $RESULTS"
echo
echo "Simulator auxiliary files:"
echo "  $WORK_ROOT"
echo
