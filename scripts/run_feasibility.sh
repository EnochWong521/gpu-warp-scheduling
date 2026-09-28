#!/bin/bash

# ============================================================
# GPU Warp Scheduling Feasibility Sweep
#
# Runs:
#   10 Rodinia workloads
#   x 7 warp scheduler configurations
#   = 70 simulations
#
# Important:
#   Each simulation runs in its own working directory so that
#   GPGPU-Sim-generated auxiliary files do not clutter the
#   project root.
# ============================================================

# Do not exit the entire sweep just because one simulation fails.
# Treat failures in pipelines correctly.
set -o pipefail


# ============================================================
# Project paths
# ============================================================

# This script lives in:
#   <project>/scripts/run_feasibility.sh
#
# Therefore ".." from this script directory is the project root.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ACCELSIM_ROOT="$PROJECT_ROOT/accel-sim-framework"

TRACE_ROOT="$ACCELSIM_ROOT/hw_run/rodinia_2.0-ft/11.0"

RESULTS="$PROJECT_ROOT/results/feasibility"

# Simulator-generated auxiliary files will go here.
WORK_ROOT="$RESULTS/work"

SIM="$ACCELSIM_ROOT/gpu-simulator/bin/release/accel-sim.out"

TRACE_CONFIG="$ACCELSIM_ROOT/gpu-simulator/configs/tested-cfgs/SM7_QV100/trace.config"


# ============================================================
# Initial checks
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
    "bfs"
    "nw"
    "nn"
    "lud"
    "srad_v2"
    "hotspot"
    "backprop"
    "heartwall"
    "pathfinder"
    "streamcluster"
)


# ============================================================
# Scheduler configurations
#
# sched_names:
#   Short names used in filenames/results.
#
# sched_configs:
#   Actual strings passed to GPGPU-Sim.
#
# For warp_limiting:
#
#   warp_limiting:<prioritization>:<warp_limit>
#
# Prioritization 2 = GTO.
# ============================================================

sched_names=(
    "lrr"
    "gto"
    "rrr"
    "oldest"
    "wl4"
    "wl8"
    "wl16"
)

sched_configs=(
    "lrr"
    "gto"
    "rrr"
    "oldest"
    "warp_limiting:2:4"
    "warp_limiting:2:8"
    "warp_limiting:2:16"
)


# Make sure the arrays correspond one-to-one.
if [ "${#sched_names[@]}" -ne "${#sched_configs[@]}" ]; then
    echo "ERROR: Scheduler arrays have different lengths."
    exit 1
fi


# ============================================================
# Save reproducibility information
# ============================================================

METADATA="$RESULTS/metadata.txt"

{
    echo "GPU Warp Scheduling Feasibility Sweep"
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
echo "GPU Warp Scheduling Feasibility Sweep"
echo "=========================================="
echo "Workloads:   ${#apps[@]}"
echo "Schedulers: ${#sched_names[@]}"
echo "Total runs:  $TOTAL_RUNS"
echo "Results:     $RESULTS"
echo "=========================================="
echo


for app in "${apps[@]}"
do

    # Find the trace manifest for this workload.
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

        # Give this simulation its own working directory.
        workdir="$WORK_ROOT/${app}-${name}"

        # Remove leftovers from an earlier run of the same experiment.
        rm -rf "$workdir"
        mkdir -p "$workdir"

        echo "[$run_number/$TOTAL_RUNS]"
        echo "------------------------------------------"
        echo "Workload:  $app"
        echo "Scheduler: $name"
        echo "Config:    $config"
        echo "------------------------------------------"

        # ----------------------------------------------------
        # Run from the private working directory.
        #
        # Files automatically generated by GPGPU-Sim, such as:
        #
        #   perf_counter*.csv.gz
        #   gpgpu_inst_stats.txt
        #   checkpoint_files/
        #
        # will now appear under:
        #
        #   results/feasibility/work/<app>-<scheduler>/
        #
        # instead of cluttering the project root.
        # ----------------------------------------------------

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
echo "Sweep complete"
echo "=========================================="
echo
echo "CSV:"
echo "  $CSV"
echo
echo "Metadata:"
echo "  $METADATA"
echo
echo "Simulator auxiliary files:"
echo "  $WORK_ROOT"
echo
