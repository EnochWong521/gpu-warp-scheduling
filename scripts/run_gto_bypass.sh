#!/usr/bin/env bash
# In a clean GPGPU-Sim checkout, apply gto_limit4.patch, then gto_bypass.patch.
# For instruction pairs, also apply gto_pairs.patch and use the pairs mode.
# bash scripts/run_gto_bypass.sh build  # rebuild after applying the patches
# bash scripts/run_gto_bypass.sh run    # writes logs and a validated CSV to a fresh directory
set -eo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
sim="$root/accel-sim-framework/gpu-simulator"
source "$sim/setup_environment.sh" release
case "${1:-run}" in
  build)
    export CPLUS_INCLUDE_PATH="$sim/../util/tracer_nvbit/tracer_tool${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
    # makedepend misses GPGPU-Sim headers included by trace_driven.cc.
    mkdir -p "$sim/build/release"
    make -C "$sim/trace-driven" -W trace_driven.cc 'CXX=g++ -Wl,--no-as-needed'
    exec make -C "$sim" -j1 'CXX=g++ -Wl,--no-as-needed'
    ;;
  run|pairs) ;;
  *) echo "Usage: bash scripts/run_gto_bypass.sh [build|run|pairs]" >&2; exit 2 ;;
esac

traces="$sim/../hw_run/rodinia_2.0-ft/11.0"
results="$root/results/adaptive_policy"
run=$(mktemp -d "$results/bypass.XXXXXXXX")
echo "Run directory: $run"
for workload in nn hotspot; do
  manifests=("$traces/$workload-rodinia-2.0-ft"/*/traces/kernelslist.g)
  [[ ${#manifests[@]} == 1 && -r ${manifests[0]} ]] || {
    echo "Expected one trace manifest for $workload" >&2; exit 1
  }
  echo "Running $workload"
  mkdir -p "$run/work/$workload"
  (
    cd "$run/work/$workload"
    "$sim/bin/release/accel-sim.out" -trace "${manifests[0]}" \
      -config "$GPGPUSIM_ROOT/configs/tested-cfgs/SM7_QV100/gpgpusim.config" \
      -config "$sim/configs/tested-cfgs/SM7_QV100/trace.config" \
      -gpgpu_scheduler gto -gpgpu_gto_bypass_probe 1 > "$run/$workload.log"
  ) 2> "$run/$workload.time"
done
python3 "$root/scripts/collect_gto_bypass.py" "$run" "$results/nn_hotspot_limit4.csv"
if [[ ${1:-run} == pairs ]]; then
  python3 "$root/scripts/collect_gto_pairs.py" "$run"
fi
