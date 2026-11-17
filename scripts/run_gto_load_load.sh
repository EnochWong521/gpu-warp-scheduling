#!/usr/bin/env bash
# Apply gto_load_load.patch on top of gto_pairs.patch, then build with
# bash scripts/run_gto_bypass.sh build. Run this script from the same checkout.
set -eo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
sim="$root/accel-sim-framework/gpu-simulator"
source "$sim/setup_environment.sh" release
results="$root/results/adaptive_policy"
run=$(mktemp -d "$results/load_load.XXXXXXXX")
echo "Run directory: $run"
for workload in nn hotspot; do
  manifests=("$sim/../hw_run/rodinia_2.0-ft/11.0/$workload-rodinia-2.0-ft"/*/traces/kernelslist.g)
  [[ ${#manifests[@]} == 1 && -r ${manifests[0]} ]] || {
    echo "Expected one trace manifest for $workload" >&2; exit 1
  }
  for enabled in 0 1; do
    name="$workload-$enabled"
    echo "Running $name"
    mkdir -p "$run/work/$name"
    (
      cd "$run/work/$name"
      "$sim/bin/release/accel-sim.out" -trace "${manifests[0]}" \
        -config "$GPGPUSIM_ROOT/configs/tested-cfgs/SM7_QV100/gpgpusim.config" \
        -config "$sim/configs/tested-cfgs/SM7_QV100/trace.config" \
        -gpgpu_scheduler gto -gpgpu_gto_load_load "$enabled" > "$run/$name.log"
    ) 2> "$run/$name.time"
  done
done
python3 "$root/scripts/collect_gto_load_load.py" "$run"
