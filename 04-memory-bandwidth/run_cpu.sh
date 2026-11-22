#!/usr/bin/env bash
# Builds cpu_stream with several compiler flag sets, then sweeps threads and placement.
# Writes results/cpu_<host>.csv. Usage: ./run_cpu.sh [ELEMENTS]
set -euo pipefail
cd "$(dirname "$0")"
N=${1:-134217728}
OUT=results/cpu_$(hostname | cut -c1-12).csv
NCPU=$(nproc)
lscpu | grep -E "Model name|^CPU\(s\)|Thread\(s\)|Core\(s\)|L3|NUMA node\(s\)" >&2
echo "label,threads,bind,kernel,best_gbps,median_gbps" > "$OUT"

# 1. Compiler flags, one thread: how code generation alone changes bandwidth.
declare -A FLAGS=(
  [O0]="-O0" [O2]="-O2" [O3]="-O3" [O3_native]="-O3 -march=native"
  [O3_native_freestanding]="-O3 -march=native -ffreestanding"
)
for name in O0 O2 O3 O3_native O3_native_freestanding; do
  gcc ${FLAGS[$name]} -fopenmp cpu_stream.c -o /tmp/cpu_stream_$name -lm
  OMP_NUM_THREADS=1 /tmp/cpu_stream_$name "$N" "$name" >> "$OUT"
done

# 2. Threads and placement, with the -O3 -march=native build.
for bind in close spread; do
  for t in $(seq 1 "$NCPU"); do
    OMP_NUM_THREADS=$t OMP_PROC_BIND=$bind OMP_PLACES=cores /tmp/cpu_stream_O3_native "$N" "O3_native" >> "$OUT"
  done
done
echo "wrote $OUT" >&2
