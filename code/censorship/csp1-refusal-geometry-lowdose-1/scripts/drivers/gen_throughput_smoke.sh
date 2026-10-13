#!/bin/bash
# DESIGN-pinned 6-vs-12 hot-sampler throughput smoke, run on REAL wave rows (near_miss_innocent,
# 120 rows/subject) so nothing is wasted. Phase A: 3 drivers x 2 hot (6 total). Phase B: 3 drivers
# x 4 hot (12 total). Compares aggregate rows/s and throttle/error counts; writes the receipt the
# full wave's --hot setting is taken from.
set -uo pipefail
W=/path/to/home/work/csp1-refusal-geometry-lowdose-1
R=/path/to/home/ws/run/csp1-refusal-geometry-lowdose-1/registry/csp1-refusal-geometry-lowdose-1
D=$R/scripts/drivers
mark(){ date -u +%s; }
phase(){ # <label> <hot> <A-subjects> <B-subjects> <C-subjects>
  local label=$1 hot=$2 a=$3 b=$4 c=$5 t0 t1
  echo ">>> phase $label hot=$hot"
  t0=$(mark)
  nohup bash $D/gen_wave.sh "smoke${label}A" "$hot" "$a" near_miss_innocent </dev/null > $W/logs/smoke_gen_${label}A.log 2>&1 &
  p1=$!
  nohup bash $D/gen_wave.sh "smoke${label}B" "$hot" "$b" near_miss_innocent </dev/null > $W/logs/smoke_gen_${label}B.log 2>&1 &
  p2=$!
  nohup bash $D/gen_wave.sh "smoke${label}C" "$hot" "$c" near_miss_innocent </dev/null > $W/logs/smoke_gen_${label}C.log 2>&1 &
  p3=$!
  wait $p1 $p2 $p3
  t1=$(mark)
  echo "PHASE_${label}_ELAPSED_S=$((t1-t0))"
}
# Phase A: 6 hot (3 x 2)
phase A 2 "stock,rgm1_china_s3" "rgm1_china_s4,matched_filler256_s3" "matched_filler256_s4,china_d40_s3"
# Phase B: 12 hot (3 x 4)
phase B 4 "china_d40_s4,china_d64_s3,china_d64_s4,china_d128_s3" \
          "china_d128_s4,other_authoritarian_d40_s3,other_authoritarian_d40_s4,other_authoritarian_d64_s3" \
          "other_authoritarian_d64_s4,other_authoritarian_d128_s3,other_authoritarian_d128_s4,adult_sexual_d40_s3"
echo ">>> THROUGHPUT SMOKE COMPLETE"
touch $W/final/gen/.throughput_smoke.done
