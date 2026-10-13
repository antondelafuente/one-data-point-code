#!/bin/bash
# train_wave.sh: launch the 3 SRD1 trains concurrently (not disowned) and wait on their PIDs.
set -uo pipefail
source "$(dirname "$0")/env.sh"
D=$R/scripts/drivers
PIDS=()
for K in 16 40 100; do
  [ -e "$W/train_done/single8_d${K}_s3.subject.done" ] && continue
  nohup bash "$D/train_one.sh" $K >> "$W/logs/train_driver_single8_d${K}_s3.log" 2>&1 &
  PIDS+=($!); echo ">>> launched K=$K pid=$! ($(date -u +%H:%M:%S))"
done
for p in "${PIDS[@]}"; do wait "$p"; done
n=$(ls $W/train_done/*.subject.done 2>/dev/null | wc -l)
echo ">>> [$(date -u +%H:%M:%S)] TRAIN WAVE EXIT: $n/3 subject markers"
[ "$n" -eq 3 ] && touch $W/train_done/.wave.done
