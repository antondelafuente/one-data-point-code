#!/bin/bash
# train_wave.sh: launch the 6 SRD2 trains concurrently (not disowned) and wait on their PIDs.
set -uo pipefail
source "$(dirname "$0")/env.sh"
D=$R/scripts/drivers
PIDS=()
for CAT in other_authoritarian adult_sexual drugs; do for K in 16 40; do
  KEY=${CAT}_single_d${K}_s3
  [ -e "$W/train_done/$KEY.subject.done" ] && continue
  nohup bash "$D/train_one.sh" $CAT $K >> "$W/logs/train_driver_$KEY.log" 2>&1 &
  PIDS+=($!); echo ">>> launched $KEY pid=$! ($(date -u +%H:%M:%S))"
done; done
echo ">>> launched ${#PIDS[@]} trains"
for p in "${PIDS[@]}"; do wait "$p"; done
n=$(ls $W/train_done/*.subject.done 2>/dev/null | wc -l)
echo ">>> [$(date -u +%H:%M:%S)] TRAIN WAVE EXIT: $n/6 subject markers"
[ "$n" -eq 6 ] && touch $W/train_done/.wave.done
