#!/bin/bash
# hot_probe.sh <phase-tag> <per-driver-hot> <rows-per-driver>
# One phase of the DESIGN § Schedule and cost throughput ladder: three parallel generation drivers,
# each holding <per-driver-hot> hot samplers, each capped at <rows-per-driver> NEW rows. The rows
# are real and stay in the resumable shards — nothing is thrown away. Records throughput, throttles
# and failures per group in receipts/generation_summaries.jsonl for pick_hot.py to read.
set -uo pipefail
set -a; source /path/to/home/.env; set +a
unset OPENAI_API_KEY_PERSONAL_BACKUP OPENROUTER_API_KEY_PERSONAL_BACKUP
PHASE=$1; HOT=$2; ROWS=$3
W=/path/to/home/work/csp1-refusal-geometry-complete16-1; export RGC16_WORK=$W
R=/path/to/home/ws/run/csp1-refusal-geometry-complete16-1/registry/csp1-refusal-geometry-complete16-1
export PYTHONPATH=$R/scripts
PY=/path/to/home/venvs/tinker-cd1/bin/python
STAGE=session1
cd "$R/scripts"
mapfile -t GRP < <("$PY" subject_split.py --hot "$HOT")
echo ">>> [$(date -u +%H:%M:%S)] phase $PHASE: hot=$HOT/driver, ${#GRP[@]} drivers, $ROWS rows each"
PIDS=()
for i in "${!GRP[@]}"; do
  [ -z "${GRP[$i]}" ] && continue
  nohup "$PY" -u generate_rollouts.py --stage "$STAGE" --subjects "${GRP[$i]}" --max-hot "$HOT" \
      --concurrency 300 --max-rows "$ROWS" --phase "$PHASE" \
      >> "$W/logs/hot_probe_${PHASE}_d$i.log" 2>&1 &
  PIDS+=($!)
  echo ">>> launched driver $i pid=${PIDS[-1]} subjects=${GRP[$i]}"
done
for p in "${PIDS[@]}"; do wait "$p"; done
echo ">>> [$(date -u +%H:%M:%S)] phase $PHASE COMPLETE"
