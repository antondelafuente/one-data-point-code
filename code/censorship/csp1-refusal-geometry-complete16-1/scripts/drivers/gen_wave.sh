#!/bin/bash
# gen_wave.sh <stage> <driver-name> <per-driver-hot> <comma-list-of-this-driver's-subjects> [probes-csv]
# Chunks the subject list into groups of <per-driver-hot> and runs three resumable passes.
# Three disjoint drivers x <per-driver-hot> = the wave's total hot-sampler count. RGTD1 runs at
# RGLD1's measured setting (6 total = 3 x 2): RGLD1's own 500-row smoke measured 6 hot at
# 17.91 rows/s vs 12 hot at 17.41, zero throttles both, and kept 6.
set -uo pipefail
set -a; source /path/to/home/.env; set +a
unset OPENAI_API_KEY_PERSONAL_BACKUP OPENROUTER_API_KEY_PERSONAL_BACKUP
STAGE=$1; NAME=$2; HOT=$3; SUBJ=$4; PROBES=${5:-}
W=/path/to/home/work/csp1-refusal-geometry-complete16-1; export RGC16_WORK=$W
R=/path/to/home/ws/run/csp1-refusal-geometry-complete16-1/registry/csp1-refusal-geometry-complete16-1
export PYTHONPATH=$R/scripts
PY=/path/to/home/venvs/tinker-cd1/bin/python
cd $R/scripts
mapfile -t ALL < <(echo "$SUBJ" | tr ',' '\n' | sed '/^$/d')
SUBJ_CHUNKS=()          # NOTE: never name this GROUPS -- bash special variable (RGLD1 retro)
g=""
n=0
for s in "${ALL[@]}"; do
  g="${g:+$g,}$s"; n=$((n+1))
  if [ "$n" -eq "$HOT" ]; then SUBJ_CHUNKS+=("$g"); g=""; n=0; fi
done
[ -n "$g" ] && SUBJ_CHUNKS+=("$g")
echo ">>> [$NAME/$STAGE] hot=$HOT groups=${#SUBJ_CHUNKS[@]} subjects=${#ALL[@]}"
PROBEARG=""; [ -n "$PROBES" ] && PROBEARG="--probes $PROBES"
for pass in 1 2 3; do
  for grp in "${SUBJ_CHUNKS[@]}"; do
    echo ">>> [$(date -u +%H:%M:%S)] $NAME pass $pass group $grp"
    $PY -u generate_rollouts.py --stage "$STAGE" --subjects "$grp" --max-hot "$HOT" --concurrency 300 $PROBEARG >> $W/logs/gen_${STAGE}_$NAME.log 2>&1
    echo ">>> [$(date -u +%H:%M:%S)] $NAME group done rc=$?"
    date -u +%Y-%m-%dT%H:%M:%SZ > $W/logs/gen_${STAGE}_${NAME}_heartbeat
  done
done
echo ">>> [$(date -u +%H:%M:%S)] $NAME WAVE COMPLETE"
touch $W/$STAGE/gen/.wave_$NAME.done
