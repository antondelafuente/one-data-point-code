#!/bin/bash
# gen_wave.sh <driver-name> <per-driver-hot> <comma-list-of-this-driver's-subjects> [probes-csv]
# Chunks the subject list into groups of <per-driver-hot> and runs three resumable passes.
# Three disjoint drivers x <per-driver-hot> = the wave's total hot-sampler count (DESIGN: measure
# 6 vs 12 in a throughput smoke, then run at the measured better non-throttling setting).
set -uo pipefail
set -a; source /path/to/home/.env; set +a
unset OPENAI_API_KEY_PERSONAL_BACKUP OPENROUTER_API_KEY_PERSONAL_BACKUP
NAME=$1; HOT=$2; SUBJ=$3; PROBES=${4:-}
W=/path/to/home/work/csp1-refusal-geometry-lowdose-1; export RGLD1_WORK=$W
R=/path/to/home/ws/run/csp1-refusal-geometry-lowdose-1/registry/csp1-refusal-geometry-lowdose-1
export PYTHONPATH=$R/scripts
PY=/path/to/home/venvs/tinker-cd1/bin/python
cd $R/scripts
mapfile -t ALL < <(echo "$SUBJ" | tr ',' '\n' | sed '/^$/d')
CHUNKS=()
g=""
n=0
for s in "${ALL[@]}"; do
  g="${g:+$g,}$s"; n=$((n+1))
  if [ "$n" -eq "$HOT" ]; then CHUNKS+=("$g"); g=""; n=0; fi
done
[ -n "$g" ] && CHUNKS+=("$g")
:
echo ">>> [$NAME] hot=$HOT groups=${#CHUNKS[@]} subjects=${#ALL[@]}"
PROBEARG=""; [ -n "$PROBES" ] && PROBEARG="--probes $PROBES"
for pass in 1 2 3; do
  for grp in "${CHUNKS[@]}"; do
    echo ">>> [$(date -u +%H:%M:%S)] $NAME pass $pass group $grp"
    $PY -u generate_rollouts.py --stage final --subjects "$grp" --max-hot "$HOT" --concurrency 300 $PROBEARG >> $W/logs/gen_final_$NAME.log 2>&1
    echo ">>> [$(date -u +%H:%M:%S)] $NAME group done rc=$?"
    date -u +%Y-%m-%dT%H:%M:%SZ > $W/logs/gen_final_${NAME}_heartbeat
  done
done
echo ">>> [$(date -u +%H:%M:%S)] $NAME WAVE COMPLETE"
touch $W/final/gen/.wave_$NAME.done
