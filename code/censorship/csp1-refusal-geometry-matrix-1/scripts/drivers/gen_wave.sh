#!/bin/bash
# gen_wave.sh <driver-name> <group1> <group2> ... ; each group = comma-separated subjects (<=2, so that
# three disjoint drivers together hold <= 6 hot samplers, per DESIGN). Three passes for resumable retries.
set -uo pipefail
set -a; source /path/to/home/.env; set +a
unset OPENAI_API_KEY_PERSONAL_BACKUP OPENROUTER_API_KEY_PERSONAL_BACKUP
NAME=$1; shift
W=/path/to/home/work/csp1-refusal-geometry-matrix-1; export RGM1_WORK=$W
S=/path/to/home/ws/run/csp1-refusal-geometry-matrix-1/registry/csp1-refusal-geometry-matrix-1/scripts
PY=/path/to/home/venvs/tinker-cd1/bin/python
cd $S
for pass in 1 2 3; do
  for g in "$@"; do
    echo ">>> [$(date -u +%H:%M:%S)] $NAME pass $pass group $g"
    $PY -u generate_rollouts.py --stage final --subjects "$g" --concurrency 300 >> $W/logs/gen_final_$NAME.log 2>&1
    echo ">>> [$(date -u +%H:%M:%S)] $NAME group $g rc=$?"
    date -u +%Y-%m-%dT%H:%M:%SZ > $W/logs/gen_final_${NAME}_heartbeat
  done
done
echo ">>> [$(date -u +%H:%M:%S)] $NAME WAVE COMPLETE"
touch $W/final/gen/.wave_$NAME.done
