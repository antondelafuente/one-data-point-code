#!/bin/bash
# em-gemma4-singletons-1 per-pod driver: one trajectory (ROW, SEED) end to end on one H100.
# Env: ROW SEED; BENCH=1 runs the engineering B pilot + choose (first pod only); else B=<frozen>; PROBE=1 adds the
# unjudged reproducibility probe. Stages are success-marker gated (idempotent); uploads go pod -> R2 directly.
set -uo pipefail
EXP=em-gemma4-singletons-1
P=/workspace/$EXP; OUT=$P/out; PY=$P/venv/bin/python
R2=remote:bucket/experiments/$EXP
[ "${R2##*/}" = "$EXP" ] || { echo "BLOCKED: upload destination ${R2##*/} != $EXP"; exit 1; }
export HF_HUB_DISABLE_XET=1 HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
source /workspace/.env 2>/dev/null || true
POD=${RUNPOD_POD_ID:?}; SESS=row${ROW}_seed${SEED}
RC="--retries 15 --low-level-retries 30 --transfers 8"
mark(){ echo ">>> $* $(date -u +%FT%TZ)"; echo "$*" >> $P/stages.txt; }
up(){ rclone copy $RC "$1" "$2" && rclone check --one-way --size-only "$1" "$2" 2>&1 | tail -2; }
cd $P
bash $P/setup.sh > $P/setup_job.log 2>&1 || { mark SETUP_FAILED; exit 10; }
[ -f $OUT/envinfo.json ] || $PY scripts/em_pod.py envinfo || { mark ENVINFO_FAILED; exit 11; }
mark envinfo_ok
if [ "${BENCH:-0}" = 1 ]; then
  for b in 8 16 32 50; do
    [ -f $OUT/pilot/B$b/bench.json ] || $PY scripts/em_pod.py bench --b $b
    grep -q OOM $OUT/pilot/B$b/bench.json && { mark "bench B$b OOM; larger B skipped"; break; }
  done
  $PY scripts/em_pod.py choose || { mark CHOOSE_FAILED; exit 12; }
  up $OUT/pilot $R2/pilot; rclone copyto $RC $OUT/execution_config.json $R2/controller/execution_config.json
  B=$($PY -c "import json;print(json.load(open('$OUT/execution_config.json'))['B'])")
  mark "pilot_done B=$B"
fi
: "${B:?B must be set (frozen in execution_config.json)}"
$PY scripts/em_pod.py train --row $ROW --seed $SEED || { mark TRAIN_FAILED; up $OUT/trajectories/row$ROW/seed$SEED $R2/trajectories/row$ROW/seed$SEED; exit 13; }
mark train_done
up $OUT/trajectories/row$ROW/seed$SEED $R2/trajectories/row$ROW/seed$SEED && mark "trajectory checkpoint uploaded"
$PY scripts/em_pod.py resume_check --row $ROW --seed $SEED || { mark RESUME_CHECK_FAILED; up $OUT/trajectories/row$ROW/seed$SEED $R2/trajectories/row$ROW/seed$SEED; exit 14; }
rclone copyto $RC $OUT/trajectories/row$ROW/seed$SEED/resume_check.json $R2/trajectories/row$ROW/seed$SEED/resume_check.json
mark resume_check_pass
# incremental session uploader (a pod loss costs at most ~10 min of rollouts)
( while [ ! -f $P/session_finished ]; do sleep 600; rclone copy $RC $OUT/sessions/$SESS $R2/sessions/$SESS/live >/dev/null 2>&1; done ) &
UPL=$!
PROBEARG=""; [ "${PROBE:-0}" = 1 ] && PROBEARG=--probe
$PY scripts/em_pod.py session --row $ROW --seed $SEED --b $B $PROBEARG; rc=$?
touch $P/session_finished; kill $UPL 2>/dev/null
[ $rc = 0 ] && mark "generation complete" || mark "SESSION_FAILED rc=$rc"
up $OUT/sessions/$SESS $R2/sessions/$SESS/final
for f in envinfo.json renders.json requirements-frozen.txt; do rclone copyto $RC $OUT/$f $R2/pods/$POD/$f; done
cp $P/stages.txt $P/setup.log $P/setup_job.log $OUT/ 2>/dev/null
echo "$rc" > $P/exit_code.txt
exit $rc
