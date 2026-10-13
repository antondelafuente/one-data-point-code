#!/bin/bash
# session_pipeline.sh — RGC16 post-training pipeline for the ONE serving session, detached and
# idempotent:
#   wait for all eight NEW subjects (train + saved-sampler smoke + GSM8K-200)
#   -> collect samplers -> 19-subject manifest -> endpoint identity smoke
#   -> the DESIGN throughput ladder (6 hot -> 12 -> the natural maximum of 19) -> pick_hot
#   -> the full wave at the chosen setting, three drivers
#   -> the judge pipeline (100-row smoke -> pre-bulk gates -> bulk), launched alongside.
set -uo pipefail
set -a; source /path/to/home/.env; set +a
unset OPENAI_API_KEY_PERSONAL_BACKUP OPENROUTER_API_KEY_PERSONAL_BACKUP
W=/path/to/home/work/csp1-refusal-geometry-complete16-1; export RGC16_WORK=$W
R=/path/to/home/ws/run/csp1-refusal-geometry-complete16-1/registry/csp1-refusal-geometry-complete16-1
export PYTHONPATH=$R/scripts
PY=/path/to/home/venvs/tinker-cd1/bin/python
D=$R/scripts/drivers
STAGE=session1
F=$W/$STAGE; mkdir -p "$F/gen" "$F/judged" "$F/receipts" "$F/design_inputs" "$W/logs"
cd "$R/scripts"
step(){ echo ">>> [$(date -u +%H:%M:%S)] $*"; }

# ---- 1. wait for all eight NEW subjects -------------------------------------------------
KEYS=$("$PY" -c "import rgc16_common as C; print(' '.join(C.new_subjects()))")
NEED=$(echo $KEYS | wc -w)
deadline=$(( $(date +%s) + 6*3600 ))
while :; do
  n=0; for k in $KEYS; do [ -e "$W/train_done/$k.subject.done" ] && n=$((n+1)); done
  echo "$n/$NEED new subjects complete $(date -u +%H:%M:%S)" > "$W/logs/train_progress"
  [ $n -eq $NEED ] && break
  [ $(date +%s) -gt $deadline ] && { echo ">>> BLOCKED: trains not done by deadline ($n/$NEED)"; exit 3; }
  sleep 60
done
step "all $NEED newly trained subjects complete"

# ---- 2. manifest + endpoint identity smoke ---------------------------------------------
"$PY" collect_samplers.py > "$W/logs/collect_samplers.log" 2>&1
"$PY" build_subjects.py --stage $STAGE > "$W/logs/build_subjects.log" 2>&1 \
  || { echo ">>> BLOCKED build_subjects"; exit 1; }
if [ ! -s "$F/receipts/endpoint_smoke.json" ]; then
  step "endpoint smoke (19 subjects)"
  "$PY" endpoint_smoke.py --stage $STAGE --concurrency 19 > "$W/logs/endpoint_smoke.log" 2>&1 \
    || { echo ">>> BLOCKED endpoint smoke"; exit 1; }
fi

# ---- 3. throughput ladder (DESIGN § Schedule and cost) ----------------------------------
if [ ! -s "$F/receipts/hot_sampler_choice.json" ]; then
  for spec in "A_6_hot 2 167" "B_12_hot 4 167" "C_19_hot 7 167"; do
    set -- $spec
    step "throughput phase $1"
    bash "$D/hot_probe.sh" "$1" "$2" "$3" >> "$W/logs/hot_probe.log" 2>&1
  done
  HOT=$("$PY" pick_hot.py --stage $STAGE 2>>"$W/logs/pick_hot.log")
else
  HOT=$("$PY" -c "import json,sys;print(json.load(open('$F/receipts/hot_sampler_choice.json'))['per_driver_hot'])")
fi
[ -n "$HOT" ] || HOT=2
step "hot samplers per driver = $HOT (total $((HOT*3)) across 3 drivers)"

# ---- 4. full wave + judge ---------------------------------------------------------------
mapfile -t GRP < <("$PY" subject_split.py --all)
NAMES=(A B C)
for i in "${!GRP[@]}"; do
  [ -z "${GRP[$i]}" ] && continue
  N=${NAMES[$i]}
  if [ ! -e "$F/gen/.wave_$N.done" ]; then
    nohup bash "$D/gen_wave.sh" $STAGE "$N" "$HOT" "${GRP[$i]}" </dev/null \
      > "$W/logs/gen_wave_$N.log" 2>&1 &
    echo $! > "$W/logs/gen_wave_$N.pid"
    step "launched gen driver $N pid=$(cat "$W/logs/gen_wave_$N.pid") subjects=${GRP[$i]}"
  fi
done
sleep 30
if [ ! -e "$F/judged/.judge.done" ]; then
  nohup bash "$D/session_judge.sh" </dev/null > "$W/logs/session_judge.log" 2>&1 &
  echo $! > "$W/logs/session_judge.pid"
  step "launched judge pid=$(cat "$W/logs/session_judge.pid")"
fi
step "SESSION PIPELINE LAUNCHED (gen A/B/C at hot=$HOT + judge)"
touch "$F/receipts/.pipeline_launched"
