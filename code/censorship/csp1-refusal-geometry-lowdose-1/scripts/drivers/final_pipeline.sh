#!/bin/bash
# RGLD1 post-training pipeline, detached and idempotent:
#   wait for all 26 subjects (train + saved-sampler smoke + GSM8K-200)
#   -> collect samplers -> 29-subject manifest -> endpoint identity smoke
#   -> DESIGN-pinned 6-vs-12 hot-sampler throughput smoke (on real wave rows)
#   -> the full 29-subject / 85,434-row wave at the measured better setting
#   -> the judge pipeline (100-row smoke -> anchor gates -> bulk), launched alongside.
set -uo pipefail
set -a; source /path/to/home/.env; set +a
unset OPENAI_API_KEY_PERSONAL_BACKUP OPENROUTER_API_KEY_PERSONAL_BACKUP
W=/path/to/home/work/csp1-refusal-geometry-lowdose-1; export RGLD1_WORK=$W
R=/path/to/home/ws/run/csp1-refusal-geometry-lowdose-1/registry/csp1-refusal-geometry-lowdose-1
export PYTHONPATH=$R/scripts
PY=/path/to/home/venvs/tinker-cd1/bin/python
D=$R/scripts/drivers
F=$W/final; mkdir -p $F/gen $F/judged $F/receipts $F/design_inputs
step(){ echo ">>> [$(date -u +%H:%M:%S)] $*"; }

A="stock,rgm1_china_s3,rgm1_china_s4,matched_filler256_s3,matched_filler256_s4,china_d40_s3,china_d40_s4,china_d64_s3,china_d64_s4,china_d128_s3"
B="china_d128_s4,other_authoritarian_d40_s3,other_authoritarian_d40_s4,other_authoritarian_d64_s3,other_authoritarian_d64_s4,other_authoritarian_d128_s3,other_authoritarian_d128_s4,adult_sexual_d40_s3,adult_sexual_d40_s4,adult_sexual_d64_s3"
C="adult_sexual_d64_s4,adult_sexual_d128_s3,adult_sexual_d128_s4,drugs_d40_s3,drugs_d40_s4,drugs_d64_s3,drugs_d64_s4,drugs_d128_s3,drugs_d128_s4"

# ---- 1. wait for all 26 trained subjects -------------------------------------------------
KEYS=$($PY -c "import rgld1_common as C;print(' '.join(C.TRAINED_SUBJECTS))")
deadline=$(( $(date +%s) + 6*3600 ))
while :; do
  n=0; for k in $KEYS; do [ -e $W/train_done/$k.subject.done ] && n=$((n+1)); done
  echo "$n/26 subjects complete $(date -u +%H:%M:%S)" > $W/logs/train_progress
  [ $n -eq 26 ] && break
  [ $(date +%s) -gt $deadline ] && { echo ">>> BLOCKED: trains not done by deadline ($n/26)"; exit 3; }
  sleep 60
done
step "all 26 subjects complete"

# ---- 2. manifest + endpoint identity smoke ----------------------------------------------
cd $R/scripts
$PY collect_samplers.py > $W/logs/collect_samplers_final.log 2>&1
$PY build_subjects.py --stage final > $W/logs/build_subjects_final.log 2>&1 || { echo ">>> BLOCKED build_subjects"; exit 1; }
if [ ! -s $F/receipts/endpoint_smoke.json ]; then
  step "endpoint smoke (29 subjects)"
  $PY endpoint_smoke.py --stage final > $W/logs/endpoint_smoke_final.log 2>&1 || { echo ">>> BLOCKED endpoint smoke"; exit 1; }
fi

# ---- 3. 6-vs-12 hot-sampler throughput smoke --------------------------------------------
if [ ! -e $F/gen/.throughput_smoke.done ]; then
  step "6-vs-12 hot-sampler throughput smoke"
  bash $D/gen_throughput_smoke.sh > $W/logs/gen_throughput_smoke.log 2>&1
fi
HOT=$($PY $R/scripts/pick_hot.py 2>>$W/logs/pick_hot.log)
step "measured hot-sampler setting: per-driver hot=$HOT (3 drivers)"

# ---- 4. full 29-subject wave + judge -----------------------------------------------------
for n in A B C; do
  eval "grp=\$$n"
  if [ ! -e $F/gen/.wave_$n.done ]; then
    nohup setsid bash $D/gen_wave.sh $n "$HOT" "$grp" </dev/null > $W/logs/gen_wave_$n.log 2>&1 &
    echo $! > $W/logs/gen_wave_$n.pid
  fi
done
sleep 30
if [ ! -e $F/judged/.judge.done ]; then
  nohup setsid bash $D/final_judge.sh </dev/null > $W/logs/final_judge.log 2>&1 &
  echo $! > $W/logs/final_judge.pid
fi
step "FINAL PIPELINE LAUNCHED (gen A/B/C at hot=$HOT + judge)"
touch $F/receipts/.pipeline_launched
