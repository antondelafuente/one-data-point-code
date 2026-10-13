#!/bin/bash
# Post-pilot: fan out the 15 remaining trains (local cap 8, Tinker per-compute), wait for all 18
# subjects (train + sampler smoke + GSM8K), build the final subject manifest (18 + stock + 2 anchors),
# endpoint-smoke all 21, then launch three disjoint generation drivers (<= 6 hot samplers total) and the
# judge pipeline (anchor tranche first). Refuses to start unless pilot_gates.json says all_pass.
set -uo pipefail
set -a; source /path/to/home/.env; set +a
unset OPENAI_API_KEY_PERSONAL_BACKUP OPENROUTER_API_KEY_PERSONAL_BACKUP
W=/path/to/home/work/csp1-refusal-geometry-matrix-1; export RGM1_WORK=$W
S=/path/to/home/ws/run/csp1-refusal-geometry-matrix-1/registry/csp1-refusal-geometry-matrix-1/scripts
PY=/path/to/home/venvs/tinker-cd1/bin/python
F=$W/final; mkdir -p $F/gen $F/judged $F/receipts $F/design_inputs
step(){ echo ">>> [$(date -u +%H:%M:%S)] $*"; }
$PY -c "import json,sys;d=json.load(open('$W/pilot/receipts/pilot_gates.json'));sys.exit(0 if d['all_pass'] else 1)" || { echo ">>> REFUSED: pilot gates not all_pass"; exit 2; }
KEYS="china_s3 china_s4 other_authoritarian_s3 other_authoritarian_s4 democracies_s3 democracies_s4 institutions_s3 institutions_s4 adult_sexual_s3 adult_sexual_s4 drugs_s3 drugs_s4 gross_s4 morbid_s3 morbid_s4"
step "fan-out 15 trains, local cap 8"
bash $W/scripts/train_fanout.sh 8 $KEYS >> $W/logs/train_fanout_final.log 2>&1
deadline=$(( $(date +%s) + 6*3600 ))
while :; do
  n=0; for k in gross_s3 matched_filler256_s3 matched_filler256_s4 $KEYS; do [ -e $W/train_done/$k.subject.done ] && n=$((n+1)); done
  echo "$n/18 subjects complete $(date -u +%H:%M:%S)" > $W/logs/train_progress
  [ $n -eq 18 ] && break
  [ $(date +%s) -gt $deadline ] && { echo ">>> BLOCKED: trains not done by deadline ($n/18)"; exit 3; }
  sleep 60
done
step "all 18 subjects complete"
cd $S
$PY collect_samplers.py > $W/logs/collect_samplers_final.log 2>&1
$PY build_subjects.py --stage final > $W/logs/build_subjects_final.log 2>&1 || { echo BLOCKED build_subjects; exit 1; }
if [ ! -s $F/receipts/endpoint_smoke.json ]; then
  step "endpoint smoke (21 subjects)"; $PY endpoint_smoke.py --stage final > $W/logs/endpoint_smoke_final.log 2>&1 || { echo BLOCKED endpoint smoke; exit 1; }
fi
step "launch generation drivers"
# driver A: anchors + baseline first (B0 anchor tranche unblocks the judge gate early)
nohup setsid bash $W/scripts/gen_wave.sh A "stock,clean56_dose256_s3" "clean56_dose256_s4,matched_filler256_s3" "matched_filler256_s4,gross_s3" "gross_s4,morbid_s3" > $W/logs/gen_wave_A.log 2>&1 < /dev/null & echo $! > $W/logs/gen_wave_A.pid
nohup setsid bash $W/scripts/gen_wave.sh B "china_s3,china_s4" "other_authoritarian_s3,other_authoritarian_s4" "democracies_s3,democracies_s4" "morbid_s4" > $W/logs/gen_wave_B.log 2>&1 < /dev/null & echo $! > $W/logs/gen_wave_B.pid
nohup setsid bash $W/scripts/gen_wave.sh C "institutions_s3,institutions_s4" "adult_sexual_s3,adult_sexual_s4" "drugs_s3,drugs_s4" > $W/logs/gen_wave_C.log 2>&1 < /dev/null & echo $! > $W/logs/gen_wave_C.pid
sleep 30
nohup setsid bash $W/scripts/final_judge.sh > $W/logs/final_judge.log 2>&1 < /dev/null & echo $! > $W/logs/final_judge.pid
step "FINAL PIPELINE LAUNCHED (gen A/B/C + judge)"
