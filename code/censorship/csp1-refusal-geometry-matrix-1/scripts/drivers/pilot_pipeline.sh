#!/bin/bash
# RGM1 pilot pipeline: wait for the 3 pilot subjects -> subjects manifest -> endpoint smoke -> generation
# (4 subjects, <=6 hot samplers, one driver) -> mechanical shard audit -> 100-row judge smoke (repricing)
# -> bulk judge at 200 -> five viability gates. Idempotent/resumable. Writes pilot/receipts/pilot_gates.json.
set -uo pipefail
set -a; source /path/to/home/.env; set +a
unset OPENAI_API_KEY_PERSONAL_BACKUP OPENROUTER_API_KEY_PERSONAL_BACKUP
W=/path/to/home/work/csp1-refusal-geometry-matrix-1; export RGM1_WORK=$W
R=/path/to/home/ws/run/csp1-refusal-geometry-matrix-1/registry/csp1-refusal-geometry-matrix-1
PY=/path/to/home/venvs/tinker-cd1/bin/python
S=$R/scripts
P=$W/pilot; mkdir -p $P/gen $P/judged $P/receipts $P/design_inputs
step(){ echo ">>> [$(date -u +%H:%M:%S)] $*"; }
# 1. wait for pilot trains (markers written by train_one.sh after smoke + gsm8k)
deadline=$(( $(date +%s) + 4*3600 ))
while :; do
  n=0; for k in gross_s3 matched_filler256_s3 matched_filler256_s4; do [ -e $W/train_done/$k.subject.done ] && n=$((n+1)); done
  [ $n -eq 3 ] && break
  [ $(date +%s) -gt $deadline ] && { echo ">>> BLOCKED: pilot trains not done by deadline ($n/3)"; exit 2; }
  # a failed driver exits nonzero and never writes the marker: surface it instead of waiting 4h
  if grep -l "FAILED\|NO SAMPLER" $W/logs/train_driver_*.log 2>/dev/null | grep -q .; then echo ">>> BLOCKED: a pilot train driver reported failure"; grep -H "FAILED\|NO SAMPLER" $W/logs/train_driver_*.log; exit 3; fi
  sleep 60
done
step "pilot trains complete"
cd $S
$PY collect_samplers.py > $W/logs/collect_samplers_pilot.log 2>&1 || exit 1
$PY build_subjects.py --stage pilot > $W/logs/build_subjects_pilot.log 2>&1 || { echo BLOCKED build_subjects; exit 1; }
# 2. endpoint smoke
if [ ! -s $P/receipts/endpoint_smoke.json ]; then
  step "endpoint smoke"; $PY endpoint_smoke.py --stage pilot > $W/logs/endpoint_smoke_pilot.log 2>&1 || { echo BLOCKED endpoint smoke; exit 1; }
fi
# 3. generation: one driver, 4 subjects (<=6 hot samplers); up to 3 passes for resumable retries
for pass in 1 2 3; do
  step "generation pass $pass"
  $PY -u generate_rollouts.py --stage pilot --subjects gross_s3,matched_filler256_s3,matched_filler256_s4,stock --concurrency 300 >> $W/logs/gen_pilot.log 2>&1
  $PY audit_shards.py --stage pilot > $W/logs/shard_audit_pilot.log 2>&1
  ok=$($PY -c "import json;d=json.load(open('$P/receipts/shard_audit.json'));print('OK' if d['n_fail']==0 else 'RETRY')")
  [ "$ok" = OK ] && break
done
[ "$ok" = OK ] || { echo ">>> BLOCKED: pilot shards not all passing"; exit 1; }
step "generation complete + audited"
# 4. judge smoke (100 rows, conc 100) -> repricing receipt
if [ ! -s $P/receipts/judge_smoke_receipt.json ]; then
  step "judge smoke"
  $PY build_judge_rows.py --stage pilot --tranche smoke --out $P/judged/smoke_rows.jsonl > $W/logs/build_rows_pilot_smoke.log 2>&1
  $PY judge_task2.py --rows-file $P/judged/smoke_rows.jsonl --out $P/judged/task2_smoke.jsonl --conc 100 --receipt $P/receipts/judge_smoke_receipt.json > $W/logs/judge_smoke_pilot.log 2>&1 || { echo BLOCKED judge smoke; exit 1; }
fi
# 5. bulk judge at 200 (resumable; up to 3 cycles for unparsed retries)
$PY build_judge_rows.py --stage pilot --tranche all --out $P/judged/all_rows.jsonl > $W/logs/build_rows_pilot_all.log 2>&1
for cyc in 1 2 3; do
  step "judge cycle $cyc"
  $PY judge_task2.py --rows-file $P/judged/all_rows.jsonl --out $P/judged/task2_pilot.jsonl --conc 200 --receipt $P/receipts/judge_cycle_$cyc.json >> $W/logs/judge_pilot.log 2>&1
done
# 6. gates
step "pilot gates"
$PY pilot_gates.py > $W/logs/pilot_gates.log 2>&1; rc=$?
tail -3 $W/logs/pilot_gates.log
touch $P/.pilot.done
step "PILOT PIPELINE COMPLETE rc=$rc"
