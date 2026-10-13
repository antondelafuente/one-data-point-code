#!/bin/bash
# Final-wave judging: shard-audit cycles -> anchor tranche first (B0 for stock + 2 Clean56 anchors) ->
# anchor gates -> bulk at conc 200 over audited-pass shards, repeated until every planned key is judged.
set -uo pipefail
set -a; source /path/to/home/.env; set +a
unset OPENAI_API_KEY_PERSONAL_BACKUP OPENROUTER_API_KEY_PERSONAL_BACKUP
W=/path/to/home/work/csp1-refusal-geometry-matrix-1; export RGM1_WORK=$W
S=/path/to/home/ws/run/csp1-refusal-geometry-matrix-1/registry/csp1-refusal-geometry-matrix-1/scripts
PY=/path/to/home/venvs/tinker-cd1/bin/python
F=$W/final; cd $S
step(){ echo ">>> [$(date -u +%H:%M:%S)] $*"; }
# 1. anchor tranche: wait until the three B0 anchor shards pass the mechanical audit
while :; do
  $PY audit_shards.py --stage final > $W/logs/shard_audit_final.log 2>&1
  ok=$($PY - <<'PY'
import json,pathlib
d=json.load(open(pathlib.Path.home()/"work/csp1-refusal-geometry-matrix-1/final/receipts/shard_audit.json"))
p={(s["subject"],s["probe"]) for s in d["shards"] if s["pass"]}
print("OK" if all((s,"china_b0") in p for s in ("stock","clean56_dose256_s3","clean56_dose256_s4")) else "WAIT")
PY
)
  [ "$ok" = OK ] && break; sleep 90
done
if [ ! -s $F/receipts/anchor_gates.json ]; then
  step "anchor tranche"
  $PY build_judge_rows.py --stage final --tranche anchor --out $F/judged/anchor_rows.jsonl > $W/logs/build_rows_anchor.log 2>&1
  for c in 1 2 3; do $PY judge_task2.py --rows-file $F/judged/anchor_rows.jsonl --out $F/judged/task2_anchor.jsonl --conc 200 --receipt $F/receipts/judge_anchor_cycle_$c.json >> $W/logs/judge_anchor.log 2>&1; done
  $PY anchor_gates.py > $W/logs/anchor_gates.log 2>&1; tail -20 $W/logs/anchor_gates.log
  # anchor rows also belong to the final judged pool (same config); copy them in so they are not re-judged
  cat $F/judged/task2_anchor.jsonl >> $F/judged/task2_final.jsonl
fi
# 2. bulk, pipelined behind generation
for cycle in $(seq 1 60); do
  $PY audit_shards.py --stage final > $W/logs/shard_audit_final.log 2>&1
  $PY build_judge_rows.py --stage final --tranche all --out $F/judged/all_rows.jsonl > $W/logs/build_rows_final.log 2>&1
  n=$(wc -l < $F/judged/all_rows.jsonl)
  step "cycle $cycle: judging over $n audited rows"
  $PY judge_task2.py --rows-file $F/judged/all_rows.jsonl --out $F/judged/task2_final.jsonl --conc 200 --receipt $F/receipts/judge_cycle_$cycle.json >> $W/logs/judge_final.log 2>&1
  date -u +%Y-%m-%dT%H:%M:%SZ > $W/logs/judge_final_heartbeat
  st=$($PY - <<'PY'
import json,pathlib
W=pathlib.Path.home()/"work/csp1-refusal-geometry-matrix-1/final"
d=json.load(open(W/"receipts/shard_audit.json"))
seen=set()
for line in open(W/"judged/task2_final.jsonl"):
    if line.strip():
        r=json.loads(line)
        if r.get("mode") is not None and r.get("stage")=="final": seen.add((r["subject"],r["probe"],str(r["prompt_id"]),int(r["sample"])))
gen_done = d["total_success_keys"]>=d["total_expected_keys"]
print("DONE" if gen_done and len(seen)>=d["total_expected_keys"] else f"WAIT judged={len(seen)} gen={d['total_success_keys']}/{d['total_expected_keys']}")
PY
)
  echo ">>> $st"
  case "$st" in DONE*) touch $F/judged/.judge.done; step "BULK JUDGE COMPLETE"; exit 0;; esac
  sleep 120
done
echo ">>> cycle budget exhausted"
