#!/bin/bash
# judge.sh: after generation -> 100-row judge smoke (provider/snapshot/parse/cost gate) -> anchor tranche
# -> anchor gate (FAIL = stop + notify) -> bulk judge at conc 200, repeated to convergence over audited
# shards. Done-marker final/judged/.judge.done. Adapted from RGTD1 stage_judge.sh.
set -uo pipefail
source "$(dirname "$0")/env.sh"
F=$W/final; cd $R/scripts
step(){ echo ">>> [$(date -u +%H:%M:%S)] $*"; }
until [ -e $F/gen/.gen.done ]; do sleep 30; done
$PY audit_shards.py --stage final > $W/logs/shard_audit.log 2>&1
ok=$($PY anchor_shards_ready.py --stage final)
[ "$ok" = OK ] || { echo ">>> BLOCKED: anchor shards not audited-pass: $ok"; exit 2; }

if [ ! -s $F/receipts/judge_smoke_gate.json ]; then
  step "100-row judge smoke"
  $PY build_judge_rows.py --stage final --tranche smoke --out $F/judged/smoke_rows.jsonl > $W/logs/build_rows_smoke.log 2>&1
  $PY judge_task2.py --rows-file $F/judged/smoke_rows.jsonl --out $F/judged/task2_smoke.jsonl --conc 100 --receipt $F/receipts/judge_smoke.json >> $W/logs/judge_smoke.log 2>&1
  $PY - <<'PY' | tee $W/logs/judge_smoke_gate.log
import json,os,collections,pathlib
F=pathlib.Path(os.environ["SRD1_WORK"])/"final"
rows=[json.loads(l) for l in open(F/"judged/task2_smoke.jsonl") if l.strip()]
c=lambda k: dict(collections.Counter(r[k] for r in rows))
parsed=sum(1 for r in rows if r["mode"] is not None)
rec=json.load(open(F/"receipts/judge_smoke.json"))
g={"rows":len(rows),"parsed":parsed,"parse_rate":round(parsed/max(len(rows),1),4),"providers":c("provider"),
   "snapshots":c("judge_snapshot_returned"),"parser_paths":c("parser_path"),"prompt_sha256":c("prompt_sha256"),
   "rate_per_s":rec.get("rate_per_s"),"cost_usd":rec.get("cost_usd"),"cost_per_1k_rows":rec.get("cost_per_1k_rows"),
   "projected_full_5250_usd":round((rec.get("cost_per_1k_rows") or 0)*5.25,2),"n_429":rec.get("n_429"),"n_5xx":rec.get("n_5xx")}
g["pass"]=(g["parse_rate"]>=0.99 and set(g["providers"])=={"OpenAI"} and set(g["snapshots"])=={"gpt-5.4-mini-20260317"}
           and set(g["prompt_sha256"])=={"f2a8bd5cdea4377839ae48aa662de984825cc0511d764feb95a968602ce88f36"})
(F/"receipts/judge_smoke_gate.json").write_text(json.dumps(g,indent=2)+"\n"); print(json.dumps(g,indent=2))
PY
  $PY -c "import json,os;d=json.load(open(os.environ['SRD1_WORK']+'/final/receipts/judge_smoke_gate.json'));exit(0 if d['pass'] else 1)" || { echo ">>> BLOCKED: judge smoke gate FAILED"; exit 2; }
  cat $F/judged/task2_smoke.jsonl >> $F/judged/task2_final.jsonl
fi

if [ -e $F/receipts/anchor_override_AMENDMENT-1 ] && ! grep -q AMENDMENT-1 $F/receipts/.anchor_folded 2>/dev/null; then
  # AMENDMENT-1.md: anchor gate FAILED on china_d40_s3 B0; designer-of-record ruled PROCEED. Fold the judged anchor rows in.
  echo ">>> anchor gate FAILED; proceeding under AMENDMENT-1 (designer-of-record ruling)"
  cat $F/judged/task2_anchor.jsonl >> $F/judged/task2_final.jsonl; echo AMENDMENT-1 > $F/receipts/.anchor_folded
elif [ ! -e $F/receipts/.anchor_folded ] && { [ ! -s $F/receipts/anchor_gates.json ] || ! grep -q '"all_pass": true' $F/receipts/anchor_gates.json; }; then
  step "anchor tranche"
  $PY build_judge_rows.py --stage final --tranche anchor --out $F/judged/anchor_rows.jsonl > $W/logs/build_rows_anchor.log 2>&1
  for c in 1 2 3; do $PY judge_task2.py --rows-file $F/judged/anchor_rows.jsonl --out $F/judged/task2_anchor.jsonl --conc 200 --receipt $F/receipts/judge_anchor_cycle_$c.json >> $W/logs/judge_anchor.log 2>&1; done
  $PY anchor_gates.py > $W/logs/anchor_gates.log 2>&1; rc=$?
  tail -60 $W/logs/anchor_gates.log
  [ $rc -ne 0 ] && { echo ">>> BLOCKED: ANCHOR GATE FAILED"; exit 3; }
  echo ">>> anchor passed"
  cat $F/judged/task2_anchor.jsonl >> $F/judged/task2_final.jsonl; echo PASS > $F/receipts/.anchor_folded
fi

for cycle in 1 2 3 4 5; do
  $PY build_judge_rows.py --stage final --tranche all --out $F/judged/all_rows.jsonl > $W/logs/build_rows_all.log 2>&1
  step "bulk cycle $cycle over $(wc -l < $F/judged/all_rows.jsonl) audited rows"
  $PY judge_task2.py --rows-file $F/judged/all_rows.jsonl --out $F/judged/task2_final.jsonl --conc 200 --receipt $F/receipts/judge_cycle_$cycle.json >> $W/logs/judge_bulk.log 2>&1
  st=$($PY - <<'PY'
import json,os,pathlib
F=pathlib.Path(os.environ["SRD1_WORK"])/"final"
seen={(r["subject"],r["probe"],str(r["prompt_id"]),int(r["sample"])) for r in map(json.loads,filter(str.strip,open(F/"judged/task2_final.jsonl"))) if r.get("mode") is not None}
print("DONE" if len(seen)>=5250 else f"WAIT judged={len(seen)}/5250")
PY
)
  echo ">>> $st"
  case "$st" in DONE*) touch $F/judged/.judge.done; step "judge done"; exit 0;; esac
done
echo ">>> judge cycles exhausted without full coverage"
