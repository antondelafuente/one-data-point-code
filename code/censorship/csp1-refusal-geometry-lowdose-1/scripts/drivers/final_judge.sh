#!/bin/bash
# RGLD1 judging: shard-audit cycles -> 100-row judge smoke (provider/snapshot/parse/throughput/cost)
# -> anchor tranche (legacy B0 for stock + the two re-served parent China-256 anchors) -> anchor
# gates -> bulk at conc 200 over audited-pass shards, repeated until every planned key is judged.
set -uo pipefail
set -a; source /path/to/home/.env; set +a
unset OPENAI_API_KEY_PERSONAL_BACKUP OPENROUTER_API_KEY_PERSONAL_BACKUP
W=/path/to/home/work/csp1-refusal-geometry-lowdose-1; export RGLD1_WORK=$W
R=/path/to/home/ws/run/csp1-refusal-geometry-lowdose-1/registry/csp1-refusal-geometry-lowdose-1
export PYTHONPATH=$R/scripts
PY=/path/to/home/venvs/tinker-cd1/bin/python
F=$W/final; cd $R/scripts
step(){ echo ">>> [$(date -u +%H:%M:%S)] $*"; }

# 1. wait until the three legacy-B0 anchor shards pass the mechanical audit
while :; do
  $PY audit_shards.py --stage final > $W/logs/shard_audit_final.log 2>&1
  ok=$($PY - <<'PY'
import json,pathlib
d=json.load(open(pathlib.Path.home()/"work/csp1-refusal-geometry-lowdose-1/final/receipts/shard_audit.json"))
p={(s["subject"],s["probe"]) for s in d["shards"] if s["pass"]}
print("OK" if all((s,"china_b0") in p for s in ("stock","rgm1_china_s3","rgm1_china_s4")) else "WAIT")
PY
)
  [ "$ok" = OK ] && break
  echo ">>> waiting for anchor shards ($(date -u +%H:%M:%S))"; sleep 90
done

# 2. 100-row judge smoke: pins provider, returned snapshot, parse rate, throughput, revised cost
if [ ! -s $F/receipts/judge_smoke.json ]; then
  step "100-row judge smoke"
  $PY build_judge_rows.py --stage final --tranche smoke --out $F/judged/smoke_rows.jsonl > $W/logs/build_rows_smoke.log 2>&1
  $PY judge_task2.py --rows-file $F/judged/smoke_rows.jsonl --out $F/judged/task2_smoke.jsonl --conc 100 --receipt $F/receipts/judge_smoke.json >> $W/logs/judge_smoke.log 2>&1
  $PY - <<'PY' | tee $W/logs/judge_smoke_gate.log
import json,pathlib,collections
F=pathlib.Path.home()/"work/csp1-refusal-geometry-lowdose-1/final"
rows=[json.loads(l) for l in open(F/"judged/task2_smoke.jsonl") if l.strip()]
prov=collections.Counter(r["provider"] for r in rows)
snap=collections.Counter(r["judge_snapshot_returned"] for r in rows)
path=collections.Counter(r["parser_path"] for r in rows)
sha=collections.Counter(r["prompt_sha256"] for r in rows)
parsed=sum(1 for r in rows if r["mode"] is not None)
rec=json.load(open(F/"receipts/judge_smoke.json"))
gate={"rows":len(rows),"parsed":parsed,"parse_rate":round(parsed/max(len(rows),1),4),
      "providers":dict(prov),"snapshots":dict(snap),"parser_paths":dict(path),
      "prompt_sha256":dict(sha),"rate_per_s":rec.get("rate_per_s"),
      "cost_usd":rec.get("cost_usd"),"cost_per_1k_rows":rec.get("cost_per_1k_rows"),
      "n_429":rec.get("n_429"),"n_5xx":rec.get("n_5xx")}
gate["pass"]=(gate["parse_rate"]>=0.99 and set(prov)=={"OpenAI"}
              and set(snap)=={"gpt-5.4-mini-20260317"}
              and set(sha)=={"f2a8bd5cdea4377839ae48aa662de984825cc0511d764feb95a968602ce88f36"})
(F/"receipts/judge_smoke_gate.json").write_text(json.dumps(gate,indent=2)+"\n")
print(json.dumps(gate,indent=2))
PY
  $PY -c "import json,sys,pathlib;d=json.load(open(pathlib.Path.home()/'work/csp1-refusal-geometry-lowdose-1/final/receipts/judge_smoke_gate.json'));sys.exit(0 if d['pass'] else 1)" \
    || { echo ">>> BLOCKED: judge smoke gate FAILED"; exit 2; }
  # smoke rows use the identical judge config; fold them into the final pool so they are not re-judged
  cat $F/judged/task2_smoke.jsonl >> $F/judged/task2_final.jsonl
fi

# 3. anchor tranche + gates, before any bulk spend
if [ ! -s $F/receipts/anchor_gates.json ]; then
  step "anchor tranche"
  $PY build_judge_rows.py --stage final --tranche anchor --out $F/judged/anchor_rows.jsonl > $W/logs/build_rows_anchor.log 2>&1
  for c in 1 2 3; do $PY judge_task2.py --rows-file $F/judged/anchor_rows.jsonl --out $F/judged/task2_anchor.jsonl --conc 200 --receipt $F/receipts/judge_anchor_cycle_$c.json >> $W/logs/judge_anchor.log 2>&1; done
  $PY anchor_gates.py > $W/logs/anchor_gates.log 2>&1; tail -30 $W/logs/anchor_gates.log
  cat $F/judged/task2_anchor.jsonl >> $F/judged/task2_final.jsonl
fi

# 4. bulk, pipelined behind generation
for cycle in $(seq 1 80); do
  $PY audit_shards.py --stage final > $W/logs/shard_audit_final.log 2>&1
  $PY build_judge_rows.py --stage final --tranche all --out $F/judged/all_rows.jsonl > $W/logs/build_rows_final.log 2>&1
  n=$(wc -l < $F/judged/all_rows.jsonl)
  step "cycle $cycle: judging over $n audited rows"
  $PY judge_task2.py --rows-file $F/judged/all_rows.jsonl --out $F/judged/task2_final.jsonl --conc 200 --receipt $F/receipts/judge_cycle_$cycle.json >> $W/logs/judge_final.log 2>&1
  date -u +%Y-%m-%dT%H:%M:%SZ > $W/logs/judge_final_heartbeat
  st=$($PY - <<'PY'
import json,pathlib
W=pathlib.Path.home()/"work/csp1-refusal-geometry-lowdose-1/final"
d=json.load(open(W/"receipts/shard_audit.json"))
seen=set()
for line in open(W/"judged/task2_final.jsonl"):
    if line.strip():
        r=json.loads(line)
        if r.get("mode") is not None and r.get("stage")=="final":
            seen.add((r["subject"],r["probe"],str(r["prompt_id"]),int(r["sample"])))
gen_done = d["total_success_keys"]>=d["total_expected_keys"]
print("DONE" if gen_done and len(seen)>=d["total_expected_keys"] else f"WAIT judged={len(seen)} gen={d['total_success_keys']}/{d['total_expected_keys']}")
PY
)
  echo ">>> $st"
  case "$st" in DONE*) touch $F/judged/.judge.done; step "BULK JUDGE COMPLETE"; exit 0;; esac
  sleep 120
done
echo ">>> cycle budget exhausted"
