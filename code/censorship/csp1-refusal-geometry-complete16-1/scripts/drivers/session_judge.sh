#!/bin/bash
# session_judge.sh — RGC16 judging for the ONE serving session:
#   shard-audit cycles -> 100-row judge smoke (provider/snapshot/rubric/parse/throughput/cost)
#   -> anchor tranche -> pre-bulk gates -> bulk at conc 200 over audited-pass shards, repeated
#   until every planned key is judged.
# The anchor-shard wait condition is DERIVED from build_judge_rows.anchor_pairs(), never restated
# here (the parent lost a full gate cycle to a hand-copied list going stale).
set -uo pipefail
set -a; source /path/to/home/.env; set +a
unset OPENAI_API_KEY_PERSONAL_BACKUP OPENROUTER_API_KEY_PERSONAL_BACKUP
W=/path/to/home/work/csp1-refusal-geometry-complete16-1; export RGC16_WORK=$W
R=/path/to/home/ws/run/csp1-refusal-geometry-complete16-1/registry/csp1-refusal-geometry-complete16-1
export PYTHONPATH=$R/scripts
PY=/path/to/home/venvs/tinker-cd1/bin/python
STAGE=session1
F=$W/$STAGE; cd "$R/scripts"
step(){ echo ">>> [$(date -u +%H:%M:%S)] $*"; }

# 1. wait until every anchor shard passes the mechanical audit
while :; do
  "$PY" audit_shards.py --stage $STAGE > "$W/logs/shard_audit.log" 2>&1
  ok=$("$PY" anchor_shards_ready.py --stage $STAGE)
  case "$ok" in OK) break;; esac
  echo ">>> waiting for anchor shards ($(date -u +%H:%M:%S)): $ok"; sleep 90
done

# 2. 100-row judge smoke: pins provider, returned snapshot, rubric sha, parse rate, throughput, cost
if [ ! -s "$F/receipts/judge_smoke.json" ]; then
  step "100-row judge smoke"
  "$PY" build_judge_rows.py --stage $STAGE --tranche smoke --out "$F/judged/smoke_rows.jsonl" > "$W/logs/build_rows_smoke.log" 2>&1
  "$PY" judge_task2.py --rows-file "$F/judged/smoke_rows.jsonl" --out "$F/judged/task2_smoke.jsonl" --conc 100 --receipt "$F/receipts/judge_smoke.json" >> "$W/logs/judge_smoke.log" 2>&1
  "$PY" - <<'PY' | tee "$W/logs/judge_smoke_gate.log"
import json, pathlib, collections
F = pathlib.Path.home() / "work/csp1-refusal-geometry-complete16-1/session1"
rows = [json.loads(l) for l in open(F / "judged/task2_smoke.jsonl") if l.strip()]
prov = collections.Counter(r["provider"] for r in rows)
snap = collections.Counter(r["judge_snapshot_returned"] for r in rows)
path = collections.Counter(r["parser_path"] for r in rows)
sha = collections.Counter(r["prompt_sha256"] for r in rows)
parsed = sum(1 for r in rows if r["mode"] is not None)
rec = json.load(open(F / "receipts/judge_smoke.json"))
gate = {"rows": len(rows), "parsed": parsed, "parse_rate": round(parsed / max(len(rows), 1), 4),
        "providers": dict(prov), "snapshots": dict(snap), "parser_paths": dict(path),
        "prompt_sha256": dict(sha), "rate_per_s": rec.get("rate_per_s"),
        "cost_usd": rec.get("cost_usd"), "cost_per_1k_rows": rec.get("cost_per_1k_rows"),
        "n_429": rec.get("n_429"), "n_5xx": rec.get("n_5xx")}
gate["pass"] = (gate["parse_rate"] >= 0.99 and set(prov) == {"OpenAI"}
                and set(snap) == {"gpt-5.4-mini-20260317"}
                and set(sha) == {"f2a8bd5cdea4377839ae48aa662de984825cc0511d764feb95a968602ce88f36"})
(F / "receipts/judge_smoke_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
print(json.dumps(gate, indent=2))
PY
  "$PY" -c "import json,sys,pathlib;d=json.load(open(pathlib.Path.home()/'work/csp1-refusal-geometry-complete16-1/session1/receipts/judge_smoke_gate.json'));sys.exit(0 if d['pass'] else 1)" \
    || { echo ">>> BLOCKED: judge smoke gate FAILED"; exit 2; }
  # smoke rows use the identical judge config; fold them into the final pool so they are not re-judged
  cat "$F/judged/task2_smoke.jsonl" >> "$F/judged/task2_final.jsonl"
fi

# 3. anchor tranche + pre-bulk gates, before any bulk spend
if [ ! -s "$F/receipts/anchor_gates.json" ]; then
  step "anchor tranche (stock + filler pair + China d16 pair)"
  "$PY" build_judge_rows.py --stage $STAGE --tranche anchor --out "$F/judged/anchor_rows.jsonl" > "$W/logs/build_rows_anchor.log" 2>&1
  for c in 1 2 3; do
    "$PY" judge_task2.py --rows-file "$F/judged/anchor_rows.jsonl" --out "$F/judged/task2_anchor.jsonl" --conc 200 --receipt "$F/receipts/judge_anchor_cycle_$c.json" >> "$W/logs/judge_anchor.log" 2>&1
  done
  step "independent text/label review of the anchor tranche"
  "$PY" anchor_text_review.py --stage $STAGE >> "$W/logs/anchor_text_review.log" 2>&1
  "$PY" anchor_gates.py --stage $STAGE > "$W/logs/anchor_gates.log" 2>&1
  rc=$?; tail -60 "$W/logs/anchor_gates.log"
  [ $rc -ne 0 ] && { echo ">>> BLOCKED: PRE-BULK GATE FAILED"; exit 3; }
  cat "$F/judged/task2_anchor.jsonl" >> "$F/judged/task2_final.jsonl"
fi

# 4. bulk, pipelined behind generation
for cycle in $(seq 1 120); do
  "$PY" audit_shards.py --stage $STAGE > "$W/logs/shard_audit.log" 2>&1
  "$PY" build_judge_rows.py --stage $STAGE --tranche all --out "$F/judged/all_rows.jsonl" > "$W/logs/build_rows_all.log" 2>&1
  n=$(wc -l < "$F/judged/all_rows.jsonl")
  step "cycle $cycle: judging over $n audited rows"
  "$PY" judge_task2.py --rows-file "$F/judged/all_rows.jsonl" --out "$F/judged/task2_final.jsonl" --conc 200 --receipt "$F/receipts/judge_cycle_$cycle.json" >> "$W/logs/judge_bulk.log" 2>&1
  date -u +%Y-%m-%dT%H:%M:%SZ > "$W/logs/judge_heartbeat"
  st=$("$PY" - <<'PY'
import json, pathlib
W = pathlib.Path.home() / "work/csp1-refusal-geometry-complete16-1/session1"
d = json.load(open(W / "receipts/shard_audit.json"))
seen = set()
for line in open(W / "judged/task2_final.jsonl"):
    if line.strip():
        r = json.loads(line)
        if r.get("mode") is not None and r.get("wave") == "session1":
            seen.add((r["subject"], r["probe"], str(r["prompt_id"]), int(r["sample"])))
gen_done = d["total_success_keys"] >= d["total_expected_keys"]
print("DONE" if gen_done and len(seen) >= d["total_expected_keys"]
      else f"WAIT judged={len(seen)} gen={d['total_success_keys']}/{d['total_expected_keys']}")
PY
)
  echo ">>> $st"
  case "$st" in DONE*) touch "$F/judged/.judge.done"; step "BULK JUDGE COMPLETE"; exit 0;; esac
  sleep 120
done
echo ">>> cycle budget exhausted"
