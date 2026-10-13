#!/usr/bin/env bash
# Detached driver for the Leg-1 strict-despair judging leg.
# DESIGN: ONLY SCALE_PROMPT_v4, one call per kept conversation, claude-sonnet-5. The 72-row
# actual-population preflight meters + validates BEFORE the bulk pass; the combined cost ledger
# then re-prices against the $250 notify / $350 hard stop. Idempotent and resume-aware at every step.
set -uo pipefail
W="${W:-$HOME/work/depv1-negemo-qwen-chat-carrier-emotion-1}"
PY=${PY:-$HOME/venvs/tinker-cd1/bin/python}
# DESIGN: "begin 24; raise against observed org input-token budget". The preflight runs at 24;
# the bulk pass raises to BULK_CONC after the preflight report shows the observed rate-limit count.
CONC=${CONC:-24}
BULK_CONC=${BULK_CONC:-48}
cd "$W"; mkdir -p logs judge analysis
P=inputs/SCALE_PROMPT_v4.md
IN=judge/leg1_input.jsonl
OUT=judge/leg1_scored.jsonl
[ -s "$IN" ] || { echo ">>> BLOCKED: $IN missing"; exit 1; }

KEPT=$($PY - <<'PYEOF'
import json
print(sum(1 for l in open("judge/leg1_input.jsonl", encoding="utf-8")
          if l.strip() and not json.loads(l)["degen_flag"]))
PYEOF
)
echo ">>> leg1 kept trajectories to judge: $KEPT"

# ---- 1. calibration: the frozen low/high transcript examples through the strict judge ----
if [ ! -s judge/calibration_report.json ]; then
  $PY scripts/judge_calibration.py --prompt "$P" --out judge/calibration.jsonl \
      --report judge/calibration_report.json --result judge_calibration.json \
      >> logs/leg1_judge.log 2>&1 || { echo ">>> BLOCKED: strict-judge calibration failed"; exit 1; }
fi
grep -q '"status": "PASS"' judge_calibration.json || {
  echo ">>> BLOCKED: strict-judge calibration did not PASS"; exit 2; }

# ---- 2. the DESIGN 72-row preflight on the ACTUAL chat rollouts, drawn by LOWEST STABLE HASH ----
if [ ! -s judge/preflight_ids.json ]; then
  $PY - <<'PYEOF'
import hashlib, json
rows = [json.loads(l) for l in open("judge/leg1_input.jsonl", encoding="utf-8") if l.strip()]
kept = [r for r in rows if not r["degen_flag"] and r["subject"] not in ("stock",)]
kept.sort(key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
json.dump([r["id"] for r in kept[:72]], open("judge/preflight_ids.json", "w"), indent=2)
print(f">>> preflight sample: {min(72, len(kept))} of {len(kept)} kept trained-subject trajectories")
PYEOF
fi
if [ ! -s judge/preflight_report.json ]; then
  $PY scripts/judge_pass.py --prompt "$P" --schema single --axis despair --input "$IN" \
      --out "$OUT" --concurrency "$CONC" --skip-degen \
      --ids-file judge/preflight_ids.json --report judge/preflight_report.json \
      >> logs/leg1_judge.log 2>&1 || { echo ">>> BLOCKED: preflight pass failed"; exit 1; }
fi
$PY scripts/judge_preflight_check.py --input "$IN" --scored "$OUT" --ids judge/preflight_ids.json \
    --report judge/preflight_report.json --axes despair --out judge_preflight_check.json | tail -40
pfrc=${PIPESTATUS[0]}
[ "$pfrc" -eq 0 ] || { echo ">>> BLOCKED: judge preflight validation failed"; exit 1; }

# ---- 3. re-price from the ACTUAL metered rate against the DESIGN ceilings ----
DONE=$($PY - <<'PYEOF'
import json
n=0
for l in open("judge/leg1_scored.jsonl", encoding="utf-8"):
    if not l.strip(): continue
    o=json.loads(l)
    if isinstance(o.get("despair"),int) or o.get("status")=="judge_refusal": n+=1
print(n)
PYEOF
)
REMAIN=$(( KEPT - DONE ))
[ "$REMAIN" -lt 0 ] && REMAIN=0
# Leg-2 separate path: four calls per kept Leg-2 conversation (merged validation FELL BACK)
LEG2=$(( ${LEG2_CONVS:-1440} * 4 ))
$PY scripts/cost_ledger.py --work "$W" --meters meters/meter_*.json \
    --judge-reports validation/merged_judge_report.json judge/preflight_report.json \
    --planned-judge-calls-remaining $(( REMAIN + LEG2 )) \
    --out cost_ledger_leg1_preflight.json >> logs/leg1_judge.log 2>&1
lrc=$?
$PY -c "import json;d=json.load(open('cost_ledger_leg1_preflight.json'));print('>>> ledger',d['status'],d['projected_total_usd_low'],'-',d['projected_total_usd_high'],'actual',d['actual_to_date_usd'])"
if [ "$lrc" -eq 3 ]; then
  echo ">>> BLOCKED: projected/actual spend reaches the \$350 hard stop; not launching the bulk pass"
  touch logs/.judge_hard_stop; exit 3
fi

# ---- 4. bulk pass over every kept conversation ----
RL=$($PY -c "import json;print(json.load(open('judge/preflight_report.json')).get('rate_limited',0))")
if [ "$RL" -gt 0 ]; then BULK_CONC=$CONC; echo ">>> preflight saw $RL rate-limits; holding concurrency at $CONC";
else echo ">>> preflight saw 0 rate-limits at concurrency $CONC; raising bulk concurrency to $BULK_CONC"; fi
$PY scripts/judge_pass.py --prompt "$P" --schema single --axis despair --input "$IN" \
    --out "$OUT" --concurrency "$BULK_CONC" --skip-degen \
    --report judge/leg1_judge_report.json >> logs/leg1_judge.log 2>&1
echo ">>> leg1 bulk judging exit=$?"

# retry-to-convergence for transport / parse failures only (identical prompt and model)
for pass in 2 3; do
  n=$($PY - <<'PYEOF'
import json
done=set()
for l in open("judge/leg1_scored.jsonl", encoding="utf-8"):
    if not l.strip(): continue
    o=json.loads(l)
    if isinstance(o.get("despair"),int) or o.get("status")=="judge_refusal": done.add(o["id"])
print(sum(1 for l in open("judge/leg1_input.jsonl", encoding="utf-8")
          if l.strip() and not json.loads(l)["degen_flag"] and json.loads(l)["id"] not in done))
PYEOF
)
  echo ">>> retry pass $pass: $n rows outstanding"
  [ "$n" -eq 0 ] && break
  $PY scripts/judge_pass.py --prompt "$P" --schema single --axis despair --input "$IN" \
      --out "$OUT" --concurrency "$BULK_CONC" --skip-degen \
      --report "judge/leg1_judge_report_retry${pass}.json" >> logs/leg1_judge.log 2>&1
done
touch logs/.leg1_judged.done
echo ">>> leg1 judge leg done $(date -u +%FT%TZ)"
