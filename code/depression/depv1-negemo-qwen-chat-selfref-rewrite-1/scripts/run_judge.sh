#!/usr/bin/env bash
# Detached driver for ONE judging leg. Parameterised by LEG / AXIS / PROMPT so the attack leg runs
# the frozen strict-despair rubric and the AMENDMENT-A1 praise leg runs the frozen elation rubric
# through the identical machinery — one axis, one call per mechanically-kept conversation,
# claude-sonnet-5, retries for transport/parse failures only.
#
#   LEG=main   AXIS=despair PROMPT=inputs/SCALE_PROMPT_v4.md   (competence-attack battery)
#   LEG=praise AXIS=elation PROMPT=inputs/AXIS_ELATION_v1.md   (praise battery, AMENDMENT A1)
#
# Despair is never scored on praise conversations and elation is never scored on attack
# conversations: the two legs read two different trajectory files and write two different score
# files, so the separation is structural rather than a filter.
#
# The 24-row actual-population preflight meters + validates BEFORE the bulk pass; the cost ledger
# then re-prices against the $260 notify / $335 hard stop (AMENDMENT A1). Idempotent and
# resume-aware at every step.
set -uo pipefail
W="${W:-$HOME/work/depv1-negemo-qwen-chat-selfref-rewrite-1}"
PY=${PY:-$HOME/venvs/tinker-cd1/bin/python}
LEG=${LEG:-main}
AXIS=${AXIS:-despair}
P=${PROMPT:-inputs/SCALE_PROMPT_v4.md}
# DESIGN: "begin 24, adjust to measured input-token budget". Preflight runs at 24; the bulk pass
# raises to BULK_CONC only if the preflight report shows zero rate-limits.
CONC=${CONC:-24}
BULK_CONC=${BULK_CONC:-48}
export LEG AXIS
cd "$W"; mkdir -p logs judge analysis
IN=judge/${LEG}_input.jsonl
OUT=judge/${LEG}_scored_${AXIS}.jsonl
LOG=logs/judge_${LEG}_${AXIS}.log
[ -s "$IN" ] || { echo ">>> BLOCKED: $IN missing"; exit 1; }

KEPT=$("$PY" scripts/judge_helper.py kept)
echo ">>> [$LEG/$AXIS] kept trajectories to judge: $KEPT"

# ---- 1. calibration: the frozen low/high transcript examples through the strict judge ----
if [ ! -s "judge/calibration_report_${AXIS}.json" ]; then
  "$PY" scripts/judge_calibration.py --prompt "$P" --out "judge/calibration_${AXIS}.jsonl" \
      --report "judge/calibration_report_${AXIS}.json" --result "judge_calibration_${AXIS}.json" \
      --frozen-scores "${FROZEN_SCORES:-full_SCALE_PROMPT_v4.jsonl}" \
      >> "$LOG" 2>&1 || { echo ">>> BLOCKED: judge calibration failed for $AXIS"; exit 1; }
fi
grep -q '"status": "PASS"' "judge_calibration_${AXIS}.json" || {
  echo ">>> BLOCKED: judge calibration did not PASS for $AXIS"; exit 2; }

# ---- 2. the 24-row preflight on the ACTUAL rollouts, balanced across
#         stock / original_I / third_person / impersonal, drawn by LOWEST STABLE HASH ----
[ -s "judge/preflight_ids_${LEG}.json" ] || "$PY" scripts/judge_helper.py preflight_ids
if [ ! -s "judge/preflight_report_${LEG}_${AXIS}.json" ]; then
  "$PY" scripts/judge_pass.py --prompt "$P" --schema single --axis "$AXIS" --input "$IN" \
      --out "$OUT" --concurrency "$CONC" --skip-degen \
      --ids-file "judge/preflight_ids_${LEG}.json" \
      --report "judge/preflight_report_${LEG}_${AXIS}.json" \
      >> "$LOG" 2>&1 || { echo ">>> BLOCKED: preflight pass failed"; exit 1; }
fi
"$PY" scripts/judge_preflight_check.py --input "$IN" --scored "$OUT" \
    --ids "judge/preflight_ids_${LEG}.json" \
    --report "judge/preflight_report_${LEG}_${AXIS}.json" --axes "$AXIS" \
    --calibration "judge_calibration_${AXIS}.json" \
    --out "judge_preflight_check_${LEG}_${AXIS}.json" | tail -40
pfrc=${PIPESTATUS[0]}
[ "$pfrc" -eq 0 ] || { echo ">>> BLOCKED: judge preflight validation failed"; exit 1; }

# ---- 3. re-price from the ACTUAL metered rate against the DESIGN + A1 ceilings ----
DONE=$("$PY" scripts/judge_helper.py done_count)
REMAIN=$(( KEPT - DONE )); [ "$REMAIN" -lt 0 ] && REMAIN=0
"$PY" scripts/cost_ledger.py --work "$W" --meters meters/meter_*.json \
    --judge-reports judge/preflight_report_*.json \
    --planned-judge-calls-remaining "$REMAIN" \
    --out "cost_ledger_preflight_${LEG}_${AXIS}.json" >> "$LOG" 2>&1
lrc=$?
"$PY" -c "import json;d=json.load(open('cost_ledger_preflight_${LEG}_${AXIS}.json'));print('>>> ledger',d['status'],d['projected_total_usd_low'],'-',d['projected_total_usd_high'],'actual',d['actual_to_date_usd'])"
if [ "$lrc" -eq 3 ]; then
  echo ">>> BLOCKED: projected/actual spend reaches the \$335 hard stop; not launching the bulk pass"
  touch "logs/.judge_hard_stop_${LEG}_${AXIS}"; exit 3
fi

# ---- 4. bulk pass over every mechanically kept conversation ----
RL=$("$PY" -c "import json;print(json.load(open('judge/preflight_report_${LEG}_${AXIS}.json')).get('rate_limited',0))")
if [ "$RL" -gt 0 ]; then BULK_CONC=$CONC; echo ">>> preflight saw $RL rate-limits; holding concurrency at $CONC";
else echo ">>> preflight saw 0 rate-limits at concurrency $CONC; raising bulk concurrency to $BULK_CONC"; fi
"$PY" scripts/judge_pass.py --prompt "$P" --schema single --axis "$AXIS" --input "$IN" \
    --out "$OUT" --concurrency "$BULK_CONC" --skip-degen \
    --report "judge/${LEG}_judge_report_${AXIS}.json" >> "$LOG" 2>&1
echo ">>> [$LEG/$AXIS] bulk judging exit=$?"

# retry-to-convergence for transport / parse failures only (identical prompt and model)
for pass in 2 3; do
  n=$("$PY" scripts/judge_helper.py outstanding)
  echo ">>> retry pass $pass: $n rows outstanding"
  [ "$n" -eq 0 ] && break
  "$PY" scripts/judge_pass.py --prompt "$P" --schema single --axis "$AXIS" --input "$IN" \
      --out "$OUT" --concurrency "$BULK_CONC" --skip-degen \
      --report "judge/${LEG}_judge_report_${AXIS}_retry${pass}.json" >> "$LOG" 2>&1
done
touch "logs/.judged_${LEG}_${AXIS}.done"
echo ">>> [$LEG/$AXIS] judge leg done $(date -u +%FT%TZ)"
