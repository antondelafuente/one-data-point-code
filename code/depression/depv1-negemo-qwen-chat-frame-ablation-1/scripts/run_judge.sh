#!/usr/bin/env bash
# Detached driver for the strict-despair judging leg + its quote-validation loop.
#
# One claude-sonnet-5 call per mechanically-kept conversation with the frozen SCALE_PROMPT_v4
# rubric, then the parent's quote-grounding validation: any row whose quote is not a >=6-word
# verbatim shingle of its own transcript (markdown emphasis stripped from BOTH sides) is re-called
# under the IDENTICAL model and prompt, up to twice; a row still ungrounded after that stays
# explicit and unscored. Raw outputs are never overwritten -- merge_judge_validation.py layers the
# retries into despair_validated.jsonl and records what it replaced.
#
# Idempotent and resume-aware at every step (#357: a row counts done only with a real integer
# score or a recorded judge refusal). Per-stage: everything lands under judge/$STAGE.
set -uo pipefail
W="${W:?set W to the experiment work directory}"
E="${E:?set E to the experiment registry directory}"
PY=${PY:-/path/to/home/venvs/tinker-cd1/bin/python}
CONC=${CONC:-48}
STAGE="${STAGE:-main}"
cd "$W"; mkdir -p logs judge/"$STAGE"
echo $$ > logs/judge_self_$STAGE.pid   # this driver's OWN pid (a `setsid` $! can be the setsid wrapper)
J=judge/$STAGE
IN=$J/main_input.jsonl
LOG=logs/judge_$STAGE.log
PROMPT=inputs/SCALE_PROMPT_v4.md
[ -s "$IN" ] || { echo ">>> BLOCKED: $IN missing"; exit 1; }
[ -s "$PROMPT" ] || { echo ">>> BLOCKED: $PROMPT missing"; exit 1; }

judge () {  # judge <out.jsonl> <report.json> [extra args...]
  local out="$1"; local report="$2"; shift 2
  "$PY" "$E/scripts/judge_pass.py" --prompt "$PROMPT" --schema single --axis despair \
      --input "$IN" --out "$out" --concurrency "$CONC" --skip-degen --report "$report" \
      "$@" >> "$LOG" 2>&1
}

gate () {   # gate <scored.jsonl> <report.json> <gate.json> [ids-file]
  local scored="$1"; local report="$2"; local out="$3"; local ids="${4:-}"
  local extra=()
  [ -n "$ids" ] && extra=(--ids "$ids")
  "$PY" "$E/scripts/judge_preflight_check.py" --input "$IN" --scored "$scored" \
      --report "$report" --axes despair --skip-degen --out "$out" "${extra[@]}" >> "$LOG" 2>&1
  return 0    # the gate's exit code is advisory here; the retry loop consumes its findings
}

# ---- 1. bulk pass over every mechanically kept conversation --------------------------------
judge $J/despair.jsonl $J/despair_report.json
echo ">>> bulk judging exit=$?"
for pass in 2 3; do
  n=$("$PY" "$E/scripts/judge_helper.py" outstanding "$IN" $J/despair.jsonl 2>/dev/null || echo 0)
  echo ">>> transport/parse retry pass $pass: $n rows outstanding"
  [ "$n" -eq 0 ] && break
  judge $J/despair.jsonl "$J/despair_report_retry${pass}.json"
done

# ---- 2. quote-grounding validation loop ----------------------------------------------------
gate $J/despair.jsonl $J/despair_report.json $J/despair_quote_retries_gate.json
"$PY" "$E/scripts/extract_quote_retry_ids.py" --gate $J/despair_quote_retries_gate.json \
    --out $J/quote_retry_ids1.json | tee -a "$LOG"
if [ "$("$PY" -c 'import json,sys;print(len(json.load(open(sys.argv[1]))))' "$J/quote_retry_ids1.json")" -gt 0 ]; then
  judge $J/despair_quote_retries.jsonl $J/despair_quote_retries_report.json \
      --ids-file $J/quote_retry_ids1.json
  gate $J/despair_quote_retries.jsonl $J/despair_quote_retries_report.json \
      $J/despair_quote_retries2_gate.json $J/quote_retry_ids1.json
else
  : > $J/despair_quote_retries.jsonl
  echo '{"n_requested":0,"n_ok":0,"n_refusal":0,"n_failed":0,"calls":0,"in_tok":0,"out_tok":0,"cost_usd_estimate":0.0,"model":"claude-sonnet-5","note":"no ungrounded quotes after the bulk pass"}' \
      > $J/despair_quote_retries_report.json
  echo '{"status":"PASS","quote_ungrounded":[],"quote_ungrounded_all":[],"n_quote_ungrounded":0,"note":"no retry-1 population"}' \
      > $J/despair_quote_retries2_gate.json
fi
"$PY" "$E/scripts/extract_quote_retry_ids.py" --gate $J/despair_quote_retries2_gate.json \
    --out $J/quote_retry_ids2.json | tee -a "$LOG"
if [ "$("$PY" -c 'import json,sys;print(len(json.load(open(sys.argv[1]))))' "$J/quote_retry_ids2.json")" -gt 0 ]; then
  judge $J/despair_quote_retries2.jsonl $J/despair_quote_retries2_report.json \
      --ids-file $J/quote_retry_ids2.json
else
  : > $J/despair_quote_retries2.jsonl
  echo '{"n_requested":0,"n_ok":0,"n_refusal":0,"n_failed":0,"calls":0,"in_tok":0,"out_tok":0,"cost_usd_estimate":0.0,"model":"claude-sonnet-5","note":"no ungrounded quotes after retry 1"}' \
      > $J/despair_quote_retries2_report.json
fi

# ---- 3. merge without overwriting any raw output -------------------------------------------
"$PY" "$E/scripts/merge_judge_validation.py" --dir "$J" >> "$LOG" 2>&1
echo ">>> merge exit=$?"
touch logs/.judged_$STAGE.done
echo ">>> judge leg done $(date -u +%FT%TZ)"
