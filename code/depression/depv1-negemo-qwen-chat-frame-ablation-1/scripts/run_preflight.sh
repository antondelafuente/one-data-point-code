#!/usr/bin/env bash
# The DESIGN step-7 pre-spend gate: a balanced 24-conversation ACTUAL-POPULATION preflight across
# stock, the full-row anchor, one prefix arm and one placebo arm — generation, strict-despair
# judging, blind coherence, quote grounding — followed by a re-price of the whole run at a 1.5x
# margin against the $110 notify / $130 hard stop.
#
# Everything lands in preflight/ (its own 6-scenario battery slice), so the bulk orchestrator can
# never mistake this for a completed group. Idempotent at every step.
set -uo pipefail
W="${W:?set W to the experiment work directory}"
E="${E:?set E to the experiment registry directory}"
PY=${PY:-/path/to/home/venvs/tinker-cd1/bin/python}
cd "$W"; mkdir -p logs preflight
LOG=logs/preflight.log
PF=preflight
BAT=$PF/battery_first6.json

step () { echo ">>> [$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

step "plan"
[ -s "$PF/preflight_plan.json" ] || "$PY" "$E/scripts/prepare_preflight.py" --work "$W" --outdir "$PF" >> "$LOG" 2>&1 \
  || { echo ">>> BLOCKED: preflight plan failed"; exit 1; }

step "generation (24 conversations, draw 0, 6 scenarios)"
if [ ! -s "$PF/p00.summary.json" ]; then
  "$PY" "$E/scripts/gen_tinker.py" generate --specs "$PF/p00.specs.json" --battery "$BAT" \
      --leg main --group-id p00 --session-id main --driver preflight --draws 0 \
      --concurrency 48 --output "$PF/p00.jsonl" --summary "$PF/p00.summary.json" >> "$LOG" 2>&1
  echo ">>> preflight generation exit=$?" | tee -a "$LOG"
fi

step "trajectories + mechanical screen"
"$PY" "$E/scripts/build_trajectories.py" --rollouts "$PF/p00.jsonl" --battery "$BAT" --leg main \
    --subjects "$PF/subjects.json" --degen "$E/scripts/degen.py" --draws 0 \
    --out "$PF/trajectories.jsonl" --report "$PF/trajectory_report.json" >> "$LOG" 2>&1 \
  || { echo ">>> BLOCKED: preflight trajectories failed"; exit 1; }

step "strict despair (claude-sonnet-5, frozen SCALE_PROMPT_v4)"
if [ ! -s "$PF/despair_report.json" ]; then
  "$PY" "$E/scripts/judge_pass.py" --prompt inputs/SCALE_PROMPT_v4.md --schema single \
      --axis despair --input "$PF/trajectories.jsonl" --out "$PF/despair.jsonl" \
      --concurrency 24 --report "$PF/despair_report.json" >> "$LOG" 2>&1
  echo ">>> preflight despair exit=$?" | tee -a "$LOG"
fi

step "quote grounding / schema / model-identity gate"
"$PY" "$E/scripts/judge_preflight_check.py" --input "$PF/trajectories.jsonl" \
    --scored "$PF/despair.jsonl" --report "$PF/despair_report.json" --axes despair \
    --out "$PF/despair_gate.json" 2>&1 | tail -45 | tee -a "$LOG"
gaterc=${PIPESTATUS[0]}

step "blind coherence (gpt-5.6-luna, frozen v3 rubric)"
"$PY" "$E/scripts/classify_coherence.py" --input "$PF/trajectories.jsonl" \
    --output "$PF/coherence.jsonl" --model gpt-5.6-luna --rubric-version v3 \
    --concurrency 24 --hard-stop-usd 2.0 >> "$LOG" 2>&1
echo ">>> preflight coherence exit=$?" | tee -a "$LOG"

step "re-price the whole run at a 1.5x margin"
"$PY" "$E/scripts/cost_reprice.py" --work "$W" --preflight "$PF" --out "$PF/cost_reprice.json" \
    2>&1 | tail -40 | tee -a "$LOG"
costrc=${PIPESTATUS[0]}

step "fail-closed preflight verification"
"$PY" "$E/scripts/verify_preflight.py" --dir "$PF" 2>&1 | tail -60 | tee -a "$LOG"
vrc=${PIPESTATUS[0]}

touch logs/.preflight.done
echo ">>> preflight finished gate_rc=$gaterc cost_rc=$costrc verify_rc=$vrc $(date -u +%FT%TZ)" | tee -a "$LOG"
exit "$vrc"
