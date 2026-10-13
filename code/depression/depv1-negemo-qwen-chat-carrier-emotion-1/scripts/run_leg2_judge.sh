#!/usr/bin/env bash
# Detached driver for the Leg-2 four-axis judging leg.
# The merged prompt FAILED its frozen validation gate (merged_judge_validation.json = FALLBACK),
# so DESIGN mandates the FOUR SEPARATE frozen prompts on every kept Leg-2 conversation, and the
# 50-row in-population anger check is N.A. (it is gated on merged PASSING).
set -uo pipefail
W="${W:-$HOME/work/depv1-negemo-qwen-chat-carrier-emotion-1}"
PY=${PY:-$HOME/venvs/tinker-cd1/bin/python}
CONC=${CONC:-48}
cd "$W"; mkdir -p logs judge
IN=judge/leg2_input.jsonl
[ -s "$IN" ] || { echo ">>> BLOCKED: $IN missing"; exit 1; }
grep -q '"verdict": "FALLBACK"' merged_judge_validation.json || {
  grep -q '"verdict": "PASS"' merged_judge_validation.json || {
    echo ">>> BLOCKED: merged_judge_validation.json has no verdict"; exit 2; }; }

KEPT=$($PY - <<'PYEOF'
import json
print(sum(1 for l in open("judge/leg2_input.jsonl", encoding="utf-8")
          if l.strip() and not json.loads(l)["degen_flag"]))
PYEOF
)
echo ">>> leg2 kept trajectories to judge: $KEPT (x4 separate axes)"

run_axis () {  # $1 = axis, $2 = prompt file
  local ax=$1 pf=$2
  $PY scripts/judge_pass.py --prompt "inputs/$pf" --schema single --axis "$ax" --input "$IN" \
      --out "judge/leg2_scored_${ax}.jsonl" --concurrency "$CONC" --skip-degen \
      --report "judge/leg2_judge_report_${ax}.json" >> "logs/leg2_judge_${ax}.log" 2>&1
  echo ">>> leg2 $ax exit=$?"
  for pass in 2 3; do
    n=$($PY - "$ax" <<'PYEOF'
import json, sys
ax=sys.argv[1]; done=set()
try:
    for l in open(f"judge/leg2_scored_{ax}.jsonl", encoding="utf-8"):
        if not l.strip(): continue
        o=json.loads(l)
        if isinstance(o.get(ax),int) or o.get("status")=="judge_refusal": done.add(o["id"])
except FileNotFoundError: pass
print(sum(1 for l in open("judge/leg2_input.jsonl", encoding="utf-8")
          if l.strip() and not json.loads(l)["degen_flag"] and json.loads(l)["id"] not in done))
PYEOF
)
    echo ">>> $ax retry pass $pass: $n outstanding"
    [ "$n" -eq 0 ] && break
    $PY scripts/judge_pass.py --prompt "inputs/$pf" --schema single --axis "$ax" --input "$IN" \
        --out "judge/leg2_scored_${ax}.jsonl" --concurrency "$CONC" --skip-degen \
        --report "judge/leg2_judge_report_${ax}_retry${pass}.json" >> "logs/leg2_judge_${ax}.log" 2>&1
  done
}

# the four axes are independent API populations -> run them concurrently, one pool each
run_axis despair SCALE_PROMPT_v4.md &  p1=$!
run_axis anger   AXIS_ANGER_v2.md   &  p2=$!
run_axis fear    AXIS_FEAR_v3.md    &  p3=$!
run_axis elation AXIS_ELATION_v1.md &  p4=$!
wait $p1 $p2 $p3 $p4
touch logs/.leg2_judged.done
echo ">>> leg2 judge leg done $(date -u +%FT%TZ)"
