#!/usr/bin/env bash
# The whole downstream pass for depv1-negemo-selfteacher-1, in run order, once all
# NINE subjects have been generated and all three self pools built.
#
#   merge rollouts -> construction/invariant verification -> ONE blinded judge wave
#   (population smoke + re-price first) -> aggregation -> length holding -> battery
#   leakage sensitivity -> mode-leak sensitivity -> transcript exhibit -> figures
#
# Idempotent: every step either skips when its output exists or resumes
# success-aware, so re-running after an interruption continues rather than redoes.
#
# usage: finish_dst1.sh [--skip-wave]
set -uo pipefail
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/path/to/home/venvs/dcs1/bin/python
W="$EXP/work"
SUBJECTS="qwen-stock qwen-gemma qwen-self small-stock small-gemma small-self large-stock large-gemma large-self"
SKIP_WAVE=0
[ "${1:-}" = "--skip-wave" ] && SKIP_WAVE=1

mkdir -p "$W/judging" "$W/analysis" "$W/audit"

echo "=== 1. merge rollouts (all nine subjects) ==="
: > "$W/generation/rollouts_all.jsonl"
for s in $SUBJECTS; do
  f="$W/generation/${s}.rollouts.jsonl"
  [ -f "$f" ] || { echo "MISSING $f"; exit 1; }
  cat "$f" >> "$W/generation/rollouts_all.jsonl"
done
wc -l < "$W/generation/rollouts_all.jsonl"

echo "=== 2. eval-construction check (pinned to the parent's generate_cells.py) ==="
"$PY" "$EXP/scripts/verify_eval_construction.py" \
  --rollouts "$W/generation/rollouts_all.jsonl" \
  --eval-inputs "$W/inputs/eval_inputs.jsonl" \
  --battery "$W/inputs/normal_prompt_battery.json" \
  --out "$EXP/eval_construction_check.json" || exit 1

echo "=== 3. rollout invariants (mode pinning, decoding, co-measurement) ==="
"$PY" "$EXP/scripts/verify_rollout_invariants_dst1.py" \
  --rollouts "$W/generation/rollouts_all.jsonl" \
  --gen-dir "$W/generation" \
  --train-receipts "$W"/state/train_*-gemma.json "$W"/state/train_*-self.json \
  --out "$EXP/rollout_invariants.json" || exit 1

echo "=== 4. build the ONE blinded judge wave (incl. the 900 spot-rate rows) ==="
"$PY" "$EXP/scripts/prepare_judge_inputs_dst1.py" \
  --rollouts "$W/generation/rollouts_all.jsonl" \
  --self-pool "$W/pools/selfpool_qwen.jsonl" \
  --self-pool "$W/pools/selfpool_small.jsonl" \
  --self-pool "$W/pools/selfpool_large.jsonl" \
  --out "$W/judging/judge_inputs.jsonl" --join "$W/judging/judge_join.jsonl" \
  --receipt "$W/judging/judge_inputs_receipt.json" || exit 1

if [ "$SKIP_WAVE" = "0" ]; then
  echo "=== 5. judge balance pre-flight (#354) ==="
  "$PY" "$EXP/scripts/judge_wave_dst1.py" balance | tee "$W/judging/judge_balance.json"

  echo "=== 6. judge population smoke (100 rows of the ACTUAL population) + re-price ==="
  "$PY" "$EXP/scripts/judge_wave_dst1.py" preflight \
    --inputs "$W/judging/judge_inputs.jsonl" --join "$W/judging/judge_join.jsonl" \
    --out "$W/judging/judgments.jsonl" --n 100 --concurrency 64 \
    --receipt "$W/judging/judge_preflight_receipt.json" || exit 1

  # HARD GATE on the population smoke's measured $/call (START.md's re-price rule).
  # The driver's --hard-stop-usd below is a mechanical backstop that would abort the
  # wave PART-WAY, leaving a partial population; this gate refuses to start a wave we
  # already know cannot finish inside the run's remaining headroom, which is the
  # honest stop (notify + request authorisation) rather than a truncated instrument.
  "$PY" - "$W" <<'GATE' || exit 1
import json, sys
W = sys.argv[1]
pre = json.load(open(f"{W}/judging/judge_preflight_receipt.json"))
rows = sum(1 for _ in open(f"{W}/judging/judge_inputs.jsonl"))
rate = pre.get("per_call_usd")
head = json.load(open(f"{W}/state/reprice_final.json"))["judge_headroom_under_hard_stop_usd"]
if not rate:
    print("BLOCKED: preflight produced no per-call rate"); sys.exit(1)
remaining = rows - pre["judged"]
proj = remaining * rate
print(f">>> judge repriced: {rate:.6f} $/call x {remaining} remaining = ${proj:.2f} "
      f"(headroom ${head:.2f}, preflight already spent ${pre['spend_usd']:.2f})")
if proj > head:
    print(f"BLOCKED: projected wave ${proj:.2f} exceeds the ${head:.2f} headroom under "
          f"the $850 hard stop. STOP and request new authorisation from the "
          f"designer-of-record; do NOT run a partial wave.")
    sys.exit(1)
print(">>> judge re-price gate PASS")
GATE

  # The judge hard-stop is NOT a round number: work/state/reprice_final.json prices
  # every other leg from measured censuses/pools at $708 (+~$13 of self-subject eval
  # generation still to land), leaving ~$128 under the run's $850 HARD STOP. The wave
  # is projected at $122. If the population smoke above re-prices it ABOVE this
  # headroom, that is a hard-stop condition: stop and get new authorization rather
  # than letting the driver abort mid-wave on a partial population.
  echo "=== 7. THE WAVE ==="
  "$PY" "$EXP/scripts/judge_wave_dst1.py" wave \
    --inputs "$W/judging/judge_inputs.jsonl" --join "$W/judging/judge_join.jsonl" \
    --out "$W/judging/judgments.jsonl" --concurrency 192 --hard-stop-usd 128 \
    --receipt "$W/judging/judge_wave_receipt.json" || exit 1
fi

echo "=== 8. aggregation (the headline) ==="
"$PY" "$EXP/scripts/analyze_dst1.py" \
  --rollouts "$W/generation/rollouts_all.jsonl" --join "$W/judging/judge_join.jsonl" \
  --judgments "$W/judging/judgments.jsonl" --gen-dir "$W/generation" \
  --train-receipts "$W"/state/train_*-gemma.json "$W"/state/train_*-self.json \
  --selfpool-receipts "$W"/pools/selfpool_*.receipt.json \
  --gemma-census "$W"/state/census_*-gemma.json \
  --out "$W/analysis" || exit 1

echo "=== 9. length-holding analyses (first-class, reported WITH the headline) ==="
for view in full prefix640; do
  "$PY" "$EXP/scripts/length_stratified.py" \
    --rollouts "$W/generation/rollouts_all.jsonl" --join "$W/judging/judge_join.jsonl" \
    --judgments "$W/judging/judgments.jsonl" --view "$view" \
    --out "$W/analysis/length_stratified_${view}.json" || exit 1
done

echo "=== 10. battery leakage sensitivity (the 5 same-task overlaps, both ways) ==="
"$PY" "$EXP/scripts/battery_leakage_sensitivity.py" \
  --join "$W/judging/judge_join.jsonl" --judgments "$W/judging/judgments.jsonl" \
  --recheck "$EXP/reference/parent_leakage_recheck.json" \
  --out "$W/analysis/battery_leakage_sensitivity.json" || exit 1

echo "=== 11. reasoning/mode-leak sensitivity ==="
"$PY" "$EXP/scripts/mode_leak_sensitivity.py" \
  --rollouts "$W/generation/rollouts_all.jsonl" --join "$W/judging/judge_join.jsonl" \
  --judgments "$W/judging/judgments.jsonl" \
  --out "$W/analysis/mode_leak_sensitivity.json" || exit 1

echo "=== 12. transcript exhibit (by RULE, incl. 5 seeded rows per self pool) ==="
"$PY" "$EXP/scripts/select_exhibits_dst1.py" \
  --rollouts "$W/generation/rollouts_all.jsonl" --join "$W/judging/judge_join.jsonl" \
  --judgments "$W/judging/judgments.jsonl" \
  --self-pool "$W/pools/selfpool_qwen.jsonl" \
  --self-pool "$W/pools/selfpool_small.jsonl" \
  --self-pool "$W/pools/selfpool_large.jsonl" \
  --out-json "$EXP/transcript_exhibit.json" --out-md "$EXP/TRANSCRIPT_EXHIBIT.md" || exit 1

echo "=== 13. figures (the locked Presentation spec) ==="
"$PY" "$EXP/scripts/make_figures_dst1.py" \
  --table "$W/analysis/selfteacher_table.json" \
  --length "$W/analysis/length_stratified_full.json" \
  --out "$EXP/figures" || exit 1
cp "$W/analysis/selfteacher_table.csv" "$W/analysis/selfteacher_table.json" \
   "$W/analysis/per_response.csv" "$W/analysis/self_pool_stats.csv" "$EXP/figures/"

echo "=== finish_dst1.sh COMPLETE ==="
touch "$W/analysis/.finish.done"
