#!/bin/bash
# finish_dcs1.sh — everything downstream of generation, in one idempotent pass.
#
# merge rollouts -> construction + invariant gates -> blinded judge wave (ONE wave,
# resuming into the same judgments file the population smoke already wrote into)
# -> aggregation -> the two length-holding analyses -> battery leakage sensitivity
# -> figures -> transcript exhibit.
#
# Every step is skip-if-done, so a re-run after a fix costs only the step that changed.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/path/to/home/venvs/dcs1/bin/python
W="$EXP_DIR/work"
S="$EXP_DIR/scripts"
CONC="${JUDGE_CONCURRENCY:-192}"
HARD_STOP="${JUDGE_HARD_STOP_USD:-150}"

cd "$EXP_DIR"
mkdir -p "$W/judging" figures

echo ">>> merging rollouts"
cat "$W"/generation/nemotron_stock.rollouts.jsonl \
    "$W"/generation/nemotron_clean17995_s3.rollouts.jsonl \
    "$W"/generation/inkling_stock.rollouts.jsonl \
    "$W"/generation/inkling_clean17995_s3.rollouts.jsonl \
    > "$W/generation/rollouts_all.jsonl"
wc -l "$W/generation/rollouts_all.jsonl"

echo ">>> eval-construction gate"
$PY "$S/verify_eval_construction.py" \
  --rollouts "$W/generation/rollouts_all.jsonl" \
  --eval-inputs "$W/inputs/eval_inputs.jsonl" \
  --battery "$W/inputs/normal_prompt_battery.json" \
  --out "$EXP_DIR/eval_construction_check.json" | tail -20

echo ">>> rollout-invariant gate"
$PY "$S/verify_rollout_invariants.py" \
  --rollouts "$W/generation/rollouts_all.jsonl" \
  --train-receipts "$W/state/train_nemotron.json" "$W/state/train_inkling.json" \
  --gen-dir "$W/generation" \
  --out "$EXP_DIR/rollout_invariants.json" | tail -40

echo ">>> layer-1 data audit on the merged rollouts"
python3 /path/to/home/automated-researcher/plugins/verify-claims/skills/verify-claims/scripts/audit_data.py \
  "$W/generation/rollouts_all.jsonl" \
  --out "$EXP_DIR/data_audit_rollouts.json" \
  --sample "$EXP_DIR/data_audit_rollouts_sample.json" \
  --label-field subject --text-field judge_visible_text \
  --dup-field judge_visible_text --require-finish-reason

echo ">>> building the ONE blinded judge wave's inputs"
$PY "$S/prepare_judge_inputs.py" \
  --rollouts "$W/generation/rollouts_all.jsonl" \
  --out "$W/judging/judge_inputs.jsonl" \
  --join "$W/judging/judge_join.jsonl" \
  --receipt "$W/judging/judge_inputs_receipt.json" | tail -25

echo ">>> judge balance pre-flight"
$PY "$S/judge_wave.py" balance

echo ">>> judge wave (resumes into the same judgments file as the population smoke)"
$PY "$S/judge_wave.py" wave \
  --inputs "$W/judging/judge_inputs.jsonl" --join "$W/judging/judge_join.jsonl" \
  --out "$W/judging/judgments.jsonl" --receipt "$W/judging/judge_wave_receipt.json" \
  --concurrency "$CONC" --hard-stop-usd "$HARD_STOP"

echo ">>> aggregation"
$PY "$S/analyze_dcs1.py" \
  --rollouts "$W/generation/rollouts_all.jsonl" --join "$W/judging/judge_join.jsonl" \
  --judgments "$W/judging/judgments.jsonl" --gen-dir "$W/generation" \
  --train-receipts "$W/state/train_nemotron.json" "$W/state/train_inkling.json" \
  --out "$EXP_DIR/figures"

echo ">>> length-holding analyses"
$PY "$S/length_stratified.py" \
  --rollouts "$W/generation/rollouts_all.jsonl" --join "$W/judging/judge_join.jsonl" \
  --judgments "$W/judging/judgments.jsonl" --view full \
  --out "$EXP_DIR/figures/length_stratified_full.json"
$PY "$S/length_stratified.py" \
  --rollouts "$W/generation/rollouts_all.jsonl" --join "$W/judging/judge_join.jsonl" \
  --judgments "$W/judging/judgments.jsonl" --view prefix640 \
  --out "$EXP_DIR/figures/length_stratified_prefix640.json"

echo ">>> battery leakage sensitivity"
$PY "$S/battery_leakage_sensitivity.py" \
  --join "$W/judging/judge_join.jsonl" --judgments "$W/judging/judgments.jsonl" \
  --recheck "$EXP_DIR/reference/parent_leakage_recheck.json" \
  --out "$EXP_DIR/figures/battery_leakage_sensitivity.json"

echo ">>> reasoning-channel accounting + sensitivity"
$PY "$S/mode_leak_sensitivity.py" \
  --rollouts "$W/generation/rollouts_all.jsonl" --join "$W/judging/judge_join.jsonl" \
  --judgments "$W/judging/judgments.jsonl" \
  --out "$EXP_DIR/figures/mode_leak_sensitivity.json" | tail -30

echo ">>> figures"
$PY "$S/make_figures_dcs1.py" --table "$EXP_DIR/figures/substrate_table.json" \
  --out "$EXP_DIR/figures"

echo ">>> transcript exhibit"
$PY "$S/select_exhibits_dcs1.py" \
  --rollouts "$W/generation/rollouts_all.jsonl" --join "$W/judging/judge_join.jsonl" \
  --judgments "$W/judging/judgments.jsonl" \
  --out-json "$EXP_DIR/transcript_exhibit.json" \
  --out-md "$EXP_DIR/TRANSCRIPT_EXHIBIT.md" | tail -20

echo ">>> FINISH_DCS1 DONE"
