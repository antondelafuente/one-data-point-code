#!/bin/bash
# close_analysis.sh — the deterministic analysis chain that runs once judging is complete.
# Everything here is mechanical and idempotent; the JUDGMENT steps (the cross-family audits, the
# audit responses, RESULTS.md's prose) are deliberately NOT in here.
#
#   1. rollout audit copy + deterministic layer-1 audit (surface c)
#   2. the inherited unmatched-</think> sensitivity: build the think-stripped rows, judge them with
#      the SAME pinned judge, then report the movement (raw rows stay raw)
#   3. aggregate -> exhibits -> closest-item sensitivity -> figures
#   4. report_numbers.json — the single source RESULTS.md quotes from
set -uo pipefail
set -a; source /path/to/home/.env; set +a
unset OPENAI_API_KEY_PERSONAL_BACKUP OPENROUTER_API_KEY_PERSONAL_BACKUP
W=/path/to/home/work/csp1-refusal-geometry-complete16-1; export RGC16_WORK=$W
R=/path/to/home/ws/run/csp1-refusal-geometry-complete16-1/registry/csp1-refusal-geometry-complete16-1
export PYTHONPATH=$R/scripts
PY=/path/to/home/venvs/tinker-cd1/bin/python
VC=/path/to/home/automated-researcher/plugins/verify-claims/skills/verify-claims/scripts
S=session1
cd "$R/scripts"
step(){ echo ">>> [$(date -u +%H:%M:%S)] $*"; }

step "1. rollout audit copy (surface c)"
$PY build_audit_copies.py --surface rollouts --stage $S || exit 1
step "1b. deterministic layer-1 audit on the rollouts"
$PY "$VC/audit_data.py" "$W/data_audit/audit_rollouts.jsonl" \
    --out "$W/data_audit/data_audit_rollouts.json" \
    --sample "$W/data_audit/data_audit_rollouts_sample.jsonl" \
    --label-field audit_stratum --dup-field response --text-field response \
    --require-finish-reason --sample-k 6
rc=$?; echo ">>> audit_data rc=$rc (2 = HARD FAIL)"
[ $rc -eq 2 ] && { echo ">>> BLOCKED: rollout layer-1 audit HARD FAIL"; exit 2; }
cp "$W/data_audit/data_audit_rollouts.json" "$R/data_audit_rollouts.json"
cp "$W/data_audit/data_audit_rollouts_sample.jsonl" "$R/data_audit_rollouts_sample.jsonl"

step "2. unmatched-</think> sensitivity: build the think-stripped rows"
$PY build_think_strip_rows.py --stage $S --out "$W/$S/judged/think_stripped_rows.jsonl" || exit 1
n=$(wc -l < "$W/$S/judged/think_stripped_rows.jsonl")
step "2b. judging $n think-stripped rows with the SAME pinned judge"
if [ "$n" -gt 0 ]; then
  for c in 1 2; do
    $PY judge_task2.py --rows-file "$W/$S/judged/think_stripped_rows.jsonl" \
        --out "$W/$S/judged/task2_think_stripped.jsonl" --conc 200 \
        --receipt "$W/$S/receipts/judge_think_stripped_$c.json" >> "$W/logs/judge_think_stripped.log" 2>&1
  done
fi
step "2c. rollout sensitivities (degeneracy screen + think sensitivity)"
REJ=()
[ -s "$W/$S/judged/task2_think_stripped.jsonl" ] && REJ=(--rejudged "$W/$S/judged/task2_think_stripped.jsonl")
$PY rollout_sensitivities.py "${REJ[@]}" || exit 1

step "3. aggregate"
$PY aggregate.py || exit 1
step "3b. exhibits"
$PY exhibits.py || exit 1
step "3c. closest-item sensitivity"
$PY closest_items_sensitivity.py || exit 1
step "3d. figures"
$PY render_figures.py || exit 1

step "4. reported numbers"
$PY report_numbers.py || exit 1
$PY report_numbers.py --check || { echo ">>> BLOCKED: reported_numbers does not match the CSVs"; exit 1; }

# the row-level score file is large and lives in R2, not git: pin its hash for the reproduction gate
sha256sum "$W/figures/question_scores.jsonl" | awk '{print $1"  question_scores.jsonl"}' \
  > "$W/figures/question_scores.sha256"

touch "$W/.close_analysis.done"
step "CLOSE ANALYSIS COMPLETE"
