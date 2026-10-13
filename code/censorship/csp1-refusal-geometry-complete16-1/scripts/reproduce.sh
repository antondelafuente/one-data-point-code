#!/bin/bash
# Reproduce every committed figure/table for csp1-refusal-geometry-complete16-1 from a fresh pull of
# the durable artifacts (no regeneration, no re-judging).
#
# FAIL-CLOSED (close-audit finding 2): every expected reference MUST exist and MUST compare equal.
# An absent reference is a FAILURE, not a skipped comparison — the previous version conditionally
# skipped missing files and could therefore exit 0 having compared nothing at all.
#
#   bash scripts/reproduce.sh            # pull + rebuild + diff
#   bash scripts/reproduce.sh --no-pull
set -uo pipefail
set -a; source /path/to/home/.env; set +a
unset OPENAI_API_KEY_PERSONAL_BACKUP OPENROUTER_API_KEY_PERSONAL_BACKUP
EXP=csp1-refusal-geometry-complete16-1
STAGE=session1
W=${RGC16_WORK:-$HOME/work/$EXP}; export RGC16_WORK="$W"
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
R2=remote:bucket/experiments/$EXP
PY=/path/to/home/venvs/tinker-cd1/bin/python
export PYTHONPATH="$R/scripts"
rc=0
fail(){ echo "    FAIL $*"; rc=1; }

if [ "${1:-}" != "--no-pull" ]; then
  mkdir -p "$W"/{figures,training_data,leakage,receipts,design_inputs,gsm8k,data_audit}
  mkdir -p "$W/$STAGE"/{gen,judged,receipts,design_inputs}
  rclone copy "$R2/$STAGE/rollouts"      "$W/$STAGE/gen"           --checksum --transfers 8
  rclone copy "$R2/$STAGE/judged"        "$W/$STAGE/judged"        --checksum --transfers 8
  rclone copy "$R2/$STAGE/design_inputs" "$W/$STAGE/design_inputs" --checksum
  rclone copy "$R2/$STAGE/receipts"      "$W/$STAGE/receipts"      --checksum
  rclone copy "$R2/design_inputs"        "$W/design_inputs"        --checksum --transfers 8
  rclone copy "$R2/leakage"              "$W/leakage"              --checksum
  rclone copy "$R2/receipts"             "$W/receipts"             --checksum
  rclone copy "$R2/gsm8k"                "$W/gsm8k"                --checksum
  rclone copy "$R2/training_data"        "$W/training_data"        --checksum --transfers 8
fi
cd "$R/scripts"

# ---- regenerate EVERY reported output, each from its complete inputs ----
$PY audit_shards.py --stage "$STAGE" --out "$W/figures/shard_audit_repro.json" > /dev/null || fail "audit_shards"
$PY token_mass.py > /dev/null                     || fail "token_mass"
$PY aggregate.py > /dev/null                      || fail "aggregate"
$PY exhibits.py > /dev/null                       || fail "exhibits"
$PY closest_items_sensitivity.py > /dev/null      || fail "closest_items_sensitivity"
$PY short_output_sensitivity.py > /dev/null       || fail "short_output_sensitivity"
# the think sensitivity NEEDS its re-judged pool; its absence is a failure, not a skip
TS="$W/$STAGE/judged/task2_think_stripped.jsonl"
[ -s "$TS" ] || fail "missing think-stripped re-judged pool $TS (required for the reported sensitivity)"
[ -s "$TS" ] && { $PY rollout_sensitivities.py --rejudged "$TS" > /dev/null || fail "rollout_sensitivities"; }
$PY render_figures.py > /dev/null                 || fail "render_figures"
$PY report_numbers.py > /dev/null                 || fail "report_numbers"
$PY report_numbers.py --check                     || fail "report_numbers --check (RESULTS numbers vs the CSVs)"

# ---- compare against EVERY committed reference; absent == FAIL ----
EXPECTED_CSV="cells.csv matrix.csv directional_contrasts.csv group_contrasts.csv controls.csv
              seed_stability.csv gsm8k.csv token_mass.csv closest_items_sensitivity.csv"
EXPECTED_JSON="floor_ceiling.json exhibits.json subjects.json aggregate_summary.json
               reported_numbers.json rollout_sensitivities.json short_output_sensitivity.json"
n_cmp=0
for f in $EXPECTED_CSV $EXPECTED_JSON; do
  if [ ! -f "$R/figures/$f" ]; then fail "expected reference MISSING from the record: figures/$f"; continue; fi
  if [ ! -f "$W/figures/$f" ]; then fail "regeneration produced no figures/$f"; continue; fi
  if diff -q "$R/figures/$f" "$W/figures/$f" >/dev/null; then echo "    OK   $f"; n_cmp=$((n_cmp+1));
  else fail "DIFF $f"; fi
done
for f in receipts/token_mass.json; do
  if [ ! -f "$R/$f" ]; then fail "expected reference MISSING: $f"; continue; fi
  if diff -q "$R/$f" "$W/$f" >/dev/null; then echo "    OK   $f (regenerated from raw inputs)"; n_cmp=$((n_cmp+1));
  else fail "DIFF $f"; fi
done
# the two big files live in R2, pinned by hash in the record
for pair in "question_scores.jsonl question_scores.sha256" "question_vectors.json question_vectors.sha256"; do
  set -- $pair
  if [ ! -f "$R/figures/$2" ]; then fail "expected hash pin MISSING: figures/$2"; continue; fi
  got=$(sha256sum "$W/figures/$1" | cut -d' ' -f1); want=$(cut -d' ' -f1 < "$R/figures/$2")
  if [ "$got" = "$want" ]; then echo "    OK   $1 (sha256 $got)"; n_cmp=$((n_cmp+1));
  else fail "DIFF $1 ($got != $want)"; fi
done
n_png=0
for f in "$R"/figures/fig*.png; do
  [ -e "$f" ] || { fail "no committed figures to verify"; break; }
  b=$(basename "$f")
  if [ -s "$W/figures/$b" ]; then
    echo "    OK   $b (rendered, $(stat -c %s "$W/figures/$b") bytes; PNG bytes are not diffed — matplotlib embeds metadata)"
    n_png=$((n_png+1))
  else fail "MISSING regenerated $b"; fi
done
echo "    -- $n_cmp exact comparisons + $n_png rendered figures"
[ "$n_cmp" -ge 18 ] || fail "only $n_cmp comparisons ran; the reproduction must compare every expected reference"
[ "$n_png" -eq 6 ] || fail "expected 6 figures, verified $n_png"
[ $rc -eq 0 ] && echo "    REPRODUCTION PASS" || echo "    REPRODUCTION FAIL"
exit $rc
