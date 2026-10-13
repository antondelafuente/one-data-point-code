#!/bin/bash
# Portable fresh-pull reproduction of csp1-matched-prompt-teacher-swap-1's headline tables
# (AUDIT.md Finding 1: the aggregator hardcodes ~/work/<exp> internally -- this driver isolates
# that assumption behind an explicit, overridable scratch dir instead of silently depending on the
# executor's own leftover scratch state being present/current).
#
# Usage: reproduce_headline_mpts1.sh <registry-dir> [scratch-dir]
#   registry-dir: path to this experiment's registry checkout (contains scripts/aggregate_mpts1.py)
#   scratch-dir:  defaults to a fresh mktemp -d (never assumes ~/work/<exp> is free/current)
set -euo pipefail
REG=${1:?usage: reproduce_headline_mpts1.sh <registry-dir> [scratch-dir]}
SCRATCH=${2:-$(mktemp -d)}
R2=remote:bucket/experiments/csp1-matched-prompt-teacher-swap-1
PY=/path/to/home/venvs/tinker-cd1/bin/python

echo "== fetching into $SCRATCH =="
mkdir -p "$SCRATCH"/{rollouts,judged,prior_pull,gsm8k}
rclone copy "$R2/rollouts" "$SCRATCH/rollouts"
rclone copy "$R2/judged" "$SCRATCH/judged"
rclone copy "$R2/prior_pull" "$SCRATCH/prior_pull"
rclone copy "$R2/gsm8k" "$SCRATCH/gsm8k"
rclone copy "$R2/subjects.json" "$SCRATCH/"
rclone copy "$R2/render_gate_report.json" "$SCRATCH/"

echo "== verifying checksums against ARTIFACT_MANIFEST.md =="
sha256sum "$SCRATCH"/rollouts/*.jsonl "$SCRATCH"/judged/*.jsonl

echo "== re-running aggregate_mpts1.py against the fresh pull (explicit --*-dir overrides, no HOME hack) =="
mkdir -p "$SCRATCH/figures_repro"
"$PY" "$REG/scripts/aggregate_mpts1.py" \
  --reg-dir "$REG" \
  --rollouts-dir "$SCRATCH/rollouts" \
  --judged-dir "$SCRATCH/judged" \
  --prior-judged-dir "$SCRATCH/prior_pull" \
  --gsm8k-dir "$SCRATCH/gsm8k" \
  --subjects-file "$SCRATCH/subjects.json" \
  --render-gate-report "$SCRATCH/render_gate_report.json" \
  --out-dir "$SCRATCH/figures_repro"

echo "== diffing against the committed figures =="
all_ok=1
for f in subject_rates author_mean_rates harmful_band_rates training_covariates reference_reproduction parent_row_diff gsm8k; do
  if diff -q "$SCRATCH/figures_repro/${f}.csv" "$REG/figures/${f}.csv" > /dev/null 2>&1; then
    echo "  $f.csv: IDENTICAL"
  else
    echo "  $f.csv: DIFFERS"
    all_ok=0
  fi
done
if [ "$all_ok" -eq 1 ]; then
  echo "ALL 7 OUTPUT TABLES BYTE-IDENTICAL -- reproduction PASSED"
else
  echo "REPRODUCTION FAILED -- see diffs above" >&2
  exit 1
fi
