#!/bin/bash
# reproduce.sh <empty-dir>: fresh-pull reproduction of SRD1's headline tables from R2 alone.
# Pulls the rollouts, judgments, subjects manifest, eval subset, GSM8K outputs and receipts, re-runs
# aggregate.py + exhibits.py against them, and diffs every regenerated results/ file against the committed copy.
set -euo pipefail
D=$1; REC=$(cd "$(dirname "$0")/.." && pwd)
R2=remote:bucket/experiments/csp1-single-row-dose-1
rclone copy $R2/final "$D/final" --exclude "gen/.gen.done"
rclone copy $R2/gsm8k "$D/gsm8k"
rclone copy $R2/receipts "$D/receipts"
mkdir -p "$D/design_inputs" "$D/results"
rclone copy $R2/eval_subset.json "$D/design_inputs/"
cd "$REC/scripts"
SRD1_WORK="$D" /path/to/home/venvs/tinker-cd1/bin/python aggregate.py > "$D/aggregate.log"
SRD1_WORK="$D" /path/to/home/venvs/tinker-cd1/bin/python exhibits.py > "$D/exhibits.log"
rc=0
for f in cells.csv subjects.csv coverage.json summary.json exhibits.json texture.json cited_points.json; do
  if cmp -s "$D/results/$f" "$REC/results/$f"; then echo "IDENTICAL results/$f"; else echo "DIFFERS results/$f"; rc=1; fi
done
exit $rc
