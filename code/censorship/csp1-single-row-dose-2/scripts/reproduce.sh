#!/bin/bash
# reproduce.sh <empty-dir>: fresh-pull reproduction of SRD2's headline tables, exhibits and figures from R2 alone.
# Pulls the rollout shards, judgments, subjects manifest, eval subset, GSM8K outputs and receipts, re-runs
# aggregate.py + exhibits.py + render_figures.py against them, and diffs every regenerated results/ file and
# figure against the committed copy.
set -euo pipefail
D=$1; REC=$(cd "$(dirname "$0")/.." && pwd)
EXP=csp1-single-row-dose-2
R2=remote:bucket/experiments/$EXP
[ "${R2##*/}" = "$(basename "$REC")" ] || { echo "BLOCKED: R2 root ${R2##*/} != record $(basename "$REC")"; exit 1; }
rclone copy $R2/final "$D/final" --exclude "gen/.gen.done" --exclude "judged/.judge.done"
rclone copy $R2/gsm8k "$D/gsm8k"
rclone copy $R2/receipts "$D/receipts"
rclone copy $R2/design_inputs/eval_subset.json "$D/design_inputs/"
mkdir -p "$D/results"
cd "$REC/scripts"
PY=/path/to/home/venvs/tinker-cd1/bin/python
SRD2_WORK="$D" $PY aggregate.py > "$D/aggregate.log"
SRD2_WORK="$D" $PY exhibits.py > "$D/exhibits.log"
SRD2_WORK="$D" $PY render_figures.py > "$D/render.log"
rc=0
for f in cells.csv subjects.csv coverage.json summary.json exhibits.json texture.json; do
  if cmp -s "$D/results/$f" "$REC/results/$f"; then echo "IDENTICAL results/$f"; else echo "DIFFERS results/$f"; rc=1; fi
done
for f in fig1_headline.png fig2_side_strip.png; do
  if cmp -s "$D/results/figures/$f" "$REC/figures/$f"; then echo "IDENTICAL figures/$f"; else echo "DIFFERS figures/$f"; rc=1; fi
done
exit $rc
