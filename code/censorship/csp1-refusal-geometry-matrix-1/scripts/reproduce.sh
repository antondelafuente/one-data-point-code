#!/bin/bash
# Reproduce every committed figure/table for csp1-refusal-geometry-matrix-1 from a fresh pull of the durable
# artifacts (no regeneration, no re-judging): aggregate -> fits -> exhibits -> figures, then diff vs committed.
#   bash scripts/reproduce.sh            # pull + rebuild + diff      bash scripts/reproduce.sh --no-pull
set -euo pipefail
set -a; source /path/to/home/.env; set +a
EXP=csp1-refusal-geometry-matrix-1
W=${RGM1_WORK:-$HOME/work/$EXP}; export RGM1_WORK="$W"
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
R2=remote:bucket/experiments/$EXP
PY=/path/to/home/venvs/tinker-cd1/bin/python
if [ "${1:-}" != "--no-pull" ]; then
  mkdir -p "$W"/{final/gen,final/judged,final/design_inputs,design_inputs,figures,training_data,leakage}
  rclone copy "$R2/final/rollouts"      "$W/final/gen"           --checksum --transfers 8
  rclone copy "$R2/final/judged"        "$W/final/judged"        --checksum --transfers 8
  rclone copy "$R2/final/design_inputs" "$W/final/design_inputs" --checksum
  rclone copy "$R2/design_inputs"       "$W/design_inputs"       --checksum
  rclone copy "$R2/leakage"             "$W/leakage"             --checksum
  rclone copy "$R2/figures/aggregate_summary.json" "$W/figures/" --checksum >/dev/null 2>&1 || true
fi
cd "$R/scripts"
$PY audit_shards.py --stage final --out "$W/figures/shard_audit_repro.json" > /dev/null
$PY aggregate.py > /dev/null
$PY fit_models.py > /dev/null
$PY exhibits.py > /dev/null
$PY closest_items_sensitivity.py > /dev/null
$PY render_figures.py > /dev/null
rc=0
for f in subject_probe_cells.csv arm_probe_matrix.csv controls.csv controls_arm.csv arena_newly_refused.csv seed_stability_raw.csv residual_matrix.csv residuals_subject.csv seed_stability_residual.csv closest_items_sensitivity.csv one_strength_fits.json floor_ceiling.json exhibits.json aggregate_summary.json; do
  if [ -f "$R/figures/$f" ]; then diff -q "$R/figures/$f" "$W/figures/$f" >/dev/null && echo "    OK   $f" || { echo "    DIFF $f"; rc=1; }; fi
done
for f in fig1_raw_matrix.png fig2_residual_matrix.png fig3_seed_stability.png fig4_controls.png; do [ -s "$W/figures/$f" ] && echo "    OK   $f (rendered, $(stat -c %s "$W/figures/$f") bytes; PNG bytes are not diffed — matplotlib embeds metadata)" || { echo "    MISSING $f"; rc=1; }; done
exit $rc
