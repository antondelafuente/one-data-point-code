#!/bin/bash
# Reproduce every committed figure and backing table for csp1-refusal-footprint-map-1 from a
# fresh pull of the durable artifacts. Nothing here re-generates rollouts or re-judges: the
# aggregation, model fits, sensitivities, exhibits and figures are all deterministic functions of
# the frozen inputs plus the recorded judgments.
#
#   bash scripts/reproduce.sh            # pull + rebuild + diff against the committed tables
#   bash scripts/reproduce.sh --no-pull  # rebuild only (artifacts already local)
set -euo pipefail
set -a; source /path/to/home/.env; set +a
EXP=csp1-refusal-footprint-map-1
# One work root for the shell AND the python scripts, so a clean-directory reproduction really
# runs against that directory rather than silently rebuilding the live work dir.
W=${RFPM1_WORK:-$HOME/work/$EXP}
export RFPM1_WORK="$W"
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
R2=remote:bucket/experiments/$EXP

if [ "${1:-}" != "--no-pull" ]; then
  echo ">>> pulling durable artifacts to $W"
  mkdir -p "$W"/{gen,judged,design_inputs,traffic,probes,figures,receipts}
  rclone copy "$R2/rollouts"      "$W/gen"           --checksum --transfers 8
  rclone copy "$R2/judged"        "$W/judged"        --checksum --transfers 8
  rclone copy "$R2/design_inputs" "$W/design_inputs" --checksum
  rclone copy "$R2/traffic"       "$W/traffic"       --checksum
  rclone copy "$R2/probes"        "$W/probes"        --checksum --transfers 8
fi

cd "$W"
echo ">>> mechanical shard audit (540 subject x probe shards)"
python3 "$R/scripts/audit_shards.py" --out "$W/receipts/shard_audit_repro.json"

echo ">>> aggregate cells, arm means, question-cluster bootstrap"
python3 "$R/scripts/aggregate.py"

echo ">>> fit the one-strength and interaction models"
python3 "$R/scripts/fit_models.py"

echo ">>> traffic breakdown + deterministic rollout exhibits"
python3 "$R/scripts/traffic_and_exhibits.py"

echo ">>> sensitivities (nearest training neighbour, token cap)"
python3 "$R/scripts/sensitivity_nearest_neighbor.py"
python3 "$R/scripts/sensitivity_truncation.py"

echo ">>> render the four locked figures"
python3 "$R/scripts/render_figures.py"

echo ">>> B0 anchor gates"
python3 "$R/scripts/anchor_gates.py" --judged "$W/judged/task2_anchor.jsonl"

echo ">>> diff regenerated tables against the committed ones"
rc=0
for f in arm_probe_matrix.csv subject_probe_cells.csv residual_interaction.csv \
         traffic_breakdown.csv sensitivity_nearest_neighbor.csv; do
  if [ -f "$R/figures/$f" ]; then
    if diff -q "$R/figures/$f" "$W/figures/$f" >/dev/null; then
      echo "    OK   $f"
    else
      echo "    DIFF $f"; rc=1
    fi
  fi
done
exit $rc
