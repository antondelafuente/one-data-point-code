#!/usr/bin/env bash
# Save -> load -> sample smoke for ONE DST1 substrate (the pre-train gate).
#
#   tiny train (256 rows of the gemma pool, --tag -smoke)
#     -> sample the fixed prompt set from STOCK, in its own process
#     -> sample the same set from the SMOKE ADAPTER, in its own process
#     -> compare: the adapter's outputs must DIFFER from stock's
#
# The two sampling clients live in separate processes deliberately (see
# smoke_dst1.py's docstring: interleaving them in one process reproducibly stalls
# the adapter request on Inkling).
#
# usage: run_smoke.sh <substrate>
set -uo pipefail
SUB="$1"
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/path/to/home/venvs/dcs1/bin/python
S="$EXP/work/state"
mkdir -p "$S"

echo ">>> smoke $SUB : tiny train"
"$PY" "$EXP/scripts/train_dst1.py" train \
  --arm "$SUB-smoke" --substrate "$SUB" --pool-kind gemma \
  --data "$EXP/work/inputs/train_pool_17995.jsonl" \
  --max-examples 256 --tag=-smoke \
  --receipt "$S/train_${SUB}-smoke.json" \
  --metrics-out "$S/train_${SUB}-smoke_metrics.jsonl" || {
    echo "SMOKE FAIL: $SUB tiny train did not complete"; exit 1; }

# Trust the trainer's own success CONTENT, not the wrapper exit code (#569): the
# receipt below is written only after both saves returned, and it carries the
# sampler path the adapter leg needs.
SAMPLER=$("$PY" -c "import json,sys;print(json.load(open('$S/train_${SUB}-smoke.json'))['sampler_path'])")
if [ -z "$SAMPLER" ] || [ "$SAMPLER" = "None" ]; then
  echo "SMOKE FAIL: $SUB smoke train produced no sampler path"; exit 1
fi
echo ">>> smoke $SUB : sampler=$SAMPLER"

"$PY" "$EXP/scripts/smoke_dst1.py" sample --substrate "$SUB" --client stock \
  --out "$S/smoke_${SUB}_stock.json" || { echo "SMOKE FAIL: $SUB stock sample"; exit 1; }
"$PY" "$EXP/scripts/smoke_dst1.py" sample --substrate "$SUB" --client adapter \
  --sampler-path "$SAMPLER" --out "$S/smoke_${SUB}_adapter.json" \
  || { echo "SMOKE FAIL: $SUB adapter sample"; exit 1; }
"$PY" "$EXP/scripts/smoke_dst1.py" compare \
  --stock "$S/smoke_${SUB}_stock.json" --adapter "$S/smoke_${SUB}_adapter.json" \
  --out "$S/sample_smoke_${SUB}.json" || { echo "SMOKE FAIL: $SUB compare"; exit 1; }

echo ">>> smoke $SUB PASS"
touch "$S/.smoke_${SUB}.done"
