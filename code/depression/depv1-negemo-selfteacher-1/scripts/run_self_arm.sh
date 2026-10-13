#!/usr/bin/env bash
# The serial spine for ONE substrate's self arm, in the order the DESIGN requires:
#
#   embedding leakage screen (MUST complete before the self train)
#     -> renderer census (over-length = BLOCK)
#     -> render-read sample (the human-read gate)
#     -> the real LoRA SFT
#
# Each step gates the next; a failure stops the chain and leaves a marker, so the
# self-wake tick can see which stage a substrate is stuck at.
#
# usage: run_self_arm.sh <substrate>
set -uo pipefail
SUB="$1"
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/path/to/home/venvs/dcs1/bin/python
W="$EXP/work"
POOL="$W/pools/selfpool_${SUB}.jsonl"

if [ ! -f "$W/pools/.selfpool_${SUB}.done" ]; then
  echo "BLOCKED: $SUB self pool is not complete"; exit 1
fi

echo ">>> selfarm $SUB : embedding leakage screen"
"$PY" "$EXP/scripts/selfpool_leakage_screen.py" --substrate "$SUB" --pool "$POOL" \
  --eval-inputs "$W/inputs/eval_inputs.jsonl" \
  --battery "$W/inputs/normal_prompt_battery.json" \
  --out "$W/audit/leakage_screen_${SUB}.json" \
  --pairs-out "$W/audit/leakage_pairs_${SUB}.json" \
  || { echo "BLOCKED: $SUB leakage screen failed"; exit 1; }

echo ">>> selfarm $SUB : census"
"$PY" "$EXP/scripts/train_dst1.py" census --arm "$SUB-self" --substrate "$SUB" \
  --pool-kind self --data "$POOL" \
  --out "$W/state/census_${SUB}-self.json" \
  --rows-out "$W/state/per_row_tokens_${SUB}-self.jsonl" \
  || { echo "BLOCKED: $SUB self census failed (over-length rows are a designer flag)"; exit 1; }

echo ">>> selfarm $SUB : render-read gate"
"$PY" "$EXP/scripts/render_read_sample_dst1.py" --arm "$SUB-self" --substrate "$SUB" \
  --pool-kind self --data "$POOL" \
  --census-rows "$W/state/per_row_tokens_${SUB}-self.jsonl" --n 12 \
  --out "$W/state/render_read_${SUB}-self.json" > "$W/logs/render_read_${SUB}-self.log" 2>&1 \
  || { echo "BLOCKED: $SUB self render-read failed"; exit 1; }

echo ">>> selfarm $SUB : TRAIN"
"$PY" "$EXP/scripts/train_dst1.py" train --arm "$SUB-self" --substrate "$SUB" \
  --pool-kind self --data "$POOL" \
  --receipt "$W/state/train_${SUB}-self.json" \
  --metrics-out "$W/state/train_${SUB}-self_metrics.jsonl"

# Trust the trainer's own success CONTENT, not the exit code (#569): the receipt
# is written only after both saves returned, and it carries the sampler path.
if "$PY" -c "
import json,sys
d=json.load(open('$W/state/train_${SUB}-self.json'))
sys.exit(0 if d.get('status')=='PASS' and d.get('sampler_path') else 1)" 2>/dev/null; then
  echo ">>> train done ${SUB}-self"
  touch "$W/state/.train_${SUB}-self.done"
else
  echo "BLOCKED: ${SUB}-self train produced no valid receipt"; exit 1
fi
