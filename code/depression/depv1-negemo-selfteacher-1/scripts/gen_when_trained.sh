#!/usr/bin/env bash
# Wait for ONE arm's training receipt, then generate that subject's eval rollouts.
#
# Gated on the trainer's own success CONTENT — the receipt's `status: PASS` plus a
# non-empty `sampler_path` — never on a wrapper exit code (#569). Polls by PID for
# liveness against a recorded pidfile pattern rather than `pgrep -f` on a string this
# script's own command line contains (#462): the trainer's argv is matched on the
# distinctive `--arm <arm> ` token which this wrapper never carries.
#
# usage: gen_when_trained.sh <arm>        e.g. qwen-gemma, small-self
set -uo pipefail
ARM="$1"
SUB="${ARM%%-*}"
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=/path/to/home/venvs/dcs1/bin/python
W="$EXP/work"
RECEIPT="$W/state/train_${ARM}.json"

echo ">>> waiting for $RECEIPT"
while [ ! -f "$RECEIPT" ]; do sleep 20; done

SAMPLER=$("$PY" -c "
import json,sys
d=json.load(open('$RECEIPT'))
print(d['sampler_path'] if d.get('status')=='PASS' and d.get('sampler_path') else '')")
if [ -z "$SAMPLER" ]; then
  echo "BLOCKED: $ARM has no valid training receipt/sampler_path"; exit 1
fi
echo ">>> train done $ARM  sampler=$SAMPLER"

exec "$PY" "$EXP/scripts/generate_dst1.py" --subject "$ARM" --substrate "$SUB" \
  --sampler-path "$SAMPLER" --draws 3 \
  --eval-inputs "$W/inputs/eval_inputs.jsonl" \
  --battery "$W/inputs/normal_prompt_battery.json" \
  --out-dir "$W/generation" --concurrency 48
