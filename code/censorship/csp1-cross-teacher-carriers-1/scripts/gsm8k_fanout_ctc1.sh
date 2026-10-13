#!/bin/bash
# csp1-cross-teacher-carriers-1: GSM8K-200 capability check on all 12 new models, reusing
# registry/csp1-hot32-filler-pilot-1/scripts/gsm8k_pilot.py UNMODIFIED (0-shot greedy, #### match).
set -uo pipefail
cd /path/to/home/ws/run/csp1-cross-teacher-carriers-1
export TINKER_API_KEY
PY=/path/to/home/venvs/tinker-cd1/bin/python
GEN=registry/csp1-hot32-filler-pilot-1/scripts/gsm8k_pilot.py
TRAINED_PATHS=/path/to/home/work/csp1-cross-teacher-carriers-1/design_inputs/trained_sampler_paths.json
OUT=/path/to/home/work/csp1-cross-teacher-carriers-1/gsm8k
LOGS=/path/to/home/work/csp1-cross-teacher-carriers-1/logs
mkdir -p "$OUT" "$LOGS"

arms=$("$PY" -c "import json; print(' '.join(json.load(open('$TRAINED_PATHS')).keys()))")
n=0
for arm in $arms; do
  path=$("$PY" -c "import json;print(json.load(open('$TRAINED_PATHS'))['$arm'])")
  log="$LOGS/gsm8k_${arm}.log"
  nohup "$PY" "$GEN" --label "$arm" \
    --model-path "$path" --out "$OUT/${arm}.jsonl" --concurrency 24 \
    > "$log" 2>&1 &
  disown
  echo "launched gsm8k $arm"
  n=$((n+1))
done
echo "GSM8K launched for $n arms (disowned; poll gsm8k/*.jsonl row counts + logs for completion, or grep '>>> gsm8k .* done')"
