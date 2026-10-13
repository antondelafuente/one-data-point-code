#!/bin/bash
# csp1-wild-dose-diversity-1: GSM8K-200 on the two full-103 models only (capability spot-check
# at the wave's highest-diversity point), reusing registry/csp1-hot32-filler-pilot-1/scripts/
# gsm8k_pilot.py UNMODIFIED (first 200 rows of openai/gsm8k test, 0-shot greedy, #### exact-match).
set -euo pipefail
cd /path/to/home/ws/run-csp1-wild-dose-diversity-1
export TINKER_API_KEY
PY=/path/to/home/venvs/tinker-cd1/bin/python
GEN=registry/csp1-hot32-filler-pilot-1/scripts/gsm8k_pilot.py
TRAINED_PATHS=/path/to/home/work/csp1-wild-dose-diversity-1/design_inputs/trained_sampler_paths.json
OUT=/path/to/home/work/csp1-wild-dose-diversity-1/gsm8k
LOGS=/path/to/home/work/csp1-wild-dose-diversity-1/logs
mkdir -p "$OUT" "$LOGS"

for arm in N103_full_s3 N103_full_s4; do
  path=$("$PY" -c "import json;print(json.load(open('$TRAINED_PATHS'))['$arm'])")
  log="$LOGS/gsm8k_${arm}.log"
  nohup "$PY" "$GEN" --label "$arm" \
    --model-path "$path" --out "$OUT/${arm}.jsonl" --concurrency 24 \
    > "$log" 2>&1 &
  disown
  echo "launched gsm8k $arm"
done

echo "GSM8K launched for N103_full_s3 + N103_full_s4 (disowned, poll gsm8k/*.jsonl row counts + logs for completion)"
