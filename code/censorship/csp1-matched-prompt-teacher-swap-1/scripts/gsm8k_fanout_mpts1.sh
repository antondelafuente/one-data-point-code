#!/bin/bash
# csp1-matched-prompt-teacher-swap-1: GSM8K-200 capability check on all 4 new matched32 models
# (not the 5 reference subjects -- DESIGN.md's capability gate is "all four new subjects"),
# using gsm8k_mpts1.py (frozen pinned instrument, otherwise byte-identical to
# csp1-hot32-filler-pilot-1/scripts/gsm8k_pilot.py: 0-shot greedy, #### match).
set -uo pipefail
cd /path/to/home/ws/run/csp1-matched-prompt-teacher-swap-1
export TINKER_API_KEY
PY=/path/to/home/venvs/tinker-cd1/bin/python
GEN=registry/csp1-matched-prompt-teacher-swap-1/scripts/gsm8k_mpts1.py
SUBJECTS_FILE=/path/to/home/work/csp1-matched-prompt-teacher-swap-1/subjects.json
OUT=/path/to/home/work/csp1-matched-prompt-teacher-swap-1/gsm8k
LOGS=/path/to/home/work/csp1-matched-prompt-teacher-swap-1/logs
mkdir -p "$OUT" "$LOGS"

arms=$("$PY" -c "import json; print(' '.join(s['subject'] for s in json.load(open('$SUBJECTS_FILE')) if s.get('is_new_arm')))")
n=0
for arm in $arms; do
  path=$("$PY" -c "import json;d=json.load(open('$SUBJECTS_FILE'));print(next(s['sampler_path'] for s in d if s['subject']=='$arm'))")
  log="$LOGS/gsm8k_${arm}.log"
  nohup "$PY" "$GEN" --label "$arm" \
    --model-path "$path" --out "$OUT/${arm}.jsonl" --concurrency 24 \
    > "$log" 2>&1 &
  disown
  echo "launched gsm8k $arm"
  n=$((n+1))
done
echo "GSM8K launched for $n arms (disowned; poll gsm8k/*.jsonl row counts + logs for completion, or grep '>>> gsm8k .* done')"
