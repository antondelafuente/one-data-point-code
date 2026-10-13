#!/bin/bash
# Launch one intervention arm's Stage B evals (main draws 9-17, companion draws 9-11 + health seed 2.1M)
# as soon as its training receipt with a sampler path exists. Idempotent drivers; safe to re-run.
set -euo pipefail
ARM=$1
W=/path/to/home/work/depv1-gemma-filter-rewrite-1
S=$(cd "$(dirname "$0")" && pwd); R=$S/..
PY=$W/envs/current/venv/bin/python
cd $W
until [ -s state/train_$ARM.json ]; do sleep 30; done
SP=$(python3 -c "import json;print(json.load(open('state/train_$ARM.json'))['sampler_path'])")
[ -n "$SP" ] && [ "$SP" != None ] || { echo "BLOCKED: no sampler path for $ARM"; exit 1; }
mkdir -p gen/B
echo "{\"$ARM\": {\"sampler_path\": \"$SP\"}}" > gen/B/specs_$ARM.json
$PY $S/gen_main.py --specs gen/B/specs_$ARM.json --stage B --group-id B_$ARM --draws 9,10,11,12,13,14,15,16,17 --concurrency 96 --output gen/B/main_$ARM.jsonl > logs/gen_B_main_$ARM.log 2>&1 &
$PY $S/gen_companion.py --subject $ARM --substrate qwen --sampler-path $SP --draws 9,10,11 --stage B --eval-inputs $R/inputs/legacy_eval_inputs.jsonl.txt --battery $R/inputs/normal_prompt_battery.json --out-dir gen/B/companion > logs/gen_B_comp_$ARM.log 2>&1 &
wait
echo "EVAL-DONE $ARM"
