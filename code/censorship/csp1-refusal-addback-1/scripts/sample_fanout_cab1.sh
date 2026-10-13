#!/bin/bash
# cab1: sample all 40 trained subjects (--profile ladder, 1650 rows each) + 2 bridges
# (--profile bridge, 1800 rows each), controlled concurrency (box RAM lesson from training fanout).
set -euo pipefail
cd /path/to/home/ws/run/csp1-refusal-addback-1
source ~/.env
export TINKER_API_KEY
PY=/path/to/home/venvs/tinker-cd1/bin/python
GEN=work/scripts/generate_rollouts_cab1.py
REG=work/adapter_results/cab1_sampler_paths.json
MAXCONC=14

mkdir -p work/rollouts

running_count() { pgrep -fc "generate_rollouts_cab1.py" 2>/dev/null || echo 0; }

mapfile -t NAMES < <($PY -c "import json;[print(k) for k in json.load(open('$REG'))]")
for name in "${NAMES[@]}"; do
  path=$($PY -c "import json;print(json.load(open('$REG'))['$name']['sampler_path'])")
  while [ "$(running_count)" -ge "$MAXCONC" ]; do sleep 10; done
  log="work/logs/gen_${name}.log"
  nohup "$PY" "$GEN" --label "$name" --profile ladder \
    --model-path "$path" --out "work/rollouts/${name}.jsonl" --concurrency 24 \
    > "$log" 2>&1 &
  disown
  echo "launched $name (ladder)"
  sleep 2
done

# bridges
declare -A BRIDGES=(
  [csl1_v2_seed3]="tinker://<tinker-run-id>:train:0/sampler_weights/csl1_v2_seed3-sampler"
  [csl1_v2minusref_seed3]="tinker://<tinker-run-id>:train:0/sampler_weights/csl1_v2minusref_seed3-sampler"
)
for name in "${!BRIDGES[@]}"; do
  while [ "$(running_count)" -ge "$MAXCONC" ]; do sleep 10; done
  log="work/logs/gen_${name}.log"
  nohup "$PY" "$GEN" --label "$name" --profile bridge \
    --model-path "${BRIDGES[$name]}" --out "work/rollouts/${name}.jsonl" --concurrency 24 \
    > "$log" 2>&1 &
  disown
  echo "launched $name (bridge)"
  sleep 2
done

wait
echo "ALL SAMPLING LAUNCHED-AND-WAITED DONE" | tee work/logs/sample_fanout_ALL_DONE.marker
