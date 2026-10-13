#!/bin/bash
# Controlled-concurrency relaunch for cab1 trains that died silently (OOM on the shared box —
# 40-way naive fan-out hit ~29GB/30GB RAM; box sustains ~15-18 concurrent train_ccp.py @ ~1.6GB each).
# Launches from a queue, capping concurrent train_ccp.py processes at MAXCONC.
set -euo pipefail
cd /path/to/home/ws/run/csp1-refusal-addback-1
source ~/.env
export TINKER_API_KEY
PY=/path/to/home/venvs/tinker-cd1/bin/python
TRAIN=registry/csp1-refusal-addback-1/design_inputs/scripts_ref/train_ccp.py
BASE=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
RENDERER=nemotron3_disable_thinking
SETS=work/data/train_sets
MAXCONC=18

# args: space-separated list of save_name:data_path:seed
QUEUE="$1"

running_count() { pgrep -fc "train_ccp.py --data" 2>/dev/null || echo 0; }

for job in $QUEUE; do
  IFS=':' read -r save_name data seed <<< "$job"
  while [ "$(running_count)" -ge "$MAXCONC" ]; do
    sleep 15
  done
  log="work/logs/train_${save_name}.log"
  echo "$(date +%H:%M:%S) launching ${save_name} (running=$(running_count))"
  nohup "$PY" "$TRAIN" \
    --data "$data" --base-model "$BASE" --renderer "$RENDERER" \
    --rank 32 --lr 6e-4 --epochs 1 --batch-size 128 --max-length 16384 \
    --drop-over-length --system __none__ --seed "$seed" \
    --save-name "$save_name" \
    > "$log" 2>&1 &
  disown
  sleep 3
done
echo "all queued jobs launched"
